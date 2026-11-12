"""Tabular autoencoder stage: the existing TabularAutoencoder over the clean static inputs (tabular_inputs.py), trained POOLED with TWO objectives kept separate and configurable:
  * UNSUPERVISED reconstruction (MSE on continuous columns + BCE-with-logits on binary / missing-indicator columns), weight `training.reconstruction_weight`;
  * SUPERVISED focal multitask loss on the bottleneck embedding through two linear heads (separate alpha/gamma/task weights per task).
The bottleneck (default 64-d) is the embedding. Same population/split discipline as the GRU stage (multitask_training.py)."""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.three_modality import artifacts
from src.three_modality.config import TABULAR_AE_DEFINITION_VERSION
from src.three_modality.multitask_training import encode_all, prepare_task_population, train_multitask
from src.three_modality.tabular_inputs import load_tabular_inputs
from src.three_modality.tcd_encoder import build_task_labels, label_counts

logger = logging.getLogger("eicu_fl")


def make_tabular_ae_model(input_dim: int, enc_cfg: dict):
    import torch.nn as nn
    from src.modalities.tabular.autoencoder import AutoencoderConfig, TabularAutoencoder

    class TabularAEMultitaskModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ae = TabularAutoencoder(AutoencoderConfig(input_dim=int(input_dim), hidden_dim=int(enc_cfg["hidden_dim"]), embedding_dim=int(enc_cfg["embedding_dim"]),
                                                           dropout=float(enc_cfg["dropout"])))
            self.mortality_head = nn.Linear(int(enc_cfg["embedding_dim"]), 1)
            self.sepsis_head = nn.Linear(int(enc_cfg["embedding_dim"]), 1)

        def forward(self, x):
            emb, recon = self.ae(x)
            return emb, self.mortality_head(emb).squeeze(-1), self.sepsis_head(emb).squeeze(-1), (recon, x)

    return TabularAEMultitaskModel()


def make_reconstruction_loss(is_binary: np.ndarray, recon_cfg: dict):
    """aux_loss_fn(aux=(recon, x), idx): MSE over continuous columns + BCE-with-logits over binary columns (the decoder output is a logit there; the target is the 0/1 input)."""
    import torch
    import torch.nn.functional as F
    b = torch.from_numpy(np.asarray(is_binary, bool)); c = ~b

    def loss(aux, idx):
        recon, x = aux
        total = recon.new_zeros(())
        if bool(c.any()):
            total = total + float(recon_cfg["continuous_weight"]) * F.mse_loss(recon[:, c], x[:, c])
        if bool(b.any()):
            total = total + float(recon_cfg["binary_weight"]) * F.binary_cross_entropy_with_logits(recon[:, b], x[:, b])
        return total
    return loss


def tabular_ae_identity(inputs_manifest: dict, incident_manifest: dict, acfg: dict, window_minutes: int) -> str:
    return artifacts.hash_json({"definition": TABULAR_AE_DEFINITION_VERSION, "inputs": inputs_manifest["identity"], "incident_label": incident_manifest["identity"],
                                "mortality_source": "hospitaldischargestatus", "encoder": acfg["encoder"], "training": acfg["training"], "window_minutes": window_minutes})


def build_tabular_ae(inputs_run_dir, incident_run_dir, patient_table: pd.DataFrame, acfg: dict, tcfg: dict, out_base, run_id: "str | None" = None, force: bool = False) -> dict:
    t0 = time.time()
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    iman = json.loads((Path(incident_run_dir) / "manifest.json").read_text(encoding="utf-8"))
    xman = json.loads((Path(inputs_run_dir) / "manifest.json").read_text(encoding="utf-8"))
    identity = tabular_ae_identity(xman, iman, acfg, tcfg["window_minutes"])
    if not force:
        hit = artifacts.find_reusable_run(out_base, identity)
        if hit:
            logger.info("[TAB_AE] cache HIT: reusing verified run %s (identity %s)", hit["run_id"], identity[:12])
            return {"run_dir": hit["run_dir"], "run_id": hit["run_id"], "cache_hit": True, "manifest": hit}
    run_id, run_dir = artifacts.new_run_dir(out_base, "tabae", identity, run_id)
    logger.info("[TAB_AE] cache MISS (identity %s) -> computing into %s", identity[:12], run_dir)
    import torch

    inp = load_tabular_inputs(inputs_run_dir)
    pats, X = inp["patients"], np.ascontiguousarray(inp["X"])
    inc = pd.read_csv(Path(incident_run_dir) / "incident_sepsis_labels.csv")
    mort, sep = build_task_labels(pats, patient_table, inc)
    split = pats["split"].to_numpy()
    counts = label_counts(mort, sep, split)
    tr_cfg = acfg["training"]
    split_eff, pop = prepare_task_population(pats, patient_table, tcfg["window_minutes"], tr_cfg["landmark_only"], mort, sep)
    mort_t, sep_t = mort.copy(), sep.copy()
    mort_t[split == "test"] = np.nan; sep_t[split == "test"] = np.nan              # TEST labels never reach training or early stopping
    logger.info("[TAB_AE] population (landmark_only=%s): train %d, val %d | inputs %s (%d binary)", tr_cfg["landmark_only"], int((split_eff == "train").sum()),
                int((split_eff == "val").sum()), X.shape, int(inp["is_binary"].sum()))
    batch_fn = lambda idx: (torch.from_numpy(X[idx]),)   # noqa: E731
    model, hist = train_multitask(lambda: make_tabular_ae_model(X.shape[1], acfg["encoder"]), batch_fn, mort_t, sep_t, split_eff, tr_cfg,
                                  aux_loss_fn=make_reconstruction_loss(inp["is_binary"], tr_cfg["reconstruction"]), aux_weight=float(tr_cfg["reconstruction_weight"]))
    hist["training_population"] = pop
    emb = encode_all(model, batch_fn, len(pats))
    ids = pats["patientunitstayid"].to_numpy(np.int64)
    logger.info("[TAB_AE] best epoch %s of %d; embeddings %s", hist["best_epoch"], len(hist["epochs"]), emb.shape)
    run_dir.mkdir(parents=True)
    b = io.BytesIO(); np.savez_compressed(b, patient_ids=ids, embeddings=emb)
    artifacts.atomic_write(run_dir / "tab_ae_embeddings.npz", b.getvalue())
    bb = io.BytesIO()
    torch.save({"state_dict": model.state_dict(), "encoder_config": dict(acfg["encoder"]), "input_dim": int(X.shape[1]), "input_names": inp["schema"]["names"],
                "is_binary": inp["is_binary"], "inputs_identity": xman["identity"], "definition": TABULAR_AE_DEFINITION_VERSION}, bb)
    artifacts.atomic_write(run_dir / "tab_ae.pt", bb.getvalue())
    artifacts.atomic_write(run_dir / "training_log.json", json.dumps({**hist, "encoder": acfg["encoder"]}, indent=1, default=str))
    artifacts.atomic_write(run_dir / "label_counts.json", json.dumps(counts, indent=1))
    artifacts.atomic_write(run_dir / "config_snapshot.json", json.dumps({"encoder": acfg["encoder"], "training": tr_cfg, "inputs_run": xman["run_id"], "incident_label_run": iman["run_id"]}, indent=1))
    names = ["tab_ae_embeddings.npz", "tab_ae.pt", "training_log.json", "label_counts.json", "config_snapshot.json"]
    man = {"run_id": run_id, "definition_version": TABULAR_AE_DEFINITION_VERSION, "identity": identity, "files": {n: artifacts.sha256_file(run_dir / n) for n in names},
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "embedding_shape": list(emb.shape), "embedding_dim": int(emb.shape[1]), "inputs_run": xman["run_id"],
           "inputs_identity": xman["identity"], "incident_label_run": iman["run_id"], "incident_identity": iman["identity"], "best_epoch": hist["best_epoch"],
           "epochs_run": len(hist["epochs"]), "seed": hist["seed"], "label_counts": counts, "training_population": pop, "cache": {"hit": False, "forced_recompute": bool(force)},
           "seconds": round(time.time() - t0, 1)}
    artifacts.atomic_write(run_dir / "manifest.json", json.dumps(man, indent=1, default=str))
    logger.info("[TAB_AE] COMPLETE in %.1fs -> %s", time.time() - t0, run_dir)
    return {"run_dir": str(run_dir), "run_id": run_id, "cache_hit": False, "manifest": man, "history": hist}
