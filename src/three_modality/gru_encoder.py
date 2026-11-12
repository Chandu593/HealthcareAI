"""GRU encoder stage: the existing GRUTimeSeriesEncoder over the first-24-hour sequences (gru_inputs.py), trained pooled with the focal multitask objective
(separate alpha/gamma/weights per task), producing ONE 128-d embedding per patient. See multitask_training.py for the training/split discipline."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.three_modality import artifacts
from src.three_modality.config import GRU_ENCODER_DEFINITION_VERSION
from src.three_modality.gru_inputs import load_gru_inputs
from src.three_modality.multitask_training import encode_all, prepare_task_population, train_multitask
from src.three_modality.tcd_encoder import build_task_labels, label_counts

logger = logging.getLogger("eicu_fl")


def make_gru_model(n_vars: int, enc_cfg: dict):
    import torch
    import torch.nn as nn
    from src.modalities.timeseries.gru_encoder import GRUEncoderConfig, GRUTimeSeriesEncoder

    class GRUMultitaskModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = GRUTimeSeriesEncoder(GRUEncoderConfig(input_dim=2 * int(n_vars), hidden_dim=int(enc_cfg["hidden_dim"]), num_layers=int(enc_cfg["num_layers"]),
                                                                 dropout=float(enc_cfg["dropout"])))
            self.mortality_head = nn.Linear(int(enc_cfg["hidden_dim"]), 1)
            self.sepsis_head = nn.Linear(int(enc_cfg["hidden_dim"]), 1)

        def forward(self, values, mask):
            x = torch.cat([values, mask], dim=-1)                                              # (B, bins, 2V): [scaled value, observation mask]
            lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.long)                  # FIXED length: nothing derived from the ICU exit
            e = self.encoder(x, lengths)
            return e, self.mortality_head(e).squeeze(-1), self.sepsis_head(e).squeeze(-1), None

    return GRUMultitaskModel()


def gru_encoder_identity(inputs_manifest: dict, incident_manifest: dict, gcfg: dict, tcfg_window: int) -> str:
    return artifacts.hash_json({"definition": GRU_ENCODER_DEFINITION_VERSION, "inputs": inputs_manifest["identity"], "incident_label": incident_manifest["identity"],
                                "mortality_source": "hospitaldischargestatus", "encoder": gcfg["encoder"], "training": gcfg["training"], "window_minutes": tcfg_window})


def build_gru_encoder(inputs_run_dir, incident_run_dir, patient_table: pd.DataFrame, gcfg: dict, tcfg: dict, out_base, run_id: "str | None" = None, force: bool = False) -> dict:
    t0 = time.time()
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    iman = json.loads((Path(incident_run_dir) / "manifest.json").read_text(encoding="utf-8"))
    xman = json.loads((Path(inputs_run_dir) / "manifest.json").read_text(encoding="utf-8"))
    identity = gru_encoder_identity(xman, iman, gcfg, tcfg["window_minutes"])
    if not force:
        hit = artifacts.find_reusable_run(out_base, identity)
        if hit:
            logger.info("[GRU_ENCODER] cache HIT: reusing verified run %s (identity %s)", hit["run_id"], identity[:12])
            return {"run_dir": hit["run_dir"], "run_id": hit["run_id"], "cache_hit": True, "manifest": hit}
    run_id, run_dir = artifacts.new_run_dir(out_base, "grue", identity, run_id)
    logger.info("[GRU_ENCODER] cache MISS (identity %s) -> computing into %s", identity[:12], run_dir)
    import torch

    inp = load_gru_inputs(inputs_run_dir)
    pats = inp["patients"]
    values, mask = np.ascontiguousarray(inp["values"]), np.ascontiguousarray(inp["mask"])
    inc = pd.read_csv(Path(incident_run_dir) / "incident_sepsis_labels.csv")
    mort, sep = build_task_labels(pats, patient_table, inc)
    split = pats["split"].to_numpy()
    counts = label_counts(mort, sep, split)
    tr_cfg = gcfg["training"]
    split_eff, pop = prepare_task_population(pats, patient_table, tcfg["window_minutes"], tr_cfg["landmark_only"], mort, sep)
    mort_t, sep_t = mort.copy(), sep.copy()
    mort_t[split == "test"] = np.nan; sep_t[split == "test"] = np.nan              # TEST labels never reach training or early stopping
    logger.info("[GRU_ENCODER] population (landmark_only=%s): train %d, val %d | train mortality %s, train sepsis %s", tr_cfg["landmark_only"],
                int((split_eff == "train").sum()), int((split_eff == "val").sum()), pop["labels_used_for_training"]["train"]["mortality"], pop["labels_used_for_training"]["train"]["sepsis_incident"])
    batch_fn = lambda idx: (torch.from_numpy(values[idx]), torch.from_numpy(mask[idx].astype(np.float32)))   # noqa: E731
    model, hist = train_multitask(lambda: make_gru_model(len(inp["variables"]), gcfg["encoder"]), batch_fn, mort_t, sep_t, split_eff, tr_cfg)
    hist["training_population"] = pop
    emb = encode_all(model, batch_fn, len(pats))
    ids = pats["patientunitstayid"].to_numpy(np.int64)
    logger.info("[GRU_ENCODER] best epoch %s of %d; embeddings %s", hist["best_epoch"], len(hist["epochs"]), emb.shape)

    run_dir.mkdir(parents=True)
    import io
    b = io.BytesIO(); np.savez_compressed(b, patient_ids=ids, embeddings=emb)
    artifacts.atomic_write(run_dir / "gru_embeddings.npz", b.getvalue())
    bb = io.BytesIO()
    torch.save({"state_dict": model.state_dict(), "encoder_config": dict(gcfg["encoder"]), "n_variables": len(inp["variables"]), "variables": inp["variables"],
                "inputs_identity": xman["identity"], "definition": GRU_ENCODER_DEFINITION_VERSION}, bb)
    artifacts.atomic_write(run_dir / "gru_encoder.pt", bb.getvalue())
    artifacts.atomic_write(run_dir / "training_log.json", json.dumps({**hist, "encoder": gcfg["encoder"]}, indent=1, default=str))
    artifacts.atomic_write(run_dir / "label_counts.json", json.dumps(counts, indent=1))
    artifacts.atomic_write(run_dir / "config_snapshot.json", json.dumps({"encoder": gcfg["encoder"], "training": tr_cfg, "inputs_run": xman["run_id"], "incident_label_run": iman["run_id"]}, indent=1))
    names = ["gru_embeddings.npz", "gru_encoder.pt", "training_log.json", "label_counts.json", "config_snapshot.json"]
    man = {"run_id": run_id, "definition_version": GRU_ENCODER_DEFINITION_VERSION, "identity": identity, "files": {n: artifacts.sha256_file(run_dir / n) for n in names},
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "embedding_shape": list(emb.shape), "embedding_dim": int(emb.shape[1]), "inputs_run": xman["run_id"],
           "inputs_identity": xman["identity"], "incident_label_run": iman["run_id"], "incident_identity": iman["identity"], "best_epoch": hist["best_epoch"],
           "epochs_run": len(hist["epochs"]), "seed": hist["seed"], "label_counts": counts, "training_population": pop, "cache": {"hit": False, "forced_recompute": bool(force)},
           "seconds": round(time.time() - t0, 1)}
    artifacts.atomic_write(run_dir / "manifest.json", json.dumps(man, indent=1, default=str))
    logger.info("[GRU_ENCODER] COMPLETE in %.1fs -> %s", time.time() - t0, run_dir)
    return {"run_dir": str(run_dir), "run_id": run_id, "cache_hit": False, "manifest": man, "history": hist}
