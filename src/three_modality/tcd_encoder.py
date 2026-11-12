"""TCD encoder: maps the ONE unified TCD vector (tcd_features.py, transformed) to ONE patient embedding.

Architecture (reused, not redesigned): the existing TabularMLPEncoder (Linear -> ReLU -> Dropout -> Linear, src/modalities/tabular/mlp_encoder.py) as the trunk, plus
two linear task heads (mortality, incident sepsis proxy) used ONLY to give the embedding a supervised objective. Loss = mortality_weight * focal(mortality) +
sepsis_weight * focal(sepsis), the repo's masked_focal_loss on logits (gamma, alpha, task weights all configurable); a patient with an unknown label for a task
simply does not contribute to that task's term. The embedding is the encoder output (dim configurable, default 64) -- it is NOT a prediction.

Training population: with `training.landmark_only` (default) only patients still in the ICU at the landmark (valid ICU exit >= window) are used for gradient updates and
early stopping; embeddings are still produced for every patient.
Leakage discipline: only TRAIN rows are used for gradient updates; VALIDATION rows only drive early stopping / model selection; TEST rows are never touched by
training (they are only embedded afterwards). Test labels are not even loaded into the training function. Torch is imported lazily so this module (and the CLI)
import without it.
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.three_modality import artifacts
from src.three_modality.tcd_features import load_tcd_features

logger = logging.getLogger("eicu_fl")

TCD_ENCODER_DEFINITION_VERSION = "tcd_encoder_v1"


# =====================================================================================================================================
# labels (torch-free)
# =====================================================================================================================================
def build_task_labels(patients: pd.DataFrame, patient_table: pd.DataFrame, incident_labels: pd.DataFrame) -> tuple:
    """Float arrays aligned to `patients.patientunitstayid` (the feature run's order): mortality (hospitaldischargestatus; NaN unknown) and the incident sepsis
    proxy (1/0 for patients INCLUDED in the incident cohort under the run's censoring policy, NaN everywhere else)."""
    from src.data.cohort.mortality_label import mortality_label
    ids = patients["patientunitstayid"].to_numpy(np.int64)
    pt = patient_table.drop_duplicates("patientunitstayid").set_index("patientunitstayid")
    mort = mortality_label(pt.reindex(ids).reset_index()).to_numpy(float)
    inc = incident_labels.drop_duplicates("patientunitstayid").set_index("patientunitstayid")
    lab = inc["label"].where(inc["included"].astype(bool))
    sep = lab.reindex(ids).to_numpy(float)
    return mort, sep


def landmark_mask(patients: pd.DataFrame, patient_table: pd.DataFrame, window_minutes: int) -> np.ndarray:
    """True for patients still in the ICU at the landmark: a VALID ICU exit time >= window_minutes (the strict landmark of the paper track's primary mortality
    protocol). Patients with a missing/invalid exit time or a shorter stay are False. Aligned to `patients.patientunitstayid`."""
    pt = patient_table.drop_duplicates("patientunitstayid").set_index("patientunitstayid")
    if "unitdischargeoffset" not in pt.columns:
        raise ValueError("patient table needs the column 'unitdischargeoffset' for tcd.training.landmark_only (set it to false to train on all stays)")
    ex = pd.to_numeric(pt["unitdischargeoffset"], errors="coerce").reindex(patients["patientunitstayid"].to_numpy(np.int64)).to_numpy(float)
    return np.isfinite(ex) & (ex >= float(window_minutes))


def label_counts(mort: np.ndarray, sep: np.ndarray, split: np.ndarray) -> dict:
    out = {}
    for sp in ("train", "val", "test"):
        m = split == sp
        out[sp] = {"patients": int(m.sum()),
                   "mortality": {"known": int((~np.isnan(mort[m])).sum()), "positive": int(np.nansum(mort[m] == 1)), "negative": int(np.nansum(mort[m] == 0))},
                   "sepsis_incident": {"known": int((~np.isnan(sep[m])).sum()), "positive": int(np.nansum(sep[m] == 1)), "negative": int(np.nansum(sep[m] == 0))}}
    return out


# =====================================================================================================================================
# model + training (torch, lazy)
# =====================================================================================================================================
def make_model(input_dim: int, enc_cfg: dict):
    import torch.nn as nn
    from src.modalities.tabular.mlp_encoder import MLPEncoderConfig, TabularMLPEncoder

    class TCDMultitaskModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = TabularMLPEncoder(MLPEncoderConfig(input_dim=int(input_dim), hidden_dim=int(enc_cfg["hidden_dim"]),
                                                              embedding_dim=int(enc_cfg["embedding_dim"]), dropout=float(enc_cfg["dropout"])))
            self.mortality_head = nn.Linear(int(enc_cfg["embedding_dim"]), 1)
            self.sepsis_head = nn.Linear(int(enc_cfg["embedding_dim"]), 1)

        def forward(self, x):
            e = self.encoder(x)
            return e, self.mortality_head(e).squeeze(-1), self.sepsis_head(e).squeeze(-1)

    return TCDMultitaskModel()


def _multitask_focal(m_logit, s_logit, m_y, s_y, tcfg):
    from src.training.losses import masked_focal_loss
    parts = []
    lm, nm = masked_focal_loss(m_logit, m_y, gamma=tcfg["gamma"], alpha=tcfg["alpha"])
    ls, ns = masked_focal_loss(s_logit, s_y, gamma=tcfg["gamma"], alpha=tcfg["alpha"])
    if lm is not None:
        parts.append(tcfg["mortality_weight"] * lm)
    if ls is not None:
        parts.append(tcfg["sepsis_weight"] * ls)
    return (sum(parts) if parts else None), nm, ns


def train_tcd_encoder(X, mort, sep, split, enc_cfg: dict, tcfg: dict) -> tuple:
    """Trains on split=='train', early-stops on split=='val'. `X` may be a memmap. Returns (model with the best-validation weights, history dict)."""
    import torch
    seed = int(tcfg["seed"])
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    if len(tr) == 0:
        raise ValueError("no training patients")
    Xtr = torch.from_numpy(np.ascontiguousarray(X[tr], dtype=np.float32))
    mtr, str_ = torch.from_numpy(mort[tr].astype(np.float32)), torch.from_numpy(sep[tr].astype(np.float32))
    has_val = len(va) > 0
    if has_val:
        Xva = torch.from_numpy(np.ascontiguousarray(X[va], dtype=np.float32))
        mva, sva = torch.from_numpy(mort[va].astype(np.float32)), torch.from_numpy(sep[va].astype(np.float32))
    model = make_model(Xtr.shape[1], enc_cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    bs, best, best_state, bad = int(tcfg["batch_size"]), float("inf"), None, 0
    hist = {"epochs": [], "best_epoch": None, "stopped_early": False, "hyperparameters": dict(tcfg), "encoder": dict(enc_cfg), "input_dim": int(Xtr.shape[1]),
            "n_train": int(len(tr)), "n_val": int(len(va)), "seed": seed, "monitor": "validation total focal loss" if has_val else "training total focal loss (no validation patients)"}
    for ep in range(int(tcfg["max_epochs"])):
        model.train()
        g = torch.Generator().manual_seed(seed + ep)
        perm = torch.randperm(len(tr), generator=g)
        tot, nb = 0.0, 0
        for i in range(0, len(tr), bs):
            b = perm[i:i + bs]
            _, ml, sl = model(Xtr[b])
            loss, _, _ = _multitask_focal(ml, sl, mtr[b], str_[b], tcfg)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep + 1}")
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        model.eval()
        with torch.no_grad():
            if has_val:
                _, vm, vs = model(Xva)
                vloss, nvm, nvs = _multitask_focal(vm, vs, mva, sva, tcfg)
                vloss = float(vloss.item()) if vloss is not None else float("nan")
                aucs = _val_aucs(vm.numpy(), vs.numpy(), mva.numpy(), sva.numpy())
            else:
                vloss, aucs = tot / max(nb, 1), {}
        hist["epochs"].append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), "val_loss": vloss, **aucs})
        score = vloss if np.isfinite(vloss) else tot / max(nb, 1)
        if score < best - 1e-9:
            best, bad, hist["best_epoch"] = score, 0, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(tcfg["patience"]):
                hist["stopped_early"] = True
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    hist["best_score"] = float(best)
    return model, hist


def _val_aucs(vm, vs, mva, sva) -> dict:
    """Validation AUROC per task (sanity signal only; the TEST split is never evaluated in this stage)."""
    from sklearn.metrics import roc_auc_score
    out = {}
    for nm, lg, y in (("val_auroc_mortality", vm, mva), ("val_auroc_sepsis", vs, sva)):
        ok = ~np.isnan(y)
        out[nm] = float(roc_auc_score(y[ok], lg[ok])) if ok.sum() > 0 and len(np.unique(y[ok])) == 2 else None
    return out


def encode_all(model, X, batch: int = 8192) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            e, _, _ = model(torch.from_numpy(np.array(X[i:i + batch], dtype=np.float32)))      # copy: memmap slices are read-only
            out.append(e.numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, int(model.mortality_head.in_features)), np.float32)


# =====================================================================================================================================
# orchestration
# =====================================================================================================================================
def tcd_encoder_identity(feature_manifest: dict, incident_manifest: dict, tcfg: dict) -> str:
    return artifacts.hash_json({"definition": TCD_ENCODER_DEFINITION_VERSION, "features": feature_manifest["identity"], "incident_label": incident_manifest["identity"],
                                "mortality_source": "hospitaldischargestatus", "encoder": tcfg["encoder"], "training": tcfg["training"]})


def build_tcd_encoder(feature_run_dir, incident_run_dir, patient_table: pd.DataFrame, tcfg: dict, out_base, run_id: "str | None" = None, force: bool = False) -> dict:
    t0 = time.time()
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    fman = json.loads((Path(feature_run_dir) / "manifest.json").read_text(encoding="utf-8"))
    iman = json.loads((Path(incident_run_dir) / "manifest.json").read_text(encoding="utf-8"))
    identity = tcd_encoder_identity(fman, iman, tcfg)
    if not force:
        hit = artifacts.find_reusable_run(out_base, identity)
        if hit:
            logger.info("[TCD_ENCODER] cache HIT: reusing verified run %s (identity %s)", hit["run_id"], identity[:12])
            return {"run_dir": hit["run_dir"], "run_id": hit["run_id"], "cache_hit": True, "manifest": hit}
    run_id, run_dir = artifacts.new_run_dir(out_base, "tcde", identity, run_id)
    logger.info("[TCD_ENCODER] cache MISS (identity %s) -> computing into %s", identity[:12], run_dir)
    import torch  # noqa: F401  (fail early with a clear message if torch is missing)

    feats = load_tcd_features(feature_run_dir)
    pats = feats["patients"]
    inc_labels = pd.read_csv(Path(incident_run_dir) / "incident_sepsis_labels.csv")
    mort, sep = build_task_labels(pats, patient_table, inc_labels)
    split = pats["split"].to_numpy()
    counts = label_counts(mort, sep, split)
    # landmark restriction: patients outside the landmark get split 'excluded' -> used by neither gradient updates nor early stopping (still embedded below)
    lm_only = bool(tcfg["training"]["landmark_only"])
    landmark = landmark_mask(pats, patient_table, tcfg["window_minutes"]) if lm_only else np.ones(len(pats), bool)
    split_eff = np.where(landmark, split, "excluded")
    train_counts = label_counts(mort, sep, split_eff)
    population = {"landmark_only": lm_only, "rule": f"valid ICU exit >= {tcfg['window_minutes']} min" if lm_only else "all stays",
                  "patients_by_split": {sp: int((split == sp).sum()) for sp in ("train", "val", "test")},
                  "landmark_patients_by_split": {sp: int(((split == sp) & landmark).sum()) for sp in ("train", "val", "test")},
                  "excluded_from_training_and_early_stopping": {sp: int(((split == sp) & ~landmark).sum()) for sp in ("train", "val")},
                  "labels_used_for_training": {sp: train_counts[sp] for sp in ("train", "val")}}
    logger.info("[TCD_ENCODER] training population (landmark_only=%s): train %d patients, val %d | train mortality %s, train sepsis %s", lm_only,
                population["landmark_patients_by_split"]["train"] if lm_only else population["patients_by_split"]["train"],
                population["landmark_patients_by_split"]["val"] if lm_only else population["patients_by_split"]["val"],
                train_counts["train"]["mortality"], train_counts["train"]["sepsis_incident"])
    # TEST labels never reach training: blank them before the training function can see them
    mort_t, sep_t = mort.copy(), sep.copy()
    mort_t[split == "test"] = np.nan; sep_t[split == "test"] = np.nan
    model, hist = train_tcd_encoder(feats["X"], mort_t, sep_t, split_eff, tcfg["encoder"], tcfg["training"])
    hist["training_population"] = population
    emb = encode_all(model, feats["X"])
    ids = pats["patientunitstayid"].to_numpy(np.int64)
    logger.info("[TCD_ENCODER] best epoch %s of %d; embeddings %s", hist["best_epoch"], len(hist["epochs"]), emb.shape)

    run_dir.mkdir(parents=True)
    b = io.BytesIO(); np.savez_compressed(b, patient_ids=ids, embeddings=emb)           # same keys as the GRU / autoencoder embedding files (fusion aligns by ID)
    artifacts.atomic_write(run_dir / "tcd_embeddings.npz", b.getvalue())
    import torch as _t
    bb = io.BytesIO()
    _t.save({"state_dict": model.state_dict(), "encoder_config": dict(tcfg["encoder"]), "input_dim": int(feats["X"].shape[1]),
             "transformed_feature_names": feats["schema"]["transformed_names"], "feature_run_identity": fman["identity"], "definition": TCD_ENCODER_DEFINITION_VERSION}, bb)
    artifacts.atomic_write(run_dir / "tcd_encoder.pt", bb.getvalue())
    artifacts.atomic_write(run_dir / "training_log.json", json.dumps(hist, indent=1, default=str))
    artifacts.atomic_write(run_dir / "label_counts.json", json.dumps(counts, indent=1))
    from src.three_modality.tcd_features import overlap_audit
    audit = overlap_audit({e["variable"] for e in feats["schema"]["features"]})
    artifacts.atomic_write(run_dir / "config_snapshot.json", json.dumps({"encoder": tcfg["encoder"], "training": tcfg["training"], "feature_run": fman["run_id"],
                                                                         "incident_label_run": iman["run_id"], "overlap_audit": audit}, indent=1))
    names = ["tcd_embeddings.npz", "tcd_encoder.pt", "training_log.json", "label_counts.json", "config_snapshot.json"]
    man = {"run_id": run_id, "definition_version": TCD_ENCODER_DEFINITION_VERSION, "identity": identity, "files": {n: artifacts.sha256_file(run_dir / n) for n in names},
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "embedding_shape": list(emb.shape), "embedding_dim": int(emb.shape[1]),
           "feature_run": fman["run_id"], "feature_identity": fman["identity"], "incident_label_run": iman["run_id"], "incident_identity": iman["identity"],
           "best_epoch": hist["best_epoch"], "epochs_run": len(hist["epochs"]), "seed": hist["seed"], "label_counts": counts, "training_population": population,
           "overlap_audit": audit,
           "cache": {"hit": False, "forced_recompute": bool(force)}, "seconds": round(time.time() - t0, 1)}
    artifacts.atomic_write(run_dir / "manifest.json", json.dumps(man, indent=1, default=str))
    logger.info("[TCD_ENCODER] COMPLETE in %.1fs -> %s", time.time() - t0, run_dir)
    return {"run_dir": str(run_dir), "run_id": run_id, "cache_hit": False, "manifest": man, "history": hist}
