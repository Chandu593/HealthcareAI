"""Federated gradient-boosted trees for the three-modality downstream stage -- a thin adapter around the paper track's `src/paper/fed_gbdt.py` (which is IMPORTED, never edited).

Trees are NOT parameter-averaged. Clients (hospitals) contribute per-bin gradient / Hessian histograms of their own TRAINING rows each boosting round; the server adds them, grows one tree on
the aggregate histograms and broadcasts it, so the federated ensemble equals a centrally trained histogram GBDT on the same bins (verified in the paper-track tests). Bin edges come from
federated quantile summaries of the clients' training rows. Class counts are exchanged as two scalars per client.

What differs from the Non-FL tree classifiers (stated in every manifest): FedGBDT uses the logistic (log) loss with an optional positive-class weight (`scale_pos_weight_mode`); it has
no focal loss. Non-FL LightGBM (custom focal objective) and Non-FL sklearn GBDT (standard loss) therefore have no exact federated counterpart; FedGBDT is reported next to both.

Validation. `train_fed_gbdt` early-stops on a validation AUROC. Inside this adapter that metric is replaced (for the duration of the call only, restored in a `finally`) by the same
HISTOGRAM AUROC used by the neural algorithms, i.e. computed from per-class histograms of the validation margins that clients would send, with a recorded tie bound. The exact pooled AUROC at
the selected iteration is stored as a diagnostic. Test rows enter the call only as feature rows (they are binned with the TRAIN edges and scored after training); their label slots hold -1.

Limitation: the boosting loop is a single call, so checkpointing/resume is per (task, variant): a finished model is reloaded, an interrupted boosting run restarts from round 0.
"""
from __future__ import annotations

import contextlib
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np

from src.paper import fed_gbdt as fg
from src.three_modality.fl_server import HistAccumulator
from src.three_modality.hw_data import SPLITS

logger = logging.getLogger("eicu_fl")


def _hist_auroc_factory(bins: int, logit_range: float):
    def _hist_auroc(y, s):
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            return float("nan")
        h = HistAccumulator(bins, logit_range)
        h.add(np.asarray(s, dtype=np.float64), y)
        return h.auroc()
    return _hist_auroc


@contextlib.contextmanager
def federated_validation_metric(bins: int, logit_range: float):
    """Temporarily replace `fed_gbdt._auroc` with the histogram AUROC; ALWAYS restored."""
    original = fg._auroc
    fg._auroc = _hist_auroc_factory(bins, logit_range)
    try:
        yield
    finally:
        fg._auroc = original


def assemble(clients: list) -> dict:
    """One feature matrix over every client's train/val/test rows plus the row maps. Test labels are masked (-1): `train_fed_gbdt` only ever indexes train and validation rows."""
    ids = sorted(c.hospitalid for c in clients)
    by_id = {c.hospitalid: c for c in clients}
    Xs, ys, rows, pos = [], [], {}, 0
    for h in ids:
        c = by_id[h]
        rows[h] = {}
        for s in SPLITS:
            n = c.Z[s].shape[0]
            rows[h][s] = np.arange(pos, pos + n, dtype=np.int64)
            pos += n
            Xs.append(c.Z[s])
            ys.append(c.y[s].astype(np.int64) if s in c.y else np.full(n, -1, dtype=np.int64))
    return {"X": np.concatenate(Xs).astype(np.float64), "y": np.concatenate(ys), "rows": rows, "ids": ids}


def train_fedgbdt(clients: list, gbdt_cfg: dict, val_cfg: dict, seed: int, ckpt_dir=None, resume: bool = False, ident: str = "") -> dict:
    """Train one FedGBDT on the clients' embeddings. Returns the model, per-client validation / test margins, training history and the validation-metric bookkeeping."""
    t0 = time.time()
    final = Path(ckpt_dir) / "final.pkl" if ckpt_dir else None
    if resume and final is not None and final.is_file():
        with open(final, "rb") as fh:
            ck = pickle.load(fh)
        if ck["ident"] != ident:
            raise RuntimeError(f"checkpoint {final} belongs to a different run identity; refusing to resume")
        return {**ck["result"], "resumed_from": "final"}
    asm = assemble(clients)
    client_rows = {h: asm["rows"][h]["train"] for h in asm["ids"] if len(asm["rows"][h]["train"])}
    val_rows = np.concatenate([asm["rows"][h]["val"] for h in asm["ids"]]) if asm["ids"] else np.empty(0, np.int64)
    with federated_validation_metric(val_cfg["bins"], val_cfg["logit_range"]):
        model, hist = fg.train_fed_gbdt(asm["X"], asm["y"], client_rows, val_rows, dict(gbdt_cfg["params"]), seed, n_bins=int(gbdt_cfg["n_bins"]), grid_size=int(gbdt_cfg["grid_size"]),
                                        max_rounds=int(gbdt_cfg["max_rounds"]), early_stopping_rounds=int(gbdt_cfg["early_stopping_rounds"]), log_every=25)
    margins = {h: {"val": model.predict_raw(asm["X"][asm["rows"][h]["val"]]), "test": model.predict_raw(asm["X"][asm["rows"][h]["test"]])} for h in asm["ids"]}
    yv = np.concatenate([asm["y"][asm["rows"][h]["val"]] for h in asm["ids"]])
    mv = np.concatenate([margins[h]["val"] for h in asm["ids"]])
    acc = HistAccumulator(val_cfg["bins"], val_cfg["logit_range"])
    acc.add(mv, yv)
    from sklearn.metrics import roc_auc_score
    exact = float(roc_auc_score(yv, mv)) if len(np.unique(yv)) == 2 else float("nan")
    result = {"margins": margins, "history": hist, "best_iteration": int(hist["best_iteration"]), "n_rounds_run": int(hist["n_rounds_run"]),
              "val_auroc_hist_at_best": float(acc.auroc()), "val_auroc_tie_bound_at_best": float(acc.tie_bound()), "val_auroc_exact_diag": exact,
              "val_pos": acc.pos, "val_neg": acc.neg, "n_clients": int(hist["n_clients"]), "n_train": int(hist["n_train"]), "seconds": round(time.time() - t0, 2),
              "comm_gb_total_sparse_all_clients": hist["comm_gb_total_sparse_all_clients"], "scale_pos_weight": hist["scale_pos_weight"], "resumed_from": None, "params": hist["params"],
              "n_trees": len(model.trees)}
    if final is not None:
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"ident": ident, "result": {**result, "model": model}}, fh)
        os.replace(tmp, final)
    result["model"] = model
    return result
