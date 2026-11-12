"""Federated clients (one per hospital and task) built from the frozen cohort.

ARCHITECTURAL ISOLATION. An `FLClient` carries train / validation LABELS only. Test labels have no attribute here and are not reachable from any training, tuning, threshold or round-selection
code: the only module allowed to read them is `fl_eval.py` (after every model, round and threshold has been fixed). Test FEATURES are present (they are inputs, needed to score the held-out
patients) but are only ever passed through a model's forward pass by the evaluator. `tests/unit/test_three_modality_fl_leakage.py` enforces this three ways: no test-label attribute, a source
scan of the training modules, and a poison test that flips every test label and requires bit-identical training artefacts.

Preprocessing is hospital-local: the StandardScaler of each client is fitted on that client's own TRAINING rows only (the standing per-hospital normalisation rule). No statistic is ever pooled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.three_modality.hw_data import Cohort, SPLITS, plan_hospitals, stable_seed


@dataclass
class FLClient:
    hospitalid: int
    task: str
    counts: dict                                   # train / val -> {"n", "pos", "neg"}; test -> {"n"} ONLY (the test class counts are test-label information and are read by fl_eval alone)
    pid: dict                                      # split -> patient ids (int64)
    Z: dict                                        # split -> raw (unscaled) embeddings, float64 (n, d)
    y: dict                                        # {"train": int64, "val": int64}  -- deliberately NO "test" entry
    scaler: StandardScaler = field(repr=False, default=None)
    nonfl_eligible: bool = False                   # satisfies the Non-FL minimum-data rule (so it is in the paired comparison set)
    nonfl_skip_reason: "str | None" = None
    seed: int = 0
    info: dict = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return int(self.Z["train"].shape[0])

    @property
    def n_val(self) -> int:
        return int(self.Z["val"].shape[0])

    @property
    def n_test(self) -> int:
        return int(self.Z["test"].shape[0])

    @property
    def dim(self) -> int:
        return int(self.Z["train"].shape[1])

    def scaled(self, split: str) -> np.ndarray:
        """Hospital-local standardisation (scaler fitted on this client's training rows only), float32."""
        return self.scaler.transform(self.Z[split]).astype(np.float32)

    def with_features(self, Z: dict) -> "FLClient":
        """A new client over different input features (Config 4: the GCN's hospital-local node embeddings). Labels, patient ids and counts are unchanged; the scaler is REFITTED on the new
        training rows of this hospital only."""
        c = FLClient(hospitalid=self.hospitalid, task=self.task, counts=self.counts, pid=self.pid, Z={s: np.asarray(Z[s], dtype=np.float64) for s in SPLITS}, y=self.y,
                     nonfl_eligible=self.nonfl_eligible, nonfl_skip_reason=self.nonfl_skip_reason, seed=self.seed, info=dict(self.info))
        c.scaler = StandardScaler().fit(c.Z["train"])
        return c


def build_clients(cohort: Cohort, task: str, embedding_set: str, ds_cfg: dict, scope: str = "all", max_hospitals: "int | None" = None, base_seed: int = 42) -> tuple:
    """(clients, skipped_rows). `skipped_rows` lists every hospital that does not become a client with the reason -- nothing is dropped silently.

    scope "all"           : every hospital with >= 1 training patient;
    scope "nonfl_matched" : only hospitals passing the Non-FL minimum-data rule (the control arm)."""
    if scope not in ("all", "nonfl_matched"):
        raise ValueError("scope must be 'all' or 'nonfl_matched'")
    clients, skipped = [], []
    for d in plan_hospitals(cohort, task, max_hospitals):
        why = d.skip_reason(ds_cfg["min_samples"])
        if d.counts["train"]["n"] == 0:
            skipped.append({"hospitalid": d.hospitalid, "reason": "no training patients"})
            continue
        if scope == "nonfl_matched" and why is not None:
            skipped.append({"hospitalid": d.hospitalid, "reason": f"outside the Non-FL matched set: {why}"})
            continue
        Z = {s: np.asarray(d.X(s, embedding_set), dtype=np.float64) for s in SPLITS}
        y = {"train": d.labels("train"), "val": d.labels("val")}          # the only label reads: train and validation (labels("test") raises PermissionError by design)
        counts = {"train": d.counts["train"], "val": d.counts["val"], "test": {"n": d.counts["test"]["n"]}}
        c = FLClient(hospitalid=d.hospitalid, task=task, counts=counts, pid={s: d.patient_ids(s).astype(np.int64) for s in SPLITS}, Z=Z, y=y,
                     nonfl_eligible=why is None, nonfl_skip_reason=why, seed=stable_seed(base_seed, "fl", task, d.hospitalid))
        c.scaler = StandardScaler().fit(Z["train"])
        clients.append(c)
    return clients, skipped


def class_totals(clients: list) -> tuple:
    """Federated class counts: the sum over clients of two scalars each (positives, negatives of the TRAINING rows)."""
    pos = int(sum(int((c.y["train"] == 1).sum()) for c in clients))
    neg = int(sum(int((c.y["train"] == 0).sum()) for c in clients))
    return pos, neg


def global_alpha(clients: list, focal_cfg: dict, smote: bool) -> float:
    """Focal alpha shared by every client: with SMOTE the neutral value (as in Non-FL); otherwise the 'balanced' rule evaluated on the FEDERATED class counts."""
    if smote:
        return float(focal_cfg["alpha_with_smote"])
    a = focal_cfg["alpha"]
    if a == "balanced":
        pos, neg = class_totals(clients)
        return float(np.clip(neg / max(pos + neg, 1), 0.05, 0.95))
    return float(a)
