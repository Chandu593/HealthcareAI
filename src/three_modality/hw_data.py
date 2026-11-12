"""Hospital-wise data access for the Non-FL (and later FL) downstream stages.

Everything downstream of the cross-attention run is trained and evaluated PER HOSPITAL. This module is the single place that decides, for one (hospital, task):
  * which patients are eligible  -- mortality: strict 24 h landmark with a known hospital-discharge status; incident sepsis: the incident cohort (known label) inside the landmark;
  * which of them are train / val / test -- the PERMANENT split carried by the cross-attention run (never regenerated here);
  * whether the hospital has enough data to train, otherwise WHY it is skipped.

Test isolation by construction: `HospitalTaskData.labels("train" | "val")` is the only label accessor available to training / sampling / graph construction / checkpoint and threshold
selection. Test labels are reachable ONLY through `test_labels_for_evaluation()`, which the evaluator calls after every choice has been fixed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.three_modality import artifacts
from src.three_modality.cross_attention3 import ENCODER_EMBEDDING_FILES, align_embeddings_by_patient_id, load_cross_attention_run

TASKS = ("mortality", "sepsis")
SPLITS = ("train", "val", "test")


def stable_seed(base: int, *parts) -> int:
    """Deterministic per-(hospital, task, variant) seed that does not depend on processing order (Non-FL independence + reproducibility)."""
    h = hashlib.sha256(("|".join(str(p) for p in (base, *parts))).encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31 - 1)


@dataclass
class Cohort:
    """Row-aligned arrays (cohort order = the TCD patient order recorded in the cross-attention run)."""
    patient_ids: np.ndarray
    hospitalid: np.ndarray
    split: np.ndarray
    landmark: np.ndarray
    labels: dict                                   # task -> float array, NaN = unknown / not eligible (TEST values included: evaluation only)
    embeddings: dict                               # set name -> (N, d) float32, aligned to patient_ids
    manifest: dict
    run_dir: str

    def eligible(self, task: str) -> np.ndarray:
        y = self.labels[task]
        return self.landmark & ~np.isnan(y)

    def hospitals(self) -> np.ndarray:
        return np.unique(self.hospitalid)


def load_cohort(cross_attention_run_dir, outputs_root, sets=("fused",)) -> Cohort:
    """Loads the cross-attention run and (optionally) the individual frozen modality embeddings, each re-aligned BY PATIENT ID and hash-verified against its encoder manifest."""
    run = load_cross_attention_run(cross_attention_run_dir)
    man = run["manifest"]
    ids = run["patient_ids"]
    lab = run["labels"]
    if not np.array_equal(lab["patient_ids"], ids):
        raise ValueError("task_labels.npz patient order differs from the fused embeddings'")
    emb = {}
    for s in sets:
        if s == "fused":
            emb["fused"] = run["fused"].astype(np.float32)
            continue
        er = man["encoder_runs"][s]
        sub = {"gru": "gru/encoder", "tabular": "tabular/encoder", "tcd": "tcd/encoder"}[s]
        d = Path(outputs_root) / sub / er["run_id"]
        f = d / ENCODER_EMBEDDING_FILES[s]
        recorded = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["files"][ENCODER_EMBEDDING_FILES[s]]
        if artifacts.sha256_file(f) != recorded:
            raise ValueError(f"{f} does not match the SHA-256 in its encoder manifest")
        z = np.load(f)
        emb[s], _ = align_embeddings_by_patient_id(ids, z["patient_ids"], z["embeddings"].astype(np.float32), s)
    return Cohort(patient_ids=ids, hospitalid=lab["hospitalid"].astype(np.int64), split=lab["split"].astype(str), landmark=lab["landmark"].astype(bool),
                  labels={"mortality": lab["mortality"].astype(np.float64), "sepsis": lab["sepsis"].astype(np.float64)}, embeddings=emb, manifest=man, run_dir=str(cross_attention_run_dir))


@dataclass
class HospitalTaskData:
    hospitalid: int
    task: str
    rows: dict                                     # split -> ascending row indices into the cohort arrays
    counts: dict                                   # split -> {"n","pos","neg"}
    _cohort: Cohort = field(repr=False, default=None)
    _y: np.ndarray = field(repr=False, default=None)

    def labels(self, split: str) -> np.ndarray:
        if split not in ("train", "val"):
            raise PermissionError("test labels are only available through test_labels_for_evaluation(); training, sampling, graph construction and selection must never read them")
        return self._y[self.rows[split]].astype(np.int64)

    def test_labels_for_evaluation(self) -> np.ndarray:
        return self._y[self.rows["test"]].astype(np.int64)

    def X(self, split: str, name: str = "fused") -> np.ndarray:
        return self._cohort.embeddings[name][self.rows[split]]

    def patient_ids(self, split: str) -> np.ndarray:
        return self._cohort.patient_ids[self.rows[split]]

    def skip_reason(self, ms: dict) -> "str | None":
        """None when the hospital can be trained for this task, else a human-readable reason (all failing criteria joined)."""
        c, why = self.counts, []
        tr, va = c["train"], c["val"]
        if tr["n"] < ms["train"]:
            why.append(f"train patients {tr['n']} < {ms['train']}")
        if tr["pos"] < ms["train_pos"]:
            why.append(f"train positives {tr['pos']} < {ms['train_pos']}")
        if tr["neg"] < ms["train_neg"]:
            why.append(f"train negatives {tr['neg']} < {ms['train_neg']}")
        if va["pos"] < ms["val_pos"]:
            why.append(f"validation positives {va['pos']} < {ms['val_pos']}")
        if va["neg"] < ms["val_neg"]:
            why.append(f"validation negatives {va['neg']} < {ms['val_neg']}")
        return "; ".join(why) if why else None


def hospital_task_data(cohort: Cohort, hospitalid: int, task: str) -> HospitalTaskData:
    y = cohort.labels[task]
    elig = cohort.eligible(task) & (cohort.hospitalid == hospitalid)
    rows, counts = {}, {}
    for s in SPLITS:
        r = np.flatnonzero(elig & (cohort.split == s))
        rows[s] = r
        pos = int((y[r] == 1).sum())
        counts[s] = {"n": int(len(r)), "pos": pos, "neg": int(len(r) - pos)}
    return HospitalTaskData(hospitalid=int(hospitalid), task=task, rows=rows, counts=counts, _cohort=cohort, _y=y)


def plan_hospitals(cohort: Cohort, task: str, max_hospitals: "int | None" = None) -> list:
    """All hospitals with at least one eligible patient for the task, ascending by hospitalid, as HospitalTaskData (trainable or not) -- skipped ones are reported, never dropped."""
    out = []
    for h in cohort.hospitals():
        d = hospital_task_data(cohort, int(h), task)
        if sum(d.counts[s]["n"] for s in SPLITS) > 0:
            out.append(d)
    return out[:max_hospitals] if max_hospitals else out
