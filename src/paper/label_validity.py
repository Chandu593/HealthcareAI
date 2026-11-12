"""Label-validity helpers for the sepsis PROXY (antibiotic AND SOFA-like >= 2).

The antibiotic half of the label can only fire if a hospital's medication records exist. Hospitals with (almost) no medication rows therefore
get (almost) no positives -- a data-availability artefact that hospital-level covariates can exploit. These helpers measure each hospital's
medication-record coverage (share of its stays with >= 1 row in any medication table) so the sepsis task can be restricted to hospitals where
the label is meaningful. Coverage uses no outcome information."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.paper.features import _resolve_table


def _ids_with_rows(path, chunksize: int = 2_000_000) -> set:
    ids: set = set()
    for ch in pd.read_csv(path, usecols=["patientunitstayid"], chunksize=chunksize):
        ids.update(ch["patientunitstayid"].unique().tolist())
    return ids


def hospital_medication_coverage(phase1_summary: dict, cohort_patients: pd.DataFrame, cache_path=None) -> pd.Series:
    """Series indexed by hospitalid: fraction of cohort stays with >= 1 row in medication / medication_imputed / medication_imputed1."""
    if cache_path and Path(cache_path).exists():
        d = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        return pd.Series({int(k): float(v) for k, v in d.items()}, dtype="float64")
    ids: set = set()
    for t in ("medication", "medication_imputed", "medication_imputed1"):
        p = _resolve_table(phase1_summary, t)
        if p:
            ids |= _ids_with_rows(p)
    cp = cohort_patients[["patientunitstayid", "hospitalid"]].copy()
    cp["has"] = cp["patientunitstayid"].isin(ids)
    cov = cp.groupby("hospitalid")["has"].mean().astype("float64")
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps({str(int(k)): float(v) for k, v in cov.items()}), encoding="utf-8")
    return cov
