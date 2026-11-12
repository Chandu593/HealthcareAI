"""Observation-window table: ONE place that decides how far each patient's
features may look. unitdischargeoffset is used only to clip the window
(a patient discharged at hour 6 simply has hours 0-6 of data); it is never
exposed as a feature and sequence lengths are never derived from it."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_window_table(cohort_patients: pd.DataFrame, window_minutes: float, min_icu_los_minutes: float = 0.0) -> pd.DataFrame:
    """Returns a DataFrame indexed by patientunitstayid with column `window_end`
    (minutes after ICU admission, <= window_minutes). Patients whose ICU stay is
    shorter than `min_icu_los_minutes` are dropped (strict landmark cohort)."""
    df = cohort_patients[["patientunitstayid", "unitdischargeoffset"]].copy()
    df["patientunitstayid"] = df["patientunitstayid"].astype("int64")
    los = pd.to_numeric(df["unitdischargeoffset"], errors="coerce")
    if min_icu_los_minutes and min_icu_los_minutes > 0:
        keep = los >= min_icu_los_minutes
        df, los = df[keep], los[keep]
    # Missing / non-positive discharge offset => no clipping information; use the full window.
    clipped = np.where(los.notna() & (los > 0), np.minimum(los.to_numpy(dtype="float64"), float(window_minutes)), float(window_minutes))
    out = pd.DataFrame({"window_end": clipped}, index=pd.Index(df["patientunitstayid"].to_numpy(), name="patientunitstayid"))
    if out.index.has_duplicates:
        raise ValueError("cohort_patients has duplicate patientunitstayid -- cannot build a window table.")
    return out
