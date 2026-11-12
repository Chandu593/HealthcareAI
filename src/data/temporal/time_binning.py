"""
Phase 4a -- Time binning.

Converts raw event-level tables (lab, medication, infusionDrug) into a
fixed grid of time bins per patient, per the architecture's pipeline:

    patient -> temporal observations -> time alignment -> fixed time bins
    -> feature aggregation -> ...

Source tables are `lab`, `medication`, `infusionDrug` (NOT vital signs --
no vitalPeriodic available, see schema_registry.py). Bin index 0 covers
[0, bin_minutes), bin index 1 covers [bin_minutes, 2*bin_minutes), etc.,
relative to ICU admission (offset 0). Events with a negative offset
(pre-ICU-admission) are dropped from the time series -- they belong to
the tabular/admission-context modality (Phase 5), not the temporal one.
Events beyond `max_bins * bin_minutes` are dropped (truncated), not
wrapped into the last bin, so a long stay doesn't silently corrupt the
last bin's statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.cohort.antibiotic_reference import is_antibiotic, is_vasopressor


def assign_bin(offset_minutes: pd.Series, bin_minutes: int, max_bins: int) -> pd.Series:
    """Maps a raw offset (minutes since ICU admission) to a bin index in
    [0, max_bins-1], or NaN if the offset is negative or beyond the grid
    (pre-admission / truncated, respectively -- see module docstring)."""
    bin_idx = (offset_minutes // bin_minutes).astype("float")
    bin_idx[offset_minutes < 0] = np.nan
    bin_idx[bin_idx >= max_bins] = np.nan
    return bin_idx


def select_lab_features(
    lab_df: pd.DataFrame, tracked_labnames: list[str], min_coverage_patients: int = 1, top_k_fallback: int = 6,
) -> list[str]:
    """Returns the labname substrings to actually track for this cohort.

    If none of the configured `tracked_labnames` substrings match
    anything in this specific (possibly small/debug) cohort's lab data,
    falls back to the top_k_fallback most frequent labnames observed,
    rather than silently producing an all-NaN feature block. This
    fallback is logged by the caller (run_pipeline.py), not silent.
    """
    if lab_df.empty:
        return []
    matched = [
        name for name in tracked_labnames
        if lab_df["labname"].str.contains(name, case=False, na=False).sum() >= min_coverage_patients
    ]
    if matched:
        return matched
    top = lab_df["labname"].value_counts().head(top_k_fallback).index.tolist()
    return top


def bin_lab_features(
    lab_df: pd.DataFrame, feature_names: list[str], bin_minutes: int, max_bins: int,
) -> pd.DataFrame:
    """Returns long-format (patientunitstayid, bin_idx, feature, value)
    with `value` = mean lab result within that bin for labnames matching
    each feature substring (case-insensitive contains match -- see
    schema_registry.py's note on unverified real labname vocabulary)."""
    if lab_df.empty or not feature_names:
        return pd.DataFrame(columns=["patientunitstayid", "bin_idx", "feature", "value"])

    df = lab_df.copy()
    df["bin_idx"] = assign_bin(df["labresultoffset"], bin_minutes, max_bins)
    df = df.dropna(subset=["bin_idx"])
    df["bin_idx"] = df["bin_idx"].astype(int)

    rows = []
    for feature in feature_names:
        mask = df["labname"].str.contains(feature, case=False, na=False)
        sub = df[mask]
        if sub.empty:
            continue
        agg = sub.groupby(["patientunitstayid", "bin_idx"])["labresult"].mean().reset_index()
        agg["feature"] = f"lab_{feature.replace(' ', '_')}"
        agg = agg.rename(columns={"labresult": "value"})
        rows.append(agg[["patientunitstayid", "bin_idx", "feature", "value"]])

    if not rows:
        return pd.DataFrame(columns=["patientunitstayid", "bin_idx", "feature", "value"])
    return pd.concat(rows, ignore_index=True)


def bin_medication_features(
    medication_df: pd.DataFrame, bin_minutes: int, max_bins: int,
) -> pd.DataFrame:
    """Returns long-format (patientunitstayid, bin_idx, feature, value)
    for `medication_n_active` (count of orders active during the bin)
    and `medication_antibiotic_active` (0/1) -- computed via interval
    overlap, i.e. a drug spanning bins 2-5 counts as active in ALL of
    bins 2,3,4,5, not just its start bin. This uses an explode-based
    approach that is fine at debug/subset scale; at full-mode scale this
    should be rewritten to avoid materializing one row per (drug, bin)
    pair -- flagged here rather than silently left slow.
    """
    if medication_df.empty:
        return pd.DataFrame(columns=["patientunitstayid", "bin_idx", "feature", "value"])

    df = medication_df.copy()
    df["start_bin"] = (df["drugstartoffset"] // bin_minutes).clip(lower=0)
    # Missing stop offset -> treat as confined to its start bin only
    # (a conservative point-event assumption), rather than assuming an
    # arbitrary duration that would spill into a second bin depending on
    # where within the bin the start offset happens to fall.
    stop = df["drugstopoffset"].fillna(df["drugstartoffset"])
    df["stop_bin"] = (stop // bin_minutes).clip(upper=max_bins - 1)
    df = df[df["start_bin"] <= df["stop_bin"]]
    df = df[df["start_bin"] < max_bins]
    if df.empty:
        return pd.DataFrame(columns=["patientunitstayid", "bin_idx", "feature", "value"])

    df["is_abx"] = is_antibiotic(df["drugname"]) if "drugname" in df.columns else False
    df["bin_range"] = df.apply(lambda r: list(range(int(r["start_bin"]), int(r["stop_bin"]) + 1)), axis=1)
    exploded = df[["patientunitstayid", "bin_range", "is_abx"]].explode("bin_range")
    exploded = exploded.rename(columns={"bin_range": "bin_idx"})
    exploded["bin_idx"] = exploded["bin_idx"].astype(int)

    n_active = (
        exploded.groupby(["patientunitstayid", "bin_idx"]).size()
        .rename("value").reset_index()
    )
    n_active["feature"] = "medication_n_active"

    abx_active = (
        exploded.groupby(["patientunitstayid", "bin_idx"])["is_abx"].any()
        .astype(int).rename("value").reset_index()
    )
    abx_active["feature"] = "medication_antibiotic_active"

    return pd.concat([n_active, abx_active], ignore_index=True)[
        ["patientunitstayid", "bin_idx", "feature", "value"]
    ]


def bin_infusion_features(
    infusion_df: pd.DataFrame, bin_minutes: int, max_bins: int,
) -> pd.DataFrame:
    """Returns long-format features for `infusion_n_active`,
    `infusion_vasopressor_active`, `infusion_vasopressor_rate_mean`.

    infusionDrug rows are point-in-time rate recordings (no start/stop
    interval in this table -- see schema_registry.py), so each row is
    assigned to a single bin via its `infusionoffset`, not exploded
    across an interval like medication."""
    if infusion_df.empty:
        return pd.DataFrame(columns=["patientunitstayid", "bin_idx", "feature", "value"])

    df = infusion_df.copy()
    df["bin_idx"] = assign_bin(df["infusionoffset"], bin_minutes, max_bins)
    df = df.dropna(subset=["bin_idx"])
    df["bin_idx"] = df["bin_idx"].astype(int)
    df["is_vaso"] = is_vasopressor(df["drugname"]) if "drugname" in df.columns else False

    n_active = (
        df.groupby(["patientunitstayid", "bin_idx"]).size()
        .rename("value").reset_index()
    )
    n_active["feature"] = "infusion_n_active"

    vaso_active = (
        df.groupby(["patientunitstayid", "bin_idx"])["is_vaso"].any()
        .astype(int).rename("value").reset_index()
    )
    vaso_active["feature"] = "infusion_vasopressor_active"

    vaso_rows = df[df["is_vaso"]].copy()
    if not vaso_rows.empty and "infusionrate" in vaso_rows.columns:
        vaso_rate = (
            vaso_rows.groupby(["patientunitstayid", "bin_idx"])["infusionrate"].mean()
            .rename("value").reset_index()
        )
        vaso_rate["feature"] = "infusion_vasopressor_rate_mean"
    else:
        vaso_rate = pd.DataFrame(columns=["patientunitstayid", "bin_idx", "value", "feature"])

    return pd.concat([n_active, vaso_active, vaso_rate], ignore_index=True)[
        ["patientunitstayid", "bin_idx", "feature", "value"]
    ]
