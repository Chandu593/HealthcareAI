"""
Phase 3e -- Cohort report.

Produces the report Section 9 requires (total/eligible/excluded
patients, mortality/sepsis positives-negatives, hospital distribution,
class imbalance) PLUS the additional metrics the user explicitly
requested for the dual-cohort design: sepsis prevalence per cohort,
onset distribution per cohort, hospital-wise cohort sizes, mortality
among septic patients, and overlap between the strict/broad cohorts.

`downstream_model_performance` is intentionally left as a documented
placeholder -- Phases 11-14 (training/evaluation) are not implemented
yet, so this project will not fabricate a number here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CohortReport:
    total_patients: int
    eligible_patients: int  # non-null mortality label
    excluded_patients: int

    mortality_positive: int
    mortality_negative: int

    sepsis_broad_positive: int
    sepsis_broad_negative: int
    sepsis_broad_prevalence: float

    sepsis_strict_positive: int
    sepsis_strict_negative: int
    sepsis_strict_prevalence: float

    cohort_overlap: dict = field(default_factory=dict)         # both / strict_only / broad_only / neither
    onset_distribution: dict = field(default_factory=dict)     # per cohort: mean/median/min/max (minutes)
    mortality_among_septic: dict = field(default_factory=dict) # per cohort: mortality rate within positives vs negatives
    hospital_distribution: list = field(default_factory=list)  # per-hospital breakdown
    class_imbalance: dict = field(default_factory=dict)

    downstream_model_performance: str = (
        "N/A -- Phases 11-14 (local/centralized/federated training and "
        "evaluation) are not implemented yet. This field will be "
        "populated once those phases exist; it is not fabricated here."
    )


def _onset_stats(offsets: pd.Series) -> dict:
    valid = offsets.dropna()
    if len(valid) == 0:
        return {"n": 0, "mean_minutes": None, "median_minutes": None, "min_minutes": None, "max_minutes": None}
    return {
        "n": int(len(valid)),
        "mean_minutes": round(float(valid.mean()), 1),
        "median_minutes": round(float(valid.median()), 1),
        "min_minutes": round(float(valid.min()), 1),
        "max_minutes": round(float(valid.max()), 1),
    }


def build_cohort_report(
    patient_df: pd.DataFrame,
    mortality: pd.Series,
    sepsis_df: pd.DataFrame,
) -> CohortReport:
    total = len(patient_df)
    eligible_mask = mortality.notna()
    eligible = int(eligible_mask.sum())
    excluded = total - eligible

    merged = patient_df[["patientunitstayid", "hospitalid"]].copy()
    merged["mortality"] = mortality.values
    merged = merged.merge(
        sepsis_df[[
            "patientunitstayid", "sepsis_broad", "sepsis_strict",
            "sepsis_broad_onset_offset_approx", "sepsis_strict_onset_offset_approx",
        ]],
        on="patientunitstayid", how="left",
    )
    merged["sepsis_broad"] = merged["sepsis_broad"].fillna(False).astype(bool)
    merged["sepsis_strict"] = merged["sepsis_strict"].fillna(False).astype(bool)

    mort_pos = int((merged["mortality"] == 1).sum())
    mort_neg = int((merged["mortality"] == 0).sum())

    broad_pos = int(merged["sepsis_broad"].sum())
    strict_pos = int(merged["sepsis_strict"].sum())

    both = int((merged["sepsis_broad"] & merged["sepsis_strict"]).sum())
    broad_only = int((merged["sepsis_broad"] & ~merged["sepsis_strict"]).sum())
    strict_only = int((merged["sepsis_strict"] & ~merged["sepsis_broad"]).sum())
    neither = int((~merged["sepsis_broad"] & ~merged["sepsis_strict"]).sum())

    def mortality_rate(mask: pd.Series) -> float | None:
        sub = merged.loc[mask, "mortality"].dropna()
        return round(float(sub.mean()), 4) if len(sub) else None

    hospital_rows = []
    for hid, grp in merged.groupby("hospitalid"):
        hospital_rows.append({
            "hospitalid": int(hid),
            "patient_count": int(len(grp)),
            "mortality_positive": int((grp["mortality"] == 1).sum()),
            "sepsis_broad_positive": int(grp["sepsis_broad"].sum()),
            "sepsis_strict_positive": int(grp["sepsis_strict"].sum()),
        })

    return CohortReport(
        total_patients=total,
        eligible_patients=eligible,
        excluded_patients=excluded,
        mortality_positive=mort_pos,
        mortality_negative=mort_neg,
        sepsis_broad_positive=broad_pos,
        sepsis_broad_negative=total - broad_pos,
        sepsis_broad_prevalence=round(broad_pos / total, 4) if total else 0.0,
        sepsis_strict_positive=strict_pos,
        sepsis_strict_negative=total - strict_pos,
        sepsis_strict_prevalence=round(strict_pos / total, 4) if total else 0.0,
        cohort_overlap={
            "both_strict_and_broad": both,
            "broad_only": broad_only,
            "strict_only": strict_only,
            "neither": neither,
        },
        onset_distribution={
            "broad": _onset_stats(merged["sepsis_broad_onset_offset_approx"]),
            "strict": _onset_stats(merged["sepsis_strict_onset_offset_approx"]),
        },
        mortality_among_septic={
            "broad_sepsis_positive": mortality_rate(merged["sepsis_broad"]),
            "broad_sepsis_negative": mortality_rate(~merged["sepsis_broad"]),
            "strict_sepsis_positive": mortality_rate(merged["sepsis_strict"]),
            "strict_sepsis_negative": mortality_rate(~merged["sepsis_strict"]),
        },
        hospital_distribution=hospital_rows,
        class_imbalance={
            "mortality_positive_pct": round(100 * mort_pos / eligible, 2) if eligible else None,
            "sepsis_broad_positive_pct": round(100 * broad_pos / total, 2) if total else None,
            "sepsis_strict_positive_pct": round(100 * strict_pos / total, 2) if total else None,
        },
    )
