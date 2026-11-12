"""
Phase 3d -- Sepsis label construction (dual cohort).

Combines Phase 3b (organ dysfunction) and Phase 3c (infection proxy):

    sepsis_broad  = broad_infection  AND organ_dysfunction
    sepsis_strict = strict_infection AND organ_dysfunction

Onset offset is the infection-proxy offset (antibiotic start for broad,
corroborated antibiotic+culture time for strict) -- NOT a true
trajectory-derived onset, because organ dysfunction here is a single
admission-window aggregate rather than a timestamped delta (see
sofa_like_scoring.py docstring). This is stated in every output field
name (`*_onset_offset_approx`) rather than left implicit.
"""

from __future__ import annotations

import pandas as pd


def compute_sepsis_labels(
    infection_df: pd.DataFrame,
    sofa_df: pd.DataFrame,
) -> pd.DataFrame:
    df = infection_df.merge(
        sofa_df[["patientunitstayid", "sofa_like_total", "n_domains_observed",
                 "max_possible_score", "organ_dysfunction"]],
        on="patientunitstayid",
        how="outer",
    )
    # .fillna(False) alone is NOT enough here: an outer merge can
    # introduce NaN into what was a bool column (e.g. a patient present
    # in infection_df but absent from sofa_df because they have no
    # apachePredVar row at all -- a real, common situation in eICU,
    # ~468/4064 patients in one real run). Pandas silently upcasts a
    # bool column to `object` dtype to hold that NaN, and fillna(False)
    # does NOT restore it to bool dtype -- it stays `object`, holding
    # Python bool objects. Downstream `~column` then calls Python's
    # bitwise invert (bool is an int subclass) instead of logical NOT:
    # ~True -> -2, ~False -> -1. The explicit .astype(bool) below is
    # required, not cosmetic -- this crashed in cohort_report.py before
    # it was added. See tests/unit/test_sepsis_and_cohort_report.py's
    # partial-coverage regression test.
    df["organ_dysfunction"] = df["organ_dysfunction"].fillna(False).astype(bool)
    df["broad_infection"] = df["broad_infection"].fillna(False).astype(bool)
    df["strict_infection"] = df["strict_infection"].fillna(False).astype(bool)

    df["sepsis_broad"] = (df["broad_infection"] & df["organ_dysfunction"]).astype(bool)
    df["sepsis_strict"] = (df["strict_infection"] & df["organ_dysfunction"]).astype(bool)

    df["sepsis_broad_onset_offset_approx"] = df["broad_infection_offset"].where(df["sepsis_broad"])
    df["sepsis_strict_onset_offset_approx"] = df["strict_infection_offset"].where(df["sepsis_strict"])

    return df
