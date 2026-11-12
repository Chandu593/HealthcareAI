"""
Phase 5a -- Tabular feature engineering.

Builds a numeric feature matrix from `patient` + `apachePredVar` +
`diagnosis`, per Section 11's requirements: numerical preprocessing,
categorical encoding, missing-value handling, feature selection,
leakage prevention.

DESIGN CHOICE -- ALLOWLIST, NOT BLOCKLIST: every feature this module
produces is explicitly listed in PATIENT_FEATURES / APACHEPREDVAR_FEATURES
below, with a reason. This directly satisfies "every
feature used must have a documented rationale" -- a blocklist approach
(exclude known-bad columns, use everything else) would silently pull in
any new column added to these tables later without anyone deciding it
belongs. See LEAKAGE_NOTES for what was deliberately left out and why.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature allowlists
# ---------------------------------------------------------------------------

# patient: admission-time demographics and context only. Explicitly
# EXCLUDED (not in this list): hospitaldischargestatus/unitdischargestatus
# (the labels themselves), hospitaldischargelocation/unitdischargelocation
# ("Death" is a literal possible value here), dischargeweight (measured
# at discharge, unavailable at prediction time), hospitaldischargeoffset/
# unitdischargeoffset (length-of-stay, a downstream consequence of
# severity/outcome, not an admission-time predictor), hospitaldischargeyear
# (no predictive value, drop).
PATIENT_NUMERIC_FEATURES = ["age", "admissionheight", "admissionweight"]
PATIENT_CATEGORICAL_FEATURES = [
    "gender", "ethnicity", "hospitaladmitsource", "unitadmitsource",
    "unitstaytype", "apacheadmissiondx", "hospital_region", "hospital_teachingstatus",
]

# apachePredVar: comorbidity flags + admission/hospital context ONLY.
# Explicitly EXCLUDED (not in this list):
#   - `diedinhospital`: the mortality outcome itself, present directly
#     in this table (see schema_registry.py's leakage_risk_columns).
#   - pao2, fio2, day1pao2, day1fio2, creatinine, verbal, motor, eyes,
#     day1verbal, day1motor, day1eyes, ventday1: these were CONSUMED to
#     construct the sepsis label in Phase 3 (sofa_like_scoring.py). This
#     is a different, subtler kind of leakage than diedinhospital: it's
#     not that these values reveal the future, it's that the sepsis
#     label was mechanically computed FROM them via a threshold rule
#     (sofa_like_total >= 2). Including them as input features would let
#     the model partially reconstruct that rule rather than learn a
#     genuinely predictive pattern, inflating apparent performance
#     without real validity. Since Phase 10's multi-task heads currently
#     share ONE fused embedding for both mortality and sepsis (no
#     task-specific feature subsets yet), the conservative choice is to
#     exclude them from the shared representation entirely, at some
#     cost to the mortality task's potential signal from admission
#     physiology. Revisit if/when task-specific feature masking exists.
#   - `dischargelocation`: same discharge-outcome concern as patient's
#     hospitaldischargelocation above.
APACHEPREDVAR_NUMERIC_FEATURES = [
    "diabetes", "electivesurgery", "thrombolytics", "aids", "hepaticfailure",
    "lymphoma", "metastaticcancer", "leukemia", "immunosuppression",
    "cirrhosis", "bedcount", "readmit", "visitnumber",
]
APACHEPREDVAR_CATEGORICAL_FEATURES = ["admitsource"]
# FIXED 2026-08-24: `teachtype`/`region` used to live here, sourced
# from apachePredVar. Real-data investigation (prompted by teachtype/
# region STILL showing vocab=1 even after the 2026-08-23 full-cohort
# fix below) found apachePredVar.teachtype/region are genuinely
# CONSTANT across the entire real 171,177-row apachePredVar table in
# this eICU export (teachtype=0 and region=3 for every single row --
# confirmed via direct `.value_counts()` against the raw CSV, bypassing
# this pipeline entirely). Not a code bug -- a property of this data.
# The real diversity lives in `hospital.csv`'s `teachingstatus`/`region`
# columns instead (confirmed: 189 non-teaching/19 teaching hospitals,
# 4 real regions + some missing, across the 208 real hospitals) --
# see PATIENT_CATEGORICAL_FEATURES' `hospital_region`/
# `hospital_teachingstatus` above, merged onto cohort_patients via
# `hospitalid` in run_pipeline.py's Phase 5 before vocabulary
# determination runs. See docs/tabular_modality.md and
# IMPLEMENTATION_TRACKER.md for the full incident writeup.

DEFAULT_TOP_K_CATEGORICAL = 10

_EICU_MISSING_SENTINELS = (-1, -1.0, "-1")


def parse_age(age_series: pd.Series) -> pd.Series:
    """eICU stores age as a string/object column. Values are usually a
    plain integer string ("70"), but ages over 89 are HIPAA-mandated
    top-coded as "> 89" (exact punctuation varies across exports). This
    maps that band to 90 as a reasonable point estimate rather than
    dropping every patient over 89 as missing. Anything else
    unparseable becomes NaN, imputed downstream like any other missing
    numeric feature -- not silently coerced to 0."""
    def _parse_one(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if s in ("", "nan", "None"):
            return np.nan
        if ">" in s:
            return 90.0  # HIPAA top-coded band, e.g. "> 89"
        try:
            return float(s)
        except ValueError:
            return np.nan
    return age_series.apply(_parse_one)


def clean_eicu_sentinels(series: pd.Series) -> pd.Series:
    """Same -1 'not recorded' convention documented in
    src/data/cohort/eicu_missing.py, reused here since apachePredVar's
    admission-context columns use the same convention."""
    out = series.copy()
    mask = out.isin(_EICU_MISSING_SENTINELS)
    out[mask] = np.nan
    return out


def determine_categorical_vocabulary(values: pd.Series, top_k: int = DEFAULT_TOP_K_CATEGORICAL) -> list[str]:
    """Returns up to top_k most frequent non-null values.

    Ranks whatever `values` it's given -- the caller decides the scope.
    As of the 2026-08-2x vocabulary-sampling-bias fix, Phase 5
    (run_pipeline.py) calls this directly on the FULL cohort for
    `patient`/`apachePredVar` fields via
    determine_full_cohort_vocabularies() below (safe because both
    tables are already loaded once for the whole cohort upstream, not
    hospital-sharded), and uses the sharded-aggregation variant
    determine_categorical_vocabulary_from_value_counts() for the
    genuinely large, hospital-sharded `diagnosis` table instead.

    Phase 4's `select_lab_features()` (time_binning.py) still uses a
    genuinely BOUNDED, seeded-random hospital sample for its own
    (different) purpose -- `lab` is tens of millions of rows, too large
    to load in full just to check which labname substrings exist in
    this cohort's data.

    HISTORY: prior to the 2026-08-2x fix, Phase 5 always called this on
    a sample of the 10 hospitals with the SMALLEST hospital IDs
    (`list(partition_result.partitions.keys())[:10]`, since
    `hospital_partition.py` builds that dict via
    `sorted(available_hospitals)`) -- not a representative sample. On
    the real 199,108-patient / 208-hospital run this silently collapsed
    `apachePredVar.teachtype` and `apachePredVar.region` to a single
    category each. See docs/tabular_modality.md for the full writeup
    and IMPLEMENTATION_TRACKER.md for the incident record.
    """
    counts = values.dropna().value_counts()
    return counts.head(top_k).index.tolist()


def accumulate_value_counts(running: "pd.Series | None", new_values: pd.Series) -> pd.Series:
    """Folds one hospital's category counts into a running aggregate,
    without ever concatenating raw rows across hospitals.

    Used to build a full-cohort categorical vocabulary over a
    hospital-sharded table (currently just `diagnosis`) one shard at a
    time -- see determine_diagnosis_vocabulary_from_shards() below.
    Only aggregate counts (a small Series indexed by category) cross
    the per-hospital boundary, never patient-level rows -- consistent
    with the project's hospital-sharding architecture
    (docs/memory_efficiency.md), which never pools raw large-table data
    across hospitals in memory at once.
    """
    counts = new_values.dropna().value_counts()
    if running is None:
        return counts
    return running.add(counts, fill_value=0)


def determine_categorical_vocabulary_from_value_counts(
    counts: "pd.Series | None", top_k: int = DEFAULT_TOP_K_CATEGORICAL,
) -> list[str]:
    """Same ranking as determine_categorical_vocabulary(), but takes
    pre-aggregated counts (e.g. accumulate_value_counts()'s running
    total) instead of raw values -- lets a vocabulary be computed over
    an entire sharded table's category frequencies without ever holding
    more than one hospital's raw rows in memory at once."""
    if counts is None or len(counts) == 0:
        return []
    return counts.sort_values(ascending=False).head(top_k).index.tolist()


def attach_hospital_features(cohort_patients: pd.DataFrame, hospital_df: pd.DataFrame) -> pd.DataFrame:
    """Merges `hospital.csv`'s `region`/`teachingstatus` onto
    `cohort_patients` via `hospitalid`, as `hospital_region`/
    `hospital_teachingstatus`.

    FIXED 2026-08-24: `apachePredVar.teachtype`/`region` (this
    project's original source for hospital-characterization features)
    turned out to be genuinely constant across the whole real
    apachePredVar table in this eICU export -- see
    APACHEPREDVAR_CATEGORICAL_FEATURES' comment. `hospital.csv` is
    where the real per-hospital diversity actually lives (confirmed:
    189 non-teaching/19 teaching, 4 real regions + some missing, across
    208 real hospitals) -- exactly the kind of "site metadata... useful
    for characterizing non-IID FL clients" schema_registry.py already
    described that table as.

    `region`/`teachingstatus` are OPTIONAL columns in hospital's schema
    (schema_registry.py) -- if either is absent, that output column is
    filled with NaN rather than raising, matching this project's
    established defensive-missing-data pattern elsewhere.
    """
    hospital_cols = [c for c in ("region", "teachingstatus") if c in hospital_df.columns]
    rename_map = {"region": "hospital_region", "teachingstatus": "hospital_teachingstatus"}
    # Defensive: a left-merge against a hospital_id-duplicated reference
    # table would silently MULTIPLY patient rows -- hospital.csv should
    # be one row per hospital, but this guards against that silently
    # corrupting the cohort if it isn't.
    hospital_ref = hospital_df[["hospitalid"] + hospital_cols].drop_duplicates(subset="hospitalid", keep="first")
    merged = cohort_patients.merge(
        hospital_ref.rename(columns=rename_map), on="hospitalid", how="left",
    ) if hospital_cols else cohort_patients.copy()
    assert len(merged) == len(cohort_patients), "hospital merge changed row count -- hospital.csv has duplicate hospitalid rows"

    for out_col in rename_map.values():
        if out_col not in merged.columns:
            merged[out_col] = np.nan
    return merged


def determine_full_cohort_vocabularies(
    cohort_patients: pd.DataFrame,
    apache_pred_var: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K_CATEGORICAL,
    top_k_overrides: "dict[str, int] | None" = None,
) -> dict[str, list[str]]:
    """Categorical vocabularies for `patient` (including hospital-
    sourced fields attached via attach_hospital_features()) +
    `apachePredVar` fields, computed from the FULL cohort rather than a
    bounded hospital sample.

    Safe re: the project's never-pool-raw-data-across-hospitals rule
    specifically because `cohort_patients` and `apache_pred_var` are
    already loaded ONCE for the whole cohort upstream, in
    run_pipeline.py's Phase 5 -- both tables (and hospital.csv, merged
    in by the caller before this runs) are small enough that they were
    never hospital-sharded in the first place (see the "apachePredVar
    is small -- load once for the whole cohort" comment there). Using
    the full data that's already in memory introduces no NEW
    cross-hospital pooling; it just stops throwing away all but 10
    hospitals' worth of it. Contrast with
    determine_diagnosis_vocabulary_from_shards() below, which handles
    the genuinely large, hospital-sharded `diagnosis` table differently
    to preserve the no-raw-pooling architecture for tables that
    actually need it.

    `top_k_overrides` lets specific high-cardinality columns (e.g.
    `apacheadmissiondx`) use a larger top_k than the default, via
    `modalities.tabular.top_k_categorical_overrides` in config.
    """
    overrides = top_k_overrides or {}
    vocabularies: dict[str, list[str]] = {}
    for col in PATIENT_CATEGORICAL_FEATURES:
        k = overrides.get(col, top_k)
        series = cohort_patients[col] if col in cohort_patients.columns else None
        vocabularies[col] = determine_categorical_vocabulary(series, k) if series is not None else []
    for col in APACHEPREDVAR_CATEGORICAL_FEATURES:
        k = overrides.get(col, top_k)
        series = clean_eicu_sentinels(apache_pred_var[col]) if col in apache_pred_var.columns else None
        vocabularies[col] = determine_categorical_vocabulary(series, k) if series is not None else []
    return vocabularies


def determine_diagnosis_vocabulary_from_shards(
    load_shard_fn,
    hospital_ids,
    top_k: int = DEFAULT_TOP_K_CATEGORICAL,
) -> list[str]:
    """Top-level diagnosis-category vocabulary aggregated across ALL
    given hospitals' shards, one hospital at a time -- never holding
    more than one hospital's raw diagnosis rows in memory at once (see
    accumulate_value_counts()).

    Fixes the same vocabulary-sampling-bias bug as
    determine_full_cohort_vocabularies(), for the one Phase 5 source
    table (`diagnosis`) that IS genuinely hospital-sharded (too large
    to load once for the whole cohort like patient/apachePredVar) --
    so unlike that function, this one preserves the hospital-by-
    hospital, no-raw-pooling discipline while still covering every
    hospital instead of a biased 10-hospital sample.

    `load_shard_fn` is a callable(hospital_id) -> DataFrame with a
    `diagnosisstring` column -- typically `load_hospital_shard` bound
    to a manifest in run_pipeline.py, but injectable here so this can
    be unit-tested without real shard files on disk.
    """
    running_counts = None
    for hid in hospital_ids:
        shard = load_shard_fn(hid)
        if shard is None or shard.empty or "diagnosisstring" not in shard.columns:
            continue
        top_level = shard["diagnosisstring"].str.split("|").str[0]
        running_counts = accumulate_value_counts(running_counts, top_level)
    return determine_categorical_vocabulary_from_value_counts(running_counts, top_k)


def _sanitize_column_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()


def one_hot_with_vocabulary(series: pd.Series, prefix: str, vocabulary: list[str]) -> pd.DataFrame:
    """One-hot encodes `series` against a FIXED vocabulary (not derived
    from `series` itself), plus a `{prefix}_other` bucket for any
    non-null value outside the vocabulary. Using a fixed, externally
    supplied vocabulary (rather than each caller re-deriving its own)
    is what keeps column sets identical across every hospital's feature
    block -- required for hospitals' tabular features to concatenate
    into one aligned tensor later."""
    out = pd.DataFrame(index=series.index)
    for value in vocabulary:
        col_name = f"{prefix}_{_sanitize_column_token(value)}"
        out[col_name] = (series == value).astype(float)
    known_mask = series.isin(vocabulary)
    out[f"{prefix}_other"] = (series.notna() & ~known_mask).astype(float)
    return out


def build_patient_features(patient_df: pd.DataFrame, vocabularies: dict[str, list[str]]) -> pd.DataFrame:
    df = patient_df.set_index("patientunitstayid")
    blocks = [pd.DataFrame(index=df.index)]

    for col in PATIENT_NUMERIC_FEATURES:
        if col == "age" and col in df.columns:
            blocks[0][col] = parse_age(df[col])
        elif col in df.columns:
            blocks[0][col] = pd.to_numeric(df[col], errors="coerce")
        else:
            blocks[0][col] = np.nan

    for col in PATIENT_CATEGORICAL_FEATURES:
        series = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
        vocab = vocabularies.get(col, [])
        blocks.append(one_hot_with_vocabulary(series, f"patient_{col}", vocab))

    return pd.concat(blocks, axis=1)


def build_apachepredvar_features(apv_df: pd.DataFrame, vocabularies: dict[str, list[str]]) -> pd.DataFrame:
    df = apv_df.set_index("patientunitstayid")
    numeric_block = pd.DataFrame(index=df.index)
    for col in APACHEPREDVAR_NUMERIC_FEATURES:
        if col in df.columns:
            numeric_block[col] = clean_eicu_sentinels(pd.to_numeric(df[col], errors="coerce"))
        else:
            numeric_block[col] = np.nan

    blocks = [numeric_block]
    for col in APACHEPREDVAR_CATEGORICAL_FEATURES:
        series = clean_eicu_sentinels(df[col]) if col in df.columns else pd.Series(np.nan, index=df.index)
        vocab = vocabularies.get(col, [])
        blocks.append(one_hot_with_vocabulary(series, f"apv_{col}", vocab))

    return pd.concat(blocks, axis=1)


def build_diagnosis_features(
    diagnosis_df: pd.DataFrame,
    patient_ids: list[int],
    vocabulary: list[str],
    lookback_minutes: int,
    cutoff_minutes: int,
) -> pd.DataFrame:
    """Multi-hot presence indicators for the top-level diagnosis
    category (the segment before the first '|' in `diagnosisstring`,
    e.g. "cardiovascular" from "cardiovascular|chest pain / ASHD|...").

    TIME WINDOWING (leakage prevention, using
    `temporal.prediction_horizon_offset_minutes` conceptually): only
    diagnosis entries recorded within
    [-lookback_minutes, +cutoff_minutes] of ICU admission are used.
    Diagnoses added later in a long stay (e.g. a "sepsis" diagnosis
    entered on day 5) reflect clinical knowledge gained AFTER the
    observation window this project predicts from, and including them
    would leak that later knowledge back into an "admission-time"
    tabular feature set.
    """
    index = pd.Index(patient_ids, name="patientunitstayid")
    if diagnosis_df.empty:
        empty = pd.DataFrame(index=index)
        for cat in vocabulary:
            empty[f"diag_{_sanitize_column_token(cat)}"] = 0.0
        empty["diag_other"] = 0.0
        return empty

    df = diagnosis_df.copy()
    df = df[
        (df["diagnosisoffset"] >= -lookback_minutes) & (df["diagnosisoffset"] <= cutoff_minutes)
    ]
    df["top_level"] = df["diagnosisstring"].str.split("|").str[0]

    result = pd.DataFrame(index=index)
    for cat in vocabulary:
        col_name = f"diag_{_sanitize_column_token(cat)}"
        patients_with_cat = set(df.loc[df["top_level"] == cat, "patientunitstayid"])
        result[col_name] = result.index.to_series().isin(patients_with_cat).astype(float)

    known_cats = set(vocabulary)
    other_patients = set(df.loc[~df["top_level"].isin(known_cats), "patientunitstayid"])
    result["diag_other"] = result.index.to_series().isin(other_patients).astype(float)

    return result


def assemble_tabular_features(
    patient_df: pd.DataFrame,
    apv_df: pd.DataFrame,
    diagnosis_df: pd.DataFrame,
    patient_ids: list[int],
    vocabularies: dict,
    diagnosis_vocabulary: list[str],
    diagnosis_lookback_minutes: int,
    diagnosis_cutoff_minutes: int,
) -> pd.DataFrame:
    """Merges the three feature blocks into one (patients x features)
    frame, reindexed to `patient_ids` so every hospital's block has
    identical columns in identical order regardless of which of the
    three source tables happened to have data for which patients."""
    patient_features = build_patient_features(patient_df, vocabularies)
    apv_features = build_apachepredvar_features(apv_df, vocabularies)
    diag_features = build_diagnosis_features(
        diagnosis_df, patient_ids, diagnosis_vocabulary,
        diagnosis_lookback_minutes, diagnosis_cutoff_minutes,
    )

    combined = patient_features.join(apv_features, how="outer").join(diag_features, how="outer")
    combined = combined.reindex(patient_ids)
    return combined
