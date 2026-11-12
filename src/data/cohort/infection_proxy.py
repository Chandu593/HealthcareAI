"""
Phase 3c -- Suspected-infection proxy.

Per the user's explicit design decision (2026-08-08 chat), this builds
TWO cohort definitions rather than picking one:

  BROAD  (sensitivity analysis): antibiotic administration alone.
  STRICT (primary analysis):     antibiotic administration WITH a
                                  microLab culture order within
                                  `corroboration_window_minutes` of the
                                  antibiotic start.

Neither is claimed to be a validated, published implementation of
Sepsis-3's "suspected infection" criterion -- the reference papers in
this project do not specify their extraction logic at that resolution,
so this is this project's own defensible approximation, reported as
both a strict primary and a broad sensitivity cohort rather than a
single unexplained choice (see docs/sepsis_label_methodology.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.cohort.antibiotic_reference import is_antibiotic


DEDUP_COLUMNS = ["patientunitstayid", "drugname", "drugstartoffset", "drugstopoffset"]


def combine_medication_sources(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Union `medication`, `medication_imputed`, and `medication_imputed1`
    (whichever are actually passed in) into one frame for antibiotic
    detection, deduplicating exact repeated (patient, drugname, start,
    stop) rows.

    WHY UNION + DEDUP RATHER THAN PICKING ONE TABLE: this project has not
    confirmed whether `medication_imputed`/`medication_imputed1` are
    independent order sets or different imputation passes over the same
    underlying `medication` rows (see schema_registry.py docstring point
    2). Picking just one risks silently discarding real antibiotic
    evidence the others might contain; naively concatenating all three
    without dedup risks double-counting the same clinical event as two
    separate antibiotic orders, inflating the broad/strict infection
    proxy. The exact-row-match dedup is a middle ground, not a proven
    fix -- if the imputed variants alter values (e.g. filled-in dosage)
    the same real order could still survive dedup as an apparent
    duplicate order under this heuristic. Flagged, not silently trusted:
    if this matters for your analysis, verify the actual relationship
    between these tables against your real data before treating
    `n_antibiotic_orders` as precise.
    """
    frames = []
    for source_name, df in tables.items():
        if df is None or len(df) == 0:
            continue
        tagged = df.copy()
        tagged["_source_table"] = source_name
        frames.append(tagged)

    if not frames:
        return pd.DataFrame(columns=["patientunitstayid", "drugname", "drugstartoffset", "drugstopoffset", "_source_table"])

    combined = pd.concat(frames, ignore_index=True, sort=False)
    dedup_cols = [c for c in DEDUP_COLUMNS if c in combined.columns]
    combined = combined.drop_duplicates(subset=dedup_cols, keep="first")
    return combined


@dataclass
class InfectionProxyConfig:
    corroboration_window_minutes: int = 1440  # +/- 24h between antibiotic start and culture order
    lookback_minutes: int = 1440              # allow antibiotics starting up to 24h pre-ICU-admission (offset can be negative)
    lookahead_minutes: int = 2880             # ...and up to 48h into the stay


def compute_infection_proxy(
    medication: pd.DataFrame,
    micro_lab: pd.DataFrame,
    config: InfectionProxyConfig,
) -> pd.DataFrame:
    """Returns one row per patientunitstayid with:
      - broad_infection: bool, antibiotic administered in window
      - broad_infection_offset: earliest qualifying antibiotic drugstartoffset
      - strict_infection: bool, broad_infection AND culture corroboration
      - strict_infection_offset: offset of the earliest corroborated pair
      - n_antibiotic_orders, n_cultures_ordered: raw counts for the cohort report
    """
    med = medication.copy()
    med["is_abx"] = is_antibiotic(med["drugname"])
    abx = med[
        med["is_abx"]
        & (med["drugstartoffset"] >= -config.lookback_minutes)
        & (med["drugstartoffset"] <= config.lookahead_minutes)
    ].copy()

    broad = (
        abx.groupby("patientunitstayid")["drugstartoffset"]
        .min()
        .rename("broad_infection_offset")
        .reset_index()
    )
    broad["broad_infection"] = True

    n_abx = (
        med[med["is_abx"]].groupby("patientunitstayid").size().rename("n_antibiotic_orders")
    )
    n_cultures = (
        micro_lab.groupby("patientunitstayid").size().rename("n_cultures_ordered")
    )

    # Strict: for each antibiotic order, check whether ANY culture in
    # micro_lab for the same patient falls within the corroboration
    # window of that antibiotic's start offset. Done via a merge on
    # patientunitstayid then a window filter, which is fine at this
    # project's scale (debug/subset cohorts); for full-mode scale this
    # merge should be chunked per hospital rather than done globally.
    pairs = abx[["patientunitstayid", "drugstartoffset"]].merge(
        micro_lab[["patientunitstayid", "culturetakenoffset"]],
        on="patientunitstayid",
        how="inner",
    )
    pairs["delta"] = (pairs["drugstartoffset"] - pairs["culturetakenoffset"]).abs()
    corroborated = pairs[pairs["delta"] <= config.corroboration_window_minutes]
    strict = (
        corroborated.groupby("patientunitstayid")["drugstartoffset"]
        .min()
        .rename("strict_infection_offset")
        .reset_index()
    )
    strict["strict_infection"] = True

    all_patients = pd.Index(
        set(medication["patientunitstayid"]) | set(micro_lab["patientunitstayid"]),
        name="patientunitstayid",
    )
    result = pd.DataFrame(index=all_patients).reset_index()
    result = result.merge(broad, on="patientunitstayid", how="left")
    result = result.merge(strict, on="patientunitstayid", how="left")
    result = result.merge(n_abx, on="patientunitstayid", how="left")
    result = result.merge(n_cultures, on="patientunitstayid", how="left")

    result["broad_infection"] = result["broad_infection"].fillna(False).astype(bool)
    result["strict_infection"] = result["strict_infection"].fillna(False).astype(bool)
    result["n_antibiotic_orders"] = result["n_antibiotic_orders"].fillna(0).astype(int)
    result["n_cultures_ordered"] = result["n_cultures_ordered"].fillna(0).astype(int)

    return result
