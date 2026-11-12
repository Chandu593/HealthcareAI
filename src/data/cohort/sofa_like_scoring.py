"""
Phase 3b -- SOFA-like organ dysfunction scoring.

IMPORTANT LIMITATIONS (read docs/sepsis_label_methodology.md for full
justification -- summarized here so this stays visible at the code
site, not just in a doc file no one opens):

1. This is a SINGLE-TIMEPOINT score aggregated over an admission-window
   (default first 24h, `observation_window_minutes`), NOT a continuous
   SOFA trajectory. Sepsis-3's "SOFA increase >= 2 from baseline" cannot
   be computed without vitalPeriodic-style repeated measurements. The
   threshold here is applied to an ABSOLUTE score, not a delta.

2. Renal domain omits urine output (no intakeOutput table) -- creatinine
   only.

3. Cardiovascular domain omits MAP (no vitalPeriodic table) and cannot
   reliably parse vasopressor dose-per-kg from the free-text `drugrate`
   field (inconsistent units across drug names) -- it can only detect
   vasopressor PRESENCE, scored as a flat 2 points, not the standard
   0/1/2/3/4 dose-tiered SOFA cardiovascular scale.

4. `lab.labname` string matching for platelets/bilirubin uses
   case-insensitive substring matching (e.g. "platelet" matches
   "platelets", "platelets x 1000") because this project has not
   verified the exact labname vocabulary against real eICU data (only
   a 5-row sample was available). VERIFY against your actual
   `lab['labname'].unique()` before trusting Phase 3 output on real
   data -- if the substring doesn't match anything, that domain's score
   will be entirely NaN/excluded, silently under-scoring dysfunction.

Because of (2) and (3), `max_possible_score` per patient can be less
than 24 (the full 6-domain x 4-point SOFA max) even when scoring, and is
reported alongside the raw sum so a patient with genuinely fewer
observed domains isn't compared apples-to-oranges against one with all
six.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.cohort.antibiotic_reference import is_vasopressor
from src.data.cohort.eicu_missing import clean_sentinels

RESPIRATORY_MAX = 4
COAGULATION_MAX = 4
LIVER_MAX = 4
RENAL_MAX = 4  # urine-output component not available -- creatinine-only ceiling still 4
CNS_MAX = 4
CARDIOVASCULAR_MAX = 2  # capped at 2 (presence-only, no dose-tiered 3/4) -- see module docstring point 3


def _score_respiratory(pao2: pd.Series, fio2: pd.Series, on_vent: pd.Series) -> pd.Series:
    fio2_frac = fio2.where(fio2 <= 1.0, fio2 / 100.0)  # normalize percent vs fraction inputs
    ratio = pao2 / fio2_frac.replace(0, np.nan)
    score = pd.Series(np.nan, index=pao2.index)
    score[ratio >= 400] = 0
    score[(ratio >= 300) & (ratio < 400)] = 1
    score[(ratio >= 200) & (ratio < 300)] = 2
    score[(ratio >= 100) & (ratio < 200)] = 3
    score[ratio < 100] = 4
    # Approximation: SOFA bands 3-4 formally require ventilatory support;
    # without continuous vent data we only bump a band-2 result to 3 if
    # ventday1 indicates ventilation on day 1. Documented, not silent.
    bump_mask = (score == 2) & (on_vent == 1)
    score[bump_mask] = 3
    return score


def _score_coagulation(platelets: pd.Series) -> pd.Series:
    score = pd.Series(np.nan, index=platelets.index)
    score[platelets >= 150] = 0
    score[(platelets >= 100) & (platelets < 150)] = 1
    score[(platelets >= 50) & (platelets < 100)] = 2
    score[(platelets >= 20) & (platelets < 50)] = 3
    score[platelets < 20] = 4
    return score


def _score_liver(bilirubin: pd.Series) -> pd.Series:
    score = pd.Series(np.nan, index=bilirubin.index)
    score[bilirubin < 1.2] = 0
    score[(bilirubin >= 1.2) & (bilirubin < 2.0)] = 1
    score[(bilirubin >= 2.0) & (bilirubin < 6.0)] = 2
    score[(bilirubin >= 6.0) & (bilirubin < 12.0)] = 3
    score[bilirubin >= 12.0] = 4
    return score


def _score_renal(creatinine: pd.Series) -> pd.Series:
    score = pd.Series(np.nan, index=creatinine.index)
    score[creatinine < 1.2] = 0
    score[(creatinine >= 1.2) & (creatinine < 2.0)] = 1
    score[(creatinine >= 2.0) & (creatinine < 3.5)] = 2
    score[(creatinine >= 3.5) & (creatinine < 5.0)] = 3
    score[creatinine >= 5.0] = 4
    return score


def _score_cns(gcs_total: pd.Series) -> pd.Series:
    score = pd.Series(np.nan, index=gcs_total.index)
    score[gcs_total == 15] = 0
    score[(gcs_total >= 13) & (gcs_total < 15)] = 1
    score[(gcs_total >= 10) & (gcs_total < 13)] = 2
    score[(gcs_total >= 6) & (gcs_total < 10)] = 3
    score[gcs_total < 6] = 4
    return score


@dataclass
class SofaLikeConfig:
    threshold: int = 2
    observation_window_minutes: int = 1440  # first 24h
    renal_urine_output_available: bool = False
    cardiovascular_map_available: bool = False


def compute_sofa_like_scores(
    apache_pred_var: pd.DataFrame,
    lab: pd.DataFrame,
    infusion_drug: pd.DataFrame,
    config: SofaLikeConfig,
) -> pd.DataFrame:
    """Returns one row per patientunitstayid with per-domain scores,
    the summed sofa_like_total, and max_possible_score for that patient
    given which domains actually had data.
    """
    apv = clean_sentinels_frame(apache_pred_var, [
        "pao2", "fio2", "day1pao2", "day1fio2", "creatinine",
        "verbal", "motor", "eyes", "day1verbal", "day1motor", "day1eyes",
        "ventday1",
    ])

    pao2 = apv["day1pao2"].where(apv["day1pao2"].notna(), apv.get("pao2"))
    fio2 = apv["day1fio2"].where(apv["day1fio2"].notna(), apv.get("fio2"))
    verbal = apv["day1verbal"].where(apv["day1verbal"].notna(), apv.get("verbal"))
    motor = apv["day1motor"].where(apv["day1motor"].notna(), apv.get("motor"))
    eyes = apv["day1eyes"].where(apv["day1eyes"].notna(), apv.get("eyes"))
    gcs_total = verbal + motor + eyes
    on_vent = apv.get("ventday1")

    resp = _score_respiratory(pao2, fio2, on_vent)
    renal = _score_renal(apv["creatinine"])
    cns = _score_cns(gcs_total)

    domain_df = pd.DataFrame({
        "patientunitstayid": apv["patientunitstayid"],
        "respiratory_score": resp,
        "renal_score": renal,
        "cns_score": cns,
    })

    lab_window = lab[
        (lab["labresultoffset"] >= 0) & (lab["labresultoffset"] <= config.observation_window_minutes)
    ]
    platelet_mask = lab_window["labname"].str.contains("platelet", case=False, na=False)
    bilirubin_mask = lab_window["labname"].str.contains("bilirubin", case=False, na=False)

    worst_platelets = (
        lab_window[platelet_mask].groupby("patientunitstayid")["labresult"].min()
    )  # lower platelets = worse
    worst_bilirubin = (
        lab_window[bilirubin_mask].groupby("patientunitstayid")["labresult"].max()
    )  # higher bilirubin = worse

    coag = _score_coagulation(worst_platelets).rename("coagulation_score")
    liver = _score_liver(worst_bilirubin).rename("liver_score")

    domain_df = domain_df.merge(coag, left_on="patientunitstayid", right_index=True, how="left")
    domain_df = domain_df.merge(liver, left_on="patientunitstayid", right_index=True, how="left")

    inf_window = infusion_drug[
        (infusion_drug["infusionoffset"] >= 0)
        & (infusion_drug["infusionoffset"] <= config.observation_window_minutes)
    ].copy()
    inf_window["is_vaso"] = is_vasopressor(inf_window["drugname"])
    vaso_present = inf_window.groupby("patientunitstayid")["is_vaso"].any()
    cardio = vaso_present.map({True: 2, False: 0}).rename("cardiovascular_score")

    domain_df = domain_df.merge(cardio, left_on="patientunitstayid", right_index=True, how="left")
    # Patients with zero infusionDrug rows in the window get NaN from the
    # merge (not 0) -- absence of infusion records is not the same claim
    # as "confirmed no vasopressor," though in practice for eICU it
    # usually does mean none was given. Left as NaN to be explicit about
    # what wasn't observed vs. what was observed-and-negative.

    score_cols = [
        "respiratory_score", "coagulation_score", "liver_score",
        "renal_score", "cns_score", "cardiovascular_score",
    ]
    domain_df["sofa_like_total"] = domain_df[score_cols].sum(axis=1, min_count=1)
    domain_df["n_domains_observed"] = domain_df[score_cols].notna().sum(axis=1)
    max_per_domain = {
        "respiratory_score": RESPIRATORY_MAX, "coagulation_score": COAGULATION_MAX,
        "liver_score": LIVER_MAX, "renal_score": RENAL_MAX, "cns_score": CNS_MAX,
        "cardiovascular_score": CARDIOVASCULAR_MAX,
    }
    domain_df["max_possible_score"] = sum(
        domain_df[c].notna() * max_per_domain[c] for c in score_cols
    )
    domain_df["organ_dysfunction"] = domain_df["sofa_like_total"] >= config.threshold

    return domain_df


def clean_sentinels_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = clean_sentinels(out[c])
    return out
