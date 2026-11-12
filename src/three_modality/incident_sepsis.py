"""INCIDENT sepsis-proxy label: a new sepsis-related event with onset strictly after the first ICU hours.

Completely separate from the Fed-THOR sepsis label (src/data/cohort/sepsis_label.py + sofa_like_scoring.py combine antibiotics from
-24h..+48h with FIRST-24h-WORST SOFA-like values, which overlaps the observation window). Nothing here imports or calls that
combination logic; only (a) the antibiotic/vasopressor keyword lists and (b) the pure SOFA band functions for coagulation/liver/renal are
reused so the bands stay identical. See docs/three_modality_incident_sepsis.md for the full definition.

TEMPORAL LOGIC (all times = minutes since ICU unit admission, binned to 1-hour bins: hour = ceil(minutes / 60))
  C = cutoff_hours (24), H = horizon_end_hours (96), W = co_occurrence_window_hours (24), P = prevalent_lookback_hours (24)
  * Observation window (model inputs, handled elsewhere): offsets <= C*60.  Incident-event window: C < offset <= H*60  (bins C+1..H).
  * Evidence is dropped if it falls after min(ICU exit, H*60): the at-risk window ends when the patient leaves the ICU.
  * Infection evidence (default): an antibiotic START (keyword match on drugname, any configured medication table).
        "antibiotic_plus_culture": antibiotic start AND a microLab culture within +/- W hours of it, BOTH inside the window being evaluated.
  * Organ dysfunction at grid hour g: SOFA-like total >= threshold, where each domain takes its WORST value among evidence in the trailing
        window (max(window_start, g-W+1) .. g).  Domains: respiratory (paO2/FiO2 pairs with identical lab offset; bands 3-4 need a
        mechanical-ventilation record in the same trailing window), coagulation (platelets), liver (total bilirubin), renal (creatinine),
        cardiovascular (vasopressor infusion present = 2, presence only).  CNS, MAP and urine output are NOT available time-resolved
        in this data export (no nurseCharting/vitalPeriodic/intakeOutput), so the maximum attainable score is 18, not 24.
  * Criteria are MET at grid hour g when organ dysfunction holds at g AND an infection-evidence event lies in the trailing window
        (max(window_start, g-W+1) .. g).  I.e. both elements occurred within W hours of each other, whichever came first; the event hour is
        when the second element arrived.
  * INCIDENT:  criteria first met at some g in [C+1, H], using ONLY evidence in bins C+1..H (window_start = C+1).
  * PREVALENT: criteria met at some g <= C, using evidence in bins -P..C (window_start = -P): excluded, never labelled incident.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cohort.antibiotic_reference import is_antibiotic, is_vasopressor
from src.data.cohort.sofa_like_scoring import _score_coagulation, _score_liver, _score_renal
from src.paper.features import _resolve_table
from src.three_modality.artifacts import atomic_write as _atomic_write
from src.three_modality.artifacts import fingerprint as _fingerprint
from src.three_modality.artifacts import find_reusable_run, list_runs, new_run_dir  # noqa: F401  (list_runs re-exported for run_three_modality)
from src.three_modality.artifacts import sha256_file as _sha256
from src.three_modality.config import INCIDENT_LABEL_DEFINITION_VERSION

logger = logging.getLogger("eicu_fl")

MAX_ATTAINABLE_SOFA_LIKE = 4 + 4 + 4 + 4 + 2   # resp + coag + liver + renal + cardio(presence-only); CNS unavailable time-resolved

# ---- statuses (every patient in the split manifest ends up with exactly one) ---------------------------------------------------
PRE_NOT_IN_PATIENT_TABLE = "pre_excluded_not_in_patient_table"
PRE_INVALID_EXIT = "pre_excluded_invalid_icu_exit_time"
PRE_ENDED_BY_CUTOFF = "pre_excluded_icu_stay_ended_by_cutoff"
EXCL_LABEL_INVALID_HOSPITAL = "excluded_label_invalid_hospital"
EXCL_PREVALENT = "excluded_prevalent"
POSITIVE = "incident_positive"
NEG_COMPLETE = "incident_negative_complete_followup"
NEG_ALIVE_DISCHARGE = "incident_negative_alive_discharge"
CENS_ALIVE = "censored_alive_discharge"
CENS_DEATH = "censored_death_no_event"
CENS_UNKNOWN = "censored_unknown_exit_status"
EXCL_POS_INCOMPLETE = "excluded_positive_incomplete_followup"
PRE_STATUSES = (PRE_NOT_IN_PATIENT_TABLE, PRE_INVALID_EXIT, PRE_ENDED_BY_CUTOFF)
CENSORED_STATUSES = (CENS_ALIVE, CENS_DEATH, CENS_UNKNOWN)
NEGATIVE_STATUSES = (NEG_COMPLETE, NEG_ALIVE_DISCHARGE)

# policy-independent classification of eligible, label-valid, non-prevalent patients
_B_EVENT_FULL, _B_EVENT_EARLY = "event_full_followup", "event_early_exit"
_B_NOEV_FULL, _B_NOEV_ALIVE, _B_NOEV_DEATH, _B_NOEV_UNKNOWN = "noevent_full_followup", "noevent_early_alive", "noevent_early_death", "noevent_early_unknown"

LABEL_COLUMNS = ["patientunitstayid", "hospitalid", "split", "icu_exit_hours", "died_in_icu", "status", "label", "included",
                 "event_hour", "prevalent", "abx_pre_cutoff", "org_dysfunction_incident_ever", "infection_evidence_incident_ever",
                 "max_sofa_like_incident", "followup_hours_in_event_window"]


def hour_bin(minutes) -> np.ndarray:
    """1-hour bin index of an offset in minutes: (h-1)*60 < minutes <= h*60 -> h.  Offset 1440 -> 24 (observation side); 1441 -> 25 (event side)."""
    return np.ceil(np.asarray(minutes, dtype="float64") / 60.0).astype("int64")


# =====================================================================================================================================
# 1. universe + eligibility
# =====================================================================================================================================
def build_universe(patient_df: pd.DataFrame, manifest, cfg: dict) -> pd.DataFrame:
    """One row per split-manifest patient (all partitions; `manifest=None` -> every patient-table row) with columns
    patientunitstayid, hospitalid, split, icu_exit_minutes, died_in_icu, pre_status (NaN = passes the landmark / validity rules)."""
    need = {"patientunitstayid", "hospitalid", "unitdischargeoffset", "unitdischargestatus"}
    missing = need - set(patient_df.columns)
    if missing:
        raise ValueError(f"patient table is missing required column(s) {sorted(missing)}")
    p = patient_df[sorted(need)].copy()
    p["patientunitstayid"] = p["patientunitstayid"].astype("int64")
    if p["patientunitstayid"].duplicated().any():
        raise ValueError("patient table has duplicate patientunitstayid")
    if manifest is not None:
        split_map = {int(k): v for k, v in manifest.patient_to_split.items()}
        u = pd.DataFrame({"patientunitstayid": np.array(sorted(split_map), dtype="int64")})
        u["split"] = u["patientunitstayid"].map(split_map)
    else:
        u = pd.DataFrame({"patientunitstayid": p["patientunitstayid"].to_numpy()})
        u["split"] = "none"
    u = u.merge(p, on="patientunitstayid", how="left")
    in_table = u["patientunitstayid"].isin(p["patientunitstayid"])
    exit_min = pd.to_numeric(u["unitdischargeoffset"], errors="coerce")
    u["icu_exit_minutes"] = exit_min
    st = u["unitdischargestatus"].astype("string").str.strip().str.lower()
    unknown_status = st.isna().to_numpy()
    dead = st.eq("expired").fillna(False).to_numpy(dtype=bool)
    u["died_in_icu"] = np.where(unknown_status, np.nan, dead.astype(float))      # NaN = unknown exit status
    cutoff_min = cfg["cutoff_hours"] * 60
    valid_exit = exit_min.notna() & (exit_min > 0)
    pre = pd.Series(np.nan, index=u.index, dtype="object")
    pre[valid_exit & (exit_min <= cutoff_min)] = PRE_ENDED_BY_CUTOFF
    pre[~valid_exit & in_table] = PRE_INVALID_EXIT
    pre[~in_table] = PRE_NOT_IN_PATIENT_TABLE
    u["pre_status"] = pre
    if u["hospitalid"].notna().all():
        u["hospitalid"] = u["hospitalid"].astype("int64")
    return u[["patientunitstayid", "hospitalid", "split", "icu_exit_minutes", "died_in_icu", "pre_status"]].reset_index(drop=True)


# =====================================================================================================================================
# 2. time-resolved evidence collection (chunked; only eligible patients, only rows inside [-P*60, min(ICU exit, H*60)])
# =====================================================================================================================================
@dataclass
class EvidenceEvents:
    abx: pd.DataFrame = field(default_factory=lambda: _empty("pidx", "hour"))
    culture: pd.DataFrame = field(default_factory=lambda: _empty("pidx", "hour"))
    plt: pd.DataFrame = field(default_factory=lambda: _empty("pidx", "hour", "value"))
    bili: pd.DataFrame = field(default_factory=lambda: _empty("pidx", "hour", "value"))
    creat: pd.DataFrame = field(default_factory=lambda: _empty("pidx", "hour", "value"))
    pf: pd.DataFrame = field(default_factory=lambda: _empty("pidx", "hour", "value"))
    vent: pd.DataFrame = field(default_factory=lambda: _empty("pidx", "hour"))
    vaso: pd.DataFrame = field(default_factory=lambda: _empty("pidx", "hour"))


def _empty(*cols) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64" if c == "value" else "int64") for c in cols})


def _read_filtered(path, usecols, off_col, id_set, idx, cap, lower, chunksize, row_filter=None, on_raw=None):
    """Chunked read -> rows for eligible patients, passing `row_filter`, with -lower <= offset <= that patient's cap. Adds `pidx`, `off`."""
    parts, n_raw = [], 0
    cols = list(dict.fromkeys(["patientunitstayid", off_col] + list(usecols)))
    for ch in pd.read_csv(path, usecols=cols, chunksize=chunksize, low_memory=False):
        ch = ch[ch["patientunitstayid"].isin(id_set)]
        if ch.empty:
            continue
        n_raw += len(ch)
        if on_raw is not None:
            on_raw(ch)
        if row_filter is not None:
            ch = ch[row_filter(ch)]
            if ch.empty:
                continue
        off = pd.to_numeric(ch[off_col], errors="coerce")
        keep = off.notna() & (off >= lower) & (off <= ch["patientunitstayid"].map(cap))
        if not keep.any():
            continue
        ch = ch[keep].copy()
        ch["off"] = off[keep].to_numpy(dtype="float64")
        ch["pidx"] = ch["patientunitstayid"].map(idx).to_numpy(dtype="int64")
        parts.append(ch)
    if not parts:
        return pd.DataFrame(columns=list(cols) + ["off", "pidx"]), n_raw
    return pd.concat(parts, ignore_index=True), n_raw


def _events(df: pd.DataFrame, value_col=None) -> pd.DataFrame:
    if df.empty:
        return _empty("pidx", "hour", *(["value"] if value_col else []))
    out = pd.DataFrame({"pidx": df["pidx"].to_numpy("int64"), "hour": hour_bin(df["off"].to_numpy())})
    if value_col:
        out["value"] = pd.to_numeric(df[value_col], errors="coerce").to_numpy("float64")
        out = out[np.isfinite(out["value"])]
    return out.reset_index(drop=True)


def collect_evidence(src: dict, eligible: pd.DataFrame, cfg: dict) -> tuple[EvidenceEvents, dict]:
    """`src` = phase-1 summary dict (table_paths / data_root).  `eligible` needs patientunitstayid + icu_exit_minutes.
    Returns (events, diagnostics).  Positional patient index `pidx` = row position in `eligible`."""
    H, P, chunk = cfg["horizon_end_hours"], cfg["prevalent_lookback_hours"], int(cfg["chunksize"])
    ids = eligible["patientunitstayid"].to_numpy("int64")
    idx = pd.Series(np.arange(len(ids), dtype="int64"), index=ids)
    cap = pd.Series(np.minimum(eligible["icu_exit_minutes"].to_numpy("float64"), H * 60.0), index=ids)
    id_set, lower = set(ids.tolist()), -P * 60.0
    diag: dict = {"tables_used": {}, "tables_missing": [], "rows_in_cohort": {}, "rows_kept": {}}
    ev = EvidenceEvents()
    od = cfg["organ_dysfunction"]

    def _path(name, required=False):
        p = _resolve_table(src, name)
        if p is None:
            if required:
                raise FileNotFoundError(f"table '{name}' not found (looked in phase-1 table_paths and data_root) -- required for the incident sepsis label")
            diag["tables_missing"].append(name)
            logger.warning("[INCIDENT_SEPSIS] table '%s' not found -- its evidence is treated as unobserved", name)
        else:
            diag["tables_used"][name] = str(p)
        return p

    # ---- medication tables: antibiotic starts + 'has any medication row' (hospital label validity) ----
    ids_with_med: set = set()
    abx_parts = []
    n_med = [0, 0]
    any_med_table = False
    for t in cfg["infection"]["medication_tables"]:
        p = _path(t)
        if p is None:
            continue
        any_med_table = True
        df, n_raw = _read_filtered(p, ["drugname"], "drugstartoffset", id_set, idx, cap, lower, chunk,
                                   row_filter=lambda c: is_antibiotic(c["drugname"]).to_numpy(),
                                   on_raw=lambda c: ids_with_med.update(c["patientunitstayid"].unique().tolist()))
        n_med[0] += n_raw; n_med[1] += len(df)
        abx_parts.append(_events(df))
    if not any_med_table:
        raise FileNotFoundError(f"none of the configured medication tables {cfg['infection']['medication_tables']} were found -- the antibiotic evidence cannot be built")
    ev.abx = pd.concat(abx_parts, ignore_index=True).drop_duplicates() if abx_parts else _empty("pidx", "hour")
    diag["rows_in_cohort"]["medication(all)"], diag["rows_kept"]["medication(all)"] = n_med

    # ---- microLab (only when culture corroboration is requested) ----
    if cfg["infection"]["evidence"] == "antibiotic_plus_culture":
        p = _path("microLab", required=True)
        df, n_raw = _read_filtered(p, [], "culturetakenoffset", id_set, idx, cap, lower, chunk)
        ev.culture = _events(df).drop_duplicates()
        diag["rows_in_cohort"]["microLab"], diag["rows_kept"]["microLab"] = n_raw, len(df)

    # ---- lab: platelets / bilirubin / creatinine / paO2 / FiO2 (exact lab names) ----
    name_to_dom = {n.strip().lower(): d for d, names in od["lab_names"].items() for n in names}
    p = _path("lab", required=True)
    lab_df, n_raw = _read_filtered(p, ["labname", "labresult"], "labresultoffset", id_set, idx, cap, lower, chunk,
                                   row_filter=lambda c: c["labname"].astype("string").str.strip().str.lower().isin(name_to_dom).to_numpy())
    diag["rows_in_cohort"]["lab"], diag["rows_kept"]["lab"] = n_raw, len(lab_df)
    if not lab_df.empty:
        lab_df["dom"] = lab_df["labname"].astype("string").str.strip().str.lower().map(name_to_dom)
        lab_df["val"] = pd.to_numeric(lab_df["labresult"], errors="coerce")
        lab_df = lab_df[lab_df["val"].notna() & (lab_df["val"] >= 0)]
        for dom, attr in (("platelets", "plt"), ("bilirubin", "bili"), ("creatinine", "creat")):
            setattr(ev, attr, _events(lab_df[lab_df["dom"] == dom], "val"))
        pao2 = lab_df[(lab_df["dom"] == "pao2") & (lab_df["val"] > 0)][["pidx", "off", "val"]].rename(columns={"val": "pao2"})
        fio2 = lab_df[lab_df["dom"] == "fio2"][["pidx", "off", "val"]].copy()
        fio2["val"] = fio2["val"].where(fio2["val"] <= 1.0, fio2["val"] / 100.0)          # percent vs fraction, as in sofa_like_scoring
        fio2 = fio2[(fio2["val"] > 0) & (fio2["val"] <= 1.0)].rename(columns={"val": "fio2"})
        if len(pao2) and len(fio2):
            pair = pao2.merge(fio2, on=["pidx", "off"], how="inner")                       # paO2 and FiO2 must share the exact lab offset
            pair["ratio"] = pair["pao2"] / pair["fio2"]
            ev.pf = _events(pair, "ratio")
    # ---- infusionDrug: vasopressor presence ----
    p = _path("infusionDrug")
    if p is not None:
        df, n_raw = _read_filtered(p, ["drugname"], "infusionoffset", id_set, idx, cap, lower, chunk,
                                   row_filter=lambda c: is_vasopressor(c["drugname"]).to_numpy())
        ev.vaso = _events(df).drop_duplicates()
        diag["rows_in_cohort"]["infusionDrug"], diag["rows_kept"]["infusionDrug"] = n_raw, len(df)
    # ---- treatment: mechanical ventilation records (respiratory bands 3-4) ----
    if od["use_ventilation_support"]:
        p = _path("treatment")
        if p is not None:
            subs = [s.lower() for s in od["ventilation_substrings"]]

            def _vent(c):
                s = c["treatmentstring"].astype("string").str.lower()
                m = pd.Series(False, index=c.index)
                for sub in subs:
                    m |= s.str.contains(sub, regex=False, na=False)
                return m.to_numpy()
            df, n_raw = _read_filtered(p, ["treatmentstring"], "treatmentoffset", id_set, idx, cap, lower, chunk, row_filter=_vent)
            ev.vent = _events(df).drop_duplicates()
            diag["rows_in_cohort"]["treatment"], diag["rows_kept"]["treatment"] = n_raw, len(df)
    diag["ids_with_any_medication_row"] = ids_with_med
    return ev, diag


# =====================================================================================================================================
# 3. window evaluation (vectorised hourly grid)
# =====================================================================================================================================
def infection_events(ev: EvidenceEvents, lo: int, hi: int, cfg: dict) -> pd.DataFrame:
    """(pidx, hour) infection-evidence events with the evidence INSIDE bins lo..hi."""
    a = ev.abx[(ev.abx["hour"] >= lo) & (ev.abx["hour"] <= hi)]
    if cfg["infection"]["evidence"] == "antibiotic":
        return a[["pidx", "hour"]].drop_duplicates().reset_index(drop=True)
    c = ev.culture[(ev.culture["hour"] >= lo) & (ev.culture["hour"] <= hi)]
    if a.empty or c.empty:
        return _empty("pidx", "hour")
    m = a.merge(c, on="pidx", suffixes=("_a", "_c"))
    m = m[(m["hour_a"] - m["hour_c"]).abs() <= cfg["co_occurrence_window_hours"]]
    return pd.DataFrame({"pidx": m["pidx"], "hour": np.maximum(m["hour_a"], m["hour_c"])}).drop_duplicates().reset_index(drop=True)


def _fill(df: pd.DataFrame, p0: int, p1: int, lo: int, hi: int, kind: str) -> np.ndarray:
    m, K = p1 - p0, hi - lo + 1
    arr = np.zeros((m, K), dtype=bool) if kind == "any" else np.full((m, K), np.inf if kind == "min" else -np.inf)
    if len(df):
        sel = df[(df["pidx"] >= p0) & (df["pidx"] < p1) & (df["hour"] >= lo) & (df["hour"] <= hi)]
        if len(sel):
            r, c = sel["pidx"].to_numpy() - p0, sel["hour"].to_numpy() - lo
            if kind == "any":
                arr[r, c] = True
            else:
                (np.minimum if kind == "min" else np.maximum).at(arr, (r, c), sel["value"].to_numpy("float64"))
    if kind != "any":
        arr[~np.isfinite(arr)] = np.nan
    return arr


def _trail(arr: np.ndarray, W: int, kind: str) -> np.ndarray:
    """Trailing-window aggregate over the last W bins (clipped at the first column): 'min'/'max' ignore NaN, 'any' is OR."""
    K, out = arr.shape[1], arr.copy()
    for j in range(1, min(W, K)):
        if kind == "any":
            out[:, j:] |= arr[:, :K - j]
        else:
            out[:, j:] = (np.fmin if kind == "min" else np.fmax)(out[:, j:], arr[:, :K - j])
    return out


def _resp_score(pf: np.ndarray, supported: np.ndarray, use_support: bool) -> np.ndarray:
    s = np.full(pf.shape, np.nan)
    with np.errstate(invalid="ignore"):
        s[pf >= 400] = 0
        s[(pf >= 300) & (pf < 400)] = 1
        s[(pf >= 200) & (pf < 300)] = 2
        s[(pf >= 100) & (pf < 200)] = 3
        s[pf < 100] = 4
        if use_support:
            s[(s >= 3) & ~supported] = 2      # formal SOFA: respiratory 3-4 require ventilatory support
    return np.nan_to_num(s)


def _band(fn, arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(fn(pd.Series(arr.ravel())).to_numpy("float64")).reshape(arr.shape)


def evaluate_window(ev: EvidenceEvents, n_patients: int, lo: int, hi: int, cfg: dict, chunk_patients: int = 20000) -> pd.DataFrame:
    """Evaluates the proxy criteria on grid hours lo..hi using ONLY evidence in bins lo..hi (trailing windows clipped at `lo`).
    Returns a frame indexed by pidx: met (bool), first_hour (float, NaN if never), max_sofa (int), org_any (bool), infection_any (bool)."""
    od, W = cfg["organ_dysfunction"], cfg["co_occurrence_window_hours"]
    thr, use_support = int(od["sofa_threshold"]), bool(od["use_ventilation_support"])
    inf_ev = infection_events(ev, lo, hi, cfg)
    met = np.zeros(n_patients, dtype=bool); first = np.full(n_patients, np.nan); mx = np.zeros(n_patients, dtype="int64")
    org_any = np.zeros(n_patients, dtype=bool); inf_any = np.zeros(n_patients, dtype=bool)
    for p0 in range(0, n_patients, chunk_patients):
        p1 = min(n_patients, p0 + chunk_patients)
        plt_w = _trail(_fill(ev.plt, p0, p1, lo, hi, "min"), W, "min")
        bil_w = _trail(_fill(ev.bili, p0, p1, lo, hi, "max"), W, "max")
        cre_w = _trail(_fill(ev.creat, p0, p1, lo, hi, "max"), W, "max")
        pf_w = _trail(_fill(ev.pf, p0, p1, lo, hi, "min"), W, "min")
        vent_w = _trail(_fill(ev.vent, p0, p1, lo, hi, "any"), W, "any")
        vaso_w = _trail(_fill(ev.vaso, p0, p1, lo, hi, "any"), W, "any")
        total = (_resp_score(pf_w, vent_w, use_support) + _band(_score_coagulation, plt_w) + _band(_score_liver, bil_w)
                 + _band(_score_renal, cre_w) + 2.0 * vaso_w)
        org = total >= thr
        inf = _trail(_fill(inf_ev, p0, p1, lo, hi, "any"), W, "any")
        crit = org & inf
        sl = slice(p0, p1)
        met[sl] = crit.any(axis=1)
        first[sl] = np.where(met[sl], lo + crit.argmax(axis=1), np.nan)
        mx[sl] = total.max(axis=1).astype("int64") if total.shape[1] else 0
        org_any[sl] = org.any(axis=1)
        inf_any[sl] = _fill(inf_ev, p0, p1, lo, hi, "any").any(axis=1)
    return pd.DataFrame({"met": met, "first_hour": first, "max_sofa": mx, "org_any": org_any, "infection_any": inf_any})


# =====================================================================================================================================
# 4. classification (eligibility / censoring / competing risk)
# =====================================================================================================================================
def apply_followup_policy(base: pd.Series, policy: str) -> pd.Series:
    """Maps policy-independent base classes to final statuses (see config.py for the three policies; alive_discharge_as_negative is the primary)."""
    m = {_B_EVENT_FULL: POSITIVE, _B_EVENT_EARLY: POSITIVE, _B_NOEV_FULL: NEG_COMPLETE, _B_NOEV_ALIVE: CENS_ALIVE,
         _B_NOEV_DEATH: CENS_DEATH, _B_NOEV_UNKNOWN: CENS_UNKNOWN}
    if policy == "alive_discharge_as_negative":
        m[_B_NOEV_ALIVE] = NEG_ALIVE_DISCHARGE
    elif policy == "require_full_followup_all":
        m[_B_EVENT_EARLY] = EXCL_POS_INCOMPLETE
    elif policy != "exclude_censored":
        raise ValueError(f"unknown censoring_policy {policy!r}")
    return base.map(m).fillna(base)       # label-invalid / prevalent pass through unchanged


def classify_patients(universe: pd.DataFrame, prev: pd.DataFrame, inc: pd.DataFrame, abx_pre: np.ndarray, hospital_coverage: pd.Series,
                      cfg: dict) -> pd.DataFrame:
    """`universe` rows with pre_status NaN, in the same order as the pidx of `prev`/`inc`.  Returns the per-patient label table (LABEL_COLUMNS)
    for the ELIGIBLE patients; `assemble_label_table` merges the pre-excluded ones back."""
    e = universe.reset_index(drop=True)
    H_min = cfg["horizon_end_hours"] * 60.0
    min_cov = cfg["label_validity"]["min_medication_coverage"]
    cov = e["hospitalid"].map(hospital_coverage)
    # coverage unknown (hospital absent from the coverage table) counts as label-invalid whenever the filter is on
    invalid_hosp = (cov.isna() | (cov < min_cov)) if min_cov > 0 else pd.Series(False, index=e.index)
    prevalent = prev["met"].to_numpy()
    event = inc["met"].to_numpy()
    exit_min = e["icu_exit_minutes"].to_numpy("float64")
    full = exit_min >= H_min
    died = e["died_in_icu"].to_numpy("float64")
    base = np.select(
        [invalid_hosp.to_numpy(), prevalent, event & full, event & ~full, ~event & full, ~event & ~full & (died == 0), ~event & ~full & (died == 1)],
        [EXCL_LABEL_INVALID_HOSPITAL, EXCL_PREVALENT, _B_EVENT_FULL, _B_EVENT_EARLY, _B_NOEV_FULL, _B_NOEV_ALIVE, _B_NOEV_DEATH],
        default=_B_NOEV_UNKNOWN)
    base = pd.Series(base, index=e.index, dtype="object")
    status = apply_followup_policy(base, cfg["followup"]["censoring_policy"])
    label = np.where(status == POSITIVE, 1.0, np.where(status.isin(NEGATIVE_STATUSES), 0.0, np.nan))
    cutoff_min = cfg["cutoff_hours"] * 60.0
    out = pd.DataFrame({
        "patientunitstayid": e["patientunitstayid"], "hospitalid": e["hospitalid"], "split": e["split"],
        "icu_exit_hours": exit_min / 60.0, "died_in_icu": died, "status": status.to_numpy(), "label": label,
        "included": ~np.isnan(label), "event_hour": inc["first_hour"].to_numpy(), "prevalent": prevalent,
        "abx_pre_cutoff": abx_pre, "org_dysfunction_incident_ever": inc["org_any"].to_numpy(),
        "infection_evidence_incident_ever": inc["infection_any"].to_numpy(), "max_sofa_like_incident": inc["max_sofa"].to_numpy(),
        "followup_hours_in_event_window": (np.minimum(exit_min, H_min) - cutoff_min) / 60.0,
    })
    out["_base"] = base.to_numpy()
    return out


def assemble_label_table(universe: pd.DataFrame, eligible_table: pd.DataFrame) -> pd.DataFrame:
    """All manifest patients: eligible rows from `eligible_table`, pre-excluded ones with their pre-status and NaN label."""
    pre = universe[universe["pre_status"].notna()]
    pre_rows = pd.DataFrame({
        "patientunitstayid": pre["patientunitstayid"], "hospitalid": pre["hospitalid"], "split": pre["split"],
        "icu_exit_hours": pre["icu_exit_minutes"] / 60.0, "died_in_icu": pre["died_in_icu"], "status": pre["pre_status"], "label": np.nan,
        "included": False})
    allr = pd.concat([eligible_table.drop(columns=["_base"], errors="ignore"), pre_rows], ignore_index=True, sort=False)
    for c in LABEL_COLUMNS:
        if c not in allr.columns:
            allr[c] = np.nan
    return allr[LABEL_COLUMNS].sort_values("patientunitstayid").reset_index(drop=True)


# =====================================================================================================================================
# 5. cohort report
# =====================================================================================================================================
def _n(mask) -> int:
    return int(np.asarray(mask).sum())


def build_cohort_report(labels: pd.DataFrame, eligible_table: pd.DataFrame, coverage_by_hosp: pd.Series, diag: dict, ev: EvidenceEvents,
                        inc_lo: int, inc_hi: int, cfg: dict) -> dict:
    s = labels["status"]
    n_elig = _n(~s.isin(PRE_STATUSES))
    cnt = s.value_counts().to_dict()
    pos, neg = _n(s == POSITIVE), _n(s.isin(NEGATIVE_STATUSES))
    base = eligible_table["_base"].value_counts().to_dict()
    inv_hosp = sorted(int(h) for h, c in coverage_by_hosp.items() if cfg["label_validity"]["min_medication_coverage"] > 0 and c < cfg["label_validity"]["min_medication_coverage"])
    flow = {
        "manifest_patients": len(labels),
        "pre_excluded": {"not_in_patient_table": cnt.get(PRE_NOT_IN_PATIENT_TABLE, 0), "invalid_icu_exit_time": cnt.get(PRE_INVALID_EXIT, 0),
                         "icu_stay_ended_by_cutoff": cnt.get(PRE_ENDED_BY_CUTOFF, 0)},
        "initial_eligible_cohort": n_elig,
        "other_exclusions": {"label_invalid_hospital_patients": cnt.get(EXCL_LABEL_INVALID_HOSPITAL, 0), "label_invalid_hospitals": len(inv_hosp)},
        "excluded_prevalent": cnt.get(EXCL_PREVALENT, 0),
        "followup": {
            "censored_alive_discharge": cnt.get(CENS_ALIVE, 0), "censored_death_no_event": cnt.get(CENS_DEATH, 0),
            "censored_unknown_exit_status": cnt.get(CENS_UNKNOWN, 0), "excluded_positive_incomplete_followup": cnt.get(EXCL_POS_INCOMPLETE, 0),
            "total_excluded_or_censored": _n(s.isin(CENSORED_STATUSES + (EXCL_POS_INCOMPLETE,))),
            "negatives_with_complete_followup": cnt.get(NEG_COMPLETE, 0), "negatives_alive_discharge_counted": cnt.get(NEG_ALIVE_DISCHARGE, 0),
            "positives_with_incomplete_followup_kept": 0 if cfg["followup"]["censoring_policy"] == "require_full_followup_all" else base.get(_B_EVENT_EARLY, 0),
        },
        "incident_positive": pos, "incident_negative": neg, "final_cohort": pos + neg,
        "incident_prevalence": (pos / (pos + neg)) if (pos + neg) else None,
    }
    recon = (flow["other_exclusions"]["label_invalid_hospital_patients"] + flow["excluded_prevalent"] + flow["followup"]["total_excluded_or_censored"] + pos + neg)
    flow["reconciliation"] = {"eligible": n_elig, "sum_of_parts": recon, "ok": bool(recon == n_elig)}
    # policy sensitivity: the same patients under each of the three policies (no recomputation needed)
    ev_full, ev_early = base.get(_B_EVENT_FULL, 0), base.get(_B_EVENT_EARLY, 0)
    nf, na, nd, nu = base.get(_B_NOEV_FULL, 0), base.get(_B_NOEV_ALIVE, 0), base.get(_B_NOEV_DEATH, 0), base.get(_B_NOEV_UNKNOWN, 0)
    sens = {
        "exclude_censored": {"positive": ev_full + ev_early, "negative": nf},
        "alive_discharge_as_negative": {"positive": ev_full + ev_early, "negative": nf + na},
        "require_full_followup_all": {"positive": ev_full, "negative": nf},
    }
    for v in sens.values():
        v["prevalence"] = v["positive"] / (v["positive"] + v["negative"]) if (v["positive"] + v["negative"]) else None
    inc_labels = labels[labels["included"]]
    by_split = {sp: {"final_cohort": _n(inc_labels["split"] == sp), "positive": _n((inc_labels["split"] == sp) & (inc_labels["label"] == 1)),
                     "negative": _n((inc_labels["split"] == sp) & (inc_labels["label"] == 0)),
                     "eligible": _n((labels["split"] == sp) & ~s.isin(PRE_STATUSES)), "manifest": _n(labels["split"] == sp)}
                for sp in sorted(labels["split"].dropna().unique())}
    n_win = lambda df: int(df[(df["hour"] >= inc_lo) & (df["hour"] <= inc_hi)]["pidx"].nunique()) if len(df) else 0   # noqa: E731
    evidence = {"patients_with_evidence_in_incident_window": {
        "antibiotic_start": n_win(ev.abx), "culture": n_win(ev.culture), "platelets": n_win(ev.plt), "bilirubin": n_win(ev.bili),
        "creatinine": n_win(ev.creat), "pao2_fio2_pair": n_win(ev.pf), "mechanical_ventilation": n_win(ev.vent), "vasopressor_infusion": n_win(ev.vaso)},
        "tables_used": diag["tables_used"], "tables_missing": diag["tables_missing"],
        "rows_in_eligible_cohort": diag["rows_in_cohort"], "rows_kept_in_time_window": diag["rows_kept"],
        "max_attainable_sofa_like": MAX_ATTAINABLE_SOFA_LIKE}
    negs = labels[labels["status"].isin(NEGATIVE_STATUSES)]
    pos_rows = labels[labels["status"] == POSITIVE]
    blocks = {}
    span = (inc_hi - inc_lo + 1)
    if len(pos_rows) and span > 0:
        step = max(1, span // 3)
        edges = list(range(inc_lo, inc_hi + 1, step)) + [inc_hi + 1]
        for a, b in zip(edges[:-1], edges[1:]):
            blocks[f"{a}-{b - 1}h"] = _n((pos_rows["event_hour"] >= a) & (pos_rows["event_hour"] < b))
    diagnostics = {
        "negatives_with_antibiotic_start_before_or_at_cutoff": _n(negs["abx_pre_cutoff"].fillna(False).astype(bool)),
        "negatives_with_organ_dysfunction_in_event_window_but_no_qualifying_infection_evidence":
            _n(negs["org_dysfunction_incident_ever"].fillna(False).astype(bool) & ~negs["infection_evidence_incident_ever"].fillna(False).astype(bool)),
        "negatives_with_infection_evidence_but_no_organ_dysfunction":
            _n(negs["infection_evidence_incident_ever"].fillna(False).astype(bool) & ~negs["org_dysfunction_incident_ever"].fillna(False).astype(bool)),
        "positive_event_hour_blocks": blocks,
        "positive_event_hour_median": float(pos_rows["event_hour"].median()) if len(pos_rows) else None,
    }
    return {"definition_version": INCIDENT_LABEL_DEFINITION_VERSION, "flow": flow, "policy_sensitivity": sens, "by_split": by_split,
            "by_status": {str(k): int(v) for k, v in cnt.items()}, "evidence_coverage": evidence, "diagnostics": diagnostics,
            "label_invalid_hospital_ids": inv_hosp, "parameters": cfg}


def assert_flow_reconciles(report: dict) -> None:
    r = report["flow"]["reconciliation"]
    if not r["ok"]:
        raise AssertionError(f"cohort flow does not reconcile: eligible={r['eligible']} but parts sum to {r['sum_of_parts']}")
    f = report["flow"]
    pre = sum(f["pre_excluded"].values())
    if f["manifest_patients"] != pre + f["initial_eligible_cohort"]:
        raise AssertionError("manifest patients != pre-excluded + eligible")
    by = report["by_status"]
    if sum(by.values()) != f["manifest_patients"]:
        raise AssertionError("per-status counts do not sum to the manifest size")


def render_cohort_report_md(rep: dict, run_id: str) -> str:
    f, p = rep["flow"], rep["parameters"]
    L = [f"# Incident sepsis-proxy cohort report -- {run_id}", "",
         f"Definition `{rep['definition_version']}`. Observation window = first {p['cutoff_hours']} ICU h (inputs); incident-event window = hours {p['cutoff_hours']}-{p['horizon_end_hours']} (exclusive/inclusive). "
         f"Censoring policy: `{p['followup']['censoring_policy']}`.", "",
         "## Cohort flow", "", "| Step | Patients |", "|---|---|",
         f"| Patients in the split manifest | {f['manifest_patients']} |",
         f"| - not in the patient table | {f['pre_excluded']['not_in_patient_table']} |",
         f"| - invalid / missing ICU exit time | {f['pre_excluded']['invalid_icu_exit_time']} |",
         f"| - ICU stay ended at or before hour {p['cutoff_hours']} (not at risk at the landmark) | {f['pre_excluded']['icu_stay_ended_by_cutoff']} |",
         f"| **Initial eligible cohort** | **{f['initial_eligible_cohort']}** |",
         f"| Other exclusions: hospitals with medication coverage < {p['label_validity']['min_medication_coverage']} ({f['other_exclusions']['label_invalid_hospitals']} hospitals) | {f['other_exclusions']['label_invalid_hospital_patients']} |",
         f"| Excluded: PREVALENT proxy (criteria met at or before hour {p['cutoff_hours']}) | {f['excluded_prevalent']} |",
         f"| Censored / excluded for follow-up: alive ICU discharge before hour {p['horizon_end_hours']}, no event | {f['followup']['censored_alive_discharge']} |",
         f"| Censored / excluded for follow-up: ICU death before hour {p['horizon_end_hours']}, no event (competing risk) | {f['followup']['censored_death_no_event']} |",
         f"| Censored / excluded for follow-up: unknown ICU exit status, no event | {f['followup']['censored_unknown_exit_status']} |",
         f"| Excluded positives with incomplete follow-up (policy `require_full_followup_all` only) | {f['followup']['excluded_positive_incomplete_followup']} |",
         f"| **Incident positive** | **{f['incident_positive']}** |",
         f"| **Incident negative** | **{f['incident_negative']}** (complete follow-up {f['followup']['negatives_with_complete_followup']}, alive discharge counted {f['followup']['negatives_alive_discharge_counted']}) |",
         f"| Final cohort | {f['final_cohort']} (prevalence {f['incident_prevalence'] if f['incident_prevalence'] is None else round(f['incident_prevalence'], 4)}) |", "",
         f"Reconciliation: eligible {f['reconciliation']['eligible']} = sum of parts {f['reconciliation']['sum_of_parts']} -> {'OK' if f['reconciliation']['ok'] else 'MISMATCH'}.",
         f"Positives kept despite leaving the ICU before hour {p['horizon_end_hours']}: {f['followup']['positives_with_incomplete_followup_kept']}.", "",
         "## Same patients under each censoring policy (no recomputation)", "", "| Policy | Positive | Negative | Prevalence |", "|---|---|---|---|"]
    for k, v in rep["policy_sensitivity"].items():
        L.append(f"| `{k}` | {v['positive']} | {v['negative']} | {None if v['prevalence'] is None else round(v['prevalence'], 4)} |")
    L += ["", "## By split (final cohort)", "", "| Split | Manifest | Eligible | Final | Positive | Negative |", "|---|---|---|---|---|---|"]
    for k, v in rep["by_split"].items():
        L.append(f"| {k} | {v['manifest']} | {v['eligible']} | {v['final_cohort']} | {v['positive']} | {v['negative']} |")
    ec = rep["evidence_coverage"]
    L += ["", "## Evidence coverage (eligible patients with >= 1 qualifying record inside the incident window)", "", "| Source | Patients |", "|---|---|"]
    for k, v in ec["patients_with_evidence_in_incident_window"].items():
        L.append(f"| {k} | {v} |")
    L += ["", f"Tables used: {', '.join(ec['tables_used']) or 'none'}. Tables missing (treated as unobserved): {', '.join(ec['tables_missing']) or 'none'}.",
          f"Maximum attainable SOFA-like score with these tables: {ec['max_attainable_sofa_like']} (CNS, MAP, urine output unavailable time-resolved).",
          "**Check the lab-name rows above against the real data**: a domain with ~0 patients usually means the configured lab name did not match.", "",
          "## Diagnostics (label-definition limitations, quantified)", ""]
    for k, v in rep["diagnostics"].items():
        L.append(f"- {k.replace('_', ' ')}: {v}")
    return "\n".join(L) + "\n"


def render_label_definition_md(cfg: dict) -> str:
    C, H, W, P = cfg["cutoff_hours"], cfg["horizon_end_hours"], cfg["co_occurrence_window_hours"], cfg["prevalent_lookback_hours"]
    od, inf = cfg["organ_dysfunction"], cfg["infection"]
    return f"""# Incident sepsis-proxy label -- exact definition (`{INCIDENT_LABEL_DEFINITION_VERSION}`)

Times are minutes since ICU unit admission, binned to 1-hour bins: hour = ceil(minutes/60). Offset {C*60} (hour {C}) is the LAST observation-side minute; offset {C*60 + 1} is the first event-side minute.

Parameters in this run: C={C} h, H={H} h, W={W} h, P={P} h, SOFA-like threshold {od['sofa_threshold']}, infection evidence `{inf['evidence']}`, censoring policy `{cfg['followup']['censoring_policy']}`, ventilation support required for respiratory bands 3-4: {od['use_ventilation_support']}.

1. **Eligibility.** A manifest patient is at risk if the ICU stay lasts beyond hour {C} (unitdischargeoffset > {C*60}) and has a valid exit time. Hospitals whose medication-record coverage is below {cfg['label_validity']['min_medication_coverage']} are excluded (the antibiotic evidence cannot fire there).
2. **Evidence cap.** Only records with offset <= min(ICU exit, {H*60}) are used. Evidence after the patient leaves the ICU never counts.
3. **Infection evidence.** Antibiotic START offsets (keyword match on drugname) in {', '.join(inf['medication_tables'])}.{' A microLab culture must lie within +/- ' + str(W) + ' h of the antibiotic start, both inside the window being evaluated; the evidence hour is the later of the two.' if inf['evidence'] == 'antibiotic_plus_culture' else ''}
4. **Organ dysfunction at grid hour g.** SOFA-like total >= {od['sofa_threshold']}. Each domain uses its worst value among evidence in the trailing window (max(window start, g-{W-1}) .. g): respiratory = paO2/FiO2 from lab rows sharing the identical offset (bands <400:1, <300:2, <200:3, <100:4; 3-4 require a mechanical-ventilation treatment record in the same trailing window, else 2), coagulation = platelets, liver = total bilirubin, renal = creatinine (bands as in sofa_like_scoring), cardiovascular = vasopressor infusion present -> 2. Unobserved domains contribute 0. Maximum attainable score: {MAX_ATTAINABLE_SOFA_LIKE}.
5. **Criteria met at grid hour g** = organ dysfunction at g AND an infection-evidence event within the trailing window (max(window start, g-{W-1}) .. g). Both elements therefore occurred within {W} h of each other, whichever came first; the event hour is when the second element arrived.
6. **PREVALENT** = criteria met at some g <= {C} using evidence in bins {-P}..{C} (window start {-P}). Excluded; never labelled incident. Evidence exactly at hour {C} counts as prevalent ("before or at hour {C}").
7. **INCIDENT POSITIVE** = not prevalent and criteria first met at some g in [{C+1}, {H}] using ONLY evidence in bins {C+1}..{H} (window start {C+1}; nothing from the observation window contributes).
8. **Negative** = not prevalent, criteria never met, and follow-up complete (ICU stay lasts to hour {H} or beyond). Patients who leave the ICU before hour {H} without an event are NOT automatically negative: alive discharge -> `censored_alive_discharge`, ICU death -> `censored_death_no_event` (competing risk, excluded), unknown exit status -> `censored_unknown_exit_status`. Policy `alive_discharge_as_negative` counts the alive discharges as negatives; `require_full_followup_all` also excludes positives who left the ICU before hour {H}.
"""


# =====================================================================================================================================
# 6. persistence, identity, orchestration
# =====================================================================================================================================
def split_manifest_hash(manifest) -> str:
    if manifest is None:
        return "none"
    return hashlib.sha256(json.dumps(manifest.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()


def compute_identity(cfg: dict, manifest_hash: str, input_files: dict) -> str:
    """Everything that can change the label: definition version, parameters (not chunksize), the split manifest, and the input files."""
    body = {"definition": INCIDENT_LABEL_DEFINITION_VERSION, "params": {k: v for k, v in cfg.items() if k != "chunksize"},
            "manifest": manifest_hash, "inputs": {k: _fingerprint(v) for k, v in sorted(input_files.items())}}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def current_identity(src: dict, manifest, cfg: dict) -> tuple:
    """(identity hash, split-manifest hash, {table: path}) for the CURRENT inputs/parameters -- computed without reading any table body."""
    mh = split_manifest_hash(manifest)
    tables = {}
    for name in ["patient", "lab", "infusionDrug", "treatment", "microLab"] + list(cfg["infection"]["medication_tables"]):
        p = _resolve_table(src, name)
        if p:
            tables[name] = p
    return compute_identity(cfg, mh, tables), mh, tables


def load_incident_labels(run_dir) -> pd.DataFrame:
    return pd.read_csv(Path(run_dir) / "incident_sepsis_labels.csv")


def build_incident_sepsis_label(src: dict, patient_df: pd.DataFrame, manifest, cfg: dict, out_base, run_id: "str | None" = None,
                                force: bool = False) -> dict:
    """End to end.  Returns {run_dir, run_id, cache_hit, report, manifest}.  Never overwrites an existing run directory: a verified
    identical run is REUSED (cache_hit=True) unless force=True, in which case a NEW run directory is created."""
    t0 = time.time()
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    identity, mh, tables_for_identity = current_identity(src, manifest, cfg)
    if not force:
        hit = find_reusable_run(out_base, identity)
        if hit:
            logger.info("[INCIDENT_SEPSIS] cache HIT: reusing verified run %s (identity %s)", hit["run_id"], identity[:12])
            return {"run_dir": hit["run_dir"], "run_id": hit["run_id"], "cache_hit": True,
                    "report": json.loads((Path(hit["run_dir"]) / "cohort_report.json").read_text(encoding="utf-8")), "manifest": hit}
    run_id, run_dir = new_run_dir(out_base, "inc", identity, run_id)
    logger.info("[INCIDENT_SEPSIS] cache MISS (identity %s) -> computing into %s", identity[:12], run_dir)

    # --- compute ---
    C, H, P = cfg["cutoff_hours"], cfg["horizon_end_hours"], cfg["prevalent_lookback_hours"]
    universe = build_universe(patient_df, manifest, cfg)
    elig = universe[universe["pre_status"].isna()].reset_index(drop=True)
    logger.info("[INCIDENT_SEPSIS] manifest=%d eligible(at risk at hour %d)=%d", len(universe), C, len(elig))
    ev, diag = collect_evidence(src, elig, cfg)
    has_med = elig["patientunitstayid"].isin(diag.pop("ids_with_any_medication_row"))
    coverage = has_med.groupby(elig["hospitalid"]).mean().astype("float64")
    prev = evaluate_window(ev, len(elig), -P, C, cfg)
    inc = evaluate_window(ev, len(elig), C + 1, H, cfg)
    abx_pre = np.zeros(len(elig), dtype=bool)
    a_pre = ev.abx[(ev.abx["hour"] >= -P) & (ev.abx["hour"] <= C)]
    abx_pre[a_pre["pidx"].unique()] = True
    elig_table = classify_patients(elig, prev, inc, abx_pre, coverage, cfg)
    labels = assemble_label_table(universe, elig_table)
    report = build_cohort_report(labels, elig_table, coverage, diag, ev, C + 1, H, cfg)
    assert_flow_reconciles(report)

    # --- hospital summary ---
    L = elig_table.copy()
    hosp = L.groupby("hospitalid").agg(
        n_eligible=("patientunitstayid", "size"), n_prevalent=("status", lambda s: int((s == EXCL_PREVALENT).sum())),
        n_positive=("status", lambda s: int((s == POSITIVE).sum())), n_negative=("status", lambda s: int(s.isin(NEGATIVE_STATUSES).sum())),
        n_censored=("status", lambda s: int(s.isin(CENSORED_STATUSES).sum())),
        label_invalid=("status", lambda s: bool((s == EXCL_LABEL_INVALID_HOSPITAL).all()))).reset_index()
    hosp["medication_coverage"] = hosp["hospitalid"].map(coverage)

    # --- persist atomically; manifest.json LAST (its presence marks the run complete) ---
    run_dir.mkdir(parents=True)
    _atomic_write(run_dir / "incident_sepsis_labels.csv", labels.to_csv(index=False))
    _atomic_write(run_dir / "hospital_summary.csv", hosp.to_csv(index=False))
    _atomic_write(run_dir / "cohort_report.json", json.dumps(report, indent=2, default=str))
    _atomic_write(run_dir / "cohort_report.md", render_cohort_report_md(report, run_id))
    _atomic_write(run_dir / "label_definition.md", render_label_definition_md(cfg))
    _atomic_write(run_dir / "config_snapshot.json", json.dumps({"incident_sepsis": cfg}, indent=2, default=str))
    files = {n: _sha256(run_dir / n) for n in ("incident_sepsis_labels.csv", "hospital_summary.csv", "cohort_report.json", "cohort_report.md",
                                                  "label_definition.md", "config_snapshot.json")}
    manifest_doc = {"run_id": run_id, "definition_version": INCIDENT_LABEL_DEFINITION_VERSION, "identity": identity,
                    "created_utc": datetime.now(timezone.utc).isoformat(), "split_manifest_hash": mh, "files": files,
                    "counts": {k: report["flow"][k] for k in ("manifest_patients", "initial_eligible_cohort", "excluded_prevalent",
                                                               "incident_positive", "incident_negative", "final_cohort")},
                    "cache": {"hit": False, "forced_recompute": bool(force)}, "seconds": round(time.time() - t0, 2),
                    "inputs": {k: {"path": str(v), "size_mtime_ns": _fingerprint(v)} for k, v in tables_for_identity.items()}}
    _atomic_write(run_dir / "manifest.json", json.dumps(manifest_doc, indent=2))
    f = report["flow"]
    logger.info("[INCIDENT_SEPSIS] eligible=%d prevalent=%d censored/excluded-followup=%d positive=%d negative=%d (%.1fs)", f["initial_eligible_cohort"],
                f["excluded_prevalent"], f["followup"]["total_excluded_or_censored"], f["incident_positive"], f["incident_negative"], time.time() - t0)
    return {"run_dir": str(run_dir), "run_id": run_id, "cache_hit": False, "report": report, "manifest": manifest_doc, "labels": labels}
