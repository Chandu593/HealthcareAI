"""Temporal Clinical Dynamics (TCD): ONE unified patient-level feature vector from the first 24 ICU hours.

Seven feature families, ALL in one vector (no per-family encoders):
  A statistical / variability   B trends / rates of change   C abnormality burden & recovery   D derived physiological indicators
  E laboratory dynamics (lab variables go through A-C and F; priority labs are always attempted)   F observation & missingness patterns
  G cross-variable interactions

Data reality of this project (13-table eICU export): the only time-stamped VALUE series are in `lab`. There is no vitalPeriodic / nurseCharting, so
shock index, pulse pressure and MAP cannot be computed here: they are declared, recorded as UNAVAILABLE (with the reason) and never imputed. If a
vitals table is present the same machinery uses it (optional path; hourly-mean binned).

TEMPORAL CUTOFF: only rows with 0 <= offset <= min(ICU exit, window_minutes) are ever read (the reader enforces it), AND compute_tcd_feature_matrix clips
again itself (defence in depth); both are tested, and the stage runs a poison audit on real data (rows after the window are corrupted -> features must be
bit-identical). Pre-ICU (negative-offset) rows are not used. ICU exit only clips the window; it is never a feature.

Fitting discipline: variable selection (coverage on TRAINING patients) and all preprocessing (missingness filter, winsorisation, imputation, scaling,
missing indicators) are fitted on training patients only; validation/test rows are only transformed.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.paper.features import _resolve_table
from src.paper.window import build_window_table
from src.three_modality import artifacts
from src.three_modality.config import TCD_FEATURE_DEFINITION_VERSION

logger = logging.getLogger("eicu_fl")

# variables whose raw values are ALSO ingredients of the incident sepsis label's organ-dysfunction score (label uses their values AFTER hour 24; TCD
# uses them BEFORE it: legitimate forecasting signal, not leakage, but documented as overlap in the schema)
LABEL_OVERLAP_VARS = {"platelets", "bilirubin", "creatinine", "pao2", "fio2", "pf_ratio"}
# Ventilator / oxygen-support SETTINGS that the eICU `lab` table also carries (kept as TCD inputs -- known by hour 24, legitimate predictors). They are markers of
# the mechanical-ventilation evidence the label's respiratory domain uses after hour 24, so they are flagged as overlap too (fio2 is already an ingredient above).
VENT_SUPPORT_VARS = {"peep", "tv", "vent_rate", "pressure_support", "lpm_o2"}


def overlap_class(variable: str) -> tuple:
    """(class, basis) of a variable's overlap with the incident label's ingredients."""
    if variable in LABEL_OVERLAP_VARS:
        return "direct", "ingredient of the label's organ-dysfunction score (used by the label AFTER hour 24, by TCD BEFORE it)"
    if variable in VENT_SUPPORT_VARS:
        return "direct", "ventilator / oxygen-support setting: a marker of the mechanical-ventilation evidence the label uses after hour 24"
    return "none", ""


def overlap_audit(variables) -> dict:
    """Which of a feature run's variables overlap with the incident label's ingredients (computed from CURRENT code, so it stays correct even for feature runs
    sealed before a flag existed)."""
    vs = sorted({v for v in variables if "&" not in v and v != "patient"})
    return {"lab_ingredients_of_the_label": [v for v in vs if v in LABEL_OVERLAP_VARS], "ventilation_support_settings": [v for v in vs if v in VENT_SUPPORT_VARS]}

# (suffix, family, requirement, definition)  -- fixed order = column order
VAR_FEATURES = [
    ("mean", "A", "n>=1", "mean of the measurements in the window"),
    ("median", "A", "n>=1", "median of the measurements"),
    ("min", "A", "n>=1", "minimum measurement"),
    ("max", "A", "n>=1", "maximum measurement"),
    ("std", "A", "n>=2", "sample standard deviation (ddof=1)"),
    ("iqr", "A", "n>=4", "interquartile range p75-p25 (linear interpolation)"),
    ("mad", "A", "n>=3", "median absolute deviation: median(|x - median(x)|)"),
    ("p10", "A", "n>=4", "10th percentile"),
    ("p90", "A", "n>=4", "90th percentile"),
    ("cv", "A", "n>=2 and |mean|>0", "coefficient of variation std/|mean|"),
    ("first", "A", "n>=1", "earliest measurement"),
    ("last", "A", "n>=1", "latest measurement in the window (the value known at the cutoff)"),
    ("early_mean", "A", ">=1 in first half", "mean of measurements in [0, 12h)"),
    ("late_mean", "A", ">=1 in second half", "mean of measurements in [12h, 24h]"),
    ("slope_per_h", "B", "n>=3 and span>=4h", "ordinary-least-squares slope of value on time, units per hour"),
    ("late_minus_early", "B", "both halves observed", "late_mean - early_mean"),
    ("rate_first_last_per_h", "B", "n>=2 and span>=2h", "(last - first) / hours between them"),
    ("median_abs_change", "B", "n>=3", "median |difference| between consecutive measurements"),
    ("max_abs_change_6h", "B", ">=1 consecutive pair 0.5-6h apart", "largest |difference| between consecutive measurements taken 0.5-6 h apart"),
    ("frac_abnormal", "C", "range defined, n>=1", "fraction of measurements outside the reference range"),
    ("n_abnormal", "C", "range defined, n>=1", "count of measurements outside the reference range"),
    ("n_abn_episodes", "C", "range defined, n>=1", "number of maximal runs of consecutive abnormal measurements"),
    ("abn_time_frac", "C", "range defined, >=1 usable segment", "fraction of observed time outside range (value carried forward <= 12h between measurements; longer gaps are unobserved)"),
    ("longest_abn_h", "C", "range defined, >=1 usable segment", "longest uninterrupted abnormal stretch in hours (same carry-forward rule)"),
    ("first_abnormal", "C", "range defined, n>=1", "1 if the earliest measurement is outside range else 0"),
    ("last_abnormal", "C", "range defined, n>=1", "1 if the latest measurement is outside range else 0"),
    ("recovered", "C", "range defined, n>=2, first abnormal", "1 if first abnormal and last normal; 0 if first and last abnormal (persistence); NaN if first normal"),
    ("deteriorated", "C", "range defined, n>=2, first normal", "1 if first normal and last abnormal; 0 if both normal; NaN if first abnormal"),
    ("n_meas", "F", "always defined (0 if none)", "number of measurement times in the window"),
    ("any_meas", "F", "always defined", "1 if at least one measurement else 0"),
    ("mean_interval_h", "F", "n>=2", "mean hours between consecutive measurements"),
    ("max_gap_h", "F", "n>=2", "longest gap between consecutive measurements, hours"),
    ("hours_since_last", "F", "n>=1", "hours from the latest measurement to the window cutoff (24h)"),
    ("first_meas_h", "F", "n>=1", "hours from ICU admission to the first measurement"),
    ("hourly_coverage", "F", "always defined", "fraction of the 24 hourly bins containing >=1 measurement"),
]
RANGED_ONLY = {"frac_abnormal", "n_abnormal", "n_abn_episodes", "abn_time_frac", "longest_abn_h", "first_abnormal", "last_abnormal", "recovered", "deteriorated"}
ZERO_DEFAULT = {"n_meas", "any_meas", "hourly_coverage"}

PT_FEATURES = [   # patient-level, from the lab DRAWS (distinct timestamps carrying >=1 selected lab); (suffix, family, definition)
    ("n_meas_total", "F", "total lab measurements across all selected lab variables"),
    ("n_vars_measured", "F", "number of selected lab variables with >=1 measurement"),
    ("n_draws", "F", "number of distinct lab draw times (timestamps with >=1 selected lab)"),
    ("mean_draw_interval_h", "F", "mean hours between consecutive lab draws (>=2 draws)"),
    ("max_draw_gap_h", "F", "longest gap between consecutive lab draws, hours (>=2 draws)"),
    ("hours_since_last_draw", "F", "hours from the latest lab draw to the cutoff (>=1 draw)"),
    ("first_draw_h", "F", "hours from ICU admission to the first lab draw (>=1 draw)"),
    ("draw_hourly_coverage", "F", "fraction of the 24 hourly bins with >=1 lab draw"),
    ("n_vars_ever_abnormal", "C", "number of ranged lab variables with >=1 abnormal measurement (>=1 ranged measurement)"),
    ("frac_vars_ever_abnormal", "C", "n_vars_ever_abnormal / ranged lab variables measured"),
    ("n_vars_recovered", "C", "ranged lab variables flagged recovered"),
    ("n_vars_deteriorated", "C", "ranged lab variables flagged deteriorated"),
    ("max_abn_per_draw", "G", "max number of abnormal ranged tests within one lab draw (joint abnormality)"),
    ("mean_abn_frac_per_draw", "G", "mean over ranged draws of (abnormal ranged tests / ranged tests measured)"),
    ("n_draws_multi_abn", "G", "number of draws with >= joint_abnormal_min abnormal ranged tests"),
    ("last_draw_n_abn", "G", "abnormal ranged tests in the latest ranged draw"),
]
IX_FEATURES = [("n_aligned", "pairs measured at the identical offset"), ("n_joint", "aligned pairs with both in the stated abnormal direction"),
               ("frac_joint", "n_joint / n_aligned (n_aligned>=1)")]


# =====================================================================================================================================
# helpers
# =====================================================================================================================================
def _layout(ids: np.ndarray):
    n = len(ids)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    return starts, np.diff(np.r_[starts, n]), ids[starts]


def _seg_quantile(sv, starts, counts, q):
    pos = q * (counts - 1)
    lo, hi = np.floor(pos).astype(np.int64), np.ceil(pos).astype(np.int64)
    frac = pos - lo
    return sv[starts + lo] * (1 - frac) + sv[starts + hi] * frac


def sanitize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_") or "unnamed"


def _norm(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


def dedupe_mean(pidx, t, v):
    """Sorted by (pidx, t); several values at the identical (pidx, t) -> their mean (one measurement per time)."""
    if len(pidx) == 0:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
    df = pd.DataFrame({"p": pidx, "t": t, "v": v}).groupby(["p", "t"], sort=True)["v"].mean().reset_index()
    return df["p"].to_numpy(np.int64), df["t"].to_numpy(float), df["v"].to_numpy(float)


def reserved_keys(tcfg: dict) -> set:
    return set(tcfg["variables"]["roles"]) | {d["name"] for d in tcfg["derived"]} | set(tcfg["vitals"]["columns"])


def name_to_key_map(tcfg: dict) -> dict:
    """normalised lab name -> role key for every role name; other names are keyed by their sanitised name at read time."""
    return {n.strip().lower(): role for role, names in tcfg["variables"]["roles"].items() for n in names}


# =====================================================================================================================================
# readers (the window is enforced HERE and again inside compute_tcd_feature_matrix)
# =====================================================================================================================================
def lab_coverage(src, ids: np.ndarray, train_mask: np.ndarray, window_end: np.ndarray, tcfg: dict) -> dict:
    """Pass 1 -- per variable key, number of TRAINING patients with >=1 measurement in their window. Returns {n_train, counts{key: n}, key_of_name{name: key}}."""
    path = _resolve_table(src, "lab")
    if path is None:
        raise FileNotFoundError("lab table not found -- required for TCD features")
    train_ids = ids[train_mask]
    tpos = pd.Series(np.arange(len(train_ids)), index=train_ids)
    cap = pd.Series(window_end[train_mask], index=train_ids)
    role_of = name_to_key_map(tcfg)
    reserved = reserved_keys(tcfg)
    seen: dict = {}
    key_of_name: dict = {}
    for ch in pd.read_csv(path, usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"], chunksize=int(tcfg["variables"]["chunksize"]), low_memory=False):
        ch = ch[ch["patientunitstayid"].isin(tpos.index)]
        if ch.empty:
            continue
        off = pd.to_numeric(ch["labresultoffset"], errors="coerce")
        val = pd.to_numeric(ch["labresult"], errors="coerce")
        ok = off.notna() & val.notna() & np.isfinite(val) & (off >= 0) & (off <= ch["patientunitstayid"].map(cap))
        ch = ch[ok]
        if ch.empty:
            continue
        nm = _norm(ch["labname"])
        local = {str(n): (role_of.get(str(n)) or (sanitize(n) + "_lab" if sanitize(n) in reserved else sanitize(n))) for n in nm.dropna().unique()}
        key_of_name.update(local)
        keys = nm.map(local)
        pos = ch["patientunitstayid"].map(tpos).to_numpy()
        for k, grp in pd.Series(pos).groupby(keys.to_numpy(), dropna=True):
            arr = seen.setdefault(k, np.zeros(len(train_ids), dtype=bool))
            arr[grp.to_numpy()] = True
    return {"n_train": int(len(train_ids)), "counts": {k: int(a.sum()) for k, a in seen.items()}, "key_of_name": key_of_name}


def select_variables(cov: dict, tcfg: dict) -> tuple:
    """Fitted on TRAINING coverage only. Roles with >= priority_min coverage + top-k non-role keys with >= min coverage. Returns (ordered keys, table rows)."""
    v, n_tr = tcfg["variables"], max(1, cov["n_train"])
    roles = list(v["roles"])
    rows, selected = [], []
    for k in roles:
        n = cov["counts"].get(k, 0)
        ok = n / n_tr >= float(v["priority_min_train_coverage"])
        rows.append({"variable": k, "kind": "role", "train_patients": n, "train_coverage": n / n_tr, "selected": bool(ok),
                     "reason": "selected (role)" if ok else f"role coverage {n / n_tr:.4f} < {v['priority_min_train_coverage']}"})
        if ok:
            selected.append(k)
    others = sorted(((k, n) for k, n in cov["counts"].items() if k not in v["roles"]), key=lambda kv: (-kv[1], kv[0]))
    for rank, (k, n) in enumerate(others):
        ok = rank < int(v["top_k_by_coverage"]) and n / n_tr >= float(v["min_train_coverage"])
        rows.append({"variable": k, "kind": "data-driven", "train_patients": n, "train_coverage": n / n_tr, "selected": bool(ok),
                     "reason": "selected (top-k)" if ok else (f"rank {rank + 1} > top_k" if rank >= int(v["top_k_by_coverage"]) else f"coverage {n / n_tr:.4f} < {v['min_train_coverage']}")})
        if ok:
            selected.append(k)
    return selected, rows


def read_lab_series(src, ids: np.ndarray, window_end: np.ndarray, selected: list, cov: dict, tcfg: dict, enforce: bool = True,
                    upper_minutes: "float | None" = None) -> tuple:
    """Pass 2 -- {key: (pidx, t_minutes, value)} (deduped, sorted) for the selected variables, plus read statistics.
    enforce=True: only 0 <= offset <= that patient's window_end. enforce=False (poison audit only): 0 <= offset <= upper_minutes, NOT clipped to the window."""
    path = _resolve_table(src, "lab")
    if path is None:
        raise FileNotFoundError("lab table not found -- required for TCD features")
    sel = set(selected)
    names = {n: k for n, k in cov["key_of_name"].items() if k in sel}
    idx = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids)
    cap = pd.Series(window_end, index=ids)
    parts = {k: [] for k in selected}
    st = {"rows_candidate": 0, "rows_dropped_after_window": 0, "rows_dropped_before_icu": 0, "max_offset_used_minutes": -1.0}
    for ch in pd.read_csv(path, usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"], chunksize=int(tcfg["variables"]["chunksize"]), low_memory=False):
        ch = ch[ch["patientunitstayid"].isin(idx.index)]
        if ch.empty:
            continue
        nm = _norm(ch["labname"])
        m = nm.isin(names).to_numpy(dtype=bool)
        if not m.any():
            continue
        ch, nm = ch[m], nm[m]
        off = pd.to_numeric(ch["labresultoffset"], errors="coerce").to_numpy(float)
        val = pd.to_numeric(ch["labresult"], errors="coerce").to_numpy(float)
        good = np.isfinite(off) & np.isfinite(val)
        cap_row = ch["patientunitstayid"].map(cap).to_numpy(float)
        st["rows_candidate"] += int(good.sum())
        st["rows_dropped_before_icu"] += int((good & (off < 0)).sum())
        st["rows_dropped_after_window"] += int((good & (off >= 0) & (off > cap_row)).sum())
        keep = good & (off >= 0) & ((off <= cap_row) if enforce else (off <= float(upper_minutes)))
        if not keep.any():
            continue
        if enforce:
            st["max_offset_used_minutes"] = max(st["max_offset_used_minutes"], float(off[keep].max()))
        pi = ch["patientunitstayid"].map(idx).to_numpy(np.int64)[keep]
        keys = nm.map(names).to_numpy()[keep]
        for k in pd.unique(keys):
            mk = keys == k
            parts[k].append((pi[mk], off[keep][mk], val[keep][mk]))
    out = {}
    for k, lst in parts.items():
        if lst:
            out[k] = dedupe_mean(np.concatenate([a for a, _, _ in lst]), np.concatenate([b for _, b, _ in lst]), np.concatenate([c for _, _, c in lst]))
        else:
            out[k] = (np.zeros(0, np.int64), np.zeros(0), np.zeros(0))
    return out, st


def read_vitals(src, ids: np.ndarray, window_end: np.ndarray, tcfg: dict) -> tuple:
    """OPTIONAL vitals source (inactive when the table is absent, as in the 13-table export). Hourly MEAN per role, t = bin midpoint (<= 23.5h)."""
    vc = tcfg["vitals"]
    if vc["enabled"] == "off":
        return {}, {"available": False, "reason": "tcd.vitals.enabled = off"}
    path = _resolve_table(src, vc["table"])
    if path is None:
        return {}, {"available": False, "reason": f"table '{vc['table']}' not present in this export"}
    cols = sorted({c for cs in vc["columns"].values() for c in cs})
    idx = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids)
    cap = pd.Series(window_end, index=ids)
    nb = tcfg["window_minutes"] // 60
    acc: dict = {}
    for ch in pd.read_csv(path, usecols=["patientunitstayid", "observationoffset"] + cols, chunksize=int(tcfg["variables"]["chunksize"]), low_memory=False):
        ch = ch[ch["patientunitstayid"].isin(idx.index)]
        off = pd.to_numeric(ch["observationoffset"], errors="coerce")
        ch = ch[off.notna() & (off >= 0) & (off <= ch["patientunitstayid"].map(cap))]
        if ch.empty:
            continue
        off = pd.to_numeric(ch["observationoffset"], errors="coerce").to_numpy(float)
        hbin = np.clip(np.ceil(off / 60.0), 1, nb).astype(np.int64)
        pi = ch["patientunitstayid"].map(idx).to_numpy(np.int64)
        for role, cs in vc["columns"].items():
            vals = pd.to_numeric(ch[cs[0]], errors="coerce")
            for c in cs[1:]:
                vals = vals.fillna(pd.to_numeric(ch[c], errors="coerce"))
            ok = vals.notna().to_numpy()
            if ok.any():
                g = pd.DataFrame({"p": pi[ok], "h": hbin[ok], "v": vals.to_numpy(float)[ok]}).groupby(["p", "h"])["v"].agg(["sum", "count"]).reset_index()
                acc.setdefault(role, []).append(g)
    series = {}
    for role, lst in acc.items():
        g = pd.concat(lst).groupby(["p", "h"], sort=True)[["sum", "count"]].sum().reset_index()
        series[role] = (g["p"].to_numpy(np.int64), (g["h"].to_numpy(float) - 1) * 60 + 30, (g["sum"] / g["count"]).to_numpy(float))
    return series, {"available": True, "roles": sorted(series)}


# =====================================================================================================================================
# derived indicators (family D) -- identical-offset alignment, never imputed
# =====================================================================================================================================
def _aligned(series: dict, inputs: list) -> "pd.DataFrame | None":
    df = None
    for nm in inputs:
        if nm not in series or len(series[nm][0]) == 0:
            return None
        cur = pd.DataFrame({"p": series[nm][0], "t": series[nm][1], nm: series[nm][2]})
        df = cur if df is None else df.merge(cur, on=["p", "t"], how="inner")
    return df


def derive_series(series: dict, kinds: dict, tcfg: dict) -> tuple:
    """Adds derived series (kind 'derived'). Returns (series, kinds, unavailable[{name, reason}])."""
    series, kinds, unavailable = dict(series), dict(kinds), []
    for spec in tcfg["derived"]:
        name, inputs, op = spec["name"], spec["inputs"], spec["op"]
        if spec.get("only_if_missing") and name in series and len(series[name][0]):
            continue                                                        # MAP measured -> not derived
        missing = [i for i in inputs if i not in series or len(series[i][0]) == 0]
        if missing:
            unavailable.append({"name": name, "reason": f"input(s) not available: {', '.join(missing)}"})
            continue
        df = _aligned(series, inputs)
        if df is None or df.empty:
            unavailable.append({"name": name, "reason": "inputs never measured at the identical offset"})
            continue
        if op == "ratio_fio2":
            f = df[inputs[1]].where(df[inputs[1]] <= 1.0, df[inputs[1]] / 100.0)
            ok = (f > 0) & (f <= 1.0)
            val = (df[inputs[0]] / f)[ok]
            df = df[ok]
        elif op == "ratio":
            ok = df[inputs[1]] != 0
            val, df = (df[inputs[0]] / df[inputs[1]])[ok], df[ok]
        elif op == "difference":
            val = df[inputs[0]] - df[inputs[1]]
        elif op == "anion_gap":
            val = df[inputs[0]] - df[inputs[1]] - df[inputs[2]]
        elif op == "map_from_sbp_dbp":
            val = (df[inputs[0]] + 2 * df[inputs[1]]) / 3.0
        else:
            raise ValueError(f"unknown derived op {op!r}")
        val = val.replace([np.inf, -np.inf], np.nan)
        ok = val.notna().to_numpy()
        if not ok.any():
            unavailable.append({"name": name, "reason": "no finite derived values"})
            continue
        series[name] = dedupe_mean(df["p"].to_numpy(np.int64)[ok], df["t"].to_numpy(float)[ok], val.to_numpy(float)[ok])
        kinds[name] = "derived"
    return series, kinds, unavailable


# =====================================================================================================================================
# per-variable features (families A, B, C, F) -- vectorised over all patients at once
# =====================================================================================================================================
def variable_features(pidx, t, v, n_pat: int, tcfg: dict, rng: "list | None") -> dict:
    """Input sorted by (pidx, t), unique per (pidx, t), already inside each patient's window. Returns {suffix: float32[n_pat]}; NaN = not computable."""
    W, half = float(tcfg["window_minutes"]), float(tcfg["half_split_minutes"])
    mc, span, nb = tcfg["min_counts"], tcfg["min_span_hours"], int(tcfg["window_minutes"] // 60)
    F = {s: np.full(n_pat, np.nan, np.float32) for s, *_ in VAR_FEATURES if rng is not None or s not in RANGED_ONLY}
    for s in ZERO_DEFAULT:
        F[s][:] = 0.0
    n = len(pidx)
    if n == 0:
        return F
    starts, counts, gid = _layout(pidx)
    ends = starts + counts - 1

    def put(name, vals, ok=None):
        if name in F:
            if ok is None:
                F[name][gid] = vals
            else:
                F[name][gid[ok]] = np.asarray(vals)[ok]

    s = np.add.reduceat(v, starts)
    mean = s / counts
    sv = v[np.lexsort((v, pidx))]
    median = _seg_quantile(sv, starts, counts, 0.5)
    put("mean", mean); put("median", median); put("min", sv[starts]); put("max", sv[ends])
    put("first", v[starts]); put("last", v[ends])
    dev = v - np.repeat(mean, counts)
    std = np.sqrt(np.add.reduceat(dev * dev, starts) / np.maximum(counts - 1, 1))
    put("std", std, counts >= mc["std"])
    put("cv", std / np.where(np.abs(mean) > 1e-9, np.abs(mean), 1), (counts >= 2) & (np.abs(mean) > 1e-9))
    q4 = counts >= mc["quantiles"]
    put("iqr", _seg_quantile(sv, starts, counts, 0.75) - _seg_quantile(sv, starts, counts, 0.25), q4)
    put("p10", _seg_quantile(sv, starts, counts, 0.10), q4); put("p90", _seg_quantile(sv, starts, counts, 0.90), q4)
    ad = np.abs(v - np.repeat(median, counts))
    put("mad", _seg_quantile(ad[np.lexsort((ad, pidx))], starts, counts, 0.5), counts >= mc["mad"])
    early = t < half
    ec = np.add.reduceat(early.astype(np.int64), starts)
    es = np.add.reduceat(np.where(early, v, 0.0), starts)
    lc, ls = counts - ec, s - es
    put("early_mean", es / np.maximum(ec, 1), ec > 0); put("late_mean", ls / np.maximum(lc, 1), lc > 0)
    # ---- B ----
    th = t / 60.0
    St, Stv, Stt = np.add.reduceat(th, starts), np.add.reduceat(th * v, starts), np.add.reduceat(th * th, starts)
    denom = counts * Stt - St * St
    span_h = (t[ends] - t[starts]) / 60.0
    put("slope_per_h", (counts * Stv - St * s) / np.where(denom > 1e-9, denom, 1), (counts >= mc["slope"]) & (span_h >= span["slope"]) & (denom > 1e-9))
    put("late_minus_early", ls / np.maximum(lc, 1) - es / np.maximum(ec, 1), (ec > 0) & (lc > 0))
    put("rate_first_last_per_h", (v[ends] - v[starts]) / np.where(span_h > 0, span_h, 1), (counts >= 2) & (span_h >= span["rate"]))
    same = pidx[1:] == pidx[:-1]
    pp, dt, dv = pidx[:-1][same], ((t[1:] - t[:-1])[same]) / 60.0, (v[1:] - v[:-1])[same]
    pstarts = pcounts = pgid = None
    if len(pp):
        pstarts, pcounts, pgid = _layout(pp)
        adv = np.abs(dv)
        sa = adv[np.lexsort((adv, pp))]
        F["median_abs_change"][pgid[pcounts >= mc["median_abs_change"] - 1]] = _seg_quantile(sa, pstarts, pcounts, 0.5)[pcounts >= mc["median_abs_change"] - 1]
        F["max_gap_h"][pgid] = np.maximum.reduceat(dt, pstarts)
        lo_h, hi_h = tcfg["short_change_hours"]
        msk = (dt >= lo_h) & (dt <= hi_h)
        if msk.any():
            tmp = np.full(n_pat, -np.inf)
            np.maximum.at(tmp, pp[msk], adv[msk])
            okk = np.isfinite(tmp)
            F["max_abs_change_6h"][okk] = tmp[okk]
    # ---- F ----
    F["n_meas"][gid] = counts; F["any_meas"][gid] = 1.0
    put("mean_interval_h", (t[ends] - t[starts]) / 60.0 / np.maximum(counts - 1, 1), counts >= 2)
    put("hours_since_last", (W - t[ends]) / 60.0); put("first_meas_h", t[starts] / 60.0)
    bins = np.minimum((t // 60).astype(np.int64), nb - 1)
    uk = np.unique(pidx.astype(np.int64) * nb + bins)
    F["hourly_coverage"][:] = (np.bincount(uk // nb, minlength=n_pat) / nb).astype(np.float32)
    # ---- C ----
    if rng is not None:
        lo = -np.inf if rng[0] is None else float(rng[0])
        hi = np.inf if rng[1] is None else float(rng[1])
        abn = (v < lo) | (v > hi)
        n_abn = np.add.reduceat(abn.astype(np.int64), starts)
        put("n_abnormal", n_abn); put("frac_abnormal", n_abn / counts)
        fa, la = abn[starts], abn[ends]
        put("first_abnormal", fa.astype(float)); put("last_abnormal", la.astype(float))
        is_start = np.zeros(n, bool); is_start[starts] = True
        ep = abn & (is_start | ~np.r_[False, abn[:-1]])
        put("n_abn_episodes", np.add.reduceat(ep.astype(np.int64), starts))
        put("recovered", (~la).astype(float), (counts >= 2) & fa)
        put("deteriorated", la.astype(float), (counts >= 2) & ~fa)
        if len(pp):
            seg_abn = abn[:-1][same]
            incl = dt <= float(tcfg["locf_max_gap_hours"])
            wdt = dt * incl
            obs = np.add.reduceat(wdt, pstarts)
            okt = obs > 0
            F["abn_time_frac"][pgid[okt]] = (np.add.reduceat(wdt * seg_abn, pstarts) / np.where(okt, obs, 1))[okt]
            a = incl & seg_abn
            run_start = a & (~np.r_[False, a[:-1]] | np.r_[True, pp[1:] != pp[:-1]])
            longest = np.zeros(n_pat)
            if run_start.any():
                run_id = np.cumsum(run_start)
                dur = np.bincount(run_id[a], weights=dt[a])
                np.maximum.at(longest, pp[run_start], dur[1:])
            F["longest_abn_h"][pgid[okt]] = longest[pgid[okt]]
    return F


def _state_mask(v: np.ndarray, rng: list, state: str) -> np.ndarray:
    lo = -np.inf if rng[0] is None else float(rng[0])
    hi = np.inf if rng[1] is None else float(rng[1])
    if state == "high":
        return v > hi
    if state == "low":
        return v < lo
    return (v < lo) | (v > hi)


# =====================================================================================================================================
# assembly: ONE unified vector (variables + patient-level + interactions)
# =====================================================================================================================================
def clip_series(series: dict, window_end: np.ndarray, window_minutes: float) -> dict:
    """Defence in depth: drop anything before ICU admission or after min(that patient's window end, window_minutes). Idempotent."""
    out = {}
    for k, (p, t, v) in series.items():
        if len(p) == 0:
            out[k] = (p, t, v)
            continue
        keep = (t >= 0) & (t <= np.minimum(window_end[p], float(window_minutes)))
        out[k] = (p[keep], t[keep], v[keep]) if not keep.all() else (p, t, v)
    return out


def _entry(name, variable, kind, family, suffix, definition, requires, overlap, basis=""):
    return {"name": name, "variable": variable, "kind": kind, "family": family, "suffix": suffix, "definition": definition, "requires": requires,
            "overlap_with_incident_label": overlap, "overlap_basis": basis}


def _pt_and_ix_features(series, kinds, n_pat, tcfg, vf) -> tuple:
    """Patient-level (draw patterns, abnormality aggregates, joint abnormality) and interaction features. Returns (cols, entries, unavailable_rules)."""
    W, nb, jmin = float(tcfg["window_minutes"]), int(tcfg["window_minutes"] // 60), int(tcfg["joint_abnormal_min"])
    ranges = tcfg["variables"]["ranges"]
    cols, entries, unavailable = [], [], []
    lab_keys = [k for k, kd in kinds.items() if kd == "lab" and k in series and len(series[k][0])]
    pt = {s: np.full(n_pat, np.nan, np.float32) for s, *_ in PT_FEATURES}
    if lab_keys:
        ap = np.concatenate([series[k][0] for k in lab_keys])
        at = np.concatenate([series[k][1] for k in lab_keys])
        M = int(max(2048, np.ceil(float(at.max())) + 2))          # key = patient * M + minute; M is data-driven so a key can never spill into another patient
        pt["n_meas_total"][:] = np.bincount(ap, minlength=n_pat)
        pt["n_vars_measured"][:] = sum((vf[k]["any_meas"] > 0).astype(np.float32) for k in lab_keys)
        uk = np.unique(ap.astype(np.int64) * M + np.rint(at).astype(np.int64))
        dp, dtm = uk // M, (uk % M).astype(float)
        starts, counts, gid = _layout(dp)
        ends = starts + counts - 1
        pt["n_draws"][:] = 0
        pt["n_draws"][gid] = counts
        pt["first_draw_h"][gid] = dtm[starts] / 60.0
        pt["hours_since_last_draw"][gid] = (W - dtm[ends]) / 60.0
        two = counts >= 2
        pt["mean_draw_interval_h"][gid[two]] = ((dtm[ends] - dtm[starts]) / 60.0 / np.maximum(counts - 1, 1))[two]
        same = dp[1:] == dp[:-1]
        if same.any():
            ppd, gap = dp[:-1][same], np.diff(dtm)[same] / 60.0
            gs, _, gg = _layout(ppd)
            pt["max_draw_gap_h"][gg] = np.maximum.reduceat(gap, gs)
        ub = np.unique(dp * nb + np.minimum((dtm // 60).astype(np.int64), nb - 1))
        pt["draw_hourly_coverage"][:] = (np.bincount(ub // nb, minlength=n_pat) / nb).astype(np.float32)
    ranged = [k for k in lab_keys if k in ranges]
    if ranged:
        meas = sum((vf[k]["any_meas"] > 0).astype(np.float32) for k in ranged)
        has = meas > 0
        for name, fn in (("n_vars_ever_abnormal", lambda k: np.nan_to_num(vf[k]["n_abnormal"]) > 0), ("n_vars_recovered", lambda k: np.nan_to_num(vf[k]["recovered"]) == 1),
                         ("n_vars_deteriorated", lambda k: np.nan_to_num(vf[k]["deteriorated"]) == 1)):
            tot = sum(fn(k).astype(np.float32) for k in ranged)
            pt[name][has] = tot[has]
        pt["frac_vars_ever_abnormal"][has] = (pt["n_vars_ever_abnormal"][has] / meas[has])
        rp = np.concatenate([series[k][0] for k in ranged]); rt = np.concatenate([series[k][1] for k in ranged])
        ra = np.concatenate([_state_mask(series[k][2], ranges[k], "abnormal") for k in ranged]).astype(np.float64)
        M2 = int(max(2048, np.ceil(float(rt.max())) + 2))
        uk, inv = np.unique(rp.astype(np.int64) * M2 + np.rint(rt).astype(np.int64), return_inverse=True)
        nmeas, nabn = np.bincount(inv).astype(float), np.bincount(inv, weights=ra)
        dp = uk // M2
        starts, counts, gid = _layout(dp)
        ends = starts + counts - 1
        pt["max_abn_per_draw"][gid] = np.maximum.reduceat(nabn, starts)
        pt["mean_abn_frac_per_draw"][gid] = np.add.reduceat(nabn / nmeas, starts) / counts
        pt["n_draws_multi_abn"][gid] = np.add.reduceat((nabn >= jmin).astype(np.int64), starts)
        pt["last_draw_n_abn"][gid] = nabn[ends]
    for s, fam, d in PT_FEATURES:
        cols.append(pt[s])
        entries.append(_entry(f"pt__{s}", "patient", "patient-level", fam, s, d, "see definition", "partial", "aggregates over all selected lab variables, including label ingredients and ventilation-support settings"))
    # ---- G: pairwise interactions at the IDENTICAL offset ----
    for rule in tcfg["interactions"]:
        a, b = rule["a"], rule["b"]
        miss = [x for x in (a, b) if x not in series or len(series[x][0]) == 0]
        if miss:
            unavailable.append({"name": f"ix__{rule['name']}", "reason": f"variable(s) not available: {', '.join(miss)}"}); continue
        norng = [x for x in (a, b) if x not in ranges]
        if norng:
            unavailable.append({"name": f"ix__{rule['name']}", "reason": f"no reference range for: {', '.join(norng)}"}); continue
        df = _aligned(series, [a, b])
        if df is None or df.empty:
            unavailable.append({"name": f"ix__{rule['name']}", "reason": "never measured at the identical offset"}); continue
        joint = _state_mask(df[a].to_numpy(float), ranges[a], rule["a_state"]) & _state_mask(df[b].to_numpy(float), ranges[b], rule["b_state"])
        p = df["p"].to_numpy(np.int64)
        n_al = np.bincount(p, minlength=n_pat).astype(np.float32)
        n_jt = np.bincount(p, weights=joint.astype(float), minlength=n_pat).astype(np.float32)
        frac = np.where(n_al > 0, n_jt / np.where(n_al > 0, n_al, 1), np.nan).astype(np.float32)
        ov = "partial" if (a in LABEL_OVERLAP_VARS | VENT_SUPPORT_VARS or b in LABEL_OVERLAP_VARS | VENT_SUPPORT_VARS) else "none"
        for sfx, d in IX_FEATURES:
            arr = {"n_aligned": n_al, "n_joint": n_jt, "frac_joint": frac}[sfx]
            cols.append(arr)
            entries.append(_entry(f"ix__{rule['name']}__{sfx}", f"{a}&{b}", "interaction", "G", sfx,
                                  f"{rule['name']}: {a} {rule['a_state']} AND {b} {rule['b_state']} -- {d}", "n_aligned>=1" if sfx == "frac_joint" else "always defined", ov))
    return cols, entries, unavailable


def compute_tcd_feature_matrix(series: dict, kinds: dict, n_pat: int, window_end: np.ndarray, tcfg: dict) -> tuple:
    """series: {key: (pidx, t_minutes, value)}, kinds: {key: 'lab'|'vital'}. Returns (X float32 [n_pat, F], names, schema entries, unavailable list).
    Clips to each patient's window itself (defence in depth) before computing anything."""
    series = clip_series(series, np.asarray(window_end, dtype=float), tcfg["window_minutes"])
    series, kinds, unavailable = derive_series(series, dict(kinds), tcfg)
    ranges = tcfg["variables"]["ranges"]
    order = [k for k in series if kinds.get(k) in ("lab", "vital")] + [k for k in series if kinds.get(k) == "derived"]
    cols, entries, vf = [], [], {}
    fam_text = {s: (fam, d, req) for s, fam, req, d in VAR_FEATURES}
    for k in order:
        p, t, v = series[k]
        rng = ranges.get(k)
        vf[k] = variable_features(p, t, v, n_pat, tcfg, rng)
        ov, basis = overlap_class(k)
        for s, fam, req, d in VAR_FEATURES:
            if s in vf[k]:
                cols.append(vf[k][s])
                entries.append(_entry(f"{k}__{s}", k, kinds[k], fam, s, f"{d}" + (f" (reference range {rng})" if fam == "C" else ""), req, ov, basis))
    c2, e2, un2 = _pt_and_ix_features(series, kinds, n_pat, tcfg, vf)
    cols += c2; entries += e2; unavailable += un2
    X = np.column_stack(cols).astype(np.float32) if cols else np.zeros((n_pat, 0), np.float32)
    return X, [e["name"] for e in entries], entries, unavailable


# =====================================================================================================================================
# preprocessing -- FITTED ON TRAINING PATIENTS ONLY
# =====================================================================================================================================
def fit_preprocessing(raw: np.ndarray, names: list, train_mask: np.ndarray, pp: dict) -> dict:
    tr = np.flatnonzero(train_mask)
    if len(tr) == 0:
        raise ValueError("no training patients -- cannot fit preprocessing")
    F = raw.shape[1]
    keep, dropped = [], []
    lo_l, hi_l, med_l, mean_l, std_l, miss_bits, miss_frac = [], [], [], [], [], [], []
    for j in range(F):
        col = raw[tr, j].astype(np.float64)
        miss = np.isnan(col)
        mf = float(miss.mean())
        if mf > float(pp["max_missing_fraction"]):
            dropped.append({"feature": names[j], "code": "too_missing", "reason": f"train missing fraction {mf:.4f} > {pp['max_missing_fraction']}"}); continue
        obs = col[~miss]
        if obs.size == 0:
            dropped.append({"feature": names[j], "code": "never_observed", "reason": "no observed training values"}); continue
        lo, hi = np.percentile(obs, pp["winsor_pct"])
        w = np.clip(obs, lo, hi)
        sd = float(w.std())
        if sd < 1e-8:
            dropped.append({"feature": names[j], "code": "constant", "reason": "constant in training after winsorisation"}); continue
        keep.append(j); lo_l.append(lo); hi_l.append(hi); med_l.append(float(np.median(w))); mean_l.append(float(w.mean())); std_l.append(sd)
        miss_frac.append(mf); miss_bits.append(np.packbits(miss).tobytes() if mf >= float(pp["indicator_min_missing"]) else None)
    groups: dict = {}
    for pos, b in enumerate(miss_bits):
        if b is not None:
            groups.setdefault(b, []).append(pos)
    group_list = [g for g in groups.values()]
    return {"kept_idx": np.array(keep, dtype=np.int64), "lo": np.array(lo_l), "hi": np.array(hi_l), "median": np.array(med_l), "mean": np.array(mean_l),
            "std": np.array(std_l), "kept_names": [names[j] for j in keep], "dropped": dropped, "train_missing_fraction": np.array(miss_frac),
            "indicator_groups": group_list, "indicator_names": [f"{names[keep[g[0]]]}__missing" for g in group_list], "n_train": int(len(tr)),
            "params": {k: pp[k] for k in ("max_missing_fraction", "winsor_pct", "indicator_min_missing")}}


def transform_features(raw: np.ndarray, state: dict, chunk: int = 50000) -> np.ndarray:
    kept, groups = state["kept_idx"], state["indicator_groups"]
    nk, ng = len(kept), len(groups)
    out = np.zeros((raw.shape[0], nk + ng), dtype=np.float32)
    reps = np.array([g[0] for g in groups], dtype=np.int64)
    for i0 in range(0, raw.shape[0], chunk):
        X = np.asarray(raw[i0:i0 + chunk][:, kept], dtype=np.float64)
        miss = np.isnan(X)
        if ng:
            out[i0:i0 + chunk, nk:] = miss[:, reps]
        X = np.clip(X, state["lo"], state["hi"])
        X = np.where(miss, state["median"], X)
        out[i0:i0 + chunk, :nk] = (X - state["mean"]) / state["std"]
    return out


def save_state(state: dict, run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    import io
    buf = io.BytesIO()
    np.savez(buf, kept_idx=state["kept_idx"], lo=state["lo"], hi=state["hi"], median=state["median"], mean=state["mean"], std=state["std"],
             train_missing_fraction=state["train_missing_fraction"])
    artifacts.atomic_write(run_dir / "preprocessing_state.npz", buf.getvalue())
    meta = {k: state[k] for k in ("kept_names", "dropped", "indicator_groups", "indicator_names", "n_train", "params")}
    artifacts.atomic_write(run_dir / "preprocessing_meta.json", json.dumps(meta, indent=1))
    return {"preprocessing_state.npz": None, "preprocessing_meta.json": None}


def load_state(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    z = np.load(run_dir / "preprocessing_state.npz")
    meta = json.loads((run_dir / "preprocessing_meta.json").read_text(encoding="utf-8"))
    return {**{k: z[k] for k in z.files}, **meta}


def transformed_names(state: dict) -> list:
    return list(state["kept_names"]) + list(state["indicator_names"])


# =====================================================================================================================================
# reports
# =====================================================================================================================================
def availability_tables(raw: np.ndarray, names: list, entries: list, split: np.ndarray, state: dict) -> pd.DataFrame:
    rows = []
    kept = {n: i for i, n in enumerate(state["kept_names"])}
    dropped = {d["feature"]: d["reason"] for d in state["dropped"]}
    masks = {sp: split == sp for sp in ("train", "val", "test")}
    for j, e in enumerate(entries):
        col = raw[:, j]
        r = {"feature": e["name"], "family": e["family"], "variable": e["variable"], "kind": e["kind"], "overlap_with_incident_label": e["overlap_with_incident_label"],
             "kept": e["name"] in kept, "drop_reason": dropped.get(e["name"], "")}
        for sp, m in masks.items():
            r[f"observed_{sp}"] = float(np.mean(~np.isnan(col[m]))) if m.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def render_feature_definitions_md(entries: list, state: dict, unavailable: list, vit: dict) -> str:
    kept = set(state["kept_names"])
    L = [f"# TCD feature definitions (`{TCD_FEATURE_DEFINITION_VERSION}`)", "",
         "ONE unified vector per patient from the first 24 ICU hours. `kept` = survived the training-only missingness/constant filter (the encoder input). "
         "`overlap` = the variable is also an ingredient of the incident sepsis label's organ-dysfunction score (TCD uses it BEFORE hour 24, the label AFTER: "
         "legitimate forecasting signal, documented not hidden; `partial` = aggregate that includes such variables).", "",
         "| Feature | Family | Variable | Needs | Definition | Overlap | Kept |", "|---|---|---|---|---|---|---|"]
    for e in entries:
        L.append(f"| `{e['name']}` | {e['family']} | {e['variable']} | {e['requires']} | {e['definition']} | {e['overlap_with_incident_label']}{(' (' + e['overlap_basis'].split(':')[0] + ')') if e.get('overlap_basis') and e['overlap_with_incident_label'] == 'direct' else ''} | {'yes' if e['name'] in kept else 'no'} |")
    L += ["", "## Unavailable (declared, never imputed)", ""]
    L += [f"- `{u['name']}`: {u['reason']}" for u in unavailable] or ["- none"]
    L += ["", f"Vitals source: {'available (' + ', '.join(vit.get('roles', [])) + ')' if vit.get('available') else 'NOT available -- ' + vit.get('reason', '')}.", "",
          "## Missing indicators", ""]
    L += [f"- `{n}`" for n in state["indicator_names"]] or ["- none"]
    return "\n".join(L) + "\n"


# =====================================================================================================================================
# real-data audit: nothing after the window can influence the features
# =====================================================================================================================================
def run_cutoff_audit(src, ids, window_end, selected, cov, tcfg, X_main, main_names, n_patients: int, horizon_minutes: float, seed: int) -> dict:
    """Re-reads a random patient subset THREE ways and compares the resulting feature matrices:
      A = rows read WITHOUT the reader's window (offsets up to horizon_minutes), features computed (relies on the in-function clipping alone);
      B = the same rows with every value AFTER the patient's window corrupted (x1000 + 12345);
      C = rows read by the normal window-enforcing reader.
    A == B bit for bit proves late rows cannot influence features; A == C shows reader and clipping agree; A vs the main run (by feature NAME) shows the
    audit is checking the real pipeline. `passed` also requires that corrupted rows actually existed (otherwise the audit had no teeth)."""
    n = min(int(n_patients), len(ids))
    pos = np.sort(np.random.default_rng(seed).choice(len(ids), size=n, replace=False))
    sub_ids, sub_we = ids[pos], window_end[pos]
    raw_series, st_raw = read_lab_series(src, sub_ids, sub_we, selected, cov, tcfg, enforce=False, upper_minutes=horizon_minutes)
    enf_series, st_enf = read_lab_series(src, sub_ids, sub_we, selected, cov, tcfg, enforce=True)
    poisoned, n_poison = {}, 0
    for k, (p, t, v) in raw_series.items():
        late = t > sub_we[p]
        n_poison += int(late.sum())
        poisoned[k] = (p, t, np.where(late, v * 1000.0 + 12345.0, v))
    kd = {k: "lab" for k in raw_series}
    Xa, na, _, _ = compute_tcd_feature_matrix(raw_series, kd, n, sub_we, tcfg)
    Xb, nb_, _, _ = compute_tcd_feature_matrix(poisoned, kd, n, sub_we, tcfg)
    Xc, nc, _, _ = compute_tcd_feature_matrix(enf_series, {k: "lab" for k in enf_series}, n, sub_we, tcfg)
    same_ab = bool(na == nb_ and np.array_equal(Xa, Xb, equal_nan=True))
    same_ac = bool(na == nc and np.array_equal(Xa, Xc, equal_nan=True))
    main_idx = {nm: i for i, nm in enumerate(main_names)}
    common = [(i, main_idx[nm]) for i, nm in enumerate(na) if nm in main_idx]
    if common:
        ia, im = zip(*common)
        sub_main = X_main[pos][:, list(im)]
        match_main = bool(np.allclose(Xa[:, list(ia)], sub_main, rtol=1e-6, atol=1e-6, equal_nan=True))
        diff = np.abs(np.nan_to_num(Xa[:, list(ia)].astype(np.float64), nan=-9e9) - np.nan_to_num(sub_main.astype(np.float64), nan=-9e9))
        max_diff = float(diff.max()) if diff.size else 0.0
    else:
        match_main, max_diff = False, float("nan")
    return {"n_patients": int(n), "n_features_compared_with_main": len(common), "rows_after_window_present_and_corrupted": int(n_poison),
            "rows_before_icu_ignored": int(st_enf["rows_dropped_before_icu"]), "rows_after_window_dropped_by_reader": int(st_enf["rows_dropped_after_window"]),
            "reader_max_offset_used_minutes": float(st_enf["max_offset_used_minutes"]), "window_minutes": int(tcfg["window_minutes"]),
            "features_identical_with_and_without_corruption": same_ab, "reader_and_clipping_agree": same_ac, "features_match_main_run": match_main,
            "max_abs_diff_vs_main": max_diff, "vitals_audited": False,
            "passed": bool(same_ab and same_ac and match_main and n_poison > 0 and st_enf["max_offset_used_minutes"] <= tcfg["window_minutes"])}


# =====================================================================================================================================
# orchestration: identity, cache reuse, persistence
# =====================================================================================================================================
FEATURE_FILES = ["patients.csv", "tcd_features_raw.npy", "tcd_features_transformed.npy", "feature_schema.json", "preprocessing_state.npz", "preprocessing_meta.json",
                 "feature_availability.csv", "availability_report.json", "availability_report.md", "feature_definitions.md", "variable_coverage.json",
                 "cutoff_audit.json", "config_snapshot.json"]


def tcd_features_identity(src: dict, manifest, tcfg: dict) -> tuple:
    """(identity, split-manifest hash, {table: path}). Covers the definition version, every parameter that can change a feature (NOT training/encoder/audit/
    chunksize), the split manifest (variable selection and preprocessing are fitted on its train partition) and the input tables."""
    from src.three_modality.incident_sepsis import split_manifest_hash
    mh = split_manifest_hash(manifest)
    tables = {}
    for name in ("patient", "lab", tcfg["vitals"]["table"]):
        p = _resolve_table(src, name)
        if p:
            tables[name] = p
    params = {k: v for k, v in tcfg.items() if k not in ("training", "encoder", "audit")}
    params["variables"] = {k: v for k, v in params["variables"].items() if k != "chunksize"}
    ident = artifacts.hash_json({"definition": TCD_FEATURE_DEFINITION_VERSION, "params": params, "manifest": mh,
                                 "inputs": {k: artifacts.fingerprint(v) for k, v in sorted(tables.items())}})
    return ident, mh, tables


def _universe(patient_df: pd.DataFrame, manifest, tcfg: dict) -> tuple:
    need = {"patientunitstayid", "hospitalid", "unitdischargeoffset"}
    if need - set(patient_df.columns):
        raise ValueError(f"patient table is missing column(s) {sorted(need - set(patient_df.columns))}")
    split_map = {int(k): v for k, v in manifest.patient_to_split.items()}
    pdf = patient_df[list(need)].copy()
    pdf["patientunitstayid"] = pdf["patientunitstayid"].astype("int64")
    if pdf["patientunitstayid"].duplicated().any():
        raise ValueError("patient table has duplicate patientunitstayid")
    pdf = pdf[pdf["patientunitstayid"].isin(split_map)].sort_values("patientunitstayid").reset_index(drop=True)
    ids = pdf["patientunitstayid"].to_numpy(np.int64)
    split = np.array([split_map[i] for i in ids])
    we = build_window_table(pdf, tcfg["window_minutes"]).loc[ids, "window_end"].to_numpy(float)
    return pdf, ids, split, we, len(split_map) - len(ids)


def build_tcd_features(src: dict, patient_df: pd.DataFrame, manifest, tcfg: dict, out_base, run_id: "str | None" = None, force: bool = False) -> dict:
    t0 = time.time()
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    identity, mh, tables = tcd_features_identity(src, manifest, tcfg)
    if not force:
        hit = artifacts.find_reusable_run(out_base, identity)
        if hit:
            logger.info("[TCD] feature cache HIT: reusing verified run %s (identity %s)", hit["run_id"], identity[:12])
            return {"run_dir": hit["run_dir"], "run_id": hit["run_id"], "cache_hit": True, "manifest": hit}
    run_id, run_dir = artifacts.new_run_dir(out_base, "tcdf", identity, run_id)
    logger.info("[TCD] feature cache MISS (identity %s) -> computing into %s", identity[:12], run_dir)

    pdf, ids, split, we, n_not_in_table = _universe(patient_df, manifest, tcfg)
    train_mask = split == "train"
    if not train_mask.any():
        raise ValueError("the split manifest has no training patients among the patient table rows")
    logger.info("[TCD] universe: %d patients (train %d / val %d / test %d); window %d min", len(ids), train_mask.sum(), (split == "val").sum(), (split == "test").sum(), tcfg["window_minutes"])
    cov = lab_coverage(src, ids, train_mask, we, tcfg)
    selected, cov_rows = select_variables(cov, tcfg)
    logger.info("[TCD] selected %d lab variables from %d training patients (roles %d + top-k)", len(selected), cov["n_train"], sum(1 for k in selected if k in tcfg["variables"]["roles"]))
    series, read_stats = read_lab_series(src, ids, we, selected, cov, tcfg, enforce=True)
    kinds = {k: "lab" for k in series}
    vseries, vit = read_vitals(src, ids, we, tcfg)
    for k, v in vseries.items():
        series[k] = dedupe_mean(*v)
        kinds[k] = "vital"
    raw, names, entries, unavailable = compute_tcd_feature_matrix(series, kinds, len(ids), we, tcfg)
    logger.info("[TCD] raw feature matrix %s (%d unavailable derived/interaction groups)", raw.shape, len(unavailable))
    state = fit_preprocessing(raw, names, train_mask, tcfg["preprocessing"])
    X = transform_features(raw, state)
    tnames = transformed_names(state)
    logger.info("[TCD] preprocessing fitted on %d training patients: kept %d / %d features, %d missing-indicator columns -> transformed %s",
                state["n_train"], len(state["kept_names"]), len(names), len(state["indicator_names"]), X.shape)

    avail = availability_tables(raw, names, entries, split, state)
    audit = {"skipped": True, "reason": "tcd.audit.n_patients = 0"}
    if int(tcfg["audit"]["n_patients"]) > 0:
        audit = run_cutoff_audit(src, ids, we, selected, cov, tcfg, raw, names, tcfg["audit"]["n_patients"], tcfg["audit"]["horizon_minutes"], tcfg["audit"]["seed"])
        logger.info("[TCD] cutoff audit: passed=%s (%d patients, %d late rows corrupted)", audit["passed"], audit["n_patients"], audit["rows_after_window_present_and_corrupted"])
        if not audit["passed"]:
            raise AssertionError(f"TCD cutoff audit FAILED: {audit}")

    by_fam = lambda df: {f: int((df["family"] == f).sum()) for f in sorted(df["family"].unique())}   # noqa: E731
    report = {
        "definition_version": TCD_FEATURE_DEFINITION_VERSION,
        "patients": {"total": int(len(ids)), "train": int(train_mask.sum()), "val": int((split == "val").sum()), "test": int((split == "test").sum()),
                     "in_manifest_but_not_in_patient_table": int(n_not_in_table), "window_clipped_at_icu_exit": int((we < tcfg["window_minutes"]).sum())},
        "features": {"raw": len(names), "kept": len(state["kept_names"]), "missing_indicator_columns": len(state["indicator_names"]), "encoder_input_dim": int(X.shape[1]),
                     "raw_by_family": by_fam(avail), "kept_by_family": by_fam(avail[avail["kept"]]),
                     "dropped_by_reason": pd.Series([d["code"] for d in state["dropped"]], dtype="object").value_counts().to_dict()},
        "variables": cov_rows, "n_variables_selected": len(selected), "unavailable": unavailable, "vitals": vit,
        "read_statistics_lab": read_stats, "cutoff_audit": audit, "overlap_with_incident_label": {o: int((avail["overlap_with_incident_label"] == o).sum()) for o in ("direct", "partial", "none")},
    }
    run_dir.mkdir(parents=True)
    ptab = pd.DataFrame({"patientunitstayid": ids, "hospitalid": pdf["hospitalid"].to_numpy(), "split": split, "window_end_minutes": we})
    artifacts.atomic_write(run_dir / "patients.csv", ptab.to_csv(index=False))
    for fname, arr in (("tcd_features_raw.npy", raw), ("tcd_features_transformed.npy", X)):
        artifacts.atomic_save_npy(run_dir / fname, arr)
    save_state(state, run_dir)
    artifacts.atomic_write(run_dir / "feature_schema.json", json.dumps({"definition_version": TCD_FEATURE_DEFINITION_VERSION, "features": entries, "transformed_names": tnames,
                                                                        "selected_variables": selected, "unavailable": unavailable, "vitals": vit}, indent=1))
    artifacts.atomic_write(run_dir / "feature_availability.csv", avail.to_csv(index=False))
    artifacts.atomic_write(run_dir / "availability_report.json", json.dumps(report, indent=1, default=str))
    artifacts.atomic_write(run_dir / "availability_report.md", render_availability_md(report, run_id, tcfg))
    artifacts.atomic_write(run_dir / "feature_definitions.md", render_feature_definitions_md(entries, state, unavailable, vit))
    artifacts.atomic_write(run_dir / "variable_coverage.json", json.dumps(cov))
    artifacts.atomic_write(run_dir / "cutoff_audit.json", json.dumps(audit, indent=1, default=str))
    artifacts.atomic_write(run_dir / "config_snapshot.json", json.dumps({"tcd": {k: v for k, v in tcfg.items() if k not in ("training", "encoder")}}, indent=1, default=str))
    files = {n: artifacts.sha256_file(run_dir / n) for n in FEATURE_FILES}
    man = {"run_id": run_id, "definition_version": TCD_FEATURE_DEFINITION_VERSION, "identity": identity, "split_manifest_hash": mh, "files": files,
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shapes": {"raw": list(raw.shape), "transformed": list(X.shape)},
           "counts": report["patients"], "cache": {"hit": False, "forced_recompute": bool(force)}, "seconds": round(time.time() - t0, 1),
           "inputs": {k: {"path": str(v), "size_mtime_ns": artifacts.fingerprint(v)} for k, v in tables.items()}, "cutoff_audit_passed": bool(audit.get("passed", False))}
    artifacts.atomic_write(run_dir / "manifest.json", json.dumps(man, indent=1))
    logger.info("[TCD] features COMPLETE in %.1fs -> %s", time.time() - t0, run_dir)
    return {"run_dir": str(run_dir), "run_id": run_id, "cache_hit": False, "manifest": man, "report": report}


def render_availability_md(rep: dict, run_id: str, tcfg: dict) -> str:
    p, f = rep["patients"], rep["features"]
    L = [f"# TCD feature availability report -- {run_id}", "", f"Definition `{rep['definition_version']}`; window = first {tcfg['window_minutes'] // 60} ICU h (offsets 0..min(ICU exit, window)).", "",
         "## Patients", "", "| Split | Patients |", "|---|---|", f"| train | {p['train']} |", f"| val | {p['val']} |", f"| test | {p['test']} |", f"| **total** | **{p['total']}** |", "",
         f"Patients whose window is clipped at ICU exit (<{tcfg['window_minutes'] // 60} h stays): {p['window_clipped_at_icu_exit']}. In the manifest but not in the patient table: {p['in_manifest_but_not_in_patient_table']}.", "",
         "## Features", "", f"- raw features: {f['raw']} (by family: {f['raw_by_family']})",
         f"- kept after the TRAINING-only filter: {f['kept']} (by family: {f['kept_by_family']})", f"- missing-indicator columns: {f['missing_indicator_columns']}",
         f"- encoder input dimension: {f['encoder_input_dim']}", f"- dropped by reason: {f['dropped_by_reason']}", f"- overlap with the incident-label ingredients: {rep['overlap_with_incident_label']}", "",
         f"## Variables ({rep['n_variables_selected']} selected; coverage = share of TRAINING patients with >=1 measurement in the window)", "",
         "| Variable | Kind | Train patients | Train coverage | Selected | Reason |", "|---|---|---|---|---|---|"]
    for r in sorted(rep["variables"], key=lambda r: (not r["selected"], -r["train_coverage"]))[:80]:
        L.append(f"| {r['variable']} | {r['kind']} | {r['train_patients']} | {r['train_coverage']:.4f} | {'yes' if r['selected'] else 'no'} | {r['reason']} |")
    L += ["", "## Unavailable (declared, never imputed)", ""]
    L += [f"- `{u['name']}`: {u['reason']}" for u in rep["unavailable"]] or ["- none"]
    v = rep["vitals"]
    L += ["", f"Vitals source: {'available' if v.get('available') else 'NOT available -- ' + v.get('reason', '')}.", "", "## Temporal cutoff evidence", ""]
    rs = rep["read_statistics_lab"]
    L += [f"- lab rows read for the selected variables: {rs['rows_candidate']}; dropped for being AFTER the patient's window: {rs['rows_dropped_after_window']}; dropped for being BEFORE ICU admission: {rs['rows_dropped_before_icu']}",
          f"- maximum offset used by any feature: {rs['max_offset_used_minutes']} min (window = {tcfg['window_minutes']} min)"]
    a = rep["cutoff_audit"]
    if a.get("skipped"):
        L.append(f"- poison audit skipped: {a['reason']}")
    else:
        L += [f"- poison audit on {a['n_patients']} random patients: {a['rows_after_window_present_and_corrupted']} rows AFTER the window were present and corrupted (x1000 + 12345); "
              f"features identical with vs without corruption = **{a['features_identical_with_and_without_corruption']}**; reader-enforced == clipped = {a['reader_and_clipping_agree']}; "
              f"matches the main run = {a['features_match_main_run']} (max abs diff {a['max_abs_diff_vs_main']}); **passed = {a['passed']}**"]
    return "\n".join(L) + "\n"


def load_tcd_features(run_dir) -> dict:
    d = Path(run_dir)
    schema = json.loads((d / "feature_schema.json").read_text(encoding="utf-8"))
    return {"patients": pd.read_csv(d / "patients.csv"), "raw": np.load(d / "tcd_features_raw.npy", mmap_mode="r"), "X": np.load(d / "tcd_features_transformed.npy", mmap_mode="r"),
            "schema": schema, "state": load_state(d), "manifest": json.loads((d / "manifest.json").read_text(encoding="utf-8"))}
