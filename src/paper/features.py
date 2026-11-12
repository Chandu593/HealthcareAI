"""Leakage-safe, window-bounded feature builder for the paper track.

Design rules (each one is asserted in tests/unit/test_paper_features.py):
  * Every time-stamped event is used only if offset <= window_end (and >=
    -lookback). window_end = min(window_minutes, unitdischargeoffset).
  * Vocabularies (top labs / diagnoses / drugs / categories) are chosen from
    TRAIN patients only.
  * Outcome-bearing or post-window columns (discharge*, died*, actual*,
    predicted*) are never features. APACHE IVa predicted mortality is kept
    only as a benchmark vector, never as an input.
  * No sequence length / LOS is exposed. Missingness is explicit (NaN +
    count features), not derived from discharge time.
Everything streams the big CSVs in chunks and accumulates into (N x K)
arrays, so peak memory is independent of table size.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cohort.antibiotic_reference import is_antibiotic, is_vasopressor
from src.data.cohort.eicu_missing import clean_columns
from src.paper.window import build_window_table

logger = logging.getLogger("eicu_fl")

FORBIDDEN_FEATURE_SUBSTRINGS = ("discharge", "died", "actual", "predicted", "mortality", "unabridged")
SOFA_LAB_PATTERN = re.compile(r"platelet|bili|creatinine", re.I)
INFECTION_DX_PATTERN = re.compile(r"sepsis|septic|infect", re.I)

# apachePredVar: whitelist -> group. Anything not listed (notably diedinhospital,
# dischargelocation, saps3today/yesterday) is never read.
APV_GROUPS = {
    "apache_physiology": ["verbal", "motor", "eyes", "day1verbal", "day1motor", "day1eyes", "pao2", "fio2",
                          "day1pao2", "day1fio2", "creatinine", "ventday1", "oobventday1", "oobintubday1", "meds", "day1meds"],
    "apache_comorbidity": ["aids", "hepaticfailure", "lymphoma", "metastaticcancer", "leukemia", "immunosuppression",
                           "cirrhosis", "diabetes", "thrombolytics", "electivesurgery", "readmit", "ima", "midur",
                           "ejectfx", "graftcount", "activetx", "sicuday", "var03hspxlos"],
    "apache_score": ["saps3day1"],
}
APV_BINARY_LIKE = set(APV_GROUPS["apache_comorbidity"]) | {"ventday1", "oobventday1", "oobintubday1", "thrombolytics"}


@dataclass
class FeatureBundle:
    patient_ids: np.ndarray
    hospital_ids: np.ndarray
    X: np.ndarray
    feature_names: list
    feature_groups: list
    y_mortality: np.ndarray
    y_sepsis: np.ndarray
    apache_pred_mortality: np.ndarray
    seq_values: "np.ndarray | None" = None
    seq_labs: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def columns_excluding(self, exclude_groups, exclude_name_regex=None) -> np.ndarray:
        """Column indices kept after dropping whole groups and/or any feature whose NAME matches a regex
        (case-insensitive) -- e.g. end-of-life treatment tokens for the robustness feature set."""
        ex = set(exclude_groups or [])
        pats = [re.compile(r, re.I) for r in (exclude_name_regex or [])]
        return np.array([i for i, (n, g) in enumerate(zip(self.feature_names, self.feature_groups))
                         if g not in ex and not any(pt.search(n) for pt in pats)], dtype=np.int64)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(patient_ids=self.patient_ids, hospital_ids=self.hospital_ids, X=self.X,
                      y_mortality=self.y_mortality, y_sepsis=self.y_sepsis, apache_pred_mortality=self.apache_pred_mortality)
        if self.seq_values is not None:
            arrays["seq_values"] = self.seq_values
        np.savez_compressed(path, **arrays, _json=np.array(json.dumps(dict(
            feature_names=self.feature_names, feature_groups=self.feature_groups, seq_labs=self.seq_labs, meta=self.meta))))

    @classmethod
    def load(cls, path) -> "FeatureBundle":
        z = np.load(path, allow_pickle=False)
        j = json.loads(str(z["_json"]))
        return cls(patient_ids=z["patient_ids"], hospital_ids=z["hospital_ids"], X=z["X"],
                   feature_names=j["feature_names"], feature_groups=j["feature_groups"], y_mortality=z["y_mortality"],
                   y_sepsis=z["y_sepsis"], apache_pred_mortality=z["apache_pred_mortality"],
                   seq_values=z["seq_values"] if "seq_values" in z.files else None, seq_labs=j["seq_labs"], meta=j["meta"])


@dataclass
class WindowContext:
    patient_ids: np.ndarray
    id_set: set
    idx: pd.Series
    wend: pd.Series
    lookback: float
    train_mask: np.ndarray
    chunksize: int


def _safe_name(s) -> str:
    return (re.sub(r"[^0-9A-Za-z]+", "_", str(s)).strip("_").lower()[:48] or "x")


def _dedupe_names(names: list) -> list:
    seen, out = Counter(), []
    for n in names:
        seen[n] += 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def _resolve_table(phase1_summary: dict, name: str) -> "str | None":
    p = (phase1_summary.get("table_paths") or {}).get(name)
    if p and Path(p).exists():
        return str(p)
    root = phase1_summary.get("data_root")
    if root:
        for cand in Path(root).glob("*.csv"):
            if cand.stem.lower() == name.lower():
                return str(cand)
    return None


def _header(path: str) -> list:
    return list(pd.read_csv(path, nrows=0).columns)


def iter_windowed(path, usecols, offset_col, ctx: WindowContext, filter_fn=None, lookback=None):
    """Yield chunks restricted to cohort patients and to -lookback <= offset <= window_end,
    with `_off` (float) and `_pidx` (row position in ctx.patient_ids) attached."""
    lb = ctx.lookback if lookback is None else lookback
    cols = list(dict.fromkeys(["patientunitstayid", offset_col] + list(usecols)))
    for chunk in pd.read_csv(path, usecols=cols, chunksize=ctx.chunksize, low_memory=False):
        chunk = chunk[chunk["patientunitstayid"].isin(ctx.id_set)]
        if chunk.empty:
            continue
        if filter_fn is not None:
            chunk = filter_fn(chunk)
            if chunk.empty:
                continue
        off = pd.to_numeric(chunk[offset_col], errors="coerce")
        wend = chunk["patientunitstayid"].map(ctx.wend)
        keep = off.notna() & (off <= wend) & (off >= -lb)
        if not keep.any():
            continue
        chunk = chunk[keep]
        yield chunk.assign(_off=off[keep].to_numpy(dtype="float64"),
                           _pidx=chunk["patientunitstayid"].map(ctx.idx).to_numpy(dtype="int64"))


def _select_vocab(counter: Counter, top_k: int, min_patients: int) -> list:
    items = [(t, c) for t, c in counter.items() if c >= min_patients and isinstance(t, str) and t != ""]
    return [t for t, _ in sorted(items, key=lambda kv: (-kv[1], kv[0]))[:top_k]]


def _norm_drug(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.extract(r"^\s*([a-z][a-z\-]*(?: [a-z\-]+)?)", expand=False).str.strip()


def _hier_token(s: pd.Series, depth: int) -> pd.Series:
    return s.astype(str).str.lower().str.split("|").str[:depth].str.join("|").str.strip()


def _vocab_from_table(path, usecols, offset_col, token_fn, ctx, top_k, min_patients, filter_fn=None, lookback=None) -> list:
    counts: Counter = Counter()
    for ch in iter_windowed(path, usecols, offset_col, ctx, filter_fn, lookback):
        tok = token_fn(ch)
        df = pd.DataFrame({"p": ch["_pidx"].to_numpy(), "t": tok.to_numpy()}).dropna().drop_duplicates()
        df = df[ctx.train_mask[df["p"].to_numpy()]]
        counts.update(df["t"].value_counts().to_dict())
    return _select_vocab(counts, top_k, min_patients)


def _accumulate_tokens(path, usecols, offset_col, token_fn, ctx, vocab, binary, filter_fn=None, lookback=None) -> np.ndarray:
    K = len(vocab)
    M = np.zeros((len(ctx.patient_ids), max(K, 1)), dtype=np.float32)
    if K == 0:
        return M[:, :0]
    col = {t: i for i, t in enumerate(vocab)}
    for ch in iter_windowed(path, usecols, offset_col, ctx, filter_fn, lookback):
        t = token_fn(ch).map(col)
        ok = t.notna().to_numpy()
        if not ok.any():
            continue
        df = pd.DataFrame({"p": ch["_pidx"].to_numpy()[ok], "t": t.to_numpy()[ok].astype("int64")})
        g = df.groupby(["p", "t"]).size()
        p, tt = g.index.get_level_values(0).to_numpy(), g.index.get_level_values(1).to_numpy()
        if binary:
            M[p, tt] = 1.0
        else:
            M[p, tt] += g.to_numpy(dtype="float32")
    return M


class LabAccumulator:
    def __init__(self, n, L, n_bins=0, n_seq_labs=0, bin_minutes=60):
        z = lambda dt=np.float64, v=0.0: np.full((n, L), v, dtype=dt)
        self.cnt, self.s, self.ss = z(np.float64), z(np.float64), z(np.float64)
        self.mn, self.mx = z(np.float32, np.inf), z(np.float32, -np.inf)
        self.fo, self.lo = z(np.float64, np.inf), z(np.float64, -np.inf)
        self.fv, self.lv = z(np.float32, np.nan), z(np.float32, np.nan)
        self.n_bins, self.n_seq, self.bin_minutes = n_bins, n_seq_labs, bin_minutes
        if n_bins and n_seq_labs:
            self.seq_sum = np.zeros((n, n_bins, n_seq_labs), dtype=np.float32)
            self.seq_cnt = np.zeros((n, n_bins, n_seq_labs), dtype=np.float32)
        else:
            self.seq_sum = self.seq_cnt = None

    def add(self, pidx, lidx, off, val):
        df = pd.DataFrame({"p": pidx, "l": lidx, "o": off, "v": val})
        df = df[np.isfinite(df["v"].to_numpy())]
        if df.empty:
            return
        df = df.assign(v2=df["v"] * df["v"]).sort_values("o", kind="stable")
        agg = df.groupby(["p", "l"], sort=False).agg(
            c=("v", "size"), s=("v", "sum"), ss=("v2", "sum"), mn=("v", "min"), mx=("v", "max"),
            fo=("o", "first"), fv=("v", "first"), lo=("o", "last"), lv=("v", "last"))
        p, l = agg.index.get_level_values(0).to_numpy(), agg.index.get_level_values(1).to_numpy()
        self.cnt[p, l] += agg["c"].to_numpy(); self.s[p, l] += agg["s"].to_numpy(); self.ss[p, l] += agg["ss"].to_numpy()
        self.mn[p, l] = np.minimum(self.mn[p, l], agg["mn"].to_numpy(dtype="float32"))
        self.mx[p, l] = np.maximum(self.mx[p, l], agg["mx"].to_numpy(dtype="float32"))
        u = agg["fo"].to_numpy() < self.fo[p, l]
        self.fo[p[u], l[u]] = agg["fo"].to_numpy()[u]; self.fv[p[u], l[u]] = agg["fv"].to_numpy(dtype="float32")[u]
        u = agg["lo"].to_numpy() >= self.lo[p, l]
        self.lo[p[u], l[u]] = agg["lo"].to_numpy()[u]; self.lv[p[u], l[u]] = agg["lv"].to_numpy(dtype="float32")[u]
        if self.seq_sum is not None:
            sq = df[(df["l"] < self.n_seq) & (df["o"] >= 0)]
            if not sq.empty:
                b = np.minimum((sq["o"].to_numpy() // self.bin_minutes).astype("int64"), self.n_bins - 1)
                g = pd.DataFrame({"p": sq["p"].to_numpy(), "b": b, "l": sq["l"].to_numpy(), "v": sq["v"].to_numpy()}).groupby(["p", "b", "l"])["v"].agg(["sum", "size"])
                ip, ib, il = (g.index.get_level_values(i).to_numpy() for i in range(3))
                self.seq_sum[ip, ib, il] += g["sum"].to_numpy(dtype="float32")
                self.seq_cnt[ip, ib, il] += g["size"].to_numpy(dtype="float32")

    def finalize(self, stats: list) -> dict:
        c = self.cnt
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(c > 0, self.s / c, np.nan)
            var = np.where(c > 1, np.maximum(self.ss / c - mean ** 2, 0.0) * c / (c - 1), np.nan)
        out = {"mean": mean, "std": np.sqrt(var), "count": c, "min": np.where(c > 0, self.mn, np.nan),
               "max": np.where(c > 0, self.mx, np.nan), "first": self.fv, "last": self.lv,
               "delta": np.where(c > 1, self.lv - self.fv, np.nan)}
        return {k: out[k].astype(np.float32) for k in stats}

    def seq(self):
        if self.seq_sum is None:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.seq_cnt > 0, self.seq_sum / self.seq_cnt, np.nan).astype(np.float32)


def _onehot(all_vals: pd.Series, train_vals: pd.Series, top_k: int, prefix: str, group: str):
    vc = train_vals.dropna().astype(str).value_counts()
    cats = [c for c in sorted(vc.index, key=lambda c: (-vc[c], c))[:top_k]]
    cols, names = [], []
    s = all_vals.astype(str).where(all_vals.notna())
    for c in cats:
        cols.append((s == c).to_numpy(dtype="float32")); names.append(f"{prefix}__{_safe_name(c)}")
    return cols, _dedupe_names(names), [group] * len(cols)


def _labels_by_patient(cohort_patients: pd.DataFrame, mortality, sepsis_df, sepsis_cohort: str):
    ids = cohort_patients["patientunitstayid"].to_numpy(dtype="int64")
    if mortality.index.equals(cohort_patients.index):
        mort = pd.Series(mortality.to_numpy(), index=ids)
    else:
        mort = pd.Series(mortality.to_numpy(), index=mortality.index.to_numpy(dtype="int64"))
    ym = pd.to_numeric(mort.reindex(ids), errors="coerce").to_numpy(dtype="float32")
    col = "sepsis_strict" if sepsis_cohort == "strict" else "sepsis_broad"
    ys = np.full(len(ids), np.nan, dtype="float32")
    if sepsis_df is not None and col in sepsis_df.columns:
        sv = sepsis_df.set_index("patientunitstayid")[col]
        sv = sv.astype("object").map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else float(bool(v)))
        ys = pd.to_numeric(sv.reindex(ids), errors="coerce").to_numpy(dtype="float32")
    return ym, ys


def build_feature_bundle(pcfg: dict, phase1_summary: dict, cohort_patients: pd.DataFrame, mortality, sepsis_df,
                         train_patient_ids, sepsis_cohort: str = "broad") -> FeatureBundle:
    wcfg, fcfg, scfg = pcfg["window"], pcfg["features"], pcfg["sequence"]
    win = build_window_table(cohort_patients, wcfg["minutes"], wcfg.get("min_icu_los_minutes", 0))
    cp = cohort_patients.copy()
    cp["patientunitstayid"] = cp["patientunitstayid"].astype("int64")
    cp = cp[cp["patientunitstayid"].isin(win.index)].sort_values(["hospitalid", "patientunitstayid"], kind="stable")
    ym_all, ys_all = _labels_by_patient(cohort_patients.assign(patientunitstayid=cohort_patients["patientunitstayid"].astype("int64")), mortality, sepsis_df, sepsis_cohort)
    lab_series_m = pd.Series(ym_all, index=cohort_patients["patientunitstayid"].astype("int64").to_numpy())
    lab_series_s = pd.Series(ys_all, index=cohort_patients["patientunitstayid"].astype("int64").to_numpy())
    pids = cp["patientunitstayid"].to_numpy(dtype="int64")
    N = len(pids)
    train_set = set(int(i) for i in train_patient_ids)
    train_mask = np.array([int(i) in train_set for i in pids], dtype=bool)
    ctx = WindowContext(patient_ids=pids, id_set=set(pids.tolist()), idx=pd.Series(np.arange(N), index=pids),
                        wend=win["window_end"].reindex(pids), lookback=float(wcfg["lookback_minutes"]),
                        train_mask=train_mask, chunksize=int(fcfg["chunksize"]))
    cp = cp.reset_index(drop=True)
    min_pat = int(fcfg["min_vocab_patients"])
    cols: list = []; names: list = []; groups: list = []

    def add(arrs, nm, grp):
        cols.extend(arrs); names.extend(nm); groups.extend(grp)

    # ---- demographics / admission context (known at ICU admission) --------
    age = pd.to_numeric(cp["age"].astype(str).str.replace(">", "", regex=False).str.strip(), errors="coerce")
    h = pd.to_numeric(cp["admissionheight"], errors="coerce"); h = h.where((h >= 50) & (h <= 250))
    w = pd.to_numeric(cp["admissionweight"], errors="coerce"); w = w.where((w >= 20) & (w <= 400))
    num = {"age": age, "height_cm": h, "weight_kg": w, "bmi": w / ((h / 100.0) ** 2),
           "hosp_to_icu_minutes": -pd.to_numeric(cp["hospitaladmitoffset"], errors="coerce"),
           "unit_visit_number": pd.to_numeric(cp["unitvisitnumber"], errors="coerce") if "unitvisitnumber" in cp else np.nan}
    add([pd.Series(v).to_numpy(dtype="float32") for v in num.values()], list(num), ["demographics"] * len(num))
    tm = train_mask
    for col, k in [("gender", 4), ("ethnicity", fcfg["top_categories"]), ("unittype", fcfg["top_categories"]),
                   ("unitadmitsource", fcfg["top_categories"]), ("hospitaladmitsource", fcfg["top_categories"]),
                   ("unitstaytype", 6), ("apacheadmissiondx", fcfg["top_admission_dx"])]:
        if col in cp:
            c, n, g = _onehot(cp[col], cp.loc[tm, col], k, f"cat_{col}", "demographics"); add(c, n, g)
    hpath = _resolve_table(phase1_summary, "hospital")
    if hpath:
        hdf = pd.read_csv(hpath).drop_duplicates("hospitalid").set_index("hospitalid")
        for col in ("region", "teachingstatus", "numbedscategory"):
            if col in hdf:
                c, n, g = _onehot(cp["hospitalid"].map(hdf[col]), cp.loc[tm, "hospitalid"].map(hdf[col]), 8, f"hosp_{col}", "hospital"); add(c, n, g)
    hn = cp.groupby("hospitalid")["patientunitstayid"].transform("size").to_numpy(dtype="float32")
    add([np.log1p(hn)], ["hosp_log_n_patients"], ["hospital"])

    # ---- apachePredVar (day-1 inputs; whitelist only) and APACHE scores ----
    apv_path = _resolve_table(phase1_summary, "apachePredVar")
    if apv_path:
        avail = set(_header(apv_path))
        want = [c for g in APV_GROUPS.values() for c in g if c in avail]
        df = pd.read_csv(apv_path, usecols=["patientunitstayid"] + want)
        df = df[df["patientunitstayid"].isin(ctx.id_set)].drop_duplicates("patientunitstayid").set_index("patientunitstayid").reindex(pids)
        df = clean_columns(df.reset_index(), [c for c in want if c not in ("ventday1",)]).set_index("patientunitstayid")
        for grp, members in APV_GROUPS.items():
            if grp == "apache_score" and not fcfg["include_apache_scores"]:
                continue
            present = [c for c in members if c in df.columns]
            add([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float32") for c in present], [f"apv_{c}" for c in present], [grp] * len(present))
        if all(c in df.columns for c in ("verbal", "motor", "eyes")):
            gcs = sum(pd.to_numeric(df[c], errors="coerce") for c in ("verbal", "motor", "eyes"))
            add([gcs.to_numpy(dtype="float32")], ["apv_gcs_total"], ["apache_physiology"])
    apache_pred = np.full(N, np.nan, dtype="float32")
    apr_path = _resolve_table(phase1_summary, "apachePatientResult")
    if apr_path:
        avail = set(_header(apr_path))
        use = ["patientunitstayid"] + [c for c in ("acutephysiologyscore", "apachescore", "predictedhospitalmortality", "apacheversion") if c in avail]
        df = pd.read_csv(apr_path, usecols=use)
        df = df[df["patientunitstayid"].isin(ctx.id_set)]
        if "apacheversion" in df:
            df = df.assign(_v=(df["apacheversion"].astype(str).str.upper() != "IVA")).sort_values("_v", kind="stable")
        df = df.drop_duplicates("patientunitstayid").set_index("patientunitstayid").reindex(pids)
        if fcfg["include_apache_scores"]:
            sc = [c for c in ("acutephysiologyscore", "apachescore") if c in df.columns]
            vals = clean_columns(df[sc].reset_index(), sc).set_index("patientunitstayid") if sc else df[sc]
            add([pd.to_numeric(vals[c], errors="coerce").to_numpy(dtype="float32") for c in sc], [f"apr_{c}" for c in sc], ["apache_score"] * len(sc))
        if "predictedhospitalmortality" in df.columns:
            pm = pd.to_numeric(df["predictedhospitalmortality"], errors="coerce").where(lambda s: (s >= 0) & (s <= 1))
            apache_pred = pm.to_numpy(dtype="float32")   # BENCHMARK ONLY -- never a feature

    # ---- labs: train-only vocabulary, then window stats (+ optional hourly sequence) ----
    lab_path = _resolve_table(phase1_summary, "lab")
    seq_values, seq_labs = None, []
    if lab_path:
        logger.info("[PAPER_FEATURES] lab vocabulary pass")
        vocab = _vocab_from_table(lab_path, ["labname"], "labresultoffset", lambda c: c["labname"].astype(str), ctx, fcfg["top_labs"], min_pat)
        L = len(vocab)
        if L:
            n_seq = min(int(scfg["n_labs"]), L) if scfg["enabled"] else 0
            n_bins = int(np.ceil(wcfg["minutes"] / scfg["bin_minutes"])) if n_seq else 0
            acc = LabAccumulator(N, L, n_bins, n_seq, scfg["bin_minutes"])
            lcol = {t: i for i, t in enumerate(vocab)}
            logger.info("[PAPER_FEATURES] lab statistics pass (%d labs)", L)
            for ch in iter_windowed(lab_path, ["labname", "labresult"], "labresultoffset", ctx):
                li = ch["labname"].astype(str).map(lcol)
                ok = li.notna().to_numpy()
                if ok.any():
                    acc.add(ch["_pidx"].to_numpy()[ok], li.to_numpy()[ok].astype("int64"), ch["_off"].to_numpy()[ok],
                            pd.to_numeric(ch["labresult"], errors="coerce").to_numpy(dtype="float64")[ok])
            st = acc.finalize(list(fcfg["lab_stats"]))
            seq_tmp = acc.seq()
            del acc
            lab_names = _dedupe_names([_safe_name(v) for v in vocab])
            for j, nm in enumerate(lab_names):
                grp = "lab_sofa" if SOFA_LAB_PATTERN.search(vocab[j]) else "lab"
                for s in fcfg["lab_stats"]:
                    add([st[s][:, j]], [f"lab_{nm}_{s}"], [grp])
            if n_seq:
                seq_values, seq_labs = seq_tmp, lab_names[:n_seq]
            del st

    # ---- token tables: treatment / diagnosis / admission drugs / medication / infusion / microLab ----
    def token_block(table, usecols, off_col, token_fn, top_k, prefix, group_fn, binary, filter_fn=None):
        path = _resolve_table(phase1_summary, table)
        if not path or top_k <= 0:
            return
        avail = set(_header(path))
        if not set(usecols) | {off_col} <= avail:
            logger.warning("[PAPER_FEATURES] %s missing columns %s -- skipped", table, sorted((set(usecols) | {off_col}) - avail))
            return
        voc = _vocab_from_table(path, usecols, off_col, token_fn, ctx, top_k, min_pat, filter_fn)
        M = _accumulate_tokens(path, usecols, off_col, token_fn, ctx, voc, binary, filter_fn)
        nm = _dedupe_names([f"{prefix}_{_safe_name(t)}" for t in voc])
        add([M[:, j] for j in range(M.shape[1])], nm, [group_fn(t) for t in voc])
        if not binary and M.shape[1]:
            add([M.sum(axis=1)], [f"{prefix}_total_events"], [group_fn("")])

    dd, td = int(fcfg["diagnosis_depth"]), int(fcfg["treatment_depth"])
    token_block("treatment", ["treatmentstring"], "treatmentoffset", lambda c: _hier_token(c["treatmentstring"], td),
                fcfg["top_treatments"], "tx", lambda t: "treatment", False)
    token_block("diagnosis", ["diagnosisstring"], "diagnosisoffset", lambda c: _hier_token(c["diagnosisstring"], dd),
                fcfg["top_diagnoses"], "dx", lambda t: "diagnosis_infection" if INFECTION_DX_PATTERN.search(t) else "diagnosis", False)
    token_block("admissionDrug", ["drugname"], "drugoffset", lambda c: _norm_drug(c["drugname"]),
                fcfg["top_admission_drugs"], "admdrug", lambda t: "admission_drug", True)
    notcancel = lambda c: c[c["drugordercancelled"].astype(str).str.lower() != "yes"] if "drugordercancelled" in c else c
    med_paths = [p for p in (_resolve_table(phase1_summary, t) for t in ("medication", "medication_imputed", "medication_imputed1")) if p]
    if med_paths and fcfg["top_medications"] > 0:
        cnt: Counter = Counter()
        tf = lambda c: _norm_drug(c["drugname"])
        for p in med_paths:
            for ch in iter_windowed(p, ["drugname", "drugordercancelled"], "drugstartoffset", ctx, notcancel):
                d = pd.DataFrame({"p": ch["_pidx"].to_numpy(), "t": tf(ch).to_numpy()}).dropna().drop_duplicates()
                d = d[ctx.train_mask[d["p"].to_numpy()]]
                cnt.update(d["t"].value_counts().to_dict())
        voc = _select_vocab(cnt, fcfg["top_medications"], min_pat)
        M = np.zeros((N, len(voc)), dtype=np.float32); abx = np.zeros(N, dtype=np.float32); vaso = np.zeros(N, dtype=np.float32)
        col = {t: i for i, t in enumerate(voc)}
        for p in med_paths:
            for ch in iter_windowed(p, ["drugname", "drugordercancelled"], "drugstartoffset", ctx, notcancel):
                nd = ch["drugname"].astype(str)
                uniq = nd.unique()
                abx_map = dict(zip(uniq, is_antibiotic(pd.Series(uniq)).to_numpy()))
                vas_map = dict(zip(uniq, is_vasopressor(pd.Series(uniq)).to_numpy()))
                pi = ch["_pidx"].to_numpy()
                abx[pi[nd.map(abx_map).to_numpy(dtype=bool)]] = 1.0
                vaso[pi[nd.map(vas_map).to_numpy(dtype=bool)]] = 1.0
                ti = _norm_drug(nd).map(col)
                ok = ti.notna().to_numpy()
                if ok.any():
                    M[pi[ok], ti.to_numpy()[ok].astype("int64")] = 1.0
        add([M[:, j] for j in range(M.shape[1])], _dedupe_names([f"med_{_safe_name(t)}" for t in voc]), ["medication"] * len(voc))
        add([M.sum(axis=1) if len(voc) else np.zeros(N, dtype=np.float32), abx, vaso],
            ["med_n_top_drugs", "med_antibiotic_any", "med_vasopressor_any"], ["medication", "medication_antibiotic", "medication_vasopressor"])
    inf_path = _resolve_table(phase1_summary, "infusionDrug")
    if inf_path and fcfg["top_infusions"] > 0:
        voc = _vocab_from_table(inf_path, ["drugname"], "infusionoffset", lambda c: _norm_drug(c["drugname"]), ctx, fcfg["top_infusions"], min_pat)
        M = _accumulate_tokens(inf_path, ["drugname"], "infusionoffset", lambda c: _norm_drug(c["drugname"]), ctx, voc, True)
        add([M[:, j] for j in range(M.shape[1])], _dedupe_names([f"inf_{_safe_name(t)}" for t in voc]), ["infusion"] * len(voc))
        vaso_i = np.zeros(N, dtype=np.float32)
        for ch in iter_windowed(inf_path, ["drugname"], "infusionoffset", ctx):
            nd = ch["drugname"].astype(str); u = nd.unique()
            m = dict(zip(u, is_vasopressor(pd.Series(u)).to_numpy()))
            vaso_i[ch["_pidx"].to_numpy()[nd.map(m).to_numpy(dtype=bool)]] = 1.0
        add([vaso_i], ["inf_vasopressor_any"], ["infusion_vasopressor"])
    ml_path = _resolve_table(phase1_summary, "microLab")
    if ml_path:
        nc = np.zeros(N, dtype=np.float32)
        for ch in iter_windowed(ml_path, ["culturesite"], "culturetakenoffset", ctx):
            g = pd.Series(ch["_pidx"].to_numpy()).value_counts()
            nc[g.index.to_numpy()] += g.to_numpy(dtype="float32")
        add([nc, (nc > 0).astype(np.float32)], ["micro_n_cultures", "micro_any_culture"], ["microlab"] * 2)

    X = np.column_stack(cols).astype(np.float32) if cols else np.zeros((N, 0), dtype=np.float32)
    names = _dedupe_names(names)
    for n, g in zip(names, groups):
        if g in ("demographics", "hospital", "apache_physiology", "apache_comorbidity", "apache_score", "lab", "lab_sofa") and \
                any(s in n.lower() for s in FORBIDDEN_FEATURE_SUBSTRINGS):
            raise ValueError(f"Forbidden (outcome/post-window) feature name leaked into the matrix: {n!r}")
    key = hashlib.sha1(json.dumps([int(pids[0]) if N else 0, N, wcfg, {k: v for k, v in fcfg.items() if k != 'cache'}, scfg,
                                   int(train_mask.sum())], sort_keys=True, default=str).encode()).hexdigest()[:16]
    return FeatureBundle(patient_ids=pids, hospital_ids=cp["hospitalid"].to_numpy(dtype="int64"), X=X, feature_names=names,
                         feature_groups=groups, y_mortality=lab_series_m.reindex(pids).to_numpy(dtype="float32"),
                         y_sepsis=lab_series_s.reindex(pids).to_numpy(dtype="float32"), apache_pred_mortality=apache_pred,
                         seq_values=seq_values, seq_labs=seq_labs, meta={"cache_key": key, "n_train": int(train_mask.sum()),
                                                                       "window": wcfg, "sepsis_cohort": sepsis_cohort})
