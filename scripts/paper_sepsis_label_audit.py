"""Label-validity audit of the sepsis PROXY (antibiotic AND SOFA-like >= 2) on the persisted Phase-3 labels.

Checks (a) how often the proxy coincides with a clinician-DOCUMENTED sepsis diagnosis, (b) whether its positive rate tracks each hospital's
medication-record coverage (a data-availability artefact -- the antibiotic half of the label cannot fire without medication rows), and
(c) what restricting to well-covered hospitals would keep. Unknown labels (stays without an apachePredVar row) are EXCLUDED, never counted as negative.

  python scripts/paper_sepsis_label_audit.py --data-root "%EICU_DATA_ROOT%" --output-dir outputs --min-los 1440
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEPSIS_RX = re.compile(r"sepsis|septic", re.I)
SEPSIS_ICD = ("038", "995.91", "995.92", "785.52", "99591", "99592", "78552")
COVERAGE_BINS = [(0.0, 0.1), (0.1, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0001)]


def _ids_with_rows(path, chunksize=2_000_000) -> set:
    ids = set()
    if Path(path).exists():
        for ch in pd.read_csv(path, usecols=["patientunitstayid"], chunksize=chunksize):
            ids.update(ch["patientunitstayid"].unique().tolist())
    return ids


def audit(data_root, output_dir, min_los=0.0, thresholds=(0.5, 0.8, 0.9, 0.95), min_hospital_n=30) -> dict:
    root, out = Path(data_root), Path(output_dir) / "cohort_reports"
    coh = pd.read_csv(out / "cohort_patients.csv", usecols=lambda c: c in ("patientunitstayid", "hospitalid", "unittype", "unitdischargeoffset"))
    lab = pd.read_csv(out / "sepsis_labels.csv")
    keep = [c for c in ("patientunitstayid", "sepsis_broad", "broad_infection", "organ_dysfunction") if c in lab.columns]
    df = coh.merge(lab[keep], on="patientunitstayid", how="left")
    df = df[pd.to_numeric(df["unitdischargeoffset"], errors="coerce") >= min_los].copy()
    n_all = len(df); known = df["sepsis_broad"].notna()
    df["y"] = df["sepsis_broad"].astype("float64")          # NaN = unknown label; never coerced to False
    r = {"n": n_all, "n_unknown_label": int((~known).sum()), "proxy_prevalence_known": float(df.loc[known, "y"].mean())}
    if "broad_infection" in df:
        r["antibiotic_prevalence_known"] = float(pd.to_numeric(df.loc[known, "broad_infection"], errors="coerce").mean())
        r["organ_dysfunction_prevalence_known"] = float(pd.to_numeric(df.loc[known, "organ_dysfunction"], errors="coerce").mean())
    dk = df[known]
    r["by_unit_type"] = {k: (int(v["n"]), float(v["p"])) for k, v in dk.groupby("unittype")["y"].agg(n="size", p="mean").sort_values("n", ascending=False).head(8).iterrows()}
    apv = root / "apachePredVar.csv"
    if apv.exists():
        g = pd.read_csv(apv, usecols=["patientunitstayid", "graftcount"]).drop_duplicates("patientunitstayid")
        r["graftcount_value_counts"] = {str(k): int(v) for k, v in g["graftcount"].value_counts(dropna=False).head(6).items()}   # semantics are NOT assumed
    dx = root / "diagnosis.csv"
    if dx.exists():
        flagged = set(); cols = ["patientunitstayid", "diagnosisstring"] + (["icd9code"] if "icd9code" in pd.read_csv(dx, nrows=0).columns else [])
        for ch in pd.read_csv(dx, usecols=cols, chunksize=1_000_000, low_memory=False):
            hit = ch["diagnosisstring"].astype(str).str.contains(SEPSIS_RX, na=False)
            if "icd9code" in ch:
                hit |= ch["icd9code"].astype(str).str.replace(" ", "").str.startswith(SEPSIS_ICD, na=False)
            flagged.update(ch.loc[hit, "patientunitstayid"].tolist())
        doc = dk["patientunitstayid"].isin(flagged); pos = dk["y"] == 1; both = int((doc & pos).sum())
        r["documented"] = {"n": int(doc.sum()), "prevalence": float(doc.mean()), "proxy_recall_of_documented": both / max(int(doc.sum()), 1),
                           "documented_among_proxy_positives": both / max(int(pos.sum()), 1)}
    med = set()
    for t in ("medication", "medication_imputed", "medication_imputed1"):
        med |= _ids_with_rows(root / f"{t}.csv")
    df["has_med"] = df["patientunitstayid"].isin(med)
    hp = df.groupby("hospitalid").agg(n=("y", "size"), n_known=("y", "count"), prev=("y", "mean"), med_cov=("has_med", "mean"))
    hp = hp[hp["n_known"] >= min_hospital_n]
    r["hospital"] = {"n_hospitals": int(len(hp)), "prevalence_quantiles": [float(x) for x in hp["prev"].quantile([0, .1, .25, .5, .75, .9, 1]).to_numpy()],
                     "hospitals_with_zero_positives": int((hp["prev"] == 0).sum()), "hospitals_with_med_coverage_below_50pct": int((hp["med_cov"] < 0.5).sum()),
                     "spearman_prevalence_vs_med_coverage": float(hp["prev"].corr(hp["med_cov"], method="spearman")) if len(hp) > 2 else float("nan")}
    bins = []
    for lo, hi in COVERAGE_BINS:
        m = (hp["med_cov"] >= lo) & (hp["med_cov"] < hi)
        bins.append({"coverage": f"[{lo:.2f}, {min(hi, 1.0):.2f}{']' if hi > 1 else ')'}", "hospitals": int(m.sum()), "patients": int(hp.loc[m, "n_known"].sum()),
                     "proxy_rate": float(np.average(hp.loc[m, "prev"], weights=hp.loc[m, "n_known"])) if m.any() else float("nan")})
    r["coverage_bins"] = bins
    r["threshold_options"] = [{"min_coverage": t, "hospitals_kept": int((hp["med_cov"] >= t).sum()), "patients_kept": int(hp.loc[hp["med_cov"] >= t, "n_known"].sum()),
                               "proxy_rate_kept": float(np.average(hp.loc[hp["med_cov"] >= t, "prev"], weights=hp.loc[hp["med_cov"] >= t, "n_known"])) if (hp["med_cov"] >= t).any() else float("nan")}
                              for t in thresholds]
    return r


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--data-root", required=True); ap.add_argument("--output-dir", default="outputs"); ap.add_argument("--min-los", type=float, default=0)
    a = ap.parse_args(); r = audit(a.data_root, a.output_dir, a.min_los)
    print(f"cohort n={r['n']} ({r['n_unknown_label']} with unknown label, excluded); proxy prevalence among known {100 * r['proxy_prevalence_known']:.1f}%"
          + (f"; antibiotic {100 * r['antibiotic_prevalence_known']:.1f}%; organ dysfunction {100 * r['organ_dysfunction_prevalence_known']:.1f}%" if "antibiotic_prevalence_known" in r else ""))
    print("proxy prevalence by unit type (n, rate):", {k: (n, round(p, 3)) for k, (n, p) in r["by_unit_type"].items()})
    if "graftcount_value_counts" in r:
        print("apachePredVar.graftcount value counts (semantics not assumed):", r["graftcount_value_counts"])
    if "documented" in r:
        d = r["documented"]; print(f"documented sepsis diagnosis: n={d['n']} ({100 * d['prevalence']:.1f}% of labelled stays); proxy catches {100 * d['proxy_recall_of_documented']:.1f}% of them; only {100 * d['documented_among_proxy_positives']:.1f}% of proxy positives have one")
    h = r["hospital"]; q = h["prevalence_quantiles"]
    print(f"hospitals (>=30 labelled stays): {h['n_hospitals']}; proxy rate min/p10/p25/median/p75/p90/max = " + "/".join(f"{x:.2f}" for x in q))
    print(f"  zero-positive hospitals: {h['hospitals_with_zero_positives']}; hospitals with <50% medication-record coverage: {h['hospitals_with_med_coverage_below_50pct']}; Spearman(prevalence, coverage) = {h['spearman_prevalence_vs_med_coverage']:.2f}")
    print("proxy rate by hospital medication-record coverage:")
    for b in r["coverage_bins"]:
        print(f"  coverage {b['coverage']:>14}: {b['hospitals']:4d} hospitals, {b['patients']:7d} labelled stays, proxy rate {b['proxy_rate']:.3f}")
    print("what a coverage filter would keep:")
    for t in r["threshold_options"]:
        print(f"  coverage >= {t['min_coverage']:.2f}: {t['hospitals_kept']:4d} hospitals, {t['patients_kept']:7d} labelled stays, proxy rate {t['proxy_rate_kept']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
