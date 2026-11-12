"""Label-definition check for the FLICS comparison: FLICS predicts mortality at ICU DISCHARGE; the paper track predicts IN-HOSPITAL mortality.
This scores the saved in-hospital-mortality predictions against the ICU-discharge death label on the same test patients, and reports how many hospital
deaths happen after ICU discharge. Read-only: it trains nothing and changes nothing.

  python scripts/paper_label_variant_check.py --run-dir outputs/paper/fedgbdt_strict_final --experiment mortality__no_apache_scores --arm fed_gbdt --data-root "%EICU_DATA_ROOT%"

ICU deaths are a subset of hospital deaths, so the ICU-discharge prevalence is never higher than the in-hospital prevalence of the same cohort.

Optional (off by default, so the printed output is unchanged without it): --paired-bootstrap N also prints the AUROC difference between the two labels with a paired bootstrap interval
(one resample of the test patients feeds both labels, so the interval is for the difference itself, not a combination of the two marginal intervals).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paper.evaluation import bootstrap_ci, fast_auroc  # noqa: E402


def death_ids(patient_csv) -> tuple:
    """-> (ids that died in the ICU (unitdischargestatus), ids that died in hospital (hospitaldischargestatus)) as sets."""
    df = pd.read_csv(patient_csv, usecols=["patientunitstayid", "unitdischargestatus", "hospitaldischargestatus"], low_memory=False)
    icu = df["unitdischargestatus"].astype(str).str.strip().str.lower().eq("expired")
    hosp = df["hospitaldischargestatus"].astype(str).str.strip().str.lower().eq("expired")
    return set(df.loc[icu, "patientunitstayid"].tolist()), set(df.loc[hosp, "patientunitstayid"].tolist())


def paired_label_difference(hosp, icu, s, n: int, alpha: float = 0.05, seed: int = 0) -> dict:
    """AUROC(score vs ICU-discharge death) - AUROC(score vs in-hospital death) on the SAME resampled test patients (one resample feeds both labels)."""
    hosp = np.asarray(hosp); icu = np.asarray(icu); s = np.asarray(s, dtype="float64"); rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        i = rng.integers(0, len(s), len(s)); a = fast_auroc(icu[i], s[i]); b = fast_auroc(hosp[i], s[i])
        if np.isfinite(a) and np.isfinite(b):
            vals.append(a - b)
    point = fast_auroc(icu, s) - fast_auroc(hosp, s)
    if not vals or not np.isfinite(point):
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_resamples": 0}
    return {"point": float(point), "lo": float(np.quantile(vals, alpha / 2)), "hi": float(np.quantile(vals, 1 - alpha / 2)), "n_resamples": len(vals)}


def check(pred, icu_dead: set, hosp_dead: set, boot_n: int = 300, paired_boot_n: int = 0) -> dict:
    sp = np.asarray(pred["split"]).astype(str); m = sp == "test"
    pid = np.asarray(pred["patient_id"])[m]; s = np.asarray(pred["score"], dtype="float64")[m]
    hosp = np.isin(pid, list(hosp_dead)).astype(int); icu = np.isin(pid, list(icu_dead)).astype(int)
    r = {"n_test": int(m.sum()), "hospital_prevalence": float(hosp.mean()), "icu_prevalence": float(icu.mean()),
         "share_of_hospital_deaths_after_icu_discharge": float(((hosp == 1) & (icu == 0)).sum() / max(hosp.sum(), 1)),
         "auroc_vs_hospital_death": bootstrap_ci(hosp, s, "auroc", boot_n, 0.05, 0), "auroc_vs_icu_death": bootstrap_ci(icu, s, "auroc", boot_n, 0.05, 0)}
    r["auroc_vs_hospital_death_check"] = fast_auroc(hosp, s)
    if paired_boot_n > 0:
        r["auroc_difference_icu_minus_hospital"] = paired_label_difference(hosp, icu, s, paired_boot_n)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); ap.add_argument("--experiment", required=True)
    ap.add_argument("--arm", default="fed_gbdt"); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--data-root", required=True)
    ap.add_argument("--paired-bootstrap", type=int, default=0, help="resamples for the paired interval of the AUROC difference between the two labels (0 = off)")
    a = ap.parse_args()
    f = Path(a.run_dir) / "predictions" / f"{a.experiment}__{a.arm}__seed{a.seed}.npz"
    if not f.exists():
        print(f"missing {f}"); return 1
    icu, hosp = death_ids(Path(a.data_root) / "patient.csv")
    r = check(np.load(f, allow_pickle=False), icu, hosp, paired_boot_n=a.paired_bootstrap)
    h, c = r["auroc_vs_hospital_death"], r["auroc_vs_icu_death"]
    print(f"{a.experiment} / {a.arm} (seed {a.seed}), test n={r['n_test']}: in-hospital mortality {100 * r['hospital_prevalence']:.1f}%, ICU-discharge mortality {100 * r['icu_prevalence']:.1f}%")
    print(f"  share of hospital deaths that occur AFTER ICU discharge: {100 * r['share_of_hospital_deaths_after_icu_discharge']:.1f}%")
    print(f"  AUROC vs in-hospital death (training target): {h['point']:.4f} [{h['lo']:.4f}, {h['hi']:.4f}]")
    print(f"  AUROC vs ICU-discharge death (FLICS's label): {c['point']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]")
    d = r.get("auroc_difference_icu_minus_hospital")
    if d is not None:
        print(f"  AUROC difference, ICU-discharge label minus in-hospital label: {d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] (paired bootstrap over test patients, {d['n_resamples']} resamples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
