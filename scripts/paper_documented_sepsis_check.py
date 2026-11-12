"""External check of a sepsis model trained on the PROXY label: does its risk score also rank clinician-DOCUMENTED sepsis diagnoses
(diagnosis table: 'sepsis'/'septic' text or ICD-9 038 / 995.91 / 995.92 / 785.52 at any time in the stay), which were never a training target?
Infection-related diagnosis tokens are excluded from the clean feature sets, so a high AUROC is not a trivial lookup.

  python scripts/paper_documented_sepsis_check.py --run-dir outputs/paper/sepsis24_valid --experiment sepsis__sepsis_clean --arm lightgbm_central --data-root "%EICU_DATA_ROOT%"
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paper.evaluation import auprc, bootstrap_ci, fast_auroc  # noqa: E402

SEPSIS_RX = re.compile(r"sepsis|septic", re.I)
SEPSIS_ICD = ("038", "995.91", "995.92", "785.52", "99591", "99592", "78552")


def documented_ids(dx_path, chunksize=1_000_000) -> set:
    cols = ["patientunitstayid", "diagnosisstring"] + (["icd9code"] if "icd9code" in pd.read_csv(dx_path, nrows=0).columns else [])
    out = set()
    for ch in pd.read_csv(dx_path, usecols=cols, chunksize=chunksize, low_memory=False):
        hit = ch["diagnosisstring"].astype(str).str.contains(SEPSIS_RX, na=False)
        if "icd9code" in ch:
            hit |= ch["icd9code"].astype(str).str.replace(" ", "").str.startswith(SEPSIS_ICD, na=False)
        out.update(ch.loc[hit, "patientunitstayid"].tolist())
    return out


def check(pred, flagged: set, boot_n: int = 300) -> dict:
    sp = np.asarray(pred["split"]).astype(str); m = sp == "test"
    pid = np.asarray(pred["patient_id"])[m]; y = np.asarray(pred["y"])[m]; p = np.asarray(pred["p_calibrated"], dtype="float64")[m]
    doc = np.isin(pid, list(flagged)).astype(int)
    r = {"n_test": int(m.sum()), "proxy_prevalence": float(y.mean()), "documented_prevalence": float(doc.mean()),
         "auroc_vs_proxy": fast_auroc(y, p), "auroc_vs_documented": bootstrap_ci(doc, p, "auroc", boot_n, 0.05, 0), "auprc_vs_documented": auprc(doc, p)}
    neg = y == 0
    if (doc[neg] == 1).sum() >= 20:
        r["auroc_vs_documented_among_proxy_negatives"] = fast_auroc(doc[neg], p[neg])      # documented cases the proxy MISSED
    q = pd.qcut(pd.Series(p), 5, labels=False, duplicates="drop")
    r["documented_rate_by_risk_quintile"] = [float(doc[(q == k).to_numpy()].mean()) for k in sorted(q.unique())]
    return r


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); ap.add_argument("--experiment", required=True)
    ap.add_argument("--arm", default="lightgbm_central"); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--data-root", required=True)
    a = ap.parse_args()
    f = Path(a.run_dir) / "predictions" / f"{a.experiment}__{a.arm}__seed{a.seed}.npz"
    if not f.exists():
        print(f"missing {f}"); return 1
    r = check(np.load(f, allow_pickle=False), documented_ids(Path(a.data_root) / "diagnosis.csv"))
    ci = r["auroc_vs_documented"]
    print(f"{a.experiment} / {a.arm} (seed {a.seed}), test n={r['n_test']}: proxy prevalence {100 * r['proxy_prevalence']:.1f}%, documented-sepsis prevalence {100 * r['documented_prevalence']:.1f}%")
    print(f"  AUROC vs the proxy label (training target): {r['auroc_vs_proxy']:.4f}")
    print(f"  AUROC vs DOCUMENTED sepsis (never a target): {ci['point']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]; AUPRC {r['auprc_vs_documented']:.4f}")
    if "auroc_vs_documented_among_proxy_negatives" in r:
        print(f"  among proxy-negative stays, AUROC for the documented cases the proxy missed: {r['auroc_vs_documented_among_proxy_negatives']:.4f}")
    print("  documented-sepsis rate by predicted-risk quintile (low -> high): " + ", ".join(f"{x:.3f}" for x in r["documented_rate_by_risk_quintile"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
