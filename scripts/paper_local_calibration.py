"""Re-score SAVED paper-track predictions with per-hospital intercept calibration (no retraining).

Question it answers: is a model's pooled-AUROC deficit mostly about between-hospital score offsets (fixable by each hospital
fitting ONE scalar on its own validation rows) or about within-hospital ranking (needs a better model)?

  python scripts/paper_local_calibration.py --run-dir outputs/paper/mort_fed_s42 --experiment mortality__no_apache_scores
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paper.evaluation import (apply_local_intercepts, auprc, brier, ece, fast_auroc, fit_local_intercepts,  # noqa: E402
                                  per_hospital_summary, platt_fit, sigmoid)


def rescore(npz_path, prior_sd=1.0) -> dict:
    z = np.load(npz_path, allow_pickle=False); sp = z["split"].astype(str); y = z["y"]; s = z["score"].astype("float64"); h = z["hospital_id"]
    va, te = sp == "val", sp == "test"
    a, b = platt_fit(s[va], y[va]); pg = sigmoid(a * s + b)
    pl = apply_local_intercepts(s, h, a, b, fit_local_intercepts(s[va], y[va], h[va], a, b, prior_sd))
    row = {}
    for name, p in (("global", pg), ("local", pl)):
        row[name] = {"auroc": fast_auroc(y[te], p[te]), "auprc": auprc(y[te], p[te]), "brier": brier(y[te], p[te]), "ece": ece(y[te], p[te]),
                     "per_hosp_macro_auroc": per_hospital_summary(y[te], p[te], h[te]).get("macro_auroc", float("nan"))}
    return row


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); ap.add_argument("--experiment", default=None)
    ap.add_argument("--prior-sd", type=float, default=1.0)
    a = ap.parse_args(); files = sorted((Path(a.run_dir) / "predictions").glob("*.npz"))
    if a.experiment:
        files = [f for f in files if f.name.startswith(a.experiment + "__")]
    if not files:
        print("no prediction files found"); return 1
    lines = ["| experiment / arm / seed | pooled AUROC (global cal.) | pooled AUROC (per-hospital cal.) | per-hospital macro AUROC | AUPRC global -> local | Brier global -> local |", "|---|---|---|---|---|---|"]
    for f in files:
        r = rescore(f, a.prior_sd); g, l = r["global"], r["local"]
        lines.append(f"| {f.stem} | {g['auroc']:.4f} | {l['auroc']:.4f} | {l['per_hosp_macro_auroc']:.4f} | {g['auprc']:.4f} -> {l['auprc']:.4f} | {g['brier']:.4f} -> {l['brier']:.4f} |")
    md = "\n".join(lines); print(md)
    out = Path(a.run_dir) / "local_calibration.md"; out.write_text(md, encoding="utf-8"); print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
