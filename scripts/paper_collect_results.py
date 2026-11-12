"""One-file summary of every paper-track run, so you can send a single file for analysis.

  python scripts/paper_collect_results.py            # scans outputs/paper/*/results.json
  python scripts/paper_collect_results.py --paper-dir outputs/paper --out outputs/paper/summary_all_runs.md

Per run / experiment / arm: seeds, test AUROC mean+-sd, AUPRC, Brier, ECE, per-hospital macro AUROC, AUROC with per-hospital calibration
(n/a for runs made before that feature existed), validation AUROC (the number to select on), feature count and split sizes.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def _na(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"


def collect(paper_dir) -> list:
    rows = []
    for rj in sorted(Path(paper_dir).glob("*/results.json")):
        try:
            res = json.loads(rj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for key, blk in res.get("experiments", {}).items():
            by = {}
            for r in blk.get("records", []):
                by.setdefault(r["arm"], []).append(r)
            for arm, rs in by.items():
                g = lambda f: np.array([f(r) for r in rs], dtype=float)
                loc = [r["local_calibrated"]["test"]["auroc"] for r in rs if "local_calibrated" in r]
                ph = [r["test_per_hospital"].get("macro_auroc") for r in rs if r.get("test_per_hospital", {}).get("macro_auroc") is not None]
                rows.append({"run": rj.parent.name, "experiment": key, "arm": arm, "seeds": len(rs), "n_features": blk.get("n_features"),
                             "test_auroc": float(np.nanmean(g(lambda r: r["test"]["auroc"]))), "test_auroc_sd": float(np.nanstd(g(lambda r: r["test"]["auroc"]))),
                             "ci_lo": rs[0].get("test_ci", {}).get("auroc", {}).get("lo"), "ci_hi": rs[0].get("test_ci", {}).get("auroc", {}).get("hi"),
                             "test_auprc": float(np.nanmean(g(lambda r: r["test"]["auprc"]))), "brier": float(np.nanmean(g(lambda r: r["test"]["brier"]))),
                             "ece": float(np.nanmean(g(lambda r: r["test"]["ece"]))), "per_hosp_macro_auroc": float(np.mean(ph)) if ph else None,
                             "auroc_local_cal": float(np.mean(loc)) if loc else None, "val_auroc": float(np.nanmean(g(lambda r: r["val"]["auroc"]))),
                             "n_test": rs[0]["test"]["n"]})
    return rows


def to_markdown(rows) -> str:
    L = ["| run | experiment | arm | seeds | feats | test AUROC (mean±sd) [seed-0 95% CI] | AUPRC | per-hosp macro | AUROC local-cal | val AUROC | Brier | ECE | n_test |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["experiment"], -r["test_auroc"])):
        ci = f" [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]" if r["ci_lo"] is not None else ""
        L.append(f"| {r['run']} | {r['experiment']} | {r['arm']} | {r['seeds']} | {r['n_features']} | {r['test_auroc']:.4f}±{r['test_auroc_sd']:.4f}{ci} | {_na(r['test_auprc'])} | "
                 f"{_na(r['per_hosp_macro_auroc'])} | {_na(r['auroc_local_cal'])} | {_na(r['val_auroc'])} | {_na(r['brier'])} | {_na(r['ece'])} | {r['n_test']} |")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--paper-dir", default="outputs/paper"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); rows = collect(a.paper_dir)
    if not rows:
        print(f"no results.json found under {a.paper_dir}"); return 1
    md = to_markdown(rows); out = Path(a.out or Path(a.paper_dir) / "summary_all_runs.md"); out.write_text(md, encoding="utf-8")
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(md); print(f"\nwritten: {out} and {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
