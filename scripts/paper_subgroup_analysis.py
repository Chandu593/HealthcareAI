"""Subgroup / robustness analysis of one paper-track arm from its saved predictions.

  python scripts/paper_subgroup_analysis.py --run-dir outputs/paper/<run_id> --experiment mortality__no_apache_scores \
      --arm lightgbm_central --data-root "$EICU_DATA_ROOT"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paper.subgroups import render_markdown, subgroup_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True); ap.add_argument("--experiment", required=True, help="e.g. mortality__full")
    ap.add_argument("--arm", default="lightgbm_central"); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data-root", required=True); ap.add_argument("--window-minutes", type=float, default=1440)
    a = ap.parse_args()
    run = Path(a.run_dir); res = json.loads((run / "results.json").read_text(encoding="utf-8"))
    blk = res["experiments"][a.experiment.replace("__", "/")]
    recs = [r for r in blk["records"] if r["arm"] == a.arm and (a.seed is None or r["seed"] == a.seed)]
    if not recs:
        print(f"no record for arm={a.arm} seed={a.seed} in {a.experiment}"); return 1
    r = recs[0]
    pred = np.load(run / "predictions" / f"{a.experiment}__{a.arm}__seed{r['seed']}.npz", allow_pickle=False)
    root = Path(a.data_root)
    pat = pd.read_csv(root / "patient.csv", usecols=lambda c: c in ("patientunitstayid", "unitdischargeoffset", "age", "unittype", "gender"))
    apr = root / "apachePatientResult.csv"
    apache = pd.read_csv(apr, usecols=lambda c: c in ("patientunitstayid", "predictedhospitalmortality", "apacheversion")) if apr.exists() else None
    rep = subgroup_report(pred, pat, apache, r["test"]["threshold"], a.window_minutes)
    md = render_markdown(f"{a.experiment} / {a.arm} (seed {r['seed']})", rep)
    out = run / f"subgroups__{a.experiment}__{a.arm}.md"; out.write_text(md, encoding="utf-8"); print(md); print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
