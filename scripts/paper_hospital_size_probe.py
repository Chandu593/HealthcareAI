"""Read-only probe for the hospital-size cohort filter: how many hospitals / patients survive each hospital-size threshold under the
configured window and landmark rule? Prints to the terminal, writes nothing, trains nothing, never touches the split manifest.

  python scripts/paper_hospital_size_probe.py --config configs/paper_fedgbdt.yaml --target-hospitals 3
  python scripts/paper_hospital_size_probe.py --config configs/paper_fedgbdt.yaml --thresholds 2699,3758 --target-hospitals 10

Use it to see where each threshold lands (for example the thresholds that keep exactly the 3 or the 10 largest hospitals), then set
`paper.cohort_filter.min_hospital_patients` (or pass --set) deliberately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_pipeline import load_phase1_summary, load_phase3_result  # noqa: E402
from src.paper.cohort_filters import hospital_size_profile  # noqa: E402
from src.paper.config import get_paper_cfg  # noqa: E402
from src.paper.overrides import apply_set_overrides  # noqa: E402
from src.paper.window import build_window_table  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--thresholds", default="200,300,400,500,600,800,1000,1500")
    ap.add_argument("--target-hospitals", type=int, default=10)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    cfg = load_config(a.config)
    apply_set_overrides(cfg.raw, a.set)
    pcfg = get_paper_cfg(cfg)
    p1 = load_phase1_summary(cfg)
    p3 = load_phase3_result(cfg, p1, need_shard_manifests=False)
    cp = p3["cohort_patients"]
    w = pcfg["window"]
    win = build_window_table(cp, w["minutes"], w.get("min_icu_los_minutes", 0))
    hosp = cp.set_index("patientunitstayid").loc[win.index, "hospitalid"].to_numpy()
    prof = hospital_size_profile(hosp, [int(x) for x in a.thresholds.split(",")], a.target_hospitals)
    print(f"window={w['minutes']} min, landmark (min ICU stay)={w.get('min_icu_los_minutes', 0)} min")
    print(f"eligible cohort: {prof['patients_total']:,} stays in {prof['hospitals_total']} hospitals (full cohort {len(cp):,})")
    print()
    print(f"{'threshold':>10} {'hospitals':>10} {'patients':>10} {'median size':>12}")
    for r in prof["rows"]:
        med = "-" if r["median_size"] is None else "%.0f" % r["median_size"]
        print(f"{r['threshold']:>10} {r['hospitals']:>10} {r['patients']:>10,} {med:>12}")
    t = prof.get("target")
    if t and t["threshold_for_target"] is not None:
        print(f"\nthreshold that keeps exactly the {t['hospitals']} largest hospitals: {t['threshold_for_target']} ({t['patients_at_target']:,} patients)")
    elif t:
        print(f"\nfewer than {t['hospitals']} hospitals in the eligible cohort")
    print("\nSet it with: --set paper.cohort_filter.min_hospital_patients=<threshold>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
