"""Paper-track entry point (additive; does not touch run_pipeline.py / run_ablation.py behaviour).

  python run_paper.py --config configs/paper.yaml --reuse-phase1to3
  python run_paper.py --config configs/paper.yaml --reuse-phase1to3 --tasks mortality --feature-sets no_apache_scores --arms lightgbm_central,logreg_central --seeds 42

Reuses the permanent split manifest (ablation.split.manifest_path) so every number is on the same patients as the earlier ablation.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from run_pipeline import (load_partition_result, load_phase1_summary, load_phase3_result, run_phase1_ingestion,
                          run_phase2_hospital_partition, run_phase3_cohort_construction)
from src.paper.overrides import apply_set_overrides
from src.paper.runner import KNOWN_ARMS, run_paper_track
from src.utils.config import load_config
from src.utils.logging_setup import close_logging, setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--reuse-phase1to3", action="store_true", help="load saved Phase 1-3 outputs instead of recomputing")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--tasks", default=None, help="comma list, e.g. mortality,sepsis")
    ap.add_argument("--feature-sets", default=None, help="comma list, e.g. no_apache_scores,sepsis_controlled")
    ap.add_argument("--arms", default=None, help="run EXACTLY these arms (overrides the config's arm switches), e.g. lightgbm_central,fedavg_mlp,mlp_local_only")
    ap.add_argument("--seeds", default=None, help="comma list overriding paper.seeds, e.g. 42,43,44")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config value, repeatable, e.g. --set paper.window.min_icu_los_minutes=1440")
    args = ap.parse_args()
    cfg = load_config(args.config)
    try:
        apply_set_overrides(cfg.raw, args.set)
    except ValueError as exc:
        ap.error(str(exc))
    paper_over = cfg.raw.setdefault("paper", {})
    if args.seeds:
        paper_over["seeds"] = [int(x) for x in args.seeds.split(",")]
    if args.arms:
        chosen = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = sorted(set(chosen) - set(KNOWN_ARMS))
        if unknown:
            ap.error(f"unknown arm(s) {unknown}; choose from {list(KNOWN_ARMS)}")
        paper_over["arms"] = {a: (a in chosen) for a in KNOWN_ARMS}
    logger = setup_logging(cfg.get("run.log_dir", "outputs/logs"))
    run_id = args.run_id or "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        if args.reuse_phase1to3:
            p1 = load_phase1_summary(cfg); part = load_partition_result(cfg); p3 = load_phase3_result(cfg, p1)
        else:
            p1 = run_phase1_ingestion(cfg); part, pdf, _ = run_phase2_hospital_partition(cfg, p1)
            p3 = run_phase3_cohort_construction(cfg, p1, part, pdf)
        run_paper_track(cfg, p1, part, p3, run_id, tasks=args.tasks.split(",") if args.tasks else None,
                        fs_filter=set(args.feature_sets.split(",")) if args.feature_sets else None)
    finally:
        close_logging()
    return 0


if __name__ == "__main__":
    sys.exit(main())
