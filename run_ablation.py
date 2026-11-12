"""
Ablation & split-aware federated pipeline -- single-command orchestrator.

Execution order (decision #8, confirmed 2026-09-10, extended
2026-09-11 per the split-aware audit): Phase 1-3 (existing, UNCHANGED,
no fitting/leakage risk) -> permanent split manifest -> split-aware
Phase 4/5 (normalization/vocabulary fit from TRAIN patients only) ->
Section 1's 72-config ablation -> Section 2's graph-embedding
ablation -> split-aware Phase 7/8/11/12/13 (fusion/graph reused
unchanged, fed split-aware tensors; local/FedAvg/baselines trained on
a TRAIN-ONLY SUBGRAPH per hospital) -> comprehensive validation report.

CRITICAL: Phase 4/5 are ALWAYS run split-aware here (train_patient_ids
threaded through to both), for EVERY step that needs them -- this
closes a real gap found while auditing the ablation framework's own
preprocessing (2026-09-11): the flat model's ENCODER weights were
already correctly trained train-only, but the RAW input tensors feeding
it were being normalized/vocabulary-fit against the FULL cohort
(val/test included) by the original, non-split-aware Phase 4/5 call.
"Fit representation" (requirement #3) means the whole representation
pipeline, not just the neural encoder's own weights.

OUTPUT ISOLATION (requirement #11 -- keep full-cohort Phase 11-13
results intact, separate directories for split-aware experiments):
Phase 4/5/7/8 write under `run.output_dir`, which they derive fresh
from whatever `cfg` they're given -- so split-aware calls are made with
a CFG COPY whose `run.output_dir` is redirected to
`{original_output_dir}/split_aware`, leaving the original, full-cohort
Phase 4/5/7/8 output completely untouched. Phase 11/12/13 write under
`run.checkpoint_dir` and already self-isolate via their own
`checkpoint_dir_name` defaulting (`*_splitaware` suffix) when given
`train_patient_ids` -- they're called with the ORIGINAL cfg (they don't
read `run.output_dir` at all, only `run.checkpoint_dir` and
hyperparameters, so no redirection is needed there).

USAGE (see README.md / IMPLEMENTATION_TRACKER.md for the full writeup):

  # Everything, one command, fresh Phase 1-3:
  python run_ablation.py --config configs/subset.yaml

  # Re-run just the ablation/federated steps against an
  # ALREADY-COMPLETED Phase 1-3 (same run.output_dir), for fast
  # iteration -- Phase 4/5 are cheap enough (seconds, not minutes) to
  # always redo fresh rather than also caching them:
  python run_ablation.py --config configs/subset.yaml --reuse-phase1to3 --steps embedding_ablation,graph_ablation

  # Only the permanent split manifest (idempotent):
  python run_ablation.py --config configs/subset.yaml --steps split

  # Only the split-aware federated stage (Phase 7/8/11/12/13):
  python run_ablation.py --config configs/subset.yaml --steps federated_split_aware

  # A fast smoke test bounded to the first N hospitals:
  python run_ablation.py --config configs/subset.yaml --max-hospitals 3

Results land under `{ablation.output_dir}/{run_id}/` for the ablation
records/summary tables, `{run.checkpoint_dir}/phase1{1,2,3}_*_splitaware/`
for the federated stage, and `{ablation.output_dir}/{run_id}/
validation_report.{json,md}` for the comprehensive report (requirement
#12) -- exact patient counts per partition, hospital-wise counts,
mortality/sepsis distributions, a zero-overlap confirmation,
confirmation that preprocessing fitting used training data only, Phase
11/12/13 results, ablation results, and any failures.
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import sys
import time
from pathlib import Path

from run_pipeline import (
    load_partition_result,
    load_phase1_summary,
    load_phase3_result,
    run_phase1_ingestion,
    run_phase2_hospital_partition,
    run_phase3_cohort_construction,
    run_phase4_timeseries_modality,
    run_phase5_tabular_modality,
    run_phase7_multimodal_fusion,
    run_phase8_patient_graph_construction,
    run_phase11_local_training,
    run_phase12_fedavg,
    run_phase13_baselines,
)
from src.ablation.config_grid import validate_ablation_config
from src.ablation.embedding_ablation_runner import run_embedding_ablation
from src.ablation.graph_comparison_runner import run_graph_comparison
from src.ablation.graph_embedding_ablation_runner import run_graph_embedding_ablation
from src.ablation.results_writer import ResultsWriter, new_run_id
from src.ablation.sampling_comparison_runner import run_sampling_comparison
from src.data.splitting.patient_split import build_and_save_split_manifest
from src.split_aware.evaluation import (
    evaluate_final_model_on_full_graphs,
    evaluate_hospital_results,
    evaluate_model_predictions,
    evaluate_predictions_by_hospital_dict,
)
from src.split_aware.reporting import build_validation_report, save_validation_report
from src.utils.config import load_config
from src.utils.logging_setup import close_logging, setup_logging

VALID_STEPS = ("split", "embedding_ablation", "graph_ablation", "sampling_comparison", "graph_comparison", "federated_split_aware")


def _get_phase1to3_fresh(cfg):
    phase1_summary = run_phase1_ingestion(cfg)
    partition_result, patient_df, _hospital_df = run_phase2_hospital_partition(cfg, phase1_summary)
    phase3_result = run_phase3_cohort_construction(cfg, phase1_summary, partition_result, patient_df)
    return phase1_summary, partition_result, phase3_result


def _get_phase1to3_reused(cfg):
    phase1_summary = load_phase1_summary(cfg)
    partition_result = load_partition_result(cfg)
    phase3_result = load_phase3_result(cfg, phase1_summary)
    return phase1_summary, partition_result, phase3_result


def _make_split_aware_output_cfg(cfg):
    """Returns a COPY of cfg whose run.output_dir is redirected to a
    `split_aware` subdirectory -- see module docstring's "OUTPUT
    ISOLATION" section. Only affects Phase 4/5/7/8, which derive their
    file paths from `run.output_dir` fresh on every call; Phase
    11/12/13 read `run.checkpoint_dir` (left untouched here) and
    self-isolate via their own `checkpoint_dir_name` parameter instead.
    """
    new_cfg = copy.deepcopy(cfg)
    original_output_dir = cfg.get("run.output_dir", "outputs")
    new_cfg.raw.setdefault("run", {})["output_dir"] = str(Path(original_output_dir) / "split_aware")
    return new_cfg


def _resolve_under_output_dir(cfg, relative_or_absolute_path: str) -> str:
    """Every other phase in this project writes under `run.output_dir`
    (e.g. Phase 8 -> `{run.output_dir}/graphs`), so a bare relative
    `ablation.output_dir`/`ablation.split.manifest_path` value (e.g.
    'outputs/ablation') should resolve the SAME way -- relative to
    `run.output_dir`'s PARENT (since these config defaults already
    include the leading 'outputs/' segment themselves, matching every
    other path already written into base.yaml, e.g.
    `run.checkpoint_dir: checkpoints`). An already-absolute path is
    left untouched.
    """
    path = Path(relative_or_absolute_path)
    if path.is_absolute():
        return str(path)
    run_output_dir = Path(cfg.get("run.output_dir", "outputs"))
    return str(run_output_dir.parent / path)


def run_federated_split_aware(cfg, split_aware_output_cfg, phase3_result, partition_result, phase4_result, phase5_result, train_patient_ids, split_manifest, logger):
    """Phase 7/8 (reused unchanged, redirected output dir) -> Phase
    11 -> Phase 12 -> Phase 13, all split-aware. Returns
    (phase11_result, phase11_eval, phase12_result, phase12_eval, phase13_result, phase13_eval).

    `phase12_eval`/the `local_only` entry of `phase13_eval` close a real
    gap found during the 2026-09-11 pilot review: Phase 12's own
    per-round local training only ever produces predictions from each
    hospital's POST-LOCAL-TRAINING weights, never from the FINAL
    AGGREGATED weights FedAvg actually outputs -- so without an extra,
    dedicated frozen-inference pass (`evaluate_final_model_on_full_
    graphs()`), the report could describe local/centralized but not
    the federated result itself, which is the central research
    question this whole baseline suite exists to answer. Likewise,
    Phase 13's LOCAL-ONLY baseline (the fairest direct comparison
    point against both centralized and federated, since all three share
    the same total epoch budget) was being trained and checkpointed
    correctly but never evaluated in the report -- only centralized was.
    """
    logger.info("[FEDERATED_SPLIT_AWARE] Phase 7 (fusion, reused unchanged, split-aware output dir)")
    phase7_result = run_phase7_multimodal_fusion(split_aware_output_cfg, phase3_result, phase4_result, phase5_result)

    logger.info("[FEDERATED_SPLIT_AWARE] Phase 8 (intra-hospital graph, reused unchanged, split-aware output dir)")
    phase8_result = run_phase8_patient_graph_construction(split_aware_output_cfg, phase7_result, partition_result)

    logger.info("[FEDERATED_SPLIT_AWARE] Phase 11 (local training, train-only subgraph)")
    phase11_result = run_phase11_local_training(
        cfg, phase3_result, phase4_result, phase5_result, phase8_result, train_patient_ids=train_patient_ids,
    )
    phase11_eval = evaluate_hospital_results(phase11_result["hospital_results"], split_manifest)

    logger.info("[FEDERATED_SPLIT_AWARE] Phase 12 (FedAvg, train-only subgraph per round)")
    phase12_result = run_phase12_fedavg(
        cfg, phase3_result, phase4_result, phase5_result, phase8_result, train_patient_ids=train_patient_ids,
    )
    sepsis_cohort = cfg.get("multitask.sepsis_cohort", "broad")
    logger.info("[FEDERATED_SPLIT_AWARE] evaluating Phase 12's FINAL AGGREGATED model (one extra frozen inference pass)")
    phase12_predictions = evaluate_final_model_on_full_graphs(
        cfg, phase3_result, phase4_result, phase5_result, phase8_result,
        phase12_result["final_global_model_path"], train_patient_ids, sepsis_cohort,
    )
    phase12_eval = evaluate_predictions_by_hospital_dict(phase12_predictions, split_manifest)

    logger.info("[FEDERATED_SPLIT_AWARE] Phase 13 (local-only + centralized-sequential baselines, train-only subgraph)")
    phase13_result = run_phase13_baselines(
        cfg, phase3_result, phase4_result, phase5_result, phase8_result, train_patient_ids=train_patient_ids,
    )
    phase13_eval = {
        "local_only": evaluate_hospital_results(phase13_result["local_only"]["hospital_results"], split_manifest),
        "centralized_sequential": {"mortality": {}, "sepsis": {}},
    }
    preds_by_hospital = phase13_result.get("centralized_sequential", {}).get("predictions_by_hospital")
    if preds_by_hospital:
        phase13_eval["centralized_sequential"] = {
            "mortality": evaluate_model_predictions(preds_by_hospital, split_manifest, "mortality"),
            "sepsis": evaluate_model_predictions(preds_by_hospital, split_manifest, "sepsis"),
        }

    return phase11_result, phase11_eval, phase12_result, phase12_eval, phase13_result, phase13_eval


def _build_cumulative_ablation_summary(results_writer: ResultsWriter) -> "dict | None":
    """Builds the validation report's Ablation summary from EVERY
    PERSISTED record under this run_id (`records/*.json`), not from
    what this single process invocation happened to compute.

    BUG FOUND AND FIXED 2026-09-16 (found via a real, in-progress
    full-scale run, not hypothetical): the 72-config grid is run in
    separate batches (`--steps embedding_ablation` invoked once per
    encoder, plus a separate `--steps graph_ablation` call), each a
    fresh process. The previous version of this function used the
    in-memory `results` dict, which only ever holds what THIS
    invocation computed -- so each new batch's `validation_report.md`
    silently overwrote the previous one's Ablation summary section,
    making it look like only the most recent batch had ever run.
    Fixed by reading `results_writer.build_summary_table()` (which
    scans every JSON record on disk for this run_id, regardless of
    which of possibly many separate CLI invocations wrote it) instead
    -- the report's Ablation summary is now correct no matter how many
    sittings the grid took, and is even correct when this specific
    invocation ran zero ablation experiments (e.g. `--steps
    federated_split_aware` alone still reports on whatever ablation
    batches already completed earlier under this run_id).
    """
    rows = results_writer.build_summary_table()
    if not rows:
        return None
    summary: dict = {}
    for kind in ("embedding_ablation", "graph_ablation"):
        kind_rows = [r for r in rows if r.get("experiment_kind") == kind]
        if not kind_rows:
            continue
        combos = sorted({r["experiment_id"].rsplit("_clf-", 1)[0] for r in kind_rows})
        best = kind_rows[0]  # build_summary_table() already sorts by test AUROC descending
        summary[kind] = {
            "n_experiments_recorded": len(kind_rows),
            "n_distinct_combos": len(combos),
            "combos": combos,
            "best_by_test_auroc": {
                "experiment_id": best["experiment_id"],
                "test_auroc": (best.get("metrics", {}).get("test", {}) or {}).get("auroc"),
            },
        }
    return summary or None


def run_ablation_suite(cfg, ablation_cfg: dict, steps: list, reuse_phase1to3: bool, run_id: str, overwrite_split: bool) -> dict:
    logger = logging.getLogger("eicu_fl")
    validate_ablation_config(ablation_cfg)
    failures: list = []

    logger.info(f"[ABLATION] steps={steps} reuse_phase1to3={reuse_phase1to3} run_id={run_id}")
    if reuse_phase1to3:
        _phase1_summary, partition_result, phase3_result = _get_phase1to3_reused(cfg)
    else:
        _phase1_summary, partition_result, phase3_result = _get_phase1to3_fresh(cfg)

    split_cfg = ablation_cfg.get("split", {})
    manifest_path = _resolve_under_output_dir(cfg, split_cfg.get("manifest_path", "outputs/ablation/split_manifest.json"))
    split_manifest = build_and_save_split_manifest(
        partition_result, phase3_result["cohort_patients"], phase3_result["mortality"], phase3_result["sepsis_df"],
        manifest_path, train_frac=split_cfg.get("train_frac", 0.7), val_frac=split_cfg.get("val_frac", 0.15),
        test_frac=split_cfg.get("test_frac", 0.15), seed=cfg.get("run.seed", 42),
        stratify_on=split_cfg.get("stratify_on", "mortality"), sepsis_cohort=cfg.get("multitask.sepsis_cohort", "broad"),
        overwrite=overwrite_split,
    )
    train_patient_ids = set(split_manifest.all_patient_ids("train"))
    logger.info(
        f"[ABLATION] split manifest: {split_manifest.n_patients_total} patients, "
        f"{split_manifest.n_hospitals_total} hospitals ({split_manifest.n_hospitals_stratified} stratified, "
        f"{split_manifest.n_hospitals_fallback_unstratified} unstratified fallback, "
        f"{split_manifest.n_hospitals_all_train_too_small} too small for a 3-way split)."
    )

    results: dict = {"split": {
        "n_patients_total": split_manifest.n_patients_total, "n_hospitals_total": split_manifest.n_hospitals_total,
    }}

    needs_phase4_5 = any(
        s in steps for s in
        ("embedding_ablation", "graph_ablation", "sampling_comparison", "graph_comparison", "federated_split_aware")
    )
    phase4_result = phase5_result = None
    split_aware_output_cfg = _make_split_aware_output_cfg(cfg)
    if needs_phase4_5:
        logger.info(f"[ABLATION] running Phase 4/5 SPLIT-AWARE ({len(train_patient_ids)} train patients) -> {split_aware_output_cfg.get('run.output_dir')}")
        phase4_result = run_phase4_timeseries_modality(split_aware_output_cfg, phase3_result, partition_result, train_patient_ids=train_patient_ids)
        phase5_result = run_phase5_tabular_modality(split_aware_output_cfg, _phase1_summary, phase3_result, partition_result, train_patient_ids=train_patient_ids)

    ablation_output_dir = _resolve_under_output_dir(cfg, ablation_cfg.get("output_dir", "outputs/ablation"))
    results_writer = ResultsWriter(ablation_output_dir, run_id=run_id)

    if "embedding_ablation" in steps:
        try:
            start = time.time()
            results["embedding_ablation"] = run_embedding_ablation(
                cfg, ablation_cfg, phase3_result, phase4_result, phase5_result, partition_result, split_manifest, results_writer,
            )
            logger.info(f"[ABLATION] embedding_ablation step complete in {time.time() - start:.1f}s")
        except Exception as e:
            logger.exception("[ABLATION] embedding_ablation step failed")
            failures.append(f"embedding_ablation: {e!r}")

    if "graph_ablation" in steps:
        try:
            start = time.time()
            results["graph_ablation"] = run_graph_embedding_ablation(
                cfg, ablation_cfg, phase3_result, phase4_result, phase5_result, partition_result, split_manifest, results_writer,
            )
            logger.info(f"[ABLATION] graph_ablation step complete in {time.time() - start:.1f}s")
        except Exception as e:
            logger.exception("[ABLATION] graph_ablation step failed")
            failures.append(f"graph_ablation: {e!r}")

    if "sampling_comparison" in steps:
        try:
            start = time.time()
            sampling_cfg = ablation_cfg.get("sampling", {}) or {}
            reference_master_rows = None
            reference_csv_path = sampling_cfg.get("reference_master_results_csv")
            if reference_csv_path and Path(reference_csv_path).exists():
                with open(reference_csv_path, newline="") as f:
                    reference_master_rows = list(csv.DictReader(f))
                logger.info(f"[ABLATION] sampling_comparison: loaded {len(reference_master_rows)} reference rows from {reference_csv_path} for the no_sampling reproduction check")
            elif reference_csv_path:
                logger.warning(f"[ABLATION] sampling_comparison: reference_master_results_csv={reference_csv_path!r} does not exist -- proceeding without the no_sampling reproduction check.")
            results["sampling_comparison"] = run_sampling_comparison(
                cfg, ablation_cfg, phase3_result, phase4_result, phase5_result, partition_result, split_manifest,
                results_writer, reference_master_rows=reference_master_rows,
            )
            if results["sampling_comparison"].get("stopped_before_sampling"):
                logger.error(
                    f"[ABLATION] sampling_comparison STOPPED before undersample/oversample -- reproduction check "
                    f"failed at combo {results['sampling_comparison'].get('stopped_at_combo')!r}. "
                    f"{results['sampling_comparison'].get('n_reproduction_mismatches')} mismatch(es) out of "
                    f"{results['sampling_comparison'].get('n_reproduction_checked')} checked. Diagnose via the "
                    f"written no_sampling records' training_info.reproduction_check_detail before re-running."
                )
            else:
                logger.info(f"[ABLATION] sampling_comparison step complete in {time.time() - start:.1f}s")
        except Exception as e:
            logger.exception("[ABLATION] sampling_comparison step failed")
            failures.append(f"sampling_comparison: {e!r}")

    if "graph_comparison" in steps:
        try:
            start = time.time()
            graph_comparison_cfg = ablation_cfg.get("graph_comparison", {}) or {}
            best_flat_config = graph_comparison_cfg.get("best_flat_config")
            if not best_flat_config:
                raise ValueError(
                    "ablation.graph_comparison.best_flat_config is not set -- graph_comparison requires the "
                    "winning {ts_model, tab_model, fusion_mode} from the real sampling comparison's results "
                    "(see src/ablation/graph_comparison_runner.py's select_best_flat_config_from_sampling_results()). "
                    "Never guessed automatically."
                )
            results["graph_comparison"] = run_graph_comparison(
                cfg, ablation_cfg, phase3_result, phase4_result, phase5_result, partition_result, split_manifest,
                results_writer, best_flat_config=best_flat_config,
            )
            logger.info(f"[ABLATION] graph_comparison step complete in {time.time() - start:.1f}s")
        except Exception as e:
            logger.exception("[ABLATION] graph_comparison step failed")
            failures.append(f"graph_comparison: {e!r}")

    if "embedding_ablation" in steps or "graph_ablation" in steps or "sampling_comparison" in steps or "graph_comparison" in steps:
        summary_path = results_writer.save_summary_csv()
        logger.info(f"[ABLATION] summary table written to {summary_path}")
        results["summary_csv_path"] = str(summary_path)
        results["run_dir"] = str(results_writer.run_dir)

    phase11_result = phase11_eval = phase12_result = phase12_eval = phase13_result = phase13_eval = None
    if "federated_split_aware" in steps:
        try:
            start = time.time()
            phase11_result, phase11_eval, phase12_result, phase12_eval, phase13_result, phase13_eval = run_federated_split_aware(
                cfg, split_aware_output_cfg, phase3_result, partition_result, phase4_result, phase5_result,
                train_patient_ids, split_manifest, logger,
            )
            results["federated_split_aware"] = {
                "n_hospitals_trained_phase11": phase11_result["n_hospitals_trained"],
                "n_rounds_phase12": phase12_result["n_rounds"],
                "n_hospitals_phase13_centralized": phase13_result["centralized_sequential"]["n_hospitals"],
            }
            logger.info(f"[ABLATION] federated_split_aware step complete in {time.time() - start:.1f}s")
        except Exception as e:
            logger.exception("[ABLATION] federated_split_aware step failed")
            failures.append(f"federated_split_aware: {e!r}")

    report = build_validation_report(
        split_manifest, phase3_result["cohort_patients"], phase3_result["mortality"], phase3_result["sepsis_df"],
        phase11_result=phase11_result, phase11_eval=phase11_eval, phase12_result=phase12_result,
        phase12_eval=phase12_eval, phase13_result=phase13_result, phase13_eval=phase13_eval,
        ablation_summary=_build_cumulative_ablation_summary(results_writer),
        failures=failures,
    )
    report_json_path, report_md_path = save_validation_report(report, results_writer.run_dir)
    logger.info(f"[ABLATION] validation report written to {report_md_path}")
    results["validation_report_path"] = str(report_md_path)
    results["failures"] = failures

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Split-aware ablation + federated pipeline orchestrator.")
    parser.add_argument("--config", required=True, help="Path to a mode config, e.g. configs/subset.yaml")
    parser.add_argument(
        "--steps", default="split,embedding_ablation,graph_ablation,federated_split_aware",
        help=(
            f"Comma-separated subset of {VALID_STEPS} to run "
            f"(default: split,embedding_ablation,graph_ablation,federated_split_aware -- unchanged from before "
            f"sampling_comparison/graph_comparison existed, so neither runs unless explicitly requested, e.g. "
            f"--steps split,sampling_comparison to reuse the existing 48/6-row ablation without rerunning it, "
            f"then --steps split,graph_comparison once ablation.graph_comparison.best_flat_config is set from "
            f"the real sampling comparison's winning row)."
        ),
    )
    parser.add_argument(
        "--reuse-phase1to3", action="store_true",
        help="Load Phase 1-3 output from disk (run.output_dir) instead of running fresh -- requires "
             "`python run_pipeline.py --config <same config> --phase 3` to have already completed. "
             "Phase 4/5 are always re-run split-aware regardless (cheap, seconds not minutes).",
    )
    parser.add_argument("--run-id", default=None, help="Results directory name under ablation.output_dir (default: a fresh UTC timestamp).")
    parser.add_argument("--overwrite-split", action="store_true", help="Force-regenerate the split manifest even if one already exists.")
    parser.add_argument("--max-hospitals", type=int, default=None, help="Override ablation.local_training.max_hospitals for a quick smoke test.")
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    for s in steps:
        if s not in VALID_STEPS:
            parser.error(f"--steps contains {s!r}, must be one of {VALID_STEPS}")

    cfg = load_config(args.config)
    setup_logging(cfg.get("run.log_dir", "outputs/logs"))
    logger = logging.getLogger("eicu_fl")

    ablation_cfg = cfg.get("ablation", {}) or {}
    if not ablation_cfg.get("enabled", True):
        logger.warning("[ABLATION] ablation.enabled is false in config -- nothing to do.")
        close_logging()
        return
    if args.max_hospitals is not None:
        ablation_cfg.setdefault("local_training", {})["max_hospitals"] = args.max_hospitals

    run_id = args.run_id or new_run_id()
    start = time.time()
    try:
        results = run_ablation_suite(cfg, ablation_cfg, steps, args.reuse_phase1to3, run_id, args.overwrite_split)
    except Exception:
        logger.exception("[ABLATION] run failed")
        close_logging()
        raise
    logger.info(
        f"[ABLATION] ALL REQUESTED STEPS COMPLETE ({time.time() - start:.1f}s) -- "
        f"validation report: {results.get('validation_report_path')}"
    )
    if results.get("failures"):
        logger.warning(f"[ABLATION] completed WITH {len(results['failures'])} step failure(s): {results['failures']}")
    close_logging()


if __name__ == "__main__":
    sys.exit(main())