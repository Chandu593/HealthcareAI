"""
Ablation framework -- class-imbalance sampling comparison (confirmed
with the user 2026-09-20).

Compares three per-(combo, classifier, task) arms over the ALREADY-
COMPLETED 48 non-PCA combos (4 TS x 2 non-PCA TAB x 2 fusion, PCA
combos excluded here regardless of how the grid is configured -- PCA
stays deferred):

  - no_sampling: refits the classifier on the SAME frozen embeddings
    the original ablation used. Exists in this module purely as an
    IN-RUN REPRODUCTION CHECK against the already-written
    master_results.csv row for that (combo, classifier, task) -- not a
    new independent measurement. Still written to `sampling_results.csv`
    (never master_results.csv) so the three arms sit side by side for
    comparison/plotting.
  - undersample / oversample (always on by default) and, since
    2026-10-03, adasyn / enn / adasyn_enn (config-selectable via
    `ablation.sampling.strategies`, see run_sampling_comparison()'s own
    docstring) -- all five are hospital-wise, train-only resampling
    (src/classifiers/imbalance_sampling.py), with a small validation-
    driven ratio search per (combo, classifier, task) -- selection is
    F1 -> recall -> ... -> Brier (confirmed with the user), never AUPRC
    (already used once, for the original no-sampling ablation) and
    never test metrics.

REAL, CONCRETE GAP THIS MODULE ORIGINALLY WORKED AROUND (found
2026-09-20 while building this, not hypothetical): at the time,
`run_embedding_ablation()` never persisted the frozen per-hospital-
trained embeddings it computes -- only the resulting classifier
METRICS were written to master_results.csv. There was therefore no
cached artifact to "reuse" for a new classifier fit on resampled data;
the per-hospital encoder training (the expensive part of the original
ablation) had to run again once per combo to regenerate those
embeddings. This was NOT the same as rerunning the completed ablation
to reconfirm its own results (explicitly forbidden) -- it was
producing a missing intermediate artifact needed for genuinely new
work, which the user's own stated exceptions for "missing output"
cover. To make this a ONE-TIME cost rather than a recurring one,
embeddings are cached to
`{ablation.sampling.embeddings_cache_dir}/{combo_id}.npz` the first
time they're computed and reloaded from there on any later run --
via `get_or_build_combo_embeddings()`, now defined in
`embedding_ablation_runner.py` (see that function's docstring).

UPDATE 2026-09-29: `run_embedding_ablation()` (the flat, no-sampling
48-config ablation) now ALSO goes through this same cache-first path
-- previously only this module and `graph_comparison_runner.py` did.
This closes the gap described above from the other direction too: a
combo already trained by a `--steps embedding_ablation` batch (e.g.
`--config configs/full_batch_lstm.yaml`) is now reused here rather
than retrained, and vice versa. All three ablation steps therefore
share one on-disk embedding cache, keyed by `combo_id` alone (not by
which step computed it first).

Since the encoder-training process itself is untouched (the exact same
`_train_combo_and_extract_embeddings()` the original ablation uses,
same config, same seed), the regenerated embeddings are expected to
reproduce the ORIGINAL no-sampling classifier metrics exactly (within
floating-point tolerance) -- checked directly against
`reference_master_rows` when supplied.

GATING BEHAVIOR (confirmed with the user 2026-09-23): the reproduction
check compares AUROC on ALL THREE splits (train/val/test), not test
alone (see `_check_reproduction()`). Every no_sampling record is
written regardless of outcome -- pass or fail -- so a mismatch is
fully diagnosable from `sampling_results.csv` itself (each record's
`training_info.reproduction_check_detail` carries the exact
reference/regenerated/abs_diff per split). But if ANY of a combo's
checks explicitly mismatch, `run_sampling_comparison()` STOPS
immediately, before running undersample/oversample for that combo OR
any combo after it (returned as `stopped_before_sampling=True`,
`stopped_at_combo=<combo_id>`) -- a mismatch signals something
systemic (wrong data root, wrong split manifest, wrong `run.seed`,
code drift since the reference was produced), not a fluke isolated to
one combo, so later combos are never even attempted rather than
burning more of the expensive embedding-regeneration compute on a run
that's already known to be untrustworthy.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

from src.ablation.config_grid import expand_embedding_combo_grid
from src.ablation.embedding_ablation_runner import (
    _build_ts_tab_lookup,
    get_or_build_combo_embeddings,
)
from src.ablation.results_writer import ExperimentRecord, ResultsWriter
from src.classifiers.boosting_classifiers import fit_and_evaluate_classifier
from src.classifiers.imbalance_sampling import search_sampling_ratio
from src.data.splitting.patient_split import SplitManifest
from src.heads.multitask_heads import align_labels_to_patient_ids

DEFAULT_CANDIDATE_RATIOS = (0.5, 0.75, 1.0)


def _hospital_id_lookup(partition_result) -> dict:
    """patient_id -> hospital_id, built once from Phase 2's partition
    (the same source of truth every other phase uses for hospital
    membership) -- needed here because a combo's pooled, cross-hospital
    `patient_ids` array doesn't carry hospital identity by itself, and
    hospital-wise resampling requires it."""
    lookup: dict = {}
    for hid, partition in partition_result.partitions.items():
        for pid in partition.patient_ids:
            lookup[int(pid)] = hid
    return lookup


REPRODUCTION_CHECK_SPLITS = ("train", "val", "test")
REPRODUCTION_CHECK_METRIC = "auroc"
REPRODUCTION_CHECK_TOLERANCE = 1e-6


def _reference_metric(reference_row: dict, split: str, metric: str) -> "float | None":
    """Reads one metric from a reference row, accepting either shape: a
    flat CSV row (csv.DictReader over master_results.csv -- dotted
    keys, e.g. "test.auroc", values are strings) OR a nested JSON
    record (e.g. ResultsWriter.build_summary_table() -- metrics.test.
    auroc, values are already floats)."""
    val = reference_row.get(f"{split}.{metric}")
    if val is None:
        val = ((reference_row.get("metrics") or {}).get(split) or {}).get(metric)
    if val in (None, "", "None"):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _check_reproduction(
    result, reference_row: "dict | None", logger: logging.Logger, experiment_id: str,
    tol: float = REPRODUCTION_CHECK_TOLERANCE,
) -> "tuple[bool | None, dict]":
    """Compares the freshly-regenerated no_sampling classifier against
    the SAME experiment_id's already-written master_results.csv row,
    when one is supplied -- confirmed with the user 2026-09-23 to check
    AUROC on ALL THREE splits (train, val, test), not test alone, so a
    discrepancy confined to one split (e.g. only training data lining
    up, or only the held-out sets) is still caught rather than averaged
    away or missed.

    Returns `(overall_matched, detail)`:
      - `overall_matched` is `True` only if every split that had BOTH a
        reference and a regenerated value matched within `tol`; `False`
        if any comparable split mismatched; `None` if there was nothing
        comparable at all (e.g. no reference row, or the reference row
        has no AUROC recorded for any split) -- distinct from `False`
        so callers can tell "checked and matched", "checked and
        mismatched", and "nothing to check" apart.
      - `detail` is `{split: {"reference": float|None, "regenerated":
        float|None, "matched": bool|None, "abs_diff": float|None}}` for
        all three splits, always -- written into the record's
        `training_info` either way, so a mismatch (or a clean match) is
        traceable per split without re-running anything.

    Logs a WARNING -- never silently swallowed -- on any real mismatch,
    since the encoder-training process is deterministic given the same
    config/seed (see `_hospital_model_seed()`), so a mismatch means
    something upstream genuinely changed (wrong data root, wrong split
    manifest, wrong `run.seed`, code drift since the reference was
    produced) and needs diagnosis, not a silent pass-through.
    """
    detail: dict = {split: {"reference": None, "regenerated": None, "matched": None, "abs_diff": None} for split in REPRODUCTION_CHECK_SPLITS}
    if reference_row is None:
        return None, detail

    any_compared = False
    all_matched = True
    for split in REPRODUCTION_CHECK_SPLITS:
        ref_val = _reference_metric(reference_row, split, REPRODUCTION_CHECK_METRIC)
        got_val = (getattr(result, f"{split}_metrics") or {}).get(REPRODUCTION_CHECK_METRIC)
        if ref_val is None or got_val is None:
            continue
        any_compared = True
        abs_diff = abs(ref_val - got_val)
        split_matched = abs_diff <= tol
        detail[split] = {"reference": ref_val, "regenerated": got_val, "matched": split_matched, "abs_diff": abs_diff}
        if not split_matched:
            all_matched = False

    if not any_compared:
        return None, detail

    if not all_matched:
        mismatched = {s: d for s, d in detail.items() if d["matched"] is False}
        logger.warning(
            f"[SAMPLING_COMPARISON] {experiment_id}: regenerated no_sampling {REPRODUCTION_CHECK_METRIC.upper()} does NOT "
            f"match the reference master_results.csv row within tolerance={tol} for: {mismatched} -- investigate "
            f"before trusting anything built on these regenerated embeddings."
        )
    return all_matched, detail


def _write_sampling_record(
    results_writer: ResultsWriter, combo, classifier_name: str, task: str, strategy: str,
    ratio: "float | None", result, split_manifest: SplitManifest, extra_training_info: dict,
) -> None:
    experiment_id = f"{combo.combo_id}_clf-{classifier_name}_task-{task}_samp-{strategy}"
    record = ExperimentRecord(
        experiment_id=experiment_id, experiment_kind="sampling_comparison",
        config={
            "ts_model": combo.ts_model, "tab_model": combo.tab_model, "fusion_mode": combo.fusion_mode,
            "classifier": classifier_name, "sampling_strategy": strategy, "sampling_ratio": ratio,
        },
        task=task, split_manifest_seed=split_manifest.seed,
        metrics={"train": result.train_metrics, "val": result.val_metrics, "test": result.test_metrics},
        training_info={
            "classifier_skipped": result.skipped, "classifier_skip_reason": result.skip_reason,
            "n_train_used": result.n_train_used, "n_val_used": result.n_val_used, "n_test_used": result.n_test_used,
            **extra_training_info,
        },
        runtime_seconds=result.fit_runtime_seconds, timestamp=datetime.now(timezone.utc).isoformat(),
    )
    results_writer.write(record)


def run_sampling_comparison(
    cfg, ablation_cfg: dict, phase3_result: dict, phase4_result: dict, phase5_result: dict,
    partition_result, split_manifest: SplitManifest, results_writer: ResultsWriter,
    reference_master_rows: "list[dict] | None" = None,
) -> dict:
    """Single entry point for the whole class-imbalance sampling
    comparison -- one call covers embedding regeneration/caching,
    the no_sampling reproduction check, and the ratio-tuned search for
    every strategy in `ablation.sampling.strategies` (default
    `["undersample", "oversample"]`; `"adasyn"`, `"enn"`, and
    `"adasyn_enn"` are also valid, added 2026-10-03 -- see
    imbalance_sampling.py's module docstring) for every (combo,
    classifier, task), writing every result to `sampling_results.csv`
    via `results_writer`. Wired into `run_ablation.py` as the
    `sampling_comparison` step so the whole thing runs from a single
    command (confirmed with the user 2026-09-20).

    `reference_master_rows`, optional: the already-loaded
    master_results.csv rows from the original no-sampling ablation,
    keyed by matching `experiment_id` -- either flat CSV rows (e.g.
    `list(csv.DictReader(f))`) or nested JSON records (e.g.
    `ResultsWriter.build_summary_table()`), used only for the
    reproduction check (see `_check_reproduction()`). The undersample/
    oversample arms don't depend on it at all, so omitting it doesn't
    block them.
    """
    logger = logging.getLogger("eicu_fl")
    sepsis_cohort = cfg.get("multitask.sepsis_cohort", "broad")
    local_training_cfg = ablation_cfg.get("local_training", {})
    max_hospitals = local_training_cfg.get("max_hospitals")
    sampling_cfg = ablation_cfg.get("sampling", {}) or {}
    seed = sampling_cfg.get("seed", 42)
    undersample_ratios = sampling_cfg.get("candidate_undersample_ratios", list(DEFAULT_CANDIDATE_RATIOS))
    oversample_ratios = sampling_cfg.get("candidate_oversample_ratios", list(DEFAULT_CANDIDATE_RATIOS))
    adasyn_ratios = sampling_cfg.get("candidate_adasyn_ratios", list(DEFAULT_CANDIDATE_RATIOS))
    enn_n_neighbors = sampling_cfg.get("enn_n_neighbors", 3)
    cache_dir = sampling_cfg.get("embeddings_cache_dir", "outputs/ablation/embeddings_cache")

    # Which strategies actually run, config-driven (added 2026-10-03
    # alongside adasyn/enn/adasyn_enn -- see imbalance_sampling.py's
    # module docstring "UPDATE" note). Default is EXACTLY the original
    # two, unchanged, so an existing config that doesn't set
    # `ablation.sampling.strategies` runs identically to before this
    # change.
    strategy_ratio_lookup = {
        "undersample": undersample_ratios, "oversample": oversample_ratios,
        "adasyn": adasyn_ratios, "enn": [None], "adasyn_enn": adasyn_ratios,
    }
    strategies_to_run = sampling_cfg.get("strategies", ["undersample", "oversample"])
    unknown = [s for s in strategies_to_run if s not in strategy_ratio_lookup]
    if unknown:
        raise ValueError(
            f"ablation.sampling.strategies contains unknown strategy name(s) {unknown}; "
            f"valid options are {sorted(strategy_ratio_lookup)}."
        )
    strategies_with_ratios = [(s, strategy_ratio_lookup[s]) for s in strategies_to_run]

    ts_row_of_patient, tab_row_of_patient = _build_ts_tab_lookup(phase4_result, phase5_result)
    ts_array = phase4_result["normalized_array"]
    ts_seq_length = phase4_result["sequence_length"]
    ts_input_dim = len(phase4_result["feature_names"])
    tab_array = phase5_result["normalized_array"]
    tab_raw_input_dim = len(phase5_result["feature_names"])

    hospital_ids = sorted(partition_result.partitions.keys())
    if max_hospitals is not None:
        hospital_ids = hospital_ids[:max_hospitals]
    hospital_id_of_patient = _hospital_id_lookup(partition_result)

    combos = expand_embedding_combo_grid(ablation_cfg)
    n_before = len(combos)
    combos = [c for c in combos if c.tab_model != "pca"]
    if len(combos) < n_before:
        logger.info(
            f"[SAMPLING_COMPARISON] excluding {n_before - len(combos)} PCA combo(s) -- PCA stays deferred "
            f"(confirmed with the user), regardless of what ablation.tab_models/subset.combos configures."
        )

    subset = ablation_cfg.get("subset") or {}
    classifiers = subset.get("classifiers") or ablation_cfg.get("classifiers", [])
    tasks = subset.get("tasks") or ["mortality", "sepsis"]
    classifier_params_cfg = ablation_cfg.get("classifier_params", {})
    reference_by_id = {r["experiment_id"]: r for r in (reference_master_rows or [])}

    combo_summaries: dict = {}
    reproduction_checks: list = []

    for combo_i, combo in enumerate(combos):
        logger.info(f"[SAMPLING_COMPARISON] ({combo_i + 1}/{len(combos)}) combo {combo.combo_id}")
        combo_result, combo_summary = get_or_build_combo_embeddings(
            combo, hospital_ids, partition_result, ts_row_of_patient, tab_row_of_patient, ts_array, ts_seq_length,
            ts_input_dim, tab_array, tab_raw_input_dim, cfg, ablation_cfg, phase3_result, sepsis_cohort,
            local_training_cfg, split_manifest, cache_dir, logger,
        )
        combo_summaries[combo.combo_id] = combo_summary
        if combo_result is None:
            continue

        label_info = align_labels_to_patient_ids(
            combo_result.patient_ids.tolist(), phase3_result["cohort_patients"], phase3_result["mortality"],
            phase3_result["sepsis_df"], sepsis_cohort,
        )
        labels_by_task = {"mortality": np.asarray(label_info["mortality"]), "sepsis": np.asarray(label_info["sepsis"])}
        train_mask = combo_result.split_by_row == "train"
        val_mask = combo_result.split_by_row == "val"
        test_mask = combo_result.split_by_row == "test"
        hospital_ids_by_row = np.array([hospital_id_of_patient[int(pid)] for pid in combo_result.patient_ids])
        hospital_ids_train = hospital_ids_by_row[train_mask]

        # --- Stage 1 for this combo: no_sampling fit + reproduction
        # check for every (task, classifier). Confirmed with the user
        # 2026-09-23: the reproduction check GATES the sampling work --
        # every no_sampling record is written regardless (so a mismatch
        # is fully diagnosable from sampling_results.csv), but if ANY
        # of this combo's checks explicitly mismatch, the run STOPS
        # before Stage 2 (undersample/oversample) runs for this combo
        # OR any combo after it -- a mismatch signals something
        # systemic (wrong data root, wrong split manifest, wrong
        # run.seed, code drift since the reference was produced), not
        # a per-combo fluke, so later combos are never even attempted
        # rather than burning more (expensive) embedding-regeneration
        # compute on a run that's already known to be untrustworthy.
        combo_no_sampling_results: dict = {}
        combo_had_mismatch = False
        for task in tasks:
            y = labels_by_task[task]
            for classifier_name in classifiers:
                params = classifier_params_cfg.get(classifier_name)
                no_sampling_result = fit_and_evaluate_classifier(
                    classifier_name, params, combo_result.embeddings[train_mask], y[train_mask],
                    combo_result.embeddings[val_mask], y[val_mask], combo_result.embeddings[test_mask], y[test_mask],
                    task=task, logger=logger,
                )
                reference_id = f"{combo.combo_id}_clf-{classifier_name}_task-{task}"
                matched, detail = _check_reproduction(no_sampling_result, reference_by_id.get(reference_id), logger, reference_id)
                reproduction_checks.append({"experiment_id": reference_id, "matched_reference": matched, "detail": detail})
                _write_sampling_record(
                    results_writer, combo, classifier_name, task, "no_sampling", None, no_sampling_result,
                    split_manifest,
                    {
                        "reproduction_check_matched_reference": matched, "reproduction_check_detail": detail,
                        "embeddings_from_cache": combo_summary.get("from_cache"),
                    },
                )
                combo_no_sampling_results[(task, classifier_name)] = no_sampling_result
                if matched is False:
                    combo_had_mismatch = True

        if combo_had_mismatch:
            n_mismatches = sum(1 for c in reproduction_checks if c["matched_reference"] is False)
            n_checked = sum(1 for c in reproduction_checks if c["matched_reference"] is not None)
            logger.error(
                f"[SAMPLING_COMPARISON] STOPPING before undersample/oversample: combo {combo.combo_id}'s "
                f"no_sampling reproduction check did not match the reference master_results.csv row(s) -- "
                f"{n_mismatches}/{n_checked} checked rows mismatched so far. No sampling (undersample/oversample) "
                f"will be run for this combo or any remaining combo. See reproduction_checks[*]['detail'] and "
                f"each written no_sampling record's training_info.reproduction_check_detail for exactly which "
                f"split(s)/metric(s) diverged."
            )
            return {
                "n_combos": len(combos), "combos": combo_summaries,
                "reproduction_checks": reproduction_checks,
                "n_reproduction_checked": n_checked, "n_reproduction_mismatches": n_mismatches,
                "stopped_before_sampling": True, "stopped_at_combo": combo.combo_id,
            }

        # --- Stage 2 for this combo: reproduction confirmed (or
        # nothing to check against) -- safe to spend the ratio-search
        # compute on it. ---
        for (task, classifier_name), _no_sampling_result in combo_no_sampling_results.items():
            y = labels_by_task[task]
            params = classifier_params_cfg.get(classifier_name)
            for strategy, ratios in strategies_with_ratios:
                best_result, trials = search_sampling_ratio(
                    classifier_name, params, combo_result.embeddings[train_mask], y[train_mask],
                    hospital_ids_train, combo_result.embeddings[val_mask], y[val_mask],
                    combo_result.embeddings[test_mask], y[test_mask], strategy=strategy,
                    candidate_ratios=ratios, task=task, seed=seed, enn_n_neighbors=enn_n_neighbors, logger=logger,
                )
                best_ratio = next((t["ratio"] for t in trials if t["result"] is best_result), None)
                _write_sampling_record(
                    results_writer, combo, classifier_name, task, strategy, best_ratio, best_result,
                    split_manifest,
                    {
                        "candidate_ratios_tried": ratios,
                        "ratio_search_trials": [
                            {
                                "ratio": t["ratio"], "val_metrics": t["result"].val_metrics,
                                "test_metrics": t["result"].test_metrics, "skipped": t["result"].skipped,
                            }
                            for t in trials
                        ],
                    },
                )
        logger.info(f"[SAMPLING_COMPARISON] combo {combo.combo_id} done")

    n_mismatches = sum(1 for c in reproduction_checks if c["matched_reference"] is False)
    n_checked = sum(1 for c in reproduction_checks if c["matched_reference"] is not None)
    logger.info(
        f"[SAMPLING_COMPARISON] no_sampling reproduction check complete: {n_checked - n_mismatches}/{n_checked} "
        f"matched the reference master_results.csv (0 mismatches -> full comparison ran; "
        f"n_checked==0 means no reference rows were available to check against)."
    )

    return {
        "n_combos": len(combos), "combos": combo_summaries,
        "reproduction_checks": reproduction_checks,
        "n_reproduction_checked": n_checked, "n_reproduction_mismatches": n_mismatches,
        "stopped_before_sampling": False,
    }
