"""End-to-end integration test for
src/ablation/sampling_comparison_runner.py against mock data --
exercises embedding regeneration + on-disk caching, the no_sampling
reproduction check against a prior real embedding_ablation run, and
the ratio-tuned undersample/oversample search, writing everything to
`sampling_results.csv` via ResultsWriter.
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_mock_eicu import generate  # noqa: E402
from run_pipeline import (  # noqa: E402
    run_phase1_ingestion,
    run_phase2_hospital_partition,
    run_phase3_cohort_construction,
    run_phase4_timeseries_modality,
    run_phase5_tabular_modality,
)
from src.ablation.embedding_ablation_runner import run_embedding_ablation  # noqa: E402
from src.ablation.results_writer import ResultsWriter  # noqa: E402
from src.ablation.sampling_comparison_runner import run_sampling_comparison  # noqa: E402
from src.data.splitting.patient_split import build_split_manifest  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestSamplingComparisonRunnerEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        # Enough hospitals/patients that both classes are usually
        # present per hospital, so undersample/oversample have real
        # work to do rather than hitting the single-class-hospital
        # passthrough on every hospital.
        generate(str(self.data_root), n_hospitals=6, n_patients_per_hospital=50, seed=21)

        self._old_env = os.environ.get("EICU_DATA_ROOT")
        os.environ["EICU_DATA_ROOT"] = str(self.data_root)

        self.cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        self.cfg.raw["run"]["output_dir"] = str(self.output_dir)
        self.cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")
        self.cfg.raw["cohort"]["patient_subset"] = 60
        self.cfg.raw.setdefault("modalities", {}).setdefault("timeseries", {})["encoder"] = "lstm"
        self.cfg.raw["modalities"].setdefault("tabular", {})["encoder"] = "autoencoder"

        phase1_summary = run_phase1_ingestion(self.cfg)
        self.partition_result, patient_df, hospital_df = run_phase2_hospital_partition(self.cfg, phase1_summary)
        self.phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, self.partition_result, patient_df)
        self.phase4_result = run_phase4_timeseries_modality(self.cfg, self.phase3_result, self.partition_result)
        self.phase5_result = run_phase5_tabular_modality(self.cfg, phase1_summary, self.phase3_result, self.partition_result)

        self.split_manifest = build_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], seed=42, stratify_on="mortality",
        )

        self.ablation_cfg = {
            "local_training": {"epochs": 2, "learning_rate": 0.01},
            "subset": {
                "combos": [{"ts_model": "lstm", "tab_model": "mlp", "fusion_mode": "concatenation"}],
                "classifiers": ["adaboost"], "tasks": ["mortality"],
            },
            "classifier_params": {"adaboost": {"n_estimators": 10}},
            "sampling": {
                "seed": 42, "candidate_undersample_ratios": [0.5, 1.0], "candidate_oversample_ratios": [0.5, 1.0],
                "embeddings_cache_dir": str(self.output_dir / "ablation" / "embeddings_cache"),
            },
        }

    def tearDown(self):
        close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def _run_sampling(self, run_id, reference_master_rows=None):
        results_writer = ResultsWriter(self.output_dir / "ablation", run_id=run_id)
        summary = run_sampling_comparison(
            self.cfg, self.ablation_cfg, self.phase3_result, self.phase4_result, self.phase5_result,
            self.partition_result, self.split_manifest, results_writer,
            reference_master_rows=reference_master_rows,
        )
        return summary, results_writer

    def test_writes_no_sampling_undersample_oversample_for_every_classifier_task(self):
        summary, results_writer = self._run_sampling(run_id="run_sampling_basic")
        self.assertEqual(summary["n_combos"], 1)

        record_files = sorted(p.name for p in results_writer.records_dir.glob("*.json"))
        # 1 combo x 1 classifier x 1 task x 3 strategies (no_sampling, undersample, oversample).
        self.assertEqual(len(record_files), 3)
        self.assertTrue(any("samp-no_sampling" in f for f in record_files))
        self.assertTrue(any("samp-undersample" in f for f in record_files))
        self.assertTrue(any("samp-oversample" in f for f in record_files))

    def test_advanced_strategies_run_end_to_end_via_config(self):
        """ablation.sampling.strategies (added 2026-10-03) is the
        opt-in switch for adasyn/enn/adasyn_enn -- confirms all three
        run cleanly end-to-end (including their own per-hospital
        fallback paths, since this mock fixture's hospitals are small
        enough to exercise them) through the real orchestrator, not
        just the isolated imbalance_sampling.py unit tests."""
        self.ablation_cfg["sampling"]["strategies"] = ["undersample", "oversample", "adasyn", "enn", "adasyn_enn"]
        self.ablation_cfg["sampling"]["candidate_adasyn_ratios"] = [0.5, 1.0]
        summary, results_writer = self._run_sampling(run_id="run_sampling_advanced")
        self.assertEqual(summary["n_combos"], 1)

        record_files = sorted(p.name for p in results_writer.records_dir.glob("*.json"))
        # 1 combo x 1 classifier x 1 task x 6 strategies.
        self.assertEqual(len(record_files), 6)
        for strat in ["no_sampling", "undersample", "oversample", "adasyn", "enn", "adasyn_enn"]:
            self.assertTrue(any(f"samp-{strat}" in f for f in record_files), f"missing a record for {strat}")

        sampling_csv = results_writer.run_dir / "sampling_results.csv"
        with open(sampling_csv) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 6)
        enn_row = next(r for r in rows if r["config.sampling_strategy"] == "enn")
        self.assertEqual(enn_row["config.sampling_ratio"], "")  # ratio genuinely has no meaning for enn

    def test_unknown_strategy_name_in_config_raises_clearly(self):
        self.ablation_cfg["sampling"]["strategies"] = ["undersample", "smote"]  # "smote" was never added
        with self.assertRaises(ValueError):
            self._run_sampling(run_id="run_sampling_bad_strategy")

    def test_results_routed_to_sampling_results_csv_not_master_results(self):
        _summary, results_writer = self._run_sampling(run_id="run_sampling_routing")
        sampling_csv = results_writer.run_dir / "sampling_results.csv"
        self.assertTrue(sampling_csv.exists())
        self.assertFalse(results_writer.master_csv_path.exists())  # nothing else written this run
        with open(sampling_csv) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        strategies = {r["config.sampling_strategy"] for r in rows}
        self.assertEqual(strategies, {"no_sampling", "undersample", "oversample"})

    def test_embeddings_are_cached_and_reused_on_a_second_run(self):
        """Direct regression target for the caching claim: a second
        run_sampling_comparison() call must NOT retrain the encoder --
        confirmed via combo_summaries reporting from_cache=True, not by
        timing (unreliable in CI)."""
        summary1, _writer1 = self._run_sampling(run_id="run_sampling_cache_1")
        self.assertFalse(summary1["combos"]["ts-lstm_tab-mlp_fusion-beforeca"]["from_cache"])

        summary2, _writer2 = self._run_sampling(run_id="run_sampling_cache_2")
        self.assertTrue(summary2["combos"]["ts-lstm_tab-mlp_fusion-beforeca"]["from_cache"])

    def test_undersample_and_oversample_records_carry_a_selected_ratio(self):
        _summary, results_writer = self._run_sampling(run_id="run_sampling_ratio_recorded")
        for path in results_writer.records_dir.glob("*samp-undersample*.json"):
            with open(path) as f:
                data = json.load(f)
            self.assertIn(data["config"]["sampling_ratio"], [0.5, 1.0])
            self.assertIn("ratio_search_trials", data["training_info"])
            self.assertEqual(len(data["training_info"]["ratio_search_trials"]), 2)  # 2 candidate ratios configured

    def test_no_sampling_reproduction_check_matches_a_real_prior_embedding_ablation_run(self):
        """Runs the ORIGINAL embedding_ablation step first (writing a
        real master_results.csv row), then feeds those rows in as
        `reference_master_rows` -- the regenerated no_sampling result
        must match the original within tolerance, proving the cache/
        regeneration path reproduces the SAME embeddings the original
        ablation used, not silently different ones."""
        original_writer = ResultsWriter(self.output_dir / "ablation", run_id="run_original_embedding_ablation")
        run_embedding_ablation(
            self.cfg, self.ablation_cfg, self.phase3_result, self.phase4_result, self.phase5_result,
            self.partition_result, self.split_manifest, original_writer,
        )
        original_rows = original_writer.build_summary_table()

        # A fresh cache dir -- forces genuine regeneration, not a reuse
        # of whatever the original run happened to leave behind (the
        # original embedding_ablation run never writes to the cache at
        # all, so this is guaranteed regardless, but made explicit).
        fresh_cache_dir = str(self.output_dir / "ablation" / "embeddings_cache_repro_check")
        self.ablation_cfg["sampling"]["embeddings_cache_dir"] = fresh_cache_dir

        summary, results_writer = self._run_sampling(
            run_id="run_sampling_repro_check", reference_master_rows=original_rows,
        )
        self.assertGreater(summary["n_reproduction_checked"], 0)
        self.assertEqual(summary["n_reproduction_mismatches"], 0)
        for check in summary["reproduction_checks"]:
            self.assertTrue(check["matched_reference"])

    def test_stops_before_sampling_when_reproduction_check_mismatches(self):
        """Direct regression target for the gating requirement
        (confirmed with the user 2026-09-23): a deliberately-wrong
        reference row must stop the run BEFORE any undersample/
        oversample record is written -- only the no_sampling record
        (needed for diagnosis) should exist."""
        original_writer = ResultsWriter(self.output_dir / "ablation", run_id="run_original_for_mismatch_test")
        run_embedding_ablation(
            self.cfg, self.ablation_cfg, self.phase3_result, self.phase4_result, self.phase5_result,
            self.partition_result, self.split_manifest, original_writer,
        )
        original_rows = original_writer.build_summary_table()
        for row in original_rows:
            row["metrics"]["test"]["auroc"] = 0.01  # deliberately wrong

        summary, results_writer = self._run_sampling(
            run_id="run_sampling_mismatch_stop", reference_master_rows=original_rows,
        )
        self.assertTrue(summary["stopped_before_sampling"])
        self.assertEqual(summary["stopped_at_combo"], "ts-lstm_tab-mlp_fusion-beforeca")
        self.assertGreaterEqual(summary["n_reproduction_mismatches"], 1)

        record_files = sorted(p.name for p in results_writer.records_dir.glob("*.json"))
        self.assertTrue(any("samp-no_sampling" in f for f in record_files))
        self.assertFalse(any("samp-undersample" in f for f in record_files))
        self.assertFalse(any("samp-oversample" in f for f in record_files))

    def test_combo_after_a_mismatch_is_never_processed(self):
        """A second regression target: with TWO combos where the FIRST
        one's reference mismatches, the SECOND combo must never even
        have its (expensive) embeddings regenerated -- proves the stop
        is a genuine early-exit, not just a skip of the sampling step
        for the bad combo alone."""
        self.ablation_cfg["subset"]["combos"] = [
            {"ts_model": "lstm", "tab_model": "mlp", "fusion_mode": "concatenation"},
            {"ts_model": "rnn", "tab_model": "mlp", "fusion_mode": "concatenation"},
        ]
        self.ablation_cfg.setdefault("ts_models", ["lstm", "rnn"])

        original_writer = ResultsWriter(self.output_dir / "ablation", run_id="run_original_for_two_combo_test")
        run_embedding_ablation(
            self.cfg, self.ablation_cfg, self.phase3_result, self.phase4_result, self.phase5_result,
            self.partition_result, self.split_manifest, original_writer,
        )
        original_rows = original_writer.build_summary_table()
        for row in original_rows:
            if row["experiment_id"].startswith("ts-lstm_"):
                row["metrics"]["test"]["auroc"] = 0.01  # only the FIRST combo's reference is wrong

        summary, results_writer = self._run_sampling(
            run_id="run_sampling_two_combo_stop", reference_master_rows=original_rows,
        )
        self.assertTrue(summary["stopped_before_sampling"])
        self.assertEqual(summary["stopped_at_combo"], "ts-lstm_tab-mlp_fusion-beforeca")
        # The rnn combo must never appear at all -- not attempted.
        self.assertNotIn("ts-rnn_tab-mlp_fusion-beforeca", summary["combos"])
        # NOTE: as of 2026-09-29, run_embedding_ablation() and
        # run_sampling_comparison() share ONE on-disk embedding cache
        # (see get_or_build_combo_embeddings() in
        # embedding_ablation_runner.py) -- so the rnn combo's .npz DOES
        # exist here, cached by the run_embedding_ablation() call above
        # (which trains both combos to fabricate this test's fake
        # "original_rows" reference), not by run_sampling_comparison.
        # The real proof that run_sampling_comparison itself never
        # attempted the rnn combo is that it never wrote a result
        # record for it (assertNotIn above already covers the in-memory
        # side; this covers what actually landed on disk from THIS
        # run's own results_writer).
        cache_dir = Path(self.ablation_cfg["sampling"]["embeddings_cache_dir"])
        self.assertTrue((cache_dir / "ts-rnn_tab-mlp_fusion-beforeca.npz").exists())
        record_files = [p.name for p in results_writer.records_dir.glob("*.json")]
        self.assertFalse(any("ts-rnn" in f for f in record_files))

    def test_reproduction_check_compares_all_three_splits_not_test_alone(self):
        """Direct regression target for the multi-split rigor
        (confirmed with the user 2026-09-23): a reference row with a
        CORRECT test AUROC but a deliberately WRONG train AUROC must
        still be caught as an overall mismatch -- a test-only check
        would have missed this."""
        original_writer = ResultsWriter(self.output_dir / "ablation", run_id="run_original_for_split_check_test")
        run_embedding_ablation(
            self.cfg, self.ablation_cfg, self.phase3_result, self.phase4_result, self.phase5_result,
            self.partition_result, self.split_manifest, original_writer,
        )
        original_rows = original_writer.build_summary_table()
        for row in original_rows:
            row["metrics"]["train"]["auroc"] = 0.01  # only TRAIN is corrupted; test/val stay correct

        summary, _results_writer = self._run_sampling(
            run_id="run_sampling_train_only_mismatch", reference_master_rows=original_rows,
        )
        self.assertTrue(summary["stopped_before_sampling"])
        detail = summary["reproduction_checks"][0]["detail"]
        self.assertFalse(detail["train"]["matched"])
        self.assertTrue(detail["val"]["matched"])
        self.assertTrue(detail["test"]["matched"])

    def test_pca_combos_are_never_run_even_if_configured(self):
        """PCA stays deferred (confirmed with the user) -- must be
        excluded from the sampling comparison regardless of what
        ablation_cfg configures, not just by convention."""
        self.ablation_cfg["subset"]["combos"] = [
            {"ts_model": "lstm", "tab_model": "mlp", "fusion_mode": "concatenation"},
            {"ts_model": "lstm", "tab_model": "pca", "fusion_mode": "concatenation"},
        ]
        summary, _writer = self._run_sampling(run_id="run_sampling_pca_excluded")
        self.assertEqual(summary["n_combos"], 1)
        self.assertNotIn("ts-lstm_tab-pca_fusion-beforeca", summary["combos"])

    def test_results_never_silently_overwritten_across_two_runs_same_id(self):
        results_writer = ResultsWriter(self.output_dir / "ablation", run_id="run_sampling_dup_check")
        run_sampling_comparison(
            self.cfg, self.ablation_cfg, self.phase3_result, self.phase4_result, self.phase5_result,
            self.partition_result, self.split_manifest, results_writer,
        )
        with self.assertRaises(FileExistsError):
            run_sampling_comparison(
                self.cfg, self.ablation_cfg, self.phase3_result, self.phase4_result, self.phase5_result,
                self.partition_result, self.split_manifest, results_writer,
            )


if __name__ == "__main__":
    unittest.main()
