import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.ablation.results_writer import ExperimentRecord, ResultsWriter


def _make_record(experiment_id="exp1", test_auroc=0.8):
    return ExperimentRecord(
        experiment_id=experiment_id, experiment_kind="embedding_ablation",
        config={"ts_model": "lstm", "tab_model": "mlp", "fusion_mode": "concatenation", "classifier": "xgboost"},
        task="mortality", split_manifest_seed=42,
        metrics={"train": {"auroc": 0.9, "n": 100}, "val": {"auroc": 0.85, "n": 20}, "test": {"auroc": test_auroc, "n": 20}},
        training_info={"n_hospitals_trained": 4, "epochs": 5},
        runtime_seconds=1.23, timestamp="2026-09-10T00:00:00Z",
    )


class TestResultsWriter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_write_creates_json_and_appends_csv(self):
        writer = ResultsWriter(self.tmpdir.name, run_id="run_test")
        path = writer.write(_make_record())
        self.assertTrue(path.exists())
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["experiment_id"], "exp1")

        with open(writer.master_csv_path) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["experiment_id"], "exp1")
        self.assertEqual(rows[0]["test.auroc"], "0.8")

    def test_refuses_to_silently_overwrite(self):
        writer = ResultsWriter(self.tmpdir.name, run_id="run_test")
        writer.write(_make_record("dup_exp"))
        with self.assertRaises(FileExistsError):
            writer.write(_make_record("dup_exp"))

    def test_overwrite_true_allows_replacing(self):
        writer = ResultsWriter(self.tmpdir.name, run_id="run_test", overwrite=True)
        writer.write(_make_record("dup_exp", test_auroc=0.5))
        writer.write(_make_record("dup_exp", test_auroc=0.99))
        with open(writer.records_dir / "dup_exp.json") as f:
            data = json.load(f)
        self.assertEqual(data["metrics"]["test"]["auroc"], 0.99)

    def test_different_run_ids_never_collide(self):
        writer1 = ResultsWriter(self.tmpdir.name, run_id="run_a")
        writer2 = ResultsWriter(self.tmpdir.name, run_id="run_b")
        writer1.write(_make_record("same_id"))
        writer2.write(_make_record("same_id"))  # must NOT raise -- different run directories
        self.assertTrue((writer1.records_dir / "same_id.json").exists())
        self.assertTrue((writer2.records_dir / "same_id.json").exists())

    def test_summary_table_sorted_by_test_auroc_descending(self):
        writer = ResultsWriter(self.tmpdir.name, run_id="run_test")
        writer.write(_make_record("low", test_auroc=0.6))
        writer.write(_make_record("high", test_auroc=0.95))
        writer.write(_make_record("mid", test_auroc=0.75))
        rows = writer.build_summary_table()
        ids_in_order = [r["experiment_id"] for r in rows]
        self.assertEqual(ids_in_order, ["high", "mid", "low"])

    def test_summary_table_handles_missing_test_auroc_without_crashing(self):
        writer = ResultsWriter(self.tmpdir.name, run_id="run_test")
        record_missing = _make_record("no_auroc")
        record_missing.metrics["test"] = {"auroc": None, "n": 20}
        writer.write(record_missing)
        writer.write(_make_record("has_auroc", test_auroc=0.7))
        rows = writer.build_summary_table()
        # The None-auroc record must sort LAST, not crash the sort.
        self.assertEqual(rows[-1]["experiment_id"], "no_auroc")

    def test_save_summary_csv_produces_a_readable_file(self):
        writer = ResultsWriter(self.tmpdir.name, run_id="run_test")
        writer.write(_make_record("exp1", test_auroc=0.7))
        writer.write(_make_record("exp2", test_auroc=0.8))
        out_path = writer.save_summary_csv()
        self.assertTrue(out_path.exists())
        with open(out_path) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertIn("test.auroc", rows[0])

    def test_sampling_comparison_routes_to_its_own_dedicated_csv(self):
        """Added 2026-09-20 alongside graph_ablation's existing
        dedicated-file routing, for the identical reason (module
        docstring's 'BUG FOUND AND FIXED' section): sampling_comparison
        rows carry extra config keys (sampling_strategy, sampling_ratio)
        that embedding_ablation rows don't -- must never land in
        master_results.csv."""
        writer = ResultsWriter(self.tmpdir.name, run_id="run_test")
        embedding_record = _make_record("ts-lstm_tab-mlp_fusion-beforeca_clf-xgboost_task-mortality")
        sampling_record = ExperimentRecord(
            experiment_id="ts-lstm_tab-mlp_fusion-beforeca_clf-xgboost_task-mortality_samp-undersample",
            experiment_kind="sampling_comparison",
            config={
                "ts_model": "lstm", "tab_model": "mlp", "fusion_mode": "concatenation", "classifier": "xgboost",
                "sampling_strategy": "undersample", "sampling_ratio": 0.75,
            },
            task="mortality", split_manifest_seed=42,
            metrics={"train": {"auroc": 0.9, "n": 60}, "val": {"auroc": 0.8, "n": 20}, "test": {"auroc": 0.78, "n": 20}},
            training_info={"ratio_search_trials": [{"ratio": 0.5}, {"ratio": 0.75}, {"ratio": 1.0}]},
            runtime_seconds=2.0, timestamp="2026-09-20T00:00:00Z",
        )
        writer.write(embedding_record)
        writer.write(sampling_record)

        with open(writer.master_csv_path) as f:
            master_rows = list(csv.DictReader(f))
        self.assertEqual(len(master_rows), 1)
        self.assertNotIn("config.sampling_strategy", master_rows[0])

        sampling_csv_path = writer.run_dir / "sampling_results.csv"
        self.assertTrue(sampling_csv_path.exists())
        with open(sampling_csv_path) as f:
            sampling_rows = list(csv.DictReader(f))
        self.assertEqual(len(sampling_rows), 1)
        self.assertEqual(sampling_rows[0]["config.sampling_strategy"], "undersample")

    def test_new_run_directories_are_distinguishable_by_kind_in_filenames(self):
        """Before-CA / After-CA / Graph experiments must be
        distinguishable purely from filenames (Section 3 requirement)."""
        writer = ResultsWriter(self.tmpdir.name, run_id="run_test")
        writer.write(_make_record("ts-lstm_tab-mlp_fusion-beforeca_clf-xgboost_task-mortality"))
        writer.write(_make_record("ts-lstm_tab-mlp_fusion-afterca_clf-xgboost_task-mortality"))
        writer.write(_make_record("graph-embedding_clf-xgboost_task-mortality"))
        filenames = sorted(p.name for p in writer.records_dir.glob("*.json"))
        self.assertTrue(any("beforeca" in f for f in filenames))
        self.assertTrue(any("afterca" in f for f in filenames))
        self.assertTrue(any("graph-embedding" in f for f in filenames))


if __name__ == "__main__":
    unittest.main()
