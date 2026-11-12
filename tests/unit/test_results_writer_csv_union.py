import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.ablation.results_writer import ExperimentRecord, ResultsWriter  # noqa: E402


def rec(i, extra=None, kind="sampling_comparison"):
    cfg = {"classifier": "xgboost", "sampling_strategy": "oversample"}
    cfg.update(extra or {})
    m = {s: {"auroc": 0.7 + i / 100, "f1": 0.3} for s in ("train", "val", "test")}
    return ExperimentRecord(f"exp_{i}", kind, cfg, "mortality", 42, m, {"n": i}, 1.0, f"2026-10-0{i}T00:00:00")


class TestCsvUnionHeader(unittest.TestCase):
    def test_later_row_with_extra_keys_never_shifts_columns(self):
        with tempfile.TemporaryDirectory() as d:
            w = ResultsWriter(d, "r")
            w.write(rec(1))
            w.write(rec(2, {"classifier_params": "{'a': 1}", "sampling_ratio": 0.5}))      # NEW keys appear only on the 2nd row
            rows = list(csv.DictReader(open(w.run_dir / "sampling_results.csv")))
            self.assertEqual(len(rows), 2)
            self.assertIn("config.classifier_params", rows[0].keys())
            self.assertEqual(rows[0]["config.classifier_params"], "")                       # row 1 has none -> blank, not shifted
            self.assertEqual(rows[1]["config.sampling_ratio"], "0.5")
            self.assertEqual(rows[1]["config.sampling_strategy"], "oversample")
            self.assertAlmostEqual(float(rows[1]["test.auroc"]), 0.72)                       # a metric still sits under its own header

    def test_rebuild_heals_a_previously_corrupted_csv(self):
        with tempfile.TemporaryDirectory() as d:
            w = ResultsWriter(d, "r"); w.write(rec(1)); w.write(rec(2, {"sampling_ratio": 0.5}))
            (w.run_dir / "sampling_results.csv").write_text("garbage,header\n1,2\n")
            healed = w.rebuild_dedicated_csvs()
            self.assertEqual([p.name for p in healed], ["sampling_results.csv"])
            self.assertEqual(len(list(csv.DictReader(open(w.run_dir / "sampling_results.csv")))), 2)

    def test_rows_stay_in_their_own_kind_file(self):
        with tempfile.TemporaryDirectory() as d:
            w = ResultsWriter(d, "r"); w.write(rec(1, kind="embedding_ablation")); w.write(rec(2))
            self.assertEqual(len(list(csv.DictReader(open(w.master_csv_path)))), 1)
            self.assertEqual(len(list(csv.DictReader(open(w.run_dir / "sampling_results.csv")))), 1)


if __name__ == "__main__":
    unittest.main()
