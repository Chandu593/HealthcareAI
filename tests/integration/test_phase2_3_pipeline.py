"""End-to-end Phase 2 + Phase 3 integration test against synthetic data."""

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
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestPhase2And3Integration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        generate(str(self.data_root), n_hospitals=3, n_patients_per_hospital=25, seed=99)

        self._old_env = os.environ.get("EICU_DATA_ROOT")
        os.environ["EICU_DATA_ROOT"] = str(self.data_root)

        self.cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        self.cfg.raw["run"]["output_dir"] = str(self.output_dir)
        self.cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")
        self.cfg.raw["cohort"]["patient_subset"] = 15  # smaller than hospital 1's 25, exercises normal path

    def tearDown(self):
        close_logging()  # must close file handlers before deleting the tempdir they write into (Windows) -- see logging_setup.close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def test_phase2_partitions_three_hospitals_with_no_cross_contamination(self):
        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, hospital_df = run_phase2_hospital_partition(self.cfg, phase1_summary)

        self.assertEqual(sorted(partition_result.eligible_hospitals), [1, 2, 3])
        seen_ids = set()
        for hid, p in partition_result.partitions.items():
            self.assertEqual(p.sampled_patient_count, 15)
            # No patient ID should appear under more than one hospital bucket
            self.assertTrue(seen_ids.isdisjoint(p.patient_ids))
            seen_ids.update(p.patient_ids)

        partition_report_path = self.output_dir / "cohort_reports" / "hospital_partition.json"
        self.assertTrue(partition_report_path.exists())
        with open(partition_report_path) as f:
            json.load(f)  # must be valid JSON (regression check for the int64 bug)

    def test_phase3_produces_internally_consistent_cohort_report(self):
        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)

        report_path = Path(phase3_result["cohort_report_path"])
        self.assertTrue(report_path.exists())
        with open(report_path) as f:
            report = json.load(f)

        # Structural invariants that must hold regardless of the random seed
        self.assertEqual(report["total_patients"], 45)  # 3 hospitals x 15 sampled
        self.assertEqual(
            report["cohort_overlap"]["both_strict_and_broad"] + report["cohort_overlap"]["strict_only"],
            report["sepsis_strict_positive"],
        )
        self.assertLessEqual(report["sepsis_strict_positive"], report["sepsis_broad_positive"])
        self.assertEqual(
            report["mortality_positive"] + report["mortality_negative"], report["eligible_patients"]
        )
        self.assertEqual(len(report["hospital_distribution"]), 3)
        self.assertIn("N/A", report["downstream_model_performance"])

        sepsis_df = phase3_result["sepsis_df"]
        # Every strict-positive patient must also be broad-positive (subset invariant)
        strict_positive_ids = set(sepsis_df.loc[sepsis_df["sepsis_strict"], "patientunitstayid"])
        broad_positive_ids = set(sepsis_df.loc[sepsis_df["sepsis_broad"], "patientunitstayid"])
        self.assertTrue(strict_positive_ids.issubset(broad_positive_ids))


if __name__ == "__main__":
    unittest.main()
