"""
End-to-end Phase 1 integration test.

Generates a tiny synthetic eICU dataset, points EICU_DATA_ROOT at it, and
runs the actual discovery -> validation -> data_report path used by
run_pipeline.py. This is the "verified via synthetic schema-compatible
data" evidence referenced in IMPLEMENTATION_TRACKER.md -- it does NOT
constitute verification against real eICU data (Section 43).
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_mock_eicu import generate  # noqa: E402
from run_pipeline import run_phase1_ingestion  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestPhase1Integration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        generate(str(self.data_root), n_hospitals=3, n_patients_per_hospital=15, seed=7)

        self._old_env = os.environ.get("EICU_DATA_ROOT")
        os.environ["EICU_DATA_ROOT"] = str(self.data_root)

    def tearDown(self):
        close_logging()  # must close file handlers before deleting the tempdir they write into (Windows) -- see logging_setup.close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def test_phase1_end_to_end_on_mock_data(self):
        cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        cfg.raw["run"]["output_dir"] = str(self.output_dir)
        cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")

        summary = run_phase1_ingestion(cfg)

        # Core tables must be found (treatment + medication_imputed* now
        # all confirmed-required per the user, 2026-08-08 -- mock
        # generator produces all 13)
        for expected in ["patient", "hospital", "apachePatientResult", "apachePredVar",
                          "diagnosis", "lab", "medication", "admissionDrug",
                          "infusionDrug", "microLab", "treatment",
                          "medication_imputed", "medication_imputed1"]:
            self.assertIn(expected, summary["found_tables"], f"{expected} not discovered")

        # No tables remain in the unconfirmed-availability bucket
        self.assertEqual(summary["missing_unconfirmed_tables"], [])

        # Per-table JSON reports actually written to disk
        for table_name, report_path in summary["table_reports"].items():
            self.assertTrue(Path(report_path).exists())
            with open(report_path) as f:
                report = json.load(f)
            self.assertGreater(report["row_count"], 0)
            self.assertTrue(report["required_columns_present"])

        # patient table row count matches what the generator produced (3*15)
        with open(summary["table_reports"]["patient"]) as f:
            patient_report = json.load(f)
        self.assertEqual(patient_report["row_count"], 45)
        self.assertEqual(patient_report["unique_hospital_count"], 3)

        # Leakage-risk columns on apachePatientResult must be flagged
        with open(summary["table_reports"]["apachePatientResult"]) as f:
            apr_report = json.load(f)
        self.assertIn("actualhospitalmortality", apr_report["leakage_risk_columns_present"])

        # lab.csv must have used the chunked strategy (large_table=True in registry)
        with open(summary["table_reports"]["lab"]) as f:
            lab_report = json.load(f)
        self.assertEqual(lab_report["read_strategy"], "chunked")

    def test_phase1_fails_loudly_without_patient_table(self):
        broken_root = Path(self.tmpdir.name) / "broken_eicu"
        broken_root.mkdir()
        shutil.copy(self.data_root / "hospital.csv", broken_root / "hospital.csv")
        os.environ["EICU_DATA_ROOT"] = str(broken_root)

        cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        cfg.raw["run"]["output_dir"] = str(self.output_dir)
        cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")

        with self.assertRaises(RuntimeError):
            run_phase1_ingestion(cfg)


if __name__ == "__main__":
    unittest.main()
