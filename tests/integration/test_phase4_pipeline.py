"""End-to-end Phase 4 integration test against synthetic data."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_mock_eicu import generate  # noqa: E402
from run_pipeline import (  # noqa: E402
    run_phase1_ingestion,
    run_phase2_hospital_partition,
    run_phase3_cohort_construction,
    run_phase4_timeseries_modality,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestPhase4Integration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        generate(str(self.data_root), n_hospitals=3, n_patients_per_hospital=20, seed=5)

        self._old_env = os.environ.get("EICU_DATA_ROOT")
        os.environ["EICU_DATA_ROOT"] = str(self.data_root)

        self.cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        self.cfg.raw["run"]["output_dir"] = str(self.output_dir)
        self.cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")
        self.cfg.raw["cohort"]["patient_subset"] = 12

    def tearDown(self):
        close_logging()  # must close file handlers before deleting the tempdir they write into (Windows) -- see logging_setup.close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def test_phase4_produces_valid_sequence_tensor(self):
        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)
        phase4_result = run_phase4_timeseries_modality(self.cfg, phase3_result, partition_result)

        # Sequence tensor artifact must exist and be loadable
        seq_path = Path(phase4_result["sequences_path"])
        self.assertTrue(seq_path.exists())
        data = np.load(seq_path, allow_pickle=True)

        n_patients = 3 * 12  # 3 hospitals x 12 sampled
        self.assertEqual(data["array"].shape[0], n_patients)
        self.assertEqual(data["array"].shape[1], self.cfg.get("temporal.max_sequence_length"))
        self.assertEqual(len(data["feature_names"]), data["array"].shape[2])

        # No NaNs should survive normalization -- fully imputed
        self.assertFalse(np.any(np.isnan(data["array"])))

        # Sequence lengths must be within [1, max_bins]
        self.assertTrue(np.all(data["sequence_length"] >= 1))
        self.assertTrue(np.all(data["sequence_length"] <= self.cfg.get("temporal.max_sequence_length")))

        # Padding regions (beyond sequence_length) must be exactly zero
        max_bins = data["array"].shape[1]
        for i, sl in enumerate(data["sequence_length"]):
            if sl < max_bins:
                self.assertTrue(np.all(data["array"][i, sl:, :] == 0.0), f"patient index {i} padding not zeroed")

        # torch_available flag must accurately reflect this environment
        self.assertIn("torch_available", phase4_result)
        if not phase4_result["torch_available"]:
            self.assertIsNone(phase4_result["embeddings_path"])
        else:
            self.assertIsNotNone(phase4_result["embeddings_path"])
            self.assertTrue(Path(phase4_result["embeddings_path"]).exists())

    def test_phase4_feature_names_are_stable_and_prefixed(self):
        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)
        phase4_result = run_phase4_timeseries_modality(self.cfg, phase3_result, partition_result)

        feature_names = phase4_result["feature_names"]
        self.assertTrue(any(f.startswith("lab_") for f in feature_names))
        self.assertIn("medication_antibiotic_active", feature_names)
        self.assertIn("infusion_vasopressor_active", feature_names)


if __name__ == "__main__":
    unittest.main()
