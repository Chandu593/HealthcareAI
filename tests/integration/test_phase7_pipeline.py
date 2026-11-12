"""End-to-end Phase 7 (multimodal fusion) integration test against
synthetic data -- runs the actual Phase 1 -> 2 -> 3 -> 4 -> 5 -> 7
pipeline path, not just the isolated fusion helper functions (those
are covered in tests/unit/test_multimodal_fusion.py, including the
critical misaligned-storage-order regression test).
"""

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
    run_phase5_tabular_modality,
    run_phase7_multimodal_fusion,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestPhase7Integration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        generate(str(self.data_root), n_hospitals=3, n_patients_per_hospital=25, seed=3)

        self._old_env = os.environ.get("EICU_DATA_ROOT")
        os.environ["EICU_DATA_ROOT"] = str(self.data_root)

        self.cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        self.cfg.raw["run"]["output_dir"] = str(self.output_dir)
        self.cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")
        self.cfg.raw["run"]["cache_dir"] = str(self.output_dir / "cache")
        self.cfg.raw["cohort"]["patient_subset"] = 15

    def tearDown(self):
        close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def _run_through_phase5(self):
        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)
        phase4_result = run_phase4_timeseries_modality(self.cfg, phase3_result, partition_result)
        phase5_result = run_phase5_tabular_modality(self.cfg, phase1_summary, phase3_result, partition_result)
        return phase3_result, phase4_result, phase5_result

    def test_phase7_produces_correctly_shaped_fused_embeddings(self):
        phase3_result, phase4_result, phase5_result = self._run_through_phase5()
        if not phase4_result["embeddings_path"] or not phase5_result["embeddings_path"]:
            self.skipTest("torch not available in this environment -- Phase 4/5 embeddings not produced")

        phase7_result = run_phase7_multimodal_fusion(self.cfg, phase3_result, phase4_result, phase5_result)

        fused_path = Path(phase7_result["fused_embeddings_path"])
        self.assertTrue(fused_path.exists())
        data = np.load(fused_path)

        n_patients = 3 * 15
        lstm_data = np.load(phase4_result["embeddings_path"])
        ae_data = np.load(phase5_result["embeddings_path"])
        expected_dim = lstm_data["embeddings"].shape[1] + ae_data["embeddings"].shape[1]

        self.assertEqual(data["embeddings"].shape, (n_patients, expected_dim))
        self.assertEqual(len(data["patient_ids"]), n_patients)
        self.assertEqual(phase7_result["n_patients"], n_patients)
        self.assertEqual(phase7_result["fused_dim"], expected_dim)

    def test_fused_patient_ids_match_cohort_exactly(self):
        phase3_result, phase4_result, phase5_result = self._run_through_phase5()
        if not phase4_result["embeddings_path"] or not phase5_result["embeddings_path"]:
            self.skipTest("torch not available in this environment -- Phase 4/5 embeddings not produced")

        phase7_result = run_phase7_multimodal_fusion(self.cfg, phase3_result, phase4_result, phase5_result)
        data = np.load(Path(phase7_result["fused_embeddings_path"]))

        cohort_ids = phase3_result["cohort_patients"]["patientunitstayid"].tolist()
        self.assertEqual(data["patient_ids"].tolist(), cohort_ids)

    def test_fused_row_equals_source_rows_for_the_same_patient(self):
        """Direct end-to-end correctness check, not just shape: for a
        real patient, the fused row must equal that SAME patient's own
        LSTM row concatenated with their own autoencoder row -- pulled
        independently from each source file by patient ID, not by
        matching row position."""
        phase3_result, phase4_result, phase5_result = self._run_through_phase5()
        if not phase4_result["embeddings_path"] or not phase5_result["embeddings_path"]:
            self.skipTest("torch not available in this environment -- Phase 4/5 embeddings not produced")

        phase7_result = run_phase7_multimodal_fusion(self.cfg, phase3_result, phase4_result, phase5_result)
        fused_data = np.load(Path(phase7_result["fused_embeddings_path"]))
        lstm_data = np.load(phase4_result["embeddings_path"])
        ae_data = np.load(phase5_result["embeddings_path"])

        sample_patient = int(fused_data["patient_ids"][7])  # arbitrary mid-list patient

        fused_row = fused_data["embeddings"][list(fused_data["patient_ids"]).index(sample_patient)]
        lstm_row = lstm_data["embeddings"][list(lstm_data["patient_ids"]).index(sample_patient)]
        ae_row = ae_data["embeddings"][list(ae_data["patient_ids"]).index(sample_patient)]

        np.testing.assert_array_almost_equal(fused_row, np.concatenate([lstm_row, ae_row]))

    def test_raises_clearly_when_an_embeddings_path_is_missing(self):
        phase3_result, phase4_result, phase5_result = self._run_through_phase5()
        broken_phase5_result = dict(phase5_result)
        broken_phase5_result["embeddings_path"] = None

        with self.assertRaises(RuntimeError) as ctx:
            run_phase7_multimodal_fusion(self.cfg, phase3_result, phase4_result, broken_phase5_result)
        self.assertIn("Phase 5", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
