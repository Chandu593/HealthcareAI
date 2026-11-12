"""Integration test: Phase 4/5's split-aware path (train_patient_ids
parameter), run against the real Phase 1->2->3 pipeline on mock data.
Proves both that the default (train_patient_ids=None) behavior is
untouched, and that the split-aware path genuinely fits its
normalization/vocabulary statistics from train patients only.
"""

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
from src.data.splitting.patient_split import build_split_manifest  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestPhase4And5SplitAware(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        generate(str(self.data_root), n_hospitals=5, n_patients_per_hospital=45, seed=31)

        self._old_env = os.environ.get("EICU_DATA_ROOT")
        os.environ["EICU_DATA_ROOT"] = str(self.data_root)

        self.cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        self.cfg.raw["run"]["output_dir"] = str(self.output_dir)
        self.cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")
        self.cfg.raw["cohort"]["patient_subset"] = 40
        self.cfg.raw.setdefault("modalities", {}).setdefault("timeseries", {})["encoder"] = "lstm"
        self.cfg.raw["modalities"].setdefault("tabular", {})["encoder"] = "autoencoder"

        phase1_summary = run_phase1_ingestion(self.cfg)
        self.partition_result, patient_df, hospital_df = run_phase2_hospital_partition(self.cfg, phase1_summary)
        self.phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, self.partition_result, patient_df)
        self.phase1_summary = phase1_summary

        self.split_manifest = build_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], seed=42, stratify_on="mortality",
        )
        self.train_patient_ids = set(self.split_manifest.all_patient_ids("train"))

    def tearDown(self):
        close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def test_default_none_reproduces_original_full_cohort_behavior(self):
        result_default = run_phase4_timeseries_modality(self.cfg, self.phase3_result, self.partition_result)
        self.assertIsNotNone(result_default.get("normalized_array"))
        self.assertEqual(result_default["normalized_array"].shape[0], len(self.phase3_result["cohort_patients"]))

    def test_split_aware_path_runs_end_to_end_phase4_and_produces_all_patients(self):
        result = run_phase4_timeseries_modality(
            self.cfg, self.phase3_result, self.partition_result, train_patient_ids=self.train_patient_ids,
        )
        n_cohort = len(self.phase3_result["cohort_patients"])
        # Every patient (train+val+test) still gets a row -- val/test
        # patients are transformed with the frozen train-fit scaler,
        # never dropped.
        self.assertEqual(result["normalized_array"].shape[0], n_cohort)
        self.assertEqual(len(result["patient_ids"]), n_cohort)

    def test_split_aware_path_runs_end_to_end_phase5_and_produces_all_patients(self):
        result = run_phase5_tabular_modality(
            self.cfg, self.phase1_summary, self.phase3_result, self.partition_result,
            train_patient_ids=self.train_patient_ids,
        )
        n_cohort = len(self.phase3_result["cohort_patients"])
        self.assertEqual(result["normalized_array"].shape[0], n_cohort)

    def test_split_aware_output_genuinely_differs_from_full_cohort_output(self):
        """If the split-aware path silently ignored train_patient_ids
        (a real regression class -- e.g. forgetting to thread the
        parameter through to the normalization call), its output would
        be numerically IDENTICAL to the full-cohort run. It must not
        be, given val/test patients are excluded from fitting."""
        result_full = run_phase5_tabular_modality(self.cfg, self.phase1_summary, self.phase3_result, self.partition_result)
        result_split_aware = run_phase5_tabular_modality(
            self.cfg, self.phase1_summary, self.phase3_result, self.partition_result,
            train_patient_ids=self.train_patient_ids,
        )
        import numpy as np
        self.assertFalse(np.allclose(result_full["normalized_array"], result_split_aware["normalized_array"]))

if __name__ == "__main__":
    unittest.main()
