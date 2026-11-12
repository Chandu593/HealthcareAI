"""Integration tests for split-aware Phase 13 baselines
(run_phase13_baselines's `train_patient_ids` parameter, added
2026-09-11).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

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
    run_phase8_patient_graph_construction,
    run_phase12_fedavg,
    run_phase13_baselines,
)
from src.data.splitting.patient_split import build_split_manifest  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestPhase13SplitAware(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        self.checkpoint_dir = Path(self.tmpdir.name) / "checkpoints"
        generate(str(self.data_root), n_hospitals=4, n_patients_per_hospital=25, seed=13)

        self._old_env = os.environ.get("EICU_DATA_ROOT")
        os.environ["EICU_DATA_ROOT"] = str(self.data_root)

        self.cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        self.cfg.raw["run"]["output_dir"] = str(self.output_dir)
        self.cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")
        self.cfg.raw["run"]["checkpoint_dir"] = str(self.checkpoint_dir)
        self.cfg.raw["cohort"]["patient_subset"] = 15
        self.cfg.raw.setdefault("graph", {})
        self.cfg.raw["graph"]["method"] = "knn"
        self.cfg.raw["graph"]["k"] = 3
        self.cfg.raw.setdefault("gnn", {})
        self.cfg.raw["gnn"]["type"] = "graphsage"
        self.cfg.raw["gnn"]["hidden_dim"] = 16
        self.cfg.raw["gnn"]["layers"] = 2
        self.cfg.raw["gnn"]["dropout"] = 0.0
        self.cfg.raw.setdefault("multitask", {})
        self.cfg.raw["multitask"]["sepsis_cohort"] = "broad"
        self.cfg.raw["multitask"]["head_hidden_dim"] = 8
        self.cfg.raw["multitask"]["head_dropout"] = 0.0
        self.cfg.raw.setdefault("training", {})
        self.cfg.raw["training"]["local_epochs"] = 2
        self.cfg.raw["training"]["learning_rate"] = 0.01
        self.cfg.raw["training"]["weight_decay"] = 0.0
        self.cfg.raw["training"]["optimizer"] = "adam"
        self.cfg.raw["training"]["reconstruction_weight"] = 0.5
        self.cfg.raw["training"]["max_hospitals"] = None
        self.cfg.raw.setdefault("federated", {})
        self.cfg.raw["federated"]["n_rounds"] = 2
        self.cfg.raw["federated"]["reuse_existing_local_checkpoints_for_round_1"] = False

        phase1_summary = run_phase1_ingestion(self.cfg)
        self.partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        self.phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, self.partition_result, patient_df)
        self.phase4_result = run_phase4_timeseries_modality(self.cfg, self.phase3_result, self.partition_result)
        self.phase5_result = run_phase5_tabular_modality(self.cfg, phase1_summary, self.phase3_result, self.partition_result)
        phase7_result = run_phase7_multimodal_fusion(self.cfg, self.phase3_result, self.phase4_result, self.phase5_result)
        self.phase8_result = run_phase8_patient_graph_construction(self.cfg, phase7_result, self.partition_result)

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

    def test_writes_to_a_separate_checkpoint_directory(self):
        result = run_phase13_baselines(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
            train_patient_ids=self.train_patient_ids,
        )
        self.assertIn("splitaware", Path(result["checkpoint_dir"]).name)

    def test_centralized_sequential_predictions_cover_every_patient(self):
        result = run_phase13_baselines(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
            train_patient_ids=self.train_patient_ids,
        )
        preds_by_hid = result["centralized_sequential"]["predictions_by_hospital"]
        self.assertTrue(preds_by_hid)
        found_non_train = False
        for hid, preds in preds_by_hid.items():
            for pid, is_train in zip(preds["patient_ids"], preds["is_train_row"]):
                expected = "train" if pid in self.train_patient_ids else "not_train"
                self.assertEqual(is_train, expected)
                if expected == "not_train":
                    found_non_train = True
        self.assertTrue(found_non_train)

    def test_federated_reference_points_to_split_aware_fedavg_manifest(self):
        # Run split-aware Phase 12 first so a split-aware fedavg
        # manifest actually exists to be referenced.
        run_phase12_fedavg(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
            train_patient_ids=self.train_patient_ids,
        )
        result = run_phase13_baselines(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
            train_patient_ids=self.train_patient_ids,
        )
        self.assertIsNotNone(result["federated_reference_model_path"])
        self.assertIn("splitaware", result["federated_reference_model_path"])

    def test_federated_reference_does_NOT_fall_back_to_the_original_full_cohort_manifest(self):
        """If only the ORIGINAL (non-split-aware) Phase 12 has run, a
        split-aware Phase 13 must NOT reference it -- that would compare
        a split-aware local/centralized baseline against a federated
        result that saw the full cohort, which is not apples-to-apples."""
        run_phase12_fedavg(self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result)
        result = run_phase13_baselines(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
            train_patient_ids=self.train_patient_ids,
        )
        self.assertIsNone(result["federated_reference_model_path"])


if __name__ == "__main__":
    unittest.main()
