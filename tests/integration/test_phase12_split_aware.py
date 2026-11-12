"""Integration tests for split-aware Phase 12 (run_phase12_fedavg's
`train_patient_ids` parameter, added 2026-09-11).
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
    run_phase11_local_training,
    run_phase12_fedavg,
)
from src.data.splitting.patient_split import build_split_manifest  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestPhase12SplitAware(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        self.checkpoint_dir = Path(self.tmpdir.name) / "checkpoints"
        generate(str(self.data_root), n_hospitals=4, n_patients_per_hospital=25, seed=9)

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
        self.cfg.raw["federated"]["reuse_existing_local_checkpoints_for_round_1"] = True

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
        result = run_phase12_fedavg(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
            train_patient_ids=self.train_patient_ids,
        )
        self.assertIn("splitaware", Path(result["checkpoint_dir"]).name)

    def test_split_aware_round1_never_reuses_the_original_full_cohort_checkpoints(self):
        """The central Phase-12-specific leakage guard: even with reuse
        enabled, a split-aware run must never pick up the ORIGINAL
        (full-cohort, potentially leaky) Phase 11 checkpoints -- only a
        prior SPLIT-AWARE Phase 11 run is a legitimate reuse source."""
        # Populate the ORIGINAL (non-split-aware) Phase 11 checkpoint
        # directory with real checkpoints.
        run_phase11_local_training(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
        )
        original_dir = self.checkpoint_dir / "phase11_local"
        self.assertTrue(any(original_dir.glob("hospital_*_local_weights.pt")))

        # Split-aware Phase 12 with reuse=True must NOT find/use those
        # -- it looks under phase11_local_splitaware, which doesn't
        # exist yet, so round 1 falls back to fresh training (logged,
        # not a crash).
        result = run_phase12_fedavg(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
            train_patient_ids=self.train_patient_ids,
        )
        self.assertEqual(result["round_summaries"][0]["source"], "fresh_local_training")

    def test_split_aware_round1_DOES_reuse_a_prior_split_aware_phase11_run(self):
        run_phase11_local_training(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
            train_patient_ids=self.train_patient_ids,
        )
        split_aware_dir = self.checkpoint_dir / "phase11_local_splitaware"
        self.assertTrue(any(split_aware_dir.glob("hospital_*_local_weights.pt")))

        result = run_phase12_fedavg(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
            train_patient_ids=self.train_patient_ids,
        )
        self.assertEqual(result["round_summaries"][0]["source"], "reused_phase11_checkpoints")

    def test_end_to_end_split_aware_fedavg_produces_a_final_model(self):
        result = run_phase12_fedavg(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
            train_patient_ids=self.train_patient_ids,
        )
        self.assertEqual(result["n_rounds"], 2)
        self.assertTrue(Path(result["final_global_model_path"]).exists())
        final_state = torch.load(result["final_global_model_path"], weights_only=True)
        self.assertTrue(all(torch.isfinite(v).all() for v in final_state.values()))


if __name__ == "__main__":
    unittest.main()
