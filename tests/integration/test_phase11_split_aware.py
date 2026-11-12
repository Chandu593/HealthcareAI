"""Integration tests for split-aware Phase 11 local training
(run_phase11_local_training/_run_local_training_round's `train_patient_ids`
parameter, added 2026-09-11). The centerpiece is a direct revert->fail
->restore->pass-style proof of the reconstruction-term leakage fix
described in _run_local_training_round()'s own docstring: without the
fix, a non-train (val/test) patient's raw tabular VALUES still
influence the trained model's weights (via the autoencoder's
unsupervised reconstruction loss, which was never masked); with the
fix, they must not.
"""

import copy
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
)
from src.data.splitting.patient_split import build_split_manifest  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestPhase11SplitAware(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        self.checkpoint_dir = Path(self.tmpdir.name) / "checkpoints"
        generate(str(self.data_root), n_hospitals=4, n_patients_per_hospital=25, seed=5)

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
        self.cfg.raw["gnn"]["hidden_dim"] = 32
        self.cfg.raw["gnn"]["layers"] = 2
        self.cfg.raw["gnn"]["dropout"] = 0.0
        self.cfg.raw.setdefault("multitask", {})
        self.cfg.raw["multitask"]["sepsis_cohort"] = "broad"
        self.cfg.raw["multitask"]["head_hidden_dim"] = 16
        self.cfg.raw["multitask"]["head_dropout"] = 0.0
        self.cfg.raw.setdefault("training", {})
        self.cfg.raw["training"]["local_epochs"] = 5
        self.cfg.raw["training"]["learning_rate"] = 0.01
        self.cfg.raw["training"]["weight_decay"] = 0.0
        self.cfg.raw["training"]["optimizer"] = "adam"
        self.cfg.raw["training"]["reconstruction_weight"] = 0.5
        self.cfg.raw["training"]["max_hospitals"] = None

        phase1_summary = run_phase1_ingestion(self.cfg)
        self.partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        self.phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, self.partition_result, patient_df)
        self.phase4_result = run_phase4_timeseries_modality(self.cfg, self.phase3_result, self.partition_result)
        self.phase5_result = run_phase5_tabular_modality(self.cfg, phase1_summary, self.phase3_result, self.partition_result)
        phase7_result = run_phase7_multimodal_fusion(self.cfg, self.phase3_result, self.phase4_result, self.phase5_result)
        self.phase8_result = run_phase8_patient_graph_construction(self.cfg, phase7_result, self.partition_result)

        if not phase7_result.get("fused_embeddings_path"):
            self.skipTest("torch not available in this environment")

        self.split_manifest = build_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], seed=42, stratify_on="mortality",
        )
        self.train_patient_ids = set(self.split_manifest.all_patient_ids("train"))
        non_train = set(self.split_manifest.all_patient_ids("val")) | set(self.split_manifest.all_patient_ids("test"))
        self.assertTrue(non_train, "test fixture needs at least one non-train patient")
        self.corrupt_patient_id = sorted(non_train)[0]
        self.assertNotIn(self.corrupt_patient_id, self.train_patient_ids)
        self.corrupt_patient_hospital_id = str(next(
            hid for hid, p in self.partition_result.partitions.items()
            if self.corrupt_patient_id in [int(pid) for pid in p.patient_ids]
        ))

    def tearDown(self):
        close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def _corrupted_phase5_result(self):
        """Returns a deep-enough copy of self.phase5_result with ONE
        non-train patient's raw tabular row wildly perturbed."""
        corrupted = dict(self.phase5_result)
        corrupted["normalized_array"] = self.phase5_result["normalized_array"].copy()
        patient_ids = [int(pid) for pid in self.phase5_result["patient_ids"]]
        row = patient_ids.index(self.corrupt_patient_id)
        corrupted["normalized_array"][row] = corrupted["normalized_array"][row] + 1000.0
        return corrupted

    def test_writes_to_a_separate_checkpoint_directory_by_default(self):
        result_original = run_phase11_local_training(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
        )
        result_split_aware = run_phase11_local_training(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
            train_patient_ids=self.train_patient_ids,
        )
        self.assertIn("phase11_local", Path(result_original["checkpoint_dir"]).name)
        self.assertNotEqual(Path(result_original["checkpoint_dir"]).name, Path(result_split_aware["checkpoint_dir"]).name)
        self.assertIn("splitaware", Path(result_split_aware["checkpoint_dir"]).name)
        # The original run's own checkpoint files must still exist --
        # the split-aware run must not have touched them.
        self.assertTrue(Path(result_original["checkpoint_dir"]).exists())
        self.assertTrue(Path(result_split_aware["checkpoint_dir"]).exists())

    def test_predictions_cover_every_patient_with_correct_train_tagging(self):
        result = run_phase11_local_training(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
            train_patient_ids=self.train_patient_ids,
        )
        found_any_non_train = False
        for hid, hospital_result in result["hospital_results"].items():
            preds = hospital_result["predictions"]
            for pid, is_train in zip(preds["patient_ids"], preds["is_train_row"]):
                expected = "train" if pid in self.train_patient_ids else "not_train"
                self.assertEqual(is_train, expected)
                if expected == "not_train":
                    found_any_non_train = True
        self.assertTrue(found_any_non_train, "test fixture should include at least one non-train prediction")

    def test_non_split_aware_training_IS_sensitive_to_a_non_train_patients_raw_values(self):
        """Establishes the test is actually meaningful: WITHOUT split-
        awareness, corrupting a patient's raw tabular row measurably
        changes the trained weights (proving the corruption is a real,
        detectable perturbation, not a no-op)."""
        torch.manual_seed(0)
        result_clean = run_phase11_local_training(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
            checkpoint_dir_name="phase11_local_sensitivity_clean",
        )
        torch.manual_seed(0)
        corrupted_phase5 = self._corrupted_phase5_result()
        result_corrupted = run_phase11_local_training(
            self.cfg, self.phase3_result, self.phase4_result, corrupted_phase5, self.phase8_result,
            checkpoint_dir_name="phase11_local_sensitivity_corrupted",
        )
        # Must compare the CORRUPTED PATIENT'S OWN hospital -- training
        # is per-hospital-independent, so any OTHER hospital's
        # checkpoint is correctly unaffected regardless of split-
        # awareness (that's the privacy boundary working correctly,
        # not a reason the sensitivity check should pass trivially).
        hid = self.corrupt_patient_hospital_id
        self.assertIn(hid, result_clean["hospital_results"])
        self.assertIn(hid, result_corrupted["hospital_results"])
        clean_ckpt = torch.load(result_clean["hospital_results"][hid]["checkpoint_path"], weights_only=True)
        corrupted_ckpt = torch.load(result_corrupted["hospital_results"][hid]["checkpoint_path"], weights_only=True)
        differs = any(not torch.allclose(clean_ckpt[k], corrupted_ckpt[k]) for k in clean_ckpt)
        self.assertTrue(differs, "corrupting a patient's raw values should have changed SOME weight without split-awareness")

    def test_split_aware_training_is_NOT_sensitive_to_a_non_train_patients_raw_values(self):
        """THE central leakage-fix proof: with split-awareness enabled,
        corrupting a NON-TRAIN patient's raw tabular row must leave the
        trained weights byte-identical -- this is the direct regression
        target for the reconstruction-term leakage fix described in
        _run_local_training_round()'s docstring. Uses the SAME hospital/
        seed as the previous test, which already proved the corruption
        is a real, detectable perturbation when NOT masked."""
        torch.manual_seed(0)
        result_clean = run_phase11_local_training(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
            train_patient_ids=self.train_patient_ids, checkpoint_dir_name="phase11_splitaware_sensitivity_clean",
        )
        torch.manual_seed(0)
        corrupted_phase5 = self._corrupted_phase5_result()
        result_corrupted = run_phase11_local_training(
            self.cfg, self.phase3_result, self.phase4_result, corrupted_phase5, self.phase8_result,
            train_patient_ids=self.train_patient_ids, checkpoint_dir_name="phase11_splitaware_sensitivity_corrupted",
        )
        # Compare every hospital that trained successfully in both runs.
        common_hids = set(result_clean["hospital_results"]) & set(result_corrupted["hospital_results"])
        self.assertTrue(common_hids)
        for hid in common_hids:
            clean_ckpt = torch.load(result_clean["hospital_results"][hid]["checkpoint_path"], weights_only=True)
            corrupted_ckpt = torch.load(result_corrupted["hospital_results"][hid]["checkpoint_path"], weights_only=True)
            for k in clean_ckpt:
                torch.testing.assert_close(
                    clean_ckpt[k], corrupted_ckpt[k],
                    msg=f"hospital {hid} parameter {k} changed after corrupting a NON-TRAIN patient's raw "
                        f"values under split-aware training -- the reconstruction-term leakage fix is not holding.",
                )

    def test_hospital_with_zero_train_patients_is_skipped_gracefully(self):
        # Deliberately empty train set -> every hospital has zero train patients.
        result = run_phase11_local_training(
            self.cfg, self.phase3_result, self.phase4_result, self.phase5_result, self.phase8_result,
            train_patient_ids=set(),
        )
        self.assertEqual(result["n_hospitals_trained"], 0)
        self.assertGreater(result["n_hospitals_skipped_no_train_patients"], 0)


if __name__ == "__main__":
    unittest.main()
