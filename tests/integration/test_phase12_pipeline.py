"""End-to-end Phase 12 (FedAvg) integration test -- runs the actual
Phase 1->2->3->4->5->7->8->12 pipeline against synthetic data.
Aggregation-function correctness in isolation is covered in
tests/unit/test_fedavg.py; this file's job is proving the real
multi-round wiring holds: round 1 reuse of existing checkpoints (with
NaN exclusion), the fallback-to-fresh path, rounds 2+ genuinely
starting from the previous round's aggregate (not a fresh random
init), and manifest persistence.
"""

import json
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
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestPhase12Integration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        self.checkpoint_dir = Path(self.tmpdir.name) / "checkpoints"
        generate(str(self.data_root), n_hospitals=4, n_patients_per_hospital=25, seed=3)

        self._old_env = os.environ.get("EICU_DATA_ROOT")
        os.environ["EICU_DATA_ROOT"] = str(self.data_root)

        self.cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        self.cfg.raw["run"]["output_dir"] = str(self.output_dir)
        self.cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")
        self.cfg.raw["run"]["cache_dir"] = str(self.output_dir / "cache")
        self.cfg.raw["run"]["checkpoint_dir"] = str(self.checkpoint_dir)
        self.cfg.raw["cohort"]["patient_subset"] = 15
        self.cfg.raw.setdefault("graph", {})
        self.cfg.raw["graph"]["method"] = "knn"
        self.cfg.raw["graph"]["k"] = 3
        self.cfg.raw["graph"]["metric"] = "cosine"
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
        self.cfg.raw["training"]["local_epochs"] = 3
        self.cfg.raw["training"]["learning_rate"] = 0.01
        self.cfg.raw["training"]["weight_decay"] = 0.0
        self.cfg.raw["training"]["optimizer"] = "adam"
        self.cfg.raw["training"]["reconstruction_weight"] = 0.5
        self.cfg.raw["training"]["max_hospitals"] = None
        self.cfg.raw.setdefault("federated", {})
        self.cfg.raw["federated"]["n_rounds"] = 3
        self.cfg.raw["federated"]["reuse_existing_local_checkpoints_for_round_1"] = False

    def tearDown(self):
        close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def _run_through_phase8(self):
        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)
        phase4_result = run_phase4_timeseries_modality(self.cfg, phase3_result, partition_result)
        phase5_result = run_phase5_tabular_modality(self.cfg, phase1_summary, phase3_result, partition_result)
        phase7_result = run_phase7_multimodal_fusion(self.cfg, phase3_result, phase4_result, phase5_result)
        phase8_result = run_phase8_patient_graph_construction(self.cfg, phase7_result, partition_result)
        return phase3_result, phase4_result, phase5_result, phase7_result, phase8_result

    def _skip_if_no_torch(self, phase7_result):
        if not phase7_result.get("fused_embeddings_path"):
            self.skipTest("torch not available in this environment -- Phase 7 embeddings not produced")

    def test_runs_all_configured_rounds_fresh(self):
        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        result = run_phase12_fedavg(self.cfg, phase3, phase4, phase5, phase8)

        self.assertEqual(result["n_rounds"], 3)
        self.assertEqual(len(result["round_summaries"]), 3)
        for r in result["round_summaries"]:
            self.assertEqual(r["source"], "fresh_local_training")
            self.assertTrue(Path(r["global_model_path"]).exists())
        self.assertTrue(Path(result["final_global_model_path"]).exists())
        self.assertEqual(result["final_global_model_path"], result["round_summaries"][-1]["global_model_path"])

    def test_round2_starts_from_round1_aggregate_not_a_fresh_random_init(self):
        """Direct regression target: the whole point of multi-round
        FedAvg is that each round builds on the PREVIOUS round's
        aggregated global model, not a fresh random restart. Proven by
        intercepting every model.load_state_dict() call across the
        run and confirming round 2's calls received EXACTLY round 1's
        saved aggregated state_dict, not the original round-1 initial
        weights and not an independently-random init."""
        import unittest.mock as mock

        from src.training.local_model import LocalPatientModel

        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        received_by_call: list = []
        original_load_state_dict = LocalPatientModel.load_state_dict

        def _spy(self, state_dict, *args, **kwargs):
            received_by_call.append({k: v.clone() for k, v in state_dict.items()})
            return original_load_state_dict(self, state_dict, *args, **kwargs)

        with mock.patch.object(LocalPatientModel, "load_state_dict", _spy):
            result = run_phase12_fedavg(self.cfg, phase3, phase4, phase5, phase8)

        round1_aggregate = torch.load(result["round_summaries"][0]["global_model_path"], weights_only=True)
        # At least one hospital in round 2 must have started from EXACTLY
        # round 1's saved aggregate.
        round1_state_count = 4  # n_hospitals in this mock dataset
        round2_first_call = received_by_call[round1_state_count]
        for key in round1_aggregate:
            torch.testing.assert_close(round2_first_call[key], round1_aggregate[key])

    def test_reuses_existing_phase11_checkpoints_for_round_1_when_enabled(self):
        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        # Produce a real prior Phase 11 run to reuse.
        run_phase11_local_training(self.cfg, phase3, phase4, phase5, phase8)

        self.cfg.raw["federated"]["reuse_existing_local_checkpoints_for_round_1"] = True
        self.cfg.raw["federated"]["n_rounds"] = 2
        result = run_phase12_fedavg(self.cfg, phase3, phase4, phase5, phase8)

        self.assertEqual(result["round_summaries"][0]["source"], "reused_phase11_checkpoints")
        self.assertEqual(result["round_summaries"][1]["source"], "fresh_local_training")

    def test_falls_back_to_fresh_when_reuse_requested_but_nothing_exists(self):
        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        # No prior Phase 11 run in this tmpdir -- checkpoints/phase11_local/ doesn't exist.
        self.cfg.raw["federated"]["reuse_existing_local_checkpoints_for_round_1"] = True
        self.cfg.raw["federated"]["n_rounds"] = 1
        result = run_phase12_fedavg(self.cfg, phase3, phase4, phase5, phase8)

        self.assertEqual(result["round_summaries"][0]["source"], "fresh_local_training")

    def test_nan_corrupted_reused_checkpoint_is_excluded_not_included(self):
        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        phase11_result = run_phase11_local_training(self.cfg, phase3, phase4, phase5, phase8)
        self.assertGreaterEqual(len(phase11_result["hospital_results"]), 2, "need >=2 trained hospitals")

        # Corrupt exactly one hospital's checkpoint in place.
        poisoned_hid = next(iter(phase11_result["hospital_results"]))
        poisoned_path = Path(phase11_result["hospital_results"][poisoned_hid]["checkpoint_path"])
        state = torch.load(poisoned_path, weights_only=True)
        for k in state:
            state[k] = state[k] * float("nan")
            break  # corrupting one tensor is enough
        torch.save(state, poisoned_path)

        self.cfg.raw["federated"]["reuse_existing_local_checkpoints_for_round_1"] = True
        self.cfg.raw["federated"]["n_rounds"] = 1
        result = run_phase12_fedavg(self.cfg, phase3, phase4, phase5, phase8)

        self.assertEqual(
            result["round_summaries"][0]["n_hospitals_aggregated"],
            len(phase11_result["hospital_results"]) - 1,
            "the poisoned hospital should have been excluded, not aggregated in.",
        )
        # And the final aggregate itself must be finite -- proving the
        # exclusion actually happened, not just that the count looked right.
        final_state = torch.load(result["final_global_model_path"], weights_only=True)
        for v in final_state.values():
            self.assertTrue(torch.isfinite(v).all())

    def test_fedavg_manifest_persisted_with_expected_fields(self):
        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        result = run_phase12_fedavg(self.cfg, phase3, phase4, phase5, phase8)

        manifest_path = Path(result["fedavg_manifest_path"])
        self.assertTrue(manifest_path.exists())
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.assertEqual(manifest["n_rounds"], 3)
        self.assertEqual(len(manifest["rounds"]), 3)
        self.assertEqual(manifest["final_global_model_path"], result["final_global_model_path"])

    def test_raises_clearly_on_invalid_n_rounds(self):
        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        self.cfg.raw["federated"]["n_rounds"] = 0
        with self.assertRaises(ValueError) as ctx:
            run_phase12_fedavg(self.cfg, phase3, phase4, phase5, phase8)
        self.assertIn("n_rounds", str(ctx.exception))

    def test_final_aggregate_matches_direct_fedavg_computation(self):
        """Direct end-to-end correctness proof, not just 'a file
        exists': for a single-round run, independently recompute the
        weighted average from the round's own saved per-hospital
        checkpoints and confirm it matches the pipeline's own
        aggregated result exactly."""
        from src.training.fedavg import fedavg_aggregate

        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        self.cfg.raw["federated"]["n_rounds"] = 1
        result = run_phase12_fedavg(self.cfg, phase3, phase4, phase5, phase8)

        round_dir = Path(result["round_summaries"][0]["checkpoint_dir"])
        with open(round_dir / "training_manifest.json") as f:
            manifest = json.load(f)

        state_dicts = {
            hid: torch.load(h["checkpoint_path"], weights_only=True) for hid, h in manifest["hospitals"].items()
        }
        weights = {hid: h["n_patients"] for hid, h in manifest["hospitals"].items()}
        expected = fedavg_aggregate(state_dicts, weights)

        actual = torch.load(result["final_global_model_path"], weights_only=True)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_each_fresh_round_checkpoint_dir_is_self_contained(self):
        """Direct regression target for a real gap found 2026-09-02 by
        double-checking a docs claim before the user could hit it:
        scripts/check_local_training_quality.py looks for
        `global_initial_weights.pt` inside whatever --checkpoint-dir
        it's pointed at. Before this fix, only Phase 11's own
        top-level checkpoint dir ever got that file -- Phase 12's
        round_2/, round_3/, etc. subdirectories didn't, silently
        breaking the diagnostic script for every round except Phase
        11's own. Every FRESH round's checkpoint_dir must now contain
        its own copy, matching what that round actually started from
        (the previous round's aggregate, for rounds 2+)."""
        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        self.cfg.raw["federated"]["n_rounds"] = 2
        result = run_phase12_fedavg(self.cfg, phase3, phase4, phase5, phase8)

        round1_aggregate = torch.load(result["round_summaries"][0]["global_model_path"], weights_only=True)
        round2_dir = Path(result["round_summaries"][1]["checkpoint_dir"])
        round2_initial_path = round2_dir / "global_initial_weights.pt"

        self.assertTrue(
            round2_initial_path.exists(),
            "round 2's checkpoint directory has no global_initial_weights.pt -- "
            "check_local_training_quality.py would fail against it.",
        )
        round2_initial = torch.load(round2_initial_path, weights_only=True)
        for key in round1_aggregate:
            torch.testing.assert_close(round2_initial[key], round1_aggregate[key])


if __name__ == "__main__":
    unittest.main()
