"""End-to-end Phase 13 (Baselines) integration test -- runs the actual
Phase 1->2->3->4->5->7->8->13 pipeline against synthetic data.

The single most important property this phase introduces: the
"centralized" baseline uses ONE continuously-updated model that is
NEVER reset between hospitals -- structurally different from every
prior training phase (Phase 11/12 explicitly reset every hospital back
to a shared starting point). test_centralized_model_constructed_exactly_once
is the direct proof.
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
    run_phase12_fedavg,
    run_phase13_baselines,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestPhase13Integration(unittest.TestCase):
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
        self.cfg.raw["federated"]["n_rounds"] = 2  # total_epochs = 3*2 = 6
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

    def test_local_baseline_gets_the_fair_matched_epoch_budget(self):
        """training.local_epochs=3, federated.n_rounds=2 -> total_epochs
        should be 6, NOT the raw training.local_epochs=3 Phase 11 alone
        would use."""
        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        result = run_phase13_baselines(self.cfg, phase3, phase4, phase5, phase8)

        self.assertEqual(result["total_epochs"], 6)
        with open(result["local_only"]["manifest_path"]) as f:
            local_manifest = json.load(f)
        self.assertEqual(local_manifest["local_epochs"], 6)
        for h in local_manifest["hospitals"].values():
            self.assertEqual(len(h["loss_history"]), 6)

    def test_centralized_baseline_also_gets_the_matched_epoch_budget(self):
        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        result = run_phase13_baselines(self.cfg, phase3, phase4, phase5, phase8)

        self.assertEqual(len(result["centralized_sequential"]["epoch_history"]), 6)

    def test_centralized_model_constructed_exactly_once(self):
        """Direct regression target for this phase's single most
        important, defining property: the centralized baseline is ONE
        continuously-updated model, never reset between hospitals --
        structurally different from Phase 11/12, which construct a
        fresh LocalPatientModel and call load_state_dict() once PER
        HOSPITAL. Proven by intercepting the constructor itself and
        confirming it's called exactly once for the whole centralized
        run (4 hospitals x 6 epochs = 24 hospital-visits, but only 1
        model)."""
        import unittest.mock as mock

        from src.training.local_model import LocalPatientModel

        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        construction_count = {"n": 0}
        original_init = LocalPatientModel.__init__

        def _spy(self, *args, **kwargs):
            construction_count["n"] += 1
            return original_init(self, *args, **kwargs)

        with mock.patch.object(LocalPatientModel, "__init__", _spy):
            run_phase13_baselines(self.cfg, phase3, phase4, phase5, phase8)

        # Exactly 2 LocalPatientModel constructions total for this whole
        # Phase 13 run: one for the local-only baseline's seeded initial
        # model (immediately handed to _run_local_training_round, which
        # THEN constructs a fresh one per hospital), and one for the
        # centralized baseline itself. The centralized one must NEVER be
        # reconstructed across its 4 hospitals x 6 epochs = 24 visits.
        n_hospitals = 4
        total_epochs = 6
        expected_local_only_constructions = 1 + n_hospitals  # 1 seeded-init + 1 per hospital
        expected_centralized_constructions = 1  # constructed ONCE, never again
        self.assertEqual(
            construction_count["n"], expected_local_only_constructions + expected_centralized_constructions,
            "LocalPatientModel was constructed a different number of times than expected -- if this is "
            "HIGHER than expected, the centralized baseline is very likely being reset per hospital or "
            "per epoch, defeating its entire purpose.",
        )

    def test_centralized_visit_order_is_not_identical_across_all_epochs(self):
        """Low-flakiness check: with 4 hospitals (24 possible orderings)
        and 6 epochs, the chance every single epoch coincidentally
        shuffles to the exact same order is astronomically small if
        shuffling is actually happening."""
        import unittest.mock as mock

        import run_pipeline as rp

        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        visit_orders: list[list[str]] = []
        original_load_batch = rp._load_hospital_batch
        current_epoch_order: list[str] = []

        def _spy(hid, *args, **kwargs):
            current_epoch_order.append(hid)
            return original_load_batch(hid, *args, **kwargs)

        # Wrap at the point where each epoch's order is reset -- simplest
        # is to just record hospital-visit sequence overall and later
        # chunk it into epochs of size n_hospitals.
        with mock.patch.object(rp, "_load_hospital_batch", side_effect=_spy):
            run_phase13_baselines(self.cfg, phase3, phase4, phase5, phase8)

        # current_epoch_order now holds the FULL sequence across both
        # local-training (per-hospital, order = sorted hospital_ids,
        # not shuffled) AND centralized training. We only care about
        # the centralized portion, which is the tail of this sequence
        # (4 hospitals visited once each by local training first, in
        # sorted order, then 4 hospitals x 6 epochs for centralized).
        n_hospitals = 4
        centralized_visits = current_epoch_order[n_hospitals:]  # skip local-only's 4 visits
        epochs = [centralized_visits[i:i + n_hospitals] for i in range(0, len(centralized_visits), n_hospitals)]
        self.assertEqual(len(epochs), 6)
        unique_orders = {tuple(e) for e in epochs}
        self.assertGreater(
            len(unique_orders), 1,
            f"all 6 epochs used the identical hospital visit order {epochs[0]} -- shuffling is very "
            f"likely not happening (or is happening but was called with the same seed state every time).",
        )

    def test_federated_reference_links_to_phase12s_actual_result(self):
        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        fedavg_result = run_phase12_fedavg(self.cfg, phase3, phase4, phase5, phase8)
        result = run_phase13_baselines(self.cfg, phase3, phase4, phase5, phase8)

        self.assertEqual(result["federated_reference_model_path"], fedavg_result["final_global_model_path"])

    def test_federated_reference_missing_is_handled_gracefully(self):
        """Phase 12 never ran in this test -- Phase 13 must still
        complete successfully (local + centralized are independent of
        it), just with federated_reference_model_path=None."""
        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        result = run_phase13_baselines(self.cfg, phase3, phase4, phase5, phase8)

        self.assertIsNone(result["federated_reference_model_path"])
        self.assertIsNotNone(result["centralized_sequential"]["final_model_path"])

    def test_non_finite_step_is_skipped_without_crashing_centralized_run(self):
        """Stubs out the local-only baseline entirely (irrelevant to
        this test, and at this test class's learning_rate=0.01 it
        naturally produces its own real NaN for one hospital -- see
        the actual log output -- which would make a call-count-based
        poison targeting 'the Nth call overall' land unpredictably).
        Isolating to centralized-only calls makes the poisoning land
        deterministically regardless of local's own training dynamics."""
        import unittest.mock as mock

        import run_pipeline as rp
        import src.training.losses as losses_module

        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        def _stub_local_training_round(*args, **kwargs):
            return {
                "checkpoint_dir": "", "manifest_path": "", "n_hospitals_trained": 0,
                "n_hospitals_skipped_empty": 0, "n_hospitals_skipped_no_labels": 0,
                "n_hospitals_skipped_non_finite": 0, "hospital_results": {},
            }

        original_compute = losses_module.compute_multitask_loss
        call_count = {"n": 0}

        def _poisoned_compute(*args, **kwargs):
            call_count["n"] += 1
            total, components, counts = original_compute(*args, **kwargs)
            if call_count["n"] == 5 and total is not None:  # poison one arbitrary centralized step
                return total * float("nan"), components, counts
            return total, components, counts

        with mock.patch.object(rp, "_run_local_training_round", side_effect=_stub_local_training_round), \
             mock.patch.object(losses_module, "compute_multitask_loss", side_effect=_poisoned_compute):
            result = run_phase13_baselines(self.cfg, phase3, phase4, phase5, phase8)

        self.assertGreater(result["centralized_sequential"]["n_non_finite_step_events"], 0)
        final_state = torch.load(result["centralized_sequential"]["final_model_path"], weights_only=True)
        for v in final_state.values():
            self.assertTrue(torch.isfinite(v).all())

    def test_baselines_manifest_persisted_with_expected_fields(self):
        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        result = run_phase13_baselines(self.cfg, phase3, phase4, phase5, phase8)

        manifest_path = Path(result["manifest_path"])
        self.assertTrue(manifest_path.exists())
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.assertEqual(manifest["total_epochs"], 6)
        self.assertIn("local_only", manifest)
        self.assertIn("centralized_sequential", manifest)
        self.assertIn("federated_reference", manifest)

    def test_local_and_centralized_final_models_genuinely_differ(self):
        phase3, phase4, phase5, phase7, phase8 = self._run_through_phase8()
        self._skip_if_no_torch(phase7)

        result = run_phase13_baselines(self.cfg, phase3, phase4, phase5, phase8)

        centralized_state = torch.load(result["centralized_sequential"]["final_model_path"], weights_only=True)
        with open(result["local_only"]["manifest_path"]) as f:
            local_manifest = json.load(f)
        any_hospital = next(iter(local_manifest["hospitals"].values()))
        local_state = torch.load(any_hospital["checkpoint_path"], weights_only=True)

        total_delta = sum((centralized_state[k] - local_state[k]).norm().item() for k in centralized_state)
        self.assertGreater(total_delta, 0.0)


if __name__ == "__main__":
    unittest.main()
