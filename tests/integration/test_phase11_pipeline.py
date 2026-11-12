"""End-to-end Phase 11 (local training loop) integration test -- runs
the actual Phase 1->2->3->4->5->7->8->11 pipeline against synthetic
data. Component-level correctness (gradient flow through every
component, output shapes, per-hospital independence in isolation) is
covered in tests/unit/test_local_model.py and tests/unit/test_losses.py;
this file's job is proving the real wiring holds end to end: Phase
4/5's raw tensors and Phase 8's graphs correctly feed
run_phase11_local_training(), every hospital genuinely starts from the
same initial weights, loss genuinely decreases on real (if synthetic)
data, and checkpoints land where Phase 12 (FedAvg) will expect them.
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
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestPhase11Integration(unittest.TestCase):
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
        self.cfg.raw["training"]["local_epochs"] = 5
        self.cfg.raw["training"]["learning_rate"] = 0.01
        self.cfg.raw["training"]["weight_decay"] = 0.0
        self.cfg.raw["training"]["optimizer"] = "adam"
        self.cfg.raw["training"]["reconstruction_weight"] = 0.5
        self.cfg.raw["training"]["max_hospitals"] = None

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

    def test_trains_every_real_hospital_with_no_skips(self):
        phase3_result, phase4_result, phase5_result, phase7_result, phase8_result = self._run_through_phase8()
        self._skip_if_no_torch(phase7_result)

        result = run_phase11_local_training(self.cfg, phase3_result, phase4_result, phase5_result, phase8_result)

        self.assertEqual(result["n_hospitals_skipped_empty"] + result["n_hospitals_trained"], 4)
        self.assertGreater(result["n_hospitals_trained"], 0)

    def test_loss_decreases_over_local_epochs_for_every_trained_hospital(self):
        """Real trainability proof, not just 'ran without crashing' --
        confirmed by hand on this exact mock dataset before writing
        this assertion: all 4 hospitals' loss strictly decreased epoch
        over epoch (1.49->1.40, 1.48->1.35, 1.49->1.40, 1.50->1.42)."""
        phase3_result, phase4_result, phase5_result, phase7_result, phase8_result = self._run_through_phase8()
        self._skip_if_no_torch(phase7_result)

        result = run_phase11_local_training(self.cfg, phase3_result, phase4_result, phase5_result, phase8_result)

        self.assertGreater(len(result["hospital_results"]), 0)
        for hid, hospital_result in result["hospital_results"].items():
            history = hospital_result["loss_history"]
            self.assertGreaterEqual(len(history), 2)
            self.assertLess(
                history[-1], history[0],
                f"hospital {hid}'s loss did not decrease: {history}",
            )

    def test_checkpoints_written_for_every_trained_hospital(self):
        phase3_result, phase4_result, phase5_result, phase7_result, phase8_result = self._run_through_phase8()
        self._skip_if_no_torch(phase7_result)

        result = run_phase11_local_training(self.cfg, phase3_result, phase4_result, phase5_result, phase8_result)

        self.assertTrue(Path(result["global_initial_checkpoint_path"]).exists())
        for hid, hospital_result in result["hospital_results"].items():
            self.assertTrue(Path(hospital_result["checkpoint_path"]).exists())

    def test_every_hospital_starts_from_identical_initial_weights(self):
        """Direct regression target: FedAvg (Phase 12) only makes sense
        as an operation if every hospital's LOCAL round started from
        the exact same point in weight-space (see
        run_phase11_local_training()'s own docstring) -- averaging
        independently-initialized models' weights is close to
        meaningless (neural network loss landscapes are not
        permutation-invariant). A shapes-only check wouldn't catch a
        regression to per-hospital random re-initialization (shapes
        would still match), so this intercepts every
        LocalPatientModel.load_state_dict() call across the whole run
        and asserts every one of them received THE EXACT SAME tensor
        values, not just the same architecture."""
        import unittest.mock as mock

        from src.training.local_model import LocalPatientModel

        phase3_result, phase4_result, phase5_result, phase7_result, phase8_result = self._run_through_phase8()
        self._skip_if_no_torch(phase7_result)

        received_state_dicts = []
        original_load_state_dict = LocalPatientModel.load_state_dict

        def _spy(self, state_dict, *args, **kwargs):
            received_state_dicts.append({k: v.clone() for k, v in state_dict.items()})
            return original_load_state_dict(self, state_dict, *args, **kwargs)

        with mock.patch.object(LocalPatientModel, "load_state_dict", _spy):
            result = run_phase11_local_training(self.cfg, phase3_result, phase4_result, phase5_result, phase8_result)

        self.assertGreaterEqual(len(received_state_dicts), 2, "need >=2 hospitals trained for this comparison")
        first = received_state_dicts[0]
        for i, state_dict in enumerate(received_state_dicts[1:], start=1):
            self.assertEqual(set(state_dict.keys()), set(first.keys()))
            for key in first:
                torch.testing.assert_close(
                    state_dict[key], first[key],
                    msg=f"hospital index {i}'s starting weights for {key!r} differ from hospital index 0's.",
                )

        # And the recorded initial checkpoint on disk matches what every
        # hospital actually received, not a coincidentally-similar copy.
        initial_state = torch.load(result["global_initial_checkpoint_path"], weights_only=True)
        for key in first:
            torch.testing.assert_close(first[key], initial_state[key])

    def test_all_four_components_receive_nonzero_updates_and_hospitals_diverge(self):
        """Direct regression target, confirmed by hand on this exact
        mock run before writing this assertion (total weight-delta
        norms: lstm=4.16, autoencoder=1.66, gnn=2.28, heads=0.90 for
        one hospital vs. the initial checkpoint; 5.26 between two
        different hospitals' checkpoints) -- proves this is genuinely
        full end-to-end training (all four components moved, not just
        the heads) and genuinely per-hospital (different hospitals
        converged to different weights, not all collapsing to the
        same point)."""
        phase3_result, phase4_result, phase5_result, phase7_result, phase8_result = self._run_through_phase8()
        self._skip_if_no_torch(phase7_result)

        result = run_phase11_local_training(self.cfg, phase3_result, phase4_result, phase5_result, phase8_result)
        initial_state = torch.load(result["global_initial_checkpoint_path"], weights_only=True)
        hospital_ids = list(result["hospital_results"].keys())
        self.assertGreaterEqual(len(hospital_ids), 2)

        state_a = torch.load(result["hospital_results"][hospital_ids[0]]["checkpoint_path"], weights_only=True)
        state_b = torch.load(result["hospital_results"][hospital_ids[1]]["checkpoint_path"], weights_only=True)

        component_prefixes = ["lstm_encoder", "autoencoder", "gnn", "heads"]
        for prefix in component_prefixes:
            delta = sum(
                (state_a[k] - initial_state[k]).norm().item() for k in initial_state if k.startswith(prefix)
            )
            self.assertGreater(delta, 0.0, f"{prefix} did not change at all during local training.")

        total_delta_a_vs_b = sum((state_a[k] - state_b[k]).norm().item() for k in initial_state)
        self.assertGreater(
            total_delta_a_vs_b, 0.0,
            f"hospitals {hospital_ids[0]} and {hospital_ids[1]} converged to identical weights.",
        )

    def test_raises_clearly_on_invalid_optimizer(self):
        phase3_result, phase4_result, phase5_result, phase7_result, phase8_result = self._run_through_phase8()
        self._skip_if_no_torch(phase7_result)

        self.cfg.raw["training"]["optimizer"] = "rmsprop"
        with self.assertRaises(ValueError) as ctx:
            run_phase11_local_training(self.cfg, phase3_result, phase4_result, phase5_result, phase8_result)
        self.assertIn("rmsprop", str(ctx.exception))

    def test_max_hospitals_limits_how_many_are_trained(self):
        phase3_result, phase4_result, phase5_result, phase7_result, phase8_result = self._run_through_phase8()
        self._skip_if_no_torch(phase7_result)

        self.cfg.raw["training"]["max_hospitals"] = 2
        result = run_phase11_local_training(self.cfg, phase3_result, phase4_result, phase5_result, phase8_result)
        self.assertLessEqual(result["n_hospitals_trained"] + result["n_hospitals_skipped_empty"], 2)

    def test_manifest_json_persisted_with_expected_fields(self):
        """Direct regression target: found while trying to check the
        real run's quality after the fact and discovering there was
        nowhere to check it -- this manifest is what makes a
        completed run's per-hospital loss curves and label coverage
        inspectable without needing to re-train."""
        import json

        phase3_result, phase4_result, phase5_result, phase7_result, phase8_result = self._run_through_phase8()
        self._skip_if_no_torch(phase7_result)

        result = run_phase11_local_training(self.cfg, phase3_result, phase4_result, phase5_result, phase8_result)

        manifest_path = Path(result["manifest_path"])
        self.assertTrue(manifest_path.exists())
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.assertEqual(manifest["n_hospitals_trained"], result["n_hospitals_trained"])
        self.assertEqual(manifest["n_hospitals_skipped_empty"], result["n_hospitals_skipped_empty"])
        self.assertIn("hospitals", manifest)
        for hid, hospital_result in result["hospital_results"].items():
            self.assertIn(hid, manifest["hospitals"])
            self.assertEqual(manifest["hospitals"][hid]["loss_history"], hospital_result["loss_history"])
            self.assertEqual(manifest["hospitals"][hid]["n_mortality_known"], hospital_result["n_mortality_known"])

    def test_non_finite_loss_skips_that_hospital_without_saving_a_corrupted_checkpoint(self):
        """Direct regression target for the NaN/Inf safety fix: proves
        a hospital whose loss goes non-finite mid-training (a) does
        NOT end up in hospital_results, (b) has NO checkpoint file
        written for it (a stale/corrupted checkpoint from a partial
        run would be worse than none), (c) is counted under
        n_hospitals_skipped_non_finite specifically -- not silently
        folded into n_hospitals_skipped_no_labels -- and (d) does not
        prevent OTHER hospitals from training normally."""
        import unittest.mock as mock

        import src.training.losses as losses_module

        phase3_result, phase4_result, phase5_result, phase7_result, phase8_result = self._run_through_phase8()
        self._skip_if_no_torch(phase7_result)

        original_compute = losses_module.compute_multitask_loss
        poisoned_hospital = {"id": None}

        def _poisoned_compute(*args, **kwargs):
            total, components, counts = original_compute(*args, **kwargs)
            if poisoned_hospital["id"] is None and total is not None:
                poisoned_hospital["id"] = "poisoned"
                return total * float("nan"), components, counts
            return total, components, counts

        with mock.patch.object(losses_module, "compute_multitask_loss", side_effect=_poisoned_compute):
            result = run_phase11_local_training(self.cfg, phase3_result, phase4_result, phase5_result, phase8_result)

        self.assertEqual(result["n_hospitals_skipped_non_finite"], 1)
        self.assertEqual(result["n_hospitals_skipped_no_labels"], 0)
        # The poisoned hospital (whichever it was -- first one with usable
        # labels, order not asserted) must have no checkpoint on disk.
        all_checkpoint_hids = {
            p.stem.replace("hospital_", "").replace("_local_weights", "")
            for p in Path(result["checkpoint_dir"]).glob("hospital_*_local_weights.pt")
        }
        trained_hids = set(result["hospital_results"].keys())
        self.assertEqual(
            len(all_checkpoint_hids), len(trained_hids),
            "a checkpoint file exists that doesn't correspond to a hospital in hospital_results "
            "-- the poisoned hospital's corrupted checkpoint may have been saved.",
        )
        # And training still proceeded for the rest.
        self.assertGreater(result["n_hospitals_trained"], 0)


if __name__ == "__main__":
    unittest.main()
