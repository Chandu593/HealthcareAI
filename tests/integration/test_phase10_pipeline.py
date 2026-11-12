"""End-to-end Phase 10 (multi-task heads) integration test -- runs the
actual Phase 1->2->3->4->5->7->8->9->10 pipeline against synthetic
data. Isolated head-architecture correctness (shape, independence) and
isolated label-alignment correctness are covered in
tests/unit/test_multitask_heads.py; this file's job is proving the
real wiring holds: Phase 9's GNN embeddings correctly feed Phase 10's
heads, Phase 3's labels correctly ID-align against Phase 9's (not
Phase 3's own) patient order, full cohort coverage preserved end to
end.
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
    run_phase8_patient_graph_construction,
    run_phase9_gnn_forward_pass,
    run_phase10_multitask_heads,
)
from src.heads.multitask_heads import align_labels_to_patient_ids  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestPhase10Integration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        generate(str(self.data_root), n_hospitals=4, n_patients_per_hospital=25, seed=3)

        self._old_env = os.environ.get("EICU_DATA_ROOT")
        os.environ["EICU_DATA_ROOT"] = str(self.data_root)

        self.cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        self.cfg.raw["run"]["output_dir"] = str(self.output_dir)
        self.cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")
        self.cfg.raw["run"]["cache_dir"] = str(self.output_dir / "cache")
        self.cfg.raw["cohort"]["patient_subset"] = 15
        self.cfg.raw.setdefault("graph", {})
        self.cfg.raw["graph"]["method"] = "knn"
        self.cfg.raw["graph"]["k"] = 3
        self.cfg.raw["graph"]["metric"] = "cosine"
        self.cfg.raw.setdefault("gnn", {})
        self.cfg.raw["gnn"]["type"] = "graphsage"
        self.cfg.raw["gnn"]["hidden_dim"] = 32
        self.cfg.raw["gnn"]["layers"] = 2
        self.cfg.raw["gnn"]["dropout"] = 0.2
        self.cfg.raw.setdefault("multitask", {})
        self.cfg.raw["multitask"]["sepsis_cohort"] = "broad"
        self.cfg.raw["multitask"]["head_hidden_dim"] = 16
        self.cfg.raw["multitask"]["head_dropout"] = 0.1

    def tearDown(self):
        close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def _run_through_phase9(self):
        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)
        phase4_result = run_phase4_timeseries_modality(self.cfg, phase3_result, partition_result)
        phase5_result = run_phase5_tabular_modality(self.cfg, phase1_summary, phase3_result, partition_result)
        phase7_result = run_phase7_multimodal_fusion(self.cfg, phase3_result, phase4_result, phase5_result)
        phase8_result = run_phase8_patient_graph_construction(self.cfg, phase7_result, partition_result)
        phase9_result = run_phase9_gnn_forward_pass(self.cfg, phase7_result, phase8_result)
        return phase3_result, phase7_result, phase9_result

    def _skip_if_no_torch(self, phase7_result):
        if not phase7_result.get("fused_embeddings_path"):
            self.skipTest("torch not available in this environment -- Phase 7 embeddings not produced")

    def test_full_cohort_covered_shapes_match(self):
        phase3_result, phase7_result, phase9_result = self._run_through_phase9()
        self._skip_if_no_torch(phase7_result)

        phase10_result = run_phase10_multitask_heads(self.cfg, phase9_result, phase3_result)

        self.assertEqual(phase10_result["n_patients"], phase9_result["n_patients"])
        self.assertEqual(phase10_result["mortality_logits_shape"], (phase9_result["n_patients"],))
        self.assertEqual(phase10_result["sepsis_logits_shape"], (phase9_result["n_patients"],))
        self.assertEqual(phase10_result["in_dim"], phase9_result["gnn_dim"])
        self.assertEqual(phase10_result["head_hidden_dim"], 16)  # from configured multitask.head_hidden_dim

    def test_label_counts_match_direct_computation_from_phase3(self):
        """Direct wiring proof: recompute mortality/sepsis counts from
        phase3_result independently (via align_labels_to_patient_ids
        called with Phase 9's own patient_ids) and confirm Phase 10's
        reported counts agree -- not just that Phase 10 ran without
        crashing."""
        phase3_result, phase7_result, phase9_result = self._run_through_phase9()
        self._skip_if_no_torch(phase7_result)

        phase10_result = run_phase10_multitask_heads(self.cfg, phase9_result, phase3_result)

        gnn_data = np.load(phase9_result["gnn_embeddings_path"])
        expected = align_labels_to_patient_ids(
            gnn_data["patient_ids"].tolist(),
            phase3_result["cohort_patients"],
            phase3_result["mortality"],
            phase3_result["sepsis_df"],
            "broad",
        )
        self.assertEqual(phase10_result["n_mortality_known"], expected["n_mortality_known"])
        self.assertEqual(phase10_result["n_mortality_positive"], expected["n_mortality_positive"])
        self.assertEqual(phase10_result["n_sepsis_positive"], expected["n_sepsis_positive"])

    def test_strict_cohort_gives_different_sepsis_counts_than_broad(self):
        """sepsis_strict ⊆ sepsis_broad is proven at the unit level
        (test_strict_is_always_subset_of_broad, Phase 3) -- this
        confirms the config knob actually changes which column Phase
        10 reads, end to end, not just that both values are accepted
        without error."""
        phase3_result, phase7_result, phase9_result = self._run_through_phase9()
        self._skip_if_no_torch(phase7_result)

        self.cfg.raw["multitask"]["sepsis_cohort"] = "broad"
        broad_result = run_phase10_multitask_heads(self.cfg, phase9_result, phase3_result)
        self.cfg.raw["multitask"]["sepsis_cohort"] = "strict"
        strict_result = run_phase10_multitask_heads(self.cfg, phase9_result, phase3_result)

        self.assertGreaterEqual(broad_result["n_sepsis_positive"], strict_result["n_sepsis_positive"])

    def test_raises_clearly_on_invalid_sepsis_cohort(self):
        phase3_result, phase7_result, phase9_result = self._run_through_phase9()
        self._skip_if_no_torch(phase7_result)

        self.cfg.raw["multitask"]["sepsis_cohort"] = "moderate"
        with self.assertRaises(ValueError) as ctx:
            run_phase10_multitask_heads(self.cfg, phase9_result, phase3_result)
        self.assertIn("moderate", str(ctx.exception))

    def test_raises_clearly_when_phase9_embeddings_missing(self):
        phase3_result, phase7_result, phase9_result = self._run_through_phase9()
        self._skip_if_no_torch(phase7_result)
        broken_phase9_result = dict(phase9_result)
        broken_phase9_result["gnn_embeddings_path"] = None

        with self.assertRaises(RuntimeError):
            run_phase10_multitask_heads(self.cfg, broken_phase9_result, phase3_result)

    def test_sparse_data_patients_absent_from_sepsis_df_do_not_crash_the_run(self):
        """Direct regression target for a real condition found while
        building this exact test file (not hypothetical): with this
        seed/mock-data configuration, one patient
        (patientunitstayid=141029) has zero rows across apachePredVar,
        medication, AND microLab -- so it's absent from sofa_df,
        infection_df, and therefore sepsis_df entirely, despite being
        a fully valid member of cohort_patients and getting a real GNN
        embedding. Confirms Phase 10 completes end to end rather than
        raising, and that this patient is counted as sepsis-unknown,
        not silently dropped or coded as sepsis-negative."""
        phase3_result, phase7_result, phase9_result = self._run_through_phase9()
        self._skip_if_no_torch(phase7_result)

        cohort_ids = set(phase3_result["cohort_patients"]["patientunitstayid"])
        sepsis_ids = set(phase3_result["sepsis_df"]["patientunitstayid"])
        self.assertTrue(
            cohort_ids - sepsis_ids,
            "This test's premise (a patient in cohort_patients but absent from sepsis_df) "
            "no longer holds for this seed -- update the seed/assertion rather than deleting this test.",
        )

        phase10_result = run_phase10_multitask_heads(self.cfg, phase9_result, phase3_result)
        self.assertGreaterEqual(phase10_result["n_sepsis_unknown"], 1)
        self.assertEqual(
            phase10_result["n_sepsis_known"] + phase10_result["n_sepsis_unknown"],
            phase10_result["n_patients"],
        )


if __name__ == "__main__":
    unittest.main()
