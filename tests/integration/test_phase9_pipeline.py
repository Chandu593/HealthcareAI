"""End-to-end Phase 9 (GNN forward pass) integration test -- runs the
actual Phase 1->2->3->4->5->7->8->9 pipeline against synthetic data.
Isolated GNN correctness (shape, empty/single-node graphs, the
differentiated edge_weight design) is covered in
tests/unit/test_patient_gnn.py; this file's job is proving the real
wiring holds: Phase 8's graphs correctly re-joined to Phase 7's node
features by patient ID, per-hospital forward passes correctly
reassembled, full cohort coverage preserved.
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
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestPhase9Integration(unittest.TestCase):
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
        return phase7_result, phase8_result

    def _skip_if_no_torch(self, phase7_result):
        if not phase7_result.get("fused_embeddings_path"):
            self.skipTest("torch not available in this environment -- Phase 7 embeddings not produced")

    def test_output_shape_matches_configured_hidden_dim(self):
        phase7_result, phase8_result = self._run_through_phase8()
        self._skip_if_no_torch(phase7_result)

        phase9_result = run_phase9_gnn_forward_pass(self.cfg, phase7_result, phase8_result)

        data = np.load(phase9_result["gnn_embeddings_path"])
        self.assertEqual(data["embeddings"].shape[1], 32)  # configured gnn.hidden_dim
        self.assertEqual(phase9_result["gnn_dim"], 32)
        self.assertEqual(phase9_result["gnn_type"], "graphsage")

    def test_full_cohort_covered_no_loss_no_duplication(self):
        phase7_result, phase8_result = self._run_through_phase8()
        self._skip_if_no_torch(phase7_result)

        phase9_result = run_phase9_gnn_forward_pass(self.cfg, phase7_result, phase8_result)

        fused_data = np.load(phase7_result["fused_embeddings_path"])
        gnn_data = np.load(phase9_result["gnn_embeddings_path"])

        self.assertEqual(set(gnn_data["patient_ids"].tolist()), set(fused_data["patient_ids"].tolist()))
        self.assertEqual(len(gnn_data["patient_ids"]), len(fused_data["patient_ids"]))  # no duplication either

    def test_works_for_gat_and_gcn_too(self):
        for gnn_type in ("gat", "gcn"):
            with self.subTest(gnn_type=gnn_type):
                self.cfg.raw["gnn"]["type"] = gnn_type
                phase7_result, phase8_result = self._run_through_phase8()
                self._skip_if_no_torch(phase7_result)

                phase9_result = run_phase9_gnn_forward_pass(self.cfg, phase7_result, phase8_result)
                data = np.load(phase9_result["gnn_embeddings_path"])
                self.assertEqual(data["embeddings"].shape[1], 32)
                self.assertFalse(np.isnan(data["embeddings"]).any())

    def test_raises_clearly_when_phase8_manifest_missing(self):
        phase7_result, phase8_result = self._run_through_phase8()
        self._skip_if_no_torch(phase7_result)
        broken_phase8_result = dict(phase8_result)
        broken_phase8_result["manifest_path"] = None

        with self.assertRaises(RuntimeError):
            run_phase9_gnn_forward_pass(self.cfg, phase7_result, broken_phase8_result)


if __name__ == "__main__":
    unittest.main()
