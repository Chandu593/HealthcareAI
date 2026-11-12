"""End-to-end Phase 8 (intra-hospital patient graph construction)
integration test -- runs the actual Phase 1->2->3->4->5->7->8 pipeline
path against synthetic data. Isolated graph-construction correctness
(exact-vs-chunked equivalence, node-identity alignment, edge cases)
is covered in tests/unit/test_patient_graph.py; this file's job is
proving the fix holds through the real per-hospital wiring, most
importantly: NO edges ever cross a hospital boundary.
"""

import json
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
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestPhase8Integration(unittest.TestCase):
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

    def tearDown(self):
        close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def _run_through_phase7(self):
        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)
        phase4_result = run_phase4_timeseries_modality(self.cfg, phase3_result, partition_result)
        phase5_result = run_phase5_tabular_modality(self.cfg, phase1_summary, phase3_result, partition_result)
        phase7_result = run_phase7_multimodal_fusion(self.cfg, phase3_result, phase4_result, phase5_result)
        return partition_result, phase7_result

    def test_one_graph_file_per_hospital(self):
        partition_result, phase7_result = self._run_through_phase7()
        if not phase7_result.get("fused_embeddings_path"):
            self.skipTest("torch not available in this environment -- Phase 7 embeddings not produced")

        phase8_result = run_phase8_patient_graph_construction(self.cfg, phase7_result, partition_result)

        self.assertEqual(phase8_result["n_hospitals"], 4)
        self.assertEqual(len(phase8_result["hospital_graph_paths"]), 4)
        for hid, path in phase8_result["hospital_graph_paths"].items():
            self.assertTrue(Path(path).exists(), f"missing graph file for hospital {hid}")

    def test_no_edges_ever_cross_a_hospital_boundary(self):
        """The single most important property of Phase 8: this is
        checked directly against each hospital's own saved graph file,
        not inferred from the construction code -- a real end-to-end
        proof that the intra-hospital-only architecture held, not just
        an assertion that the code intends it to."""
        partition_result, phase7_result = self._run_through_phase7()
        if not phase7_result.get("fused_embeddings_path"):
            self.skipTest("torch not available in this environment -- Phase 7 embeddings not produced")

        phase8_result = run_phase8_patient_graph_construction(self.cfg, phase7_result, partition_result)

        patient_to_hospital = {}
        for hid, partition in partition_result.partitions.items():
            for pid in partition.patient_ids:
                patient_to_hospital[int(pid)] = hid

        for hid, path in phase8_result["hospital_graph_paths"].items():
            data = np.load(path)
            node_patient_ids = data["patient_ids"]
            for i, j in zip(data["edge_index"][0], data["edge_index"][1]):
                pid_i, pid_j = int(node_patient_ids[i]), int(node_patient_ids[j])
                self.assertEqual(
                    patient_to_hospital[pid_i], patient_to_hospital[pid_j],
                    f"edge between patients {pid_i} (hospital {patient_to_hospital[pid_i]}) and "
                    f"{pid_j} (hospital {patient_to_hospital[pid_j]}) crosses a hospital boundary",
                )
                self.assertEqual(int(hid), patient_to_hospital[pid_i])

    def test_manifest_lists_every_hospital(self):
        partition_result, phase7_result = self._run_through_phase7()
        if not phase7_result.get("fused_embeddings_path"):
            self.skipTest("torch not available in this environment -- Phase 7 embeddings not produced")

        phase8_result = run_phase8_patient_graph_construction(self.cfg, phase7_result, partition_result)

        with open(phase8_result["manifest_path"]) as f:
            manifest = json.load(f)
        self.assertEqual(set(manifest["hospital_graph_paths"].keys()), set(str(h) for h in partition_result.partitions.keys()))

    def test_total_nodes_matches_cohort_size(self):
        partition_result, phase7_result = self._run_through_phase7()
        if not phase7_result.get("fused_embeddings_path"):
            self.skipTest("torch not available in this environment -- Phase 7 embeddings not produced")

        phase8_result = run_phase8_patient_graph_construction(self.cfg, phase7_result, partition_result)

        fused_data = np.load(phase7_result["fused_embeddings_path"])
        self.assertEqual(phase8_result["total_nodes"], len(fused_data["patient_ids"]))


if __name__ == "__main__":
    unittest.main()
