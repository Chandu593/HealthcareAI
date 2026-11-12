"""End-to-end integration test for
src/ablation/graph_embedding_ablation_runner.py against mock data."""

import json
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
from src.ablation.graph_embedding_ablation_runner import run_graph_embedding_ablation  # noqa: E402
from src.ablation.results_writer import ResultsWriter  # noqa: E402
from src.data.splitting.patient_split import build_split_manifest  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestGraphEmbeddingAblationRunnerEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        generate(str(self.data_root), n_hospitals=4, n_patients_per_hospital=50, seed=21)

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
        self.phase4_result = run_phase4_timeseries_modality(self.cfg, self.phase3_result, self.partition_result)
        self.phase5_result = run_phase5_tabular_modality(self.cfg, phase1_summary, self.phase3_result, self.partition_result)

        self.split_manifest = build_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], seed=42, stratify_on="mortality",
        )

    def tearDown(self):
        close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def test_graph_ablation_runs_end_to_end_and_writes_results(self):
        ablation_cfg = {
            "local_training": {"epochs": 2, "learning_rate": 0.01},
            "graph_classifiers": ["adaboost", "xgboost"],
            "subset": {"tasks": ["mortality"]},
        }
        results_writer = ResultsWriter(self.output_dir / "ablation", run_id="run_graph")
        summary = run_graph_embedding_ablation(
            self.cfg, ablation_cfg, self.phase3_result, self.phase4_result, self.phase5_result,
            self.partition_result, self.split_manifest, results_writer,
        )
        self.assertFalse(summary.get("skipped_entirely", False))
        self.assertEqual(summary["n_experiments"], 2)  # 2 classifiers x 1 task
        self.assertGreater(summary["training_info"]["n_hospitals_trained"], 0)

        record_files = sorted(results_writer.records_dir.glob("*.json"))
        self.assertEqual(len(record_files), 2)
        for path in record_files:
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["experiment_kind"], "graph_ablation")
            self.assertIn("graph-embedding", data["experiment_id"])
            self.assertIn("train", data["metrics"])

    def test_different_gnn_type_is_a_free_config_override_not_a_new_grid_axis(self):
        """Decision #6: GNN type is NOT multiplied into the grid -- but
        the runner still accepts it as a simple override so a person
        CAN compare gnn_type values one at a time without a code
        change, just not as an automatic cross product."""
        ablation_cfg = {
            "local_training": {"epochs": 1, "learning_rate": 0.01},
            "graph_classifiers": ["adaboost"], "subset": {"tasks": ["mortality"]},
        }
        results_writer = ResultsWriter(self.output_dir / "ablation", run_id="run_graph_gcn")
        summary = run_graph_embedding_ablation(
            self.cfg, ablation_cfg, self.phase3_result, self.phase4_result, self.phase5_result,
            self.partition_result, self.split_manifest, results_writer, gnn_type="gcn",
        )
        self.assertEqual(summary["training_info"]["gnn_type"], "gcn")


    def test_gnn_training_is_not_sensitive_to_a_non_train_patients_features(self):
        """Direct leakage-fix confirmation for THIS runner's GNN stage
        (reuses run_pipeline.py's `_build_train_only_subgraph()`, see
        that function's docstring for the full design-correction
        history): monkeypatching `build_hospital_graph` to inject one
        deliberately huge edge_weight anomaly for a non-train node must
        not change the trained GNN+heads weights, since that node is
        never part of the training subgraph."""
        import numpy as np
        import torch
        from src.data.splitting.patient_split import build_split_manifest
        from src.ablation.embedding_ablation_runner import _build_ts_tab_lookup, _load_hospital_raw_tensors
        from src.training.flat_multimodal_model import FlatModelConfig
        from src.ablation.embedding_ablation_runner import _train_one_hospital_flat_model
        from src.ablation.config_grid import EmbeddingComboSpec
        from src.graph.patient_graph import build_hospital_graph
        from src.gnn.patient_gnn import PatientGNN
        from src.heads.multitask_heads import MultiTaskHeads, align_labels_to_patient_ids
        from src.training.losses import compute_multitask_loss
        import run_pipeline as rp

        split_manifest = build_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], seed=42, stratify_on="mortality",
        )
        # Pick a hospital with both train and non-train patients.
        hid = None
        for candidate_hid, p in self.partition_result.partitions.items():
            pids = [int(x) for x in p.patient_ids]
            splits = {split_manifest.split_of(pid) for pid in pids}
            if "train" in splits and len(splits) > 1:
                hid = candidate_hid
                break
        self.assertIsNotNone(hid, "test fixture needs a hospital with a mixed train/non-train split")

        ts_row_of_patient, tab_row_of_patient = _build_ts_tab_lookup(self.phase4_result, self.phase5_result)
        ts_array = self.phase4_result["normalized_array"]
        ts_seq_length = self.phase4_result["sequence_length"]
        tab_array = self.phase5_result["normalized_array"]
        partition = self.partition_result.partitions[hid]
        hospital_patient_ids = sorted(int(pid) for pid in partition.patient_ids)
        timeseries_x, timeseries_lengths, tabular_x_raw = _load_hospital_raw_tensors(
            hospital_patient_ids, ts_row_of_patient, ts_array, ts_seq_length, tab_row_of_patient, tab_array,
        )
        label_info = align_labels_to_patient_ids(
            hospital_patient_ids, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], "broad",
        )
        mortality_labels = torch.tensor(label_info["mortality"], dtype=torch.float32)
        sepsis_labels = torch.tensor(label_info["sepsis"], dtype=torch.float32)
        split_of_row = np.array([split_manifest.split_of(pid) or "train" for pid in hospital_patient_ids])
        train_row_mask = split_of_row == "train"
        non_train_idx = np.where(~train_row_mask)[0][0]

        combo = EmbeddingComboSpec(ts_model="lstm", tab_model="autoencoder", fusion_mode="concatenation")
        model_config = FlatModelConfig(
            ts_model="lstm", tab_model="autoencoder", fusion_mode="concatenation",
            ts_input_dim=timeseries_x.shape[2], tab_input_dim=tabular_x_raw.shape[1], ts_max_bins=timeseries_x.shape[1],
            ts_hidden_dim=16, tab_hidden_dim=16, tab_embedding_dim=8, head_hidden_dim=8,
        )
        tabular_x = torch.tensor(tabular_x_raw, dtype=torch.float32)
        torch.manual_seed(0)
        flat_model, skip_reason, _ = _train_one_hospital_flat_model(
            combo, timeseries_x, timeseries_lengths, tabular_x, train_row_mask, mortality_labels, sepsis_labels,
            model_config, {"epochs": 2, "learning_rate": 0.01}, __import__("logging").getLogger("eicu_fl"), hid,
        )
        self.assertIsNone(skip_reason)
        flat_model.eval()
        with torch.no_grad():
            fused, _ = flat_model.encode(timeseries_x, timeseries_lengths, tabular_x)
        fused_np = fused.numpy()

        graph = build_hospital_graph(hospital_patient_ids, fused_np, method="knn", k=3)
        edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
        edge_weight = torch.tensor(graph["edge_weight"], dtype=torch.float32)

        def _train_gnn(x_np):
            x_full = torch.tensor(x_np, dtype=torch.float32)
            train_edge_index, train_edge_weight, train_indices = rp._build_train_only_subgraph(
                edge_index, edge_weight, torch.tensor(train_row_mask),
            )
            x_train = x_full[train_indices]
            mort_train = mortality_labels[train_indices]
            sepsis_train = sepsis_labels[train_indices]
            torch.manual_seed(1)
            gnn = PatientGNN(in_dim=x_np.shape[1], hidden_dim=16, n_layers=2, gnn_type="graphsage", dropout=0.0)
            heads = MultiTaskHeads(16, head_hidden_dim=8, dropout=0.0)
            optimizer = torch.optim.Adam(list(gnn.parameters()) + list(heads.parameters()), lr=0.01)
            for _ in range(5):
                optimizer.zero_grad()
                out = gnn(x_train, train_edge_index, train_edge_weight)
                head_out = heads(out)
                loss, _, _ = compute_multitask_loss(
                    head_out["mortality_logits"], head_out["sepsis_logits"], mort_train, sepsis_train,
                    1.0, 1.0, "weighted_bce",
                )
                loss.backward()
                optimizer.step()
            return gnn.state_dict()

        clean_state = _train_gnn(fused_np)
        corrupted_np = fused_np.copy()
        corrupted_np[non_train_idx] = corrupted_np[non_train_idx] + 1000.0
        corrupted_state = _train_gnn(corrupted_np)

        for k in clean_state:
            torch.testing.assert_close(
                clean_state[k], corrupted_state[k],
                msg=f"GNN parameter {k} changed after corrupting a non-train node's fused embedding.",
            )


if __name__ == "__main__":
    unittest.main()
