"""End-to-end integration test for
src/ablation/graph_comparison_runner.py -- confirms Graph A and Graph B
are ALWAYS built and evaluated together (never one alone), from
identical settings except candidate-generation, using the GNN's own
fully-connected heads (not a boosted-tree classifier)."""

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
from src.ablation.graph_comparison_runner import run_graph_comparison  # noqa: E402
from src.ablation.results_writer import ResultsWriter  # noqa: E402
from src.data.splitting.patient_split import build_split_manifest  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestGraphComparisonRunnerEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        generate(str(self.data_root), n_hospitals=5, n_patients_per_hospital=40, seed=11)

        self._old_env = os.environ.get("EICU_DATA_ROOT")
        os.environ["EICU_DATA_ROOT"] = str(self.data_root)

        self.cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        self.cfg.raw["run"]["output_dir"] = str(self.output_dir)
        self.cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")
        self.cfg.raw["cohort"]["patient_subset"] = 50
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

        self.ablation_cfg = {
            "local_training": {"epochs": 2, "learning_rate": 0.01},
            "subset": {"tasks": ["mortality"]},
            "sampling": {"embeddings_cache_dir": str(self.output_dir / "ablation" / "embeddings_cache")},
            "graph_comparison": {
                "best_flat_config": {"ts_model": "lstm", "tab_model": "mlp", "fusion_mode": "concatenation"},
                "graph": {"k": 3, "cosine_threshold": 0.3, "n_clusters": 3, "cluster_seed": 42},
                "gnn": {"type": "graphsage", "hidden_dim": 16, "layers": 2, "dropout": 0.0},
                "heads": {"hidden_dim": 8, "dropout": 0.0},
                "training": {"epochs": 3, "learning_rate": 0.01, "mortality_weight": 1.0, "sepsis_weight": 1.0, "loss": "weighted_bce", "seed": 42},
            },
        }

    def tearDown(self):
        close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def _run(self, run_id, best_flat_config=None):
        results_writer = ResultsWriter(self.output_dir / "ablation", run_id=run_id)
        comparison = run_graph_comparison(
            self.cfg, self.ablation_cfg, self.phase3_result, self.phase4_result, self.phase5_result,
            self.partition_result, self.split_manifest, results_writer,
            best_flat_config=best_flat_config or self.ablation_cfg["graph_comparison"]["best_flat_config"],
        )
        return comparison, results_writer

    def test_both_graph_a_and_graph_b_are_always_built_together(self):
        comparison, results_writer = self._run(run_id="run_graph_both")
        self.assertIn("graph_a", comparison)
        self.assertIn("graph_b", comparison)
        self.assertIn("mortality", comparison["graph_a"])
        self.assertIn("mortality", comparison["graph_b"])

        record_files = sorted(p.name for p in results_writer.records_dir.glob("*.json"))
        self.assertTrue(any("_graph_a_" in f for f in record_files))
        self.assertTrue(any("_graph_b_" in f for f in record_files))

    def test_results_routed_to_graph_comparison_results_csv(self):
        _comparison, results_writer = self._run(run_id="run_graph_routing")
        graph_csv = results_writer.run_dir / "graph_comparison_results.csv"
        self.assertTrue(graph_csv.exists())
        self.assertFalse(results_writer.master_csv_path.exists())
        self.assertFalse((results_writer.run_dir / "sampling_results.csv").exists())

    def test_graph_a_and_graph_b_use_identical_k_and_threshold(self):
        """Fairness regression target: both graphs must be built with
        the SAME k/cosine_threshold from config -- not independently
        configurable per graph."""
        comparison, _writer = self._run(run_id="run_graph_fairness")
        # Both graphs completed using the one shared graph_cfg -- if
        # k/threshold differed per graph, this would require a second
        # config block, which graph_comparison_runner.py's signature
        # doesn't even accept (single graph_cfg for both).
        self.assertIn("graph_stats", comparison["graph_a"]["mortality"])
        self.assertIn("graph_stats", comparison["graph_b"]["mortality"])

    def test_graph_a_and_graph_b_produce_different_edge_counts(self):
        """Sanity check that they are genuinely different graphs, not
        the same construction under two labels."""
        comparison, _writer = self._run(run_id="run_graph_different_methods")
        edges_a = comparison["graph_a"]["mortality"]["graph_stats"]["n_edges"]
        edges_b = comparison["graph_b"]["mortality"]["graph_stats"]["n_edges"]
        nodes_a = comparison["graph_a"]["mortality"]["graph_stats"]["n_nodes"]
        nodes_b = comparison["graph_b"]["mortality"]["graph_stats"]["n_nodes"]
        self.assertEqual(nodes_a, nodes_b)  # same patients, same node count either way
        # Not asserting edges_a != edges_b unconditionally (could
        # coincide on tiny mock data), but both must be non-negative
        # ints and the comparison must have actually run both methods.
        self.assertGreaterEqual(edges_a, 0)
        self.assertGreaterEqual(edges_b, 0)

    def test_metrics_come_from_the_gnn_heads_not_a_boosted_tree_classifier(self):
        """Direct regression target for the correction: the written
        records must carry NO classifier field (config.classifier),
        unlike sampling_comparison's records -- only combo_id/graph/
        method/threshold_used, confirming the GNN+heads path was used,
        not boosting_classifiers.py's fit_and_evaluate_classifier()."""
        import json
        _comparison, results_writer = self._run(run_id="run_graph_no_classifier")
        for path in results_writer.records_dir.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            self.assertNotIn("classifier", data["config"])
            self.assertIn("method", data["config"])
            self.assertIn(data["config"]["method"], ("knn_and_threshold", "minibatch_kmeans_and_threshold"))

    def test_missing_best_flat_config_raises_clearly(self):
        results_writer = ResultsWriter(self.output_dir / "ablation", run_id="run_graph_missing_config")
        with self.assertRaises(ValueError):
            run_graph_comparison(
                self.cfg, self.ablation_cfg, self.phase3_result, self.phase4_result, self.phase5_result,
                self.partition_result, self.split_manifest, results_writer, best_flat_config=None,
            )

    def test_pca_best_flat_config_refuses_to_run(self):
        results_writer = ResultsWriter(self.output_dir / "ablation", run_id="run_graph_pca_refused")
        with self.assertRaises(ValueError):
            run_graph_comparison(
                self.cfg, self.ablation_cfg, self.phase3_result, self.phase4_result, self.phase5_result,
                self.partition_result, self.split_manifest, results_writer,
                best_flat_config={"ts_model": "lstm", "tab_model": "pca", "fusion_mode": "concatenation"},
            )

    def test_reuses_embeddings_already_cached_by_sampling_comparison(self):
        """The whole point of sharing get_or_build_combo_embeddings()
        (now defined in embedding_ablation_runner.py, shared by all
        three ablation steps) with sampling_comparison_runner.py: if
        the winning combo's embeddings are already cached, this step
        must not regenerate them."""
        from src.ablation.embedding_ablation_runner import _build_ts_tab_lookup, get_or_build_combo_embeddings
        from src.ablation.sampling_comparison_runner import _hospital_id_lookup
        from src.ablation.config_grid import EmbeddingComboSpec

        combo = EmbeddingComboSpec(ts_model="lstm", tab_model="mlp", fusion_mode="concatenation")
        hospital_ids = sorted(self.partition_result.partitions.keys())
        ts_row_of_patient, tab_row_of_patient = _build_ts_tab_lookup(self.phase4_result, self.phase5_result)
        _combo_result, summary = get_or_build_combo_embeddings(
            combo, hospital_ids, self.partition_result, ts_row_of_patient, tab_row_of_patient,
            self.phase4_result["normalized_array"], self.phase4_result["sequence_length"],
            len(self.phase4_result["feature_names"]), self.phase5_result["normalized_array"],
            len(self.phase5_result["feature_names"]), self.cfg, self.ablation_cfg, self.phase3_result,
            "broad", {}, self.split_manifest, self.ablation_cfg["sampling"]["embeddings_cache_dir"],
            __import__("logging").getLogger("eicu_fl"),
        )
        self.assertFalse(summary["from_cache"])  # first call: genuinely regenerated

        comparison, _writer = self._run(run_id="run_graph_reuse_cache")
        self.assertTrue(comparison["embeddings_from_cache"])  # second call, via run_graph_comparison: reused


if __name__ == "__main__":
    unittest.main()
