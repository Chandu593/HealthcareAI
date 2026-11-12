import unittest

from src.ablation.config_grid import (
    expand_embedding_combo_grid,
    expand_embedding_experiment_grid,
    expand_graph_experiment_grid,
    validate_ablation_config,
)

FULL_ABLATION_CFG = {
    "ts_models": ["lstm", "rnn", "gru", "transformer"],
    "tab_models": ["autoencoder", "mlp", "pca"],
    "fusion_modes": ["concatenation", "cross_attention"],
    "classifiers": ["adaboost", "xgboost", "lightgbm"],
    "graph_classifiers": ["adaboost", "xgboost", "lightgbm"],
}


class TestExpandGrid(unittest.TestCase):
    def test_full_default_grid_has_72_combos(self):
        combos = expand_embedding_combo_grid(FULL_ABLATION_CFG)
        self.assertEqual(len(combos), 4 * 3 * 2)

    def test_full_experiment_grid_has_144_experiments_before_tasks_wait_correct_math(self):
        experiments = expand_embedding_experiment_grid(FULL_ABLATION_CFG)
        # 24 combos x 3 classifiers x 2 tasks = 144
        self.assertEqual(len(experiments), 24 * 3 * 2)

    def test_graph_grid_has_6_experiments(self):
        experiments = expand_graph_experiment_grid(FULL_ABLATION_CFG)
        self.assertEqual(len(experiments), 3 * 2)

    def test_all_experiment_ids_are_unique(self):
        experiments = expand_embedding_experiment_grid(FULL_ABLATION_CFG)
        ids = [e.experiment_id for e in experiments]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_graph_experiment_ids_are_unique(self):
        experiments = expand_graph_experiment_grid(FULL_ABLATION_CFG)
        ids = [e.experiment_id for e in experiments]
        self.assertEqual(len(ids), len(set(ids)))

    def test_explicit_combo_subset_overrides_full_cross_product(self):
        cfg = dict(FULL_ABLATION_CFG, subset={
            "combos": [{"ts_model": "lstm", "tab_model": "mlp", "fusion_mode": "concatenation"}],
        })
        combos = expand_embedding_combo_grid(cfg)
        self.assertEqual(len(combos), 1)
        self.assertEqual(combos[0].ts_model, "lstm")

    def test_explicit_classifier_and_task_subset(self):
        cfg = dict(FULL_ABLATION_CFG, subset={"classifiers": ["xgboost"], "tasks": ["mortality"]})
        experiments = expand_embedding_experiment_grid(cfg)
        self.assertEqual(len(experiments), 24 * 1 * 1)
        self.assertTrue(all(e.classifier == "xgboost" and e.task == "mortality" for e in experiments))

    def test_validate_rejects_unknown_ts_model(self):
        cfg = dict(FULL_ABLATION_CFG, ts_models=["lstm", "not_a_real_model"])
        with self.assertRaises(ValueError):
            validate_ablation_config(cfg)

    def test_validate_rejects_unknown_classifier(self):
        cfg = dict(FULL_ABLATION_CFG, classifiers=["adaboost", "randomforest"])
        with self.assertRaises(ValueError):
            validate_ablation_config(cfg)

    def test_validate_accepts_the_full_default_config(self):
        validate_ablation_config(FULL_ABLATION_CFG)  # must not raise

    def test_combo_id_distinguishes_before_and_after_ca(self):
        combos = expand_embedding_combo_grid(FULL_ABLATION_CFG)
        before = next(c for c in combos if c.ts_model == "lstm" and c.tab_model == "mlp" and c.fusion_mode == "concatenation")
        after = next(c for c in combos if c.ts_model == "lstm" and c.tab_model == "mlp" and c.fusion_mode == "cross_attention")
        self.assertIn("beforeca", before.combo_id)
        self.assertIn("afterca", after.combo_id)
        self.assertNotEqual(before.combo_id, after.combo_id)


if __name__ == "__main__":
    unittest.main()
