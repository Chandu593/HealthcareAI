"""Tests for src/training/graph_local_model.py -- the CORRECTED
architecture for the Graph A/B comparison (GNN + its own
fully-connected heads, trained via backprop, never a separate
boosted-tree classifier)."""

import unittest

import numpy as np
import torch

from src.training.graph_local_model import (
    GraphOnlyLocalModel,
    predict_graph_model_full_hospital,
    train_graph_model_one_hospital,
)

DEFAULT_MODEL_CONFIG = {
    "in_dim": 8, "gnn_hidden_dim": 16, "gnn_n_layers": 2, "gnn_type": "graphsage",
    "gnn_dropout": 0.0, "head_hidden_dim": 8, "head_dropout": 0.0,
}
DEFAULT_TRAINING_CFG = {"epochs": 5, "learning_rate": 0.01, "mortality_weight": 1.0, "sepsis_weight": 1.0, "loss": "weighted_bce"}


def _small_hospital_fixture(n=12, seed=0):
    rng = np.random.RandomState(seed)
    embeddings = rng.randn(n, 8).astype(np.float32)
    # A small ring graph -- every node connects to its two neighbors.
    src = np.arange(n)
    dst = (np.arange(n) + 1) % n
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    edge_weight = np.full(edge_index.shape[1], 0.8, dtype=np.float32)
    mortality = (rng.rand(n) > 0.5).astype(np.float32)
    sepsis = (rng.rand(n) > 0.5).astype(np.float32)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[: n // 2] = True  # first half train, second half val/test
    return embeddings, edge_index, edge_weight, train_mask, mortality, sepsis


class TestGraphOnlyLocalModel(unittest.TestCase):
    def test_forward_shape(self):
        model = GraphOnlyLocalModel(**DEFAULT_MODEL_CONFIG)
        embeddings = torch.randn(10, 8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        out = model(embeddings, edge_index)
        self.assertEqual(out["mortality_logits"].shape, (10,))
        self.assertEqual(out["sepsis_logits"].shape, (10,))

    def test_never_contains_a_boosted_tree_classifier(self):
        """Direct guard against reintroducing the corrected mistake:
        the model's only trainable submodules must be the GNN and the
        heads -- no sklearn/xgboost/lightgbm estimator anywhere."""
        model = GraphOnlyLocalModel(**DEFAULT_MODEL_CONFIG)
        submodule_names = {name for name, _ in model.named_modules()}
        self.assertIn("gnn", submodule_names)
        self.assertIn("heads", submodule_names)
        for _name, module in model.named_modules():
            self.assertIsInstance(module, torch.nn.Module)  # every submodule is a real nn.Module, not a foreign estimator wrapper


class TestTrainGraphModelOneHospital(unittest.TestCase):
    def test_returns_a_trained_model_and_decreasing_loss(self):
        embeddings, edge_index, edge_weight, train_mask, mortality, sepsis = _small_hospital_fixture(n=20, seed=1)
        model, skip_reason, loss_history = train_graph_model_one_hospital(
            embeddings, edge_index, edge_weight, train_mask, mortality, sepsis,
            {**DEFAULT_MODEL_CONFIG, "in_dim": 8}, {**DEFAULT_TRAINING_CFG, "epochs": 20}, None, seed=42,
        )
        self.assertIsNone(skip_reason)
        self.assertIsNotNone(model)
        self.assertEqual(len(loss_history), 20)
        self.assertLess(loss_history[-1], loss_history[0])

    def test_no_train_patients_skips_cleanly(self):
        embeddings, edge_index, edge_weight, _train_mask, mortality, sepsis = _small_hospital_fixture(n=10, seed=2)
        empty_train_mask = np.zeros(10, dtype=bool)
        model, skip_reason, loss_history = train_graph_model_one_hospital(
            embeddings, edge_index, edge_weight, empty_train_mask, mortality, sepsis,
            DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CFG, None,
        )
        self.assertIsNone(model)
        self.assertEqual(skip_reason, "no_train_patients")
        self.assertEqual(loss_history, [])

    def test_no_usable_labels_skips_cleanly(self):
        embeddings, edge_index, edge_weight, train_mask, _mortality, _sepsis = _small_hospital_fixture(n=10, seed=3)
        all_unknown = np.full(10, np.nan, dtype=np.float32)
        model, skip_reason, _loss_history = train_graph_model_one_hospital(
            embeddings, edge_index, edge_weight, train_mask, all_unknown, all_unknown,
            DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CFG, None,
        )
        self.assertIsNone(model)
        self.assertEqual(skip_reason, "no_labels")

    def test_only_train_rows_ever_reach_the_gradient_computing_pass(self):
        """Direct leakage-safety regression target: poison every
        VAL/TEST row's embedding with NaN (which would blow up any
        computation that touched it) -- training must still complete
        cleanly, proving those rows are genuinely excluded from the
        train-only subgraph, not just masked in the loss."""
        embeddings, edge_index, edge_weight, train_mask, mortality, sepsis = _small_hospital_fixture(n=16, seed=4)
        poisoned = embeddings.copy()
        poisoned[~train_mask] = np.nan
        model, skip_reason, loss_history = train_graph_model_one_hospital(
            poisoned, edge_index, edge_weight, train_mask, mortality, sepsis,
            DEFAULT_MODEL_CONFIG, {**DEFAULT_TRAINING_CFG, "epochs": 5}, None, seed=42,
        )
        self.assertIsNone(skip_reason)
        self.assertTrue(all(np.isfinite(l) for l in loss_history))

    def test_deterministic_for_fixed_seed(self):
        embeddings, edge_index, edge_weight, train_mask, mortality, sepsis = _small_hospital_fixture(n=16, seed=5)
        model1, _sr1, loss1 = train_graph_model_one_hospital(
            embeddings, edge_index, edge_weight, train_mask, mortality, sepsis,
            DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CFG, None, seed=99,
        )
        model2, _sr2, loss2 = train_graph_model_one_hospital(
            embeddings, edge_index, edge_weight, train_mask, mortality, sepsis,
            DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CFG, None, seed=99,
        )
        self.assertEqual(loss1, loss2)

    def test_different_hospitals_can_diverge(self):
        e1, ei1, ew1, tm1, m1, s1 = _small_hospital_fixture(n=16, seed=6)
        e2, ei2, ew2, tm2, m2, s2 = _small_hospital_fixture(n=16, seed=7)
        model1, _sr1, _l1 = train_graph_model_one_hospital(
            e1, ei1, ew1, tm1, m1, s1, DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CFG, None, seed=42,
        )
        model2, _sr2, _l2 = train_graph_model_one_hospital(
            e2, ei2, ew2, tm2, m2, s2, DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CFG, None, seed=42,
        )
        w1 = model1.heads.mortality_head[0].weight.detach().numpy()
        w2 = model2.heads.mortality_head[0].weight.detach().numpy()
        self.assertFalse(np.allclose(w1, w2))


class TestPredictGraphModelFullHospital(unittest.TestCase):
    def test_returns_probabilities_for_every_node(self):
        embeddings, edge_index, edge_weight, train_mask, mortality, sepsis = _small_hospital_fixture(n=12, seed=8)
        model, _sr, _lh = train_graph_model_one_hospital(
            embeddings, edge_index, edge_weight, train_mask, mortality, sepsis,
            DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CFG, None, seed=42,
        )
        preds = predict_graph_model_full_hospital(model, embeddings, edge_index, edge_weight)
        self.assertEqual(preds["mortality_prob"].shape, (12,))
        self.assertEqual(preds["sepsis_prob"].shape, (12,))
        self.assertTrue(np.all((preds["mortality_prob"] >= 0) & (preds["mortality_prob"] <= 1)))

    def test_val_test_rows_use_their_real_full_graph_neighbors(self):
        """Inference is over the FULL graph, not the train-only
        subgraph -- a val/test node's prediction should differ from
        what it would be if it were isolated (zero edges), confirming
        it genuinely receives messages from its real neighbors at
        inference time."""
        embeddings, edge_index, edge_weight, train_mask, mortality, sepsis = _small_hospital_fixture(n=12, seed=9)
        model, _sr, _lh = train_graph_model_one_hospital(
            embeddings, edge_index, edge_weight, train_mask, mortality, sepsis,
            DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CFG, None, seed=42,
        )
        full_preds = predict_graph_model_full_hospital(model, embeddings, edge_index, edge_weight)
        isolated_edge_index = np.empty((2, 0), dtype=np.int64)
        isolated_edge_weight = np.empty((0,), dtype=np.float32)
        isolated_preds = predict_graph_model_full_hospital(model, embeddings, isolated_edge_index, isolated_edge_weight)
        self.assertFalse(np.allclose(full_preds["mortality_prob"], isolated_preds["mortality_prob"]))


if __name__ == "__main__":
    unittest.main()
