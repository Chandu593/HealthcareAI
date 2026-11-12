"""Tests for src/gnn/patient_gnn.py (Phase 9).

Covers: correct output shapes for all three architectures, the
empty-graph and single-node edge cases (confirmed to occur in the real
data -- at least one real hospital has zero patients), and the
differentiated edge_weight design decision -- GCN must genuinely use
edge_weight (output changes when it changes), GraphSAGE/GAT must
genuinely ignore it (output is identical regardless), not silently do
the opposite of what was designed and documented.
"""

import unittest

import torch

from src.gnn.patient_gnn import PatientGNN


def _fixed_seed_model(gnn_type: str, in_dim: int = 16, hidden_dim: int = 8, n_layers: int = 2, **kwargs) -> PatientGNN:
    torch.manual_seed(1)
    model = PatientGNN(in_dim, hidden_dim, n_layers=n_layers, gnn_type=gnn_type, **kwargs)
    model.eval()
    return model


class TestPatientGNNConstruction(unittest.TestCase):
    def test_raises_on_unknown_gnn_type(self):
        with self.assertRaises(ValueError):
            PatientGNN(16, 8, gnn_type="not_a_real_type")

    def test_raises_on_zero_layers(self):
        with self.assertRaises(ValueError):
            PatientGNN(16, 8, n_layers=0, gnn_type="graphsage")

    def test_raises_when_hidden_dim_not_divisible_by_gat_heads(self):
        with self.assertRaises(ValueError):
            PatientGNN(16, 10, gnn_type="gat", gat_heads=4)  # 10 % 4 != 0


class TestPatientGNNForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.n, self.in_dim, self.hidden_dim = 12, 16, 8
        self.x = torch.randn(self.n, self.in_dim)
        self.edge_index = torch.randint(0, self.n, (2, 24))
        self.edge_weight = torch.rand(24)

    def test_output_shape_correct_for_every_architecture(self):
        for gnn_type in ("graphsage", "gat", "gcn"):
            with self.subTest(gnn_type=gnn_type):
                model = _fixed_seed_model(gnn_type, self.in_dim, self.hidden_dim)
                with torch.no_grad():
                    out = model(self.x, self.edge_index, edge_weight=self.edge_weight)
                self.assertEqual(out.shape, (self.n, self.hidden_dim))

    def test_output_has_no_nan_or_inf(self):
        for gnn_type in ("graphsage", "gat", "gcn"):
            with self.subTest(gnn_type=gnn_type):
                model = _fixed_seed_model(gnn_type, self.in_dim, self.hidden_dim)
                with torch.no_grad():
                    out = model(self.x, self.edge_index, edge_weight=self.edge_weight)
                self.assertFalse(torch.isnan(out).any())
                self.assertFalse(torch.isinf(out).any())

    def test_empty_graph_does_not_crash(self):
        """Confirmed to occur in real data -- at least one real
        hospital has zero patients in the final cohort (see Phase 8's
        real-data verification, IMPLEMENTATION_TRACKER.md)."""
        x_empty = torch.empty((0, self.in_dim))
        edge_index_empty = torch.empty((2, 0), dtype=torch.long)
        for gnn_type in ("graphsage", "gat", "gcn"):
            with self.subTest(gnn_type=gnn_type):
                model = _fixed_seed_model(gnn_type, self.in_dim, self.hidden_dim)
                with torch.no_grad():
                    out = model(x_empty, edge_index_empty)
                self.assertEqual(out.shape, (0, self.hidden_dim))

    def test_single_node_zero_edges_does_not_crash(self):
        x_single = torch.randn(1, self.in_dim)
        edge_index_single = torch.empty((2, 0), dtype=torch.long)
        for gnn_type in ("graphsage", "gat", "gcn"):
            with self.subTest(gnn_type=gnn_type):
                model = _fixed_seed_model(gnn_type, self.in_dim, self.hidden_dim)
                with torch.no_grad():
                    out = model(x_single, edge_index_single)
                self.assertEqual(out.shape, (1, self.hidden_dim))

    def test_variable_n_layers_respected(self):
        for n_layers in (1, 3, 4):
            with self.subTest(n_layers=n_layers):
                model = _fixed_seed_model("graphsage", self.in_dim, self.hidden_dim, n_layers=n_layers)
                self.assertEqual(len(model.layers), n_layers)
                with torch.no_grad():
                    out = model(self.x, self.edge_index)
                self.assertEqual(out.shape, (self.n, self.hidden_dim))


class TestEdgeWeightUsageIsDifferentiatedByArchitecture(unittest.TestCase):
    """Direct regression target for the design decision documented in
    patient_gnn.py's module docstring: GCN must genuinely use
    edge_weight; GraphSAGE/GAT must genuinely ignore it. Silently
    reversing either half of this (e.g. GCN ignoring edge_weight, or
    GraphSAGE accidentally being sensitive to it through some
    unintended path) would defeat the entire point of the
    differentiated design."""

    def setUp(self):
        torch.manual_seed(0)
        self.n, self.in_dim, self.hidden_dim = 10, 16, 8
        self.x = torch.randn(self.n, self.in_dim)
        self.edge_index = torch.randint(0, self.n, (2, 20))
        self.weight_a = torch.ones(20)
        self.weight_b = torch.rand(20) * 5  # deliberately very different from weight_a

    def test_gcn_output_changes_with_edge_weight(self):
        model = _fixed_seed_model("gcn", self.in_dim, self.hidden_dim)
        with torch.no_grad():
            out_a = model(self.x, self.edge_index, edge_weight=self.weight_a)
            out_b = model(self.x, self.edge_index, edge_weight=self.weight_b)
        self.assertFalse(
            torch.allclose(out_a, out_b),
            "GCN's output did not change when edge_weight changed -- edge_weight is not being used.",
        )

    def test_graphsage_output_unaffected_by_edge_weight(self):
        model = _fixed_seed_model("graphsage", self.in_dim, self.hidden_dim)
        with torch.no_grad():
            out_a = model(self.x, self.edge_index, edge_weight=self.weight_a)
            out_b = model(self.x, self.edge_index, edge_weight=self.weight_b)
        self.assertTrue(
            torch.allclose(out_a, out_b),
            "GraphSAGE's output changed with edge_weight -- it should be standard/unweighted, per design.",
        )

    def test_gat_output_unaffected_by_edge_weight(self):
        model = _fixed_seed_model("gat", self.in_dim, self.hidden_dim)
        with torch.no_grad():
            out_a = model(self.x, self.edge_index, edge_weight=self.weight_a)
            out_b = model(self.x, self.edge_index, edge_weight=self.weight_b)
        self.assertTrue(
            torch.allclose(out_a, out_b),
            "GAT's output changed with edge_weight -- it should use standard self-attention only, per design.",
        )


if __name__ == "__main__":
    unittest.main()
