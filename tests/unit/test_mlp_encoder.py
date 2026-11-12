"""Tests for TabularMLPEncoder (src/modalities/tabular/mlp_encoder.py),
added 2026-09-06 as a sibling to TabularAutoencoder."""

import unittest

import torch

from src.modalities.tabular.mlp_encoder import MLPEncoderConfig, TabularMLPEncoder


def _make(**overrides):
    defaults = dict(input_dim=10, hidden_dim=16, embedding_dim=8, dropout=0.0)
    defaults.update(overrides)
    torch.manual_seed(0)
    return TabularMLPEncoder(MLPEncoderConfig(**defaults))


class TestTabularMLPEncoder(unittest.TestCase):
    def test_output_shape(self):
        model = _make()
        x = torch.randn(5, 10)
        out = model(x)
        self.assertEqual(out.shape, (5, 8))

    def test_returns_a_single_tensor_not_a_tuple(self):
        """Direct contract check: unlike TabularAutoencoder's
        (embedding, reconstruction) tuple, this returns embedding
        alone -- callers that branch on architecture must account for
        this (see run_phase5_tabular_modality())."""
        model = _make()
        out = model(torch.randn(3, 10))
        self.assertIsInstance(out, torch.Tensor)

    def test_single_patient(self):
        model = _make()
        out = model(torch.randn(1, 10))
        self.assertEqual(out.shape, (1, 8))

    def test_empty_input_does_not_crash(self):
        model = _make()
        out = model(torch.empty((0, 10)))
        self.assertEqual(out.shape, (0, 8))

    def test_output_has_no_nan_or_inf(self):
        model = _make()
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, 10))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_raises_on_zero_input_dim(self):
        with self.assertRaises(ValueError):
            _make(input_dim=0)

    def test_raises_on_zero_hidden_dim(self):
        with self.assertRaises(ValueError):
            _make(hidden_dim=0)

    def test_raises_on_zero_embedding_dim(self):
        with self.assertRaises(ValueError):
            _make(embedding_dim=0)

    def test_gradient_flows_to_all_parameters(self):
        model = _make()
        model.train()
        out = model(torch.randn(4, 10))
        out.sum().backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} received no gradient.")


if __name__ == "__main__":
    unittest.main()
