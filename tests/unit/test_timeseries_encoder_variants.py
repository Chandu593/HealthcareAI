"""Tests for the new sibling timeseries encoders (RNN, GRU,
Transformer) added 2026-09-06 alongside LSTMTimeSeriesEncoder. Each is
checked against the SAME contract LSTMTimeSeriesEncoder already
satisfies: forward(x, lengths) -> (batch, hidden_dim), one embedding
per patient, handles variable-length (padded) sequences correctly.
"""

import unittest

import torch

from src.modalities.timeseries.gru_encoder import GRUEncoderConfig, GRUTimeSeriesEncoder
from src.modalities.timeseries.rnn_encoder import RNNEncoderConfig, RNNTimeSeriesEncoder
from src.modalities.timeseries.transformer_encoder import TransformerEncoderConfig, TransformerTimeSeriesEncoder


class _EncoderContractMixin:
    """Shared contract tests, parameterized by encoder/config class via
    subclass attributes -- RNN, GRU, and Transformer all promise the
    exact same input/output shape contract as LSTM, so the same tests
    apply to all three without duplicating them three times."""

    encoder_cls = None
    config_cls = None

    def _make(self, **overrides):
        defaults = dict(input_dim=6, hidden_dim=16)
        defaults.update(overrides)
        torch.manual_seed(0)
        return self.encoder_cls(self.config_cls(**defaults))

    def test_output_shape(self):
        model = self._make()
        x = torch.randn(5, 8, 6)
        lengths = torch.tensor([8, 6, 4, 2, 1])
        out = model(x, lengths)
        self.assertEqual(out.shape, (5, 16))

    def test_single_patient(self):
        model = self._make()
        x = torch.randn(1, 8, 6)
        lengths = torch.tensor([3])
        out = model(x, lengths)
        self.assertEqual(out.shape, (1, 16))

    def test_output_has_no_nan_or_inf(self):
        model = self._make()
        model.eval()
        x = torch.randn(4, 8, 6)
        lengths = torch.tensor([8, 5, 3, 1])
        with torch.no_grad():
            out = model(x, lengths)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_raises_on_zero_input_dim(self):
        with self.assertRaises(ValueError):
            self._make(input_dim=0)

    def test_raises_on_zero_hidden_dim(self):
        with self.assertRaises(ValueError):
            self._make(hidden_dim=0)

    def test_gradient_flows_to_all_parameters(self):
        model = self._make()
        model.train()
        x = torch.randn(4, 8, 6)
        lengths = torch.tensor([8, 5, 3, 1])
        out = model(x, lengths)
        out.sum().backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} received no gradient.")


class TestRNNTimeSeriesEncoder(_EncoderContractMixin, unittest.TestCase):
    encoder_cls = RNNTimeSeriesEncoder
    config_cls = RNNEncoderConfig


class TestGRUTimeSeriesEncoder(_EncoderContractMixin, unittest.TestCase):
    encoder_cls = GRUTimeSeriesEncoder
    config_cls = GRUEncoderConfig


class TestTransformerTimeSeriesEncoder(_EncoderContractMixin, unittest.TestCase):
    encoder_cls = TransformerTimeSeriesEncoder
    config_cls = TransformerEncoderConfig

    def _make(self, **overrides):
        defaults = dict(input_dim=6, hidden_dim=16, num_heads=4, max_bins=8)
        defaults.update(overrides)
        torch.manual_seed(0)
        return self.encoder_cls(self.config_cls(**defaults))

    def test_raises_when_hidden_dim_not_divisible_by_num_heads(self):
        with self.assertRaises(ValueError):
            self._make(hidden_dim=15, num_heads=4)

    def test_raises_when_input_exceeds_configured_max_bins(self):
        model = self._make(max_bins=8)
        x = torch.randn(2, 10, 6)  # 10 > configured max_bins=8
        lengths = torch.tensor([10, 5])
        with self.assertRaises(ValueError):
            model(x, lengths)

    def test_padding_mask_actually_excludes_padded_timesteps(self):
        """Direct regression target: if the padding mask were wired
        backwards (masking REAL timesteps instead of padded ones, or
        not applied at all), changing the PADDED region's content
        would still change the output for a real patient -- it
        shouldn't, since padded values are meaningless filler."""
        model = self._make()
        model.eval()
        x = torch.randn(1, 8, 6)
        lengths = torch.tensor([3])  # only the first 3 timesteps are "real"

        with torch.no_grad():
            out_before = model(x, lengths)

        x_corrupted = x.clone()
        x_corrupted[:, 3:, :] = 999.0  # corrupt only the PADDED region
        with torch.no_grad():
            out_after = model(x_corrupted, lengths)

        torch.testing.assert_close(out_before, out_after)


if __name__ == "__main__":
    unittest.main()
