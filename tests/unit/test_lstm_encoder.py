"""
These tests exercise the actual LSTM forward pass and are SKIPPED (not
silently passed) in any environment without torch installed -- see
src/modalities/timeseries/lstm_encoder.py's module docstring. Run
`pip install torch` (already in requirements.txt) then re-run this file
to get real verification of this component.
"""

import unittest

from src.modalities.timeseries.lstm_encoder import TORCH_AVAILABLE, LSTMEncoderConfig, LSTMTimeSeriesEncoder

if TORCH_AVAILABLE:
    import torch


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed in this environment -- see module docstring")
class TestLSTMTimeSeriesEncoder(unittest.TestCase):
    def test_output_shape(self):
        config = LSTMEncoderConfig(input_dim=5, hidden_dim=16, num_layers=1, dropout=0.0)
        encoder = LSTMTimeSeriesEncoder(config)
        x = torch.randn(4, 10, 5)  # batch=4, max_bins=10, features=5
        lengths = torch.tensor([10, 5, 1, 8])
        out = encoder(x, lengths)
        self.assertEqual(tuple(out.shape), (4, 16))

    def test_padding_does_not_affect_output(self):
        """Two sequences identical up to their real length, differing
        only in the padded region, should produce the same embedding --
        this is the whole point of pack_padded_sequence."""
        config = LSTMEncoderConfig(input_dim=3, hidden_dim=8, num_layers=1, dropout=0.0)
        encoder = LSTMTimeSeriesEncoder(config)
        encoder.eval()

        torch.manual_seed(0)
        base = torch.randn(1, 4, 3)
        x1 = base.clone()
        x2 = base.clone()
        x2[0, 3, :] = 999.0  # corrupt only the padding region (length=3)
        lengths = torch.tensor([3])

        with torch.no_grad():
            out1 = encoder(x1, lengths)
            out2 = encoder(x2, lengths)

        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))


class TestImportGuardWithoutTorch(unittest.TestCase):
    def test_clear_error_when_torch_unavailable(self):
        if TORCH_AVAILABLE:
            self.skipTest("torch IS installed here -- this test only applies to the no-torch fallback path")
        with self.assertRaises(ImportError):
            LSTMTimeSeriesEncoder(LSTMEncoderConfig(input_dim=5))


if __name__ == "__main__":
    unittest.main()
