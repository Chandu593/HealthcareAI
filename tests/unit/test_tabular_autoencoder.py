"""
Skipped in this sandbox (no torch, no internet) with an explicit
reason, same pattern as tests/unit/test_lstm_encoder.py -- these will
run for real once `pip install torch` succeeds on a machine with
internet access.
"""

import unittest

from src.modalities.tabular.autoencoder import AutoencoderConfig, TORCH_AVAILABLE, TabularAutoencoder

if TORCH_AVAILABLE:
    import torch


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed in this environment -- see module docstring")
class TestTabularAutoencoder(unittest.TestCase):
    def test_output_shapes(self):
        config = AutoencoderConfig(input_dim=20, hidden_dim=16, embedding_dim=8, dropout=0.0)
        model = TabularAutoencoder(config)
        x = torch.randn(5, 20)
        embedding, reconstruction = model(x)
        self.assertEqual(tuple(embedding.shape), (5, 8))
        self.assertEqual(tuple(reconstruction.shape), (5, 20))

    def test_deterministic_in_eval_mode(self):
        config = AutoencoderConfig(input_dim=10, hidden_dim=8, embedding_dim=4, dropout=0.5)
        model = TabularAutoencoder(config)
        model.eval()
        x = torch.randn(1, 10)
        with torch.no_grad():
            emb1, _ = model(x)
            emb2, _ = model(x)
        self.assertTrue(torch.allclose(emb1, emb2))


class TestImportGuardWithoutTorch(unittest.TestCase):
    def test_clear_error_when_torch_unavailable(self):
        if TORCH_AVAILABLE:
            self.skipTest("torch IS installed here -- this test only applies to the no-torch fallback path")
        with self.assertRaises(ImportError):
            TabularAutoencoder(AutoencoderConfig(input_dim=10))


if __name__ == "__main__":
    unittest.main()
