import itertools
import unittest

import torch

from src.training.flat_multimodal_model import (
    VALID_FUSION_MODES,
    VALID_TAB_MODELS,
    VALID_TS_MODELS,
    FlatModelConfig,
    FlatMultimodalModel,
)

N_PATIENTS = 12
MAX_BINS = 10
TS_INPUT_DIM = 6
TAB_INPUT_DIM = 15


def _make_batch(tab_input_dim=TAB_INPUT_DIM):
    torch.manual_seed(0)
    timeseries_x = torch.randn(N_PATIENTS, MAX_BINS, TS_INPUT_DIM)
    timeseries_lengths = torch.randint(1, MAX_BINS + 1, (N_PATIENTS,))
    tabular_x = torch.randn(N_PATIENTS, tab_input_dim)
    return timeseries_x, timeseries_lengths, tabular_x


class TestFlatMultimodalModelAllCombos(unittest.TestCase):
    def test_every_ts_x_nonpca_tab_x_fusion_combo_produces_correct_shapes(self):
        non_pca_tab_models = [m for m in VALID_TAB_MODELS if m != "pca"]
        for ts_model, tab_model, fusion_mode in itertools.product(VALID_TS_MODELS, non_pca_tab_models, VALID_FUSION_MODES):
            with self.subTest(ts=ts_model, tab=tab_model, fusion=fusion_mode):
                config = FlatModelConfig(
                    ts_model=ts_model, tab_model=tab_model, fusion_mode=fusion_mode,
                    ts_input_dim=TS_INPUT_DIM, tab_input_dim=TAB_INPUT_DIM, ts_max_bins=MAX_BINS,
                    ts_hidden_dim=16, tab_hidden_dim=16, tab_embedding_dim=8,
                    cross_attention_attn_dim=8, cross_attention_num_heads=2, head_hidden_dim=8,
                )
                model = FlatMultimodalModel(config)
                ts_x, ts_len, tab_x = _make_batch()
                out = model(ts_x, ts_len, tab_x)
                self.assertEqual(out["mortality_logits"].shape, (N_PATIENTS,))
                self.assertEqual(out["sepsis_logits"].shape, (N_PATIENTS,))
                self.assertEqual(out["fused_embedding"].shape, (N_PATIENTS, model.fused_dim))
                if tab_model == "autoencoder":
                    self.assertIsNotNone(out["tabular_reconstruction"])
                    self.assertEqual(out["tabular_reconstruction"].shape, (N_PATIENTS, TAB_INPUT_DIM))
                else:
                    self.assertIsNone(out["tabular_reconstruction"])

    def test_pca_combo_has_no_tab_encoder_submodule_and_passes_through(self):
        pca_embedding_dim = 8
        for ts_model, fusion_mode in itertools.product(VALID_TS_MODELS, VALID_FUSION_MODES):
            with self.subTest(ts=ts_model, fusion=fusion_mode):
                config = FlatModelConfig(
                    ts_model=ts_model, tab_model="pca", fusion_mode=fusion_mode,
                    ts_input_dim=TS_INPUT_DIM, tab_input_dim=pca_embedding_dim, ts_max_bins=MAX_BINS,
                    ts_hidden_dim=16, cross_attention_attn_dim=8, cross_attention_num_heads=2, head_hidden_dim=8,
                )
                model = FlatMultimodalModel(config)
                self.assertIsNone(model.tab_encoder)
                ts_x, ts_len, tab_x = _make_batch(tab_input_dim=pca_embedding_dim)
                out = model(ts_x, ts_len, tab_x)
                self.assertEqual(out["fused_embedding"].shape, (N_PATIENTS, model.fused_dim))
                self.assertIsNone(out["tabular_reconstruction"])

    def test_gradients_flow_to_every_trainable_component(self):
        config = FlatModelConfig(
            ts_model="lstm", tab_model="autoencoder", fusion_mode="cross_attention",
            ts_input_dim=TS_INPUT_DIM, tab_input_dim=TAB_INPUT_DIM, ts_max_bins=MAX_BINS,
            ts_hidden_dim=16, tab_hidden_dim=16, tab_embedding_dim=8,
            cross_attention_attn_dim=8, cross_attention_num_heads=2, head_hidden_dim=8,
        )
        model = FlatMultimodalModel(config)
        ts_x, ts_len, tab_x = _make_batch()
        out = model(ts_x, ts_len, tab_x)
        loss = out["mortality_logits"].sum() + out["sepsis_logits"].sum() + out["tabular_reconstruction"].sum()
        loss.backward()

        for name, submodule in [("ts_encoder", model.ts_encoder), ("tab_encoder", model.tab_encoder),
                                 ("fusion", model.fusion), ("heads", model.heads)]:
            params = list(submodule.parameters())
            self.assertGreater(len(params), 0, f"{name} has no parameters at all")
            n_with_grad = sum(1 for p in params if p.grad is not None and p.grad.norm().item() > 0)
            self.assertGreater(n_with_grad, 0, f"{name} received zero real gradient")

    def test_gradients_flow_even_without_cross_attention_or_autoencoder(self):
        """Concatenation (no params) + mlp (no reconstruction term) --
        confirms the joint loss still reaches ts_encoder/tab_encoder/
        heads even with the two 'extra gradient path' components absent."""
        config = FlatModelConfig(
            ts_model="gru", tab_model="mlp", fusion_mode="concatenation",
            ts_input_dim=TS_INPUT_DIM, tab_input_dim=TAB_INPUT_DIM, ts_max_bins=MAX_BINS,
            ts_hidden_dim=16, tab_hidden_dim=16, tab_embedding_dim=8, head_hidden_dim=8,
        )
        model = FlatMultimodalModel(config)
        self.assertIsNone(model.fusion)
        ts_x, ts_len, tab_x = _make_batch()
        out = model(ts_x, ts_len, tab_x)
        loss = out["mortality_logits"].sum() + out["sepsis_logits"].sum()
        loss.backward()
        for name, submodule in [("ts_encoder", model.ts_encoder), ("tab_encoder", model.tab_encoder), ("heads", model.heads)]:
            n_with_grad = sum(1 for p in submodule.parameters() if p.grad is not None and p.grad.norm().item() > 0)
            self.assertGreater(n_with_grad, 0, f"{name} received zero real gradient")

    def test_invalid_ts_model_raises(self):
        with self.assertRaises(ValueError):
            FlatModelConfig(ts_model="not_real", tab_model="mlp", fusion_mode="concatenation", ts_input_dim=5, tab_input_dim=5)

    def test_invalid_tab_model_raises(self):
        with self.assertRaises(ValueError):
            FlatModelConfig(ts_model="lstm", tab_model="not_real", fusion_mode="concatenation", ts_input_dim=5, tab_input_dim=5)

    def test_invalid_fusion_mode_raises(self):
        with self.assertRaises(ValueError):
            FlatModelConfig(ts_model="lstm", tab_model="mlp", fusion_mode="not_real", ts_input_dim=5, tab_input_dim=5)

    def test_local_patient_model_untouched_by_this_addition(self):
        """Sanity check per 'preserve the locked architecture': Phase
        11's LocalPatientModel must remain importable and structurally
        identical (still hardcoded to LSTM+Autoencoder+GNN+heads) --
        this new module must not have required any change there."""
        from src.training.local_model import LocalPatientModel, LocalModelConfig  # noqa: F401
        import inspect

        sig = inspect.signature(LocalPatientModel.forward)
        self.assertIn("edge_index", sig.parameters)  # still graph-based, untouched


if __name__ == "__main__":
    unittest.main()
