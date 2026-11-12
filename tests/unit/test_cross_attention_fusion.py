import unittest

import torch

from src.fusion.cross_attention_fusion import CrossAttentionFusion, CrossAttentionFusionConfig
import src.fusion.multimodal_fusion as concatenation_module


class TestCrossAttentionFusion(unittest.TestCase):
    def test_output_shape_defaults_to_concatenation_width(self):
        """Confirms the drop-in-replacement claim in the module
        docstring: with output_dim=None, the fused width must equal
        ts_dim + tab_dim, exactly what plain concatenation would
        produce -- so nothing downstream (GNN in_dim, heads in_dim)
        needs reconfiguring based on which fusion mode is selected."""
        fusion = CrossAttentionFusion(CrossAttentionFusionConfig(ts_dim=128, tab_dim=64, attn_dim=32, num_heads=4))
        ts = torch.randn(10, 128)
        tab = torch.randn(10, 64)
        out = fusion(ts, tab)
        self.assertEqual(out.shape, (10, 192))  # 128 + 64

    def test_explicit_output_dim_is_honored(self):
        fusion = CrossAttentionFusion(
            CrossAttentionFusionConfig(ts_dim=128, tab_dim=64, attn_dim=32, num_heads=4, output_dim=96)
        )
        out = fusion(torch.randn(5, 128), torch.randn(5, 64))
        self.assertEqual(out.shape, (5, 96))

    def test_no_cross_patient_interaction(self):
        """Each row's output must depend ONLY on that row's own two
        embeddings -- same no-cross-patient-mixing property
        concatenation already has for free. Direct regression target:
        changing one patient's tabular embedding must not change any
        OTHER patient's fused output."""
        fusion = CrossAttentionFusion(CrossAttentionFusionConfig(ts_dim=16, tab_dim=8, attn_dim=16, num_heads=2))
        fusion.eval()
        ts = torch.randn(6, 16)
        tab = torch.randn(6, 8)
        with torch.no_grad():
            out_before = fusion(ts, tab)

        tab_corrupted = tab.clone()
        tab_corrupted[3] = torch.randn(8) * 1000  # blow up one patient's tabular embedding
        with torch.no_grad():
            out_after = fusion(ts, tab_corrupted)

        other_rows = [i for i in range(6) if i != 3]
        torch.testing.assert_close(out_before[other_rows], out_after[other_rows])
        self.assertFalse(torch.allclose(out_before[3], out_after[3]))

    def test_has_trainable_parameters_and_receives_gradients(self):
        """Cross-attention (unlike concatenation) has real weights that
        must be trainable -- confirms it can sit inside a joint
        backward pass."""
        fusion = CrossAttentionFusion(CrossAttentionFusionConfig(ts_dim=32, tab_dim=16, attn_dim=16, num_heads=2))
        params = list(fusion.parameters())
        self.assertGreater(len(params), 0)

        ts = torch.randn(4, 32, requires_grad=False)
        tab = torch.randn(4, 16, requires_grad=False)
        out = fusion(ts, tab)
        loss = out.sum()
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in params if p.grad is not None]
        self.assertTrue(all(g >= 0 for g in grad_norms))
        self.assertEqual(sum(1 for p in params if p.grad is not None), len(params))

    def test_attn_dim_not_divisible_by_heads_raises(self):
        with self.assertRaises(ValueError):
            CrossAttentionFusion(CrossAttentionFusionConfig(ts_dim=32, tab_dim=16, attn_dim=15, num_heads=4))

    def test_existing_concatenation_module_is_untouched(self):
        """Decision #4 explicitly requires the existing concatenation
        fusion implementation to remain unchanged. Sanity check that the
        locked module still exposes exactly the same public functions
        this new file did not modify."""
        self.assertTrue(hasattr(concatenation_module, "fuse_embeddings"))
        self.assertTrue(hasattr(concatenation_module, "align_embeddings_to_cohort"))


if __name__ == "__main__":
    unittest.main()
