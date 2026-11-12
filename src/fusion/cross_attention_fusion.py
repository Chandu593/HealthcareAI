"""
Ablation framework -- Cross-Attention fusion, added 2026-09-10 as a
SIBLING to Phase 7's locked concatenation fusion (multimodal_fusion.py,
completely untouched by this addition -- same "new file, existing file
never modified" discipline as the RNN/GRU/Transformer/MLP encoder
additions on 2026-09-06).

Confirmed with the user (ablation-framework decision #4): Cross-
Attention is another FUSION MECHANISM alongside plain concatenation,
selectable via `ablation.fusion_modes` / a flat-model's `fusion_type`.
Unlike concatenation (zero parameters, nothing to train), this module
has real trainable weights (Q/K/V projections) -- it is trained
JOINTLY with whichever TS/TAB encoders feed it, inside the SAME local,
per-hospital training loop (src/training/flat_multimodal_model.py), not
frozen and not federated (per decision #2: the 72-config ablation is
entirely pre-federated).

DESIGN: each patient's two modality embeddings (time-series, tabular)
are treated as a 2-token sequence. Both tokens attend to both tokens
(nn.MultiheadAttention self-attention over the 2-token sequence) --
this is the natural way to let the TS embedding query the TAB
embedding and vice versa without picking an arbitrary "which modality
queries which" direction. A residual + LayerNorm (standard Transformer-
block practice) stabilizes training since attn_dim is typically smaller
than either input embedding's own width. The two attended tokens are
then flattened and projected to `output_dim`, which DEFAULTS to
ts_dim + tab_dim (concatenation's own width) specifically so
Cross-Attention is a drop-in alternative -- nothing downstream (a GNN's
`in_dim`, or MultiTaskHeads' `in_dim`) needs to be reconfigured
differently depending on which fusion mode was selected.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # placeholder so the module still imports without torch


@dataclass
class CrossAttentionFusionConfig:
    ts_dim: int
    tab_dim: int
    attn_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.1
    output_dim: "int | None" = None  # None -> defaults to ts_dim + tab_dim (see module docstring)


if TORCH_AVAILABLE:

    class CrossAttentionFusion(nn.Module):
        def __init__(self, config: CrossAttentionFusionConfig):
            super().__init__()
            if config.ts_dim < 1 or config.tab_dim < 1:
                raise ValueError(f"ts_dim and tab_dim must both be >= 1, got {config.ts_dim}, {config.tab_dim}.")
            if config.attn_dim % config.num_heads != 0:
                raise ValueError(
                    f"attn_dim ({config.attn_dim}) must be divisible by num_heads ({config.num_heads})."
                )
            self.config = config
            self.output_dim = config.output_dim if config.output_dim is not None else (config.ts_dim + config.tab_dim)

            self.ts_proj = nn.Linear(config.ts_dim, config.attn_dim)
            self.tab_proj = nn.Linear(config.tab_dim, config.attn_dim)
            self.attn = nn.MultiheadAttention(
                embed_dim=config.attn_dim, num_heads=config.num_heads, dropout=config.dropout, batch_first=True,
            )
            self.norm = nn.LayerNorm(config.attn_dim)
            self.out_proj = nn.Linear(config.attn_dim * 2, self.output_dim)

        def forward(self, ts_embedding: "torch.Tensor", tab_embedding: "torch.Tensor") -> "torch.Tensor":
            """ts_embedding: (N, ts_dim), tab_embedding: (N, tab_dim) -> (N, output_dim).
            No cross-PATIENT interaction at all (each row is independent,
            same as concatenation) -- only cross-MODALITY attention
            within a single patient's own two embeddings."""
            ts_p = self.ts_proj(ts_embedding).unsqueeze(1)   # (N, 1, attn_dim)
            tab_p = self.tab_proj(tab_embedding).unsqueeze(1)  # (N, 1, attn_dim)
            tokens = torch.cat([ts_p, tab_p], dim=1)          # (N, 2, attn_dim)
            attended, _ = self.attn(tokens, tokens, tokens)
            attended = self.norm(attended + tokens)           # residual, standard transformer-block practice
            flat = attended.reshape(attended.shape[0], -1)    # (N, 2 * attn_dim)
            return self.out_proj(flat)

else:

    class CrossAttentionFusion:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "torch is not installed -- install it (`pip install -r requirements.txt` or "
                "`pip install torch`) to use CrossAttentionFusion."
            )
