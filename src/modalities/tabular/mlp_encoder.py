"""
Phase 5 -- Tabular MLP encoder, added 2026-09-06 as a sibling to
TabularAutoencoder (autoencoder.py, untouched by this addition) so
`modalities.tabular.model_type` can select between architectures. See
docs/tabular_modality.md for the full design writeup.

A plain feedforward projection (input_dim -> hidden_dim ->
embedding_dim) with NO decoder and NO reconstruction objective --
unlike TabularAutoencoder, which also builds a decoder specifically
for a reconstruction-loss training objective (see autoencoder.py's own
docstring, and src/training/local_model.py's use of it). This is for
users who want a pure feature-embedding projection without committing
to the unsupervised reconstruction-based approach at all.

Shares AutoencoderConfig's exact field names (input_dim, hidden_dim,
embedding_dim, dropout) for direct side-by-side comparability, even
though this class doesn't use a separate config dataclass with a
decoder-specific field the way AutoencoderConfig might.

forward() returns the embedding ONLY (a single tensor), NOT a
(embedding, reconstruction) tuple the way TabularAutoencoder does --
there's nothing to reconstruct. Callers that branch on architecture
(see run_phase5_tabular_modality()) must account for this difference.
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
class MLPEncoderConfig:
    input_dim: int
    hidden_dim: int = 128
    embedding_dim: int = 64
    dropout: float = 0.1


if TORCH_AVAILABLE:

    class TabularMLPEncoder(nn.Module):
        def __init__(self, config: MLPEncoderConfig):
            super().__init__()
            if config.input_dim < 1:
                raise ValueError(f"input_dim must be >= 1, got {config.input_dim}.")
            if config.hidden_dim < 1:
                raise ValueError(f"hidden_dim must be >= 1, got {config.hidden_dim}.")
            if config.embedding_dim < 1:
                raise ValueError(f"embedding_dim must be >= 1, got {config.embedding_dim}.")
            self.config = config
            self.encoder = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.embedding_dim),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """x: (N, input_dim) -> (N, embedding_dim). A single
            tensor, NOT a tuple -- see module docstring."""
            return self.encoder(x)

else:

    class TabularMLPEncoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "torch is not installed -- install it (`pip install -r requirements.txt` or "
                "`pip install torch`) to use TabularMLPEncoder."
            )
