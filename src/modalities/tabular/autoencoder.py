"""
Phase 5c -- Tabular autoencoder.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object


@dataclass
class AutoencoderConfig:
    input_dim: int
    hidden_dim: int = 128
    embedding_dim: int = 64
    dropout: float = 0.1


if TORCH_AVAILABLE:

    class TabularAutoencoder(nn.Module):
        """Encoder: input_dim -> hidden_dim -> embedding_dim
        Decoder:  embedding_dim -> hidden_dim -> input_dim

        forward() returns (embedding, reconstruction). Only `embedding`
        is used downstream (Phase 7 fusion); `reconstruction` exists for
        an eventual reconstruction-loss training objective (not yet
        implemented -- see module docstring)."""

        def __init__(self, config: AutoencoderConfig):
            super().__init__()
            self.config = config
            self.encoder = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.embedding_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(config.embedding_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.input_dim),
            )

        def forward(self, x: "torch.Tensor"):
            embedding = self.encoder(x)
            reconstruction = self.decoder(embedding)
            return embedding, reconstruction

else:

    class TabularAutoencoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "torch is not installed. Install it via `pip install -r "
                "requirements.txt` (or `pip install torch`) to use "
                "TabularAutoencoder. The rest of Phase 5 (feature "
                "engineering, normalization) does not require torch and "
                "can run without it."
            )
