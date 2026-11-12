"""
Phase 4 -- Transformer time-series encoder, added 2026-09-06 as a
sibling to LSTMTimeSeriesEncoder (lstm_encoder.py, untouched by this
addition) so `modalities.timeseries.model_type` can select between
architectures. See docs/timeseries_modality.md for the full design
writeup.

Same input/output contract as LSTMTimeSeriesEncoder: forward(x,
lengths) -> (batch, hidden_dim). Mechanically different, though, since
a Transformer has no native notion of "padding" the way an RNN-family
cell does:

- Padded timesteps are excluded from self-attention via a
  `src_key_padding_mask` (derived from `lengths`), NOT
  `pack_padded_sequence` (a pure RNN-family mechanism with no
  Transformer equivalent).
- The patient embedding is taken from EACH PATIENT'S OWN LAST REAL
  timestep's output position -- matching LSTM/RNN/GRU's "final hidden
  state" semantic exactly (rather than e.g. mean-pooling over every
  real timestep), so all four architectures extract a patient
  embedding via the same conceptual rule, keeping any comparison
  between them fair rather than confounded by a different pooling
  choice.
- `hidden_dim` doubles as the Transformer's `d_model` -- kept as the
  same name/config key as LSTM's own `hidden_dim` so the output
  embedding width matches exactly regardless of which architecture is
  selected (this is what keeps Phase 7's fusion, and everything after
  it, completely oblivious to which architecture produced the
  embeddings it's fusing -- see docs/timeseries_modality.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # placeholder so the module still imports without torch


@dataclass
class TransformerEncoderConfig:
    input_dim: int
    hidden_dim: int = 128  # doubles as d_model -- see module docstring
    num_layers: int = 2
    num_heads: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.2
    max_bins: int = 48  # must match temporal.max_sequence_length -- sizes the (fixed,
                         # non-learned) positional encoding table


if TORCH_AVAILABLE:

    class _SinusoidalPositionalEncoding(nn.Module):
        """Standard fixed positional encoding (Vaswani et al. 2017,
        "Attention Is All You Need") -- computed once at construction,
        not learned, so it adds no trainable parameters."""

        def __init__(self, d_model: int, max_len: int):
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return x + self.pe[: x.shape[1]].unsqueeze(0)

    class TransformerTimeSeriesEncoder(nn.Module):
        def __init__(self, config: TransformerEncoderConfig):
            super().__init__()
            if config.input_dim < 1:
                raise ValueError(f"input_dim must be >= 1, got {config.input_dim}.")
            if config.hidden_dim < 1:
                raise ValueError(f"hidden_dim must be >= 1, got {config.hidden_dim}.")
            if config.hidden_dim % config.num_heads != 0:
                raise ValueError(
                    f"hidden_dim ({config.hidden_dim}) must be divisible by num_heads "
                    f"({config.num_heads}) -- required by nn.MultiheadAttention."
                )
            self.config = config
            self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
            self.positional_encoding = _SinusoidalPositionalEncoding(config.hidden_dim, config.max_bins)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        def forward(self, x: "torch.Tensor", lengths: "torch.Tensor") -> "torch.Tensor":
            """x: (batch, max_bins, input_dim), lengths: (batch,) real
            (non-padded) timestep counts -> (batch, hidden_dim)."""
            batch_size, max_bins, _ = x.shape
            if max_bins > self.config.max_bins:
                raise ValueError(
                    f"input has {max_bins} timesteps but this encoder's positional encoding was "
                    f"only sized for max_bins={self.config.max_bins} -- construct with a matching "
                    f"max_bins (temporal.max_sequence_length)."
                )

            projected = self.input_projection(x)
            projected = self.positional_encoding(projected)

            # True at PADDED positions -- nn.TransformerEncoder's own
            # convention for src_key_padding_mask (True = ignore).
            position_idx = torch.arange(max_bins, device=x.device).unsqueeze(0)
            padding_mask = position_idx >= lengths.unsqueeze(1)

            encoded = self.transformer(projected, src_key_padding_mask=padding_mask)

            # Each patient's own last REAL timestep -- lengths-1, same
            # "final hidden state" semantic as the RNN-family encoders.
            # Same implicit lengths>=1 assumption LSTMTimeSeriesEncoder
            # already relies on (pack_padded_sequence requires it too);
            # clamped defensively rather than left to index -1 silently.
            last_real_idx = (lengths - 1).clamp(min=0)
            batch_idx = torch.arange(batch_size, device=x.device)
            return encoded[batch_idx, last_real_idx]

else:

    class TransformerTimeSeriesEncoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "torch is not installed -- install it (`pip install -r requirements.txt` or "
                "`pip install torch`) to use TransformerTimeSeriesEncoder."
            )
