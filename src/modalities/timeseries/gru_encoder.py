"""
Phase 4 -- GRU time-series encoder, added 2026-09-06 as a sibling to
LSTMTimeSeriesEncoder (lstm_encoder.py, untouched by this addition) so
`modalities.timeseries.model_type` can select between architectures.
See docs/timeseries_modality.md for the full design writeup.

Same input/output contract as LSTMTimeSeriesEncoder: forward(x,
lengths) -> (batch, hidden_dim), one embedding per patient taken from
their own final real (non-padded) timestep, via the same
pack_padded_sequence handling of variable-length sequences. A drop-in
swap wherever LSTMTimeSeriesEncoder is used, EXCEPT that `nn.GRU`
(like plain RNN, unlike LSTM) has no cell state, so its hidden output
is a single tensor `h_n`, not the `(h_n, c_n)` tuple LSTM returns.
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
class GRUEncoderConfig:
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2


if TORCH_AVAILABLE:

    class GRUTimeSeriesEncoder(nn.Module):
        def __init__(self, config: GRUEncoderConfig):
            super().__init__()
            if config.input_dim < 1:
                raise ValueError(f"input_dim must be >= 1, got {config.input_dim}.")
            if config.hidden_dim < 1:
                raise ValueError(f"hidden_dim must be >= 1, got {config.hidden_dim}.")
            self.config = config
            self.gru = nn.GRU(
                input_size=config.input_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                batch_first=True,
            )

        def forward(self, x: "torch.Tensor", lengths: "torch.Tensor") -> "torch.Tensor":
            """x: (batch, max_bins, input_dim), lengths: (batch,) real
            (non-padded) timestep counts -> (batch, hidden_dim)."""
            lengths_cpu = lengths.detach().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False,
            )
            _, h_n = self.gru(packed)
            return h_n[-1]  # last layer's final hidden state, one vector per patient

else:

    class GRUTimeSeriesEncoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "torch is not installed -- install it (`pip install -r requirements.txt` or "
                "`pip install torch`) to use GRUTimeSeriesEncoder."
            )
