"""
Phase 4d -- LSTM time-series encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # placeholder so class definition below doesn't NameError at import time


@dataclass
class LSTMEncoderConfig:
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2


if TORCH_AVAILABLE:

    class LSTMTimeSeriesEncoder(nn.Module):
        """Produces one patient-level embedding per input sequence.

        Input:  x (batch, max_bins, input_dim) float tensor -- the
                normalized, imputed, zero-padded array from
                normalization.py's NormalizationResult.array
                lengths (batch,) int tensor -- tensor.sequence_length,
                the REAL (non-padding) bin count per patient
        Output: (batch, hidden_dim) patient embedding -- taken from the
                last real timestep's hidden state (via
                pack_padded_sequence), not from a padding step.
        """

        def __init__(self, config: LSTMEncoderConfig):
            super().__init__()
            self.config = config
            self.lstm = nn.LSTM(
                input_size=config.input_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                batch_first=True,
            )

        def forward(self, x: "torch.Tensor", lengths: "torch.Tensor") -> "torch.Tensor":
            lengths_cpu = lengths.detach().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False,
            )
            _, (h_n, _) = self.lstm(packed)
            # h_n: (num_layers, batch, hidden_dim) -- take the last layer's
            # final hidden state as the patient embedding.
            return h_n[-1]

else:

    class LSTMTimeSeriesEncoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "torch is not installed. Install it via `pip install -r "
                "requirements.txt` (or `pip install torch`) to use "
                "LSTMTimeSeriesEncoder. The rest of Phase 4 (time binning, "
                "sequence construction, normalization) does not require "
                "torch and can run without it."
            )
