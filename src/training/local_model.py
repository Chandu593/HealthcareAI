"""
Phase 11 -- LocalPatientModel: the composite, end-to-end trainable
model each hospital locally trains its own copy of.

Chains the EXACT SAME nn.Module classes every prior phase already
built and real-full-scale-verified (LSTMTimeSeriesEncoder, Phase 4;
TabularAutoencoder, Phase 5; PatientGNN, Phase 9; MultiTaskHeads,
Phase 10) into one differentiable forward pass, so a single
loss.backward() call updates all four components jointly. This is a
NEW, ADDITIVE module -- run_phase4/5/7/8/9/10's existing runner
functions in run_pipeline.py are completely untouched and remain
exactly as valid as before for their own forward-pass-verification and
`--start-phase` purposes. Those functions communicate between phases
via saved .npz files (detached numpy arrays, no autograd graph) by
design, which is fine for standalone per-phase verification but would
silently break end-to-end backpropagation if reused directly here --
LocalPatientModel instead re-imports the same underlying classes and
chains them as torch tensors, in memory, for exactly one hospital's
data per forward call.

DESIGN DECISION (confirmed with the user, 2026-08-30): full end-to-end
training -- LSTM, autoencoder, GNN, AND heads are all trained jointly,
not just the heads. Directly grounded in two things found in the
existing code, not assumed: (1) Phase 9's own docstring already
describes "ONE model (shared weights, what a genuine federated
deployment would eventually FedAvg), applied independently to each
hospital's own graph" -- i.e., the whole stack, not just the heads, is
what a genuine federated deployment would train and average; (2)
TabularAutoencoder's decoder already exists with an inline comment
flagging it as built for "an eventual reconstruction-loss training
objective (not yet implemented)" -- this phase is what implements it
(see src/training/losses.py's reconstruction term).

Fusion is Phase 7's own locked design (Option A, concatenation, no
learned projection layer) -- reimplemented here as a plain torch.cat
rather than importing Phase 7's own run_phase7_multimodal_fusion(),
since that function operates on saved-to-disk numpy arrays for two
already-computed, independent embedding files, not two in-memory
tensors mid-forward-pass. The concatenation LOGIC is identical; only
the calling convention differs, for the reason given above.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.gnn.patient_gnn import PatientGNN
from src.heads.multitask_heads import MultiTaskHeads
from src.modalities.tabular.autoencoder import AutoencoderConfig, TabularAutoencoder
from src.modalities.timeseries.lstm_encoder import LSTMEncoderConfig, LSTMTimeSeriesEncoder


@dataclass
class LocalModelConfig:
    lstm: LSTMEncoderConfig
    autoencoder: AutoencoderConfig
    gnn_hidden_dim: int = 128
    gnn_n_layers: int = 2
    gnn_type: str = "graphsage"
    gnn_dropout: float = 0.2
    head_hidden_dim: int = 64
    head_dropout: float = 0.1


class LocalPatientModel(nn.Module):
    """One hospital's full local model: LSTM -> Autoencoder -> concat
    fusion -> GNN -> MultiTaskHeads, chained end to end.

    `fusion_dim` (LSTM hidden_dim + AE embedding_dim) becomes the
    GNN's `in_dim` automatically -- there is no separate fusion
    "layer" with its own learned weights (Phase 7's locked Option A),
    so nothing here needs its own dimension configured independently
    of the two upstream encoders' own output widths.
    """

    def __init__(self, config: LocalModelConfig):
        super().__init__()
        self.config = config
        self.lstm_encoder = LSTMTimeSeriesEncoder(config.lstm)
        self.autoencoder = TabularAutoencoder(config.autoencoder)

        fusion_dim = config.lstm.hidden_dim + config.autoencoder.embedding_dim
        self.gnn = PatientGNN(
            fusion_dim, config.gnn_hidden_dim, n_layers=config.gnn_n_layers,
            gnn_type=config.gnn_type, dropout=config.gnn_dropout,
        )
        self.heads = MultiTaskHeads(
            config.gnn_hidden_dim, head_hidden_dim=config.head_hidden_dim, dropout=config.head_dropout,
        )

    def forward(
        self,
        timeseries_x: torch.Tensor,
        timeseries_lengths: torch.Tensor,
        tabular_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: "torch.Tensor | None" = None,
    ) -> dict[str, torch.Tensor]:
        """All five inputs are for ONE hospital's patients only, in the
        SAME patient order (the graph's own node order -- see
        run_phase11_local_training()'s docstring for how callers must
        slice Phase 4/5's raw tensors to match Phase 8's graph node
        order, by ID, before calling this).

        timeseries_x:       (N, max_bins, timeseries_input_dim)
        timeseries_lengths: (N,) int -- real (non-padding) bin count per patient
        tabular_x:           (N, tabular_input_dim)
        edge_index:          (2, E) -- this hospital's own graph, never another's
        edge_weight:          (E,) or None
        """
        lstm_embedding = self.lstm_encoder(timeseries_x, timeseries_lengths)
        ae_embedding, tabular_reconstruction = self.autoencoder(tabular_x)
        fused = torch.cat([lstm_embedding, ae_embedding], dim=1)
        gnn_embedding = self.gnn(fused, edge_index, edge_weight=edge_weight)
        head_out = self.heads(gnn_embedding)
        return {
            "mortality_logits": head_out["mortality_logits"],
            "sepsis_logits": head_out["sepsis_logits"],
            "tabular_reconstruction": tabular_reconstruction,
        }
