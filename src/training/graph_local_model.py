"""
Graph A / Graph B comparison model -- CORRECTED 2026-09-23.

The flat-model ablation framework (src/ablation/embedding_ablation_
runner.py, src/classifiers/boosting_classifiers.py) fits separate
boosted-tree classifiers (AdaBoost/XGBoost/LightGBM) on frozen
per-patient embeddings -- a deliberate, additive shortcut built so many
(ts_model, tab_model, fusion_mode) combinations could be compared
cheaply without training a full GNN for each one.

That shortcut does NOT carry over to the graph-comparison stage. Once
the best FLAT configuration has been selected (from the real sampling
comparison, via `src/ablation/sampling_comparison_runner.py`), Graph A
and Graph B must each be evaluated with the GNN's OWN natural
fully-connected prediction head (`src/gnn/patient_gnn.py`'s
`PatientGNN` + `src/heads/multitask_heads.py`'s `MultiTaskHeads`, both
reused UNCHANGED and untouched here), trained end-to-end via backprop
-- never a separate classifier fit on top of GNN output. This module
exists specifically to correct an earlier plan that would have reused
the boosted-tree classifiers for this stage too.

`GraphOnlyLocalModel` takes FROZEN, already-computed per-patient
embeddings (the winning flat config's frozen encoder output) as fixed
node features -- it never re-trains, or even re-runs, an encoder; only
the GNN's own message-passing weights and the heads' own linear layers
are trained here. Per-hospital LOCAL training only, no FedAvg rounds
(confirmed with the user 2026-09-20: "skip the federated-learning
concept" for this work) -- each hospital trains its own independent
copy, exactly like `src/ablation/embedding_ablation_runner.py`'s flat
encoders do, and exactly like Phase 9-13's own precedent of never
pooling raw data or gradients across hospitals.

Leakage safety: the GRADIENT-COMPUTING forward/backward pass runs on a
TRAIN-ONLY SUBGRAPH (`_build_train_only_subgraph()`, imported from
`run_pipeline.py` -- the SAME function the split-aware Phase 11/13
track already uses, not a second reimplementation that could drift).
Validation/test rows are never present in any tensor the training step
touches. Evaluation runs the TRAINED model over the FULL hospital
graph (inductive inference, same GraphSAGE-style pattern
`_build_train_only_subgraph()`'s own docstring describes), then the
caller extracts val/test rows' predictions from that full-graph
forward pass -- so a val/test patient's OWN prediction can legitimately
use its real neighbors (some of which may be train patients), while
its features/label never once entered a gradient computation.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from run_pipeline import _build_train_only_subgraph
from src.gnn.patient_gnn import PatientGNN
from src.heads.multitask_heads import MultiTaskHeads
from src.training.losses import compute_multitask_loss


class GraphOnlyLocalModel(nn.Module):
    """PatientGNN -> MultiTaskHeads, nothing else. `in_dim` must match
    the frozen embeddings' width (e.g. 192 for the flat ablation
    framework's LSTM(128)+MLP/autoencoder(64) concatenation)."""

    def __init__(
        self, in_dim: int, gnn_hidden_dim: int = 128, gnn_n_layers: int = 2, gnn_type: str = "graphsage",
        gnn_dropout: float = 0.2, gat_heads: int = 4, head_hidden_dim: int = 64, head_dropout: float = 0.1,
    ):
        super().__init__()
        self.gnn = PatientGNN(
            in_dim=in_dim, hidden_dim=gnn_hidden_dim, n_layers=gnn_n_layers, gnn_type=gnn_type,
            dropout=gnn_dropout, gat_heads=gat_heads,
        )
        self.heads = MultiTaskHeads(in_dim=gnn_hidden_dim, head_hidden_dim=head_hidden_dim, dropout=head_dropout)

    def forward(self, embeddings: torch.Tensor, edge_index: torch.Tensor, edge_weight: "torch.Tensor | None" = None) -> dict:
        gnn_out = self.gnn(embeddings, edge_index, edge_weight)
        return self.heads(gnn_out)


def train_graph_model_one_hospital(
    embeddings: np.ndarray, edge_index: np.ndarray, edge_weight: np.ndarray, train_row_mask: np.ndarray,
    mortality_labels: np.ndarray, sepsis_labels: np.ndarray, model_config: dict, training_cfg: dict,
    logger: logging.Logger, seed: "int | None" = None,
) -> "tuple[GraphOnlyLocalModel | None, str | None, list]":
    """Trains ONE hospital's own, independently-initialized
    GraphOnlyLocalModel using ONLY the train-only subgraph (see module
    docstring). Returns `(model_or_None, skip_reason_or_None,
    loss_history)` -- never raises for a routine skip condition (too
    few train patients, no usable labels, non-finite loss), matching
    `_train_one_hospital_flat_model()`'s and Phase 11's own
    established skip-reason discipline exactly.
    """
    n_train = int(train_row_mask.sum())
    if n_train < 1:
        return None, "no_train_patients", []

    embeddings_t = torch.tensor(embeddings, dtype=torch.float32)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    edge_weight_t = torch.tensor(edge_weight, dtype=torch.float32) if edge_weight is not None and len(edge_weight) else None
    train_mask_t = torch.tensor(train_row_mask, dtype=torch.bool)
    mortality_t = torch.tensor(mortality_labels, dtype=torch.float32)
    sepsis_t = torch.tensor(sepsis_labels, dtype=torch.float32)

    sub_edge_index, sub_edge_weight, train_indices = _build_train_only_subgraph(edge_index_t, edge_weight_t, train_mask_t)
    train_embeddings = embeddings_t[train_indices]
    train_mortality = mortality_t[train_indices]
    train_sepsis = sepsis_t[train_indices]

    if seed is not None:
        torch.manual_seed(seed)
    model = GraphOnlyLocalModel(**model_config)
    model.train()
    lr = training_cfg.get("learning_rate", 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mortality_weight = training_cfg.get("mortality_weight", 1.0)
    sepsis_weight = training_cfg.get("sepsis_weight", 1.0)
    loss_type = training_cfg.get("loss", "weighted_bce")
    epochs = training_cfg.get("epochs", 10)

    loss_history: list = []
    for _epoch in range(epochs):
        optimizer.zero_grad()
        out = model(train_embeddings, sub_edge_index, sub_edge_weight)
        total_loss, _components, _counts = compute_multitask_loss(
            out["mortality_logits"], out["sepsis_logits"], train_mortality, train_sepsis,
            mortality_weight=mortality_weight, sepsis_weight=sepsis_weight, loss_type=loss_type,
        )
        if total_loss is None:
            return None, "no_labels", loss_history
        if not torch.isfinite(total_loss):
            return None, "non_finite_loss", loss_history
        total_loss.backward()
        optimizer.step()
        loss_history.append(float(total_loss.item()))

    return model, None, loss_history


def predict_graph_model_full_hospital(
    model: GraphOnlyLocalModel, embeddings: np.ndarray, edge_index: np.ndarray, edge_weight: np.ndarray,
) -> dict:
    """Inductive inference over the FULL hospital graph (every node,
    not just train) -- see module docstring's "Leakage safety"
    section for why this is safe: the model's WEIGHTS never saw a
    val/test row during training, only their features are used now,
    at inference time, exactly like a deployed model scoring a new
    patient. Returns `{"mortality_prob": (n,) np.ndarray,
    "sepsis_prob": (n,) np.ndarray}`.
    """
    embeddings_t = torch.tensor(embeddings, dtype=torch.float32)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    edge_weight_t = torch.tensor(edge_weight, dtype=torch.float32) if edge_weight is not None and len(edge_weight) else None
    model.eval()
    with torch.no_grad():
        out = model(embeddings_t, edge_index_t, edge_weight_t)
    return {
        "mortality_prob": torch.sigmoid(out["mortality_logits"]).numpy(),
        "sepsis_prob": torch.sigmoid(out["sepsis_logits"]).numpy(),
    }
