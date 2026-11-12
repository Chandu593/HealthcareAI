"""
Ablation framework -- FlatMultimodalModel: the graph-free joint model
used for the 72-config TS x TAB x Fusion x Classifier ablation
(IMPLEMENTATION_TRACKER.md's ablation-framework entry; decisions
#1/#2/#8 confirmed with the user 2026-09-10).

Chains a configurable TS encoder + TAB encoder (or a frozen PCA
embedding, see below) + a configurable fusion mechanism + MultiTaskHeads
-- deliberately WITHOUT Phase 9's GNN, since the 72-config ablation is
explicitly about TS/TAB/fusion embedding quality BEFORE the graph stage
even runs (decision #8's execution order: split -> TS/TAB embeddings ->
fusion -> 72-config classifier ablation -> THEN graph/GNN -> THEN
federated learning). This is a NEW, ADDITIVE module, completely
separate from src/training/local_model.py's LocalPatientModel (Phase
11, hardcoded to LSTM+Autoencoder+GraphSAGE/GAT/GCN+heads) -- that
module and every phase that depends on it (11/12/13) are UNTOUCHED by
this addition, per "preserve the locked architecture."

TRAINING SCOPE (decision #2): each hospital trains its OWN, independent
copy of this model -- "locally trained representations," never
FedAvg'd/FedProto'd, and only on that hospital's TRAIN-split patients
(see src/ablation/embedding_ablation_runner.py, which is the only
intended caller of the training loop; this module itself has no
opinion about which patients it's given). MLP heads (mortality+sepsis)
exist here ONLY to give the TS/TAB encoders + fusion mechanism a real
supervised training signal -- per decision #1, AdaBoost/XGBoost/
LightGBM are fit completely separately and afterwards, on this model's
FROZEN fused-embedding output (`encode()`'s return value), never on the
heads' own predictions and never inside this model's backward pass.

TAB=PCA IS SPECIAL: PCA (src/modalities/tabular/pca_encoder.py) has no
gradient and is not an nn.Module -- for a PCA combo, the caller
pre-computes each hospital's per-hospital-fit PCA embedding OUTSIDE
this model (fit on that hospital's train-split rows only, see
pca_encoder.py's own docstring) and passes it in as `tabular_x`
directly, already at the target embedding width. This model then has
NO tabular-encoder submodule at all for a PCA combo (`tab_encoder is
None`) -- `tabular_x` IS the embedding, passed straight into fusion.
Gradients still flow into the TS encoder and the fusion mechanism (if
it has parameters, i.e. cross_attention) exactly as normal; they simply
have nowhere to flow on the frozen PCA side, same as any other
frozen/detached input.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.fusion.cross_attention_fusion import CrossAttentionFusion, CrossAttentionFusionConfig
from src.heads.multitask_heads import MultiTaskHeads
from src.modalities.tabular.autoencoder import AutoencoderConfig, TabularAutoencoder
from src.modalities.tabular.mlp_encoder import MLPEncoderConfig, TabularMLPEncoder
from src.modalities.timeseries.gru_encoder import GRUEncoderConfig, GRUTimeSeriesEncoder
from src.modalities.timeseries.lstm_encoder import LSTMEncoderConfig, LSTMTimeSeriesEncoder
from src.modalities.timeseries.rnn_encoder import RNNEncoderConfig, RNNTimeSeriesEncoder
from src.modalities.timeseries.transformer_encoder import TransformerEncoderConfig, TransformerTimeSeriesEncoder

VALID_TS_MODELS = ("lstm", "rnn", "gru", "transformer")
VALID_TAB_MODELS = ("autoencoder", "mlp", "pca")
VALID_FUSION_MODES = ("concatenation", "cross_attention")

_TS_ENCODER_CLASSES = {
    "lstm": (LSTMTimeSeriesEncoder, LSTMEncoderConfig),
    "rnn": (RNNTimeSeriesEncoder, RNNEncoderConfig),
    "gru": (GRUTimeSeriesEncoder, GRUEncoderConfig),
    "transformer": (TransformerTimeSeriesEncoder, TransformerEncoderConfig),
}


@dataclass
class FlatModelConfig:
    ts_model: str
    tab_model: str
    fusion_mode: str
    ts_input_dim: int
    tab_input_dim: int          # for tab_model="pca": this is the ALREADY-EMBEDDED pca width, not raw feature count
    ts_max_bins: int = 48       # must match the actual timeseries_x tensor's bin dimension (transformer only)
    ts_hidden_dim: int = 128
    ts_num_layers: int = 2
    ts_dropout: float = 0.2
    ts_transformer_num_heads: int = 4
    ts_transformer_dim_feedforward: int = 256
    tab_hidden_dim: int = 128
    tab_embedding_dim: int = 64
    tab_dropout: float = 0.1
    cross_attention_attn_dim: int = 128
    cross_attention_num_heads: int = 4
    head_hidden_dim: int = 64
    head_dropout: float = 0.1

    def __post_init__(self):
        if self.ts_model not in VALID_TS_MODELS:
            raise ValueError(f"ts_model must be one of {VALID_TS_MODELS}, got {self.ts_model!r}.")
        if self.tab_model not in VALID_TAB_MODELS:
            raise ValueError(f"tab_model must be one of {VALID_TAB_MODELS}, got {self.tab_model!r}.")
        if self.fusion_mode not in VALID_FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {VALID_FUSION_MODES}, got {self.fusion_mode!r}.")


class FlatMultimodalModel(nn.Module):
    """One hospital's local, graph-free model: TS_encoder ->
    (TAB_encoder | frozen PCA passthrough) -> fusion -> MultiTaskHeads.
    """

    def __init__(self, config: FlatModelConfig):
        super().__init__()
        self.config = config

        ts_cls, ts_cfg_cls = _TS_ENCODER_CLASSES[config.ts_model]
        if config.ts_model == "transformer":
            ts_cfg = ts_cfg_cls(
                input_dim=config.ts_input_dim, hidden_dim=config.ts_hidden_dim, num_layers=config.ts_num_layers,
                num_heads=config.ts_transformer_num_heads, dim_feedforward=config.ts_transformer_dim_feedforward,
                dropout=config.ts_dropout, max_bins=config.ts_max_bins,
            )
        else:
            ts_cfg = ts_cfg_cls(
                input_dim=config.ts_input_dim, hidden_dim=config.ts_hidden_dim,
                num_layers=config.ts_num_layers, dropout=config.ts_dropout,
            )
        self.ts_encoder = ts_cls(ts_cfg)
        ts_output_dim = config.ts_hidden_dim

        self.has_reconstruction = False
        if config.tab_model == "autoencoder":
            self.tab_encoder = TabularAutoencoder(AutoencoderConfig(
                input_dim=config.tab_input_dim, hidden_dim=config.tab_hidden_dim,
                embedding_dim=config.tab_embedding_dim, dropout=config.tab_dropout,
            ))
            tab_output_dim = config.tab_embedding_dim
            self.has_reconstruction = True
        elif config.tab_model == "mlp":
            self.tab_encoder = TabularMLPEncoder(MLPEncoderConfig(
                input_dim=config.tab_input_dim, hidden_dim=config.tab_hidden_dim,
                embedding_dim=config.tab_embedding_dim, dropout=config.tab_dropout,
            ))
            tab_output_dim = config.tab_embedding_dim
        elif config.tab_model == "pca":
            self.tab_encoder = None  # frozen, per-hospital-fit PCA embedding passed straight in as tabular_x
            tab_output_dim = config.tab_input_dim
        else:
            raise AssertionError("unreachable -- FlatModelConfig.__post_init__ already validated tab_model")

        if config.fusion_mode == "cross_attention":
            self.fusion = CrossAttentionFusion(CrossAttentionFusionConfig(
                ts_dim=ts_output_dim, tab_dim=tab_output_dim, attn_dim=config.cross_attention_attn_dim,
                num_heads=config.cross_attention_num_heads,
            ))
            fused_dim = self.fusion.output_dim
        else:
            self.fusion = None  # plain torch.cat -- Phase 7's own locked, zero-parameter design
            fused_dim = ts_output_dim + tab_output_dim

        self.fused_dim = fused_dim
        self.heads = MultiTaskHeads(fused_dim, head_hidden_dim=config.head_hidden_dim, dropout=config.head_dropout)

    def encode(
        self, timeseries_x: torch.Tensor, timeseries_lengths: torch.Tensor, tabular_x: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        """Returns (fused_embedding, tabular_reconstruction_or_None).
        Exposed separately from forward() so the ablation orchestrator
        can extract the FROZEN fused embedding (for the classifier
        stage, decision #1) without needing to also run the heads."""
        ts_embedding = self.ts_encoder(timeseries_x, timeseries_lengths)

        tabular_reconstruction = None
        if self.config.tab_model == "autoencoder":
            tab_embedding, tabular_reconstruction = self.tab_encoder(tabular_x)
        elif self.config.tab_model == "mlp":
            tab_embedding = self.tab_encoder(tabular_x)
        else:  # pca -- tabular_x IS already the frozen, per-hospital-fit embedding
            tab_embedding = tabular_x

        if self.fusion is not None:
            fused = self.fusion(ts_embedding, tab_embedding)
        else:
            fused = torch.cat([ts_embedding, tab_embedding], dim=1)
        return fused, tabular_reconstruction

    def forward(self, timeseries_x: torch.Tensor, timeseries_lengths: torch.Tensor, tabular_x: torch.Tensor) -> dict:
        fused, tabular_reconstruction = self.encode(timeseries_x, timeseries_lengths, tabular_x)
        head_out = self.heads(fused)
        return {
            "mortality_logits": head_out["mortality_logits"],
            "sepsis_logits": head_out["sepsis_logits"],
            "tabular_reconstruction": tabular_reconstruction,
            "fused_embedding": fused,
        }
