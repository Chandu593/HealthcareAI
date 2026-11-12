"""
Phase 9 -- Graph Neural Network (architecture + forward-pass
verification only, no training loop yet -- same pattern as Phase 4's
LSTM and Phase 5's autoencoder: weights are randomly initialized,
verified for correct mechanics/shapes, not learned representations).

Consumes Phase 8's per-hospital patient graphs (node features = Phase
7's fused embeddings, edge_index/edge_weight from k-NN or
cosine-threshold construction) and produces refined per-patient node
embeddings via message passing. Output stays at `hidden_dim` -- no
task head here; mortality/sepsis prediction is Phase 10's job.

Still one independent forward pass PER HOSPITAL's own graph, same
model weights, never pooled -- no exception to the privacy boundary
every prior phase has held.

ALL THREE architectures from `configs/base.yaml`'s `gnn.type` are
implemented (graphsage | gat | gcn), not just the configured default
-- matching the ablation-comparison capability a research write-up
benefits from, and consistent with how `fusion.type` and `graph.method`
were also both/all fully implemented rather than just their defaults.

EDGE WEIGHT USAGE -- differentiated by architecture, not applied
uniformly (decided 2026-08-26, confirmed with user: use it only where
it's a genuine, well-founded mechanism, not bolted on to look more
sophisticated):

- **GCN**: USES Phase 8's edge_weight (cosine similarity). PyG's
  GCNConv has a native `edge_weight` parameter -- this directly
  matches the standard, published weighted-graph extension of Kipf &
  Welling (2017)'s formulation (edge weights modulate the normalized
  graph Laplacian). Genuinely well-founded, not a hack.
- **GAT**: does NOT use edge_weight/edge_attr. GATConv computes its
  OWN attention coefficients from node features at each layer --
  re-injecting the exact cosine similarity that was already used to
  build the graph's edges in the first place would be largely circular
  (the edge exists BECAUSE of that similarity) rather than adding
  genuine new signal, and isn't how GAT is standardly used in the
  literature. Standard self-attention only.
- **GraphSAGE**: does NOT use edge weights. PyG's SAGEConv has no
  edge_weight/edge_attr parameter at all in its standard formulation
  (mean/pooling aggregation over unweighted neighbor sets, per
  Hamilton et al. 2017) -- forcing one in would require a non-standard
  custom aggregator whose benefit is unproven, deviating from what
  "GraphSAGE" means in published work.

Verified via `python -c` against the actual installed PyG API
(2.8.0.post1) before this design was finalized, not assumed from
memory -- see IMPLEMENTATION_TRACKER.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

VALID_GNN_TYPES = ("graphsage", "gat", "gcn")


class PatientGNN(nn.Module):
    """A stack of `n_layers` GNN layers of the configured `gnn_type`,
    mapping (N, in_dim) node features -> (N, hidden_dim) refined
    per-patient embeddings via message passing over `edge_index`.

    `gat_heads` only applies when gnn_type='gat': intermediate layers
    concatenate `gat_heads` attention heads (standard GAT design, per
    Velickovic et al. 2018), each producing hidden_dim // gat_heads
    channels so the concatenated width is exactly hidden_dim; the
    final layer averages heads instead (concat=False) to produce a
    clean hidden_dim-wide output, not gat_heads * hidden_dim.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        gnn_type: str = "graphsage",
        dropout: float = 0.2,
        gat_heads: int = 4,
    ):
        super().__init__()
        if gnn_type not in VALID_GNN_TYPES:
            raise ValueError(f"gnn_type={gnn_type!r} not recognized -- must be one of {VALID_GNN_TYPES}.")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}.")
        if gnn_type == "gat" and hidden_dim % gat_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be evenly divisible by gat_heads ({gat_heads}) "
                f"so concatenated intermediate-layer heads produce exactly hidden_dim channels."
            )

        self.gnn_type = gnn_type
        self.dropout = dropout
        self.layers = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * n_layers
        for i in range(n_layers):
            is_last = i == n_layers - 1
            in_channels, out_channels = dims[i], dims[i + 1]
            if gnn_type == "graphsage":
                self.layers.append(SAGEConv(in_channels, out_channels))
            elif gnn_type == "gat":
                if is_last:
                    self.layers.append(GATConv(in_channels, out_channels, heads=gat_heads, concat=False))
                else:
                    self.layers.append(
                        GATConv(in_channels, out_channels // gat_heads, heads=gat_heads, concat=True)
                    )
            elif gnn_type == "gcn":
                self.layers.append(GCNConv(in_channels, out_channels))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """`edge_weight` is only ever passed through to the underlying
        layer when gnn_type='gcn' -- see module docstring. Safe to
        always pass it in regardless of gnn_type; it's simply ignored
        for graphsage/gat.
        """
        for i, layer in enumerate(self.layers):
            is_last = i == len(self.layers) - 1
            if self.gnn_type == "gcn":
                x = layer(x, edge_index, edge_weight=edge_weight)
            else:
                x = layer(x, edge_index)
            if not is_last:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
