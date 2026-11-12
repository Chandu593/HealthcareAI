"""Neural architectures for the paper track."""

from __future__ import annotations

import torch
import torch.nn as nn


class PaperNet(nn.Module):
    """Tabular MLP (+ optional hourly-lab GRU) -> fused embedding -> risk logit.
    LayerNorm only (no BatchNorm running statistics), so FedAvg of the weights is well defined."""

    def __init__(self, d_tab: int, d_seq: int = 0, hidden=(256, 128), emb_dim: int = 64, dropout: float = 0.2, seq_hidden: int = 48):
        super().__init__()
        layers, d = [], d_tab
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.tab = nn.Sequential(*layers)
        self.seq = nn.GRU(d_seq, seq_hidden, batch_first=True) if d_seq > 0 else None
        self.fuse = nn.Sequential(nn.Linear(d + (seq_hidden if self.seq is not None else 0), emb_dim), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(emb_dim, 1)

    def forward(self, xt, xs=None):
        h = self.tab(xt)
        if self.seq is not None:
            _, hn = self.seq(xs)
            h = torch.cat([h, hn[-1]], dim=1)
        e = self.fuse(h)
        return self.head(e).squeeze(-1), e


class GraphHead(nn.Module):
    """GraphSAGE over a hospital's own patient graph, on frozen embeddings of the global model.
    SAGEConv has a root (self) weight, so with no edges it reduces to an MLP -- the graph arm can
    therefore be compared against 'no-edges' with identical capacity. `residual` adds the global
    model's own logit so the graph can only learn a correction."""

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2, residual: bool = True):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        dims = [in_dim] + [hidden] * layers
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i + 1]) for i in range(layers)])
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # starts exactly at the base logit when residual
        self.drop = nn.Dropout(dropout); self.residual = residual

    def forward(self, x, edge_index, base_logit=None):
        h = x
        for conv in self.convs:
            h = self.drop(torch.nn.functional.gelu(conv(h, edge_index)))
        z = self.out(h).squeeze(-1)
        return z + base_logit if (self.residual and base_logit is not None) else z


class LinearNet(nn.Module):
    """Logistic regression with the PaperNet call signature (used by the `fedavg_logreg` arm). Convex, so FedAvg is stable."""

    def __init__(self, d_tab: int):
        super().__init__()
        self.lin = nn.Linear(d_tab, 1)

    def forward(self, xt, xs=None):
        z = self.lin(xt).squeeze(-1)
        return z, z.unsqueeze(1)
