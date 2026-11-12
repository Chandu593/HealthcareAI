"""Hospital-local patient graphs, GCN and graph-compatible SMOTE for Configs 2-4.

Design (all of it per hospital, from that hospital's own patients only):
  * Graph A = kNN AND cosine threshold; Graph B = MiniBatchKMeans candidate restriction, then the same kNN/threshold rule inside each cluster. The TRAIN graph is built by the repo's own
    `build_hospital_graph` from the hospital's TRAINING patients only (labels are never read).
  * Validation / test patients are attached INDUCTIVELY: each query patient receives directed edges FROM its nearest training patients (same rule: top-k, cosine >= threshold, same
    cluster for Graph B). There are no val-val, test-test, val-test or query->train edges, so (a) a training node's representation is identical with or without query nodes, (b) a query
    patient's prediction never depends on any other validation / test patient, and (c) val and test are attached separately and never see each other.
  * Graph-compatible SMOTE (Config 3): synthetic POSITIVE nodes are interpolated between a training positive and one of its k nearest training positives of the SAME hospital, then wired
    with directed edges FROM real training nodes only (same attachment rule as a query patient). No synthetic->real edges, no synthetic-synthetic edges, no adjacency interpolation:
    real training nodes compute exactly what they would without SMOTE, and synthetic nodes exist only as extra supervised training nodes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.graph.patient_graph import build_hospital_graph
from src.three_modality.downstream_config import GRAPHS
from src.three_modality.hw_classifiers import focal_loss_torch, resolve_alpha, selection_score


def _l2(Z: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.where(n == 0, 1.0, n)


@dataclass
class TrainGraph:
    key: str                                   # "A" | "B"
    Z: np.ndarray                              # raw (unstandardised) training embeddings, float64 (n, d)
    Zn: np.ndarray                             # L2-normalised copy
    edge_index: np.ndarray                     # (2, E) int64 over training-node indices (symmetric)
    edge_weight: np.ndarray                    # (E,) exact cosine
    k: int
    threshold: float
    kmeans: object = None
    cluster_labels: "np.ndarray | None" = None

    @property
    def n(self) -> int:
        return int(self.Z.shape[0])


def fit_train_graph(Z_train: np.ndarray, key: str, gr: dict) -> TrainGraph:
    """Graph A / B over the hospital's TRAINING embeddings only, via the repo's `build_hospital_graph` (identical edges)."""
    Z = np.asarray(Z_train, dtype=np.float64)
    n = Z.shape[0]
    g = build_hospital_graph(list(range(n)), Z, method=GRAPHS[key], k=int(gr["k"]), cosine_threshold=float(gr["cosine_threshold"]),
                             max_nodes_full_similarity=int(gr["max_nodes_full_similarity"]), n_clusters=int(gr["n_clusters"]), cluster_seed=int(gr["cluster_seed"]))
    km, labels = None, None
    if key == "B":
        labels = g["cluster_labels"]
        eff = max(1, min(int(gr["n_clusters"]), n // 2))
        if n >= 2 and eff > 1:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=eff, random_state=int(gr["cluster_seed"]), n_init="auto").fit(_l2(Z))     # same call as the builder -> same fitted model, kept for predict()
            if not np.array_equal(km.labels_.astype(np.int64), labels):
                raise RuntimeError("re-fitted MiniBatchKMeans disagrees with the builder's cluster labels; cannot attach query patients consistently")
    return TrainGraph(key=key, Z=Z, Zn=_l2(Z), edge_index=g["edge_index"].astype(np.int64), edge_weight=np.asarray(g["edge_weight"], dtype=np.float32),
                      k=int(gr["k"]), threshold=float(gr["cosine_threshold"]), kmeans=km, cluster_labels=labels)


def attach_edges(tg: TrainGraph, Zq: np.ndarray, chunk: int = 2048):
    """Directed edges (train node -> query node). Returns (src_train_idx, dst_query_idx, weight). Query rows never connect to each other."""
    Zq = np.asarray(Zq, dtype=np.float64)
    nq = Zq.shape[0]
    if nq == 0 or tg.n == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    Qn = _l2(Zq)
    if tg.key == "B" and tg.kmeans is not None:
        qcl = tg.kmeans.predict(Qn).astype(np.int64)
    else:
        qcl = np.zeros(nq, dtype=np.int64)
    tcl = tg.cluster_labels if (tg.key == "B" and tg.kmeans is not None) else np.zeros(tg.n, dtype=np.int64)
    src, dst, wts = [], [], []
    for c in np.unique(qcl):
        qi = np.flatnonzero(qcl == c)
        members = np.flatnonzero(tcl == c)
        if len(members) == 0:
            continue
        kk = min(tg.k, len(members))
        for s in range(0, len(qi), chunk):
            qs = qi[s:s + chunk]
            sim = Qn[qs] @ tg.Zn[members].T                                        # exact cosine to the (same-cluster) training patients
            top = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
            sel = np.take_along_axis(sim, top, axis=1)
            keep = sel >= tg.threshold
            rows, cols = np.nonzero(keep)
            src.append(members[top[rows, cols]])
            dst.append(qs[rows])
            wts.append(sel[rows, cols].astype(np.float32))
    if not src:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    return np.concatenate(src).astype(np.int64), np.concatenate(dst).astype(np.int64), np.concatenate(wts)


def generate_synthetic_positives(Z_train: np.ndarray, y_train: np.ndarray, scfg: dict, seed: int):
    """Hospital-local SMOTE on TRAINING positives. Returns (Z_syn or None, info). Skipped (None) when the hospital has fewer than `min_positives` training positives."""
    y = np.asarray(y_train).astype(int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    info = {"n_train_pos": int(len(pos)), "n_train_neg": int(len(neg)), "n_synthetic": 0, "skipped": None}
    if len(pos) < int(scfg["min_positives"]):
        info["skipped"] = f"fewer than {scfg['min_positives']} training positives ({len(pos)})"
        return None, info
    target_pos = int(round(float(scfg["ratio"]) * len(neg)))
    n_syn = min(max(0, target_pos - len(pos)), int(scfg["max_synthetic_per_positive"]) * len(pos))
    if n_syn == 0:
        info["note"] = "training positives already at/above the target ratio; no synthetic node needed"
        return None, info
    from sklearn.neighbors import NearestNeighbors
    P = np.asarray(Z_train, dtype=np.float64)[pos]
    k = min(int(scfg["k_neighbors"]), len(pos) - 1)
    nn_idx = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(P, return_distance=False)[:, 1:]       # column 0 is the point itself
    rng = np.random.default_rng(seed)
    parents = np.arange(n_syn) % len(pos)
    partner = nn_idx[parents, rng.integers(0, k, size=n_syn)]
    lam = rng.random(n_syn)[:, None]
    Zs = P[parents] + lam * (P[partner] - P[parents])
    info.update({"n_synthetic": int(n_syn), "k_neighbors_used": int(k)})
    return Zs, info


# ---------------------------------------------------------------------------------------------------------------------------------
# GCN
# ---------------------------------------------------------------------------------------------------------------------------------
def make_gcn(in_dim: int, gcfg: dict):
    import torch.nn as nn
    from torch_geometric.nn import GCNConv

    class HospitalGCN(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [in_dim] + [int(gcfg["hidden"])] * int(gcfg["layers"])
            self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1], add_self_loops=True, normalize=True) for i in range(int(gcfg["layers"]))])
            self.drop = nn.Dropout(float(gcfg["dropout"]))
            self.head = nn.Linear(dims[-1], 1)

        def forward(self, x, edge_index, edge_weight):
            h = x
            for i, conv in enumerate(self.convs):
                h = conv(h, edge_index, edge_weight).relu()
                if i < len(self.convs) - 1:
                    h = self.drop(h)
            return h, self.head(h).squeeze(-1)

    return HospitalGCN()


def _graph_tensors(tg: TrainGraph, q_src, q_dst, q_w, n_extra: int, torch):
    """Train edges + directed (train -> extra) edges; extra nodes are indexed n_train + j."""
    ei = np.concatenate([tg.edge_index, np.stack([q_src, tg.n + q_dst])], axis=1) if len(q_src) else tg.edge_index
    ew = np.concatenate([tg.edge_weight, q_w]) if len(q_w) else tg.edge_weight
    return torch.from_numpy(ei.astype(np.int64)), torch.from_numpy(ew.astype(np.float32))


def train_gcn(Ztr, ytr, Zva, yva, Zte, key: str, ds_cfg: dict, seed: int, smote: bool = False, want_embeddings: bool = False) -> dict:
    """Train ONE hospital's GCN on its own training graph (+ synthetic nodes when `smote`). Validation nodes (labels) only select the checkpoint; test nodes are attached once at the end.

    Returns p_val, p_test (+ embeddings of train/val/test nodes when requested) and an `info` dict (graph statistics, SMOTE report, best epoch)."""
    import copy

    import torch

    gcfg, gr = ds_cfg["gcn"], ds_cfg["graph"]
    torch.manual_seed(seed)
    ytr, yva = np.asarray(ytr).astype(int), np.asarray(yva).astype(int)
    if len(np.unique(ytr)) < 2:
        raise ValueError("training labels contain a single class; the caller must skip this hospital")
    sc = StandardScaler().fit(Ztr)                                                 # model features: scaling fitted on this hospital's training rows only
    Xtr, Xva, Xte = (torch.from_numpy(sc.transform(a).astype(np.float32)) for a in (Ztr, Zva, Zte))
    tg = fit_train_graph(Ztr, key, gr)
    n_tr = tg.n
    focal = {**ds_cfg["focal"], **{k: gcfg[k] for k in ("gamma", "alpha") if gcfg.get(k) is not None}}      # optional per-family override (set by the tuning stage)
    gamma, alpha = float(focal["gamma"]), resolve_alpha(focal, ytr, smote)

    # ---- training graph: real train nodes (+ synthetic positive nodes wired FROM real train nodes) ----
    Zs, sinfo = (generate_synthetic_positives(Ztr, ytr, ds_cfg["smote"], seed) if smote else (None, {"n_synthetic": 0, "skipped": "smote off"}))
    m = 0 if Zs is None else Zs.shape[0]
    if m:
        s_src, s_dst, s_w = attach_edges(tg, Zs)
        sinfo["synthetic_nodes_with_no_edge"] = int(m - len(np.unique(s_dst)))
        sinfo["synthetic_edges"] = int(len(s_src))
        x_train_graph = torch.cat([Xtr, torch.from_numpy(sc.transform(Zs).astype(np.float32))])
    else:
        s_src = s_dst = np.empty(0, np.int64)
        s_w = np.empty(0, np.float32)
        x_train_graph = Xtr
    ei_tr, ew_tr = _graph_tensors(tg, s_src, s_dst, s_w, m, torch)
    y_nodes = torch.from_numpy(np.concatenate([ytr, np.ones(m, dtype=int)]))
    # ---- validation inference graph: train graph + val nodes attached inductively ----
    v_src, v_dst, v_w = attach_edges(tg, Zva)
    ei_va, ew_va = _graph_tensors(tg, v_src, v_dst, v_w, len(Zva), torch)
    x_val_graph = torch.cat([Xtr, Xva])
    yva_t = torch.from_numpy(yva)

    model = make_gcn(Xtr.shape[1], gcfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(gcfg["lr"]), weight_decay=float(gcfg["weight_decay"]))
    metric, min_ep = gcfg.get("selection_metric", "loss"), int(gcfg.get("min_epochs", 1))
    best, best_state, best_ep, bad, ep, best_loss = float("inf"), None, 0, 0, 0, float("nan")
    for ep in range(1, int(gcfg["max_epochs"]) + 1):
        model.train()
        opt.zero_grad()
        _, logit = model(x_train_graph, ei_tr, ew_tr)
        loss = focal_loss_torch(logit[: n_tr + m], y_nodes, alpha, gamma)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            _, lv = model(x_val_graph, ei_va, ew_va)
            vl = focal_loss_torch(lv[n_tr:], yva_t, alpha, gamma).item()
            pvp = torch.sigmoid(lv[n_tr:]).numpy()
        if ep < min_ep:
            continue                                                        # a checkpoint before `min_epochs` can never be selected
        sc = selection_score(metric, vl, yva, pvp)
        if sc < best - 1e-7:
            best, best_state, best_ep, bad, best_loss = sc, copy.deepcopy(model.state_dict()), ep, 0, vl
        else:
            bad += 1
            if bad >= int(gcfg["patience"]):
                break
    if best_state is None:
        best_state, best_ep, best_loss = copy.deepcopy(model.state_dict()), ep, float("nan")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ev, lv = model(x_val_graph, ei_va, ew_va)
        # test nodes are attached once, only now, in their OWN graph (never together with val nodes)
        t_src, t_dst, t_w = attach_edges(tg, Zte)
        ei_te, ew_te = _graph_tensors(tg, t_src, t_dst, t_w, len(Zte), torch)
        et, lt = model(torch.cat([Xtr, Xte]), ei_te, ew_te)
    out = {"p_val": torch.sigmoid(lv[n_tr:]).numpy().astype(np.float64), "p_test": torch.sigmoid(lt[n_tr:]).numpy().astype(np.float64),
           "info": {"best_epoch": best_ep, "epochs_run": ep, "alpha": alpha, "gamma": gamma, "best_val_focal_loss": best_loss, "selection_metric": metric, "min_epochs": min_ep, "graph": GRAPHS[key],
                    "train_nodes": n_tr, "train_edges_directed": int(tg.edge_index.shape[1]), "isolated_train_nodes": int(n_tr - len(np.unique(tg.edge_index[1]))) if n_tr else 0,
                    "val_attach_edges": int(len(v_src)), "test_attach_edges": int(len(t_src)), "smote": sinfo,
                    "n_clusters_effective": None if tg.cluster_labels is None else int(len(np.unique(tg.cluster_labels)))}}
    if want_embeddings:
        out["emb_train"], out["emb_val"], out["emb_test"] = ev[:n_tr].numpy(), ev[n_tr:].numpy(), et[n_tr:].numpy()
    return out
