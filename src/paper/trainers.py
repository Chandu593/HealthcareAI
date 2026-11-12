"""Training loops: centralized (non-federated reference), local-only, FedAvg
(+optional FedProx), and federated GraphHead training. All use mini-batches,
class-weighted BCE, per-round/epoch validation and best-state restoration
(fixing the old 5-full-batch-step, unweighted-BCE, per-hospital-encoder setup)."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.paper.evaluation import auprc, fast_auroc
from src.paper.models import GraphHead, LinearNet, PaperNet

logger = logging.getLogger("eicu_fl")


@dataclass
class NeuralData:
    Xt: torch.Tensor
    Xs: "torch.Tensor | None"
    y: np.ndarray            # float32, NaN = unknown label
    hosp: np.ndarray


def set_seed(seed: int) -> None:
    np.random.seed(seed); torch.manual_seed(seed)


def pos_weight_value(y_train: np.ndarray, mode: str) -> "float | None":
    pos = float((y_train == 1).sum()); neg = float((y_train == 0).sum())
    if mode == "none" or pos == 0 or neg == 0:
        return None
    ratio = neg / pos
    return float(np.sqrt(ratio)) if mode == "sqrt" else float(ratio)


def make_model(d_tab: int, d_seq: int, ncfg: dict, kind: str = "mlp"):
    if kind == "linear":
        return LinearNet(d_tab)
    return PaperNet(d_tab, d_seq, tuple(ncfg["hidden"]), int(ncfg["emb_dim"]), float(ncfg["dropout"]), int(ncfg["seq_hidden"]))


def _loss_fn(pw):
    return nn.BCEWithLogitsLoss(pos_weight=None if pw is None else torch.tensor([pw], dtype=torch.float32))


def _fwd(model, data: NeuralData, idx):
    xs = data.Xs[idx] if data.Xs is not None else None
    return model(data.Xt[idx], xs)


@torch.no_grad()
def predict_logits(model, data: NeuralData, idx: np.ndarray, bs: int = 4096) -> np.ndarray:
    model.eval(); out = []
    for i in range(0, len(idx), bs):
        j = torch.as_tensor(idx[i:i + bs], dtype=torch.long)
        out.append(_fwd(model, data, j)[0].cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, dtype=np.float32)


@torch.no_grad()
def embed(model, data: NeuralData, idx: np.ndarray, bs: int = 4096):
    model.eval(); lo, em = [], []
    for i in range(0, len(idx), bs):
        j = torch.as_tensor(idx[i:i + bs], dtype=torch.long)
        l, e = _fwd(model, data, j); lo.append(l.cpu().numpy()); em.append(e.cpu().numpy())
    return np.concatenate(lo).astype(np.float32), np.concatenate(em).astype(np.float32)


def _metric(y, s, name):
    return fast_auroc(y, s) if name == "auroc" else auprc(y, s)


def lr_for_round(base: float, decay: float, r: int) -> float:
    """Multiplicative per-round learning-rate decay (decay=1.0 -> constant, the historical behaviour)."""
    return float(base) * float(decay) ** int(r)


def server_update(gstate: dict, avg: dict, velocity, beta: float = 0.0, server_lr: float = 1.0):
    """Server step after averaging. beta=0 and server_lr=1 return the plain FedAvg result unchanged;
    otherwise FedAvgM: v <- beta*v + (avg - global);  global <- global + server_lr * v."""
    if beta == 0.0 and server_lr == 1.0:
        return avg, velocity
    new, vel = {}, {}
    for k, g in gstate.items():
        if g.is_floating_point():
            d = avg[k] - g
            v = d if velocity is None else beta * velocity[k] + d
            vel[k] = v; new[k] = g + server_lr * v
        else:
            new[k] = avg[k]
    return new, vel


def _train_epochs(model, data, idx, ncfg, loss_fn, rng, epochs, global_params=None, mu=0.0, opt=None, max_steps=None):
    opt = opt or torch.optim.Adam(model.parameters(), lr=float(ncfg["lr"]), weight_decay=float(ncfg["weight_decay"]))
    bs = int(ncfg["batch_size"]); y = torch.as_tensor(data.y)
    model.train(); steps = 0
    for _ in range(epochs):
        perm = rng.permutation(idx)
        for i in range(0, len(perm), bs):
            j = torch.as_tensor(perm[i:i + bs], dtype=torch.long)
            if len(j) < 2:
                continue
            opt.zero_grad()
            logit, _ = _fwd(model, data, j)
            loss = loss_fn(logit, y[j])
            if mu > 0 and global_params is not None:
                loss = loss + 0.5 * mu * sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
            loss.backward(); opt.step(); steps += 1
            if max_steps and steps >= int(max_steps):
                return opt
    return opt


def train_central(data: NeuralData, train_idx, val_idx, d_tab, d_seq, ncfg, seed, epochs=None):
    """Pooled (NON-federated) reference. Early-stops on validation."""
    set_seed(seed); rng = np.random.default_rng(seed)
    model = make_model(d_tab, d_seq, ncfg); loss_fn = _loss_fn(pos_weight_value(data.y[train_idx], ncfg["pos_weight"]))
    opt = torch.optim.Adam(model.parameters(), lr=float(ncfg["lr"]), weight_decay=float(ncfg["weight_decay"]))
    best, best_state, bad, hist = -np.inf, None, 0, []
    for ep in range(int(epochs or ncfg["central_epochs"])):
        _train_epochs(model, data, train_idx, ncfg, loss_fn, rng, 1, opt=opt)
        m = _metric(data.y[val_idx], predict_logits(model, data, val_idx), ncfg["selection_metric"])
        hist.append(float(m))
        if m > best + 1e-5:
            best, bad, best_state = m, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= int(ncfg["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_history": hist, "best_val": float(best), "epochs_run": len(hist)}


def run_fedavg(data: NeuralData, train_by_hosp: dict, val_idx, d_tab, d_seq, ncfg, fcfg, seed, kind: str = "mlp"):
    """FedAvg: every round each hospital trains on ITS OWN train rows only (mini-batch, local_epochs),
    server takes the n_train-weighted mean of weights. One SHARED model (aligned embedding space)."""
    set_seed(seed); rng = np.random.default_rng(seed)
    gmodel = make_model(d_tab, d_seq, ncfg, kind); local = make_model(d_tab, d_seq, ncfg, kind)
    ncfg = dict(ncfg)                                    # federated-specific optimisation overrides
    for k in ("batch_size", "lr"):
        if fcfg.get(k) is not None:
            ncfg[k] = fcfg[k]
    all_train = np.concatenate([v for v in train_by_hosp.values()]) if train_by_hosp else np.zeros(0, dtype=np.int64)
    loss_fn = _loss_fn(pos_weight_value(data.y[all_train], ncfg["pos_weight"]))
    clients = [(h, idx) for h, idx in train_by_hosp.items() if len(idx) >= 2]
    total_w = float(sum(len(i) for _, i in clients))
    eff = int(fcfg["local_epochs"]) * sum(len(i) ** 2 for _, i in clients) / max(total_w, 1.0) / int(ncfg["batch_size"])
    logger.info("[FEDAVG] %d clients, %.0f train rows; ~%.0f n-weighted sequential steps per round (x %d rounds)", len(clients), total_w, eff, int(fcfg["rounds"]))
    best, best_state, bad, hist = -np.inf, None, 0, []
    velocity = None; max_steps = fcfg.get("max_local_steps"); beta = float(fcfg.get("server_momentum", 0.0)); slr = float(fcfg.get("server_lr", 1.0))
    for r in range(int(fcfg["rounds"])):
        ncfg_r = dict(ncfg, lr=lr_for_round(ncfg["lr"], fcfg.get("lr_decay", 1.0), r))
        gstate = copy.deepcopy(gmodel.state_dict()); acc = {k: torch.zeros_like(v, dtype=torch.float32) if v.is_floating_point() else v.clone() for k, v in gstate.items()}
        for h, idx in clients:
            local.load_state_dict(gstate)
            gp = [p.detach().clone() for p in local.parameters()] if fcfg["fedprox_mu"] > 0 else None
            _train_epochs(local, data, idx, ncfg_r, loss_fn, rng, int(fcfg["local_epochs"]), gp, float(fcfg["fedprox_mu"]), max_steps=max_steps)
            w = len(idx) / total_w
            for k, v in local.state_dict().items():
                if v.is_floating_point():
                    acc[k] += w * v.detach()
        new_state, velocity = server_update(gstate, acc, velocity, beta, slr)
        gmodel.load_state_dict(new_state)
        m = _metric(data.y[val_idx], predict_logits(gmodel, data, val_idx), ncfg["selection_metric"])
        hist.append(float(m))
        logger.info("[FEDAVG] round %d/%d val_%s=%.4f", r + 1, fcfg["rounds"], ncfg["selection_metric"], m)
        if m > best + 1e-5:
            best, bad, best_state = m, 0, copy.deepcopy(gmodel.state_dict())
        else:
            bad += 1
            if bad >= int(fcfg["patience"]):
                break
    if best_state is not None:
        gmodel.load_state_dict(best_state)
    return gmodel, {"val_history": hist, "best_val": float(best), "rounds_run": len(hist), "n_clients": len(clients)}


def train_local_only(data: NeuralData, hosp_rows: dict, d_tab, d_seq, ncfg, seed):
    """Each hospital trains its OWN model (never averaged). Returns logits for every row of every hospital."""
    scores = np.full(len(data.y), np.nan, dtype=np.float32)
    for h, (tr, va, rows) in hosp_rows.items():
        set_seed(seed + int(h)); rng = np.random.default_rng(seed + int(h))
        model = make_model(d_tab, d_seq, ncfg)
        if len(tr) >= 2:
            lf = _loss_fn(pos_weight_value(data.y[tr], ncfg["pos_weight"])); opt = None
            best, best_state, bad = -np.inf, None, 0
            for ep in range(int(ncfg["local_epochs_local_only"])):
                opt = _train_epochs(model, data, tr, ncfg, lf, rng, 1, opt=opt)
                if len(va) >= 2 and len(np.unique(data.y[va])) == 2:
                    m = _metric(data.y[va], predict_logits(model, data, va), ncfg["selection_metric"])
                    if m > best + 1e-5:
                        best, bad, best_state = m, 0, copy.deepcopy(model.state_dict())
                    else:
                        bad += 1
                        if bad >= int(ncfg["patience"]):
                            break
            if best_state is not None:
                model.load_state_dict(best_state)
        scores[rows] = predict_logits(model, data, rows)
    return scores


# ------------------------------- graph stage -------------------------------

def build_graphs(emb: np.ndarray, hosp: np.ndarray, patient_ids: np.ndarray, split: np.ndarray, gcfg: dict, method: str) -> dict:
    """One graph PER HOSPITAL from that hospital's own embeddings only (no cross-hospital edges).
    method: 'none' | 'knn_and_threshold' (Graph A) | 'minibatch_kmeans_and_threshold' (Graph B)."""
    from src.graph.patient_graph import build_hospital_graph
    graphs = {}
    for h in np.unique(hosp):
        rows = np.where(hosp == h)[0]
        if method == "none" or len(rows) < 3:
            ei = np.zeros((2, 0), dtype=np.int64)
        else:
            g = build_hospital_graph(list(patient_ids[rows]), emb[rows], method=method, k=int(gcfg["k"]),
                                     cosine_threshold=float(gcfg["cosine_threshold"]),
                                     max_nodes_full_similarity=int(gcfg["max_nodes_full_similarity"]),
                                     n_clusters=int(min(gcfg["n_clusters"], max(len(rows) // 2, 1))), cluster_seed=int(gcfg["cluster_seed"]))
            ei = np.asarray(g["edge_index"], dtype=np.int64)
        sp = split[rows]; tr_local = np.where(sp == "train")[0]
        remap = -np.ones(len(rows), dtype=np.int64); remap[tr_local] = np.arange(len(tr_local))
        keep = (remap[ei[0]] >= 0) & (remap[ei[1]] >= 0) if ei.shape[1] else np.zeros(0, dtype=bool)
        tr_ei = np.stack([remap[ei[0][keep]], remap[ei[1][keep]]]) if keep.any() else np.zeros((2, 0), dtype=np.int64)
        graphs[int(h)] = {"rows": rows, "ei": torch.as_tensor(ei), "train_local": tr_local, "train_ei": torch.as_tensor(tr_ei), "n_edges": int(ei.shape[1])}
    return graphs


def run_graph_fedavg(emb, base_logit, y, graphs, split, gcfg, ncfg, seed, pw):
    """FedAvg of a GraphHead. Training: train-nodes-only subgraph, loss on train labels. Inference: the hospital's full graph."""
    set_seed(seed)
    E = torch.as_tensor(emb, dtype=torch.float32); B = torch.as_tensor(base_logit, dtype=torch.float32); Y = torch.as_tensor(y)
    gm = GraphHead(emb.shape[1], int(gcfg["hidden"]), int(gcfg["layers"]), float(gcfg["dropout"]), bool(gcfg["residual"]))
    local = copy.deepcopy(gm); loss_fn = _loss_fn(pw)
    clients = [(h, g) for h, g in graphs.items() if len(g["train_local"]) >= 2]
    total_w = float(sum(len(g["train_local"]) for _, g in clients))
    val_rows = np.where(split == "val")[0]

    @torch.no_grad()
    def infer_all(model):
        model.eval(); out = np.zeros(len(y), dtype=np.float32)
        for h, g in graphs.items():
            r = torch.as_tensor(g["rows"]); out[g["rows"]] = model(E[r], g["ei"], B[r]).numpy()
        return out

    def val_metric(model):
        s = infer_all(model); return _metric(y[val_rows], s[val_rows], ncfg["selection_metric"]), s

    best, best_state = val_metric(gm)[0], copy.deepcopy(gm.state_dict()); hist = [float(best)]; best_round, bad = 0, 0
    for r in range(int(gcfg["rounds"])):
        gstate = copy.deepcopy(gm.state_dict()); acc = {k: torch.zeros_like(v) for k, v in gstate.items()}
        for h, g in clients:
            local.load_state_dict(gstate); local.train()
            opt = torch.optim.Adam(local.parameters(), lr=float(gcfg["lr"]), weight_decay=float(ncfg["weight_decay"]))
            rows = torch.as_tensor(g["rows"][g["train_local"]])
            for _ in range(int(gcfg["local_epochs"])):
                opt.zero_grad(); loss_fn(local(E[rows], g["train_ei"], B[rows]), Y[rows]).backward(); opt.step()
            w = len(g["train_local"]) / total_w
            for k, v in local.state_dict().items():
                acc[k] += w * v.detach()
        gm.load_state_dict(acc)
        m, _ = val_metric(gm); hist.append(float(m))
        if m > best + 1e-5:
            best, best_round, bad, best_state = m, r + 1, 0, copy.deepcopy(gm.state_dict())
        else:
            bad += 1
            if bad >= 8:
                break
    gm.load_state_dict(best_state)
    n_edges = sum(g["n_edges"] for g in graphs.values())
    return infer_all(gm), {"val_history": hist, "best_val": float(best), "best_round": best_round, "n_edges": int(n_edges)}
