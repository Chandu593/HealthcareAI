"""Models, per-client training problems, local updates and server aggregation for the parameter-averaging algorithms: FedAvg, FedProx, FedAdam, SCAFFOLD.

These apply ONLY to models whose parameters live in one shared space (LR, MLP, GCN). Tree ensembles are never averaged: they use the separate federated-histogram GBDT (`fl_gbdt_adapter.py`).

Notation (round t, client i, weights p_i = n_i / sum n, n_i = number of REAL training patients of client i, w = global parameters, K local steps, eta_l local learning rate):
  local FedAvg       w_i <- K SGD steps from w on  F_i(w)
  local FedProx      same, on  F_i(w) + (mu/2) ||w - w_t||^2
  local SCAFFOLD     SGD with gradient  g - c_i + c ;  option II:  c_i+ = c_i - c + (w_t - w_i) / (K eta_l) ;  server  c <- c + sum_i p_i (c_i+ - c_i)
  server avg         w <- w + sum_i p_i (w_i - w)                                (FedAvg / FedProx / SCAFFOLD, server lr 1)
  server FedAdam     D = sum_i p_i (w_i - w);  m <- b1 m + (1-b1) D;  v <- b2 v + (1-b2) D^2;  w <- w + eta_s m / (sqrt(v) + tau)       (Reddi et al. 2021, no bias correction)
SCAFFOLD here uses the sample-weighted objective sum_i p_i F_i, so its server control variate is the p-weighted mean of the client variates (uniform p_i recovers the original paper).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.three_modality.hw_classifiers import focal_loss_torch, resolve_alpha
from src.three_modality.fl_clients import global_alpha as _global_alpha
from src.three_modality.hw_data import stable_seed
from src.three_modality.hw_graph import attach_edges, fit_train_graph, generate_synthetic_positives, make_gcn


# ---------------------------------------------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------------------------------------------
class FLLR(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        return self.head(x).squeeze(-1)


class FLMLP(nn.Module):
    def __init__(self, d, hidden, dropout):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


def build_model(family: str, in_dim: int, ds_cfg: dict, seed: int):
    """Freshly initialised model; the seed fixes the shared initial weights (every client starts from the SAME point)."""
    torch.manual_seed(seed)
    if family == "lr":
        return FLLR(in_dim)
    if family == "mlp":
        c = ds_cfg["classifiers"]["mlp"]
        return FLMLP(in_dim, int(c["hidden"]), float(c["dropout"]))
    if family == "gcn":
        return make_gcn(in_dim, ds_cfg["gcn"])
    raise ValueError(f"unknown family {family!r}")


def get_params(model) -> list:
    return [p.detach().clone() for p in model.parameters()]


def set_params(model, params: list) -> None:
    with torch.no_grad():
        for p, q in zip(model.parameters(), params):
            p.copy_(q)


def param_names(model) -> list:
    return [n for n, _ in model.named_parameters()]


def head_indices(model) -> list:
    """Indices of the classification-head parameters (the personalised part). LR is all head."""
    return [i for i, n in enumerate(param_names(model)) if n.startswith("head.")]


def body_indices(model) -> list:
    h = set(head_indices(model))
    return [i for i in range(len(param_names(model))) if i not in h]


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------------------------------------------------------------------
# per-client problem (data tensors + hospital-local graph)
# ---------------------------------------------------------------------------------------------------------------------------------
def _edge_tensors(tg, q_src, q_dst, q_w):
    """Train edges + directed (train -> query) edges; query nodes are indexed n_train + j."""
    ei = np.concatenate([tg.edge_index, np.stack([q_src, tg.n + q_dst])], axis=1) if len(q_src) else tg.edge_index
    ew = np.concatenate([tg.edge_weight, q_w]) if len(q_w) else tg.edge_weight
    return torch.from_numpy(ei.astype(np.int64)), torch.from_numpy(ew.astype(np.float32))


class ClientProblem:
    """Everything one client needs to compute a local loss and to score its own patients. Built from the client's OWN data only; holds no label of the test split."""

    def __init__(self, client, family: str, ds_cfg: dict, graph_key: "str | None" = None, smote: bool = False):
        self.client, self.family, self.smote, self.graph_key = client, family, smote, graph_key
        self.ytr = torch.from_numpy(client.y["train"].astype(np.int64))
        self.yva = torch.from_numpy(client.y["val"].astype(np.int64))
        self._x = {"train": torch.from_numpy(client.scaled("train")), "val": torch.from_numpy(client.scaled("val"))}
        self.info = {"hospitalid": client.hospitalid}
        if family == "gcn":
            if graph_key is None:
                raise ValueError("the GCN family needs a graph key (A or B)")
            Ztr, ytr = client.Z["train"], client.y["train"]
            self.tg = fit_train_graph(Ztr, graph_key, ds_cfg["graph"])
            self.n_tr = self.tg.n
            Zs, sinfo = (generate_synthetic_positives(Ztr, ytr, ds_cfg["smote"], client.seed) if smote else (None, {"n_synthetic": 0, "skipped": "smote off"}))
            m = 0 if Zs is None else int(Zs.shape[0])
            if m:
                s_src, s_dst, s_w = attach_edges(self.tg, Zs)                 # synthetic nodes get edges FROM real training nodes only, never to or from other clients
                sinfo["synthetic_nodes_with_no_edge"] = int(m - len(np.unique(s_dst)))
                sinfo["synthetic_edges"] = int(len(s_src))
                self.x_graph_train = torch.cat([self._x["train"], torch.from_numpy(client.scaler.transform(Zs).astype(np.float32))])
            else:
                s_src = s_dst = np.empty(0, np.int64)
                s_w = np.empty(0, np.float32)
                self.x_graph_train = self._x["train"]
            self.ei_tr, self.ew_tr = _edge_tensors(self.tg, s_src, s_dst, s_w)
            self.y_nodes = torch.from_numpy(np.concatenate([ytr, np.ones(m, dtype=np.int64)]))
            self.n_synth = m
            self._q = {}
            self.info.update({"train_nodes": self.n_tr, "train_edges_directed": int(self.tg.edge_index.shape[1]), "smote": sinfo})
        else:
            self.n_tr = client.n_train
            self.y_nodes = self.ytr
            self.n_synth = 0

    # ---- inputs ----
    def x(self, split: str) -> torch.Tensor:
        if split not in self._x:
            self._x[split] = torch.from_numpy(self.client.scaled(split))     # test FEATURES are materialised lazily, only when the evaluator asks
        return self._x[split]

    def _query_graph(self, split: str):
        """Train graph + the split's patients attached inductively (directed train -> query edges, no query-query edges). Built once per split and cached."""
        if split not in self._q:
            src, dst, w = attach_edges(self.tg, self.client.Z[split])
            ei, ew = _edge_tensors(self.tg, src, dst, w)
            self._q[split] = (torch.cat([self._x["train"], self.x(split)]), ei, ew, int(len(src)))
        return self._q[split]

    # ---- forward ----
    def logits(self, model, split: str) -> torch.Tensor:
        """Logits of this client's own patients of `split` (train: real + synthetic nodes for the GCN)."""
        if self.family != "gcn":
            return model(self.x(split))
        if split == "train":
            _, lg = model(self.x_graph_train, self.ei_tr, self.ew_tr)
            return lg[: self.n_tr + self.n_synth]
        x, ei, ew, _ = self._query_graph(split)
        _, lg = model(x, ei, ew)
        return lg[self.n_tr:]

    def embeddings(self, model, split: str) -> np.ndarray:
        """GCN node embeddings (Config 4). Train-node embeddings come from the VALIDATION-attached graph, exactly as in the Non-FL regime (a train node's representation does not depend on the
        attached query nodes: edges point train -> query only)."""
        if self.family != "gcn":
            raise ValueError("embeddings are defined for the GCN family only")
        x, ei, ew, _ = self._query_graph("val" if split == "train" else split)
        h, _ = model(x, ei, ew)
        return (h[: self.n_tr] if split == "train" else h[self.n_tr:]).detach().numpy().astype(np.float64)

    def labels(self, split: str) -> torch.Tensor:
        if split == "train":
            return self.y_nodes
        if split == "val":
            return self.yva
        raise PermissionError("training code can only read train / validation labels")

    def attach_edge_count(self, split: str) -> int:
        return self._query_graph(split)[3] if self.family == "gcn" else 0


def client_alpha(client, ds_cfg: dict, fl_cfg: dict, global_a: float, smote: bool) -> float:
    return float(global_a) if fl_cfg["focal_alpha_mode"] == "global" else resolve_alpha(ds_cfg["focal"], client.y["train"], smote)



def make_problems(clients: list, family: str, ds_cfg: dict, graph_key=None, smote: bool = False) -> tuple:
    """ClientProblem per client; a client whose hospital-local graph cannot be built is excluded and REPORTED (never dropped silently)."""
    problems, failed = {}, []
    for c in clients:
        try:
            problems[c.hospitalid] = ClientProblem(c, family, ds_cfg, graph_key, smote)
        except Exception as exc:                  # noqa: BLE001
            failed.append({"hospitalid": c.hospitalid, "reason": f"hospital-local graph construction failed: {type(exc).__name__}: {exc}"})
    return problems, failed


def make_alphas(problems: dict, ds_cfg: dict, fl_cfg: dict, smote: bool) -> dict:
    clients = [p.client for p in problems.values()]
    g = _global_alpha(clients, ds_cfg["focal"], smote)
    return {h: client_alpha(p.client, ds_cfg, fl_cfg, g, smote) for h, p in problems.items()}



# ---------------------------------------------------------------------------------------------------------------------------------
# local update
# ---------------------------------------------------------------------------------------------------------------------------------
def local_update(model, prob: ClientProblem, w_start: list, alpha: float, gamma: float, *, algo: str, optimizer: str, lr: float, wd: float, phases: list, mu: float = 0.0,
                 c_global: "list | None" = None, c_local: "list | None" = None, w_anchor: "list | None" = None, seed: int = 0, grad_clip: "float | None" = None):
    """Run the client's local steps starting from `w_start` and return (w_local, stats), or (None, stats) when the loss/weights became non-finite (the client is then excluded from this
    round's aggregation and the event is counted -- never averaged in).

    `phases` = [(n_steps, active_param_indices), ...] (FedRep uses two phases; every other method one phase over all parameters). `w_anchor` is the FedProx anchor (the round's global
    parameters); it defaults to `w_start`. SCAFFOLD needs plain SGD and a single all-parameter phase."""
    if algo == "scaffold" and (optimizer != "sgd" or len(phases) != 1):
        raise ValueError("SCAFFOLD requires the plain-SGD local optimiser and a single full-parameter phase")
    params = list(model.parameters())
    set_params(model, w_start)
    anchor = w_anchor if w_anchor is not None else w_start
    torch.manual_seed(seed)
    y = prob.labels("train")
    steps_done, last_loss = 0, float("nan")
    for n_steps, active in phases:
        act = [params[i] for i in active]
        opt = torch.optim.Adam(act, lr=lr, weight_decay=wd) if optimizer == "adam" else None
        for _ in range(int(n_steps)):
            model.train()
            loss = focal_loss_torch(prob.logits(model, "train"), y, alpha, gamma)
            last_loss = float(loss.detach())
            if not np.isfinite(last_loss):
                return None, {"nonfinite": True, "steps": steps_done}
            total = loss
            if algo == "fedprox" and mu > 0:
                total = total + 0.5 * mu * sum(((params[i] - anchor[i]) ** 2).sum() for i in active)
            if opt is not None:
                opt.zero_grad()
                total.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(act, grad_clip)
                opt.step()
            else:
                grads = torch.autograd.grad(total, act)
                with torch.no_grad():
                    for i, g in zip(active, grads):
                        gt = g + wd * params[i]
                        if algo == "scaffold":
                            gt = gt - c_local[i] + c_global[i]
                        params[i].sub_(lr * gt)
            steps_done += 1
    w_local = get_params(model)
    if not all(bool(torch.isfinite(t).all()) for t in w_local):
        return None, {"nonfinite": True, "steps": steps_done}
    return w_local, {"nonfinite": False, "steps": steps_done, "last_train_loss": last_loss}


# ---------------------------------------------------------------------------------------------------------------------------------
# server side
# ---------------------------------------------------------------------------------------------------------------------------------
def client_weights(n_train: list) -> np.ndarray:
    """p_i = n_i / sum n (McMahan et al.). Raises on an empty / non-positive federation."""
    n = np.asarray(n_train, dtype=np.float64)
    if n.size == 0 or n.sum() <= 0 or (n < 0).any():
        raise ValueError("federation has no training weight")
    return n / n.sum()


def weighted_delta(w: list, locals_: list, p: np.ndarray, indices: "list | None" = None) -> list:
    """D = sum_i p_i (w_i - w) over `indices` (default: all parameters)."""
    idx = list(range(len(w))) if indices is None else list(indices)
    return [sum(float(p[i]) * (loc[j] - w[j]) for i, loc in enumerate(locals_)) for j in idx]


def apply_server_update(algo: str, w: list, delta: list, state: dict, server_cfg: dict, indices: "list | None" = None) -> list:
    """New global parameters for `indices` (default all). FedAvg / FedProx / SCAFFOLD: w + D. FedAdam: server Adam on the pseudo-gradient D (state['m'], state['v'] are updated in place)."""
    idx = list(range(len(w))) if indices is None else list(indices)
    if algo in ("fedavg", "fedprox", "scaffold"):
        return [w[j] + d for j, d in zip(idx, delta)]
    if algo == "fedadam":
        b1, b2, tau, eta = float(server_cfg["beta1"]), float(server_cfg["beta2"]), float(server_cfg["tau"]), float(server_cfg["server_lr"])
        if state.get("m") is None:
            state["m"] = [torch.zeros_like(d) for d in delta]
            state["v"] = [torch.zeros_like(d) for d in delta]
        out = []
        for k, (j, d) in enumerate(zip(idx, delta)):
            state["m"][k] = b1 * state["m"][k] + (1.0 - b1) * d
            state["v"][k] = b2 * state["v"][k] + (1.0 - b2) * d * d
            out.append(w[j] + eta * state["m"][k] / (state["v"][k].sqrt() + tau))
        return out
    raise ValueError(f"unknown algorithm {algo!r}")


def scaffold_client_control(c_i: list, c: list, w_global: list, w_local: list, K: int, lr: float) -> list:
    """Option II:  c_i+ = c_i - c + (w_t - w_i) / (K eta_l)."""
    return [ci - cg + (wg - wl) / (K * lr) for ci, cg, wg, wl in zip(c_i, c, w_global, w_local)]


def scaffold_server_control(c: list, deltas_ci: list, p: np.ndarray) -> list:
    """c <- c + sum_i p_i (c_i+ - c_i)."""
    return [c[j] + sum(float(p[i]) * dc[j] for i, dc in enumerate(deltas_ci)) for j in range(len(c))]


def round_seed(base: int, rnd: int, hospitalid: int, tag: str = "local") -> int:
    return stable_seed(base, "fl_round", tag, rnd, hospitalid)
