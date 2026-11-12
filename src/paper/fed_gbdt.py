"""Federated histogram gradient boosting (horizontal FL, hospital = client). numpy only.

Why this exists. On the strict cohort the federated MLP (0.8617) sits ~0.018 AUROC below the pooled LightGBM
(0.8796): the gap is the model family, not the federation (federated MLP vs central MLP is only ~0.007). Tree
ensembles can be trained federatedly WITHOUT pooling raw rows if the clients only ever send gradient/hessian
HISTOGRAMS, which is the horizontal histogram protocol used by federated XGBoost/FedTree-style systems:

  1. bin edges   : each client sends a small per-feature quantile summary of ITS OWN TRAIN rows; the server merges them
                   (weighted by the client's non-missing count) into global edges. No raw values leave a client.
  2. every round : the server broadcasts the current tree frontier; each client computes (sum g, sum h) per
                   (frontier node, feature, bin) over its own train rows and sends them; the server SUMS them, picks the
                   best split per node, and broadcasts it. Leaf values come from summed (G, H).
  3. validation  : early stopping on the validation AUROC pooled at the server -- the same simulation shortcut that
                   `run_fedavg` uses for fedavg_mlp (a deployment would aggregate a validation loss instead).

Exactness. Summing client histograms is, up to float summation order, identical to the histogram of the pooled rows,
so the learned trees equal those of a pooled histogram GBDT trained on the same bin edges. That equivalence is
verified in tests/unit/test_paper_fed_gbdt.py (partition invariance and an explicit per-client protocol path). What is
NOT claimed: any privacy guarantee. Gradient histograms disclose more than a weight average, and there is no
differential privacy or secure aggregation here (same boundary as every other arm in the paper track).

Leakage rules honoured (same as the rest of the paper track): bin edges, base score, class counts and every split are
computed from TRAIN rows only; validation rows are used only for early stopping / trial selection; test rows are only
ever binned with the train edges and scored.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_LOG = logging.getLogger("eicu_fl")

FIXED_DEFAULTS = {
    "learning_rate": 0.08, "max_depth": 6, "reg_lambda": 10.0, "min_child_weight": 3.0, "min_gain": 0.0,
    "colsample_bytree": 0.6, "subsample": 1.0, "scale_pos_weight_mode": "none",
}


# --------------------------------------------------------------------------------------------- binning
def local_quantile_summary(Xk: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What ONE client sends for binning: per-feature quantiles of its own train rows (shape (Q, F)) and its
    per-feature non-missing counts (F,). Features with no observed value get NaN quantiles."""
    cnt = np.sum(~np.isnan(Xk), axis=0).astype("int64")
    q = np.full((len(grid), Xk.shape[1]), np.nan, dtype="float64")
    ok = cnt > 0
    if ok.any():
        q[:, ok] = np.nanquantile(Xk[:, ok].astype("float64"), grid, axis=0)
    return q, cnt


def merge_quantile_summaries(summaries: list, n_bins: int) -> list:
    """Server side: weighted merge of client summaries into per-feature bin edges (sorted unique values, <= n_bins-1).
    Edges are snapped to observed quantile values (no interpolation), so a split `x <= edge` is always a real threshold."""
    F = summaries[0][0].shape[1]
    probs = np.arange(1, n_bins, dtype="float64") / n_bins
    edges = []
    for j in range(F):
        vals, wts = [], []
        for q, cnt in summaries:
            if cnt[j] > 0:
                vals.append(q[:, j]); wts.append(np.full(q.shape[0], cnt[j] / q.shape[0], dtype="float64"))
        if not vals:
            edges.append(np.zeros(0, dtype="float64")); continue
        v = np.concatenate(vals); w = np.concatenate(wts)
        o = np.argsort(v, kind="mergesort"); v, w = v[o], w[o]
        cw = np.cumsum(w) / w.sum()
        pick = np.minimum(np.searchsorted(cw, probs, side="left"), len(v) - 1)
        edges.append(np.unique(v[pick]))
    return edges


def federated_bin_edges(X: np.ndarray, client_rows: dict, n_bins: int, grid_size: int = 128) -> list:
    """Bin edges from TRAIN rows only, built through the client->server summary exchange above."""
    grid = (np.arange(grid_size, dtype="float64") + 0.5) / grid_size
    summaries = [local_quantile_summary(X[np.asarray(rows)], grid) for rows in client_rows.values() if len(rows) > 0]
    return merge_quantile_summaries(summaries, n_bins)


def bin_features(X: np.ndarray, edges: list, n_bins: int) -> np.ndarray:
    """(N, F) uint8 bin index with `x <= edges[j][b]` semantics; missing values go to the dedicated bin `n_bins`."""
    out = np.empty(X.shape, dtype=np.uint8)
    for j, e in enumerate(edges):
        col = X[:, j]
        b = np.searchsorted(e, col, side="left") if len(e) else np.zeros(len(col), dtype="int64")
        b = np.minimum(b, n_bins - 1)
        b[np.isnan(col)] = n_bins
        out[:, j] = b
    return out


# --------------------------------------------------------------------------------------------- model
@dataclass
class _Tree:
    feature: np.ndarray      # int32, -1 for a leaf
    thr: np.ndarray          # int32 bin threshold (go left if bin <= thr)
    miss_left: np.ndarray    # bool, send missing left
    left: np.ndarray         # int32
    right: np.ndarray        # int32
    value: np.ndarray        # float64, learning rate already applied (leaves only)


@dataclass
class FedGBDT:
    edges: list
    n_bins: int
    base_score: float
    trees: list = field(default_factory=list)
    gain_by_feature: "np.ndarray | None" = None
    params: dict = field(default_factory=dict)

    def _tree_margin(self, tree: _Tree, binned: np.ndarray) -> np.ndarray:
        n = binned.shape[0]
        node = np.zeros(n, dtype=np.int64)
        for _ in range(64):
            f = tree.feature[node]
            internal = np.flatnonzero(f >= 0)
            if internal.size == 0:
                break
            nd = node[internal]
            b = binned[internal, f[internal]].astype(np.int64)
            go_left = (b <= tree.thr[nd]) | ((b == self.n_bins) & tree.miss_left[nd])
            node[internal] = np.where(go_left, tree.left[nd], tree.right[nd])
        return tree.value[node]

    def predict_raw(self, X: np.ndarray, n_trees: "int | None" = None) -> np.ndarray:
        binned = bin_features(X, self.edges, self.n_bins)
        return self._predict_binned(binned, n_trees)

    def _predict_binned(self, binned: np.ndarray, n_trees: "int | None" = None) -> np.ndarray:
        out = np.full(binned.shape[0], self.base_score, dtype="float64")
        for t in (self.trees if n_trees is None else self.trees[:n_trees]):
            out += self._tree_margin(t, binned)
        return out


# --------------------------------------------------------------------------------------------- training
def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def client_histograms(bins_col, g, h, client_idx, node_local, n_clients, n_nodes, n_bins1):
    """The LITERAL client message for one feature: (K, n_nodes, n_bins1) sums of g and of h, one slab per client."""
    base = (client_idx.astype(np.int64) * n_nodes + node_local) * n_bins1 + bins_col.astype(np.int64)
    size = n_clients * n_nodes * n_bins1
    hg = np.bincount(base, weights=g, minlength=size).reshape(n_clients, n_nodes, n_bins1)
    hh = np.bincount(base, weights=h, minlength=size).reshape(n_clients, n_nodes, n_bins1)
    return hg, hh


def _grow_tree(bins_T, g, h, cid, n_clients, sel_cols, n_bins, p, explicit, gain_acc, comm):
    """One level-wise tree from aggregated histograms. Returns (tree, leaf_id_per_train_row)."""
    B1 = n_bins + 1
    lam, mcw, gamma, lr, max_depth = p["reg_lambda"], p["min_child_weight"], p["min_gain"], p["learning_rate"], int(p["max_depth"])
    n_tr = len(g)
    feature, thr, missl, left, right = [-1], [0], [False], [-1], [-1]
    pos = np.zeros(n_tr, dtype=np.int64)               # position of each row's node inside the current frontier, -1 = finished
    leaf_of_row = np.zeros(n_tr, dtype=np.int64)
    frontier = [0]
    for depth in range(max_depth):
        n_front = len(frontier)
        ridx = np.flatnonzero(pos >= 0)
        if ridx.size == 0:
            break
        full = ridx.size == n_tr
        ga, ha, nl = (g, h, pos) if full else (g[ridx], h[ridx], pos[ridx])
        ca = cid if full else cid[ridx]
        Fs = len(sel_cols)
        Hg = np.empty((n_front, Fs, B1)); Hh = np.empty((n_front, Fs, B1))
        for jj, f in enumerate(sel_cols):
            col = bins_T[f] if full else bins_T[f][ridx]
            if explicit:                                # literal protocol: per-client slabs, then the server sums them
                cg, ch = client_histograms(col, ga, ha, ca, nl, n_clients, n_front, B1)
                Hg[:, jj, :], Hh[:, jj, :] = cg.sum(axis=0), ch.sum(axis=0)
            else:                                       # identical sum, without materialising the K slabs
                idx = nl * B1 + col.astype(np.int64)
                Hg[:, jj, :] = np.bincount(idx, weights=ga, minlength=n_front * B1).reshape(n_front, B1)
                Hh[:, jj, :] = np.bincount(idx, weights=ha, minlength=n_front * B1).reshape(n_front, B1)
        # communication accounting (what a deployment would send, never used for training): a dense message is every cell as two float32;
        # a sparse message only carries cells a client actually touched, at most (its active rows x selected features) per level, 12 bytes each
        cnt = np.bincount(ca, minlength=n_clients).astype("float64")
        comm["dense"] += float(n_clients) * n_front * Fs * B1 * 8
        comm["sparse"] += float(np.minimum(n_front * Fs * B1, cnt * Fs).sum()) * 12
        Gn, Hn = Hg[:, 0, :].sum(axis=1), Hh[:, 0, :].sum(axis=1)             # node totals (identical for every feature)
        cG = np.cumsum(Hg[:, :, : n_bins - 1], axis=2); cH = np.cumsum(Hh[:, :, : n_bins - 1], axis=2)
        Gm, Hm = Hg[:, :, n_bins][:, :, None], Hh[:, :, n_bins][:, :, None]
        parent = (Gn ** 2 / (Hn + lam))[:, None, None]
        best = None
        for miss_left in (True, False):
            GL = cG + Gm if miss_left else cG
            HL = cH + Hm if miss_left else cH
            GR, HR = Gn[:, None, None] - GL, Hn[:, None, None] - HL
            gain = 0.5 * (GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - parent) - gamma
            gain = np.where((HL >= mcw) & (HR >= mcw), gain, -np.inf)
            flat = gain.reshape(n_front, -1)
            arg = flat.argmax(axis=1); val = flat[np.arange(n_front), arg]
            if best is None:
                best = [val, arg, np.full(n_front, miss_left)]
            else:
                better = val > best[0]
                best[0] = np.where(better, val, best[0]); best[1] = np.where(better, arg, best[1]); best[2] = np.where(better, miss_left, best[2])
        bgain, barg, bmiss = best
        split = np.flatnonzero(bgain > 0)
        if split.size == 0:
            break
        new_frontier, child_pos_left, child_pos_right = [], np.full(n_front, -1, dtype=np.int64), np.full(n_front, -1, dtype=np.int64)
        node_feat = np.full(n_front, -1, dtype=np.int64); node_thr = np.zeros(n_front, dtype=np.int64); node_miss = np.zeros(n_front, dtype=bool)
        for k in split:
            fj, t = divmod(int(barg[k]), n_bins - 1)
            gid = frontier[k]
            lid, rid = len(feature), len(feature) + 1
            for _ in range(2):
                feature.append(-1); thr.append(0); missl.append(False); left.append(-1); right.append(-1)
            feature[gid], thr[gid], missl[gid], left[gid], right[gid] = int(sel_cols[fj]), t, bool(bmiss[k]), lid, rid
            gain_acc[int(sel_cols[fj])] += float(bgain[k])
            child_pos_left[k], child_pos_right[k] = len(new_frontier), len(new_frontier) + 1
            new_frontier += [lid, rid]
            node_feat[k], node_thr[k], node_miss[k] = int(sel_cols[fj]), t, bool(bmiss[k])
        # route this level's rows
        kk = pos[ridx]
        is_split = node_feat[kk] >= 0
        rows_s = ridx[is_split]; ks = kk[is_split]
        b = bins_T[node_feat[ks], rows_s].astype(np.int64)
        go_left = (b <= node_thr[ks]) | ((b == n_bins) & node_miss[ks])
        new_pos = np.where(go_left, child_pos_left[ks], child_pos_right[ks])
        done = ridx[~is_split]
        leaf_of_row[done] = np.asarray(frontier, dtype=np.int64)[kk[~is_split]]
        pos[done] = -1
        pos[rows_s] = new_pos
        frontier = new_frontier
    # rows still active at max depth, or whose node never split, end in the frontier node they sit in
    rem = np.flatnonzero(pos >= 0)
    if rem.size:
        leaf_of_row[rem] = np.asarray(frontier, dtype=np.int64)[pos[rem]]
    n_nodes = len(feature)
    Gl = np.bincount(leaf_of_row, weights=g, minlength=n_nodes); Hl = np.bincount(leaf_of_row, weights=h, minlength=n_nodes)
    value = np.where(np.asarray(feature) < 0, -lr * Gl / (Hl + lam), 0.0)
    tree = _Tree(np.asarray(feature, dtype=np.int32), np.asarray(thr, dtype=np.int32), np.asarray(missl, dtype=bool),
                 np.asarray(left, dtype=np.int32), np.asarray(right, dtype=np.int32), value.astype("float64"))
    return tree, leaf_of_row


def train_fed_gbdt(X, y, client_rows: dict, val_rows: np.ndarray, params: dict, seed: int, *, n_bins: int = 64,
                   grid_size: int = 128, max_rounds: int = 600, early_stopping_rounds: "int | None" = 40,
                   explicit_client_histograms: bool = False, edges: "list | None" = None, log_every: int = 25):
    """Train one federated GBDT. `client_rows` maps hospital -> TRAIN row indices; `val_rows` is used only for early stopping.
    Returns (model truncated to the best round, history dict)."""
    t0 = time.time()
    p = dict(FIXED_DEFAULTS, **(params or {}))
    if not 2 <= n_bins <= 254:
        raise ValueError("n_bins must be in [2, 254] (uint8 bins, one value reserved for missing)")
    ids = [h for h, r in client_rows.items() if len(r) > 0]
    tr = np.concatenate([np.asarray(client_rows[h], dtype=np.int64) for h in ids])
    cid = np.repeat(np.arange(len(ids)), [len(client_rows[h]) for h in ids])
    ytr = y[tr].astype("float64")
    if len(np.unique(ytr)) < 2:
        raise ValueError("federated GBDT needs both classes in the pooled train rows")
    if edges is None:
        edges = federated_bin_edges(X, {h: client_rows[h] for h in ids}, n_bins, grid_size)
    binned = bin_features(X, edges, n_bins)                                  # val/test rows are binned with TRAIN edges only
    bins_T = np.ascontiguousarray(binned[tr].T)
    n_pos, n_neg = float(ytr.sum()), float(len(ytr) - ytr.sum())             # aggregated class counts (scalars), as a client would send
    mode = p["scale_pos_weight_mode"]
    spw = 1.0 if mode == "none" else (math.sqrt(n_neg / n_pos) if mode == "sqrt" else n_neg / n_pos)
    w = np.where(ytr > 0.5, spw, 1.0)
    prior = (spw * n_pos) / (spw * n_pos + n_neg)
    base = float(math.log(prior / (1 - prior)))
    model = FedGBDT(edges=edges, n_bins=n_bins, base_score=base, gain_by_feature=np.zeros(X.shape[1]), params=dict(p))
    F_tr = np.full(len(tr), base, dtype="float64")
    vb = binned[val_rows]; yv = y[val_rows]
    F_val = np.full(len(val_rows), base, dtype="float64")
    rng = np.random.default_rng(seed)
    n_feat = X.shape[1]; n_sel = max(1, int(round(p["colsample_bytree"] * n_feat)))
    best, best_it, since, curve = -np.inf, 0, 0, []
    comm = {"dense": 0.0, "sparse": 0.0}
    for it in range(1, int(max_rounds) + 1):
        pr = _sigmoid(F_tr)
        g, h = w * (pr - ytr), w * pr * (1 - pr)
        if p["subsample"] < 1.0:
            keep = rng.random(len(tr)) < p["subsample"]
            g, h = np.where(keep, g, 0.0), np.where(keep, h, 0.0)
        sel = np.sort(rng.choice(n_feat, size=n_sel, replace=False)) if n_sel < n_feat else np.arange(n_feat)
        tree, leaf = _grow_tree(bins_T, g, h, cid, len(ids), sel, n_bins, p, explicit_client_histograms, model.gain_by_feature, comm)
        model.trees.append(tree)
        F_tr += tree.value[leaf]
        F_val += model._tree_margin(tree, vb)
        a = _auroc(yv, F_val)
        if it % 10 == 0 or it == 1:
            curve.append((it, a))
        if a > best + 1e-6:
            best, best_it, since = a, it, 0
        else:
            since += 1
        if log_every and it % log_every == 0:
            _LOG.info("[FED_GBDT] round %d val AUROC=%.4f (best %.4f @ %d) %.0fs", it, a, best, best_it, time.time() - t0)
        if early_stopping_rounds and since >= early_stopping_rounds:
            break
    n_run = len(model.trees)
    model.trees = model.trees[:best_it]
    hist = {"best_iteration": int(best_it), "n_rounds_run": int(n_run), "best_val_auroc": float(best), "n_clients": len(ids),
            "n_train": int(len(tr)), "n_bins": int(n_bins), "params": dict(p), "scale_pos_weight": float(spw),
            "comm_mb_per_client_per_round_dense": float(comm["dense"] / max(n_run, 1) / len(ids) / 1e6),
            "comm_mb_per_client_per_round_sparse": float(comm["sparse"] / max(n_run, 1) / len(ids) / 1e6),
            "comm_gb_total_sparse_all_clients": float(comm["sparse"] / 1e9),
            "val_curve": curve, "seconds": float(time.time() - t0), "explicit_client_histograms": bool(explicit_client_histograms)}
    return model, hist


def search_and_train(X, y, client_rows, val_rows, cfg: dict, seed: int, fixed_params: "dict | None" = None):
    """Validation-only selection, mirroring the LightGBM arm: trial 0 is the configured `fixed` params, the rest are random draws
    from the `search` grid; the best by validation AUROC is returned. Pass `fixed_params` (the winner from the first seed) to skip the search."""
    kw = dict(n_bins=int(cfg.get("n_bins", 64)), grid_size=int(cfg.get("quantile_grid", 128)), max_rounds=int(cfg.get("max_rounds", 600)),
              early_stopping_rounds=cfg.get("early_stopping_rounds", 40), explicit_client_histograms=bool(cfg.get("explicit_client_histograms", False)))
    ids = [h for h, r in client_rows.items() if len(r) > 0]
    edges = federated_bin_edges(X, {h: client_rows[h] for h in ids}, kw["n_bins"], kw["grid_size"])      # one exchange, reused by every trial
    if fixed_params is not None:
        m, hist = train_fed_gbdt(X, y, client_rows, val_rows, fixed_params, seed, edges=edges, **kw)
        hist["trials"] = [{"params": fixed_params, "val_auroc": hist["best_val_auroc"]}]
        return m, hist
    rng = np.random.default_rng(seed)
    trials = [dict(FIXED_DEFAULTS, **cfg.get("fixed", {}))]
    grid = cfg.get("search", {})
    for _ in range(max(int(cfg.get("n_trials", 6)) - 1, 0)):
        trials.append(dict(trials[0], **{k: v[int(rng.integers(len(v)))] for k, v in grid.items()}))
    best = None; log = []
    for i, tp in enumerate(trials):
        m, hist = train_fed_gbdt(X, y, client_rows, val_rows, tp, seed, edges=edges, **kw)
        log.append({"params": {k: tp[k] for k in sorted(tp)}, "val_auroc": hist["best_val_auroc"], "best_iteration": hist["best_iteration"]})
        _LOG.info("[FED_GBDT] trial %d/%d val AUROC=%.4f params=%s", i + 1, len(trials), hist["best_val_auroc"], log[-1]["params"])
        if best is None or hist["best_val_auroc"] > best[1]["best_val_auroc"]:
            best = (m, hist)
    best[1]["trials"] = log
    return best


def reuse_params_from_run(run_dir, exp_key: str) -> dict:
    """The federated GBDT's hyperparameters from a finished run: the first-seed `fed_gbdt` record of `run_dir/results.json` for experiment `exp_key`
    (for example "mortality/no_apache_scores"). The pooled arm (`gbdt_central`) reuses them so the two models differ only in not being federated.
    Raises if the run, the experiment, the arm or any parameter is missing; it never falls back to a search on its own."""
    p = Path(run_dir) / "results.json"
    if not p.exists():
        raise FileNotFoundError(f"gbdt_central: no results.json in {run_dir} (paper.gbdt_central.reuse_params_from must point at a finished federated-GBDT run)")
    blk = json.load(open(p, encoding="utf-8"))["experiments"].get(exp_key)
    if blk is None:
        raise KeyError(f"gbdt_central: experiment {exp_key!r} is not in {p}")
    recs = sorted((r for r in blk["records"] if r["arm"] == "fed_gbdt"), key=lambda r: r["seed"])
    if not recs:
        raise ValueError(f"gbdt_central: {p} has no fed_gbdt record for {exp_key!r}")
    prm = recs[0]["extra"].get("params", {})
    missing = [k for k in FIXED_DEFAULTS if k not in prm]
    if missing:
        raise ValueError(f"gbdt_central: the fed_gbdt record in {p} lacks parameters {missing}")
    return {k: prm[k] for k in FIXED_DEFAULTS}
