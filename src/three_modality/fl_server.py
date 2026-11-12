"""Federated round loop (full participation) for FedAvg / FedProx / FedAdam / SCAFFOLD and the personalised variants FedPer / FedRep, plus post-hoc local fine-tuning.

Federated validation. After every round each client scores ITS OWN validation patients with the current model and sends only (a) a per-class histogram of the validation LOGITS and (b) the
sum of its validation focal loss. The server adds the histograms and computes a global validation AUROC. The histogram AUROC treats every positive/negative pair that falls in the same bin
as a tie (0.5), so its absolute error is at most 0.5 * (tied-pair fraction) -- that bound is recorded each round (`val_auroc_tie_bound`) and the exact pooled AUROC at the selected round is
recorded as a diagnostic (`val_auroc_exact_diag`, never used for selection). Round selection, early stopping and the global threshold use VALIDATION data only.

Checkpointing: the complete round state (global parameters, server-optimiser moments, SCAFFOLD control variates, personalised heads, best snapshot, history) is written atomically every
`checkpoint.every_rounds` rounds; local randomness is seeded per (round, client), so a resumed run reproduces an uninterrupted one exactly.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.three_modality.fl_algorithms import (ClientProblem, apply_server_update, body_indices, build_model, client_weights, get_params, head_indices, local_update, n_params,
                                              round_seed, scaffold_client_control, scaffold_server_control, set_params, weighted_delta)
from src.three_modality.hw_classifiers import focal_loss_torch

logger = logging.getLogger("eicu_fl")


# ---------------------------------------------------------------------------------------------------------------------------------
# histogram-based federated validation
# ---------------------------------------------------------------------------------------------------------------------------------
class HistAccumulator:
    """Per-class histogram of logits over [-R, R] (values outside are clipped into the end bins). Additive across clients."""

    def __init__(self, bins: int, logit_range: float):
        self.bins, self.R = int(bins), float(logit_range)
        self.pos = np.zeros(self.bins, dtype=np.int64)
        self.neg = np.zeros(self.bins, dtype=np.int64)

    def _idx(self, logits) -> np.ndarray:
        z = np.clip(np.asarray(logits, dtype=np.float64), -self.R, self.R)
        return np.minimum(((z + self.R) / (2.0 * self.R) * self.bins).astype(np.int64), self.bins - 1)

    def add(self, logits, y) -> None:
        y = np.asarray(y).astype(int)
        i = self._idx(logits)
        self.pos += np.bincount(i[y == 1], minlength=self.bins)
        self.neg += np.bincount(i[y == 0], minlength=self.bins)

    def merge(self, other: "HistAccumulator") -> None:
        self.pos += other.pos
        self.neg += other.neg

    def auroc(self) -> float:
        P, N = int(self.pos.sum()), int(self.neg.sum())
        if P == 0 or N == 0:
            return float("nan")
        below = np.cumsum(self.neg) - self.neg
        return float((self.pos * (below + 0.5 * self.neg)).sum() / (P * N))

    def tie_bound(self) -> float:
        """Upper bound on |AUROC_hist - AUROC_exact|: 0.5 x the fraction of positive/negative pairs that share a bin."""
        P, N = int(self.pos.sum()), int(self.neg.sum())
        return float(0.5 * (self.pos * self.neg).sum() / (P * N)) if P and N else float("nan")

    def best_f1_probability_threshold(self) -> float:
        """Probability threshold maximising F1 on the pooled validation histogram (predict positive when logit >= bin lower edge). NaN when a class is absent."""
        P, N = int(self.pos.sum()), int(self.neg.sum())
        if P == 0 or N == 0:
            return float("nan")
        tp = np.cumsum(self.pos[::-1])[::-1].astype(np.float64)
        fp = np.cumsum(self.neg[::-1])[::-1].astype(np.float64)
        f1 = 2 * tp / (2 * tp + fp + (P - tp))
        b = int(np.argmax(f1))
        edge = -self.R + b * (2.0 * self.R / self.bins)
        return float(1.0 / (1.0 + np.exp(-edge)))


def compose_params(w: list, hidx: list, head: list) -> list:
    out = list(w)
    for j, h in zip(hidx, head):
        out[j] = h
    return out


def client_params(res: dict, hid: int) -> list:
    """The exact parameters used to score client `hid` with a trained result: its fine-tuned parameters, else the global body + its own head (or the mean-head fallback)."""
    if hid in res.get("client_full", {}):
        return res["client_full"][hid]
    if res.get("heads") is not None:
        return compose_params(res["w"], res["hidx"], res["heads"].get(hid, res["mean_head"]))
    return res["w"]


def federated_validation(problems: dict, model, params_of, alphas: dict, gamma: float, bins: int, logit_range: float) -> tuple:
    """(HistAccumulator, global validation focal loss). `params_of(hid)` gives the parameters a client scores with."""
    acc = HistAccumulator(bins, logit_range)
    loss_sum, n_sum = 0.0, 0
    model.eval()
    with torch.no_grad():
        for hid in sorted(problems):
            pr = problems[hid]
            if pr.client.n_val == 0:
                continue
            set_params(model, params_of(hid))
            lg = pr.logits(model, "val")
            yv = pr.labels("val")
            loss_sum += float(focal_loss_torch(lg, yv, alphas[hid], gamma)) * pr.client.n_val
            n_sum += pr.client.n_val
            acc.add(lg.numpy(), yv.numpy())
    return acc, (loss_sum / n_sum if n_sum else float("nan"))


def _atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _clone(lst):
    return [t.clone() for t in lst]


# ---------------------------------------------------------------------------------------------------------------------------------
# the round loop
# ---------------------------------------------------------------------------------------------------------------------------------
def train_federated(problems: dict, family: str, ds_cfg: dict, fl_cfg: dict, *, algo: str = "fedavg", method: str = "global", alphas: dict, seed: int, ckpt_dir=None, resume: bool = False,
                    ident: str = "", round_hook=None, tag: str = "") -> dict:
    """Train ONE federated model (one task, one family) and return the best-validation-round state.

    method "global"  : one model for all clients (algo in fedavg | fedprox | fedadam | scaffold);
    method "fedper"  : shared body (aggregated, FedAvg) + hospital-specific heads that never leave the client;
    method "fedrep"  : as FedPer, with per-round head-only steps followed by body-only steps."""
    if method not in ("global", "fedper", "fedrep"):
        raise ValueError(f"unknown method {method!r}")
    if method != "global" and algo != "fedavg":
        raise ValueError("personalised methods aggregate the shared body with FedAvg only")
    if not problems:
        raise ValueError("no clients")
    t0 = time.time()
    ids = sorted(problems)
    r_cfg, l_cfg, v_cfg = fl_cfg["rounds"], fl_cfg["local"], fl_cfg["validation"]
    gamma = float(ds_cfg["focal"]["gamma"])
    a_cfg = fl_cfg["algorithms"]
    dim = problems[ids[0]].client.dim
    model = build_model(family, dim, ds_cfg, seed)
    w = get_params(model)
    hidx, bidx = head_indices(model), body_indices(model)
    personalised = method != "global"
    eligible = {h for h in ids if problems[h].client.nonfl_eligible} if personalised else set()
    heads = {h: _clone([w[j] for j in hidx]) for h in eligible} if personalised else None
    mean_head = _clone([w[j] for j in hidx]) if personalised else None
    lr, wd = float(l_cfg["lr"][family]), float(l_cfg["weight_decay"][family])
    K = int(r_cfg["local_steps"])
    if method == "fedrep":
        fr = fl_cfg["personalization"]["fedrep"]
        phases = [(int(fr["head_steps"]), hidx), (int(fr["body_steps"]), bidx)]
    else:
        phases = [(K, list(range(len(w))))]
    K_eff = int(sum(n for n, _ in phases))
    agg_idx = bidx if personalised else list(range(len(w)))
    mu = float(a_cfg["fedprox"]["mu"]) if algo == "fedprox" else 0.0
    if algo == "fedprox" and l_cfg["optimizer"] == "sgd" and float(l_cfg["lr"][family]) * mu >= 2.0:
        raise ValueError(f"FedProx with plain SGD is unstable when lr * mu >= 2 (lr={l_cfg['lr'][family]}, mu={mu}): the proximal quadratic's step would overshoot; lower mu or the local lr")
    server_state = {"m": None, "v": None}
    c = [torch.zeros_like(t) for t in w] if algo == "scaffold" else None
    c_i = {h: [torch.zeros_like(t) for t in w] for h in ids} if algo == "scaffold" else None
    state = {"rnd": 0, "best": None, "bad": 0, "history": [], "nonfinite_updates": 0}
    state_path = Path(ckpt_dir) / "state.pt" if ckpt_dir else None
    final_path = Path(ckpt_dir) / "final.pt" if ckpt_dir else None
    bytes_per_round = 2 * n_params(model) * 4 * len(ids) * (2 if algo == "scaffold" else 1)    # download + upload of float32 parameters (x2 for SCAFFOLD's control variates)

    def _load(path):
        ck = torch.load(path, weights_only=False)
        if ck["ident"] != ident:
            raise RuntimeError(f"checkpoint {path} belongs to a different run identity; refusing to resume (it would silently mix two experiments)")
        return ck

    stop_reason = None
    if resume and final_path is not None and final_path.is_file():
        ck = _load(final_path)
        return {**ck["result"], "resumed_from": "final", "seconds": 0.0}
    if resume and state_path is not None and state_path.is_file():
        ck = _load(state_path)
        state, w = ck["state"], ck["w"]
        server_state, c, c_i, heads, mean_head = ck["server_state"], ck["c"], ck["c_i"], ck["heads"], ck["mean_head"]
        logger.info("[FL] %s resumed from round %d", tag, state["rnd"])

    def params_of(hid):
        return client_params({"w": w, "heads": heads, "mean_head": mean_head, "hidx": hidx}, hid)

    def _save_state():
        if state_path is not None:
            _atomic_torch_save({"ident": ident, "state": state, "w": w, "server_state": server_state, "c": c, "c_i": c_i, "heads": heads, "mean_head": mean_head}, state_path)

    for rnd in range(state["rnd"] + 1, int(r_cfg["max_rounds"]) + 1):
        if round_hook is not None:
            round_hook(rnd)
        tr = time.time()
        locs, used, new_ci = [], [], {}
        for hid in ids:
            pr = problems[hid]
            if personalised:
                w_start = compose_params(w, hidx, heads[hid] if hid in heads else mean_head)
            else:
                w_start = w
            wl, st = local_update(model, pr, w_start, alphas[hid], gamma, algo=("fedavg" if personalised else algo), optimizer=l_cfg["optimizer"], lr=lr, wd=wd, phases=phases, mu=mu,
                                  c_global=c, c_local=(c_i[hid] if c_i else None), w_anchor=w, seed=round_seed(seed, rnd, hid), grad_clip=l_cfg.get("grad_clip"))
            if wl is None:
                state["nonfinite_updates"] += 1
                continue
            locs.append(wl)
            used.append(hid)
            if algo == "scaffold":
                new_ci[hid] = scaffold_client_control(c_i[hid], c, w, wl, K_eff, lr)
        if not locs:
            stop_reason = "every client update was non-finite"
            break
        p = client_weights([problems[h].client.n_train for h in used])
        new_w = list(w)
        for j, val in zip(agg_idx, apply_server_update(algo, w, weighted_delta(w, locs, p, agg_idx), server_state, a_cfg["fedadam"], agg_idx)):
            new_w[j] = val
        if algo == "scaffold":
            dcs = [[n - o for n, o in zip(new_ci[h], c_i[h])] for h in used]
            c = scaffold_server_control(c, dcs, p)
            for h in used:
                c_i[h] = new_ci[h]
        if personalised:
            el = [(h, locs[i]) for i, h in enumerate(used) if h in eligible]
            for h, wl in el:
                heads[h] = _clone([wl[j] for j in hidx])
            if el:
                pe = client_weights([problems[h].client.n_train for h, _ in el])
                mean_head = [sum(float(pe[i]) * el[i][1][j] for i in range(len(el))) for j in hidx]
        w = new_w
        acc, vloss = federated_validation(problems, model, params_of, alphas, gamma, v_cfg["bins"], v_cfg["logit_range"])
        auc = acc.auroc()
        score = (auc if np.isfinite(auc) else -vloss) if v_cfg["selection_metric"] == "auroc" else -vloss
        rec = {"round": rnd, "val_auroc_hist": None if not np.isfinite(auc) else float(auc), "val_auroc_tie_bound": float(acc.tie_bound()) if np.isfinite(auc) else None,
               "val_focal_loss": float(vloss), "clients_updated": len(used), "seconds": round(time.time() - tr, 3)}
        state["history"].append(rec)
        best = state["best"]
        if best is None or score > best["score"] + 1e-6:
            state["best"] = {"score": float(score), "round": rnd, "w": _clone(w), "heads": ({h: _clone(v) for h, v in heads.items()} if personalised else None),
                             "mean_head": (_clone(mean_head) if personalised else None), "val_pos": acc.pos.copy(), "val_neg": acc.neg.copy(), "val_loss": float(vloss)}
            state["bad"] = 0
        else:
            state["bad"] += 1
        state["rnd"] = rnd
        if rnd % max(1, int(r_cfg["max_rounds"]) // 10) == 0 or rnd == 1:
            logger.info("[FL] %s round %d val AUROC(hist)=%s loss=%.4f best=%.4f@%d", tag, rnd, "nan" if not np.isfinite(auc) else f"{auc:.4f}", vloss, state["best"]["score"], state["best"]["round"])
        if rnd % int(fl_cfg["checkpoint"]["every_rounds"]) == 0:
            _save_state()
        if rnd >= int(r_cfg["min_rounds"]) and state["bad"] >= int(r_cfg["patience"]):
            stop_reason = f"early stopping (no validation improvement for {r_cfg['patience']} rounds)"
            break
    if stop_reason is None:
        stop_reason = "max_rounds reached"
    best = state["best"]
    if best is None:
        raise RuntimeError("federated training produced no usable round")
    hist = state["history"]
    result = {"w": best["w"], "heads": best["heads"], "mean_head": best["mean_head"], "hidx": hidx, "bidx": bidx, "best_round": best["round"], "best_val_score": best["score"],
              "val_pos": best["val_pos"], "val_neg": best["val_neg"], "val_focal_loss": best["val_loss"], "rounds_run": state["rnd"], "stop_reason": stop_reason,
              "history": hist, "nonfinite_updates": state["nonfinite_updates"], "algo": algo, "method": method, "family": family, "n_clients": len(ids),
              "n_params": n_params(model), "comm_mb_total": float(bytes_per_round * state["rnd"] / 1e6), "comm_mb_per_round": float(bytes_per_round / 1e6),
              "local_steps_effective": K_eff, "seconds": round(time.time() - t0, 2), "resumed_from": "state" if (resume and state_path is not None and state_path.is_file()) else None,
              "head_clients": sorted(eligible) if personalised else None}
    # diagnostic only: exact pooled validation AUROC at the selected round (validation data, never used for selection)
    hb = HistAccumulator(v_cfg["bins"], v_cfg["logit_range"])
    hb.pos, hb.neg = best["val_pos"], best["val_neg"]
    result["val_auroc_hist_at_best"] = float(hb.auroc())
    result["val_auroc_tie_bound_at_best"] = float(hb.tie_bound())
    if ckpt_dir is not None:
        _atomic_torch_save({"ident": ident, "result": result}, final_path)
        if state_path is not None and state_path.is_file():
            state_path.unlink()
    return result


def exact_val_auroc_diag(problems: dict, model, res: dict) -> float:
    """Exact pooled validation AUROC of the trained result (diagnostic for the histogram approximation error; uses validation data only)."""
    from sklearn.metrics import roc_auc_score
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for hid in sorted(problems):
            pr = problems[hid]
            if pr.client.n_val == 0:
                continue
            set_params(model, client_params(res, hid))
            ps.append(pr.logits(model, "val").numpy())
            ys.append(pr.labels("val").numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def predict_logits(res: dict, model, prob: ClientProblem, split: str) -> np.ndarray:
    set_params(model, client_params(res, prob.client.hospitalid))
    model.eval()
    with torch.no_grad():
        return prob.logits(model, split).numpy().astype(np.float64)


# ---------------------------------------------------------------------------------------------------------------------------------
# post-hoc local fine-tuning (FedAvg -> local)
# ---------------------------------------------------------------------------------------------------------------------------------
def finetune_clients(res: dict, problems: dict, family: str, ds_cfg: dict, fl_cfg: dict, alphas: dict, seed: int) -> tuple:
    """Fine-tune the best global model locally for every client that satisfies the Non-FL minimum-data rule. Epoch 0 (the global model itself) is a candidate, so fine-tuning can never
    end up worse than the global model on the client's VALIDATION loss. Selection uses the client's own validation loss only. Returns (client_full, info)."""
    ft = fl_cfg["personalization"]["finetune"]
    gamma = float(ds_cfg["focal"]["gamma"])
    model = build_model(family, problems[sorted(problems)[0]].client.dim, ds_cfg, seed)
    all_idx = list(range(len(res["w"])))
    active = res["hidx"] if ft["head_only"] else all_idx
    wd = float(fl_cfg["local"]["weight_decay"][family])
    out, info = {}, {}
    for hid in sorted(problems):
        pr = problems[hid]
        if not pr.client.nonfl_eligible:
            info[hid] = {"source": "global_fallback", "reason": "client does not satisfy the minimum-data rule for local fine-tuning"}
            continue

        def val_loss(params):
            set_params(model, params)
            model.eval()
            with torch.no_grad():
                return float(focal_loss_torch(pr.logits(model, "val"), pr.labels("val"), alphas[hid], gamma))

        cur = _clone(res["w"])
        best_l, best_p, best_e, bad = val_loss(cur), _clone(cur), 0, 0
        e = 0
        for e in range(1, int(ft["max_epochs"]) + 1):
            nxt, st = local_update(model, pr, cur, alphas[hid], gamma, algo="fedavg", optimizer="sgd", lr=float(ft["lr"]), wd=wd, phases=[(1, active)], seed=round_seed(seed, e, hid, "ft"))
            if nxt is None:
                break
            cur = nxt
            vl = val_loss(cur)
            if vl < best_l - 1e-7:
                best_l, best_p, best_e, bad = vl, _clone(cur), e, 0
            else:
                bad += 1
                if bad >= int(ft["patience"]):
                    break
        out[hid] = best_p
        info[hid] = {"source": "local_finetune", "best_epoch": best_e, "epochs_run": e, "best_val_focal_loss": best_l}
    return out, info
