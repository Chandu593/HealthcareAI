"""The four downstream classifiers, each fitted on ONE hospital's own training rows (never pooled):

  lr        : logistic regression (a single linear layer, torch) trained with focal loss;
  mlp       : small MLP (torch) trained with focal loss;
  lightgbm  : LightGBM with a CUSTOM FOCAL OBJECTIVE (analytic gradient and Hessian, validated against finite differences in the tests);
  gbdt      : sklearn GradientBoostingClassifier with its STANDARD deviance (log) loss -- sklearn exposes no loss hook, so no focal weighting and no class weighting are applied.

Feature scaling is fitted on the hospital's TRAINING rows only. Early stopping / the number of boosting stages is chosen on the hospital's VALIDATION rows only. Test rows are only ever
passed through `predict`. Every call is self-contained and seeded, so one hospital's result never depends on another hospital's data or on processing order.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

CLASSIFIER_LOSSES = {
    "lr": "focal loss (torch, linear)", "mlp": "focal loss (torch)",
    "lightgbm": "focal loss via custom objective (gradient/Hessian analytic, Hessian clipped to >= 1e-6)",
    "gbdt": "standard deviance (log) loss; NO focal weighting and NO class weighting (sklearn GradientBoostingClassifier has no loss hook)",
}


def resolve_alpha(focal_cfg: dict, y_train, smote: bool) -> float:
    """Focal alpha (weight of the POSITIVE class). 'balanced' = fraction of negatives in this hospital's training data. With SMOTE the classes are already balanced, so a neutral alpha is used
    instead of re-weighting the classes a second time."""
    if smote:
        return float(focal_cfg["alpha_with_smote"])
    a = focal_cfg["alpha"]
    if a == "balanced":
        y = np.asarray(y_train)
        return float(np.clip((y == 0).mean(), 0.05, 0.95))
    return float(a)


# ---------------------------------------------------------------------------------------------------------------------------------
# focal loss, numpy side (LightGBM) and torch side (LR / MLP / GCN)
# ---------------------------------------------------------------------------------------------------------------------------------
def focal_loss_np(z, y, alpha: float, gamma: float) -> float:
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64)
    p = expit(z)                                                          # overflow-free sigmoid
    pt = np.clip(np.where(y == 1, p, 1.0 - p), 1e-12, 1.0)
    at = np.where(y == 1, alpha, 1.0 - alpha)
    return float(np.mean(-at * (1.0 - pt) ** gamma * np.log(pt)))


def focal_grad_hess(z, y, alpha: float, gamma: float):
    """d L / d z and d^2 L / d z^2 of the per-sample focal loss  L = -a_t (1-p_t)^gamma log p_t  w.r.t. the raw score z (p = sigmoid(z), s = 2y-1, dp_t/dz = s p_t (1-p_t)):
        dL/dz   = a_t s (1-p_t)^g [ g p_t log p_t - (1-p_t) ]
        d2L/dz2 = a_t p_t (1-p_t) { -g (1-p_t)^(g-1) [ g p_t log p_t - (1-p_t) ] + (1-p_t)^g [ g log p_t + g + 1 ] }
    Returned per sample (no averaging, the booster sums them). The Hessian is the TRUE second derivative and may be slightly negative for some (p, gamma); the LightGBM wrapper clips it."""
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64)
    p = expit(z)
    s = 2.0 * y - 1.0
    pt = np.clip(np.where(y == 1, p, 1.0 - p), 1e-12, 1.0 - 1e-12)
    at = np.where(y == 1, alpha, 1.0 - alpha)
    omp, lg = 1.0 - pt, np.log(pt)
    inner = gamma * pt * lg - omp
    g = at * s * omp ** gamma * inner
    h = at * pt * omp * (-gamma * omp ** (gamma - 1.0) * inner + omp ** gamma * (gamma * lg + gamma + 1.0))
    return g, h


def focal_loss_torch(logits, y, alpha: float, gamma: float):
    import torch
    import torch.nn.functional as F
    y = y.to(logits.dtype)
    logp, log1mp = F.logsigmoid(logits), F.logsigmoid(-logits)
    p = torch.exp(logp)
    pos = -alpha * (1.0 - p).pow(gamma) * logp
    neg = -(1.0 - alpha) * p.pow(gamma) * log1mp
    return torch.where(y == 1, pos, neg).mean()


def selection_score(metric: str, val_loss: float, y_val, p_val) -> float:
    """LOWER is better. 'loss' = validation focal loss; 'auroc' / 'auprc' = negative validation ranking score. A validation set with one class cannot be ranked, so it falls back to the loss."""
    y = np.asarray(y_val)
    if metric == "loss" or len(np.unique(y)) < 2:
        return float(val_loss)
    return -float(roc_auc_score(y, p_val) if metric == "auroc" else average_precision_score(y, p_val))


# ---------------------------------------------------------------------------------------------------------------------------------
# torch classifiers
# ---------------------------------------------------------------------------------------------------------------------------------
def _fit_torch(kind, Xtr, ytr, Xva, yva, Xte, cfg, alpha, gamma, seed):
    import copy

    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    d = Xtr.shape[1]
    if kind == "lr":
        net = nn.Linear(d, 1)
    else:
        net = nn.Sequential(nn.Linear(d, int(cfg["hidden"])), nn.ReLU(), nn.Dropout(float(cfg["dropout"])), nn.Linear(int(cfg["hidden"]), 1))
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    xtr, xva, xte = (torch.from_numpy(a.astype(np.float32)) for a in (Xtr, Xva, Xte))
    ytr_t, yva_t = torch.from_numpy(np.asarray(ytr)), torch.from_numpy(np.asarray(yva))
    metric, min_ep = cfg.get("selection_metric", "loss"), int(cfg.get("min_epochs", 1))
    best, best_state, best_ep, bad, best_loss = float("inf"), None, 0, 0, float("nan")
    ep = 0
    for ep in range(1, int(cfg["max_epochs"]) + 1):
        net.train()
        opt.zero_grad()
        loss = focal_loss_torch(net(xtr).squeeze(-1), ytr_t, alpha, gamma)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            lv = net(xva).squeeze(-1)
            vl = focal_loss_torch(lv, yva_t, alpha, gamma).item()
            pvp = torch.sigmoid(lv).numpy()
        if ep < min_ep:
            continue                                                        # a checkpoint before `min_epochs` can never be selected
        sc = selection_score(metric, vl, yva, pvp)
        if sc < best - 1e-7:
            best, best_state, best_ep, bad, best_loss = sc, copy.deepcopy(net.state_dict()), ep, 0, vl
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    if best_state is None:                                                  # unreachable with min_epochs <= max_epochs, kept as a guard
        best_state, best_ep, best_loss = copy.deepcopy(net.state_dict()), ep, float("nan")
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pv, pt = (torch.sigmoid(net(x).squeeze(-1)).numpy().astype(np.float64) for x in (xva, xte))
    return pv, pt, {"best_epoch": best_ep, "epochs_run": ep, "best_val_focal_loss": best_loss, "selection_metric": metric, "min_epochs": min_ep}


# ---------------------------------------------------------------------------------------------------------------------------------
# LightGBM with the custom focal objective
# ---------------------------------------------------------------------------------------------------------------------------------
def _fit_lightgbm(Xtr, ytr, Xva, yva, Xte, cfg, alpha, gamma, seed):
    import lightgbm as lgb

    prior = float(np.clip(np.mean(ytr), 1e-3, 1 - 1e-3))
    z0 = float(np.log(prior / (1 - prior)))                              # custom objectives do not boost from the average: start from the training log-odds
    # LightGBM pre-filters every feature that cannot satisfy min_data_in_leaf and aborts (fatal) when none is left -- which happens for tiny hospitals / tuning slices. The leaf minimum is
    # therefore capped at a quarter of the training rows and the pre-filter is switched off (an unsplittable tree then just returns the prior; it can no longer crash the run).
    min_leaf = int(max(1, min(int(cfg["min_child_samples"]), len(ytr) // 4)))
    dparams = {"feature_pre_filter": False, "min_data_in_leaf": min_leaf}
    dtr = lgb.Dataset(Xtr, label=np.asarray(ytr, dtype=float), init_score=np.full(len(ytr), z0), params=dparams, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=np.asarray(yva, dtype=float), init_score=np.full(len(yva), z0), reference=dtr, free_raw_data=False)

    floor = float(cfg.get("hessian_floor", 0.05))
    metric = cfg.get("selection_metric", "loss")
    if metric != "loss" and len(np.unique(yva)) < 2:
        metric = "loss"

    def fobj(preds, data):
        y = data.get_label()
        g, h = focal_grad_hess(preds, y, alpha, gamma)
        at = np.where(y == 1, alpha, 1.0 - alpha)
        return g, np.maximum(h, floor * at)                              # the true focal Hessian is ~0/negative for confidently-wrong samples: a tiny floor made the Newton step explode

    def feval(preds, data):
        y = data.get_label()
        if metric == "loss":
            return "focal_val", focal_loss_np(preds, y, alpha, gamma), False
        return f"{metric}_val", float(roc_auc_score(y, preds) if metric == "auroc" else average_precision_score(y, preds)), True

    params = {"objective": fobj, "metric": "None", "learning_rate": float(cfg["learning_rate"]), "num_leaves": int(cfg["num_leaves"]), "min_data_in_leaf": min_leaf, "feature_pre_filter": False,
              "bagging_fraction": float(cfg["subsample"]), "bagging_freq": 1 if float(cfg["subsample"]) < 1.0 else 0, "seed": int(seed), "deterministic": True, "force_col_wise": True,
              "num_threads": 1, "verbose": -1, "lambda_l2": float(cfg.get("lambda_l2", 1.0)), "max_delta_step": float(cfg.get("max_delta_step", 1.0)),
              "min_sum_hessian_in_leaf": float(cfg.get("min_sum_hessian_in_leaf", 1e-3))}
    booster = lgb.train(params, dtr, num_boost_round=int(cfg["n_estimators"]), valid_sets=[dva], feval=feval,
                        callbacks=[lgb.early_stopping(int(cfg["early_stopping_rounds"]), first_metric_only=True, verbose=False)])
    best_it = int(booster.best_iteration or booster.current_iteration())
    raw = lambda X: booster.predict(X, raw_score=True, num_iteration=best_it) + z0     # noqa: E731
    name = "focal_val" if metric == "loss" else f"{metric}_val"
    recorded = ((booster.best_score or {}).get("valid_0") or {}).get(name)
    if recorded is None:                                                  # no tree could be grown, so no evaluation was recorded: score the prior-only model directly
        zv = raw(Xva)
        recorded = focal_loss_np(zv, yva, alpha, gamma) if metric == "loss" else float(roc_auc_score(yva, zv) if metric == "auroc" else average_precision_score(yva, zv))
    best_score = float(recorded)
    info = {"best_iteration": best_it, "n_trees_trained": int(booster.current_iteration()), "init_log_odds": z0, "selection_metric": metric,
            "min_data_in_leaf_effective": min_leaf, "best_val_score": best_score,
            "max_abs_raw_score": float(max(np.abs(raw(Xtr)).max(), np.abs(raw(Xva)).max()))}   # diagnostic from TRAIN/VAL features only
    if metric == "loss":
        info["best_val_focal_loss"] = best_score
    return expit(raw(Xva)), expit(raw(Xte)), info


# ---------------------------------------------------------------------------------------------------------------------------------
# sklearn GradientBoosting (standard loss)
# ---------------------------------------------------------------------------------------------------------------------------------
def _fit_gbdt(Xtr, ytr, Xva, yva, Xte, cfg, seed):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import log_loss

    metric = cfg.get("selection_metric", "loss")
    if metric != "loss" and len(np.unique(yva)) < 2:
        metric = "loss"
    m = GradientBoostingClassifier(n_estimators=int(cfg["n_estimators"]), learning_rate=float(cfg["learning_rate"]), max_depth=int(cfg["max_depth"]),
                                   subsample=float(cfg.get("subsample", 1.0)), random_state=int(seed))
    m.fit(Xtr, ytr)
    scores = []
    for p in m.staged_predict_proba(Xva):
        q = p[:, 1]
        scores.append(log_loss(yva, np.clip(q, 1e-7, 1 - 1e-7), labels=[0, 1]) if metric == "loss" else -(roc_auc_score(yva, q) if metric == "auroc" else average_precision_score(yva, q)))
    best = int(np.argmin(scores)) + 1                                    # number of stages chosen on VALIDATION only (ties -> fewer stages)
    pv = pt = None
    for i, (a, b) in enumerate(zip(m.staged_predict_proba(Xva), m.staged_predict_proba(Xte)), start=1):
        if i == best:
            pv, pt = a[:, 1].astype(np.float64), b[:, 1].astype(np.float64)
            break
    return pv, pt, {"best_n_estimators": best, "n_estimators_trained": len(scores), "best_val_score": float(min(scores)), "selection_metric": metric,
                    **({"best_val_log_loss": float(min(scores))} if metric == "loss" else {})}


def fit_predict(name: str, Xtr, ytr, Xva, yva, Xte, ds_cfg: dict, seed: int, smote: bool = False) -> dict:
    """Fit ONE classifier on ONE hospital's training rows; returns validation and test probabilities (test rows are only predicted). `ytr`/`yva` are 0/1 arrays."""
    if name not in CLASSIFIER_LOSSES:
        raise ValueError(f"unknown classifier {name!r}; choose from {tuple(CLASSIFIER_LOSSES)}")
    ytr, yva = np.asarray(ytr).astype(int), np.asarray(yva).astype(int)
    if len(np.unique(ytr)) < 2:
        raise ValueError("training labels contain a single class; the caller must skip this hospital")
    sc = StandardScaler().fit(Xtr)                                       # scaling statistics from THIS hospital's TRAINING rows only
    A, B, C = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    ccfg = ds_cfg["classifiers"][name]
    focal = {**ds_cfg["focal"], **{k: ccfg[k] for k in ("gamma", "alpha") if ccfg.get(k) is not None}}      # optional per-family override (set by the tuning stage)
    gamma = float(focal["gamma"])
    alpha = resolve_alpha(focal, ytr, smote)
    if name in ("lr", "mlp"):
        pv, pt, info = _fit_torch(name, A, ytr, B, yva, C, ccfg, alpha, gamma, seed)
    elif name == "lightgbm":
        pv, pt, info = _fit_lightgbm(A, ytr, B, yva, C, ccfg, alpha, gamma, seed)
    else:
        pv, pt, info = _fit_gbdt(A, ytr, B, yva, C, ccfg, seed)
    info.update({"loss": CLASSIFIER_LOSSES[name], "alpha": None if name == "gbdt" else alpha, "gamma": None if name == "gbdt" else gamma})
    return {"p_val": pv, "p_test": pt, "info": info}
