"""Centralized (non-federated) strong tabular references: tuned LightGBM, optional XGBoost, and LR.
Hyper-parameters are chosen on the VALIDATION split only."""

from __future__ import annotations

import inspect
import logging

import numpy as np

from src.paper.evaluation import fast_auroc

logger = logging.getLogger("eicu_fl")


def _sample(search: dict, rng: np.random.Generator) -> dict:
    return {k: v[int(rng.integers(len(v)))] for k, v in search.items()}


def _lgb_params(p: dict, ytr, seed: int, n_jobs: int) -> dict:
    pos, neg = float((ytr == 1).sum()), float((ytr == 0).sum())
    spw = 1.0 if p["scale_pos_weight_mode"] == "none" or pos == 0 else float(np.sqrt(neg / pos))
    return dict(objective="binary", learning_rate=p["learning_rate"], num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
                colsample_bytree=p["feature_fraction"], subsample=p["bagging_fraction"], subsample_freq=1 if p["bagging_fraction"] < 1 else 0,
                reg_lambda=p["lambda_l2"], scale_pos_weight=spw, random_state=seed, n_jobs=n_jobs, verbose=-1)


def tune_and_fit_lightgbm(Xtr, ytr, Xva, yva, lcfg: dict, seed: int, fixed_params: "dict | None" = None):
    """Random search over lcfg['search'] (n_trials), early-stopping each trial on val AUC.
    If `fixed_params` is given (other seeds), no search -- just refit with that config and this seed."""
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    trials = [fixed_params] if fixed_params is not None else [_sample(lcfg["search"], rng) for _ in range(int(lcfg["n_trials"]))]
    best = (-np.inf, None, None, None); log = []
    for p in trials:
        m = lgb.LGBMClassifier(n_estimators=int(lcfg["n_estimators"]), **_lgb_params(p, ytr, seed, int(lcfg["n_jobs"])))
        ev = {"eval_X": Xva, "eval_y": yva} if "eval_X" in inspect.signature(lgb.LGBMClassifier.fit).parameters else {"eval_set": [(Xva, yva)]}
        m.fit(Xtr, ytr, eval_metric="auc", callbacks=[lgb.early_stopping(int(lcfg["early_stopping_rounds"]), verbose=False)], **ev)
        pv = m.predict_proba(Xva, num_iteration=m.best_iteration_)[:, 1]
        a = fast_auroc(yva, pv); log.append({"params": p, "val_auroc": a, "best_iteration": int(m.best_iteration_ or 0)})
        if a > best[0]:
            best = (a, m, p, int(m.best_iteration_ or 0))
    return best[1], {"best_params": best[2], "best_val_auroc": float(best[0]), "best_iteration": best[3], "trials": log}


def predict_lightgbm(model, X) -> np.ndarray:
    return model.predict_proba(X, num_iteration=model.best_iteration_)[:, 1].astype(np.float32)


def fit_xgboost(Xtr, ytr, Xva, yva, seed: int):
    import xgboost as xgb
    pos, neg = float((ytr == 1).sum()), float((ytr == 0).sum())
    m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.6,
                          scale_pos_weight=float(np.sqrt(neg / max(pos, 1))), eval_metric="auc", early_stopping_rounds=100,
                          random_state=seed, n_jobs=-1, tree_method="hist")
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def fit_logreg(Xtr, ytr, cfg: dict, seed: int):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=float(cfg["C"]), max_iter=int(cfg["max_iter"]), random_state=seed).fit(Xtr, ytr)
