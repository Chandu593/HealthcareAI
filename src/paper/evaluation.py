"""Metrics, calibration, thresholding, bootstrap CIs and paired tests.
One threshold rule is applied identically to train/val/test (fixes the old
train@0.5 vs val-tuned-threshold inconsistency)."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve


def fast_auroc(y, p) -> float:
    y = np.asarray(y); p = np.asarray(p, dtype="float64")
    npos = int((y == 1).sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def auprc(y, p) -> float:
    y = np.asarray(y)
    return float("nan") if (y == 1).sum() == 0 or (y == 0).sum() == 0 else float(average_precision_score(y, p))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype="float64") - np.asarray(y, dtype="float64")) ** 2))


def ece(y, p, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype="float64"); p = np.asarray(p, dtype="float64")
    edges = np.linspace(0, 1, n_bins + 1); out = 0.0
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    for i in range(n_bins):
        m = b == i
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype="float64"), -30, 30)))


def platt_fit(score_val, y_val):
    """Fit p = sigmoid(a*score + b) on VALIDATION only. Returns (a, b); (1, 0) if degenerate."""
    y_val = np.asarray(y_val)
    if len(np.unique(y_val)) < 2:
        return 1.0, 0.0
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(score_val, dtype="float64").reshape(-1, 1), y_val)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def threshold_max_f1(y, p) -> float:
    y = np.asarray(y)
    if (y == 1).sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def binary_report(y, p, thr: float) -> dict:
    y = np.asarray(y); p = np.asarray(p, dtype="float64"); pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"n": int(len(y)), "n_positive": int((y == 1).sum()), "prevalence": float((y == 1).mean()) if len(y) else float("nan"),
            "threshold": float(thr), "accuracy": (tp + tn) / max(len(y), 1), "precision": prec, "recall": rec, "specificity": spec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "auroc": fast_auroc(y, p), "auprc": auprc(y, p),
            "brier": brier(y, p), "ece": ece(y, p), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


_METRICS = {"auroc": fast_auroc, "auprc": auprc}


def bootstrap_ci(y, p, metric: str = "auroc", n: int = 500, alpha: float = 0.05, seed: int = 0) -> dict:
    y = np.asarray(y); p = np.asarray(p); f = _METRICS[metric]; rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); v = f(y[i], p[i])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return {"point": float(f(y, p)), "lo": float(np.quantile(vals, alpha / 2)), "hi": float(np.quantile(vals, 1 - alpha / 2))}


def paired_bootstrap_delta(y, p1, p2, metric: str = "auroc", n: int = 500, alpha: float = 0.05, seed: int = 0) -> dict:
    """delta = metric(p1) - metric(p2) on the SAME resampled patients; two-sided bootstrap p-value."""
    y = np.asarray(y); p1 = np.asarray(p1); p2 = np.asarray(p2); f = _METRICS[metric]
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); a, b = f(y[i], p1[i]), f(y[i], p2[i])
        if np.isfinite(a) and np.isfinite(b):
            d.append(a - b)
    if not d:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_value": float("nan")}
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"delta": float(f(y, p1) - f(y, p2)), "lo": float(np.quantile(d, alpha / 2)), "hi": float(np.quantile(d, 1 - alpha / 2)),
            "p_value": float(min(max(p, 1.0 / (len(d) + 1)), 1.0))}


def per_hospital_summary(y, p, hospital_ids, min_pos: int = 5, min_neg: int = 5) -> dict:
    y = np.asarray(y); p = np.asarray(p); h = np.asarray(hospital_ids); aucs, ws = [], []
    for hid in np.unique(h):
        m = h == hid; yy = y[m]
        if (yy == 1).sum() >= min_pos and (yy == 0).sum() >= min_neg:
            aucs.append(fast_auroc(yy, p[m])); ws.append(m.sum())
    if not aucs:
        return {"n_hospitals_evaluated": 0}
    a = np.array(aucs); w = np.array(ws, dtype="float64")
    return {"n_hospitals_evaluated": int(len(a)), "macro_auroc": float(a.mean()), "median_auroc": float(np.median(a)),
            "q25_auroc": float(np.quantile(a, 0.25)), "q75_auroc": float(np.quantile(a, 0.75)),
            "weighted_auroc": float((a * w).sum() / w.sum()), "min_auroc": float(a.min()), "max_auroc": float(a.max())}


def fit_local_intercepts(score_val, y_val, hosp_val, a: float, b: float, prior_sd: float = 1.0, iters: int = 30) -> dict:
    """Per-hospital intercept shift d_h on top of the global Platt map, p = sigmoid(a*s + b + d_h).
    Each hospital fits ONE scalar on ITS OWN validation rows (penalised Newton, N(0, prior_sd^2) prior, so tiny
    hospitals stay near 0). Federated-compatible: nothing but a scalar is ever fitted, and no rows leave the hospital."""
    s_ = np.asarray(score_val, dtype="float64"); y_ = np.asarray(y_val, dtype="float64"); h_ = np.asarray(hosp_val)
    prec = 1.0 / float(prior_sd) ** 2; out = {}
    for h in np.unique(h_):
        m = h_ == h; z = a * s_[m] + b; yy = y_[m]; d = 0.0
        for _ in range(iters):
            p = sigmoid(z + d); g = (p - yy).sum() + prec * d; hess = (p * (1 - p)).sum() + prec
            step = g / hess; d -= step
            if abs(step) < 1e-8:
                break
        out[int(h)] = float(np.clip(d, -5.0, 5.0))
    return out


def apply_local_intercepts(score, hosp, a: float, b: float, shifts: dict):
    d = np.array([shifts.get(int(h), 0.0) for h in np.asarray(hosp)], dtype="float64")
    return sigmoid(a * np.asarray(score, dtype="float64") + b + d)
