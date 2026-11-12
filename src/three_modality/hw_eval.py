"""Per-hospital evaluation: validation-selected threshold, the full metric set, and macro / sample-weighted hospital averages with explicit skip accounting.

Metric definitions never fabricate a value: a metric that is mathematically undefined for a hospital's test set (AUROC with one class, recall with no positives, ...) is None and is
excluded -- with a count -- from the averages, never silently set to 0.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

METRICS = ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "specificity")


def select_threshold(y_val, p_val) -> float:
    """Threshold that maximises F1 on the VALIDATION set only (candidates = the distinct validation probabilities; ties -> the lowest threshold). With a single-class validation set no
    threshold can be learned and 0.5 is returned (callers skip such hospitals before this point)."""
    y, p = np.asarray(y_val).astype(int), np.asarray(p_val, dtype=np.float64)
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5
    best_t, best_f = 0.5, -1.0
    for t in np.unique(p):
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f > best_f + 1e-12:
            best_f, best_t = f, float(t)
    return best_t


def select_threshold_recall(y_val, p_val, target: float) -> float:
    """Secondary operating point, chosen on the VALIDATION set only: the HIGHEST threshold whose validation recall reaches `target` (highest = best precision among those that qualify)."""
    y, p = np.asarray(y_val).astype(int), np.asarray(p_val, dtype=np.float64)
    if y.sum() == 0:
        return 0.5
    for t in np.unique(p)[::-1]:
        if ((p >= t) & (y == 1)).sum() / y.sum() >= target - 1e-12:
            return float(t)
    return float(p.min())


def hospital_metrics(y_test, p_test, threshold: float) -> dict:
    y, p = np.asarray(y_test).astype(int), np.asarray(p_test, dtype=np.float64)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n, pos, neg = len(y), int(y.sum()), int(len(y) - y.sum())
    div = lambda a, b: (a / b) if b else None    # noqa: E731
    two_class = pos > 0 and neg > 0
    rep = classification_report(y, pred, labels=[0, 1], target_names=["negative", "positive"], output_dict=True, zero_division=0)
    return {"n": n, "n_pos": pos, "n_neg": neg, "threshold": float(threshold),
            "auroc": float(roc_auc_score(y, p)) if two_class else None,
            "auprc": float(average_precision_score(y, p)) if pos > 0 else None,
            "accuracy": div(tp + tn, n), "precision": div(tp, tp + fp), "recall": div(tp, tp + fn), "f1": div(2 * tp, 2 * tp + fp + fn), "specificity": div(tn, tn + fp),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}, "classification_report": rep}


_CATEGORIES = (("train patients", "too few training patients"), ("train positives", "too few training positives"), ("train negatives", "too few training negatives"),
               ("validation positives", "no/too few validation positives"), ("validation negatives", "no/too few validation negatives"))


def skip_categories(reason: str) -> list:
    out = []
    for part in (reason or "").split(";"):
        part = part.strip()
        hit = next((c for key, c in _CATEGORIES if part.startswith(key)), None)
        out.append(hit or re.sub(r"\d+", "#", part) or "unspecified")
    return out


def aggregate(rows: list, min_test_per_class_for_power: int = 5) -> dict:
    """`rows` = per-hospital result dicts: {"status": "trained"|"skipped", "skip_reason", "metrics": {...}|None, ...}."""
    trained = [r for r in rows if r.get("status") == "trained" and r.get("metrics")]
    skipped = [r for r in rows if r.get("status") != "trained"]
    cats = Counter(c for r in skipped for c in skip_categories(r.get("skip_reason", "")))
    out = {"n_hospitals": len(rows), "n_trained": len(trained), "n_skipped": len(skipped), "skip_reason_counts": dict(cats),
           "n_low_power_trained": sum(1 for r in trained if min(r["metrics"]["n_pos"], r["metrics"]["n_neg"]) < min_test_per_class_for_power),
           "low_power_definition": f"fewer than {min_test_per_class_for_power} test patients in a class", "metrics": {}}
    for m in METRICS:
        vals = [(r["metrics"][m], r["metrics"]["n"], min(r["metrics"]["n_pos"], r["metrics"]["n_neg"]) >= min_test_per_class_for_power) for r in trained if r["metrics"][m] is not None]
        if not vals:
            out["metrics"][m] = {"macro_mean": None, "sample_weighted_mean": None, "macro_mean_powered_hospitals": None, "n_defined": 0, "n_undefined": len(trained)}
            continue
        v = np.array([x[0] for x in vals])
        w = np.array([x[1] for x in vals], dtype=float)
        pw = np.array([x[2] for x in vals])
        out["metrics"][m] = {"macro_mean": float(v.mean()), "macro_std": float(v.std(ddof=0)), "sample_weighted_mean": float((v * w).sum() / w.sum()),
                             "macro_mean_powered_hospitals": float(v[pw].mean()) if pw.any() else None, "n_powered": int(pw.sum()), "n_defined": len(vals), "n_undefined": len(trained) - len(vals)}
    out["test_patients_in_trained_hospitals"] = int(sum(r["metrics"]["n"] for r in trained))
    return out
