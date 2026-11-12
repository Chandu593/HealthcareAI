"""
Ablation framework -- classifier wrapper (AdaBoost, XGBoost, LightGBM).

Confirmed with the user (ablation-framework decision #1): these
classifiers are trained SEPARATELY from the neural encoders, on FROZEN
embeddings -- never inside the encoders' gradient-based joint training
loop, and never federated (see src/training/flat_multimodal_model.py's
docstring and decision #2). This module has zero dependency on torch /
the rest of the modeling stack -- it operates purely on numpy arrays
and labels, matching the "classifier stage fully decoupled from
representation learning" design this project settled on specifically
because AdaBoost/XGBoost/LightGBM have no gradient and cannot sit
inside the same backward pass the MLP heads do.

Masking discipline matches src/training/losses.py's per-task masking:
labels are expected as a 1D float array where NaN means "unknown,
exclude this patient from this task" -- never coerced to 0/1. Each
classifier is fit/evaluated per TASK separately (mortality, sepsis),
using only that task's non-NaN rows, mirroring
align_labels_to_patient_ids()'s own contract (src/heads/multitask_heads.py).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

VALID_CLASSIFIERS = ("adaboost", "xgboost", "lightgbm")   # which of these actually RUN is controlled by config (ablation.classifiers), never by editing this tuple

DEFAULT_CLASSIFIER_PARAMS = {
    "adaboost": {"n_estimators": 100, "random_state": 0},
    "xgboost": {"n_estimators": 200, "max_depth": 4, "eval_metric": "logloss", "random_state": 0},
    "lightgbm": {"n_estimators": 200, "max_depth": -1, "verbosity": -1, "random_state": 0},
}


def build_classifier(name: str, params: "dict | None" = None):
    """Returns a fresh, unfitted sklearn-API-compatible classifier
    instance (.fit(X, y), .predict_proba(X)). `params` overrides
    DEFAULT_CLASSIFIER_PARAMS[name] on top (config-driven per
    ablation.classifier_params, never hardcoded research thresholds)."""
    if name not in VALID_CLASSIFIERS:
        raise ValueError(f"classifier name must be one of {VALID_CLASSIFIERS}, got {name!r}.")
    merged_params = dict(DEFAULT_CLASSIFIER_PARAMS[name])
    merged_params.update(params or {})

    if name == "adaboost":
        return AdaBoostClassifier(**merged_params)
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(**merged_params)
    if name == "lightgbm":
        import lightgbm as lgb   # lazy import: only needed when this classifier is actually configured
        return lgb.LGBMClassifier(**merged_params)
    raise AssertionError("unreachable -- VALID_CLASSIFIERS guard above already excludes anything else")


def _mask_known(X: np.ndarray, y: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    mask = ~np.isnan(y)
    return X[mask], y[mask].astype(int)


def compute_binary_classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Standard metrics at a given decision `threshold` (default 0.5,
    preserving this function's original behavior exactly when not
    specified). AUROC/AUPRC/Brier are threshold-independent and always
    computed; accuracy/precision/recall/F1/sensitivity/specificity
    depend on `threshold` -- see `select_threshold_from_validation()`
    below for how the ablation framework and the split-aware Phase
    11-13 pipeline choose a non-default threshold from VALIDATION data
    only (added 2026-09-11, closing a real gap: a fixed 0.5 cutoff
    under severe class imbalance, e.g. 8% mortality prevalence,
    produced precision=recall=0 for otherwise-good models in the
    2026-09-10 pilot run -- see IMPLEMENTATION_TRACKER.md).

    AUROC/AUPRC are set to None (never crashed on, never silently
    fabricated as 0.0/0.5) when a split has only one class present --
    both are mathematically undefined there, and a fabricated value
    would look like a real (and misleadingly perfect or terrible)
    result in a summary table. `sensitivity` is an explicit alias for
    recall (same quantity, clinical-literature naming, requested
    alongside `specificity` for research-grade reporting).
    """
    metrics: dict = {}
    y_pred_label = (y_pred_proba >= threshold).astype(int)
    metrics["n"] = int(len(y_true))
    metrics["n_positive"] = int(y_true.sum())
    metrics["threshold_used"] = float(threshold)
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred_label))
    metrics["precision"] = float(precision_score(y_true, y_pred_label, zero_division=0))
    recall = float(recall_score(y_true, y_pred_label, zero_division=0))
    metrics["recall"] = recall
    metrics["sensitivity"] = recall
    metrics["f1"] = float(f1_score(y_true, y_pred_label, zero_division=0))
    tn = int(((y_true == 0) & (y_pred_label == 0)).sum())
    fp = int(((y_true == 0) & (y_pred_label == 1)).sum())
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    metrics["brier_score"] = float(np.mean((y_pred_proba - y_true) ** 2))
    if len(np.unique(y_true)) < 2:
        metrics["auroc"] = None
        metrics["auprc"] = None
        metrics["metrics_note"] = "only one class present in this split -- AUROC/AUPRC undefined"
    else:
        metrics["auroc"] = float(roc_auc_score(y_true, y_pred_proba))
        metrics["auprc"] = float(average_precision_score(y_true, y_pred_proba))
    return metrics


def select_threshold_from_validation(y_val: np.ndarray, y_val_proba: np.ndarray, metric: str = "f1") -> float:
    """Scans candidate thresholds (every unique predicted probability on
    VALIDATION data, i.e. every point the decision boundary could
    meaningfully move to) and returns the one maximizing `metric`
    ("f1" or "balanced_accuracy") on validation data ONLY -- the test
    set is never touched by this function, per "use validation data for
    threshold/model selection and reserve the test set for final
    evaluation." Falls back to 0.5 if validation has fewer than 2
    labeled examples or only one class present (nothing meaningful to
    select from), logged rather than silently guessed.
    """
    if len(y_val) < 2 or len(np.unique(y_val)) < 2:
        return 0.5
    candidates = np.unique(y_val_proba)
    if len(candidates) == 0:
        return 0.5
    best_threshold = 0.5
    best_score = -1.0
    for t in candidates:
        pred = (y_val_proba >= t).astype(int)
        if metric == "balanced_accuracy":
            tp = int(((y_val == 1) & (pred == 1)).sum())
            fn = int(((y_val == 1) & (pred == 0)).sum())
            tn = int(((y_val == 0) & (pred == 0)).sum())
            fp = int(((y_val == 0) & (pred == 1)).sum())
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = (sens + spec) / 2.0
        else:
            score = f1_score(y_val, pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(t)
    return best_threshold


@dataclass
class ClassifierAblationResult:
    classifier: str
    task: str
    train_metrics: dict
    val_metrics: dict
    test_metrics: dict
    fit_runtime_seconds: float
    n_train_used: int
    n_val_used: int
    n_test_used: int
    skipped: bool = False
    skip_reason: "str | None" = None

    def to_dict(self) -> dict:
        return {
            "classifier": self.classifier, "task": self.task, "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics, "test_metrics": self.test_metrics,
            "fit_runtime_seconds": self.fit_runtime_seconds, "n_train_used": self.n_train_used,
            "n_val_used": self.n_val_used, "n_test_used": self.n_test_used, "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


def fit_and_evaluate_classifier(
    classifier_name: str, params: "dict | None",
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: "np.ndarray | None", y_val: "np.ndarray | None",
    X_test: "np.ndarray | None", y_test: "np.ndarray | None",
    task: str, logger: "logging.Logger | None" = None,
    resampler: "Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None" = None,
) -> ClassifierAblationResult:
    """Fits `classifier_name` on TRAIN embeddings/labels (masked to
    known labels for `task`), evaluates on train/val/test alike. Never
    crashes the whole ablation run over one bad combo -- returns a
    `skipped=True` result (e.g. too few positive-class training
    examples to fit a binary classifier at all) rather than raising,
    logged clearly with `skip_reason` so it's attributable in the
    results table, not silently absent.

    `resampler`, added for the class-imbalance sampling comparison
    (confirmed with the user 2026-09-20): an optional
    `(X, y) -> (X, y)` callable applied to the TRAIN split ONLY, after
    label-masking and before `.fit()` -- never to validation or test,
    which are evaluated exactly as passed in. Defaults to `None`
    (no-op), which preserves this function's exact original behavior
    for every existing caller/experiment -- see
    `src/classifiers/imbalance_sampling.py` for the hospital-wise
    resampling implementation this parameter is designed to carry.
    """
    logger = logger or logging.getLogger("eicu_fl")
    X_train_m, y_train_m = _mask_known(X_train, y_train)
    if resampler is not None:
        X_train_m, y_train_m = resampler(X_train_m, y_train_m)

    if len(y_train_m) < 2 or len(np.unique(y_train_m)) < 2:
        reason = (
            f"train split has {len(y_train_m)} labeled patient(s) for task={task!r} with "
            f"{len(np.unique(y_train_m)) if len(y_train_m) else 0} distinct class(es) -- cannot fit "
            f"a binary classifier."
        )
        logger.warning(f"[ABLATION_CLASSIFIER] {classifier_name}/{task} skipped: {reason}")
        return ClassifierAblationResult(
            classifier=classifier_name, task=task, train_metrics={}, val_metrics={}, test_metrics={},
            fit_runtime_seconds=0.0, n_train_used=len(y_train_m), n_val_used=0, n_test_used=0,
            skipped=True, skip_reason=reason,
        )

    clf = build_classifier(classifier_name, params)
    start = time.time()
    clf.fit(X_train_m, y_train_m)
    fit_runtime = time.time() - start

    train_proba = clf.predict_proba(X_train_m)[:, 1]
    train_metrics = compute_binary_classification_metrics(y_train_m, train_proba)

    # Threshold/model selection uses VALIDATION data only (never test) --
    # see select_threshold_from_validation()'s docstring. Falls back to
    # the conventional 0.5 if validation is unavailable/degenerate,
    # logged so it's visible in the results rather than silently assumed.
    selected_threshold = 0.5
    val_metrics: dict = {}
    n_val_used = 0
    val_proba = None
    y_val_m = None
    if X_val is not None and y_val is not None:
        X_val_m, y_val_m = _mask_known(X_val, y_val)
        n_val_used = len(y_val_m)
        if n_val_used > 0:
            val_proba = clf.predict_proba(X_val_m)[:, 1]
            selected_threshold = select_threshold_from_validation(y_val_m, val_proba)
            val_metrics = compute_binary_classification_metrics(y_val_m, val_proba, threshold=selected_threshold)
    if n_val_used == 0:
        logger.info(
            f"[ABLATION_CLASSIFIER] {classifier_name}/{task}: no validation data available -- "
            f"falling back to the conventional 0.5 threshold for test metrics."
        )

    test_metrics: dict = {}
    n_test_used = 0
    if X_test is not None and y_test is not None:
        X_test_m, y_test_m = _mask_known(X_test, y_test)
        n_test_used = len(y_test_m)
        if n_test_used > 0:
            test_proba = clf.predict_proba(X_test_m)[:, 1]
            # The SAME threshold chosen from validation is applied here
            # -- the test set itself is never used to pick it.
            test_metrics = compute_binary_classification_metrics(y_test_m, test_proba, threshold=selected_threshold)

    return ClassifierAblationResult(
        classifier=classifier_name, task=task, train_metrics=train_metrics, val_metrics=val_metrics,
        test_metrics=test_metrics, fit_runtime_seconds=fit_runtime, n_train_used=len(y_train_m),
        n_val_used=n_val_used, n_test_used=n_test_used,
    )
