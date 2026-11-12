"""Defaults and validation for the three-token cross-attention stage and the hospital-wise (Non-FL / FL) downstream stages.

Additive: lives beside `src/three_modality/config.py` and does not change it. Read from `three_modality.cross_attention` / `three_modality.downstream` of the YAML (all keys optional;
`--set three_modality.downstream.gcn.hidden=32` style overrides apply as for every other stage).
"""

from __future__ import annotations

import copy

from src.three_modality.config import _deep_merge, validate_focal_training

CROSS_ATTENTION_DEFAULTS = {
    # ---- model: LayerNorm+Linear projection of each frozen embedding into a common width, learned token-type embedding, ONE self-attention block over the 3 tokens
    # (residual + LayerNorm), a masked, content-free token + learned output slot for an UNAVAILABLE modality, flatten + availability flags -> fused embedding ----
    "model": {"attn_dim": 64, "num_heads": 4, "dropout": 0.1, "output_dim": 128},
    # ---- training: centrally POOLED over the training patients of all hospitals (the cross-attention is a shared pretrained representation, NOT a federated model) ----
    "training": {
        "focal": {"mortality": {"alpha": 0.25, "gamma": 2.0}, "sepsis": {"alpha": 0.25, "gamma": 2.0}},   # alpha weights the POSITIVE class (repo masked_focal_loss)
        "task_weights": {"mortality": 1.0, "sepsis": 1.0},
        "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 1024, "max_epochs": 30, "patience": 5, "grad_clip": 1.0, "seed": 42,
        "landmark_only": True,
    },
}

DOWNSTREAM_DEFAULTS = {
    "seed": 42,
    # a (hospital, task) pair is only trained when it has enough data; everything else is REPORTED as skipped with a reason
    "min_samples": {"train": 20, "train_pos": 3, "train_neg": 3, "val_pos": 1, "val_neg": 1},
    # focal loss for the torch models (LR, MLP, GCN) and LightGBM; alpha weights the POSITIVE class. "balanced" = fraction of NEGATIVES in that hospital's training data.
    # With SMOTE the classes are already balanced, so alpha is forced to `alpha_with_smote` (neutral 0.5) instead of re-weighting a second time.
    "focal": {"gamma": 2.0, "alpha": "balanced", "alpha_with_smote": 0.5},
    "smote": {"ratio": 1.0, "k_neighbors": 5, "min_positives": 2, "max_synthetic_per_positive": 20},
    # Graph A = kNN + cosine threshold ("knn_and_threshold"); Graph B = MiniBatchKMeans candidates + kNN/threshold inside each cluster. Same defaults as the paper track.
    "graph": {"k": 10, "cosine_threshold": 0.5, "n_clusters": 8, "cluster_seed": 42, "max_nodes_full_similarity": 2000},
    # `selection_metric` = what picks the checkpoint / boosting stage on the hospital's VALIDATION patients: "loss" (focal loss; the original behaviour), "auroc" or "auprc" (ranking quality).
    # `min_epochs`: epochs before which a checkpoint cannot be selected (guards against picking a near-initial model on a tiny, noisy validation set). `gamma` / `alpha` optionally override
    # the global focal setting for one model family (used by the tuning stage).
    "gcn": {"hidden": 64, "layers": 2, "dropout": 0.2, "lr": 5e-3, "weight_decay": 1e-4, "max_epochs": 200, "patience": 25, "selection_metric": "loss", "min_epochs": 1},
    "classifiers": {
        "lr": {"lr": 1e-2, "weight_decay": 1e-3, "max_epochs": 300, "patience": 30, "selection_metric": "loss", "min_epochs": 1},
        "mlp": {"hidden": 64, "dropout": 0.1, "lr": 5e-3, "weight_decay": 1e-4, "max_epochs": 300, "patience": 30, "selection_metric": "loss", "min_epochs": 1},
        # LightGBM focal objective, stabilised: the TRUE focal Hessian is ~0 or negative for confidently-wrong samples, so it is floored at `hessian_floor` x (class weight) instead of 1e-6;
        # with `lambda_l2`, `max_delta_step` and `min_sum_hessian_in_leaf` this keeps leaf values bounded (the 1e-6 floor produced exploding raw scores).
        "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 5, "early_stopping_rounds": 30, "subsample": 1.0,
                     "hessian_floor": 0.05, "lambda_l2": 1.0, "max_delta_step": 1.0, "min_sum_hessian_in_leaf": 1e-3, "selection_metric": "loss"},
        "gbdt": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "subsample": 1.0, "selection_metric": "loss"},   # sklearn GradientBoostingClassifier, STANDARD deviance loss (no focal hook)
    },
    # secondary, VALIDATION-selected operating point reported next to the F1-maximising primary one: the highest threshold whose validation recall reaches `recall_target`
    "operating_points": {"recall_target": 0.65},
    # pooled-prediction reporting (needs the saved patient-level test predictions): bootstrap resamples and FLICS-style subsets of the k largest trained hospitals
    "reporting": {"n_bootstrap": 500, "top_hospitals": [3, 10], "seed": 0},
}

CLASSIFIERS = ("lr", "mlp", "lightgbm", "gbdt")
GRAPHS = {"A": "knn_and_threshold", "B": "minibatch_kmeans_and_threshold"}
TASKS = ("mortality", "sepsis")
EMBEDDING_SETS = ("fused", "gru", "tabular", "tcd")


def validate_cross_attention_cfg(c: dict) -> dict:
    m = c["model"]
    for k in ("attn_dim", "num_heads", "output_dim"):
        if not isinstance(m[k], int) or m[k] <= 0:
            raise ValueError(f"cross_attention.model.{k} must be a positive integer, got {m[k]!r}")
    if m["attn_dim"] % m["num_heads"] != 0:
        raise ValueError("cross_attention.model.attn_dim must be divisible by num_heads")
    if not 0.0 <= float(m["dropout"]) < 1.0:
        raise ValueError("cross_attention.model.dropout must be in [0, 1)")
    validate_focal_training(c["training"], "cross_attention.training")
    return c


def get_cross_attention_cfg(cfg) -> dict:
    raw = getattr(cfg, "raw", cfg) or {}
    over = ((raw.get("three_modality") or {}).get("cross_attention")) or {}
    return validate_cross_attention_cfg(_deep_merge(copy.deepcopy(CROSS_ATTENTION_DEFAULTS), over))


def validate_downstream_cfg(c: dict) -> dict:
    ms = c["min_samples"]
    for k, v in ms.items():
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"downstream.min_samples.{k} must be a non-negative integer")
    f = c["focal"]
    if float(f["gamma"]) < 0:
        raise ValueError("downstream.focal.gamma must be >= 0")
    a = f["alpha"]
    if not (a == "balanced" or (isinstance(a, (int, float)) and 0.0 < float(a) < 1.0)):
        raise ValueError("downstream.focal.alpha must be 'balanced' or a number in (0, 1)")
    if not 0.0 < float(f["alpha_with_smote"]) < 1.0:
        raise ValueError("downstream.focal.alpha_with_smote must be in (0, 1)")
    s = c["smote"]
    if not 0.0 < float(s["ratio"]) <= 1.0:
        raise ValueError("downstream.smote.ratio must be in (0, 1]")
    if int(s["min_positives"]) < 2:
        raise ValueError("downstream.smote.min_positives must be >= 2 (interpolation needs two positives)")
    g = c["graph"]
    if int(g["k"]) < 1 or not -1.0 <= float(g["cosine_threshold"]) <= 1.0 or int(g["n_clusters"]) < 1:
        raise ValueError("downstream.graph needs k >= 1, cosine_threshold in [-1, 1], n_clusters >= 1")
    metrics = ("loss", "auroc", "auprc")
    for name, m in [("gcn", c["gcn"])] + [(f"classifiers.{k}", v) for k, v in c["classifiers"].items()]:
        if m.get("selection_metric", "loss") not in metrics:
            raise ValueError(f"downstream.{name}.selection_metric must be one of {metrics}")
        me = int(m.get("min_epochs", 1))
        if me < 1 or ("max_epochs" in m and me > int(m["max_epochs"])):
            raise ValueError(f"downstream.{name}.min_epochs must be in [1, max_epochs]")
        for k in ("gamma", "alpha"):
            if k in m and m[k] is not None and not (k == "alpha" and m[k] == "balanced") and float(m[k]) < 0:
                raise ValueError(f"downstream.{name}.{k} must be >= 0")
    lg = c["classifiers"]["lightgbm"]
    if float(lg["hessian_floor"]) <= 0 or float(lg["lambda_l2"]) < 0 or float(lg["max_delta_step"]) < 0 or float(lg["min_sum_hessian_in_leaf"]) < 0:
        raise ValueError("downstream.classifiers.lightgbm: hessian_floor must be > 0 and lambda_l2 / max_delta_step / min_sum_hessian_in_leaf >= 0")
    rt = float(c["operating_points"]["recall_target"])
    if not 0.0 < rt <= 1.0:
        raise ValueError("downstream.operating_points.recall_target must be in (0, 1]")
    return c


def get_downstream_cfg(cfg) -> dict:
    raw = getattr(cfg, "raw", cfg) or {}
    over = ((raw.get("three_modality") or {}).get("downstream")) or {}
    return validate_downstream_cfg(_deep_merge(copy.deepcopy(DOWNSTREAM_DEFAULTS), over))


HPARAM_SECTIONS = ("classifiers", "gcn", "graph", "focal", "smote")


def apply_hparams(ds_cfg: dict, hparams: "dict | None") -> dict:
    """Overlay tuned hyperparameters (the `best_hparams` of a `nonfl_tune` run) onto a downstream config. Only model / graph / loss hyperparameters can be overridden; seeds, eligibility
    (`min_samples`) and reporting settings cannot, so a tuning file can never change who is trained or how results are summarised."""
    out = copy.deepcopy(ds_cfg)
    if not hparams:
        return out
    bad = sorted(set(hparams) - set(HPARAM_SECTIONS))
    if bad:
        raise ValueError(f"hyperparameter file may only set {HPARAM_SECTIONS}; found {bad}")
    return validate_downstream_cfg(_deep_merge(out, copy.deepcopy(hparams)))


TUNING_DEFAULTS = {"n_hospitals": 60, "families": ["lr", "mlp", "lightgbm", "gbdt", "gcn"], "quick": False}


def get_tuning_cfg(cfg) -> dict:
    raw = getattr(cfg, "raw", cfg) or {}
    out = _deep_merge(copy.deepcopy(TUNING_DEFAULTS), ((raw.get("three_modality") or {}).get("tuning")) or {})
    if int(out["n_hospitals"]) < 3:
        raise ValueError("three_modality.tuning.n_hospitals must be >= 3")
    return out
