"""Defaults and validation for the federated (FL) downstream stages -- Configs 1-4 and personalisation.

Read from `three_modality.federated` of the YAML (all keys optional; `--set three_modality.federated.rounds.max_rounds=20` style overrides work as for every other stage).
Model / graph / SMOTE / focal / eligibility settings are NOT duplicated here: they come from `DOWNSTREAM_DEFAULTS` (`three_modality.downstream`) so the FL and Non-FL regimes share one
definition of the model families, the hospital-local graphs, SMOTE and the minimum-data rule.

IMPORTANT (stated in every manifest): the three encoders and the three-token cross-attention that produced the input embeddings are CENTRALLY pretrained on pooled training patients. The
FL regime here is therefore federated DOWNSTREAM learning on frozen representations -- it is not end-to-end federated learning and makes no claim about the encoders' privacy.
"""
from __future__ import annotations

import copy

from src.three_modality.config import _deep_merge

FL_DEFINITION_VERSION = "fl-v1"

FL_REGIME = ("Federated downstream learning on frozen, CENTRALLY pretrained representations: the three encoders and the three-token cross-attention were trained on pooled training patients, "
             "so this is NOT end-to-end federated learning. Only the downstream classifier / GCN is trained federatedly (clients = hospitals; only model parameters, scalar class counts and "
             "validation-score histograms leave a client; graphs, SMOTE nodes, scalers and thresholds stay hospital-local).")

NN_ALGORITHMS = ("fedavg", "fedprox", "fedadam", "scaffold")
FAMILIES = ("lr", "mlp", "gcn")                       # parameter-averaging families; trees use FedGBDT only
FL_CLASSIFIERS = ("lr", "mlp", "gbdt_fed")
PERSONALIZATION_METHODS = ("fedper", "fedrep", "finetune")
PERSONALIZED_MODELS = ("mlp", "gcn")                  # LR has a single layer: no body/head split exists
CLIENT_SCOPES = ("all", "nonfl_matched")

FL_DEFAULTS = {
    "seed": 42,
    # ---- clients: scope "all" = every hospital with >= 1 training patient (its data contributes to the federation); "nonfl_matched" = only the hospitals the Non-FL regime trained
    # (same `min_samples` rule) -- the control that isolates federation from the larger client population ----
    "scope": "all",
    "rounds": {"max_rounds": 150, "patience": 15, "min_rounds": 5, "local_steps": 5},   # full participation every round; `local_steps` full-batch steps per round
    "local": {                                       # ONE local optimiser for all four algorithms (SCAFFOLD's correction is only valid for plain SGD)
        "optimizer": "sgd",                          # "sgd" | "adam" (Adam: FedAvg / FedProx control only, state is re-initialised every round)
        "lr": {"lr": 0.3, "mlp": 0.1, "gcn": 0.1},
        "weight_decay": {"lr": 1e-3, "mlp": 1e-4, "gcn": 1e-4},   # same values as the Non-FL defaults
        "grad_clip": None,
    },
    "algorithms": {
        "fedavg": {},
        "fedprox": {"mu": 0.1},                      # proximal weight; the paper track's validation optimum was 1.0 on engineered features -- retuned here
        "fedadam": {"server_lr": 0.03, "beta1": 0.9, "beta2": 0.99, "tau": 1e-3},   # Reddi et al. 2021 (no bias correction)
        "scaffold": {},                              # option-II control variates, stateful clients, full participation, sample-weighted objective
    },
    "focal_alpha_mode": "global",                    # "global": alpha from the federated class counts (two scalars per client); "local": per-client alpha as in Non-FL (ablation)
    "scaler": "per_hospital",                        # standardisation fitted on each hospital's own training rows (the standing rule); no global statistics are ever computed
    # ---- federated validation: each client sends a histogram of its validation LOGITS per class + a loss sum; the server sums them -> global AUROC / loss. No score or label leaves a client. ----
    "validation": {"bins": 3000, "logit_range": 15.0, "selection_metric": "auroc"},
    "gbdt": {"n_bins": 64, "grid_size": 128, "max_rounds": 300, "early_stopping_rounds": 30,
             "params": {"learning_rate": 0.08, "max_depth": 4, "reg_lambda": 10.0, "min_child_weight": 3.0, "colsample_bytree": 0.6, "scale_pos_weight_mode": "none"}},
    "personalization": {
        "fedrep": {"head_steps": 3, "body_steps": 2},             # per round: head-only steps first (body frozen), then body-only steps (head frozen)
        "finetune": {"max_epochs": 60, "patience": 10, "lr": 0.05, "head_only": True, "selection": "loss"},   # local fine-tuning after FedAvg; selection on the hospital's validation loss
        "base_algorithm": "fedavg",                               # personalised bodies are aggregated with FedAvg only
    },
    "checkpoint": {"every_rounds": 10},
    "tuning": {
        "families": ["lr", "mlp", "gcn", "gbdt"], "algorithms": list(NN_ALGORITHMS), "task_for_tuning": "both", "quick": False,
        "grids": {"lr_scale": [0.3, 1.0, 3.0], "mu": [0.01, 0.1, 1.0], "server_lr": [0.01, 0.03, 0.1], "gbdt_depth": [3, 4, 6], "gbdt_spw": ["none", "sqrt"]},
    },
}

HPARAM_SECTIONS = ("local", "algorithms", "gbdt", "rounds", "personalization")


def validate_fl_cfg(c: dict) -> dict:
    r = c["rounds"]
    for k in ("max_rounds", "patience", "min_rounds", "local_steps"):
        if not isinstance(r[k], int) or r[k] < 1:
            raise ValueError(f"federated.rounds.{k} must be a positive integer")
    if r["min_rounds"] > r["max_rounds"]:
        raise ValueError("federated.rounds.min_rounds must be <= max_rounds")
    lo = c["local"]
    if lo["optimizer"] not in ("sgd", "adam"):
        raise ValueError("federated.local.optimizer must be 'sgd' or 'adam'")
    for fam in FAMILIES:
        if float(lo["lr"][fam]) <= 0 or float(lo["weight_decay"][fam]) < 0:
            raise ValueError(f"federated.local lr / weight_decay for {fam!r} must be > 0 / >= 0")
    a = c["algorithms"]
    if set(a) != set(NN_ALGORITHMS):
        raise ValueError(f"federated.algorithms must define exactly {NN_ALGORITHMS}")
    if float(a["fedprox"]["mu"]) < 0:
        raise ValueError("federated.algorithms.fedprox.mu must be >= 0")
    fa = a["fedadam"]
    if float(fa["server_lr"]) <= 0 or not 0 <= float(fa["beta1"]) < 1 or not 0 <= float(fa["beta2"]) < 1 or float(fa["tau"]) <= 0:
        raise ValueError("federated.algorithms.fedadam: server_lr > 0, beta1/beta2 in [0, 1), tau > 0")
    if c["scope"] not in CLIENT_SCOPES:
        raise ValueError(f"federated.scope must be one of {CLIENT_SCOPES}")
    if c["focal_alpha_mode"] not in ("global", "local"):
        raise ValueError("federated.focal_alpha_mode must be 'global' or 'local'")
    if c["scaler"] != "per_hospital":
        raise ValueError("federated.scaler: only 'per_hospital' is implemented (a federated/global scaler would break the per-hospital normalisation rule and needs explicit approval)")
    v = c["validation"]
    if int(v["bins"]) < 50 or float(v["logit_range"]) <= 0 or v["selection_metric"] not in ("auroc", "loss"):
        raise ValueError("federated.validation: bins >= 50, logit_range > 0, selection_metric in {auroc, loss}")
    pz = c["personalization"]
    if int(pz["fedrep"]["head_steps"]) < 1 or int(pz["fedrep"]["body_steps"]) < 1:
        raise ValueError("federated.personalization.fedrep head_steps / body_steps must be >= 1")
    if pz["base_algorithm"] != "fedavg":
        raise ValueError("federated.personalization.base_algorithm: only 'fedavg' is supported")
    if int(c["checkpoint"]["every_rounds"]) < 1:
        raise ValueError("federated.checkpoint.every_rounds must be >= 1")
    g = c["gbdt"]
    if g["params"].get("scale_pos_weight_mode", "none") not in ("none", "sqrt", "full"):
        raise ValueError("federated.gbdt.params.scale_pos_weight_mode must be none | sqrt | full")
    return c


def get_fl_cfg(cfg) -> dict:
    raw = getattr(cfg, "raw", cfg) or {}
    over = ((raw.get("three_modality") or {}).get("federated")) or {}
    return validate_fl_cfg(_deep_merge(copy.deepcopy(FL_DEFAULTS), over))


def apply_fl_hparams(fl_cfg: dict, hparams: "dict | None") -> dict:
    """Overlay validation-selected FL hyper-parameters (the `best_hparams` of an `fl_tune` run). Only optimisation / algorithm / GBDT / personalisation settings may be overridden --
    seeds, scope, scaler, validation protocol and eligibility cannot, so a tuning file can never change who is trained or how results are summarised."""
    out = copy.deepcopy(fl_cfg)
    if not hparams:
        return out
    bad = sorted(set(hparams) - set(HPARAM_SECTIONS))
    if bad:
        raise ValueError(f"FL hyper-parameter file may only set {HPARAM_SECTIONS}; found {bad}")
    return validate_fl_cfg(_deep_merge(out, copy.deepcopy(hparams)))
