"""Defaults for the paper track. Everything is overridable under a `paper:`
stanza in a YAML config (configs/paper.yaml, configs/paper_debug.yaml); no
edit to configs/base.yaml is required."""

from __future__ import annotations

import copy
from typing import Any

# Feature groups tagged on every engineered feature (see features.py).
# A "sepsis ingredient" is anything that feeds, or near-duplicates, the
# sepsis proxy label (antibiotic AND SOFA-like >= 2), so the controlled
# sepsis arm can remove it and report an honest (non-circular) number.
SEPSIS_INGREDIENT_GROUPS = (
    "apache_physiology", "apache_score", "lab_sofa", "medication_antibiotic",
    "medication_vasopressor", "infusion_vasopressor", "microlab",
    "diagnosis_infection", "treatment", "medication", "infusion",
)

# Removed in the deconfounded steps of the sepsis ladder: infection-related admitting diagnoses, organ-failure diagnosis tokens, and the whole
# arterial-blood-gas family (ABGs are drawn on ventilated / respiratory-failure patients, i.e. they stand in for the respiratory SOFA input;
# paCO2 alone was the #1 feature of the "controlled" sepsis model).
SEPSIS_DECONFOUND_REGEX = [
    r"apacheadmissiondx__.*(sepsis|septic|pneumon|infect|cellulit|abscess|mening|bacter|peritonit|urinary|osteomyel|endocard|necrotiz|fasciit|cholang|empyema|gangrene)",
    r"^dx_.*(failure|shock|insufficiency|encephalopathy|coagulopathy|thrombocytopen|ards|hypotension)",
    r"^lab_(pao2|paco2|fio2|ph|hco3|base_excess|o2_sat|sao2|spo2|bicarbonate_abg)_",
]

PAPER_DEFAULTS: dict[str, Any] = {
    "output_subdir": "paper",
    "window": {
        # First-24h observation window (APACHE-style day-1). Features only
        # use data with offset <= min(window_minutes, unitdischargeoffset);
        # unitdischargeoffset is used ONLY to clip the window, never as a feature.
        "minutes": 1440,
        "lookback_minutes": 1440,       # pre-ICU labs/diagnoses allowed this far back
        "min_icu_los_minutes": 0,       # >0 => strict landmark cohort (sensitivity analysis)
    },
    "features": {
        "top_labs": 40, "top_treatments": 60, "top_diagnoses": 60,
        "top_admission_drugs": 40, "top_medications": 60, "top_infusions": 20,
        "top_categories": 12, "top_admission_dx": 30, "min_vocab_patients": 5,
        "diagnosis_depth": 3, "treatment_depth": 3,
        "include_apache_scores": True,
        "chunksize": 1_000_000,
        "lab_stats": ["mean", "min", "max", "first", "last", "std", "count", "delta"],
        "cache": True,
    },
    "sequence": {"enabled": True, "n_labs": 12, "bin_minutes": 120},   # 12 two-hourly bins over the 24h window
    # task -> list of feature-set names to run. Feature sets are defined below.
    "tasks": {"mortality": ["full", "no_apache_scores"], "sepsis": ["sepsis_controlled"]},
    "feature_sets": {
        "full": {"exclude_groups": []},
        "no_apache_scores": {"exclude_groups": ["apache_score"]},
        "sepsis_controlled": {"exclude_groups": list(SEPSIS_INGREDIENT_GROUPS)},
        "sepsis_uncontrolled": {"exclude_groups": []},   # for the leakage-gap analysis only
        # robustness: raw measurements only AND without end-of-life / comfort-care tokens (answers "is mortality driven by
        # outcome-adjacent care decisions?")
        # Sepsis ladder (each step removes more of what is circular with the proxy label):
        #   sepsis_uncontrolled -> sepsis_controlled -> sepsis_deconfounded -> sepsis_no_labs
        # `deconfounded` also drops infection-related ADMISSION diagnoses (the admitting reason is nearly the label),
        # organ-failure diagnosis tokens, and the oxygenation labs that are the same measurement as the respiratory SOFA input.
        "sepsis_deconfounded": {"exclude_groups": list(SEPSIS_INGREDIENT_GROUPS), "exclude_name_regex": SEPSIS_DECONFOUND_REGEX},
        # + hospital-level covariates (hospital size/region identify the hospital, and the proxy label's positive rate varies by hospital
        # through antibiotic-charting coverage, so these inflate POOLED sepsis AUROC without any clinical signal)
        "sepsis_no_hospital": {"exclude_groups": list(SEPSIS_INGREDIENT_GROUPS) + ["hospital"], "exclude_name_regex": SEPSIS_DECONFOUND_REGEX},
        "sepsis_no_labs": {"exclude_groups": list(SEPSIS_INGREDIENT_GROUPS) + ["hospital", "lab"], "exclude_name_regex": SEPSIS_DECONFOUND_REGEX},
        # `sepsis_clean` = no_hospital + drops `apv_activetx` (APACHE "active treatment" = treatment intensity; the #1 feature of the
        # no-lab models and adjacent to the antibiotic half of the label). The primary sepsis feature set when labels are restricted to
        # label-valid hospitals (see sepsis_hospital_filter).
        "sepsis_clean": {"exclude_groups": list(SEPSIS_INGREDIENT_GROUPS) + ["hospital"], "exclude_name_regex": SEPSIS_DECONFOUND_REGEX + [r"^apv_activetx$"]},
        "no_apache_no_eol": {"exclude_groups": ["apache_score"],
                             "exclude_name_regex": ["palliat", "comfort", "hospice", "dnr", "do_not_resus", "withdraw", "end_of_life", "terminal"]},
    },
    # Pooled run of the SAME histogram GBDT (one client holding every training row), off by default. `reuse_params_from` = a finished federated-GBDT run directory whose winning
    # hyperparameters are reused, so the two differ only in not being federated; null = the arm runs its own validation search.
    "gbdt_central": {"reuse_params_from": None},
    "arms": {
        "logreg_central": True, "lightgbm_central": True, "xgboost_central": False,
        "mlp_central": True, "mlp_local_only": False, "fedavg_mlp": True,
        "fedavg_graph": True,
        "lightgbm_sampling": False, "fedavg_logreg": False, "fed_gbdt": False, "gbdt_central": False,      # negative-result section: hospital-wise random under/oversampling on the STRONG features
    },
    # The sepsis proxy needs medication records to fire. Restrict the SEPSIS task (never mortality) to hospitals whose medication-record
    # coverage is at least this share of their stays. Set enabled=false to reproduce the earlier unfiltered sepsis numbers.
    "sepsis_hospital_filter": {"enabled": True, "min_medication_coverage": 0.8},
    "sampling": {"strategies": ["undersample", "oversample"], "ratios": [0.25, 0.5, 1.0]},
    # Optional hospital-size cohort filter (0 = off = previous behaviour); see configs/paper_flicsstyle.yaml and scripts/paper_hospital_size_probe.py.
    "cohort_filter": {"min_hospital_patients": 0},
    # Federated histogram GBDT arm (`fed_gbdt`, off by default; see src/paper/fed_gbdt.py). Trial 0 = `fixed`; further trials are random
    # draws from `search`, all selected on validation AUROC only (mirrors the LightGBM arm). n_trials=1 trains `fixed` only.
    "fed_gbdt": {
        "n_bins": 64, "quantile_grid": 128, "max_rounds": 600, "early_stopping_rounds": 30, "n_trials": 4,
        "explicit_client_histograms": False,       # True runs the literal per-client message path (same trees, slower)
        "fixed": {"learning_rate": 0.08, "max_depth": 6, "reg_lambda": 10.0, "min_child_weight": 3.0, "min_gain": 0.0,
                  "colsample_bytree": 0.6, "subsample": 1.0, "scale_pos_weight_mode": "none"},
        "search": {"learning_rate": [0.05, 0.08, 0.12], "max_depth": [4, 6, 8], "reg_lambda": [1.0, 10.0, 30.0],
                   "min_child_weight": [1.0, 3.0, 10.0], "colsample_bytree": [0.4, 0.6, 0.8], "scale_pos_weight_mode": ["none", "sqrt"]},
    },
    "seeds": [42, 43, 44],
    "lightgbm": {
        "n_trials": 12, "early_stopping_rounds": 100, "n_estimators": 3000, "n_jobs": -1,
        "search": {
            "learning_rate": [0.02, 0.03, 0.05], "num_leaves": [15, 31, 63],
            "min_child_samples": [20, 50, 100], "feature_fraction": [0.4, 0.6, 0.8],
            "bagging_fraction": [0.7, 0.9, 1.0], "lambda_l2": [0.0, 1.0, 10.0],
            "scale_pos_weight_mode": ["none", "sqrt"],
        },
    },
    "logreg": {"C": 0.1, "max_iter": 500},
    "neural": {
        "hidden": [256, 128], "emb_dim": 64, "dropout": 0.2, "batch_size": 256,
        "lr": 1e-3, "weight_decay": 1e-4, "central_epochs": 40, "patience": 6,
        "pos_weight": "sqrt",          # none | sqrt | full
        "selection_metric": "auroc",   # auroc | auprc
        "seq_hidden": 48, "use_sequence": True, "clip": 5.0, "min_hospital_n_for_local_stats": 30,
        "missing_indicator_range": [0.02, 0.98], "local_epochs_local_only": 30,
    },
    # FedAvg progress per round ~ the n-weighted number of sequential steps a client takes, so small
# batches / several local epochs / a slightly higher lr are needed or it underfits (the same failure mode
# as the old 5-full-batch-step encoders). Effective steps per round are logged by run_fedavg.
    "federated": {"rounds": 40, "local_epochs": 5, "batch_size": 128, "lr": 3e-3, "fedprox_mu": 0.0, "patience": 8,
                  # stabilisers, all no-ops at these defaults: per-round lr decay, cap on local optimizer steps per client per round
                  # (stops the few very large hospitals dominating each round), server momentum (FedAvgM)
                  "lr_decay": 1.0, "max_local_steps": None, "server_momentum": 0.0, "server_lr": 1.0},
    "federated_logreg": {"lr": 0.01, "local_epochs": 3, "batch_size": 256, "rounds": 40, "patience": 8},
    "graph": {
        "k": 10, "cosine_threshold": 0.5, "n_clusters": 8, "cluster_seed": 42,
        "max_nodes_full_similarity": 2000, "hidden": 64, "layers": 2, "dropout": 0.2,
        "rounds": 30, "local_epochs": 5, "lr": 5e-3, "residual": True,
    },
    "evaluation": {
        "local_calibration_prior_sd": 1.0, "bootstrap_n": 500, "threshold_rule": "max_f1_val", "calibrate": True,
        "per_hospital_min_pos": 5, "per_hospital_min_neg": 5, "ci_alpha": 0.05,
        "primary_metric": "auroc", "compare_to": "lightgbm_central",
    },
    # Reported by the cited papers (values as extracted from their text; cohorts, windows and feature sets differ -- see report caveat).
    # Only the base paper (FLICS, eICU alone) is listed. Papers on MIMIC or other databases are not compared.
    "reference_results": [
        {"name": "FLICS: federated LR/LSTM, 3 hospitals", "metric": "auroc", "value": 0.87, "eicu_only": True,
         "source": "Federated_Learning_for_ICU_Mortality_Prediction (multi-hospital); 1,000-patient test set, CI 0.83-0.90"},
        {"name": "FLICS: federated LSTM, 10 hospitals", "metric": "auroc", "value": 0.84, "eicu_only": True,
         "source": "Federated_Learning_for_ICU_Mortality_Prediction (multi-hospital); federated LR is 0.83"},
    ],
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_paper_cfg(cfg) -> dict:
    """PAPER_DEFAULTS deep-merged with cfg.raw['paper'] (if any)."""
    raw = getattr(cfg, "raw", cfg) or {}
    return _deep_merge(PAPER_DEFAULTS, raw.get("paper", {}) if isinstance(raw, dict) else {})
