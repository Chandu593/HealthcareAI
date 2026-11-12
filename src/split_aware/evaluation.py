"""
Split-aware Phase 11/12/13 -- research-grade evaluation reporting.

Aggregates per-hospital predictions (produced by
`_run_local_training_round()`'s / `_run_centralized_sequential_training()`'s
split-aware `predictions` output -- per-patient logits, labels, and a
train/not_train tag) into ONE global prediction set per task, then
computes AUROC/AUPRC/accuracy/precision/recall/F1/sensitivity/
specificity/calibration (Brier score) -- selecting the decision
threshold from VALIDATION data only
(src/classifiers/boosting_classifiers.py's
`select_threshold_from_validation()`), applying that SAME threshold to
TEST, per the standing rule "use validation for threshold/model
selection, reserve test for final evaluation" (confirmed 2026-09-11,
requirement #9).

`is_train_row` (as saved by the training loops) only distinguishes
"train" from "not_train" -- the training loop itself has no reason to
know val vs test, only which rows it may touch. To report val and test
SEPARATELY, this module additionally consults the split manifest
itself (the actual three-way split membership), not just the training
loop's binary tag.
"""

from __future__ import annotations

import numpy as np

from src.classifiers.boosting_classifiers import (
    compute_binary_classification_metrics,
    select_threshold_from_validation,
)
from src.data.splitting.patient_split import SplitManifest


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def evaluate_final_model_on_full_graphs(
    cfg, phase3_result: dict, phase4_result: dict, phase5_result: dict, phase8_result: dict,
    state_dict_path: str, train_patient_ids: set, sepsis_cohort: str = "broad",
) -> dict:
    """Loads a saved model checkpoint (e.g. Phase 12's FINAL AGGREGATED
    global model) and runs ONE frozen forward pass per hospital over
    that hospital's FULL graph, producing predictions for every patient
    (train+val+test) in the same per-hospital shape
    `_run_local_training_round()`'s own split-aware `predictions`
    output uses -- so the result feeds straight into
    `evaluate_model_predictions()`/`evaluate_hospital_results()`.

    THIS CLOSES A REAL GAP found during the 2026-09-11 pilot review, not
    hypothetical: Phase 12's own per-round local training only ever
    produces predictions from each hospital's POST-LOCAL-TRAINING
    weights (that round's contribution to the average), never from the
    FINAL AGGREGATED weights that FedAvg actually produces as its
    output. The central federated-learning research question --
    "how does the federated model perform on held-out data" -- can only
    be answered by evaluating the aggregate itself, which needs this
    one extra frozen inference pass; it is not something any existing
    per-round data already contains.
    """
    import torch

    from run_pipeline import _load_hospital_batch
    from src.modalities.tabular.autoencoder import AutoencoderConfig
    from src.modalities.timeseries.lstm_encoder import LSTMEncoderConfig
    from src.training.local_model import LocalModelConfig, LocalPatientModel

    ts_input_dim = len(phase4_result["feature_names"])
    tab_input_dim = len(phase5_result["feature_names"])
    model_config = LocalModelConfig(
        lstm=LSTMEncoderConfig(
            input_dim=ts_input_dim, hidden_dim=cfg.get("modalities.timeseries.hidden_dim", 128),
            num_layers=cfg.get("modalities.timeseries.num_layers", 2),
            dropout=cfg.get("modalities.timeseries.dropout", 0.2),
        ),
        autoencoder=AutoencoderConfig(
            input_dim=tab_input_dim, hidden_dim=cfg.get("modalities.tabular.hidden_dim", 128),
            embedding_dim=cfg.get("modalities.tabular.embedding_dim", 64),
            dropout=cfg.get("modalities.tabular.dropout", 0.1),
        ),
        gnn_hidden_dim=cfg.get("gnn.hidden_dim", 128), gnn_n_layers=cfg.get("gnn.layers", 2),
        gnn_type=cfg.get("gnn.type", "graphsage"), gnn_dropout=cfg.get("gnn.dropout", 0.2),
        head_hidden_dim=cfg.get("multitask.head_hidden_dim", 64),
        head_dropout=cfg.get("multitask.head_dropout", 0.1),
    )
    model = LocalPatientModel(model_config)
    state_dict = torch.load(state_dict_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    ts_row_of_patient = {int(pid): i for i, pid in enumerate(phase4_result["patient_ids"])}
    tab_row_of_patient = {int(pid): i for i, pid in enumerate(phase5_result["patient_ids"])}
    ts_array = phase4_result["normalized_array"]
    ts_seq_length = phase4_result["sequence_length"]
    tab_array = phase5_result["normalized_array"]

    predictions_by_hospital: dict = {}
    with torch.no_grad():
        for hid in sorted(phase8_result["hospital_graph_paths"].keys()):
            batch = _load_hospital_batch(
                hid, phase8_result["hospital_graph_paths"], ts_row_of_patient, ts_array, ts_seq_length,
                tab_row_of_patient, tab_array, phase3_result, sepsis_cohort,
            )
            if batch is None:
                continue
            out = model(
                batch["timeseries_x"], batch["timeseries_lengths"], batch["tabular_x"],
                batch["edge_index"], batch["edge_weight"],
            )
            is_train_row = ["train" if pid in train_patient_ids else "not_train" for pid in batch["patient_ids"]]
            predictions_by_hospital[hid] = {
                "patient_ids": batch["patient_ids"],
                "mortality_logits": out["mortality_logits"].numpy().tolist(),
                "sepsis_logits": out["sepsis_logits"].numpy().tolist(),
                "mortality_labels": batch["mortality_labels"].numpy().tolist(),
                "sepsis_labels": batch["sepsis_labels"].numpy().tolist(),
                "is_train_row": is_train_row,
            }
    return predictions_by_hospital


def aggregate_predictions_by_hospital(hospital_predictions: dict) -> dict:
    """`hospital_predictions`: {hid: {"patient_ids", "mortality_logits",
    "sepsis_logits", "mortality_labels", "sepsis_labels", "is_train_row"}}
    -- exactly the shape of `_run_local_training_round()`'s per-hospital
    `predictions` field, or `_run_centralized_sequential_training()`'s
    `predictions_by_hospital`. Returns one global dict of numpy arrays,
    concatenated in sorted hospital-id order (deterministic, not
    insertion-order-dependent).
    """
    all_patient_ids, all_mort_logits, all_sepsis_logits = [], [], []
    all_mort_labels, all_sepsis_labels, all_is_train = [], [], []
    for hid in sorted(hospital_predictions.keys()):
        preds = hospital_predictions[hid]
        all_patient_ids.extend(preds["patient_ids"])
        all_mort_logits.extend(preds["mortality_logits"])
        all_sepsis_logits.extend(preds["sepsis_logits"])
        all_mort_labels.extend(preds["mortality_labels"])
        all_sepsis_labels.extend(preds["sepsis_labels"])
        all_is_train.extend(preds["is_train_row"])
    return {
        "patient_ids": np.array(all_patient_ids),
        "mortality_proba": _sigmoid(np.array(all_mort_logits, dtype=np.float64)),
        "sepsis_proba": _sigmoid(np.array(all_sepsis_logits, dtype=np.float64)),
        "mortality_labels": np.array(all_mort_labels, dtype=np.float64),
        "sepsis_labels": np.array(all_sepsis_labels, dtype=np.float64),
        "is_train_row": np.array(all_is_train),
    }


def evaluate_model_predictions(hospital_predictions: dict, split_manifest: SplitManifest, task: str) -> dict:
    """Returns `{"train": {...}, "val": {...}, "test": {...}}` for the
    given task ("mortality" | "sepsis") -- threshold selected on val,
    reused for test. A split with fewer than 2 known-label patients or
    only one class present returns `{}` for that split rather than
    crashing or fabricating a value (mirrors
    `fit_and_evaluate_classifier()`'s own graceful-skip behavior).
    """
    if task not in ("mortality", "sepsis"):
        raise ValueError(f"task must be 'mortality' or 'sepsis', got {task!r}.")

    agg = aggregate_predictions_by_hospital(hospital_predictions)
    proba = agg[f"{task}_proba"]
    labels = agg[f"{task}_labels"]
    patient_ids = agg["patient_ids"]

    split_of_row = np.array([split_manifest.split_of(int(pid)) or "train" for pid in patient_ids])

    def _masked(mask: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        y = labels[mask]
        p = proba[mask]
        known = ~np.isnan(y)
        return y[known], p[known]

    metrics: dict = {}

    y_train, p_train = _masked(split_of_row == "train")
    metrics["train"] = (
        compute_binary_classification_metrics(y_train, p_train) if len(y_train) and len(np.unique(y_train)) > 1 else {}
    )

    y_val, p_val = _masked(split_of_row == "val")
    threshold = 0.5
    if len(y_val) and len(np.unique(y_val)) > 1:
        threshold = select_threshold_from_validation(y_val, p_val)
        metrics["val"] = compute_binary_classification_metrics(y_val, p_val, threshold=threshold)
    else:
        metrics["val"] = {}

    y_test, p_test = _masked(split_of_row == "test")
    metrics["test"] = (
        compute_binary_classification_metrics(y_test, p_test, threshold=threshold)
        if len(y_test) and len(np.unique(y_test)) > 1 else {}
    )

    return metrics


def evaluate_predictions_by_hospital_dict(predictions_by_hospital: dict, split_manifest: SplitManifest) -> dict:
    """Convenience wrapper for the FLAT `predictions_by_hospital` shape
    (`evaluate_final_model_on_full_graphs()`'s return, or Phase 13
    centralized-sequential's `predictions_by_hospital`) -- as opposed
    to `evaluate_hospital_results()`, which unwraps Phase 11/13
    local-only's nested `hospital_results[hid]["predictions"]` shape.
    Returns `{"mortality": {...}, "sepsis": {...}}`.
    """
    if not predictions_by_hospital:
        return {"mortality": {}, "sepsis": {}}
    return {
        "mortality": evaluate_model_predictions(predictions_by_hospital, split_manifest, "mortality"),
        "sepsis": evaluate_model_predictions(predictions_by_hospital, split_manifest, "sepsis"),
    }


def evaluate_hospital_results(hospital_results: dict, split_manifest: SplitManifest) -> dict:
    """Convenience wrapper for Phase 11/12's `hospital_results` shape
    (each hospital's dict has a `"predictions"` sub-key, unlike Phase
    13 centralized's flatter `predictions_by_hospital`). Returns
    `{"mortality": {...}, "sepsis": {...}}`.
    """
    hospital_predictions = {
        hid: r["predictions"] for hid, r in hospital_results.items() if "predictions" in r
    }
    if not hospital_predictions:
        return {"mortality": {}, "sepsis": {}}
    return {
        "mortality": evaluate_model_predictions(hospital_predictions, split_manifest, "mortality"),
        "sepsis": evaluate_model_predictions(hospital_predictions, split_manifest, "sepsis"),
    }
