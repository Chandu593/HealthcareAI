"""
Graph A vs Graph B comparison (confirmed with the user 2026-09-23),
run AFTER the best flat configuration has been selected from the real
sampling comparison (`sampling_comparison_runner.py`). Never runs
alone -- every call to `run_graph_comparison()` builds and evaluates
BOTH Graph A and Graph B, from the SAME selected patient embeddings,
the SAME hospital boundaries, the SAME k / cosine threshold / edge-
weight definition, and the SAME GNN architecture, training settings,
split, and evaluation protocol. The only intended difference is the
candidate-generation strategy:

  - Graph A: `build_hospital_graph(method="knn_and_threshold")` --
    exact intra-hospital k-NN AND cosine-threshold intersection
    (top-k AND cosine >= threshold).
  - Graph B: `build_hospital_graph(method="minibatch_kmeans_and_
    threshold")` -- intra-hospital MiniBatch K-Means candidate
    restriction, then EXACT cosine similarity within each cluster, then
    the SAME top-k-AND-threshold rule.

CORRECTED 2026-09-23: the downstream predictor for Graph A/B is the
GNN's OWN natural fully-connected multitask head
(`src/training/graph_local_model.py`'s `GraphOnlyLocalModel`, trained
end-to-end via backprop) -- NOT a separate boosted-tree classifier
(AdaBoost/XGBoost/LightGBM). That classifier-based approach belongs
ONLY to the flat-model ablation/sampling-comparison stage (comparing
many encoder combinations cheaply); it is never appropriate once a
single encoder combination has been frozen and a graph built on top of
it. See `graph_local_model.py`'s own module docstring for the fuller
rationale and the leakage-safety guarantee (train-only subgraph for
the gradient-computing pass, full-graph inductive inference for
val/test).

Per-hospital LOCAL training only, no FedAvg (confirmed with the user
2026-09-20: "skip the federated-learning concept" for this work) --
each hospital trains its own independent GraphOnlyLocalModel, exactly
mirroring how the flat ablation framework's encoders are also trained
per-hospital, never pooled.

Does NOT rerun the completed 48-config ablation, and does NOT touch
PCA (still deferred) -- reuses the SAME embeddings-cache mechanism
`sampling_comparison_runner.py` already uses (`_get_or_build_combo_
embeddings()`), so if the winning combo's embeddings are already
cached from the sampling comparison, this step regenerates nothing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

from src.ablation.config_grid import EmbeddingComboSpec
from src.ablation.embedding_ablation_runner import _build_ts_tab_lookup, get_or_build_combo_embeddings
from src.ablation.results_writer import ExperimentRecord, ResultsWriter
from src.ablation.sampling_comparison_runner import _hospital_id_lookup
from src.classifiers.boosting_classifiers import compute_binary_classification_metrics, select_threshold_from_validation
from src.classifiers.imbalance_sampling import metric_sort_key, select_best_candidate
from src.data.splitting.patient_split import SplitManifest
from src.graph.patient_graph import build_hospital_graph
from src.heads.multitask_heads import align_labels_to_patient_ids
from src.training.graph_local_model import predict_graph_model_full_hospital, train_graph_model_one_hospital

GRAPH_METHODS = {
    "graph_a": "knn_and_threshold",
    "graph_b": "minibatch_kmeans_and_threshold",
}


def _row_metrics(row: dict, split: str) -> dict:
    """Same dual-shape support used throughout this module family: a
    flat CSV row (dotted keys, string values) or a nested JSON record
    (metrics.<split>, native types)."""
    flat = {k[len(f"{split}.") :]: v for k, v in row.items() if k.startswith(f"{split}.")}
    if flat:
        out = {}
        for name, val in flat.items():
            if val in (None, "", "None"):
                out[name] = None
                continue
            try:
                out[name] = float(val)
            except (TypeError, ValueError):
                out[name] = val
        return out
    return dict((row.get("metrics") or {}).get(split) or {})


def _row_config(row: dict) -> dict:
    flat = {k[len("config.") :]: v for k, v in row.items() if k.startswith("config.")}
    return flat or dict(row.get("config") or {})


def select_best_flat_config_from_sampling_results(rows: list, task: str = "mortality") -> dict:
    """Applies the SAME selection priority confirmed for the sampling
    comparison itself (F1 -> recall -> precision -> AUROC -> AUPRC ->
    balanced accuracy -> specificity -> Brier, via `metric_sort_key`/
    `select_best_candidate`) to `sampling_results.csv`'s rows, using
    VALIDATION metrics only, restricted to `task` (default
    "mortality").

    NOTE, flagged rather than silently resolved: this selects the
    single best (ts_model, tab_model, fusion_mode) ENCODER combination
    for `task` specifically. Since mortality and sepsis share one
    frozen embedding in this project's multitask design, the winning
    combo is used for BOTH tasks' graph comparison -- if mortality and
    sepsis would independently prefer DIFFERENT encoder combinations,
    this function does not reconcile that; call it once per task and
    compare, or set `ablation.graph_comparison.best_flat_config`
    explicitly to override the automatic selection.

    Raises `ValueError` if no row has usable validation metrics for
    `task` -- never silently falls back to an arbitrary row.
    """
    candidates = [r for r in rows if r.get("task") == task and _row_metrics(r, "val")]
    if not candidates:
        raise ValueError(f"No sampling_results.csv rows with usable validation metrics found for task={task!r}.")
    best_row = select_best_candidate(candidates, metrics_of=lambda r: _row_metrics(r, "val"))
    config = _row_config(best_row)
    return {
        "ts_model": config["ts_model"], "tab_model": config["tab_model"], "fusion_mode": config["fusion_mode"],
        "source_classifier": config.get("classifier"), "source_sampling_strategy": config.get("sampling_strategy"),
        "source_sampling_ratio": config.get("sampling_ratio"), "source_task": task,
        "source_experiment_id": best_row.get("experiment_id"),
        "flat_val_metrics": _row_metrics(best_row, "val"), "flat_test_metrics": _row_metrics(best_row, "test"),
    }


def _write_graph_comparison_record(
    results_writer: ResultsWriter, graph_name: str, combo_id: str, task: str, threshold: float,
    train_metrics: dict, val_metrics: dict, test_metrics: dict, graph_stats: dict, split_manifest: SplitManifest,
) -> None:
    record = ExperimentRecord(
        experiment_id=f"{combo_id}_{graph_name}_task-{task}", experiment_kind="graph_comparison",
        config={"combo_id": combo_id, "graph": graph_name, "method": GRAPH_METHODS[graph_name], "threshold_used": threshold},
        task=task, split_manifest_seed=split_manifest.seed,
        metrics={"train": train_metrics, "val": val_metrics, "test": test_metrics},
        training_info={"graph_stats": graph_stats},
        runtime_seconds=graph_stats.get("runtime_seconds", 0.0), timestamp=datetime.now(timezone.utc).isoformat(),
    )
    results_writer.write(record)


def _run_one_graph_type(
    graph_name: str, combo_result, hospital_ids: list, hospital_id_of_patient: dict, labels_by_task: dict,
    train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray, graph_cfg: dict, gnn_cfg: dict,
    heads_cfg: dict, training_cfg: dict, tasks: list, logger: logging.Logger,
) -> dict:
    """Builds `graph_name`'s per-hospital graphs from `combo_result`'s
    embeddings and trains+evaluates a GraphOnlyLocalModel on each --
    pools every hospital's own val/test predictions into one global
    array per task (matching how the flat baseline's classifier metrics
    are ALSO one pooled, cross-hospital number, so Flat/Graph A/Graph B
    stay comparable). Returns `{task: {"train": metrics, "val":
    metrics, "test": metrics, "graph_stats": {...}}}`.
    """
    method = GRAPH_METHODS[graph_name]
    embeddings = combo_result.embeddings
    patient_ids = combo_result.patient_ids
    hospital_ids_by_row = np.array([hospital_id_of_patient[int(pid)] for pid in patient_ids])

    n_nodes_total = 0
    n_edges_total = 0
    per_task_probs: dict = {task: {"train": {"y": [], "p": []}, "val": {"y": [], "p": []}, "test": {"y": [], "p": []}} for task in tasks}

    for hid in hospital_ids:
        h_mask = hospital_ids_by_row == hid
        if h_mask.sum() < 2:
            continue  # too small to form any graph -- consistent with every other phase's empty/degenerate-hospital handling
        h_embeddings = embeddings[h_mask]
        h_patient_ids = patient_ids[h_mask]
        h_train_mask = train_mask[h_mask]
        h_val_mask = val_mask[h_mask]
        h_test_mask = test_mask[h_mask]

        graph = build_hospital_graph(
            h_patient_ids.tolist(), h_embeddings, method=method, k=graph_cfg["k"],
            cosine_threshold=graph_cfg["cosine_threshold"], n_clusters=graph_cfg.get("n_clusters", 10),
            cluster_seed=graph_cfg.get("cluster_seed", 42),
        )
        n_nodes_total += graph["n_nodes"]
        n_edges_total += graph["n_edges"]
        if not h_train_mask.any():
            continue

        for task in tasks:
            y_h = labels_by_task[task][h_mask]
            model_config = {
                "in_dim": h_embeddings.shape[1], "gnn_hidden_dim": gnn_cfg["hidden_dim"],
                "gnn_n_layers": gnn_cfg["layers"], "gnn_type": gnn_cfg["type"], "gnn_dropout": gnn_cfg["dropout"],
                "head_hidden_dim": heads_cfg["hidden_dim"], "head_dropout": heads_cfg["dropout"],
            }
            model, skip_reason, _loss_history = train_graph_model_one_hospital(
                h_embeddings, graph["edge_index"], graph["edge_weight"], h_train_mask,
                labels_by_task["mortality"][h_mask] if "mortality" in tasks else np.full(h_mask.sum(), np.nan),
                labels_by_task["sepsis"][h_mask] if "sepsis" in tasks else np.full(h_mask.sum(), np.nan),
                model_config, training_cfg, logger, seed=training_cfg.get("seed", 42),
            )
            if model is None:
                logger.warning(f"[GRAPH_COMPARISON] {graph_name} hospital={hid}: skipped ({skip_reason}) for task={task}")
                continue
            full_preds = predict_graph_model_full_hospital(model, h_embeddings, graph["edge_index"], graph["edge_weight"])
            prob = full_preds[f"{task}_prob"]
            for split_name, split_mask in (("train", h_train_mask), ("val", h_val_mask), ("test", h_test_mask)):
                known = split_mask & ~np.isnan(y_h)
                if known.any():
                    per_task_probs[task][split_name]["y"].append(y_h[known])
                    per_task_probs[task][split_name]["p"].append(prob[known])

    results: dict = {}
    for task in tasks:
        y_val = np.concatenate(per_task_probs[task]["val"]["y"]) if per_task_probs[task]["val"]["y"] else np.array([])
        p_val = np.concatenate(per_task_probs[task]["val"]["p"]) if per_task_probs[task]["val"]["p"] else np.array([])
        threshold = select_threshold_from_validation(y_val, p_val) if len(y_val) else 0.5
        split_metrics = {}
        for split_name in ("train", "val", "test"):
            y = np.concatenate(per_task_probs[task][split_name]["y"]) if per_task_probs[task][split_name]["y"] else np.array([])
            p = np.concatenate(per_task_probs[task][split_name]["p"]) if per_task_probs[task][split_name]["p"] else np.array([])
            split_metrics[split_name] = compute_binary_classification_metrics(y, p, threshold=threshold) if len(y) else {}
        results[task] = {
            **split_metrics, "threshold_used": threshold,
            "graph_stats": {"n_nodes": n_nodes_total, "n_edges": n_edges_total, "n_hospitals": len(hospital_ids)},
        }
    return results


def run_graph_comparison(
    cfg, ablation_cfg: dict, phase3_result: dict, phase4_result: dict, phase5_result: dict,
    partition_result, split_manifest: SplitManifest, results_writer: ResultsWriter,
    best_flat_config: "dict | None" = None,
) -> dict:
    """Single entry point: builds and evaluates Graph A AND Graph B
    together (never one alone), from the SAME embeddings/hospital
    boundaries/k/threshold/GNN settings, writing every result to
    `graph_comparison_results.csv`. `best_flat_config` (dict with at
    least `ts_model`/`tab_model`/`fusion_mode`) is REQUIRED -- either
    passed explicitly or read from `ablation.graph_comparison.
    best_flat_config` by the caller (`run_ablation.py`); this function
    itself never guesses one, since which combo won the real sampling
    comparison is a fact about real data this sandbox cannot produce.
    """
    logger = logging.getLogger("eicu_fl")
    if not best_flat_config or not all(k in best_flat_config for k in ("ts_model", "tab_model", "fusion_mode")):
        raise ValueError(
            "run_graph_comparison() requires best_flat_config={'ts_model':..., 'tab_model':..., 'fusion_mode':...} "
            "-- set ablation.graph_comparison.best_flat_config from the real sampling comparison's winning row "
            "before running this step. Never guessed automatically here."
        )
    if best_flat_config["tab_model"] == "pca":
        raise ValueError("best_flat_config uses tab_model='pca' -- PCA stays deferred; this step refuses to run against it.")

    graph_comparison_cfg = ablation_cfg.get("graph_comparison", {}) or {}
    graph_cfg = graph_comparison_cfg.get("graph", {"k": 5, "cosine_threshold": 0.6, "n_clusters": 10, "cluster_seed": 42})
    gnn_cfg = graph_comparison_cfg.get("gnn", {"type": "graphsage", "hidden_dim": 128, "layers": 2, "dropout": 0.2})
    heads_cfg = graph_comparison_cfg.get("heads", {"hidden_dim": 64, "dropout": 0.1})
    training_cfg = graph_comparison_cfg.get("training", {"epochs": 10, "learning_rate": 0.001, "seed": 42})

    combo = EmbeddingComboSpec(
        ts_model=best_flat_config["ts_model"], tab_model=best_flat_config["tab_model"],
        fusion_mode=best_flat_config["fusion_mode"],
    )

    sepsis_cohort = cfg.get("multitask.sepsis_cohort", "broad")
    local_training_cfg = ablation_cfg.get("local_training", {})
    max_hospitals = local_training_cfg.get("max_hospitals")
    sampling_cfg = ablation_cfg.get("sampling", {}) or {}
    cache_dir = sampling_cfg.get("embeddings_cache_dir", "outputs/ablation/embeddings_cache")

    ts_row_of_patient, tab_row_of_patient = _build_ts_tab_lookup(phase4_result, phase5_result)
    ts_array = phase4_result["normalized_array"]
    ts_seq_length = phase4_result["sequence_length"]
    ts_input_dim = len(phase4_result["feature_names"])
    tab_array = phase5_result["normalized_array"]
    tab_raw_input_dim = len(phase5_result["feature_names"])

    hospital_ids = sorted(partition_result.partitions.keys())
    if max_hospitals is not None:
        hospital_ids = hospital_ids[:max_hospitals]
    hospital_id_of_patient = _hospital_id_lookup(partition_result)

    combo_result, combo_summary = get_or_build_combo_embeddings(
        combo, hospital_ids, partition_result, ts_row_of_patient, tab_row_of_patient, ts_array, ts_seq_length,
        ts_input_dim, tab_array, tab_raw_input_dim, cfg, ablation_cfg, phase3_result, sepsis_cohort,
        local_training_cfg, split_manifest, cache_dir, logger,
    )
    if combo_result is None:
        raise ValueError(f"best_flat_config={best_flat_config!r} produced zero hospitals with embeddings -- cannot build any graph.")

    label_info = align_labels_to_patient_ids(
        combo_result.patient_ids.tolist(), phase3_result["cohort_patients"], phase3_result["mortality"],
        phase3_result["sepsis_df"], sepsis_cohort,
    )
    labels_by_task = {"mortality": np.asarray(label_info["mortality"]), "sepsis": np.asarray(label_info["sepsis"])}
    train_mask = combo_result.split_by_row == "train"
    val_mask = combo_result.split_by_row == "val"
    test_mask = combo_result.split_by_row == "test"

    tasks = (ablation_cfg.get("subset") or {}).get("tasks") or ["mortality", "sepsis"]
    comparison: dict = {"combo_id": combo.combo_id, "embeddings_from_cache": combo_summary.get("from_cache"), "graph_a": {}, "graph_b": {}}

    for graph_name in ("graph_a", "graph_b"):
        logger.info(f"[GRAPH_COMPARISON] building + evaluating {graph_name} ({GRAPH_METHODS[graph_name]}) for combo {combo.combo_id}")
        graph_results = _run_one_graph_type(
            graph_name, combo_result, hospital_ids, hospital_id_of_patient, labels_by_task, train_mask, val_mask,
            test_mask, graph_cfg, gnn_cfg, heads_cfg, training_cfg, tasks, logger,
        )
        comparison[graph_name] = graph_results
        for task, task_results in graph_results.items():
            _write_graph_comparison_record(
                results_writer, graph_name, combo.combo_id, task, task_results["threshold_used"],
                task_results.get("train", {}), task_results.get("val", {}), task_results.get("test", {}),
                task_results.get("graph_stats", {}), split_manifest,
            )

    logger.info(
        f"[GRAPH_COMPARISON] complete for combo {combo.combo_id}: graph_a nodes={comparison['graph_a'].get(tasks[0], {}).get('graph_stats', {}).get('n_nodes')} "
        f"edges={comparison['graph_a'].get(tasks[0], {}).get('graph_stats', {}).get('n_edges')}; "
        f"graph_b nodes={comparison['graph_b'].get(tasks[0], {}).get('graph_stats', {}).get('n_nodes')} "
        f"edges={comparison['graph_b'].get(tasks[0], {}).get('graph_stats', {}).get('n_edges')}"
    )
    return comparison
