"""
Ablation framework -- Section 2's graph-embedding classifier ablation.

Runs on the SINGLE, existing/locked default architecture
(ts_model=lstm, tab_model=autoencoder, fusion_mode=concatenation --
"the resulting graph embeddings... as defined by the existing project
architecture," per the original request's Section 2 and decision #7),
NOT the full 24-combo grid again -- decision #6 explicitly keeps GNN
type and federated algorithm out of the 72-config grid, and this is a
separate, smaller experiment: one representation source x
graph_classifiers x task = 6 experiments with the default lists.

Reuses, UNCHANGED:
  - `_train_one_hospital_flat_model()` / `_load_hospital_raw_tensors()`
    (src/ablation/embedding_ablation_runner.py) to produce the SAME
    kind of per-hospital, train-split-only, non-federated encoder
    training as Section 1 -- run here independently for exactly the
    default combo, rather than depending on Section 1 having already
    produced it, so this runner stays callable on its own (e.g. via
    `--start-phase`).
  - `src/graph/patient_graph.py`'s `build_hospital_graph()` (Phase 8,
    completely untouched) to build each hospital's intra-hospital graph
    from those fused embeddings. The graph is built over the hospital's
    FULL patient population (train+val+test) -- graph structure itself
    is unsupervised (cosine similarity of features), so building it
    over everyone is not a fitting/leakage operation.
  - `run_pipeline.py`'s `_build_train_only_subgraph()` (shared with
    Phase 11's own split-aware training, added/corrected 2026-09-11) to
    restrict the GRADIENT-COMPUTING forward/backward pass to a
    TRAIN-ONLY SUBGRAPH -- a non-train node is not even present in the
    tensors the model touches during training, so there is no pathway
    (message-passing or otherwise) for its raw features to influence a
    trained weight. A DESIGN CORRECTION, not the original approach: an
    earlier version of this function instead ran the FULL graph every
    epoch and masked only the LABELS to NaN (the standard semi-
    supervised/transductive regime) -- a dedicated leakage-sensitivity
    test (test_phase11_split_aware.py) proved that insufficient: a
    non-train node's corrupted FEATURES still changed a train
    neighbor's trained weights via a shared edge, even with its label
    masked. The train-only-subgraph approach closes this completely.
    AFTER training, one frozen (`eval()`, `no_grad()`) forward pass over
    the FULL graph produces embeddings for val/test patients -- this is
    read-only inference against already-fixed weights (the standard
    GraphSAGE inductive-learning use case: train on a subgraph, infer
    on an expanded graph including previously-unseen nodes).
  - `src/gnn/patient_gnn.py`'s `PatientGNN` and
    `src/heads/multitask_heads.py`'s `MultiTaskHeads` +
    `src/training/losses.py`'s `compute_multitask_loss()` (Phase
    9/10/11, completely untouched) for the GNN+heads training loop --
    one independent PatientGNN per hospital (decision #2's "locally
    trained, never FedAvg'd" applies here too, per decision #8's
    ordering: this runs BEFORE "THEN federated learning").
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import numpy as np
import torch

from src.ablation.config_grid import EmbeddingComboSpec, expand_graph_experiment_grid
from src.ablation.embedding_ablation_runner import (
    _build_ts_tab_lookup,
    _load_hospital_raw_tensors,
    _train_one_hospital_flat_model,
)
from src.ablation.results_writer import ExperimentRecord, ResultsWriter
from src.classifiers.boosting_classifiers import fit_and_evaluate_classifier
from src.data.splitting.patient_split import SplitManifest
from src.graph.patient_graph import build_hospital_graph
from src.gnn.patient_gnn import PatientGNN
from src.heads.multitask_heads import MultiTaskHeads, align_labels_to_patient_ids
from src.training.flat_multimodal_model import FlatModelConfig
from src.training.losses import compute_multitask_loss

DEFAULT_TS_MODEL = "lstm"
DEFAULT_TAB_MODEL = "autoencoder"
DEFAULT_FUSION_MODE = "concatenation"


def run_graph_embedding_ablation(
    cfg, ablation_cfg: dict, phase3_result: dict, phase4_result: dict, phase5_result: dict,
    partition_result, split_manifest: SplitManifest, results_writer: ResultsWriter, gnn_type: "str | None" = None,
) -> dict:
    logger = logging.getLogger("eicu_fl")
    sepsis_cohort = cfg.get("multitask.sepsis_cohort", "broad")
    local_training_cfg = ablation_cfg.get("local_training", {})
    max_hospitals = local_training_cfg.get("max_hospitals")
    gnn_type = gnn_type or cfg.get("gnn.type", "graphsage")

    ts_row_of_patient, tab_row_of_patient = _build_ts_tab_lookup(phase4_result, phase5_result)
    ts_array = phase4_result["normalized_array"]
    ts_seq_length = phase4_result["sequence_length"]
    ts_input_dim = len(phase4_result["feature_names"])
    tab_array = phase5_result["normalized_array"]
    tab_input_dim = len(phase5_result["feature_names"])

    hospital_ids = sorted(partition_result.partitions.keys())
    if max_hospitals is not None:
        hospital_ids = hospital_ids[:max_hospitals]

    combo = EmbeddingComboSpec(ts_model=DEFAULT_TS_MODEL, tab_model=DEFAULT_TAB_MODEL, fusion_mode=DEFAULT_FUSION_MODE)

    all_patient_ids: list = []
    all_embeddings: list = []
    all_split: list = []
    n_trained = n_skipped_empty = n_skipped_no_train = n_skipped_no_labels = 0
    n_skipped_non_finite = n_skipped_empty_graph = 0

    start = time.time()
    for i, hid in enumerate(hospital_ids):
        partition = partition_result.partitions[hid]
        hospital_patient_ids = sorted(int(pid) for pid in partition.patient_ids)
        if not hospital_patient_ids:
            n_skipped_empty += 1
            continue

        timeseries_x, timeseries_lengths, tabular_x_raw = _load_hospital_raw_tensors(
            hospital_patient_ids, ts_row_of_patient, ts_array, ts_seq_length, tab_row_of_patient, tab_array,
        )
        label_info = align_labels_to_patient_ids(
            hospital_patient_ids, phase3_result["cohort_patients"], phase3_result["mortality"],
            phase3_result["sepsis_df"], sepsis_cohort,
        )
        mortality_labels_full = torch.tensor(label_info["mortality"], dtype=torch.float32)
        sepsis_labels_full = torch.tensor(label_info["sepsis"], dtype=torch.float32)

        split_of_row = np.array([split_manifest.split_of(pid) or "train" for pid in hospital_patient_ids])
        train_row_mask = split_of_row == "train"

        # ---- Step 1: this hospital's own default (lstm+autoencoder+
        # concat) flat model, trained on train-split rows only -> fused
        # embeddings for graph construction. ----
        model_config = FlatModelConfig(
            ts_model=DEFAULT_TS_MODEL, tab_model=DEFAULT_TAB_MODEL, fusion_mode=DEFAULT_FUSION_MODE,
            ts_input_dim=ts_input_dim, tab_input_dim=tab_input_dim, ts_max_bins=timeseries_x.shape[1],
            ts_hidden_dim=cfg.get("modalities.timeseries.hidden_dim", 128),
            ts_num_layers=cfg.get("modalities.timeseries.num_layers", 2),
            ts_dropout=cfg.get("modalities.timeseries.dropout", 0.2),
            tab_hidden_dim=cfg.get("modalities.tabular.hidden_dim", 128),
            tab_embedding_dim=cfg.get("modalities.tabular.embedding_dim", 64),
            tab_dropout=cfg.get("modalities.tabular.dropout", 0.1),
            head_hidden_dim=cfg.get("multitask.head_hidden_dim", 64),
            head_dropout=cfg.get("multitask.head_dropout", 0.1),
        )
        tabular_x = torch.tensor(tabular_x_raw, dtype=torch.float32)
        flat_model, skip_reason, _loss_history = _train_one_hospital_flat_model(
            combo, timeseries_x, timeseries_lengths, tabular_x, train_row_mask,
            mortality_labels_full, sepsis_labels_full, model_config, local_training_cfg, logger, hid,
        )
        if skip_reason == "no_train_patients":
            n_skipped_no_train += 1
            continue
        if skip_reason == "no_labels":
            n_skipped_no_labels += 1
            continue
        if skip_reason == "non_finite_loss":
            n_skipped_non_finite += 1
            continue

        flat_model.eval()
        with torch.no_grad():
            fused, _recon = flat_model.encode(timeseries_x, timeseries_lengths, tabular_x)
        fused_np = fused.numpy()

        # ---- Step 2: this hospital's intra-hospital graph (Phase 8,
        # unchanged) from the fused embeddings, over ALL patients. ----
        graph = build_hospital_graph(
            hospital_patient_ids, fused_np,
            method=cfg.get("graph.method", "knn"), k=cfg.get("graph.k", 5),
            cosine_threshold=cfg.get("graph.cosine_threshold", 0.7),
            max_nodes_full_similarity=cfg.get("graph.max_nodes_full_similarity", 2000),
        )
        if graph["n_nodes"] == 0:
            n_skipped_empty_graph += 1
            continue
        edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
        edge_weight = torch.tensor(graph["edge_weight"], dtype=torch.float32)

        # ---- Step 3: this hospital's own GNN+heads, trained on a
        # TRAIN-ONLY SUBGRAPH (see run_pipeline.py's
        # `_build_train_only_subgraph()` docstring for the full
        # rationale and the real design correction that led here,
        # 2026-09-11): a non-train node is not even present in the
        # tensors the gradient-computing forward/backward pass touches,
        # so there is no message-passing (or any other) pathway for its
        # raw features to influence a trained weight -- not just its
        # label. This SUPERSEDES an earlier version of this function
        # that masked only the LABELS while still running the full
        # graph every epoch -- proven insufficient by a dedicated
        # leakage-sensitivity test in test_phase11_split_aware.py (a
        # non-train node's corrupted features still changed a train
        # neighbor's weights via a shared edge). ----
        from run_pipeline import _build_train_only_subgraph

        x_full = torch.tensor(fused_np, dtype=torch.float32)
        train_edge_index, train_edge_weight, train_indices = _build_train_only_subgraph(
            edge_index, edge_weight, torch.tensor(train_row_mask),
        )
        x_train = x_full[train_indices]
        mort_train = mortality_labels_full[train_indices]
        sepsis_train = sepsis_labels_full[train_indices]

        gnn = PatientGNN(
            in_dim=fused_np.shape[1], hidden_dim=cfg.get("gnn.hidden_dim", 128),
            n_layers=cfg.get("gnn.n_layers", 2), gnn_type=gnn_type, dropout=cfg.get("gnn.dropout", 0.2),
        )
        heads = MultiTaskHeads(
            cfg.get("gnn.hidden_dim", 128), head_hidden_dim=cfg.get("multitask.head_hidden_dim", 64),
            dropout=cfg.get("multitask.head_dropout", 0.1),
        )
        optimizer = torch.optim.Adam(
            list(gnn.parameters()) + list(heads.parameters()),
            lr=local_training_cfg.get("learning_rate", 0.001), weight_decay=local_training_cfg.get("weight_decay", 0.0001),
        )

        gnn_skip_reason = None
        for _epoch in range(local_training_cfg.get("epochs", 5)):
            optimizer.zero_grad()
            node_embeddings_train = gnn(x_train, train_edge_index, train_edge_weight)
            head_out = heads(node_embeddings_train)
            total_loss, _components, _counts = compute_multitask_loss(
                head_out["mortality_logits"], head_out["sepsis_logits"], mort_train, sepsis_train,
                local_training_cfg.get("mortality_weight", 1.0), local_training_cfg.get("sepsis_weight", 1.0),
                local_training_cfg.get("loss_type", "weighted_bce"),
            )
            if total_loss is None:
                gnn_skip_reason = "no_labels"
                break
            if not torch.isfinite(total_loss):
                gnn_skip_reason = "non_finite_loss"
                break
            total_loss.backward()
            optimizer.step()

        if gnn_skip_reason == "no_labels":
            n_skipped_no_labels += 1
            continue
        if gnn_skip_reason == "non_finite_loss":
            n_skipped_non_finite += 1
            continue

        # Frozen inference over the FULL graph (every patient,
        # train+val+test) -- val/test patients only ever "pass through
        # frozen components" here, exactly as required.
        gnn.eval()
        heads.eval()
        with torch.no_grad():
            node_embeddings = gnn(x_full, edge_index, edge_weight)

        all_patient_ids.extend(hospital_patient_ids)
        all_embeddings.append(node_embeddings.numpy())
        all_split.extend(split_of_row.tolist())
        n_trained += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(hospital_ids):
            logger.info(f"[ABLATION_GRAPH] progress {i + 1}/{len(hospital_ids)} hospitals")

    training_info_summary = {
        "gnn_type": gnn_type, "n_hospitals_total": len(hospital_ids), "n_hospitals_trained": n_trained,
        "n_hospitals_skipped_empty": n_skipped_empty, "n_hospitals_skipped_no_train_patients": n_skipped_no_train,
        "n_hospitals_skipped_no_labels": n_skipped_no_labels, "n_hospitals_skipped_non_finite": n_skipped_non_finite,
        "n_hospitals_skipped_empty_graph": n_skipped_empty_graph, "runtime_seconds": time.time() - start,
    }

    if not all_embeddings:
        logger.warning("[ABLATION_GRAPH] zero hospitals produced embeddings -- skipping classifiers entirely.")
        return {"skipped_entirely": True, "training_info": training_info_summary}

    combo_embeddings = np.concatenate(all_embeddings, axis=0)
    combo_patient_ids = np.array(all_patient_ids)
    combo_split = np.array(all_split)

    label_info = align_labels_to_patient_ids(
        combo_patient_ids.tolist(), phase3_result["cohort_patients"], phase3_result["mortality"],
        phase3_result["sepsis_df"], sepsis_cohort,
    )
    labels_by_task = {"mortality": np.asarray(label_info["mortality"]), "sepsis": np.asarray(label_info["sepsis"])}

    train_mask = combo_split == "train"
    val_mask = combo_split == "val"
    test_mask = combo_split == "test"

    experiments = expand_graph_experiment_grid(ablation_cfg)
    classifier_params_cfg = ablation_cfg.get("classifier_params", {})
    for spec in experiments:
        params = classifier_params_cfg.get(spec.classifier)
        y = labels_by_task[spec.task]
        result = fit_and_evaluate_classifier(
            spec.classifier, params, combo_embeddings[train_mask], y[train_mask],
            combo_embeddings[val_mask], y[val_mask], combo_embeddings[test_mask], y[test_mask],
            task=spec.task, logger=logger,
        )
        record = ExperimentRecord(
            experiment_id=spec.experiment_id, experiment_kind="graph_ablation",
            config={"gnn_type": gnn_type, "classifier": spec.classifier, "classifier_params": params or {}},
            task=spec.task, split_manifest_seed=split_manifest.seed,
            metrics={"train": result.train_metrics, "val": result.val_metrics, "test": result.test_metrics},
            training_info=dict(
                training_info_summary, classifier_fit_runtime_seconds=result.fit_runtime_seconds,
                classifier_skipped=result.skipped, classifier_skip_reason=result.skip_reason,
            ),
            runtime_seconds=training_info_summary["runtime_seconds"] + result.fit_runtime_seconds,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results_writer.write(record)

    return {"n_experiments": len(experiments), "training_info": training_info_summary}
