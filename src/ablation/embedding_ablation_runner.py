"""
Ablation framework -- Section 1's TS x TAB x Fusion x Classifier
ablation orchestrator.

Execution order (decision #8, confirmed with the user 2026-09-10):
split -> TS/TAB embedding generation -> fusion -> THIS 72-config
frozen-embedding classifier ablation -> (separately, afterwards) graph
construction/GNN -> graph-embedding classifier experiments -> THEN
federated learning. This module owns exactly the middle step: for
every (ts_model, tab_model, fusion_mode) combo, it trains a fresh
src.training.flat_multimodal_model.FlatMultimodalModel PER HOSPITAL,
independently (decision #2: "locally trained representations," never
FedAvg'd/FedProto'd, never pooled across hospitals), using ONLY that
hospital's TRAIN-split patients (per the permanent split manifest --
src/data/splitting/patient_split.py), then extracts the FROZEN fused
embedding for every one of that hospital's patients (train+val+test
alike) using the now-fixed trained weights. Once every hospital's
embeddings are collected into one combo-level matrix, AdaBoost/XGBoost/
LightGBM (src/classifiers/boosting_classifiers.py) are fit on the TRAIN
partition and evaluated on train/val/test, per task, per decision #1 --
completely separately from the encoder training above, never inside
its gradient computation.

Mirrors `_run_local_training_round()`'s per-hospital training
discipline (run_pipeline.py, Phase 11) wherever it applies unchanged --
same masked multitask loss (src/training/losses.py, reused as-is), same
NaN/Inf-safety skip, same no-usable-labels skip, same progress logging
every 20 hospitals -- but is a NEW, separate function rather than a
modification of that one, because the two differ in three structural
ways that don't fit as optional parameters without obscuring both: (1)
no graph/edge_index at all, (2) training is restricted to a per-
hospital TRAIN-split row mask instead of every patient in the batch,
and (3) there is no shared initial state_dict to load (decision #2 --
each hospital's flat model starts from its own fresh, independently-
seeded init; there is no FedAvg round to make that requirement matter
here).
"""

from __future__ import annotations

import logging
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.ablation.config_grid import EmbeddingComboSpec, expand_embedding_combo_grid
from src.ablation.results_writer import ExperimentRecord, ResultsWriter
from src.classifiers.boosting_classifiers import fit_and_evaluate_classifier
from src.data.splitting.patient_split import SplitManifest
from src.heads.multitask_heads import align_labels_to_patient_ids
from src.modalities.tabular.pca_encoder import fit_transform_per_hospital_pca
from src.training.flat_multimodal_model import FlatModelConfig, FlatMultimodalModel
from src.training.losses import compute_multitask_loss


@dataclass
class ComboEmbeddingResult:
    combo: EmbeddingComboSpec
    patient_ids: np.ndarray               # (n_patients,) global, across every hospital
    embeddings: np.ndarray                 # (n_patients, fused_dim)
    split_by_row: np.ndarray               # (n_patients,) of "train"/"val"/"test", same order as patient_ids
    n_hospitals_total: int = 0
    n_hospitals_trained: int = 0
    n_hospitals_skipped_empty: int = 0
    n_hospitals_skipped_no_train_patients: int = 0
    n_hospitals_skipped_no_labels: int = 0
    n_hospitals_skipped_non_finite: int = 0
    n_hospitals_pca_too_small: int = 0
    train_runtime_seconds: float = 0.0


def _build_ts_tab_lookup(phase4_result: dict, phase5_result: dict):
    ts_patient_ids = [int(pid) for pid in phase4_result["patient_ids"]]
    ts_row_of_patient = {pid: i for i, pid in enumerate(ts_patient_ids)}
    tab_patient_ids = [int(pid) for pid in phase5_result["patient_ids"]]
    tab_row_of_patient = {pid: i for i, pid in enumerate(tab_patient_ids)}
    return ts_row_of_patient, tab_row_of_patient


def _load_hospital_raw_tensors(
    hospital_patient_ids: list, ts_row_of_patient: dict, ts_array, ts_seq_length,
    tab_row_of_patient: dict, tab_array,
):
    """Graph-free equivalent of run_pipeline.py's `_load_hospital_batch()`
    -- slices Phase 4/5's raw, already-normalized tensors by patient ID
    for exactly one hospital's patients (Phase 2's own partition list,
    NOT Phase 8's graph node list -- this runs BEFORE graph construction
    even happens, per decision #8's execution order)."""
    missing_ts = [pid for pid in hospital_patient_ids if pid not in ts_row_of_patient]
    missing_tab = [pid for pid in hospital_patient_ids if pid not in tab_row_of_patient]
    if missing_ts or missing_tab:
        raise RuntimeError(
            f"patient(s) present in Phase 2's partition but absent from Phase 4/5's raw tensors "
            f"(timeseries missing={missing_ts[:5]}, tabular missing={missing_tab[:5]}) -- Phase "
            f"2/3/4/5 are expected to cover the exact same cohort; investigate before proceeding."
        )
    timeseries_x = torch.tensor(ts_array[[ts_row_of_patient[pid] for pid in hospital_patient_ids]], dtype=torch.float32)
    timeseries_lengths = torch.tensor(ts_seq_length[[ts_row_of_patient[pid] for pid in hospital_patient_ids]], dtype=torch.long)
    tabular_x_raw = tab_array[[tab_row_of_patient[pid] for pid in hospital_patient_ids]]
    return timeseries_x, timeseries_lengths, tabular_x_raw


def _hospital_model_seed(base_seed: int, combo_id: str, hid) -> int:
    """Deterministic per-(combo, hospital) seed derived from
    `run.seed` -- added 2026-09-20 while building the sampling
    comparison (see `_train_combo_and_extract_embeddings()`'s
    docstring for why this was needed). Deliberately NOT Python's
    built-in `hash()`: string hashing is randomized per-process by
    default (`PYTHONHASHSEED`), which would silently reintroduce
    non-determinism here. `zlib.crc32` is stable across processes and
    interpreter versions."""
    combo_component = zlib.crc32(combo_id.encode("utf-8"))
    return (int(base_seed) * 1_000_003 + combo_component + int(hid)) % (2**31 - 1)


def _train_one_hospital_flat_model(
    combo: EmbeddingComboSpec, timeseries_x, timeseries_lengths, tabular_x, train_row_mask: np.ndarray,
    mortality_labels, sepsis_labels, model_config: FlatModelConfig, local_training_cfg: dict,
    logger: logging.Logger, hid, seed: "int | None" = None,
) -> "tuple[FlatMultimodalModel | None, str | None, list]":
    """Trains ONE hospital's own, independently-initialized
    FlatMultimodalModel using ONLY the rows where train_row_mask is
    True. Returns (model_or_None, skip_reason_or_None, loss_history).
    Never raises for a routine skip condition (too few train patients,
    no usable labels, non-finite loss) -- those are reported via
    skip_reason so the caller can count and log them, exactly like
    run_pipeline.py's `_run_local_training_round()` does for Phase 11.

    `seed`, added 2026-09-20 (backward-compatible: defaults to `None`,
    meaning "don't explicitly seed," preserving every existing caller's
    exact prior behavior -- `graph_embedding_ablation_runner.py`'s own
    call site does not pass it and is therefore unaffected): when
    given, calls `torch.manual_seed(seed)` immediately before
    constructing the model, so the SAME (combo, hospital) reproduces
    the SAME random initial weights -- and therefore the same trained
    embeddings -- across separate process invocations. See
    `_train_combo_and_extract_embeddings()`'s docstring for why this
    matters for the sampling comparison specifically.
    """
    n_train = int(train_row_mask.sum())
    if n_train < 1:
        return None, "no_train_patients", []

    if seed is not None:
        torch.manual_seed(seed)
    model = FlatMultimodalModel(model_config)
    model.train()
    lr = local_training_cfg.get("learning_rate", 0.001)
    wd = local_training_cfg.get("weight_decay", 0.0001)
    optimizer_name = local_training_cfg.get("optimizer", "adam")
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd) if optimizer_name == "adam"
        else torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    )
    epochs = local_training_cfg.get("epochs", 5)
    reconstruction_weight = local_training_cfg.get("reconstruction_weight", 0.5) if combo.tab_model == "autoencoder" else 0.0
    mortality_weight = local_training_cfg.get("mortality_weight", 1.0)
    sepsis_weight = local_training_cfg.get("sepsis_weight", 1.0)
    loss_type = local_training_cfg.get("loss_type", "weighted_bce")

    train_idx = np.where(train_row_mask)[0]
    ts_train = timeseries_x[train_idx]
    len_train = timeseries_lengths[train_idx]
    tab_train = tabular_x[train_idx]
    mort_train = mortality_labels[train_idx]
    sepsis_train = sepsis_labels[train_idx]

    loss_history: list = []
    for _epoch in range(epochs):
        optimizer.zero_grad()
        out = model(ts_train, len_train, tab_train)
        total_loss, _components, _counts = compute_multitask_loss(
            out["mortality_logits"], out["sepsis_logits"], mort_train, sepsis_train,
            mortality_weight, sepsis_weight, loss_type,
            reconstruction=out["tabular_reconstruction"], tabular_target=tab_train,
            reconstruction_weight=reconstruction_weight,
        )
        if total_loss is None:
            return None, "no_labels", loss_history
        if not torch.isfinite(total_loss):
            logger.warning(f"[ABLATION_FLAT_TRAIN] hospital {hid} ({combo.combo_id}): non-finite loss at epoch {len(loss_history)}.")
            return None, "non_finite_loss", loss_history
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())

    return model, None, loss_history


def _train_combo_and_extract_embeddings(
    combo: EmbeddingComboSpec, hospital_ids: list, partition_result, ts_row_of_patient: dict, tab_row_of_patient: dict,
    ts_array, ts_seq_length, ts_input_dim: int, tab_array, tab_raw_input_dim: int,
    cfg, ablation_cfg: dict, phase3_result: dict, sepsis_cohort: str, local_training_cfg: dict,
    split_manifest: SplitManifest, logger: logging.Logger,
) -> "tuple[ComboEmbeddingResult | None, dict]":
    """Trains ONE combo's per-hospital encoders (TRAIN-split only, per
    decision #2) and extracts frozen fused embeddings for every one of
    that combo's patients -- the exact per-combo body
    `run_embedding_ablation()` itself runs, extracted 2026-09-20 so the
    sampling-comparison runner (`src/ablation/sampling_comparison_
    runner.py`) can regenerate the SAME embeddings the original 48-
    config ablation used, via this SAME code path, rather than a
    second, potentially-drifting reimplementation. Returns
    `(combo_result_or_None, combo_summary_dict)` -- `None` for
    `combo_result` when zero hospitals produced embeddings (mirrors the
    original inline "skipped_entirely" handling exactly, byte-for-byte
    the same summary shape).

    REAL FINDING, fixed 2026-09-20 while building the sampling
    comparison: this function's per-hospital model construction was
    never explicitly seeded, so two separate process invocations of
    the SAME combo produced DIFFERENT random initial weights -- and
    therefore different trained embeddings and different classifier
    metrics -- discovered directly via a reproduction check that
    failed unexpectedly (regenerated test AUROC didn't match the
    original master_results.csv row) rather than assumed. Fixed by
    deriving a deterministic per-(combo, hospital) seed from
    `run.seed` (`_hospital_model_seed()`) and threading it through to
    `_train_one_hospital_flat_model()`. This makes every FUTURE call
    reproducible; it does NOT and cannot retroactively make the
    ALREADY-COMPLETED real 48-config ablation's historical
    master_results.csv rows re-derivable from a fresh run, since those
    were produced before this fix existed (that data remains valid and
    untouched either way -- only future regenerations are affected).
    """
    combo_start = time.time()
    base_seed = cfg.get("run.seed", 42)
    all_patient_ids: list = []
    all_embeddings: list = []
    all_split: list = []
    n_trained = n_skipped_empty = n_skipped_no_train = n_skipped_no_labels = n_skipped_non_finite = n_pca_too_small = 0

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
        mortality_labels = torch.tensor(label_info["mortality"], dtype=torch.float32)
        sepsis_labels = torch.tensor(label_info["sepsis"], dtype=torch.float32)

        split_of_row = np.array([split_manifest.split_of(pid) or "train" for pid in hospital_patient_ids])
        train_row_mask = split_of_row == "train"

        if combo.tab_model == "pca":
            pca_n_components = ablation_cfg.get("pca", {}).get("n_components", 32)
            train_idx_arr = np.where(train_row_mask)[0]
            pca_embeddings, skipped = fit_transform_per_hospital_pca(
                {hid: tabular_x_raw}, {hid: train_idx_arr}, embedding_dim=pca_n_components,
            )
            if hid in skipped:
                n_pca_too_small += 1
            tabular_x = torch.tensor(pca_embeddings[hid], dtype=torch.float32)
            tab_input_dim = pca_n_components
        else:
            tabular_x = torch.tensor(tabular_x_raw, dtype=torch.float32)
            tab_input_dim = tab_raw_input_dim

        model_config = FlatModelConfig(
            ts_model=combo.ts_model, tab_model=combo.tab_model, fusion_mode=combo.fusion_mode,
            ts_input_dim=ts_input_dim, tab_input_dim=tab_input_dim, ts_max_bins=timeseries_x.shape[1],
            ts_hidden_dim=cfg.get("modalities.timeseries.hidden_dim", 128),
            ts_num_layers=cfg.get("modalities.timeseries.num_layers", 2),
            ts_dropout=cfg.get("modalities.timeseries.dropout", 0.2),
            ts_transformer_num_heads=cfg.get("modalities.timeseries.transformer_num_heads", 4),
            ts_transformer_dim_feedforward=cfg.get("modalities.timeseries.transformer_dim_feedforward", 256),
            tab_hidden_dim=cfg.get("modalities.tabular.hidden_dim", 128),
            tab_embedding_dim=cfg.get("modalities.tabular.embedding_dim", 64),
            tab_dropout=cfg.get("modalities.tabular.dropout", 0.1),
            cross_attention_attn_dim=ablation_cfg.get("cross_attention", {}).get("attn_dim", 128),
            cross_attention_num_heads=ablation_cfg.get("cross_attention", {}).get("num_heads", 4),
            head_hidden_dim=cfg.get("multitask.head_hidden_dim", 64),
            head_dropout=cfg.get("multitask.head_dropout", 0.1),
        )

        model, skip_reason, _loss_history = _train_one_hospital_flat_model(
            combo, timeseries_x, timeseries_lengths, tabular_x, train_row_mask,
            mortality_labels, sepsis_labels, model_config, local_training_cfg, logger, hid,
            seed=_hospital_model_seed(base_seed, combo.combo_id, hid),
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

        model.eval()
        with torch.no_grad():
            fused, _recon = model.encode(timeseries_x, timeseries_lengths, tabular_x)
        all_patient_ids.extend(hospital_patient_ids)
        all_embeddings.append(fused.numpy())
        all_split.extend(split_of_row.tolist())
        n_trained += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(hospital_ids):
            logger.info(f"[ABLATION_EMBEDDING] {combo.combo_id}: progress {i + 1}/{len(hospital_ids)} hospitals")

    if not all_embeddings:
        logger.warning(f"[ABLATION_EMBEDDING] combo {combo.combo_id}: zero hospitals produced embeddings -- skipping classifiers entirely for this combo.")
        return None, {"n_hospitals_trained": 0, "skipped_entirely": True}

    combo_embeddings = np.concatenate(all_embeddings, axis=0)
    combo_patient_ids = np.array(all_patient_ids)
    combo_split = np.array(all_split)
    combo_result = ComboEmbeddingResult(
        combo=combo, patient_ids=combo_patient_ids, embeddings=combo_embeddings, split_by_row=combo_split,
        n_hospitals_total=len(hospital_ids), n_hospitals_trained=n_trained, n_hospitals_skipped_empty=n_skipped_empty,
        n_hospitals_skipped_no_train_patients=n_skipped_no_train, n_hospitals_skipped_no_labels=n_skipped_no_labels,
        n_hospitals_skipped_non_finite=n_skipped_non_finite, n_hospitals_pca_too_small=n_pca_too_small,
        train_runtime_seconds=time.time() - combo_start,
    )
    combo_summary = {
        "n_hospitals_trained": n_trained, "n_hospitals_skipped_empty": n_skipped_empty,
        "n_hospitals_skipped_no_train_patients": n_skipped_no_train, "n_hospitals_skipped_no_labels": n_skipped_no_labels,
        "n_hospitals_skipped_non_finite": n_skipped_non_finite, "n_hospitals_pca_too_small": n_pca_too_small,
        "runtime_seconds": combo_result.train_runtime_seconds, "n_patients_embedded": len(combo_patient_ids),
    }
    return combo_result, combo_summary


def _combo_cache_path(cache_dir, combo_id: str) -> Path:
    return Path(cache_dir) / f"{combo_id}.npz"


def _save_combo_embeddings_to_cache(combo_result: ComboEmbeddingResult, cache_dir, logger: logging.Logger) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = _combo_cache_path(cache_dir, combo_result.combo.combo_id)
    np.savez(
        path,
        patient_ids=combo_result.patient_ids, embeddings=combo_result.embeddings,
        split_by_row=combo_result.split_by_row,
        n_hospitals_total=combo_result.n_hospitals_total, n_hospitals_trained=combo_result.n_hospitals_trained,
        n_hospitals_skipped_empty=combo_result.n_hospitals_skipped_empty,
        n_hospitals_skipped_no_train_patients=combo_result.n_hospitals_skipped_no_train_patients,
        n_hospitals_skipped_no_labels=combo_result.n_hospitals_skipped_no_labels,
        n_hospitals_skipped_non_finite=combo_result.n_hospitals_skipped_non_finite,
        n_hospitals_pca_too_small=combo_result.n_hospitals_pca_too_small,
        train_runtime_seconds=combo_result.train_runtime_seconds,
    )
    logger.info(f"[EMBEDDING_CACHE] cached embeddings for {combo_result.combo.combo_id} -> {path}")


def _load_combo_embeddings_from_cache(combo, cache_dir, logger: logging.Logger) -> "ComboEmbeddingResult | None":
    path = _combo_cache_path(cache_dir, combo.combo_id)
    if not path.exists():
        return None
    data = np.load(path)
    logger.info(
        f"[EMBEDDING_CACHE] {combo.combo_id}: loaded cached embeddings from {path} "
        f"(n_patients={len(data['patient_ids'])}) -- skipping encoder re-training."
    )
    return ComboEmbeddingResult(
        combo=combo, patient_ids=data["patient_ids"], embeddings=data["embeddings"], split_by_row=data["split_by_row"],
        n_hospitals_total=int(data["n_hospitals_total"]), n_hospitals_trained=int(data["n_hospitals_trained"]),
        n_hospitals_skipped_empty=int(data["n_hospitals_skipped_empty"]),
        n_hospitals_skipped_no_train_patients=int(data["n_hospitals_skipped_no_train_patients"]),
        n_hospitals_skipped_no_labels=int(data["n_hospitals_skipped_no_labels"]),
        n_hospitals_skipped_non_finite=int(data["n_hospitals_skipped_non_finite"]),
        n_hospitals_pca_too_small=int(data["n_hospitals_pca_too_small"]),
        train_runtime_seconds=float(data["train_runtime_seconds"]),
    )


def get_or_build_combo_embeddings(
    combo, hospital_ids, partition_result, ts_row_of_patient, tab_row_of_patient, ts_array, ts_seq_length,
    ts_input_dim, tab_array, tab_raw_input_dim, cfg, ablation_cfg, phase3_result, sepsis_cohort,
    local_training_cfg, split_manifest: SplitManifest, cache_dir, logger: logging.Logger,
) -> "tuple[ComboEmbeddingResult | None, dict]":
    """Cache-first wrapper around `_train_combo_and_extract_embeddings()`.

    Added 2026-09-29 so `run_embedding_ablation()` (this module's own
    orchestrator, i.e. "the flat ablation") shares ONE on-disk embedding
    cache with `sampling_comparison_runner.py` and
    `graph_comparison_runner.py`, instead of each of the three re-
    training the exact same (ts_model, tab_model, fusion_mode) combo's
    per-hospital encoders from scratch independently. Previously only
    `sampling_comparison_runner.py` defined this cache-first logic
    (as a private `_get_or_build_combo_embeddings()`); it is now the
    single shared implementation all three import, so a combo trained
    by ANY of the three (in the same or a separate process/batch
    invocation) is reused by the other two rather than recomputed.
    Safe specifically because `_train_combo_and_extract_embeddings()`
    seeds every hospital's model deterministically from
    `run.seed`/`combo_id`/`hospital_id` (`_hospital_model_seed()`) --
    a cache hit is therefore bit-for-bit equivalent to a fresh train,
    not an approximation.

    NOTE: the cache has no content-based invalidation -- it is keyed
    only by `combo.combo_id` (the ts/tab/fusion identity), not by a
    hash of the encoder code itself. If `FlatMultimodalModel` or its
    training loop changes, any previously-cached combo must be deleted
    from `cache_dir` by hand before re-running, or the stale embeddings
    will be silently reused. This was already true for
    `sampling_comparison`/`graph_comparison`'s use of this cache before
    this change; it now also applies to `embedding_ablation`.
    """
    cached = _load_combo_embeddings_from_cache(combo, cache_dir, logger)
    if cached is not None:
        summary = {
            "n_hospitals_trained": cached.n_hospitals_trained, "n_patients_embedded": len(cached.patient_ids),
            "from_cache": True,
        }
        return cached, summary

    combo_result, combo_summary = _train_combo_and_extract_embeddings(
        combo, hospital_ids, partition_result, ts_row_of_patient, tab_row_of_patient, ts_array, ts_seq_length,
        ts_input_dim, tab_array, tab_raw_input_dim, cfg, ablation_cfg, phase3_result, sepsis_cohort,
        local_training_cfg, split_manifest, logger,
    )
    if combo_result is not None:
        _save_combo_embeddings_to_cache(combo_result, cache_dir, logger)
    combo_summary = dict(combo_summary)
    combo_summary["from_cache"] = False
    return combo_result, combo_summary


def run_embedding_ablation(
    cfg, ablation_cfg: dict, phase3_result: dict, phase4_result: dict, phase5_result: dict,
    partition_result, split_manifest: SplitManifest, results_writer: ResultsWriter,
) -> dict:
    """Runs every (ts_model, tab_model, fusion_mode) combo in the grid,
    trains it per-hospital (train-split only), extracts frozen
    embeddings for everyone, fits every configured classifier for every
    task, and writes one ExperimentRecord per (combo, classifier, task)
    via `results_writer`. Returns a run-level summary dict.

    Thin orchestration loop only, as of 2026-09-20 -- the actual
    per-combo encoder training/embedding extraction lives in
    `_train_combo_and_extract_embeddings()` (extracted so
    `sampling_comparison_runner.py` can reuse the exact same code path
    rather than reimplementing it); behavior here is unchanged from
    before the extraction, verified by re-running this module's
    existing test suite unmodified.
    """
    logger = logging.getLogger("eicu_fl")
    sepsis_cohort = cfg.get("multitask.sepsis_cohort", "broad")
    local_training_cfg = ablation_cfg.get("local_training", {})
    max_hospitals = local_training_cfg.get("max_hospitals")

    ts_row_of_patient, tab_row_of_patient = _build_ts_tab_lookup(phase4_result, phase5_result)
    ts_array = phase4_result["normalized_array"]
    ts_seq_length = phase4_result["sequence_length"]
    ts_input_dim = len(phase4_result["feature_names"])
    tab_array = phase5_result["normalized_array"]
    tab_raw_input_dim = len(phase5_result["feature_names"])

    hospital_ids = sorted(partition_result.partitions.keys())
    if max_hospitals is not None:
        hospital_ids = hospital_ids[:max_hospitals]

    # Shared with sampling_comparison_runner.py/graph_comparison_runner.py
    # (same config key, deliberately -- see get_or_build_combo_embeddings()'s
    # docstring): running this step per-ts-model batch (e.g.
    # `--config configs/full_batch_lstm.yaml`) now populates the SAME
    # on-disk cache those other two steps read, and vice versa.
    cache_dir = ablation_cfg.get("sampling", {}).get("embeddings_cache_dir", "outputs/ablation/embeddings_cache")

    combos = expand_embedding_combo_grid(ablation_cfg)
    combo_summaries: dict = {}

    for combo_i, combo in enumerate(combos):
        logger.info(f"[ABLATION_EMBEDDING] ({combo_i + 1}/{len(combos)}) starting combo {combo.combo_id}")
        combo_result, combo_summary = get_or_build_combo_embeddings(
            combo, hospital_ids, partition_result, ts_row_of_patient, tab_row_of_patient, ts_array, ts_seq_length,
            ts_input_dim, tab_array, tab_raw_input_dim, cfg, ablation_cfg, phase3_result, sepsis_cohort,
            local_training_cfg, split_manifest, cache_dir, logger,
        )
        combo_summaries[combo.combo_id] = combo_summary
        if combo_result is None:
            continue

        _fit_and_record_classifiers_for_combo(
            combo_result, ablation_cfg, split_manifest, results_writer, logger, phase3_result, sepsis_cohort,
        )
        cache_note = "from cache" if combo_summary.get("from_cache") else f"trained in {combo_result.train_runtime_seconds:.1f}s"
        logger.info(
            f"[ABLATION_EMBEDDING] combo {combo.combo_id} done ({cache_note}, "
            f"{combo_result.n_hospitals_trained}/{len(hospital_ids)} hospitals trained)"
        )

    return {"n_combos": len(combos), "combos": combo_summaries}


def _fit_and_record_classifiers_for_combo(
    combo_result: ComboEmbeddingResult, ablation_cfg: dict, split_manifest: SplitManifest,
    results_writer: ResultsWriter, logger: logging.Logger, phase3_result: dict, sepsis_cohort: str,
) -> None:
    # Labels for the FULL global patient list this combo produced
    # embeddings for, aligned by ID exactly ONCE here (reused across
    # every classifier/task combo below) -- reuses Phase 10's own
    # ID-based join rather than re-deriving it.
    label_info = align_labels_to_patient_ids(
        combo_result.patient_ids.tolist(), phase3_result["cohort_patients"], phase3_result["mortality"],
        phase3_result["sepsis_df"], sepsis_cohort,
    )
    labels_by_task = {"mortality": np.asarray(label_info["mortality"]), "sepsis": np.asarray(label_info["sepsis"])}

    train_mask = combo_result.split_by_row == "train"
    val_mask = combo_result.split_by_row == "val"
    test_mask = combo_result.split_by_row == "test"

    subset = ablation_cfg.get("subset") or {}
    classifiers = subset.get("classifiers") or ablation_cfg.get("classifiers", [])
    tasks = subset.get("tasks") or ["mortality", "sepsis"]
    classifier_params = ablation_cfg.get("classifier_params", {})

    for task in tasks:
        y = labels_by_task[task]
        for classifier_name in classifiers:
            params = classifier_params.get(classifier_name)
            result = fit_and_evaluate_classifier(
                classifier_name, params,
                combo_result.embeddings[train_mask], y[train_mask],
                combo_result.embeddings[val_mask], y[val_mask],
                combo_result.embeddings[test_mask], y[test_mask],
                task=task, logger=logger,
            )
            record = ExperimentRecord(
                experiment_id=f"{combo_result.combo.combo_id}_clf-{classifier_name}_task-{task}",
                experiment_kind="embedding_ablation",
                config={
                    "ts_model": combo_result.combo.ts_model, "tab_model": combo_result.combo.tab_model,
                    "fusion_mode": combo_result.combo.fusion_mode, "classifier": classifier_name,
                    "classifier_params": params or {},
                },
                task=task, split_manifest_seed=split_manifest.seed,
                metrics={"train": result.train_metrics, "val": result.val_metrics, "test": result.test_metrics},
                training_info={
                    "n_hospitals_trained": combo_result.n_hospitals_trained,
                    "n_hospitals_total": combo_result.n_hospitals_total,
                    "n_hospitals_skipped_empty": combo_result.n_hospitals_skipped_empty,
                    "n_hospitals_skipped_no_train_patients": combo_result.n_hospitals_skipped_no_train_patients,
                    "n_hospitals_skipped_no_labels": combo_result.n_hospitals_skipped_no_labels,
                    "n_hospitals_skipped_non_finite": combo_result.n_hospitals_skipped_non_finite,
                    "n_hospitals_pca_too_small": combo_result.n_hospitals_pca_too_small,
                    "encoder_train_runtime_seconds": combo_result.train_runtime_seconds,
                    "classifier_fit_runtime_seconds": result.fit_runtime_seconds,
                    "classifier_skipped": result.skipped, "classifier_skip_reason": result.skip_reason,
                },
                runtime_seconds=combo_result.train_runtime_seconds + result.fit_runtime_seconds,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            results_writer.write(record)
