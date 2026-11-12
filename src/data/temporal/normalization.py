"""
Phase 4c -- Missing-value handling + normalization.

Fits normalization statistics PER HOSPITAL, not globally. This directly
matches the architecture's "Local Hospital Preprocessing" step and the
federated privacy boundary: a hospital's feature
distribution is itself a property of that hospital's patient population,
and standardizing with global (cross-hospital) statistics would require
either centralizing raw data or leaking aggregate statistics across the
client boundary before federated training even begins. Each hospital's
scaler is fit only on that hospital's own real (non-padded) bins.

Missing-value handling: within each hospital's real (non-padded) bins,
a NaN entry means "no observation recorded in this bin" -- imputed to
that hospital's own per-feature mean (i.e., 0.0 after standardization)
rather than 0 pre-standardization, which would incorrectly imply "low
value" for a feature whose normal range doesn't include zero (e.g.
platelets). Padding bins (beyond sequence_length) are zeroed after
standardization regardless of source data, since they carry no signal.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.temporal.sequence_builder import SequenceTensor


@dataclass
class NormalizationResult:
    array: np.ndarray  # same shape as input, standardized + imputed + padding zeroed
    scalers_by_hospital: dict  # hospital_id -> sklearn StandardScaler (per-feature)
    hospitals_with_insufficient_data: list  # hospitals where a feature had zero real observations cohort-wide


def fit_transform_local(
    tensor: SequenceTensor, hospital_id_by_patient: dict[int, int],
) -> NormalizationResult:
    n_patients, max_bins, n_features = tensor.array.shape
    out = np.zeros_like(tensor.array, dtype=np.float32)
    scalers: dict = {}
    insufficient: list = []

    hospital_to_indices: dict[int, list[int]] = {}
    for i, pid in enumerate(tensor.patient_ids):
        hid = hospital_id_by_patient.get(pid)
        hospital_to_indices.setdefault(hid, []).append(i)

    for hid, indices in hospital_to_indices.items():
        idx_array = np.array(indices)
        real_mask = np.zeros((len(idx_array), max_bins), dtype=bool)
        for row_i, i in enumerate(idx_array):
            real_mask[row_i, : tensor.sequence_length[i]] = True

        hosp_values = tensor.array[idx_array]  # (n_hosp_patients, max_bins, n_features)
        flat_values = hosp_values.reshape(-1, n_features)
        flat_real_mask = real_mask.reshape(-1)

        real_values = flat_values[flat_real_mask]  # (n_real_bins_total, n_features)

        scaler = StandardScaler()
        col_has_data = ~np.all(np.isnan(real_values), axis=0) if len(real_values) else np.zeros(n_features, dtype=bool)
        if not col_has_data.all():
            insufficient.append(hid)

        # Impute NaN -> per-feature mean of this hospital's real values
        # (falls back to 0.0 for a feature with zero real observations
        # hospital-wide, which becomes a constant column the scaler maps
        # to 0 after fitting -- explicitly tracked via
        # hospitals_with_insufficient_data rather than silently present).
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_means = np.nanmean(real_values, axis=0) if len(real_values) else np.zeros(n_features)
        col_means = np.nan_to_num(col_means, nan=0.0)

        imputed_real = np.where(np.isnan(real_values), col_means, real_values) if len(real_values) else real_values
        if len(imputed_real):
            scaler.fit(imputed_real)
        scalers[hid] = scaler

        # Transform ALL bins (real + padding) for this hospital's patients,
        # imputing NaN the same way, then zero out padding afterward.
        imputed_all = np.where(np.isnan(flat_values), col_means, flat_values)
        if len(imputed_all):
            transformed = scaler.transform(imputed_all) if len(imputed_real) else imputed_all
        else:
            transformed = imputed_all
        transformed = transformed.reshape(len(idx_array), max_bins, n_features)
        transformed[~real_mask] = 0.0  # zero out structural padding

        out[idx_array] = transformed

    return NormalizationResult(
        array=out, scalers_by_hospital=scalers, hospitals_with_insufficient_data=insufficient,
    )


def fit_transform_local_split_aware(
    tensor: SequenceTensor, hospital_id_by_patient: dict[int, int], train_patient_ids: set,
) -> NormalizationResult:
    """Split-aware sibling of `fit_transform_local()` -- added 2026-09-11
    as part of making Phase 11/12/13 genuinely leakage-safe against the
    permanent train/val/test split (src/data/splitting/patient_split.py).

    Identical per-hospital design (never pools across hospitals), but
    the StandardScaler for each hospital is fit using ONLY that
    hospital's patients present in `train_patient_ids` -- val/test
    patients are still IMPUTED (using the train-only column means) and
    TRANSFORMED with the frozen, train-fit scaler, exactly like any
    other frozen/inference-only input, but never contribute to the
    fitted mean/variance itself. `fit_transform_local()` itself is
    completely untouched; every existing caller of it (full-cohort
    Phase 4 runs) is unaffected.

    A hospital with zero train patients (shouldn't happen given the
    split manifest's own MIN_PATIENTS_FOR_THREE_WAY_SPLIT guarantee
    that at least the whole hospital goes to train when it's too small
    to split three ways, but checked directly rather than assumed) logs
    into `hospitals_with_insufficient_data` and falls back to zero
    imputation the same way a real zero-observation feature already
    does in the original function.
    """
    n_patients, max_bins, n_features = tensor.array.shape
    out = np.zeros_like(tensor.array, dtype=np.float32)
    scalers: dict = {}
    insufficient: list = []

    hospital_to_indices: dict[int, list[int]] = {}
    for i, pid in enumerate(tensor.patient_ids):
        hid = hospital_id_by_patient.get(pid)
        hospital_to_indices.setdefault(hid, []).append(i)

    for hid, indices in hospital_to_indices.items():
        idx_array = np.array(indices)
        train_mask_per_row = np.array([tensor.patient_ids[i] in train_patient_ids for i in idx_array])
        train_idx_array = idx_array[train_mask_per_row]

        real_mask = np.zeros((len(idx_array), max_bins), dtype=bool)
        for row_i, i in enumerate(idx_array):
            real_mask[row_i, : tensor.sequence_length[i]] = True

        # Fit statistics from TRAIN rows' real bins only.
        train_real_mask = real_mask[train_mask_per_row] if len(train_idx_array) else np.zeros((0, max_bins), dtype=bool)
        train_values = tensor.array[train_idx_array] if len(train_idx_array) else np.zeros((0, max_bins, n_features))
        flat_train_values = train_values.reshape(-1, n_features)
        flat_train_real_mask = train_real_mask.reshape(-1)
        real_train_values = flat_train_values[flat_train_real_mask]

        scaler = StandardScaler()
        col_has_data = ~np.all(np.isnan(real_train_values), axis=0) if len(real_train_values) else np.zeros(n_features, dtype=bool)
        if not len(train_idx_array) or not col_has_data.all():
            insufficient.append(hid)

        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_means = np.nanmean(real_train_values, axis=0) if len(real_train_values) else np.zeros(n_features)
        col_means = np.nan_to_num(col_means, nan=0.0)

        imputed_train_real = np.where(np.isnan(real_train_values), col_means, real_train_values) if len(real_train_values) else real_train_values
        if len(imputed_train_real):
            scaler.fit(imputed_train_real)
        scalers[hid] = scaler

        # Transform ALL of this hospital's patients (train+val+test)
        # with the train-only-fit scaler.
        hosp_values = tensor.array[idx_array]
        flat_values = hosp_values.reshape(-1, n_features)
        imputed_all = np.where(np.isnan(flat_values), col_means, flat_values)
        if len(imputed_all):
            transformed = scaler.transform(imputed_all) if len(imputed_train_real) else imputed_all
        else:
            transformed = imputed_all
        transformed = transformed.reshape(len(idx_array), max_bins, n_features)
        transformed[~real_mask] = 0.0

        out[idx_array] = transformed

    return NormalizationResult(
        array=out, scalers_by_hospital=scalers, hospitals_with_insufficient_data=insufficient,
    )
