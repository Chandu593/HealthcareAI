"""
Phase 5b -- Per-hospital local normalization for tabular features.

Same federated-privacy rationale as src/data/temporal/normalization.py
(fit statistics per hospital, never globally), simplified for the 2D
(patients x features) case -- no time/bin dimension here, so there's no
padding concept to zero out.

One-hot / binary columns (anything the caller marks in
`categorical_column_mask`) are passed through unscaled -- standardizing
a 0/1 indicator would destroy its interpretability as "this category is
present" for no benefit, since it's already on a bounded, comparable
scale.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class TabularNormalizationResult:
    array: np.ndarray  # (n_patients, n_features), standardized (numeric cols) + imputed
    scalers_by_hospital: dict  # hospital_id -> sklearn StandardScaler (numeric columns only)
    hospitals_with_insufficient_data: list


def fit_transform_tabular_local(
    array: np.ndarray,
    patient_ids: list[int],
    hospital_id_by_patient: dict[int, int],
    numeric_column_mask: np.ndarray,  # bool, shape (n_features,) -- True = standardize, False = pass through
) -> TabularNormalizationResult:
    n_patients, n_features = array.shape
    out = array.copy().astype(np.float32)
    scalers: dict = {}
    insufficient: list = []

    numeric_idx = np.where(numeric_column_mask)[0]

    hospital_to_indices: dict = {}
    for i, pid in enumerate(patient_ids):
        hid = hospital_id_by_patient.get(pid)
        hospital_to_indices.setdefault(hid, []).append(i)

    for hid, indices in hospital_to_indices.items():
        idx_array = np.array(indices)
        hosp_values = array[idx_array][:, numeric_idx] if len(numeric_idx) else np.zeros((len(idx_array), 0))

        col_has_data = ~np.all(np.isnan(hosp_values), axis=0) if hosp_values.size else np.array([], dtype=bool)
        if len(col_has_data) and not col_has_data.all():
            insufficient.append(hid)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_means = np.nanmean(hosp_values, axis=0) if hosp_values.size else np.zeros(len(numeric_idx))
        col_means = np.nan_to_num(col_means, nan=0.0)

        imputed = np.where(np.isnan(hosp_values), col_means, hosp_values) if hosp_values.size else hosp_values
        scaler = StandardScaler()
        if len(numeric_idx) and len(imputed):
            scaler.fit(imputed)
            transformed_numeric = scaler.transform(imputed)
        else:
            transformed_numeric = imputed
        scalers[hid] = scaler

        if len(numeric_idx):
            out[np.ix_(idx_array, numeric_idx)] = transformed_numeric

        # Non-numeric (one-hot/binary) columns: pass through, but a
        # genuinely missing categorical (NaN, e.g. a raw value that was
        # itself absent before one-hot encoding never applies here since
        # one_hot_with_vocabulary always emits 0/1 -- this branch exists
        # only in case a caller passes raw NaNs in a "categorical" column
        # directly) gets imputed to 0.0 rather than left as NaN.
        cat_idx = np.where(~numeric_column_mask)[0]
        if len(cat_idx):
            cat_block = out[np.ix_(idx_array, cat_idx)]
            out[np.ix_(idx_array, cat_idx)] = np.nan_to_num(cat_block, nan=0.0)

    return TabularNormalizationResult(
        array=out, scalers_by_hospital=scalers, hospitals_with_insufficient_data=insufficient,
    )


def fit_transform_tabular_local_split_aware(
    array: np.ndarray, patient_ids: list[int], hospital_id_by_patient: dict[int, int],
    numeric_column_mask: np.ndarray, train_patient_ids: set,
) -> TabularNormalizationResult:
    """Split-aware sibling of `fit_transform_tabular_local()` -- see
    src/data/temporal/normalization.py's `fit_transform_local_split_aware()`
    for the full rationale (identical design, 2D case). The scaler for
    each hospital is fit using ONLY that hospital's train-split
    patients; val/test patients are imputed with the train-only column
    means and transformed with the frozen, train-fit scaler.
    `fit_transform_tabular_local()` itself is untouched.
    """
    n_patients, n_features = array.shape
    out = array.copy().astype(np.float32)
    scalers: dict = {}
    insufficient: list = []

    numeric_idx = np.where(numeric_column_mask)[0]
    cat_idx = np.where(~numeric_column_mask)[0]

    hospital_to_indices: dict = {}
    for i, pid in enumerate(patient_ids):
        hid = hospital_id_by_patient.get(pid)
        hospital_to_indices.setdefault(hid, []).append(i)

    for hid, indices in hospital_to_indices.items():
        idx_array = np.array(indices)
        train_mask_per_row = np.array([patient_ids[i] in train_patient_ids for i in idx_array])
        train_idx_array = idx_array[train_mask_per_row]

        train_values = array[train_idx_array][:, numeric_idx] if len(train_idx_array) and len(numeric_idx) else np.zeros((len(train_idx_array), len(numeric_idx)))
        col_has_data = ~np.all(np.isnan(train_values), axis=0) if train_values.size else np.array([], dtype=bool)
        if not len(train_idx_array) or (len(col_has_data) and not col_has_data.all()):
            insufficient.append(hid)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_means = np.nanmean(train_values, axis=0) if train_values.size else np.zeros(len(numeric_idx))
        col_means = np.nan_to_num(col_means, nan=0.0)

        imputed_train = np.where(np.isnan(train_values), col_means, train_values) if train_values.size else train_values
        scaler = StandardScaler()
        if len(numeric_idx) and len(imputed_train):
            scaler.fit(imputed_train)
        scalers[hid] = scaler

        hosp_values = array[idx_array][:, numeric_idx] if len(numeric_idx) else np.zeros((len(idx_array), 0))
        imputed_all = np.where(np.isnan(hosp_values), col_means, hosp_values) if hosp_values.size else hosp_values
        if len(numeric_idx) and len(imputed_train):
            transformed_numeric = scaler.transform(imputed_all)
        else:
            transformed_numeric = imputed_all
        if len(numeric_idx):
            out[np.ix_(idx_array, numeric_idx)] = transformed_numeric

        if len(cat_idx):
            cat_block = out[np.ix_(idx_array, cat_idx)]
            out[np.ix_(idx_array, cat_idx)] = np.nan_to_num(cat_block, nan=0.0)

    return TabularNormalizationResult(
        array=out, scalers_by_hospital=scalers, hospitals_with_insufficient_data=insufficient,
    )
