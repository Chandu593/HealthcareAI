"""
Phase 4b -- Sequence tensor construction.

Combines the long-format outputs of time_binning.py into one aligned
3D array (n_patients, max_bins, n_features), plus:
  - observed_mask: same shape, True where that specific (patient, bin,
    feature) had a real observation (vs. NaN-filled because nothing was
    recorded in that bin for that feature)
  - sequence_length: per patient, how many of the max_bins are "real"
    (within that patient's actual ICU stay) vs. structural padding
    beyond discharge -- computed from `patient.unitdischargeoffset`
    (falling back to `hospitaldischargeoffset` if missing), NOT from
    whether data happened to be observed. A patient can have a real bin
    with no lab draw in it (missing observation) which is different
    from a bin that doesn't exist because they were already discharged
    (structural padding) -- conflating the two would let the model
    "see" that a patient was discharged early via an artifact of
    zero-padding rather than an explicit signal, which is a leakage
    pattern worth avoiding by keeping the two concepts separate here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SequenceTensor:
    patient_ids: list[int]
    feature_names: list[str]
    array: np.ndarray            # (n_patients, max_bins, n_features), NaN where unobserved
    observed_mask: np.ndarray    # same shape, bool
    sequence_length: np.ndarray  # (n_patients,), int -- real (non-padding) bin count


def compute_sequence_length(
    patient_df: pd.DataFrame, bin_minutes: int, max_bins: int,
) -> pd.Series:
    """Index: patientunitstayid. Value: real bin count in [1, max_bins]."""
    discharge = patient_df["unitdischargeoffset"]
    if "hospitaldischargeoffset" in patient_df.columns:
        discharge = discharge.where(discharge.notna(), patient_df["hospitaldischargeoffset"])
    discharge = discharge.fillna(bin_minutes)  # unknown stay length -> assume at least 1 bin
    n_bins = np.ceil(discharge.clip(lower=1) / bin_minutes).astype(int).clip(lower=1, upper=max_bins)
    return pd.Series(n_bins.values, index=patient_df["patientunitstayid"].values)


def build_sequence_tensor(
    binned_frames: list[pd.DataFrame],
    patient_ids: list[int],
    feature_names: list[str],
    max_bins: int,
) -> SequenceTensor:
    n_patients = len(patient_ids)
    n_features = len(feature_names)
    array = np.full((n_patients, max_bins, n_features), np.nan, dtype=np.float32)

    patient_idx = {pid: i for i, pid in enumerate(patient_ids)}
    feature_idx = {f: i for i, f in enumerate(feature_names)}

    for frame in binned_frames:
        if frame.empty:
            continue
        for row in frame.itertuples(index=False):
            pid = getattr(row, "patientunitstayid")
            if pid not in patient_idx:
                continue
            feat = getattr(row, "feature")
            if feat not in feature_idx:
                continue
            bin_idx = int(getattr(row, "bin_idx"))
            if not (0 <= bin_idx < max_bins):
                continue
            array[patient_idx[pid], bin_idx, feature_idx[feat]] = getattr(row, "value")

    observed_mask = ~np.isnan(array)
    return SequenceTensor(
        patient_ids=patient_ids,
        feature_names=feature_names,
        array=array,
        observed_mask=observed_mask,
        sequence_length=np.zeros(n_patients, dtype=int),  # filled in by attach_sequence_length
    )


def attach_sequence_length(tensor: SequenceTensor, sequence_length_by_patient: pd.Series) -> SequenceTensor:
    lengths = np.array([
        int(sequence_length_by_patient.get(pid, tensor.array.shape[1])) for pid in tensor.patient_ids
    ])
    tensor.sequence_length = lengths
    return tensor
