"""
Ablation framework -- per-hospital PCA tabular "encoder".

Confirmed with the user (2026-09-10, ablation-framework decision #3):
PCA is fit PER-HOSPITAL, never globally / never on pooled raw data
across hospitals -- the same privacy boundary as per-hospital
normalization (Phase 5) and per-hospital local training (Phase 11).
Each hospital's PCA basis is therefore its own (not a shared,
cross-hospital-comparable component space) -- an explicit, documented
trade-off, not an oversight: a shared global PCA basis would require
either pooling raw tabular data across hospitals (breaks the privacy
rule enforced since Phase 3) or a federated/incremental PCA algorithm
(real additional engineering, out of scope for now -- flagged in
IMPLEMENTATION_TRACKER.md's ablation-framework entry).

Unlike TabularAutoencoder/TabularMLPEncoder, PCA has no gradient -- it
is FIT (not trained) once per hospital, on that hospital's TRAIN-split
patients ONLY (never val/test, to avoid leaking val/test structure into
the embedding itself), then used to TRANSFORM that hospital's full
train+val+test population. It cannot participate in the joint backward
pass the flat model uses for the autoencoder/mlp encoders -- see
src/training/flat_multimodal_model.py's docstring for how a frozen PCA
embedding is wired in as a fixed feature source instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class PCAEncoderConfig:
    input_dim: int
    embedding_dim: int = 32


class PerHospitalPCAEncoder:
    """One sklearn PCA instance, meant to be fit on exactly one
    hospital's train-split rows. The caller (the ablation orchestrator)
    is responsible for slicing to train-only before calling fit() --
    same responsibility split as TabularAutoencoder's caller deciding
    which tensors to pass into .train()/.eval().
    """

    def __init__(self, config: PCAEncoderConfig):
        if config.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {config.input_dim}.")
        if config.embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {config.embedding_dim}.")
        self.config = config
        self._n_components_requested = min(config.embedding_dim, config.input_dim)
        self._pca = PCA(n_components=self._n_components_requested, random_state=0)
        self._n_components_fit = None
        self._fitted = False

    def fit(self, X_train: np.ndarray) -> "PerHospitalPCAEncoder":
        if X_train.shape[0] < 2:
            raise ValueError(
                f"PCA needs at least 2 training rows to fit; got {X_train.shape[0]}. The caller "
                "should skip/flag a hospital this small for a PCA-based combo rather than fit on "
                "a single row (see fit_transform_per_hospital_pca())."
            )
        # PCA's own hard ceiling: n_components <= min(n_samples, n_features).
        max_components = min(self._n_components_requested, X_train.shape[0])
        if max_components != self._n_components_requested:
            self._pca = PCA(n_components=max_components, random_state=0)
        self._pca.fit(X_train)
        self._n_components_fit = max_components
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PerHospitalPCAEncoder.transform() called before fit().")
        embedded = self._pca.transform(X)
        if embedded.shape[1] < self.config.embedding_dim:
            # Zero-pad so every hospital's PCA embedding has the SAME
            # width regardless of how many components that hospital's
            # (possibly tiny) train set could support -- required
            # because the flat model's fusion step assumes a fixed
            # tabular embedding width across all hospitals. Zero
            # padding is a neutral choice: those extra dimensions carry
            # no signal for this hospital, rather than a fabricated one.
            pad = np.zeros((embedded.shape[0], self.config.embedding_dim - embedded.shape[1]), dtype=embedded.dtype)
            embedded = np.concatenate([embedded, pad], axis=1)
        return embedded.astype(np.float32)


def fit_transform_per_hospital_pca(
    tabular_x_by_hospital: dict, train_row_idx_by_hospital: dict, embedding_dim: int,
) -> "tuple[dict, dict]":
    """Ablation-orchestrator convenience wrapper.

    tabular_x_by_hospital: {hospital_id: np.ndarray (n_hospital_patients, input_dim)},
        already restricted to exactly that hospital's own patients --
        never touches another hospital's rows.
    train_row_idx_by_hospital: {hospital_id: np.ndarray of row indices INTO
        tabular_x_by_hospital[hid] belonging to that hospital's TRAIN split}.

    Returns (embeddings_by_hospital, skipped_too_small):
      embeddings_by_hospital: {hospital_id: np.ndarray (n_hospital_patients, embedding_dim)}
      skipped_too_small: {hospital_id: n_train_rows} for hospitals with
        fewer than 2 train rows -- these get an all-zero embedding
        (never a crash) so one small hospital doesn't take down the
        whole ablation run; the caller logs this using the hospital id.

    Each hospital's PCA never sees another hospital's data, at fit OR
    transform time -- this loop only ever indexes into that hospital's
    own tabular_x_by_hospital[hid] array.
    """
    out: dict = {}
    skipped_too_small: dict = {}
    for hid, X in tabular_x_by_hospital.items():
        train_idx = np.asarray(train_row_idx_by_hospital.get(hid, np.array([], dtype=int)))
        input_dim = X.shape[1]
        if len(train_idx) < 2:
            skipped_too_small[hid] = len(train_idx)
            out[hid] = np.zeros((X.shape[0], embedding_dim), dtype=np.float32)
            continue
        encoder = PerHospitalPCAEncoder(PCAEncoderConfig(input_dim=input_dim, embedding_dim=embedding_dim))
        encoder.fit(X[train_idx])
        out[hid] = encoder.transform(X)
    return out, skipped_too_small
