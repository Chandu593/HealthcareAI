"""
Phase 7 -- Multimodal fusion.

Concatenates Phase 4's LSTM time-series embeddings with Phase 5's
autoencoder tabular embeddings into one unified per-patient
representation. Text (Phase 6) is intentionally excluded -- locked
decision, no clinical notes in this eICU export; see
docs/text_modality_decision.md.

FUSION TYPE: pure concatenation (`fusion.type: concatenation` in
config), no learned projection layer. Consistent with Phase 4/5, which
also have no training loop yet -- there's no training signal yet to
justify committing to a specific projected output width, so this phase
verifies the fusion mechanics (correct patient-ID alignment, correct
concatenation) rather than a learned joint representation.

ALIGNMENT, NOT ASSUMED ORDERING: Phase 4 and Phase 5 currently derive
their patient_ids from the same phase3_result['cohort_patients'] list,
in the same order, and neither filters/drops patients afterward -- so
in practice their saved embeddings' patient_ids already match. This
module does NOT rely on that holding by convention alone: every fusion
here re-indexes each modality's embeddings by patient ID against an
explicit target order, and raises loudly if any expected patient is
missing from a modality's embeddings, rather than silently producing a
zero/misaligned row. A silent misalignment here would corrupt every
downstream sample without any obvious symptom -- exactly the failure
mode this project treats as unacceptable (see IMPLEMENTATION_TRACKER.md's
recurring theme: "no silent skips or fabrication").
"""

from __future__ import annotations

import numpy as np


def align_embeddings_to_cohort(
    patient_ids: list[int], embeddings: np.ndarray, target_order: list[int],
) -> np.ndarray:
    """Reindexes `embeddings` (currently ordered per `patient_ids`) to
    match `target_order` exactly, via an explicit ID-based lookup.

    Raises ValueError if any patient in `target_order` is missing from
    `patient_ids` -- a missing patient means this modality's embeddings
    don't cover the full cohort, which should never happen given how
    Phase 4/5 are built (both start from the same full cohort and never
    drop patients), so treat it as a real problem to investigate, not
    something to paper over with a zero/NaN row.
    """
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("patient_ids contains duplicates -- cannot build a unique ID-> row index")

    index = {pid: i for i, pid in enumerate(patient_ids)}
    missing = [pid for pid in target_order if pid not in index]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(target_order)} target patient(s) missing from this modality's "
            f"embeddings (e.g. {missing[:5]}) -- Phase 4 and Phase 5 are expected to cover the exact "
            f"same cohort; investigate before fusing rather than silently dropping or zero-filling them."
        )
    row_order = [index[pid] for pid in target_order]
    return embeddings[row_order]


def fuse_embeddings(
    lstm_patient_ids: list[int],
    lstm_embeddings: np.ndarray,
    autoencoder_patient_ids: list[int],
    autoencoder_embeddings: np.ndarray,
    target_order: list[int] | None = None,
) -> tuple[list[int], np.ndarray]:
    """Concatenates LSTM + autoencoder embeddings into one unified
    per-patient representation, aligned by patient ID.

    `target_order`, if given, is the authoritative output patient order
    (typically phase3_result['cohort_patients']'s own patient order --
    the full cohort, independent of either modality's own storage
    order). If omitted, defaults to `lstm_patient_ids`'s order, and
    `autoencoder_patient_ids` must cover that exact same set.

    Returns (patient_ids, fused_embeddings) where fused_embeddings has
    shape (len(target_order), lstm_dim + autoencoder_dim) -- LSTM
    columns first, then autoencoder columns, in that order.
    """
    order = target_order if target_order is not None else list(lstm_patient_ids)
    lstm_aligned = align_embeddings_to_cohort(lstm_patient_ids, lstm_embeddings, order)
    ae_aligned = align_embeddings_to_cohort(autoencoder_patient_ids, autoencoder_embeddings, order)
    fused = np.concatenate([lstm_aligned, ae_aligned], axis=1)
    return order, fused
