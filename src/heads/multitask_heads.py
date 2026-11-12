"""
Phase 10 -- Multi-task prediction heads (mortality + sepsis).

Architecture + forward-pass verification only (randomly initialized
weights, no training loop yet) -- same pattern as Phase 4's LSTM,
Phase 5's autoencoder, and Phase 9's GNN: this phase proves the
mechanics are correct (right shapes, correct patient-ID alignment
between embeddings and labels), not learned predictions.

NO HOSPITAL LOOPING (unlike Phase 9): Phase 9's GNN genuinely needs
each hospital's own edge_index for message passing -- a real
graph-structural dependency. A prediction head is a plain per-patient
feed-forward function with zero cross-patient interaction, exactly
like Phase 4's LSTM and Phase 5's autoencoder, both of which already
run as one batched matrix operation over the full cohort with no
hospital looping. The model's weights are shared and frozen (no
gradient, no aggregation) regardless of how the forward pass is
chunked, so running per-hospital vs. in one call are mathematically
identical here -- there is no privacy boundary to protect by looping,
only unneeded complexity. ~200K patients x 128-dim input through two
small MLPs is well under 1GB and fine in a single call -- no adaptive
batching (src/utils/batching.py) needed at this scale, unlike Phase
4's LSTM, which genuinely OOM'd unbatched at real full scale.

HEAD ARCHITECTURE (decided 2026-08-30, confirmed with user): independent
small MLP heads per task -- Linear(in_dim, head_hidden_dim) -> ReLU ->
Dropout -> Linear(head_hidden_dim, 1) for EACH of mortality and sepsis,
with NO parameter sharing beyond the GNN trunk itself (which is already
the shared representation both tasks pull from). Alternatives
considered: (a) plain linear probes -- lower capacity, rejected; (b) an
additional shared bottleneck layer before the two heads split off --
rejected in favor of independent heads so neither task's signal is
forced through a shared compression the user didn't want.

OUTPUT IS RAW LOGITS (no sigmoid), shape (N,) NOT (N,1) -- squeezed
explicitly via squeeze(-1), not squeeze(). `multitask.loss:
weighted_bce | focal` (configs/base.yaml) both operate in logit space
(BCEWithLogitsLoss / focal-loss formulations expect logits), and (N,)
vs (N,1) is a common, easy-to-miss shape bug once paired with an
(N,)-shaped label vector during training -- see
test_multitask_heads.py's explicit shape-squeeze regression test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

VALID_SEPSIS_COHORTS = ("broad", "strict")


class MultiTaskHeads(nn.Module):
    """Two independent MLP heads (mortality, sepsis) on top of a shared
    input embedding (Phase 9's GNN output). No parameter sharing
    between the two heads beyond that shared input -- see module
    docstring for the design decision.
    """

    def __init__(self, in_dim: int, head_hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}.")
        if head_hidden_dim < 1:
            raise ValueError(f"head_hidden_dim must be >= 1, got {head_hidden_dim}.")

        def _make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1),
            )

        # Two SEPARATE nn.Sequential instances (not one shared, reused
        # module) -- each call to _make_head() creates its own freshly
        # initialized Linear layers, so mortality_head and sepsis_head
        # never share a single parameter tensor. See
        # test_multitask_heads.py's
        # test_heads_are_architecturally_independent for the direct
        # regression proof.
        self.mortality_head = _make_head()
        self.sepsis_head = _make_head()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (N, in_dim) -> {"mortality_logits": (N,), "sepsis_logits": (N,)}.

        squeeze(-1) rather than squeeze(): squeeze() with N==1 would
        ALSO collapse the batch dimension, silently turning a
        single-patient (1,) output into a scalar (). squeeze(-1) only
        ever removes the trailing size-1 head-output dimension,
        regardless of N -- see test_single_patient_shape_not_collapsed.
        """
        return {
            "mortality_logits": self.mortality_head(x).squeeze(-1),
            "sepsis_logits": self.sepsis_head(x).squeeze(-1),
        }


def align_labels_to_patient_ids(
    patient_ids: list[int],
    cohort_patients: pd.DataFrame,
    mortality: pd.Series,
    sepsis_df: pd.DataFrame,
    sepsis_cohort: str,
) -> dict:
    """Joins Phase 3's mortality + sepsis labels onto `patient_ids`
    (Phase 9's GNN embedding order) by patient ID -- never by row
    position. Direct extension of the ID-based-alignment discipline
    established by align_embeddings_to_cohort() in
    src/fusion/multimodal_fusion.py (Phase 7), applied here to labels
    instead of embeddings.

    `sepsis_cohort` must be exactly "broad" or "strict" -- selects
    which of Phase 3's dual-cohort sepsis columns (sepsis_broad /
    sepsis_strict) this run's single configurable sepsis head is being
    validated against (configs/base.yaml's multitask.sepsis_cohort).
    Raises ValueError on anything else -- never silently defaults or
    guesses (locked requirement; see configs/base.yaml's
    multitask.sepsis_cohort comment and IMPLEMENTATION_TRACKER.md).

    `mortality` is expected to share `cohort_patients`' pandas index
    (exactly what run_phase3_cohort_construction()'s
    `mortality = mortality_label(cohort_patients)` produces, and what
    load_phase3_result() reconstructs) -- assigned back onto a copy of
    cohort_patients so pandas aligns by INDEX, not by trusting the two
    already share row order positionally.

    ABSENCE FROM sepsis_df IS NOT AN ERROR (found via this phase's own
    integration test against mock data, root-caused before deciding
    how to handle it -- not assumed): compute_infection_proxy()'s
    output only covers patients with >=1 medication or microLab row
    (`set(medication.patientunitstayid) | set(micro_lab.patientunitstayid)`),
    and compute_sofa_like_scores()'s output only covers patients with
    an apachePredVar row at all. compute_sepsis_labels()'s outer merge
    of the two already documents the asymmetric case (present in one,
    absent from the other) as "a real, common situation in eICU
    (~468/4064 patients in one real run)" -- a patient absent from
    BOTH (confirmed here with mock data: patient 141029 has zero rows
    in apachePredVar, medication, AND microLab) is the same phenomenon
    taken one step further, not a different problem, and gets the
    same treatment: sepsis is UNKNOWN (NaN) for that patient, exactly
    like mortality_label() already returns NaN for an unrecognized
    hospitaldischargestatus rather than silently defaulting to
    "no sepsis". `sepsis` is therefore returned as a float array
    (1.0 / 0.0 / NaN), not bool -- NaN doesn't survive a bool dtype.
    Real-data prevalence of this is NOT YET CONFIRMED (only seen in
    this phase's mock-data integration test so far) -- flag in
    IMPLEMENTATION_TRACKER.md until the user's real run confirms an
    actual count.

    A patient_id present in the GNN embeddings but ENTIRELY ABSENT
    from cohort_patients itself DOES still raise ValueError -- unlike
    sepsis_df, cohort_patients is the authoritative source every
    downstream phase's patient_ids are derived from, so its coverage
    should be exact by construction; a gap there points to real ID
    corruption somewhere upstream, not a sparse-data edge case.
    """
    if sepsis_cohort not in VALID_SEPSIS_COHORTS:
        raise ValueError(
            f"multitask.sepsis_cohort={sepsis_cohort!r} is not valid -- must be exactly one of "
            f"{VALID_SEPSIS_COHORTS}. Never silently defaulted or guessed -- see configs/base.yaml."
        )
    sepsis_col = f"sepsis_{sepsis_cohort}"
    if sepsis_col not in sepsis_df.columns:
        raise ValueError(
            f"sepsis_df is missing expected column {sepsis_col!r} -- got columns {list(sepsis_df.columns)}."
        )

    if cohort_patients["patientunitstayid"].duplicated().any():
        raise ValueError("cohort_patients has duplicate patientunitstayid values -- cannot build a unique ID join.")
    if sepsis_df["patientunitstayid"].duplicated().any():
        raise ValueError("sepsis_df has duplicate patientunitstayid values -- cannot build a unique ID join.")

    # Index-based alignment (pandas assigns `mortality` back by its own
    # index, not by positional order) -- see docstring.
    labeled = cohort_patients[["patientunitstayid"]].copy()
    labeled["mortality"] = mortality
    mortality_by_id = dict(zip(labeled["patientunitstayid"], labeled["mortality"]))
    sepsis_by_id = dict(zip(sepsis_df["patientunitstayid"], sepsis_df[sepsis_col]))

    missing_from_cohort = [pid for pid in patient_ids if pid not in mortality_by_id]
    if missing_from_cohort:
        raise ValueError(
            f"{len(missing_from_cohort)} patient(s) present in the GNN embeddings are entirely absent from "
            f"cohort_patients (e.g. {missing_from_cohort[:5]}) -- cohort_patients is the authoritative "
            f"cohort every downstream phase's patient_ids derive from; investigate ID corruption upstream "
            f"rather than dropping them here."
        )

    mortality_arr = np.array([mortality_by_id[pid] for pid in patient_ids], dtype=float)
    # NaN for both "legitimately unknown" (patient present in sepsis_df
    # with a value) cases AND "absent from sepsis_df entirely" -- see
    # docstring. sepsis_by_id.get(pid, np.nan): absent patients get NaN
    # directly from the dict lookup's default, no separate branch needed.
    sepsis_arr = np.array(
        [(np.nan if pid not in sepsis_by_id else float(bool(sepsis_by_id[pid]))) for pid in patient_ids],
        dtype=float,
    )

    n_mortality_unknown = int(np.isnan(mortality_arr).sum())
    n_sepsis_unknown = int(np.isnan(sepsis_arr).sum())
    return {
        "mortality": mortality_arr,
        "sepsis": sepsis_arr,
        "sepsis_cohort": sepsis_cohort,
        "n_patients": len(patient_ids),
        "n_mortality_known": len(patient_ids) - n_mortality_unknown,
        "n_mortality_positive": int((mortality_arr == 1).sum()),
        "n_mortality_negative": int((mortality_arr == 0).sum()),
        "n_mortality_unknown": n_mortality_unknown,
        "n_sepsis_known": len(patient_ids) - n_sepsis_unknown,
        "n_sepsis_positive": int((sepsis_arr == 1).sum()),
        "n_sepsis_negative": int((sepsis_arr == 0).sum()),
        "n_sepsis_unknown": n_sepsis_unknown,
    }
