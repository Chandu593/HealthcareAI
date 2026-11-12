"""
Phase 11 -- masked multi-task loss functions.

Both classification losses (mortality, sepsis) are MASKED per-task,
never per-patient: Phase 10 established that a patient can have a
known mortality outcome but an unknown sepsis status (or vice versa,
though only sepsis-unknown was observed at real scale so far -- see
IMPLEMENTATION_TRACKER.md's Phase 10 section). Computing one combined
mask across both tasks would silently throw away patients who DO have
a usable label for one of the two tasks. `NaN` in the label tensor
marks "unknown, exclude from this task's loss for this patient" --
never coerced to 0 (negative) or 1 (positive), consistent with
align_labels_to_patient_ids()'s own contract (src/heads/multitask_heads.py).

If a hospital happens to have zero patients with a KNOWN label for a
given task (small hospitals, or the sepsis_df-absence condition Phase
10 found applying to every single one of that hospital's patients),
that task's loss term is dropped for that hospital's training step
(returns None, not a NaN-poisoned tensor) -- logged by the caller, not
silently absorbed into a zero-valued term that would look like a
real, well-fit loss of exactly 0.0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

VALID_LOSS_TYPES = ("weighted_bce", "focal")


def masked_bce_loss(
    logits: torch.Tensor, labels: torch.Tensor, pos_weight: "float | None" = None,
) -> tuple["torch.Tensor | None", int]:
    """Standard binary cross-entropy on logits, over only the
    non-NaN entries of `labels`. Returns (None, 0) if every entry is
    NaN -- the caller must handle this, not assume a tensor comes back.

    NOTE (2026-10-07): the loss_type name "weighted_bce" elsewhere in this
    project historically meant PLAIN (unweighted) BCE, because no class
    weight was ever passed. `pos_weight` (default None == the unchanged
    historical behaviour, so every locked result stays reproducible) adds a
    real positive-class weight; the paper track (src/paper) always passes one."""
    mask = ~torch.isnan(labels)
    n_used = int(mask.sum().item())
    if n_used == 0:
        return None, 0
    pw = None if pos_weight is None else torch.tensor([float(pos_weight)], dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask].float(), pos_weight=pw), n_used


def masked_focal_loss(
    logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25,
) -> tuple["torch.Tensor | None", int]:
    """Focal loss (Lin et al. 2017) on logits, over only the non-NaN
    entries of `labels`. Same None-on-all-unknown contract as
    masked_bce_loss()."""
    mask = ~torch.isnan(labels)
    n_used = int(mask.sum().item())
    if n_used == 0:
        return None, 0
    logits_m = logits[mask]
    labels_m = labels[mask].float()
    bce = F.binary_cross_entropy_with_logits(logits_m, labels_m, reduction="none")
    p = torch.sigmoid(logits_m)
    p_t = p * labels_m + (1 - p) * (1 - labels_m)
    alpha_t = alpha * labels_m + (1 - alpha) * (1 - labels_m)
    loss = alpha_t * (1 - p_t).pow(gamma) * bce
    return loss.mean(), n_used


_LOSS_FNS = {"weighted_bce": masked_bce_loss, "focal": masked_focal_loss}


def compute_multitask_loss(
    mortality_logits: torch.Tensor,
    sepsis_logits: torch.Tensor,
    mortality_labels: torch.Tensor,
    sepsis_labels: torch.Tensor,
    mortality_weight: float,
    sepsis_weight: float,
    loss_type: str,
    reconstruction: "torch.Tensor | None" = None,
    tabular_target: "torch.Tensor | None" = None,
    reconstruction_weight: float = 0.0,
) -> tuple["torch.Tensor | None", dict, dict]:
    """Combines mortality + sepsis (masked, per multitask.loss) +
    optionally a reconstruction MSE term (unsupervised -- never masked,
    every patient has tabular input regardless of label availability)
    into one total loss.

    `loss_type` must be exactly one of VALID_LOSS_TYPES -- raises
    ValueError otherwise, same "never silently default or guess"
    discipline as multitask.sepsis_cohort's own validation
    (src/heads/multitask_heads.py).

    Returns (total_loss, components, counts). `total_loss` is None
    only if NOTHING contributed at all (both classification tasks had
    zero known labels AND reconstruction is disabled/zero-weighted) --
    the caller must skip the training step in that case, not call
    .backward() on a gradient-less tensor. `components` holds each
    included term's own scalar value (for logging); `counts` holds
    how many patients contributed to each classification term.
    """
    if loss_type not in VALID_LOSS_TYPES:
        raise ValueError(f"multitask.loss={loss_type!r} is not valid -- must be one of {VALID_LOSS_TYPES}.")
    loss_fn = _LOSS_FNS[loss_type]

    mortality_loss, n_mortality = loss_fn(mortality_logits, mortality_labels)
    sepsis_loss, n_sepsis = loss_fn(sepsis_logits, sepsis_labels)

    total = None
    components: dict = {}
    if mortality_loss is not None:
        weighted = mortality_weight * mortality_loss
        total = weighted if total is None else total + weighted
        components["mortality_loss"] = mortality_loss.item()
    if sepsis_loss is not None:
        weighted = sepsis_weight * sepsis_loss
        total = weighted if total is None else total + weighted
        components["sepsis_loss"] = sepsis_loss.item()
    if reconstruction is not None and reconstruction_weight > 0:
        recon_loss = F.mse_loss(reconstruction, tabular_target)
        weighted = reconstruction_weight * recon_loss
        total = weighted if total is None else total + weighted
        components["reconstruction_loss"] = recon_loss.item()

    counts = {"n_mortality": n_mortality, "n_sepsis": n_sepsis}
    return total, components, counts
