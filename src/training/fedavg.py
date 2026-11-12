"""
Phase 12 -- FedAvg: weighted-average aggregation of per-hospital
locally-trained model weights into one global model.

Standard formulation (McMahan et al., 2017, "Communication-Efficient
Learning of Deep Networks from Decentralized Data"): each hospital's
contribution is weighted by how many patients it trained on, NOT
averaged unweighted -- a hospital with 7,058 patients (Phase 8's real
max) should influence the global model more than one with 20, matching
how a hypothetical centralized model trained on the pooled data would
naturally weight more evidence. Deviating from this weighting would
produce a genuinely different algorithm, not an implementation detail
-- see run_phase12_fedavg()'s own docstring in run_pipeline.py for the
full reasoning. This module itself is agnostic to what `weights`
represents -- it just computes a weighted average of matching
state_dicts -- so the n_patients convention lives in the caller, not
here.
"""

from __future__ import annotations

import torch


def fedavg_aggregate(state_dicts: dict, weights: dict) -> dict:
    """Weighted average of per-hospital `state_dicts` (hospital_id ->
    nn.Module state_dict), weighted by `weights` (hospital_id -> a
    positive number, by convention n_patients).

    Raises ValueError rather than silently producing a misleading
    result for every way this could go wrong without the caller
    noticing:
    - `state_dicts` is empty (nothing to aggregate).
    - a hospital in `state_dicts` has no corresponding entry in
      `weights`.
    - total weight is <= 0 (cannot normalize).
    - any two hospitals' state_dicts have mismatched keys OR shapes
      (every hospital is expected to share the exact same
      LocalModelConfig, so this should never legitimately happen --
      a mismatch here points to a real bug upstream, e.g. hospitals
      trained under different configs being aggregated together).
    """
    if not state_dicts:
        raise ValueError("fedavg_aggregate() received zero state_dicts -- nothing to aggregate.")

    missing_weights = [hid for hid in state_dicts if hid not in weights]
    if missing_weights:
        raise ValueError(
            f"no weight provided for hospital(s) {missing_weights[:5]} -- every hospital in "
            f"state_dicts needs a corresponding entry in weights."
        )

    total_weight = sum(weights[hid] for hid in state_dicts)
    if total_weight <= 0:
        raise ValueError(
            f"total aggregation weight is {total_weight} -- cannot normalize (check for zero/negative "
            f"n_patients values)."
        )

    reference_hid = next(iter(state_dicts))
    reference_sd = state_dicts[reference_hid]
    reference_keys = set(reference_sd.keys())
    for hid, sd in state_dicts.items():
        if set(sd.keys()) != reference_keys:
            raise ValueError(
                f"hospital {hid}'s state_dict keys don't match hospital {reference_hid}'s -- "
                f"architecture mismatch. Every hospital is expected to share the exact same "
                f"LocalModelConfig; investigate rather than silently aggregating mismatched models."
            )
        for key in reference_keys:
            if sd[key].shape != reference_sd[key].shape:
                raise ValueError(
                    f"hospital {hid}'s tensor {key!r} has shape {tuple(sd[key].shape)}, expected "
                    f"{tuple(reference_sd[key].shape)} (from hospital {reference_hid}) -- architecture "
                    f"mismatch, not a routine aggregation case."
                )

    aggregated: dict = {}
    for key in reference_keys:
        weighted_sum = sum(weights[hid] * state_dicts[hid][key].float() for hid in state_dicts)
        aggregated[key] = (weighted_sum / total_weight).to(reference_sd[key].dtype)
    return aggregated
