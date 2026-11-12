"""Hospital-level cohort filters for the paper track (numpy only, additive; the default is OFF).

`hospital_size_mask` restricts an experiment to hospitals that contribute at least N cohort patients. It exists for the
FLICS-scale federations (the 3 or 10 largest hospitals), where every arm -- federated, central and local -- must see the same hospitals. Hospital sizes are counted over the whole
cohort (all splits) and never use a label, so the filter cannot leak outcome information; it only decides which sites exist.
"""

from __future__ import annotations

import numpy as np


def hospital_size_mask(hospital_ids, min_patients):
    """-> (mask, info) keeping patients of hospitals with >= min_patients cohort patients, or None when the filter is off (<= 0)."""
    if not min_patients or int(min_patients) <= 0:
        return None
    hid = np.asarray(hospital_ids)
    hosp, cnt = np.unique(hid, return_counts=True)
    keep = hosp[cnt >= int(min_patients)]
    mask = np.isin(hid, keep)
    kept_sizes = cnt[cnt >= int(min_patients)]
    info = {"rule": f"hospital size >= {int(min_patients)} cohort patients", "hospitals_total": int(len(hosp)), "hospitals_kept": int(len(keep)),
            "patients_kept": int(mask.sum()), "patients_total": int(len(mask)),
            "kept_hospital_sizes": ({"min": int(kept_sizes.min()), "median": float(np.median(kept_sizes)), "max": int(kept_sizes.max())}
                                    if len(kept_sizes) else None)}
    return mask, info


def combine_filters(parts, hospital_ids):
    """AND several (mask, info) filters. A single part is returned unchanged, so enabling nothing new reproduces the old behaviour exactly."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0]
    mask = np.logical_and.reduce([m for m, _ in parts])
    hid = np.asarray(hospital_ids)
    info = {"rule": "; ".join(i["rule"] for _, i in parts), "hospitals_total": int(len(np.unique(hid))),
            "hospitals_kept": int(len(np.unique(hid[mask]))), "patients_kept": int(mask.sum()), "patients_total": int(len(mask)),
            "components": [i for _, i in parts]}
    return mask, info


def hospital_size_profile(hospital_ids, thresholds=(200, 300, 400, 500, 600, 800, 1000, 1500), target_hospitals=None):
    """Read-only summary used to choose a size threshold: for each candidate threshold, how many hospitals and patients survive.
    With `target_hospitals`, also the largest threshold that still keeps at least that many hospitals (the size of the N-th largest hospital)."""
    hosp, cnt = np.unique(np.asarray(hospital_ids), return_counts=True)
    rows = []
    for t in thresholds:
        k = cnt >= int(t)
        rows.append({"threshold": int(t), "hospitals": int(k.sum()), "patients": int(cnt[k].sum()),
                     "median_size": float(np.median(cnt[k])) if k.any() else None})
    out = {"hospitals_total": int(len(hosp)), "patients_total": int(cnt.sum()), "rows": rows}
    if target_hospitals:
        order = np.sort(cnt)[::-1]
        n = int(target_hospitals)
        out["target"] = ({"hospitals": n, "threshold_for_target": int(order[n - 1]), "patients_at_target": int(order[:n].sum())}
                         if len(order) >= n else {"hospitals": n, "threshold_for_target": None, "patients_at_target": None})
    return out
