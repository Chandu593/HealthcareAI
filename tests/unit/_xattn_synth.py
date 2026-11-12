"""Synthetic upstream fixture for the cross-attention and hospital-wise tests: a TCD feature run, GRU/tabular input runs, three encoder runs (with embeddings whose columns carry
real signal for mortality / sepsis), an incident-label run and a patient table -- all written with the same manifest fields the real runs record. Nothing here touches real data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.three_modality import artifacts


def _write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1), encoding="utf-8")


def _save_npz(path, **arrs):
    np.savez_compressed(path, **arrs)


def build_synth_upstream(root, n_hosp=6, per=90, seed=0, dims=(10, 6, 7), shuffle_gru=False, shuffle_tab=False, tcd_order_shuffled=True):
    """`per` may be an int or a list of per-hospital patient counts. Directory layout mirrors the real outputs/three_modality tree."""
    """Returns dict(runs, bases, patient_table, ids, hosp, split, landmark, mort, sep, no_lab, emb)."""
    rs = np.random.default_rng(seed)
    root = Path(root)
    sizes = [per] * n_hosp if isinstance(per, int) else list(per)
    n_hosp = len(sizes)
    N = int(sum(sizes))
    ids = np.arange(7000, 7000 + N, dtype=np.int64)
    if tcd_order_shuffled:
        ids = rs.permutation(ids)                                              # the reference (TCD) order is NOT sorted: positional alignment would be wrong
    hosp = np.repeat(np.arange(11, 11 + n_hosp), sizes)
    rs.shuffle(hosp)
    split = np.empty(N, dtype=object)
    for h in np.unique(hosp):
        idx = np.flatnonzero(hosp == h)
        idx = rs.permutation(idx)
        n_tr, n_va = int(round(0.7 * len(idx))), int(round(0.15 * len(idx)))
        split[idx[:n_tr]], split[idx[n_tr:n_tr + n_va]], split[idx[n_tr + n_va:]] = "train", "val", "test"
    split = split.astype("U5")
    # landmark: ~75% stay >= 1440 min; mortality ~20%; sepsis only for landmark patients (~55% included), ~25% positive
    exit_min = np.where(rs.random(N) < 0.75, rs.integers(1440, 6000, N), rs.integers(60, 1400, N)).astype(int)
    landmark = exit_min >= 1440
    died = rs.random(N) < 0.2
    included = landmark & (rs.random(N) < 0.55)
    sep_lab = (rs.random(N) < 0.25).astype(float)
    patient_table = pd.DataFrame({"patientunitstayid": ids, "hospitaldischargestatus": np.where(died, "Expired", "Alive"), "unitdischargeoffset": exit_min})
    mort = died.astype(float)
    sep = np.where(included, sep_lab, np.nan)
    no_lab = rs.random(N) < 0.2                                                  # no lab observation in the window: GRU mask all-zero, TCD lab statistics NaN

    # signal-carrying embeddings
    def emb(d, which):
        e = rs.normal(0, 1, (N, d)).astype(np.float32)
        e[:, 0] += 2.0 * mort if which in ("gru", "tcd") else 0.0
        e[:, 1] += 2.0 * np.nan_to_num(sep) if which in ("gru", "tab") else 0.0
        e[:, 2] += 1.5 * mort if which == "tab" else 0.0
        return e
    E = {"gru": emb(dims[0], "gru"), "tabular": emb(dims[1], "tab"), "tcd": emb(dims[2], "tcd")}

    # ---- incident label run ----
    inc_dir = root / "incident_sepsis" / "inc_synth"
    inc_dir.mkdir(parents=True)
    inc = pd.DataFrame({"patientunitstayid": ids, "hospitalid": hosp, "split": split, "icu_exit_hours": exit_min / 60.0, "included": included, "label": np.where(included, sep_lab, np.nan)})
    inc.to_csv(inc_dir / "incident_sepsis_labels.csv", index=False)
    inc_man = {"run_id": "inc_synth", "identity": "i" * 64, "split_manifest_hash": "splithash" + "0" * 55}
    _write_json(inc_dir / "manifest.json", inc_man)

    # ---- TCD feature run ----
    f_dir = root / "tcd" / "features" / "tcdf_synth"
    f_dir.mkdir(parents=True)
    pats = pd.DataFrame({"patientunitstayid": ids, "hospitalid": hosp, "split": split, "window_end_minutes": np.minimum(exit_min, 1440).astype(float)})
    pats.to_csv(f_dir / "patients.csv", index=False)
    raw = rs.normal(0, 1, (N, 4)).astype(np.float32)
    raw[no_lab, 0:2] = np.nan
    np.save(f_dir / "tcd_features_raw.npy", raw)
    _write_json(f_dir / "feature_schema.json", {"features": [
        {"name": "lab_a__mean", "kind": "lab", "requires": "n>=1"}, {"name": "lab_b__mean", "kind": "lab", "requires": "n>=1"},
        {"name": "lab_a__count", "kind": "lab", "requires": "always defined"}, {"name": "age", "kind": "patient-level", "requires": "always defined"}]})
    feat_man = {"run_id": "tcdf_synth", "identity": "f" * 64, "split_manifest_hash": inc_man["split_manifest_hash"]}
    _write_json(f_dir / "manifest.json", feat_man)

    # ---- GRU inputs / tabular inputs ----
    g_dir = root / "gru" / "inputs" / "gruin_synth"
    t_dir = root / "tabular" / "inputs" / "tabin_synth"
    for d, rid, ident in ((g_dir, "gruin_synth", "g" * 64), (t_dir, "tabin_synth", "t" * 64)):
        d.mkdir(parents=True)
        pats.to_csv(d / "patients.csv", index=False)
        _write_json(d / "manifest.json", {"run_id": rid, "identity": ident, "tcd_feature_run": "tcdf_synth", "tcd_feature_identity": feat_man["identity"]})
    mask = (rs.random((N, 24, 5)) < 0.3).astype(np.uint8)
    mask[no_lab] = 0
    np.save(g_dir / "gru_mask.npy", mask)

    # ---- encoder runs ----
    runs = {"incident": str(inc_dir)}
    spec = {"gru": ("grue_synth", "gru/encoder", "gru_embeddings.npz", dict(inputs_run="gruin_synth", inputs_identity="g" * 64), shuffle_gru),
            "tabular": ("tabae_synth", "tabular/encoder", "tab_ae_embeddings.npz", dict(inputs_run="tabin_synth", inputs_identity="t" * 64), shuffle_tab),
            "tcd": ("tcde_synth", "tcd/encoder", "tcd_embeddings.npz", dict(feature_run="tcdf_synth", feature_identity=feat_man["identity"]), False)}
    for k, (rid, sub, fname, extra, shuf) in spec.items():
        d = root / sub / rid
        d.mkdir(parents=True)
        order = rs.permutation(N) if shuf else np.arange(N)
        _save_npz(d / fname, patient_ids=ids[order], embeddings=E[k][order])
        man = {"run_id": rid, "identity": (k[0] * 64), "embedding_dim": int(E[k].shape[1]), "embedding_shape": [N, int(E[k].shape[1])], "incident_label_run": "inc_synth",
               "incident_identity": inc_man["identity"], "files": {fname: artifacts.sha256_file(d / fname)}, **extra}
        _write_json(d / "manifest.json", man)
        runs[k] = str(d)
    bases = {"tcd_features": str(root / "tcd" / "features"), "gru_inputs": str(root / "gru" / "inputs"), "tabular_inputs": str(root / "tabular" / "inputs")}
    return {"runs": runs, "bases": bases, "patient_table": patient_table, "ids": ids, "hosp": hosp, "split": split, "landmark": landmark, "mort": mort, "sep": sep,
            "no_lab": no_lab, "emb": E, "incident_dir": str(inc_dir)}
