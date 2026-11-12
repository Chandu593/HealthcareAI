"""Tabular autoencoder inputs: the paper track's CLEAN, leakage-guarded static feature set, for every patient, in the TCD run's patient order.

Reuses `src.paper.features.build_feature_bundle` (window-bounded, allowlisted, vocabularies from TRAINING patients only) rather than the legacy Phase-5 features, and applies the
paper's `sepsis_clean` exclusions (label-ingredient groups: APACHE physiology and scores, SOFA labs, antibiotic/vasopressor/treatment/medication/infusion tokens, infection
diagnoses, cultures; hospital covariates; deconfounding name patterns; activetx) PLUS every `lab` group, which TCD already represents. What remains is admission context:
demographics, APACHE predictor inputs, non-infection diagnoses, admission drugs.

Window: paper-track convention (events from `lookback_minutes` before ICU admission up to min(ICU exit, tcd.window_minutes)); all stays are included (no minimum-stay filter) because
every patient needs an embedding. Labels are never used here (the bundle's label slots get NaN stubs).

Column types matter for the reconstruction loss: CONTINUOUS columns are winsorised, median-imputed and standardised (training statistics only, via the TCD preprocessing code);
BINARY columns (all training values in {0,1}) are kept as 0/1 so they can be reconstructed with a binary cross-entropy; missing indicators are binary too. `tab_is_binary.npy` records
the type of every input column.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.paper.config import get_paper_cfg
from src.paper.features import _resolve_table, build_feature_bundle
from src.three_modality import artifacts, tcd_features as tf
from src.three_modality.config import TABULAR_INPUT_DEFINITION_VERSION

logger = logging.getLogger("eicu_fl")

INPUT_FILES = ["patients.csv", "tab_inputs.npy", "tab_is_binary.npy", "feature_schema.json", "preprocessing_state.npz", "preprocessing_meta.json", "binary_columns.json",
               "input_report.json", "input_report.md", "config_snapshot.json"]
BUNDLE_TABLES = ["patient", "hospital", "apachePredVar", "apachePatientResult", "diagnosis", "admissionDrug", "treatment", "medication", "medication_imputed", "medication_imputed1",
                 "infusionDrug", "microLab", "lab"]
FORBIDDEN_NAME = re.compile(r"(^lab_|^apr_|discharge|died|actual|predicted|mortality)")


def paper_cfg_for(acfg: dict, tcfg: dict) -> dict:
    over = {"top_labs": 0, "include_apache_scores": False, "cache": False, **(acfg.get("bundle_overrides") or {})}
    return get_paper_cfg({"paper": {"window": {"minutes": int(tcfg["window_minutes"]), "min_icu_los_minutes": 0}, "features": over, "sequence": {"enabled": False}}})


def tabular_inputs_identity(src: dict, tcd_manifest: dict, acfg: dict, tcfg: dict) -> str:
    tables = {n: artifacts.fingerprint(p) for n in BUNDLE_TABLES if (p := _resolve_table(src, n))}
    params = {k: v for k, v in acfg.items() if k not in ("training", "encoder")}
    return artifacts.hash_json({"definition": TABULAR_INPUT_DEFINITION_VERSION, "params": params, "window_minutes": tcfg["window_minutes"],
                                "tcd_feature_identity": tcd_manifest["identity"], "tables": tables})


def split_binary_columns(raw: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """True for columns whose observed TRAINING values are all in {0, 1} (and which have at least one observation)."""
    tr = raw[train_mask]
    out = np.zeros(raw.shape[1], bool)
    for j in range(raw.shape[1]):
        o = tr[:, j][~np.isnan(tr[:, j])]
        out[j] = o.size > 0 and bool(np.all((o == 0) | (o == 1)))
    return out


def build_inputs_matrix(raw: np.ndarray, names: list, groups: list, train_mask: np.ndarray, pp: dict) -> dict:
    """Train-only preprocessing of the selected raw columns -> {X, is_binary, names, kinds, groups, continuous_state, binary_kept, dropped, ...}."""
    is_bin = split_binary_columns(raw, train_mask)
    cont_idx, bin_idx = np.flatnonzero(~is_bin), np.flatnonzero(is_bin)
    out = {"dropped": []}
    # continuous: the TCD preprocessing (drop too-missing/constant, winsorise, median-impute, standardise, deduplicated missing indicators), fitted on TRAIN
    if len(cont_idx):
        state = tf.fit_preprocessing(raw[:, cont_idx], [names[j] for j in cont_idx], train_mask, pp)
        Xc = tf.transform_features(raw[:, cont_idx], state)
        nk = len(state["kept_idx"])
        cont_block, ind_block = Xc[:, :nk], Xc[:, nk:]
        cont_names = state["kept_names"]; cont_groups = [groups[cont_idx[j]] for j in state["kept_idx"]]
        ind_names = list(state["indicator_names"])
        out["dropped"] += [{**d, "type": "continuous"} for d in state["dropped"]]
        out["continuous_state"] = state
    else:
        cont_block = ind_block = np.zeros((raw.shape[0], 0), np.float32); cont_names = cont_groups = ind_names = []; out["continuous_state"] = None
    # binary: kept as 0/1; constant-in-training or too-missing columns dropped; remaining NaN imputed to 0 (counted)
    bin_keep, bin_nan_imputed = [], {}
    tr = np.flatnonzero(train_mask)
    for j in bin_idx:
        col = raw[:, j]
        miss_tr = float(np.isnan(col[tr]).mean())
        ob = col[tr][~np.isnan(col[tr])]
        if miss_tr > float(pp["max_missing_fraction"]):
            out["dropped"].append({"feature": names[j], "code": "too_missing", "type": "binary", "reason": f"train missing fraction {miss_tr:.4f}"}); continue
        if ob.min() == ob.max():
            out["dropped"].append({"feature": names[j], "code": "constant", "type": "binary", "reason": "constant in training"}); continue
        bin_keep.append(j); bin_nan_imputed[names[j]] = int(np.isnan(col).sum())
    bin_block = np.nan_to_num(raw[:, bin_keep], nan=0.0).astype(np.float32) if bin_keep else np.zeros((raw.shape[0], 0), np.float32)
    X = np.concatenate([cont_block, bin_block, ind_block], axis=1).astype(np.float32)
    kinds = ["continuous"] * len(cont_names) + ["binary"] * len(bin_keep) + ["missing_indicator"] * len(ind_names)
    out.update({"X": X, "is_binary": np.array([k != "continuous" for k in kinds], bool), "names": cont_names + [names[j] for j in bin_keep] + ind_names, "kinds": kinds,
                "groups": cont_groups + [groups[j] for j in bin_keep] + ["missing_indicator"] * len(ind_names), "binary_kept": [names[j] for j in bin_keep],
                "binary_nan_imputed": bin_nan_imputed, "n_continuous": len(cont_names), "n_binary": len(bin_keep), "n_indicators": len(ind_names)})
    return out


def _load_tcd(tcd_run_dir) -> dict:
    d = Path(tcd_run_dir)
    return {"patients": pd.read_csv(d / "patients.csv"), "manifest": json.loads((d / "manifest.json").read_text(encoding="utf-8"))}


def build_tabular_inputs(src: dict, patient_full: pd.DataFrame, tcd_run_dir, acfg: dict, tcfg: dict, out_base, run_id: "str | None" = None, force: bool = False) -> dict:
    t0 = time.time()
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    tcd = _load_tcd(tcd_run_dir)
    identity = tabular_inputs_identity(src, tcd["manifest"], acfg, tcfg)
    if not force:
        hit = artifacts.find_reusable_run(out_base, identity)
        if hit:
            logger.info("[TAB_INPUTS] cache HIT: reusing verified run %s (identity %s)", hit["run_id"], identity[:12])
            return {"run_dir": hit["run_dir"], "run_id": hit["run_id"], "cache_hit": True, "manifest": hit}
    run_id, run_dir = artifacts.new_run_dir(out_base, "tabin", identity, run_id)
    logger.info("[TAB_INPUTS] cache MISS (identity %s) -> computing into %s", identity[:12], run_dir)
    pats = tcd["patients"]
    ids = pats["patientunitstayid"].to_numpy(np.int64)
    split = pats["split"].to_numpy()
    train_mask = split == "train"
    cp = patient_full.copy()
    cp["patientunitstayid"] = cp["patientunitstayid"].astype("int64")
    cp = cp[cp["patientunitstayid"].isin(ids)].drop_duplicates("patientunitstayid").sort_values("patientunitstayid").reset_index(drop=True)
    if len(cp) != len(ids):
        raise ValueError(f"the patient table covers {len(cp)} of the {len(ids)} patients of the TCD run")
    mort_stub = pd.Series(np.nan, index=cp["patientunitstayid"].to_numpy(np.int64))           # labels are never used for features
    bundle = build_feature_bundle(paper_cfg_for(acfg, tcfg), src, cp, mort_stub, None, train_patient_ids=ids[train_mask].tolist(), sepsis_cohort="broad")
    pos = pd.Series(np.arange(len(bundle.patient_ids)), index=np.asarray(bundle.patient_ids)).reindex(ids)
    if pos.isna().any():
        raise ValueError(f"{int(pos.isna().sum())} patients of the TCD run are missing from the feature bundle")
    order = pos.to_numpy(np.int64)
    fs = acfg["feature_set"]
    cols = list(bundle.columns_excluding(list(fs["exclude_groups"]), list(fs["exclude_name_regex"])))
    names_all = [bundle.feature_names[i] for i in cols]
    groups_all = [bundle.feature_groups[i] for i in cols]
    bad = [n for n in names_all if FORBIDDEN_NAME.search(n)]
    if bad:
        raise AssertionError(f"forbidden feature names reached the tabular inputs: {bad[:5]}")
    raw = np.asarray(bundle.X)[order][:, cols].astype(np.float32)
    logger.info("[TAB_INPUTS] clean static set: %d of %d bundle columns (groups %s)", len(cols), len(bundle.feature_names), sorted(set(groups_all)))
    res = build_inputs_matrix(raw, names_all, groups_all, train_mask, acfg["preprocessing"])
    X = res["X"]
    logger.info("[TAB_INPUTS] inputs %s: %d continuous + %d binary + %d missing indicators; dropped %d", X.shape, res["n_continuous"], res["n_binary"], res["n_indicators"], len(res["dropped"]))

    from collections import Counter
    report = {"definition_version": TABULAR_INPUT_DEFINITION_VERSION, "shape": list(X.shape),
              "patients": {sp: int((split == sp).sum()) for sp in ("train", "val", "test")} | {"total": int(len(ids))},
              "bundle_columns_total": len(bundle.feature_names), "bundle_columns_by_group": dict(Counter(bundle.feature_groups)),
              "selected_raw_columns": len(cols), "selected_by_group": dict(Counter(groups_all)), "excluded_groups": list(fs["exclude_groups"]),
              "kept": {"continuous": res["n_continuous"], "binary": res["n_binary"], "missing_indicator": res["n_indicators"], "total": int(X.shape[1])},
              "kept_by_group": dict(Counter(g for g in res["groups"])), "dropped": res["dropped"], "dropped_by_reason": dict(Counter(d["code"] for d in res["dropped"])),
              "binary_columns_with_missing_imputed_to_zero": {k: v for k, v in res["binary_nan_imputed"].items() if v},
              "forbidden_name_check": "passed", "excluded_by_design": {"apache_scores": True, "labs": True, "sequence_block": True},
              "window": {"minutes": int(tcfg["window_minutes"]), "lookback_minutes_before_icu_admission": int(paper_cfg_for(acfg, tcfg)["window"]["lookback_minutes"]), "min_icu_los_minutes": 0},
              "bundle_cache_key": bundle.meta.get("cache_key"), "tcd_feature_run": tcd["manifest"]["run_id"]}
    run_dir.mkdir(parents=True)
    artifacts.atomic_write(run_dir / "patients.csv", pats.to_csv(index=False))
    artifacts.atomic_save_npy(run_dir / "tab_inputs.npy", X)
    artifacts.atomic_save_npy(run_dir / "tab_is_binary.npy", res["is_binary"])
    artifacts.atomic_write(run_dir / "feature_schema.json", json.dumps({"names": res["names"], "kinds": res["kinds"], "groups": res["groups"]}, indent=1))
    if res["continuous_state"] is not None:
        tf.save_state(res["continuous_state"], run_dir)
    else:
        artifacts.atomic_write(run_dir / "preprocessing_state.npz", b""); artifacts.atomic_write(run_dir / "preprocessing_meta.json", "{}")
    artifacts.atomic_write(run_dir / "binary_columns.json", json.dumps({"kept": res["binary_kept"], "nan_imputed_to_zero": res["binary_nan_imputed"]}, indent=1))
    artifacts.atomic_write(run_dir / "input_report.json", json.dumps(report, indent=1, default=str))
    artifacts.atomic_write(run_dir / "input_report.md", render_report_md(report, run_id))
    artifacts.atomic_write(run_dir / "config_snapshot.json", json.dumps({"tabular_inputs": {k: v for k, v in acfg.items() if k not in ("training", "encoder")}}, indent=1, default=str))
    man = {"run_id": run_id, "definition_version": TABULAR_INPUT_DEFINITION_VERSION, "identity": identity, "files": {n: artifacts.sha256_file(run_dir / n) for n in INPUT_FILES},
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shape": report["shape"], "counts": report["patients"], "kept": report["kept"],
           "tcd_feature_run": tcd["manifest"]["run_id"], "tcd_feature_identity": tcd["manifest"]["identity"], "cache": {"hit": False, "forced_recompute": bool(force)},
           "seconds": round(time.time() - t0, 1)}
    artifacts.atomic_write(run_dir / "manifest.json", json.dumps(man, indent=1))
    logger.info("[TAB_INPUTS] COMPLETE in %.1fs -> %s", time.time() - t0, run_dir)
    return {"run_dir": str(run_dir), "run_id": run_id, "cache_hit": False, "manifest": man, "report": report}


def render_report_md(rep: dict, run_id: str) -> str:
    p, k = rep["patients"], rep["kept"]
    L = [f"# Tabular autoencoder input report -- {run_id}", "", f"Definition `{rep['definition_version']}`; shape {rep['shape']}. Clean static set from the paper-track feature bundle (window "
         f"{rep['window']['minutes']} min after ICU admission, {rep['window']['lookback_minutes_before_icu_admission']} min of pre-ICU look-back, all stays).", "",
         "## Patients", "", "| Split | Patients |", "|---|---|", f"| train | {p['train']} |", f"| val | {p['val']} |", f"| test | {p['test']} |", f"| total | {p['total']} |", "",
         "## Columns", "", f"- bundle columns: {rep['bundle_columns_total']} (by group: {rep['bundle_columns_by_group']})",
         f"- excluded groups: {', '.join(rep['excluded_groups'])}", f"- selected raw columns: {rep['selected_raw_columns']} (by group: {rep['selected_by_group']})",
         f"- kept: {k['continuous']} continuous + {k['binary']} binary + {k['missing_indicator']} missing indicators = {k['total']} (by group: {rep['kept_by_group']})",
         f"- dropped: {len(rep['dropped'])} (by reason: {rep['dropped_by_reason']})",
         f"- binary columns with missing values imputed to 0: {rep['binary_columns_with_missing_imputed_to_zero'] or 'none'}",
         f"- forbidden-name check (labs, APACHE scores, outcome-like names): {rep['forbidden_name_check']}", "",
         "APACHE scores, laboratory features and the sequence block are excluded by design; the hospital covariates and every label-ingredient group of the paper's clean set are excluded too."]
    return "\n".join(L) + "\n"


def load_tabular_inputs(run_dir) -> dict:
    d = Path(run_dir)
    return {"patients": pd.read_csv(d / "patients.csv"), "X": np.load(d / "tab_inputs.npy", mmap_mode="r"), "is_binary": np.load(d / "tab_is_binary.npy"),
            "schema": json.loads((d / "feature_schema.json").read_text(encoding="utf-8")), "manifest": json.loads((d / "manifest.json").read_text(encoding="utf-8"))}
