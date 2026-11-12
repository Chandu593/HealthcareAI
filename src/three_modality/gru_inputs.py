"""GRU inputs: fresh 24 x 1-hour sequences from the first 24 ICU hours ONLY.

NOT the legacy Phase-4 arrays: those use a 48 x 240-minute grid (192 h) and derive `sequence_length` from `unitdischargeoffset` (the ICU length of stay leaks into the
recurrent encoder). Here:
  * the window is exactly [0, min(ICU exit, window_minutes)] -- enforced by the reader (same reader as the TCD features) AND again inside bin_series (defence in depth),
    and proven by a poison audit on real data;
  * the sequence length is the FIXED number of bins for every patient; nothing is derived from the ICU exit. A patient whose stay is shorter simply has observation-mask 0
    for the bins after the exit, like any other unobserved bin (no value is ever fabricated: unobserved cells are 0 with mask 0);
  * channels = [scaled value, observation mask] for each of the lab variables the TCD run selected (training-patient coverage, so the selection is already train-only);
    within an hour bin the LAST observed value is kept;
  * per-variable scaling (winsorise at training percentiles, then standardise) is fitted on TRAINING patients' observed values only (pooled, not per hospital).
Universe, row order, splits and variable selection are inherited from the verified TCD feature run, so every modality shares one patient ordering.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.three_modality import artifacts
from src.three_modality.config import GRU_INPUT_DEFINITION_VERSION
from src.three_modality.tcd_features import clip_series, read_lab_series

logger = logging.getLogger("eicu_fl")

INPUT_FILES = ["patients.csv", "gru_values.npy", "gru_mask.npy", "variables.json", "scaling.json", "input_report.json", "input_report.md", "cutoff_audit.json", "config_snapshot.json"]


def bin_series(series: dict, window_end: np.ndarray, window_minutes: float, bin_minutes: int, n_bins: int, variables: list) -> tuple:
    """-> (values float32 [N, n_bins, V] UNSCALED, observed bool [N, n_bins, V]). Clips to each patient's window itself, then keeps the LAST value per (patient, bin).
    Bin b covers ((b)*bin_minutes, (b+1)*bin_minutes], except that offset 0 belongs to bin 0; minute `window_minutes` belongs to the last bin."""
    N, V = len(window_end), len(variables)
    series = clip_series(series, np.asarray(window_end, float), window_minutes)
    values = np.zeros((N, n_bins, V), np.float32)
    observed = np.zeros((N, n_bins, V), bool)
    for j, k in enumerate(variables):
        if k not in series or len(series[k][0]) == 0:
            continue
        p, t, v = series[k]
        order = np.lexsort((t, p))
        p, t, v = p[order], t[order], v[order]
        b = np.clip(np.ceil(t / float(bin_minutes)).astype(np.int64) - 1, 0, n_bins - 1)
        key = p.astype(np.int64) * n_bins + b
        last = np.flatnonzero(np.r_[key[1:] != key[:-1], True])        # sorted by (patient, time): the last row of each (patient, bin) run is the freshest value
        values[p[last], b[last], j] = v[last]
        observed[p[last], b[last], j] = True
    return values, observed


def fit_scaling(values: np.ndarray, observed: np.ndarray, train_mask: np.ndarray, variables: list, winsor_pct) -> dict:
    """Per variable, from TRAINING patients' observed cells only: winsorisation limits, mean and std of the winsorised values."""
    tr = np.flatnonzero(train_mask)
    out = {"variables": {}, "n_train": int(len(tr)), "winsor_pct": list(winsor_pct)}
    for j, k in enumerate(variables):
        o = observed[tr][:, :, j]
        x = values[tr][:, :, j][o].astype(np.float64)
        if x.size == 0:
            out["variables"][k] = {"n_observed_train_cells": 0, "lo": 0.0, "hi": 0.0, "mean": 0.0, "std": 1.0, "degenerate": "never observed in training"}
            continue
        lo, hi = np.percentile(x, winsor_pct)
        w = np.clip(x, lo, hi)
        sd = float(w.std())
        out["variables"][k] = {"n_observed_train_cells": int(x.size), "lo": float(lo), "hi": float(hi), "mean": float(w.mean()), "std": sd if sd > 1e-8 else 1.0,
                               "degenerate": "" if sd > 1e-8 else "constant in training: scaled to 0"}
    return out


def apply_scaling(values: np.ndarray, observed: np.ndarray, scaling: dict, variables: list) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    for j, k in enumerate(variables):
        s = scaling["variables"][k]
        x = np.clip(values[:, :, j].astype(np.float64), s["lo"], s["hi"])
        z = (x - s["mean"]) / s["std"]
        out[:, :, j] = np.where(observed[:, :, j], z, 0.0)             # unobserved cells are 0 AND mask 0 (never an imputed value)
    return out


def gru_inputs_identity(tcd_manifest: dict, gcfg: dict, tcfg: dict) -> str:
    params = {k: v for k, v in gcfg.items() if k not in ("training", "encoder", "audit")}
    return artifacts.hash_json({"definition": GRU_INPUT_DEFINITION_VERSION, "params": params, "window_minutes": tcfg["window_minutes"], "tcd_feature_identity": tcd_manifest["identity"],
                                "lab_table": tcd_manifest["inputs"].get("lab", {}).get("size_mtime_ns")})


def _load_tcd(tcd_run_dir) -> dict:
    d = Path(tcd_run_dir)
    return {"patients": pd.read_csv(d / "patients.csv"), "schema": json.loads((d / "feature_schema.json").read_text(encoding="utf-8")),
            "cov": json.loads((d / "variable_coverage.json").read_text(encoding="utf-8")), "manifest": json.loads((d / "manifest.json").read_text(encoding="utf-8"))}


def run_gru_cutoff_audit(src, ids, window_end, variables, cov, tcfg, gcfg, values_main, mask_main, scaling) -> dict:
    """Same poison audit as the TCD features, for the sequences: re-read a random subset WITHOUT the window, corrupt every value after the patient's window, and require the
    binned+scaled arrays to be bit-identical with vs without corruption, identical to the reader-enforced path, and identical to the main run's rows."""
    n = min(int(gcfg["audit"]["n_patients"]), len(ids))
    pos = np.sort(np.random.default_rng(int(gcfg["audit"]["seed"])).choice(len(ids), size=n, replace=False))
    sub_ids, sub_we = ids[pos], window_end[pos]
    W, bm, nb = tcfg["window_minutes"], int(gcfg["bin_minutes"]), int(gcfg["n_bins"])
    raw, _ = read_lab_series(src, sub_ids, sub_we, variables, cov, tcfg, enforce=False, upper_minutes=float(gcfg["audit"]["horizon_minutes"]))
    enf, st = read_lab_series(src, sub_ids, sub_we, variables, cov, tcfg, enforce=True)
    poisoned, n_poison = {}, 0
    for k, (p, t, v) in raw.items():
        late = t > sub_we[p]
        n_poison += int(late.sum())
        poisoned[k] = (p, t, np.where(late, v * 1000.0 + 12345.0, v))

    def run(series):
        v, o = bin_series(series, sub_we, W, bm, nb, variables)
        return apply_scaling(v, o, scaling, variables), o
    va, oa = run(raw); vb, ob = run(poisoned); vc, oc = run(enf)
    same_ab = bool(np.array_equal(va, vb) and np.array_equal(oa, ob))
    same_ac = bool(np.array_equal(va, vc) and np.array_equal(oa, oc))
    match = bool(np.array_equal(va, np.asarray(values_main[pos])) and np.array_equal(oa, np.asarray(mask_main[pos]).astype(bool)))
    return {"n_patients": int(n), "rows_after_window_present_and_corrupted": int(n_poison), "rows_before_icu_ignored": int(st["rows_dropped_before_icu"]),
            "rows_after_window_dropped_by_reader": int(st["rows_dropped_after_window"]), "reader_max_offset_used_minutes": float(st["max_offset_used_minutes"]),
            "window_minutes": int(W), "arrays_identical_with_and_without_corruption": same_ab, "reader_and_clipping_agree": same_ac, "arrays_match_main_run": match,
            "passed": bool(same_ab and same_ac and match and n_poison > 0 and st["max_offset_used_minutes"] <= W)}


def build_gru_inputs(src: dict, tcd_run_dir, gcfg: dict, tcfg: dict, out_base, run_id: "str | None" = None, force: bool = False) -> dict:
    t0 = time.time()
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    tcd = _load_tcd(tcd_run_dir)
    identity = gru_inputs_identity(tcd["manifest"], gcfg, tcfg)
    if not force:
        hit = artifacts.find_reusable_run(out_base, identity)
        if hit:
            logger.info("[GRU_INPUTS] cache HIT: reusing verified run %s (identity %s)", hit["run_id"], identity[:12])
            return {"run_dir": hit["run_dir"], "run_id": hit["run_id"], "cache_hit": True, "manifest": hit}
    run_id, run_dir = artifacts.new_run_dir(out_base, "gruin", identity, run_id)
    logger.info("[GRU_INPUTS] cache MISS (identity %s) -> computing into %s", identity[:12], run_dir)
    pats = tcd["patients"]
    ids = pats["patientunitstayid"].to_numpy(np.int64)
    split = pats["split"].to_numpy()
    we = pats["window_end_minutes"].to_numpy(float)
    variables = list(tcd["schema"]["selected_variables"])
    W, bm, nb = tcfg["window_minutes"], int(gcfg["bin_minutes"]), int(gcfg["n_bins"])
    series, rstats = read_lab_series(src, ids, we, variables, tcd["cov"], tcfg, enforce=True)
    values_raw, observed = bin_series(series, we, W, bm, nb, variables)
    scaling = fit_scaling(values_raw, observed, split == "train", variables, gcfg["scaling"]["winsor_pct"])
    values = apply_scaling(values_raw, observed, scaling, variables)
    del values_raw
    mask = observed.astype(np.uint8)
    logger.info("[GRU_INPUTS] sequences %s (+ mask); observed cells %.2f%%", values.shape, 100.0 * observed.mean())

    audit = {"skipped": True, "reason": "gru.audit.n_patients = 0"}
    if int(gcfg["audit"]["n_patients"]) > 0:
        audit = run_gru_cutoff_audit(src, ids, we, variables, tcd["cov"], tcfg, gcfg, values, mask, scaling)
        logger.info("[GRU_INPUTS] cutoff audit: passed=%s (%d patients, %d late rows corrupted)", audit["passed"], audit["n_patients"], audit["rows_after_window_present_and_corrupted"])
        if not audit["passed"]:
            raise AssertionError(f"GRU cutoff audit FAILED: {audit}")

    per_var = []
    for j, k in enumerate(variables):
        row = {"variable": k}
        for sp in ("train", "val", "test"):
            m = split == sp
            row[f"observed_cell_fraction_{sp}"] = float(observed[m][:, :, j].mean()) if m.any() else None
            row[f"patients_with_any_{sp}"] = float(observed[m][:, :, j].any(axis=1).mean()) if m.any() else None
        per_var.append(row)
    report = {"definition_version": GRU_INPUT_DEFINITION_VERSION, "shape": {"values": list(values.shape), "mask": list(mask.shape), "bins": nb, "bin_minutes": bm, "channels": 2 * len(variables)},
              "patients": {sp: int((split == sp).sum()) for sp in ("train", "val", "test")} | {"total": int(len(ids)), "window_clipped_at_icu_exit": int((we < W).sum())},
              "fixed_sequence_length": nb, "sequence_length_derived_from_icu_exit": False,
              "observed_cell_fraction_overall": float(observed.mean()),
              "patients_with_no_observation_at_all": {sp: int((~observed[split == sp].any(axis=(1, 2))).sum()) for sp in ("train", "val", "test")},
              "variables": per_var, "read_statistics_lab": rstats, "cutoff_audit": audit, "tcd_feature_run": tcd["manifest"]["run_id"]}
    run_dir.mkdir(parents=True)
    artifacts.atomic_write(run_dir / "patients.csv", pats.to_csv(index=False))
    artifacts.atomic_save_npy(run_dir / "gru_values.npy", values)
    artifacts.atomic_save_npy(run_dir / "gru_mask.npy", mask)
    artifacts.atomic_write(run_dir / "variables.json", json.dumps(variables))
    artifacts.atomic_write(run_dir / "scaling.json", json.dumps(scaling, indent=1))
    artifacts.atomic_write(run_dir / "input_report.json", json.dumps(report, indent=1, default=str))
    artifacts.atomic_write(run_dir / "input_report.md", render_report_md(report, run_id, W))
    artifacts.atomic_write(run_dir / "cutoff_audit.json", json.dumps(audit, indent=1, default=str))
    artifacts.atomic_write(run_dir / "config_snapshot.json", json.dumps({"gru_inputs": {k: v for k, v in gcfg.items() if k not in ("training", "encoder")}}, indent=1))
    man = {"run_id": run_id, "definition_version": GRU_INPUT_DEFINITION_VERSION, "identity": identity, "files": {n: artifacts.sha256_file(run_dir / n) for n in INPUT_FILES},
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shapes": report["shape"], "counts": report["patients"], "tcd_feature_run": tcd["manifest"]["run_id"],
           "tcd_feature_identity": tcd["manifest"]["identity"], "cutoff_audit_passed": bool(audit.get("passed", False)), "cache": {"hit": False, "forced_recompute": bool(force)},
           "seconds": round(time.time() - t0, 1)}
    artifacts.atomic_write(run_dir / "manifest.json", json.dumps(man, indent=1))
    logger.info("[GRU_INPUTS] COMPLETE in %.1fs -> %s", time.time() - t0, run_dir)
    return {"run_dir": str(run_dir), "run_id": run_id, "cache_hit": False, "manifest": man, "report": report}


def render_report_md(rep: dict, run_id: str, window_minutes: int) -> str:
    sh, p = rep["shape"], rep["patients"]
    L = [f"# GRU input report -- {run_id}", "", f"Definition `{rep['definition_version']}`. {sh['bins']} bins x {sh['bin_minutes']} min = {sh['bins'] * sh['bin_minutes']} min = the first {window_minutes // 60} ICU hours; "
         f"{sh['channels']} channels (value + observation mask for {sh['channels'] // 2} lab variables). Sequence length is FIXED at {rep['fixed_sequence_length']} for every patient; "
         f"derived from the ICU exit: **{rep['sequence_length_derived_from_icu_exit']}**.", "", "## Patients", "", "| Split | Patients |", "|---|---|",
         f"| train | {p['train']} |", f"| val | {p['val']} |", f"| test | {p['test']} |", f"| total | {p['total']} |", "",
         f"Windows clipped at ICU exit: {p['window_clipped_at_icu_exit']} (their later bins are simply unobserved). Observed cells overall: {100 * rep['observed_cell_fraction_overall']:.2f}%. "
         f"Patients with NO observation at all: {rep['patients_with_no_observation_at_all']}.", "",
         "## Variables (fraction of patient-hour cells observed / patients with >=1 observation)", "", "| Variable | cells train | cells val | cells test | patients train | patients val | patients test |", "|---|---|---|---|---|---|---|"]
    for r in rep["variables"]:
        f = lambda x: "n/a" if x is None else f"{x:.4f}"   # noqa: E731
        L.append(f"| {r['variable']} | {f(r['observed_cell_fraction_train'])} | {f(r['observed_cell_fraction_val'])} | {f(r['observed_cell_fraction_test'])} | "
                 f"{f(r['patients_with_any_train'])} | {f(r['patients_with_any_val'])} | {f(r['patients_with_any_test'])} |")
    a = rep["cutoff_audit"]
    L += ["", "## Temporal cutoff evidence", ""]
    rs = rep["read_statistics_lab"]
    L.append(f"- lab rows read: {rs['rows_candidate']}; dropped AFTER the window: {rs['rows_dropped_after_window']}; dropped BEFORE ICU admission: {rs['rows_dropped_before_icu']}; max offset used: {rs['max_offset_used_minutes']} min")
    if a.get("skipped"):
        L.append(f"- poison audit skipped: {a['reason']}")
    else:
        L.append(f"- poison audit on {a['n_patients']} random patients: {a['rows_after_window_present_and_corrupted']} rows after the window were present and corrupted; arrays identical with vs without corruption = "
                 f"**{a['arrays_identical_with_and_without_corruption']}**; reader == clipping = {a['reader_and_clipping_agree']}; match the main run = {a['arrays_match_main_run']}; **passed = {a['passed']}**")
    return "\n".join(L) + "\n"


def load_gru_inputs(run_dir) -> dict:
    d = Path(run_dir)
    return {"patients": pd.read_csv(d / "patients.csv"), "values": np.load(d / "gru_values.npy", mmap_mode="r"), "mask": np.load(d / "gru_mask.npy", mmap_mode="r"),
            "variables": json.loads((d / "variables.json").read_text(encoding="utf-8")), "scaling": json.loads((d / "scaling.json").read_text(encoding="utf-8")),
            "manifest": json.loads((d / "manifest.json").read_text(encoding="utf-8"))}
