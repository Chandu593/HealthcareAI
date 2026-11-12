"""Hyper-parameter tuning for the hospital-wise models (stage `nonfl_tune`) -- never touches test data.

Why not rank recipes on the validation set? Each recipe already uses that validation set to pick its own checkpoint (and some recipes pick it by validation AUROC), so validation AUROC is
optimistically biased toward exactly those recipes. Instead every tuning hospital's TRAINING patients are split (stratified, deterministic) into a fitting part (70%) and a held-out
part (30%): the recipe is fitted on the fitting part, its checkpoint is chosen on the hospital's validation patients exactly as in the real runs, and the recipe is SCORED on the held-out
part of train. Test patients (features and labels) are never loaded. The winner per model family is the recipe with the highest macro held-out AUROC over the tuning hospitals and both
tasks (ties -> held-out AUPRC -> recipe order); recipe 0 is always the current defaults, so tuning can only replace them with something that scored strictly better.

The chosen hyper-parameters are ONE set shared by all hospitals, selected once at design time from training/validation data of a stratified sample of hospitals (disclosed in the manifest).
"""
from __future__ import annotations

import copy
import csv
import io
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.three_modality import artifacts
from src.three_modality.downstream_config import apply_hparams
from src.three_modality.hw_classifiers import fit_predict
from src.three_modality.hw_data import Cohort, plan_hospitals, stable_seed
from src.three_modality.hw_graph import train_gcn

logger = logging.getLogger("eicu_fl")

TUNE_DEFINITION_VERSION = "tune-v2"      # v2: per-hospital failures are counted (never fatal), unstable recipes are disqualified, completed recipes are cached so a crashed run resumes
FAMILIES = ("lr", "mlp", "lightgbm", "gbdt", "gcn")
HOLDOUT_FRACTION = 0.3
MAX_FAILURE_RATE = 0.10          # a recipe that errors on more than 10% of its hospital-tasks is unstable and can never win


def default_grids() -> dict:
    """Recipe lists per family. Entry 0 is always `{}` = the current defaults. Overrides are merged over `downstream.classifiers.<family>` / `downstream.gcn` / `downstream.graph`."""
    auroc = {"selection_metric": "auroc", "min_epochs": 10}
    lr = [{}] + [({"weight_decay": wd, **sel}) for wd in (1e-3, 1e-2, 1e-1, 1.0) for sel in ({"selection_metric": "loss", "min_epochs": 10}, auroc)]
    lr += [{"weight_decay": 1e-2, "gamma": 0.0, **auroc}, {"weight_decay": 1e-2, "gamma": 1.0, **auroc}]
    mlp = [{}] + [{"hidden": h, "dropout": dp, "weight_decay": wd, **auroc} for h, dp, wd in itertools.product((32, 64), (0.1, 0.3), (1e-4, 1e-2))]
    lgb = [{}] + [{"num_leaves": nl, "min_child_samples": mc, "learning_rate": 0.05, "selection_metric": sm} for (nl, mc), sm in itertools.product(((7, 5), (7, 20), (15, 20), (4, 20)), ("loss", "auroc"))]
    gbdt = [{}] + [{"max_depth": md, "learning_rate": lrt, "selection_metric": sm} for md, lrt, sm in itertools.product((2, 3), (0.03, 0.05), ("loss", "auroc")) if (md, lrt, sm) != (3, 0.05, "loss")]
    gcn = [{}] + [{"hidden": h, "dropout": dp, "weight_decay": wd, **auroc} for h, dp, wd in ((32, 0.2, 1e-4), (64, 0.5, 1e-4), (32, 0.5, 1e-2), (64, 0.2, 1e-2), (64, 0.2, 1e-4))]
    graph = [{}, {"k": 5}, {"cosine_threshold": 0.3}, {"k": 20}, {"k": 20, "cosine_threshold": 0.3}]
    return {"lr": lr, "mlp": mlp, "lightgbm": lgb, "gbdt": gbdt, "gcn": gcn, "graph": graph}


def tuning_hospitals(cohort: Cohort, task: str, ms: dict, n: int) -> list:
    """Trainable hospitals for the task, spread evenly over the size ranking (so tiny and large hospitals are both represented). Deterministic; uses only counts."""
    ok = sorted([d for d in plan_hospitals(cohort, task) if d.skip_reason(ms) is None], key=lambda d: (d.counts["train"]["n"], d.hospitalid))
    if not ok:
        return []
    idx = sorted(set(np.linspace(0, len(ok) - 1, min(n, len(ok))).round().astype(int).tolist()))
    return [ok[i] for i in idx]


def split_train_holdout(d, seed: int):
    """Stratified 70/30 split of the hospital's TRAINING rows (labels = training labels only). Returns (fit_idx, holdout_idx) as positions into d.rows['train'], or None when the held-out
    part cannot be scored (a class missing there) or the fitting part loses a class."""
    y = d.labels("train")
    try:
        fit, ho = train_test_split(np.arange(len(y)), test_size=HOLDOUT_FRACTION, random_state=seed, stratify=y)
    except ValueError:
        return None
    if len(np.unique(y[fit])) < 2 or len(np.unique(y[ho])) < 2:
        return None
    return np.sort(fit), np.sort(ho)


def _score(y_ho, p_ho):
    return float(roc_auc_score(y_ho, p_ho)), float(average_precision_score(y_ho, p_ho))


def _evaluate_recipe(family: str, key: "str | None", cohort_hosp: dict, base_cfg: dict, overrides: dict, graph_overrides: dict, seed0: int) -> dict:
    """Mean held-out AUROC / AUPRC of one recipe over the tuning hospitals of both tasks. Returns {"auroc", "auprc", "n", "per_task": {...}}."""
    if family == "gcn":
        cfg = apply_hparams(base_cfg, {"gcn": overrides, "graph": graph_overrides})
    elif family == "graph":
        cfg = apply_hparams(base_cfg, {"graph": overrides})
    else:
        cfg = apply_hparams(base_cfg, {"classifiers": {family: overrides}})
    per_task, allv, n_failed, first_error = {}, [], 0, None
    for task, hosp in cohort_hosp.items():
        vals, t_failed = [], 0
        for d in hosp:
            seed = stable_seed(seed0, d.hospitalid, task, "tune")
            sp = split_train_holdout(d, seed)
            if sp is None:
                continue
            fit, ho = sp
            Xtr, ytr = d.X("train"), d.labels("train")
            Xva, yva = d.X("val"), d.labels("val")                            # validation: checkpoint selection only (as in the real runs)
            try:
                if family in ("gcn", "graph"):
                    keys = (key,) if key else ("A", "B")
                    ps = [train_gcn(Xtr[fit], ytr[fit], Xva, yva, Xtr[ho], k, cfg, seed)["p_test"] for k in keys]
                    aucs = [_score(ytr[ho], q) for q in ps]
                    vals.append((float(np.mean([a for a, _ in aucs])), float(np.mean([b for _, b in aucs]))))
                else:
                    out = fit_predict(family, Xtr[fit], ytr[fit], Xva, yva, Xtr[ho], cfg, seed)      # the "test" slot receives the HELD-OUT TRAIN rows; real test rows are never loaded
                    vals.append(_score(ytr[ho], out["p_test"]))
            except Exception as exc:          # noqa: BLE001  one hospital's failure is counted against the recipe, never fatal to the tuning run
                t_failed += 1
                first_error = first_error or f"{type(exc).__name__}: {str(exc)[:160]}"
        n_failed += t_failed
        per_task[task] = {"auroc": float(np.mean([v[0] for v in vals])) if vals else None, "auprc": float(np.mean([v[1] for v in vals])) if vals else None, "n_hospitals": len(vals), "n_failed": t_failed}
        allv += vals
    tasks_ok = [t for t in per_task.values() if t["auroc"] is not None]
    disq = n_failed > MAX_FAILURE_RATE * (len(allv) + n_failed)
    return {"auroc": float(np.mean([t["auroc"] for t in tasks_ok])) if tasks_ok else None, "auprc": float(np.mean([t["auprc"] for t in tasks_ok])) if tasks_ok else None,
            "n": len(allv), "n_failed": n_failed, "disqualified": bool(disq), "first_error": first_error, "per_task": per_task}


def run_tuning(cohort: Cohort, ds_cfg: dict, out_base, *, families=FAMILIES, tasks=("mortality", "sepsis"), n_hospitals: int = 60, quick: bool = False, grids: "dict | None" = None,
               run_id: "str | None" = None, force: bool = False) -> dict:
    t0 = time.time()
    for f in families:
        if f not in FAMILIES:
            raise ValueError(f"unknown family {f!r}; choose from {FAMILIES}")
    grids = copy.deepcopy(grids or default_grids())
    if quick:
        grids = {k: v[:2] for k, v in grids.items()}
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    ident = artifacts.hash_json({"definition": TUNE_DEFINITION_VERSION, "cross_attention": [cohort.manifest["run_id"], cohort.manifest["identity"]], "families": list(families), "tasks": list(tasks),
                                 "n_hospitals": n_hospitals, "grids": grids, "cfg": {k: ds_cfg[k] for k in ("seed", "min_samples", "focal", "graph", "gcn", "classifiers")}})
    if not force:
        hit = artifacts.find_reusable_run(out_base, ident)
        if hit:
            logger.info("[TUNE] cache HIT: reusing verified run %s", hit["run_id"])
            return {"run_dir": hit["run_dir"], "run_id": hit["run_id"], "cache_hit": True, "manifest": hit,
                    "best_hparams": json.loads((Path(hit["run_dir"]) / "best_hparams.json").read_text(encoding="utf-8"))}
    rid, run_dir = artifacts.new_run_dir(out_base, "tune", ident, run_id)
    logger.info("[TUNE] families %s | tasks %s | up to %d hospitals/task | cache MISS (identity %s) -> %s", list(families), list(tasks), n_hospitals, ident[:12], run_dir)
    hosp = {t: tuning_hospitals(cohort, t, ds_cfg["min_samples"], n_hospitals if "gcn" not in families else n_hospitals) for t in tasks}
    gcn_hosp = {t: h[:: max(1, int(np.ceil(len(h) / max(1, min(40, n_hospitals)))))] for t, h in hosp.items()}       # GCN recipes are ~10x costlier: at most ~40 hospitals per task
    table, best = [], {"classifiers": {}}
    seed0 = int(ds_cfg["seed"])
    partial = out_base / f"_partial_{ident[:16]}.json"                                   # finished recipes survive a crash / Ctrl-C and are reused on the next identical run
    cache = {} if (force or not partial.is_file()) else json.loads(partial.read_text(encoding="utf-8"))

    def cached(k, fn):
        if k in cache:
            logger.info("[TUNE] %s: reused from the interrupted run", k)
            return cache[k]
        r = fn()
        cache[k] = r
        artifacts.atomic_write(partial, json.dumps(cache))
        return r

    for fam in [f for f in families if f != "gcn"]:
        scores = []
        for i, ov in enumerate(grids[fam]):
            r = cached(f"{fam}:{i}", lambda fam=fam, ov=ov: _evaluate_recipe(fam, None, hosp, ds_cfg, ov, {}, seed0))
            scores.append(r)
            table.append({"family": fam, "recipe": f"{fam}_{i:02d}", "overrides": ov, **{k: r[k] for k in ("auroc", "auprc", "n", "n_failed", "disqualified", "first_error")}, "per_task": r["per_task"]})
            logger.info("[TUNE] %s recipe %d/%d holdout AUROC %s (n=%d, failed=%d%s)", fam, i + 1, len(grids[fam]), None if r["auroc"] is None else round(r["auroc"], 4), r["n"], r["n_failed"], ", DISQUALIFIED: " + str(r["first_error"]) if r["disqualified"] else "")
        j = _argbest(scores)
        best["classifiers"][fam] = grids[fam][j]
    if "gcn" in families:
        scores = []
        for i, ov in enumerate(grids["gcn"]):
            r = cached(f"gcn:{i}", lambda ov=ov: _evaluate_recipe("gcn", "A", gcn_hosp, ds_cfg, ov, {}, seed0))             # recipes ranked on Graph A (the stronger graph in the real runs), applied to both
            scores.append(r)
            table.append({"family": "gcn", "recipe": f"gcn_{i:02d}", "overrides": ov, **{k: r[k] for k in ("auroc", "auprc", "n", "n_failed", "disqualified", "first_error")}, "per_task": r["per_task"]})
            logger.info("[TUNE] gcn recipe %d/%d holdout AUROC %s (n=%d, failed=%d%s)", i + 1, len(grids["gcn"]), None if r["auroc"] is None else round(r["auroc"], 4), r["n"], r["n_failed"], ", DISQUALIFIED: " + str(r["first_error"]) if r["disqualified"] else "")
        best["gcn"] = grids["gcn"][_argbest(scores)]
        gscores = []
        for i, gov in enumerate(grids["graph"]):
            r = cached(f"graph:{i}", lambda gov=gov: _evaluate_recipe("graph", None, gcn_hosp, apply_hparams(ds_cfg, {"gcn": best["gcn"]}), gov, {}, seed0))      # graph k / threshold, averaged over Graph A and B
            gscores.append(r)
            table.append({"family": "graph", "recipe": f"graph_{i:02d}", "overrides": gov, **{k: r[k] for k in ("auroc", "auprc", "n", "n_failed", "disqualified", "first_error")}, "per_task": r["per_task"]})
            logger.info("[TUNE] graph recipe %d/%d holdout AUROC %s (n=%d, failed=%d%s)", i + 1, len(grids["graph"]), None if r["auroc"] is None else round(r["auroc"], 4), r["n"], r["n_failed"], ", DISQUALIFIED: " + str(r["first_error"]) if r["disqualified"] else "")
        best["graph"] = grids["graph"][_argbest(gscores)]
    best = {k: v for k, v in best.items() if v}
    apply_hparams(ds_cfg, best)                                                            # must validate
    run_dir.mkdir(parents=True)
    artifacts.atomic_write(run_dir / "best_hparams.json", json.dumps(best, indent=1))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["family", "recipe", "overrides", "holdout_auroc", "holdout_auprc", "n_hospital_tasks", "n_failed", "disqualified", "first_error", "per_task"])
    for t in table:
        w.writerow([t["family"], t["recipe"], json.dumps(t["overrides"], sort_keys=True), t["auroc"], t["auprc"], t["n"], t["n_failed"], t["disqualified"], t["first_error"], json.dumps(t["per_task"])])
    artifacts.atomic_write(run_dir / "tuning_table.csv", buf.getvalue())
    man = {"run_id": rid, "definition_version": TUNE_DEFINITION_VERSION, "identity": ident, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "cross_attention_run": cohort.manifest["run_id"], "families": list(families), "tasks": list(tasks), "quick": bool(quick), "n_hospitals_requested": n_hospitals,
           "tuning_hospitals": {t: [d.hospitalid for d in h] for t, h in hosp.items()}, "gcn_tuning_hospitals": {t: [d.hospitalid for d in h] for t, h in gcn_hosp.items()},
           "method": f"recipes fitted on {int((1 - HOLDOUT_FRACTION) * 100)}% of each hospital's TRAINING patients, checkpoint chosen on its VALIDATION patients, scored on the held-out {int(HOLDOUT_FRACTION * 100)}% of train",
           "test_data": "never loaded: no test features, no test labels",
           "scope": "one hyper-parameter set per model family, shared by all hospitals, chosen once at design time from training/validation data of a stratified hospital sample; "
                    "tuned on the fused embedding and reused for the individual-modality embeddings",
           "best_hparams": best, "files": {n: artifacts.sha256_file(run_dir / n) for n in ("best_hparams.json", "tuning_table.csv")}, "seconds": round(time.time() - t0, 1)}
    artifacts.atomic_write(run_dir / "manifest.json", json.dumps(man, indent=1))
    partial.unlink(missing_ok=True)
    logger.info("[TUNE] COMPLETE in %.1fs -> %s", time.time() - t0, run_dir)
    return {"run_dir": str(run_dir), "run_id": rid, "cache_hit": False, "manifest": man, "best_hparams": best, "table": table}


def _argbest(results: list) -> int:
    """Highest held-out AUROC; ties -> AUPRC; then the earlier recipe (recipe 0 = the defaults wins ties). Recipes with no score, or disqualified for erroring on > 10% of hospital-tasks, never win."""
    best_i, best_key = 0, (-1.0, -1.0)
    for i, r in enumerate(results):
        if r["auroc"] is None or r.get("disqualified"):
            continue
        key = (round(r["auroc"], 6), round(r["auprc"] or 0.0, 6))
        if key > best_key:
            best_i, best_key = i, key
    return best_i


def load_best_hparams(tuning_base, run_id: str) -> dict:
    d = Path(tuning_base) / run_id
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    for n, h in man["files"].items():
        if artifacts.sha256_file(d / n) != h:
            raise ValueError(f"tuning run {run_id}: {n} does not match its recorded SHA-256 (modified file)")
    return json.loads((d / "best_hparams.json").read_text(encoding="utf-8"))
