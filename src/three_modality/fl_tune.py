"""Validation-only tuning of the federated stages (`fl_tune`).

Every candidate is a COMPLETE federated training run on the fused embedding (scope as configured, default: every hospital with >= 1 training patient) scored by the federated validation
histogram AUROC of its best round. This module builds clients with `fl_clients.build_clients`, which exposes train/validation labels only, and never imports the evaluator: test labels and test
metrics are unreachable here by construction.

Search (one run per candidate, equal grid sizes across algorithms so no algorithm is favoured by tuning effort):
  1. local learning rate per family: default x {0.3, 1, 3} with FedAvg;
  2. FedProx mu in {0.01, 0.1, 1}, FedAdam server lr in {0.01, 0.03, 0.1} -- scored as the mean validation AUROC over the tuned families and tasks, at each family's best learning rate;
     SCAFFOLD has no extra hyper-parameter and is run once per (family, task) at the same learning rate;
  3. FedGBDT: max_depth in {3, 4, 6} x scale_pos_weight_mode in {none, sqrt}.
One hyper-parameter set per family (shared by the tasks and by all hospitals) is written to `best_hparams.json`; apply it with `--fl-hparams-run-id` on any FL stage. The per-candidate table is
`tuning_table.csv`; an interrupted tuning run resumes from `partial.json` with `--resume-existing-run`.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import logging
import time
from pathlib import Path

import numpy as np

from src.three_modality import artifacts
from src.three_modality.fl_algorithms import make_alphas, make_problems
from src.three_modality.fl_clients import build_clients
from src.three_modality.fl_config import FAMILIES, FL_DEFINITION_VERSION, NN_ALGORITHMS
from src.three_modality.fl_gbdt_adapter import train_fedgbdt
from src.three_modality.fl_server import train_federated
from src.three_modality.hw_data import Cohort, stable_seed

logger = logging.getLogger("eicu_fl")


def _grids(fl_cfg: dict, quick: bool) -> dict:
    g = fl_cfg["tuning"]["grids"]
    if quick:
        return {"lr_scale": [1.0], "mu": [float(fl_cfg["algorithms"]["fedprox"]["mu"])], "server_lr": [float(fl_cfg["algorithms"]["fedadam"]["server_lr"])],
                "gbdt_depth": [int(fl_cfg["gbdt"]["params"]["max_depth"])], "gbdt_spw": [fl_cfg["gbdt"]["params"]["scale_pos_weight_mode"]]}
    return {k: list(v) for k, v in g.items()}


def _cand_id(*parts) -> str:
    return "|".join(str(p) for p in parts)


def _nn_candidate(clients, family, ds_cfg, cfg, algo, graph_key, seed, round_hook=None) -> dict:
    t0 = time.time()
    problems, failed = make_problems(clients, family, ds_cfg, graph_key, False)
    alphas = make_alphas(problems, ds_cfg, cfg, False)
    res = train_federated(problems, family, ds_cfg, cfg, algo=algo, alphas=alphas, seed=seed, round_hook=round_hook, tag=f"tune/{family}/{algo}")
    return {"val_auroc_hist": res["val_auroc_hist_at_best"], "best_round": res["best_round"], "rounds_run": res["rounds_run"], "nonfinite_updates": res["nonfinite_updates"],
            "n_clients": res["n_clients"], "n_failed_clients": len(failed), "seconds": round(time.time() - t0, 1)}


def run_fl_tuning(cohort: Cohort, ds_cfg: dict, fl_cfg: dict, out_base, *, families=("lr", "mlp", "gcn", "gbdt"), algorithms=NN_ALGORITHMS, tasks=("mortality", "sepsis"),
                  max_hospitals=None, quick: bool = False, run_id: "str | None" = None, force: bool = False, resume: bool = False, round_hook=None) -> dict:
    t0 = time.time()
    grids = _grids(fl_cfg, quick)
    fams = [f for f in families if f in FAMILIES or f == "gbdt"]
    if len(fams) != len(families):
        raise ValueError(f"unknown tuning family in {families}; choose from {FAMILIES + ('gbdt',)}")
    ident = artifacts.hash_json({"definition": FL_DEFINITION_VERSION, "kind": "fl_tune", "cross_attention": [cohort.manifest["run_id"], cohort.manifest["identity"]], "families": fams,
                                 "algorithms": list(algorithms), "tasks": list(tasks), "max_hospitals": max_hospitals, "quick": quick, "grids": grids,
                                 "fl": {k: fl_cfg[k] for k in ("seed", "scope", "rounds", "local", "algorithms", "focal_alpha_mode", "validation", "gbdt")},
                                 "ds": {k: ds_cfg[k] for k in ("min_samples", "focal", "graph", "gcn", "classifiers")}})
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    if resume:
        d = out_base / (run_id or "")
        if not run_id or not (d / "run_state.json").is_file() or (d / "manifest.json").is_file():
            raise FileNotFoundError("--resume-existing-run needs the --run-id of an interrupted (not completed) tuning run")
        if json.loads((d / "run_state.json").read_text(encoding="utf-8"))["identity"] != ident:
            raise RuntimeError("the interrupted tuning run has a different identity; refusing to resume")
        rid, run_dir = run_id, d
    else:
        hit = None if force else artifacts.find_reusable_run(out_base, ident)
        if hit:
            return {"run_dir": hit["run_dir"], "run_id": hit["run_id"], "cache_hit": True, "best_hparams": json.loads((Path(hit["run_dir"]) / "best_hparams.json").read_text(encoding="utf-8"))}
        rid, run_dir = artifacts.new_run_dir(out_base, "fl_tune", ident, run_id)
        run_dir.mkdir(parents=True)
        artifacts.atomic_write(run_dir / "run_state.json", json.dumps({"identity": ident, "kind": "fl_tune"}, indent=1))
    partial_path = run_dir / "partial.json"
    cache = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.is_file() else {}
    table = []

    def run_cand(cid, task, family, algo, param, value, fn):
        if cid in cache:
            rec = cache[cid]
        else:
            try:
                rec = {**fn(), "status": "ok"}
            except Exception as exc:                       # noqa: BLE001 -- a failing candidate is recorded and disqualified, never silently dropped
                rec = {"val_auroc_hist": None, "status": f"failed: {type(exc).__name__}: {exc}"}
            cache[cid] = rec
            artifacts.atomic_write(partial_path, json.dumps(cache, indent=1, default=str))
        table.append({"task": task, "family": family, "algorithm": algo, "param": param, "value": value, **rec})
        return rec.get("val_auroc_hist")

    data = {t: build_clients(cohort, t, "fused", ds_cfg, fl_cfg["scope"], max_hospitals, fl_cfg["seed"])[0] for t in tasks}
    best = {"local": {"lr": {}}, "algorithms": {}, "gbdt": {"params": {}}}
    lr_scale_best, nn_fams = {}, [f for f in fams if f != "gbdt"]
    for fam in nn_fams:                                    # 1. local learning rate (FedAvg)
        scores = {}
        for sc in grids["lr_scale"]:
            vals = []
            for t in tasks:
                cfg = copy.deepcopy(fl_cfg)
                cfg["local"]["lr"][fam] = float(fl_cfg["local"]["lr"][fam]) * float(sc)
                vals.append(run_cand(_cand_id(t, fam, "fedavg", "lr_scale", sc), t, fam, "fedavg", "lr_scale", sc,
                                     lambda cfg=cfg, t=t, fam=fam: _nn_candidate(data[t], fam, ds_cfg, cfg, "fedavg", "A" if fam == "gcn" else None, stable_seed(fl_cfg["seed"], "tune", t, fam, "fedavg"), round_hook)))
            scores[sc] = float(np.mean([v for v in vals if v is not None])) if any(v is not None for v in vals) else float("-inf")
        lr_scale_best[fam] = max(scores, key=scores.get)
        best["local"]["lr"][fam] = float(fl_cfg["local"]["lr"][fam]) * float(lr_scale_best[fam])

    def cfg_at_best():
        c = copy.deepcopy(fl_cfg)
        for fam, v in best["local"]["lr"].items():
            c["local"]["lr"][fam] = v
        return c

    def tune_algo(algo, param, values, setter):            # 2. algorithm hyper-parameter, scored as the mean over families and tasks
        scores = {}
        for v in values:
            vals = []
            for fam in nn_fams:
                for t in tasks:
                    cfg = cfg_at_best()
                    setter(cfg, v)
                    vals.append(run_cand(_cand_id(t, fam, algo, param, v), t, fam, algo, param, v,
                                         lambda cfg=cfg, t=t, fam=fam: _nn_candidate(data[t], fam, ds_cfg, cfg, algo, "A" if fam == "gcn" else None, stable_seed(fl_cfg["seed"], "tune", t, fam, algo), round_hook)))
            ok = [x for x in vals if x is not None]
            scores[v] = float(np.mean(ok)) if ok else float("-inf")
        return max(scores, key=scores.get)

    if nn_fams and "fedprox" in algorithms:
        best["algorithms"]["fedprox"] = {"mu": float(tune_algo("fedprox", "mu", grids["mu"], lambda c, v: c["algorithms"]["fedprox"].__setitem__("mu", float(v))))}
    if nn_fams and "fedadam" in algorithms:
        best["algorithms"]["fedadam"] = {"server_lr": float(tune_algo("fedadam", "server_lr", grids["server_lr"], lambda c, v: c["algorithms"]["fedadam"].__setitem__("server_lr", float(v))))}
    if nn_fams and "scaffold" in algorithms:
        for fam in nn_fams:
            for t in tasks:
                run_cand(_cand_id(t, fam, "scaffold", "-", "-"), t, fam, "scaffold", "-", "-",
                         lambda t=t, fam=fam: _nn_candidate(data[t], fam, ds_cfg, cfg_at_best(), "scaffold", "A" if fam == "gcn" else None, stable_seed(fl_cfg["seed"], "tune", t, fam, "scaffold"), round_hook))
    if "gbdt" in fams:                                     # 3. FedGBDT
        scores = {}
        for depth in grids["gbdt_depth"]:
            for spw in grids["gbdt_spw"]:
                vals = []
                for t in tasks:
                    g = copy.deepcopy(fl_cfg["gbdt"])
                    g["params"]["max_depth"], g["params"]["scale_pos_weight_mode"] = int(depth), spw
                    vals.append(run_cand(_cand_id(t, "gbdt", "fedgbdt", "depth_spw", f"{depth}/{spw}"), t, "gbdt", "fedgbdt", "depth_spw", f"{depth}/{spw}",
                                         lambda g=g, t=t: (lambda r: {"val_auroc_hist": r["val_auroc_hist_at_best"], "best_round": r["best_iteration"], "rounds_run": r["n_rounds_run"],
                                                                      "seconds": r["seconds"], "n_clients": r["n_clients"]})(train_fedgbdt(data[t], g, fl_cfg["validation"], stable_seed(fl_cfg["seed"], "tune", t, "gbdt")))))
                ok = [x for x in vals if x is not None]
                scores[(depth, spw)] = float(np.mean(ok)) if ok else float("-inf")
        d_best, s_best = max(scores, key=scores.get)
        best["gbdt"]["params"] = {"max_depth": int(d_best), "scale_pos_weight_mode": s_best}
    buf = io.StringIO()
    cols = ["task", "family", "algorithm", "param", "value", "val_auroc_hist", "best_round", "rounds_run", "nonfinite_updates", "n_clients", "n_failed_clients", "seconds", "status"]
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(table)
    artifacts.atomic_write(run_dir / "tuning_table.csv", buf.getvalue())
    artifacts.atomic_write(run_dir / "best_hparams.json", json.dumps(best, indent=1))
    manifest = {"run_id": rid, "definition_version": FL_DEFINITION_VERSION, "kind": "fl_tune", "identity": ident, "cross_attention_run": cohort.manifest["run_id"], "families": fams,
                "algorithms": list(algorithms), "tasks": list(tasks), "quick": quick, "partial": bool(max_hospitals), "max_hospitals": max_hospitals, "grids": grids,
                "selection": "validation only: federated validation-histogram AUROC of each candidate's best round; test labels and test metrics are unreachable from this module",
                "scope": fl_cfg["scope"], "n_candidates": len(table), "n_failed_candidates": sum(1 for r in table if str(r["status"]).startswith("failed")), "best_hparams": best,
                "lr_scale_selected": lr_scale_best, "shared_across": "tasks and hospitals (one set per family, disclosed)", "seconds": round(time.time() - t0, 1),
                "files": {n: artifacts.sha256_file(run_dir / n) for n in ("tuning_table.csv", "best_hparams.json")}}
    artifacts.atomic_write(run_dir / "manifest.json", json.dumps(manifest, indent=1, default=str))
    return {"run_dir": str(run_dir), "run_id": rid, "cache_hit": False, "best_hparams": best, "manifest": manifest}


def load_fl_hparams(base, run_id: str) -> dict:
    d = Path(base) / run_id
    m = d / "manifest.json"
    if not m.is_file():
        raise FileNotFoundError(f"FL tuning run {run_id!r} not found or incomplete under {base}")
    man = json.loads(m.read_text(encoding="utf-8"))
    for n, h in man["files"].items():
        if artifacts.sha256_file(d / n) != h:
            raise ValueError(f"FL tuning run {run_id}: {n} does not match its recorded hash")
    return json.loads((d / "best_hparams.json").read_text(encoding="utf-8"))
