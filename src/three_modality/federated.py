"""FL regime, Configs 1-4 and the personalised experiments (federated DOWNSTREAM learning on frozen, centrally pretrained representations).

  Config 1 : fused or individual embeddings + federated LR / MLP (FedAvg, FedProx, FedAdam, SCAFFOLD) and FedGBDT;
  Config 2 : hospital-local Graph A / Graph B + federated GCN (the graphs never leave their hospital; only GCN parameters are exchanged);
  Config 3 : Config 2 + hospital-local, TRAINING-ONLY graph SMOTE (synthetic nodes live only inside the owning client's training graph);
  Config 4 : stage 1 = federated GCN (one algorithm, `c4_gcn_algorithm`); stage 2 = every client embeds its own patients with the final global GCN and the downstream classifiers
             (federated LR / MLP, FedGBDT) are trained federatedly on those hospital-local embeddings;
  Personalised (separate experiment): FedPer, FedRep (shared body + hospital-specific heads, MLP / GCN) and FedAvg followed by local fine-tuning.

This is NOT end-to-end federated learning: the three encoders and the three-token cross-attention that produced the embeddings were trained centrally on pooled training patients.

Artifacts per run (additive, `<out>/federated/<run_id>/`): per_hospital.json/csv, summary.json, pooled_summary.json (patient-bootstrap CIs and hospital-bootstrap macro CIs), classification_reports.json/md,
per_hospital_classification_reports.csv, predictions.npz (same layout as the Non-FL regime), training_logs/<variant>.json (round history, selected round, stop reason, histogram-validation bound,
communication, runtime), checkpoints/<variant>/{final.pt|final.pkl}, manifest.json, run_state.json.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from src.three_modality import artifacts
from src.three_modality.downstream_config import EMBEDDING_SETS, GRAPHS, TASKS
from src.three_modality.fl_algorithms import build_model, make_alphas, make_problems, set_params
from src.three_modality.fl_clients import build_clients
from src.three_modality.fl_config import (FL_DEFINITION_VERSION, FL_REGIME, FL_CLASSIFIERS, NN_ALGORITHMS, PERSONALIZATION_METHODS, PERSONALIZED_MODELS)
from src.three_modality.fl_eval import class_reports, evaluate_variant, summarise_variant
from src.three_modality.fl_gbdt_adapter import train_fedgbdt
from src.three_modality.fl_server import HistAccumulator, client_params, exact_val_auroc_diag, finetune_clients, predict_logits, train_federated
from src.three_modality.hw_data import Cohort, SPLITS, stable_seed
from src.three_modality.hw_reports import PredictionStore, class_report_markdown, per_hospital_class_rows, pooled_metrics

logger = logging.getLogger("eicu_fl")

CONFIGS = (1, 2, 3, 4)
HIST_NOTE = ("Federated validation: clients send per-class histograms of validation logits (bins and range in the manifest); the global validation AUROC is computed from the summed histograms "
             "and treats same-bin positive/negative pairs as ties, so |error| <= 0.5 x the tied-pair fraction (recorded per round as val_auroc_tie_bound; exact pooled AUROC at the selected "
             "round is stored as a diagnostic only).")

_CSV_COLS = ["variant", "config", "task", "embedding_set", "graph", "classifier", "sampling", "algorithm", "method", "hospitalid", "eval_set", "nonfl_eligible", "participated_in_training",
             "status", "skip_reason", "threshold", "threshold_source", "head_source", "train_n", "train_pos", "train_neg", "val_n", "val_pos", "val_neg", "test_n", "test_pos", "test_neg",
             "auroc", "auprc", "accuracy", "precision", "recall", "f1", "specificity", "tn", "fp", "fn", "tp", "rt_threshold", "rt_precision", "rt_recall", "rt_f1", "rt_accuracy",
             "rt_specificity", "seconds"]


# ---------------------------------------------------------------------------------------------------------------------------------
# specs / identity
# ---------------------------------------------------------------------------------------------------------------------------------
def fl_variant_specs(config: int, embedding_sets, classifiers, graphs, tasks, algorithms, c4_gcn_algorithm: str = "fedavg") -> list:
    for a in algorithms:
        if a not in NN_ALGORITHMS:
            raise ValueError(f"unknown algorithm {a!r}; parameter-averaging algorithms are {NN_ALGORITHMS} (trees use classifier 'gbdt_fed')")
    for c in classifiers:
        if c not in FL_CLASSIFIERS:
            raise ValueError(f"unknown FL classifier {c!r}; choose from {FL_CLASSIFIERS}")
    out = []
    for task in tasks:
        if config == 1:
            for es in embedding_sets:
                for clf in classifiers:
                    if clf == "gbdt_fed":
                        out.append({"variant": f"f1__{es}__{task}__gbdt_fed__fedgbdt", "config": 1, "task": task, "embedding_set": es, "classifier": "gbdt_fed", "graph": None, "sampling": "none",
                                    "algorithm": "fedgbdt", "method": "global"})
                    else:
                        out += [{"variant": f"f1__{es}__{task}__{clf}__{a}", "config": 1, "task": task, "embedding_set": es, "classifier": clf, "graph": None, "sampling": "none",
                                 "algorithm": a, "method": "global"} for a in algorithms]
        elif config in (2, 3):
            out += [{"variant": f"f{config}__graph{g}__{task}__{a}", "config": config, "task": task, "embedding_set": "fused", "classifier": "gcn", "graph": g,
                     "sampling": "graph_smote" if config == 3 else "none", "algorithm": a, "method": "global"} for g in graphs for a in algorithms]
        elif config == 4:
            for g in graphs:
                for clf in classifiers:
                    if clf == "gbdt_fed":
                        out.append({"variant": f"f4__graph{g}__{task}__gbdt_fed__fedgbdt", "config": 4, "task": task, "embedding_set": "fused", "classifier": "gbdt_fed", "graph": g,
                                    "sampling": "none", "algorithm": "fedgbdt", "method": "global", "stage1_algorithm": c4_gcn_algorithm})
                    else:
                        out += [{"variant": f"f4__graph{g}__{task}__{clf}__{a}", "config": 4, "task": task, "embedding_set": "fused", "classifier": clf, "graph": g, "sampling": "none",
                                 "algorithm": a, "method": "global", "stage1_algorithm": c4_gcn_algorithm} for a in algorithms]
        else:
            raise ValueError(f"config must be one of {CONFIGS}")
    return out


def fl_identity(kind: str, ca_manifest: dict, ds_cfg: dict, fl_cfg: dict, specs: list, scope: str, max_hospitals, hparams_source=None) -> str:
    rel = {"min_samples": ds_cfg["min_samples"], "focal": ds_cfg["focal"], "graph": ds_cfg["graph"], "gcn": ds_cfg["gcn"], "smote": ds_cfg["smote"], "mlp": ds_cfg["classifiers"]["mlp"],
           "operating_points": ds_cfg["operating_points"]}
    fl = {k: fl_cfg[k] for k in ("seed", "rounds", "local", "algorithms", "focal_alpha_mode", "scaler", "validation", "gbdt", "personalization")}
    return artifacts.hash_json({"definition": FL_DEFINITION_VERSION, "kind": kind, "cross_attention": [ca_manifest["run_id"], ca_manifest["identity"]], "variants": [s["variant"] for s in specs],
                                "ds": rel, "fl": fl, "scope": scope, "max_hospitals": max_hospitals, "hparams_source": hparams_source})


# ---------------------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------------------
def _nn_scorer(res: dict, problems: dict, family: str, ds_cfg: dict, seed: int, personalised: str = None, ft_info: "dict | None" = None):
    model = build_model(family, next(iter(problems.values())).client.dim, ds_cfg, seed)

    def scorer(client):
        pr = problems[client.hospitalid]
        out = {"p_val": expit(predict_logits(res, model, pr, "val")) if client.n_val else np.empty(0), "p_test": expit(predict_logits(res, model, pr, "test"))}
        if personalised in ("fedper", "fedrep"):
            out["head_source"] = "local_head" if client.hospitalid in (res["heads"] or {}) else "mean_head_fallback"
        elif personalised == "finetune":
            out["head_source"] = (ft_info or {}).get(client.hospitalid, {}).get("source", "global_fallback")
            out["info"] = (ft_info or {}).get(client.hospitalid)
        return out
    return scorer, model


def _global_threshold(res: dict, fl_cfg: dict) -> float:
    h = HistAccumulator(fl_cfg["validation"]["bins"], fl_cfg["validation"]["logit_range"])
    h.pos, h.neg = np.asarray(res["val_pos"]).copy(), np.asarray(res["val_neg"]).copy()
    return h.best_f1_probability_threshold()


def _train_log(spec: dict, res: dict, extra: "dict | None" = None) -> dict:
    keys = ("algo", "method", "family", "best_round", "best_val_score", "rounds_run", "stop_reason", "val_auroc_hist_at_best", "val_auroc_tie_bound_at_best", "val_auroc_exact_diag", "val_focal_loss",
            "nonfinite_updates", "n_clients", "n_params", "comm_mb_total", "comm_mb_per_round", "local_steps_effective", "seconds", "resumed_from", "head_clients")
    log = {"spec": spec, **{k: res.get(k) for k in keys}, "history": res.get("history")}
    if extra:
        log.update(extra)
    return log


def _gbdt_log(spec: dict, res: dict) -> dict:
    keys = ("best_iteration", "n_rounds_run", "n_trees", "val_auroc_hist_at_best", "val_auroc_tie_bound_at_best", "val_auroc_exact_diag", "n_clients", "n_train", "seconds",
            "comm_gb_total_sparse_all_clients", "scale_pos_weight", "params", "resumed_from")
    return {"spec": spec, **{k: res.get(k) for k in keys}, "val_curve": (res.get("history") or {}).get("val_curve")}


def _csv(rows: list) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_CSV_COLS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        m = r["metrics"] or {}
        flat = {k: r.get(k) for k in ("variant", "config", "task", "embedding_set", "graph", "classifier", "sampling", "algorithm", "method", "hospitalid", "eval_set", "nonfl_eligible",
                                      "participated_in_training", "status", "skip_reason", "threshold", "threshold_source", "head_source", "seconds")}
        for s in SPLITS:
            flat.update({f"{s}_n": r["counts"][s]["n"], f"{s}_pos": r["counts"][s]["pos"], f"{s}_neg": r["counts"][s]["neg"]})
        flat.update({k: m.get(k) for k in ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "specificity")})
        cm = m.get("confusion_matrix") or {}
        flat.update({k: cm.get(k) for k in ("tn", "fp", "fn", "tp")})
        rt = ((r.get("operating_points") or {}).get("recall_target")) or {}
        flat["rt_threshold"] = rt.get("threshold")
        flat.update({f"rt_{k}": (rt.get("metrics") or {}).get(k) for k in ("precision", "recall", "f1", "accuracy", "specificity")})
        w.writerow(flat)
    return buf.getvalue()


def macro_ci(rows: list, metric: str, n_boot: int, seed: int) -> "list | None":
    """95% CI of the hospital-macro mean of `metric` by bootstrapping HOSPITALS (rows with an undefined metric are excluded)."""
    vals = np.array([r["metrics"][metric] for r in rows if r.get("status") == "trained" and r.get("metrics") and r["metrics"].get(metric) is not None], dtype=float)
    if len(vals) < 3:
        return None
    rs = np.random.default_rng(seed)
    means = [vals[rs.integers(0, len(vals), len(vals))].mean() for _ in range(int(n_boot))]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _pooled_summary(variant_rows: list, store: PredictionStore, variant: str, ds_cfg: dict) -> dict:
    rep = ds_cfg["reporting"]
    arrs = store.arrays()
    pred = {k: arrs[f"{variant}@{k}"] for k in ("pid", "hid", "y", "p")} if f"{variant}@p" in arrs else None
    out = {}
    ev = [r for r in variant_rows if r["status"] == "trained"]
    thr = {r["hospitalid"]: r["threshold"] for r in ev}
    paired_ids = [r["hospitalid"] for r in ev if r["eval_set"] == "paired"]
    for name, ids in (("paired_nonfl", paired_ids), ("all_evaluated_clients", [r["hospitalid"] for r in ev])):
        sub = [r for r in ev if r["hospitalid"] in set(ids)]
        out[name] = {"pooled": pooled_metrics(pred, thr, ids, n_boot=int(rep["n_bootstrap"]), seed=int(rep["seed"])) if pred is not None and ids else None,
                     "macro_ci95": {m: macro_ci(sub, m, int(rep["n_bootstrap"]), int(rep["seed"])) for m in ("auroc", "auprc", "f1", "recall", "precision")}}
    return out


def _prepare_run(kind: str, out_base, ident: str, run_id, force: bool, resume: bool):
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    if not force and not resume:
        hit = artifacts.find_reusable_run(out_base, ident)
        if hit:
            return None, None, hit
    if resume:
        if not run_id:
            raise ValueError("--resume-existing-run needs the explicit --run-id of the interrupted run")
        d = out_base / run_id
        st = d / "run_state.json"
        if not st.is_file():
            raise FileNotFoundError(f"{d} has no run_state.json -- nothing to resume")
        if (d / "manifest.json").is_file():
            raise FileExistsError(f"{d} is already COMPLETE; completed runs are never overwritten or resumed")
        if json.loads(st.read_text(encoding="utf-8"))["identity"] != ident:
            raise RuntimeError("the interrupted run has a different identity (config / variants / upstream run differ); refusing to resume")
        return run_id, d, None
    rid, d = artifacts.new_run_dir(out_base, kind, ident, run_id)
    d.mkdir(parents=True)
    artifacts.atomic_write(d / "run_state.json", json.dumps({"identity": ident, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind}, indent=1))
    return rid, d, None


def _finalize(kind: str, rid: str, run_dir: Path, ident: str, specs: list, rows: list, logs: dict, store: PredictionStore, cohort: Cohort, ds_cfg: dict, fl_cfg: dict, extra_manifest: dict,
              t0: float, scope: str, max_hospitals, tasks) -> dict:
    summary, cls_json, pooled = {}, {}, {}
    md = [f"# Per-class classification reports — FL {kind} (run {rid})", "",
                                            "Primary operating point = the F1-maximising threshold chosen on each client's own validation patients (global federated-histogram threshold when a client's "
                                            "validation split lacks a class). Test labels were read only after every round, hyper-parameter and threshold was fixed. 'paired' = hospitals the Non-FL "
                                            "regime also evaluates (identical test patients).", ""]
    for s in specs:
        vr = [r for r in rows if r["variant"] == s["variant"]]
        a = summarise_variant(vr, ds_cfg)
        a["spec"] = s
        a["n_failed"] = int(sum(1 for r in vr if r["status"] == "failed"))
        a["training"] = {k: logs[s["variant"]].get(k) for k in ("best_round", "rounds_run", "stop_reason", "best_iteration", "n_rounds_run", "val_auroc_hist_at_best", "val_auroc_tie_bound_at_best",
                                                                  "val_auroc_exact_diag", "seconds", "comm_mb_total", "nonfinite_updates") if k in logs[s["variant"]]}
        summary[s["variant"]] = a
        cr = class_reports(vr, ds_cfg)
        cls_json[s["variant"]] = {"spec": s, **cr}
        md += [f"### {s['variant']}", "", class_report_markdown(s["variant"], cr["paired_nonfl"]["f1_max_operating_point"], "paired set, F1-max threshold"),
               class_report_markdown(s["variant"], cr["all_evaluated_clients"]["f1_max_operating_point"], "all evaluated clients, F1-max threshold"), ""]
        pooled[s["variant"]] = _pooled_summary(vr, store, s["variant"], ds_cfg)
    artifacts.atomic_write(run_dir / "per_hospital.json", json.dumps(rows, indent=1, default=str))
    artifacts.atomic_write(run_dir / "per_hospital.csv", _csv(rows))
    artifacts.atomic_write(run_dir / "summary.json", json.dumps(summary, indent=1, default=str))
    artifacts.atomic_write(run_dir / "pooled_summary.json", json.dumps(pooled, indent=1, default=str))
    artifacts.atomic_write(run_dir / "classification_reports.json", json.dumps(cls_json, indent=1, default=str))
    artifacts.atomic_write(run_dir / "classification_reports.md", "\n".join(md) + "\n")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["variant", "hospitalid", "class", "precision", "recall", "f1_score", "support"], lineterminator="\n")
    w.writeheader()
    w.writerows(per_hospital_class_rows(rows))
    artifacts.atomic_write(run_dir / "per_hospital_classification_reports.csv", buf.getvalue())
    (run_dir / "training_logs").mkdir(exist_ok=True)
    for v, lg in logs.items():
        artifacts.atomic_write(run_dir / "training_logs" / f"{v}.json", json.dumps(lg, indent=1, default=str))
    store.save(run_dir / "predictions.npz")
    names = ["per_hospital.json", "per_hospital.csv", "summary.json", "pooled_summary.json", "classification_reports.json", "classification_reports.md", "per_hospital_classification_reports.csv",
             "predictions.npz"]
    ca = cohort.manifest
    n_failed = sum(v["n_failed"] for v in summary.values())
    manifest = {
        "run_id": rid, "definition_version": FL_DEFINITION_VERSION, "kind": kind, "identity": ident, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "regime": FL_REGIME,
        "centrally_pretrained_representations": True, "end_to_end_federated": False,
        "privacy_statement": ("Only the downstream stage is federated. The encoders and cross-attention were trained centrally on pooled training patients with supervised heads, so the embeddings are "
                              "in-sample for those labels and no end-to-end privacy claim is made. Exchanged: model parameters (or FedGBDT histograms), two class-count scalars, validation-logit histograms."),
        "variants": [s["variant"] for s in specs], "tasks": list(tasks), "scope": scope, "partial": bool(max_hospitals), "max_hospitals": max_hospitals,
        "cross_attention_run": ca["run_id"], "cross_attention_identity": ca["identity"], "encoder_runs": ca["encoder_runs"], "incident_label_run": ca["incident_label_run"],
        "split_identity": ca["split_identity"],
        "eligibility": {"mortality": "strict 24 h landmark with a known hospital-discharge status", "sepsis": "incident cohort (hours 24-96 proxy, alive_discharge_as_negative primary policy) inside the landmark"},
        "selection": "rounds / boosting iterations / checkpoints chosen on the federated VALIDATION histogram AUROC only; thresholds on each client's validation patients (or the global validation histogram); test labels used only for the final metrics",
        "histogram_validation": {"bins": fl_cfg["validation"]["bins"], "logit_range": fl_cfg["validation"]["logit_range"], "note": HIST_NOTE},
        "federated_config": {k: fl_cfg[k] for k in ("seed", "scope", "rounds", "local", "algorithms", "focal_alpha_mode", "scaler", "validation", "gbdt", "personalization", "checkpoint")},
        "downstream_config": {k: ds_cfg[k] for k in ("seed", "min_samples", "focal", "graph", "gcn", "smote", "classifiers", "operating_points")},
        "counts": {"hospital_variant_rows": len(rows), "failed": n_failed}, "files": {n: artifacts.sha256_file(run_dir / n) for n in names}, "seconds": round(time.time() - t0, 1),
        "not_directly_comparable": "hospital-macro and pooled metrics are not directly comparable with figures from the base papers or the paper track (different cohorts, windows, protocols)",
        **extra_manifest,
    }
    artifacts.atomic_write(run_dir / "manifest.json", json.dumps(manifest, indent=1, default=str))
    logger.info("[FL] %s COMPLETE in %.1fs -> %s", kind, time.time() - t0, run_dir)
    return {"run_dir": str(run_dir), "run_id": rid, "cache_hit": False, "manifest": manifest, "summary": summary}


# ---------------------------------------------------------------------------------------------------------------------------------
# one variant
# ---------------------------------------------------------------------------------------------------------------------------------
def _run_nn_variant(spec, clients, skipped, cohort, ds_cfg, fl_cfg, store, ckpt_root, ident, resume, round_hook, family, graph_key=None, smote=False):
    seed = stable_seed(fl_cfg["seed"], "fl_init", spec["variant"])
    problems, failed = make_problems(clients, family, ds_cfg, graph_key, smote)
    used_clients = [problems[h].client for h in sorted(problems)]
    sk = list(skipped) + failed
    alphas = make_alphas(problems, ds_cfg, fl_cfg, smote)
    method = spec.get("method", "global")
    algo = "fedavg" if method != "global" else spec["algorithm"]
    res = train_federated(problems, family, ds_cfg, fl_cfg, algo=algo, method=method, alphas=alphas, seed=seed, ckpt_dir=Path(ckpt_root) / spec["variant"], resume=resume,
                          ident=ident + "|" + spec["variant"], round_hook=round_hook, tag=spec["variant"])
    scorer, model = _nn_scorer(res, problems, family, ds_cfg, seed, personalised=method if method in ("fedper", "fedrep") else None)
    res["val_auroc_exact_diag"] = exact_val_auroc_diag(problems, model, res)
    rows = evaluate_variant(spec, cohort, spec["task"], used_clients, sk, scorer, ds_cfg, store, _global_threshold(res, fl_cfg), seed)
    extra = {"graph_info": {h: p.info for h, p in problems.items()} if family == "gcn" else None, "alpha": {"mode": fl_cfg["focal_alpha_mode"], "value_range": [min(alphas.values()), max(alphas.values())]}}
    return rows, _train_log(spec, res, extra), res, problems, sk


def _run_gbdt_variant(spec, clients, skipped, cohort, ds_cfg, fl_cfg, store, ckpt_root, ident, resume):
    seed = stable_seed(fl_cfg["seed"], "fl_init", spec["variant"])
    res = train_fedgbdt(clients, fl_cfg["gbdt"], fl_cfg["validation"], seed, ckpt_dir=Path(ckpt_root) / spec["variant"], resume=resume, ident=ident + "|" + spec["variant"])
    scorer = lambda c: {"p_val": expit(res["margins"][c.hospitalid]["val"]), "p_test": expit(res["margins"][c.hospitalid]["test"]),
                        "info": {"best_iteration": res["best_iteration"], "n_trees": res["n_trees"]}}
    h = HistAccumulator(fl_cfg["validation"]["bins"], fl_cfg["validation"]["logit_range"])
    h.pos, h.neg = res["val_pos"], res["val_neg"]
    rows = evaluate_variant(spec, cohort, spec["task"], clients, skipped, scorer, ds_cfg, store, h.best_f1_probability_threshold(), seed)
    return rows, _gbdt_log(spec, res), res


# ---------------------------------------------------------------------------------------------------------------------------------
# Configs 1-4
# ---------------------------------------------------------------------------------------------------------------------------------
def run_fl(config: int, cohort: Cohort, ds_cfg: dict, fl_cfg: dict, out_base, *, embedding_sets=("fused",), classifiers=FL_CLASSIFIERS, graphs=("A", "B"), tasks=TASKS,
           algorithms=NN_ALGORITHMS, c4_gcn_algorithm: str = "fedavg", max_hospitals=None, run_id: "str | None" = None, force: bool = False, resume: bool = False,
           hparams_source: "str | None" = None, round_hook=None) -> dict:
    t0 = time.time()
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {CONFIGS}")
    for es in embedding_sets:
        if es not in EMBEDDING_SETS or es not in cohort.embeddings:
            raise ValueError(f"embedding set {es!r} not loaded in the cohort (loaded: {list(cohort.embeddings)})")
    for g in graphs:
        if g not in GRAPHS:
            raise ValueError(f"unknown graph {g!r}; choose A or B")
    if config != 1 and tuple(embedding_sets) != ("fused",):
        raise ValueError("Configs 2-4 use the fused embedding only")
    if c4_gcn_algorithm not in NN_ALGORITHMS:
        raise ValueError(f"c4_gcn_algorithm must be one of {NN_ALGORITHMS}")
    scope = fl_cfg["scope"]
    specs = fl_variant_specs(config, embedding_sets, classifiers, graphs, tasks, algorithms, c4_gcn_algorithm)
    ident = fl_identity(f"fl_c{config}", cohort.manifest, ds_cfg, fl_cfg, specs, scope, max_hospitals, hparams_source)
    rid, run_dir, hit = _prepare_run(f"fl_c{config}", out_base, ident, run_id, force, resume)
    if hit:
        logger.info("[FL] cache HIT: reusing verified run %s", hit["run_id"])
        return {"run_dir": hit["run_dir"], "run_id": hit["run_id"], "cache_hit": True, "manifest": hit}
    logger.info("[FL] config %d | %d variants | scope=%s | identity %s -> %s", config, len(specs), scope, ident[:12], run_dir)
    ckpt_root = run_dir / "checkpoints"
    store = PredictionStore()
    rows, logs = [], {}
    for task in tasks:
        tspecs = [s for s in specs if s["task"] == task]
        sets_needed = sorted({s["embedding_set"] for s in tspecs})
        data = {es: build_clients(cohort, task, es, ds_cfg, scope, max_hospitals, fl_cfg["seed"]) for es in sets_needed}
        if config == 1:
            for s in tspecs:
                clients, skipped = data[s["embedding_set"]]
                if s["classifier"] == "gbdt_fed":
                    r, lg, _ = _run_gbdt_variant(s, clients, skipped, cohort, ds_cfg, fl_cfg, store, ckpt_root, ident, resume)
                else:
                    r, lg, _, _, _ = _run_nn_variant(s, clients, skipped, cohort, ds_cfg, fl_cfg, store, ckpt_root, ident, resume, round_hook, s["classifier"])
                rows += r
                logs[s["variant"]] = lg
        elif config in (2, 3):
            clients, skipped = data["fused"]
            for s in tspecs:
                r, lg, _, _, _ = _run_nn_variant(s, clients, skipped, cohort, ds_cfg, fl_cfg, store, ckpt_root, ident, resume, round_hook, "gcn", s["graph"], smote=(config == 3))
                rows += r
                logs[s["variant"]] = lg
        else:                                                       # Config 4: two stages per (graph, task)
            clients, skipped = data["fused"]
            for g in graphs:
                s1 = {"variant": f"c4s1__graph{g}__{task}__{c4_gcn_algorithm}", "config": 4, "task": task, "embedding_set": "fused", "classifier": "gcn", "graph": g, "sampling": "none",
                      "algorithm": c4_gcn_algorithm, "method": "global"}
                seed1 = stable_seed(fl_cfg["seed"], "fl_init", s1["variant"])
                problems, failed = make_problems(clients, "gcn", ds_cfg, g, False)
                alphas = make_alphas(problems, ds_cfg, fl_cfg, False)
                res1 = train_federated(problems, "gcn", ds_cfg, fl_cfg, algo=c4_gcn_algorithm, alphas=alphas, seed=seed1, ckpt_dir=ckpt_root / s1["variant"], resume=resume,
                                       ident=ident + "|" + s1["variant"], round_hook=round_hook, tag=s1["variant"])
                logs[s1["variant"]] = _train_log(s1, res1, {"role": "Config 4 stage 1 (federated GCN, embedding generator)", "failed_clients": failed})
                model1 = build_model("gcn", clients[0].dim, ds_cfg, seed1)
                emb_clients = []
                for hid in sorted(problems):
                    set_params(model1, client_params(res1, hid))
                    model1.eval()
                    pr = problems[hid]
                    with torch.no_grad():
                        Z = {sp: pr.embeddings(model1, sp) for sp in SPLITS}
                    emb_clients.append(pr.client.with_features(Z))
                sk2 = list(skipped) + failed
                for s in [x for x in tspecs if x["graph"] == g]:
                    if s["classifier"] == "gbdt_fed":
                        r, lg, _ = _run_gbdt_variant(s, emb_clients, sk2, cohort, ds_cfg, fl_cfg, store, ckpt_root, ident, resume)
                    else:
                        r, lg, _, _, _ = _run_nn_variant(s, emb_clients, sk2, cohort, ds_cfg, fl_cfg, store, ckpt_root, ident, resume, round_hook, s["classifier"])
                    lg["stage1_variant"] = s1["variant"]
                    rows += r
                    logs[s["variant"]] = lg
        data.clear()
    return _finalize(f"fl_c{config}", rid, run_dir, ident, specs, rows, logs, store, cohort, ds_cfg, fl_cfg,
                     {"config": config, "algorithms": list(algorithms), "classifiers": list(classifiers) if config in (1, 4) else None, "graphs": list(graphs) if config > 1 else None,
                      "embedding_sets": list(embedding_sets) if config == 1 else ["fused"], "c4_gcn_algorithm": c4_gcn_algorithm if config == 4 else None,
                      "hparams": {"source_run": hparams_source} if hparams_source else None,
                      "tree_note": "FedGBDT = federated histogram boosting (logistic loss, optional positive-class weight; no focal). Non-FL focal LightGBM / standard-loss GBDT have no exact federated counterpart; trees are never parameter-averaged.",
                      "non_fl_comparison": "evaluation uses the paired set = hospitals passing the Non-FL minimum-data rule (identical test patients); clients outside it are reported separately as fl_only"},
                     t0, scope, max_hospitals, tasks)


# ---------------------------------------------------------------------------------------------------------------------------------
# personalised experiments
# ---------------------------------------------------------------------------------------------------------------------------------
def personalization_specs(methods, models, graph: str, tasks) -> list:
    for m in methods:
        if m not in PERSONALIZATION_METHODS:
            raise ValueError(f"unknown personalisation method {m!r}; choose from {PERSONALIZATION_METHODS}")
    for m in models:
        if m not in PERSONALIZED_MODELS:
            raise ValueError(f"personalised models are {PERSONALIZED_MODELS} (LR has no body/head split)")
    out = []
    for task in tasks:
        for model in models:
            gtag = f"__graph{graph}" if model == "gcn" else ""
            base = {"config": 5, "task": task, "embedding_set": "fused", "classifier": model, "graph": graph if model == "gcn" else None, "sampling": "none"}
            if "finetune" in methods:
                out.append({**base, "variant": f"fp__fedavg_global__{model}{gtag}__{task}", "algorithm": "fedavg", "method": "global", "role": "reference: the FedAvg global model before fine-tuning"})
                out.append({**base, "variant": f"fp__finetune__{model}{gtag}__{task}", "algorithm": "fedavg", "method": "finetune"})
            for m in methods:
                if m in ("fedper", "fedrep"):
                    out.append({**base, "variant": f"fp__{m}__{model}{gtag}__{task}", "algorithm": "fedavg", "method": m})
    return out


def run_fl_personalized(cohort: Cohort, ds_cfg: dict, fl_cfg: dict, out_base, *, methods=PERSONALIZATION_METHODS, models=PERSONALIZED_MODELS, graph: str = "A", tasks=TASKS, max_hospitals=None,
                        run_id: "str | None" = None, force: bool = False, resume: bool = False, hparams_source: "str | None" = None, round_hook=None) -> dict:
    """Separate personalisation experiment (Config id 5): FedPer / FedRep (shared body, hospital-specific heads, body aggregated with FedAvg) and FedAvg followed by local fine-tuning.
    The un-personalised FedAvg model is recorded as the reference variant `fp__fedavg_global__*` whenever fine-tuning is requested."""
    t0 = time.time()
    if graph not in GRAPHS:
        raise ValueError(f"unknown graph {graph!r}")
    scope = fl_cfg["scope"]
    specs = personalization_specs(methods, models, graph, tasks)
    ident = fl_identity("fl_personalized", cohort.manifest, ds_cfg, fl_cfg, specs, scope, max_hospitals, hparams_source)
    rid, run_dir, hit = _prepare_run("fl_pers", out_base, ident, run_id, force, resume)
    if hit:
        return {"run_dir": hit["run_dir"], "run_id": hit["run_id"], "cache_hit": True, "manifest": hit}
    logger.info("[FL] personalised | %d variants | scope=%s -> %s", len(specs), scope, run_dir)
    ckpt_root = run_dir / "checkpoints"
    store = PredictionStore()
    rows, logs = [], {}
    for task in tasks:
        clients, skipped = build_clients(cohort, task, "fused", ds_cfg, scope, max_hospitals, fl_cfg["seed"])
        for model in models:
            gkey = graph if model == "gcn" else None
            base = None                                             # the FedAvg global model of this (task, model), shared by the reference and the fine-tune variants
            for s in [x for x in specs if x["task"] == task and x["classifier"] == model]:
                if s["method"] == "finetune":
                    res, problems, sk = base
                    ref_variant = next(x["variant"] for x in specs if x["task"] == task and x["classifier"] == model and x["method"] == "global")
                    seed = stable_seed(fl_cfg["seed"], "fl_init", ref_variant)
                    alphas = make_alphas(problems, ds_cfg, fl_cfg, False)
                    t1 = time.time()
                    full, info = finetune_clients(res, problems, model, ds_cfg, fl_cfg, alphas, seed)
                    res_ft = {**res, "client_full": full}
                    scorer, _ = _nn_scorer(res_ft, problems, model, ds_cfg, seed, personalised="finetune", ft_info=info)
                    used = [problems[h].client for h in sorted(problems)]
                    rows += evaluate_variant(s, cohort, task, used, sk, scorer, ds_cfg, store, _global_threshold(res, fl_cfg), seed)
                    tuned = [v for v in info.values() if v["source"] == "local_finetune"]
                    logs[s["variant"]] = {"spec": s, "base_variant": ref_variant, "n_finetuned": len(tuned), "n_global_fallback": len(info) - len(tuned), "seconds": round(time.time() - t1, 2),
                                          "mean_best_epoch": float(np.mean([v["best_epoch"] for v in tuned])) if tuned else None, "per_client": info,
                                          "note": "epoch 0 (the global model) is a candidate, so fine-tuning cannot end worse than the global model on the client's validation loss"}
                    continue
                r, lg, res, problems, sk = _run_nn_variant(s, clients, skipped, cohort, ds_cfg, fl_cfg, store, ckpt_root, ident, resume, round_hook, model, gkey)
                rows += r
                logs[s["variant"]] = lg
                if s["method"] == "global":
                    base = (res, problems, sk)
    return _finalize("fl_personalized", rid, run_dir, ident, specs, rows, logs, store, cohort, ds_cfg, fl_cfg,
                     {"methods": list(methods), "models": list(models), "graph": graph, "hparams": {"source_run": hparams_source} if hparams_source else None,
                      "personalisation_note": ("FedPer/FedRep heads never leave their hospital; clients that do not satisfy the minimum-data rule use the sample-weighted mean head (head_source=mean_head_fallback) "
                                               "or the global model (fine-tuning); pooled AUROC of personalised models mixes the score scales of different heads"),
                      "non_fl_comparison": "paired set = hospitals passing the Non-FL minimum-data rule (identical test patients)"},
                     t0, scope, max_hospitals, tasks)
