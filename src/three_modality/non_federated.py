"""Non-FL regime, Configs 1-4: every hospital trains and evaluates its OWN models from its OWN train / val / test patients. Nothing is shared between hospitals: no parameters, no
scalers, no graphs, no pooled patient-level classification, no aggregation. (The three encoders and the three-token cross-attention that produced the input embeddings are CENTRALLY
pretrained; this regime is therefore a hospital-wise downstream study, not end-to-end learning, and makes no federated claim.)

  Config 1 : embedding set (fused or one individual modality) + LR / MLP / LightGBM(focal) / GradientBoosting(standard loss);
  Config 2 : hospital-local Graph A / Graph B + GCN, no sampling;
  Config 3 : Config 2 with hospital-local graph-compatible SMOTE (training positives only);
  Config 4 : the Config-2 GCN's hospital-local node embeddings (Graph A / B) + the four classifiers.

Per hospital and task the stage records: train/val/test counts and class distribution, the validation-selected threshold, AUROC / AUPRC / accuracy / precision / recall / F1 /
specificity, the confusion matrix and per-class report -- or the reason the hospital was skipped.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import time
from pathlib import Path

import numpy as np

from src.three_modality import artifacts
from src.three_modality.downstream_config import CLASSIFIERS, EMBEDDING_SETS, GRAPHS, TASKS, apply_hparams
from src.three_modality.hw_classifiers import CLASSIFIER_LOSSES, fit_predict
from src.three_modality.hw_data import Cohort, SPLITS, hospital_task_data, plan_hospitals, stable_seed
from src.three_modality.hw_eval import aggregate, hospital_metrics, select_threshold, select_threshold_recall
from src.three_modality.hw_reports import PredictionStore, aggregate_class_reports, class_report_markdown, per_hospital_class_rows
from src.three_modality.hw_graph import train_gcn

logger = logging.getLogger("eicu_fl")

NONFL_DEFINITION_VERSION = "nonfl-v2"      # v2: stabilised LightGBM focal objective, selectable selection metric / min_epochs, recall-target operating point, saved test predictions, aggregated per-class reports
CONFIGS = (1, 2, 3, 4)
REGIME = ("Non-FL: one independent model per hospital and task, trained only on that hospital's own training patients; no parameter sharing, no aggregation, no pooled patient-level "
          "classification. The encoders and the three-token cross-attention are centrally pretrained, so this is hospital-wise downstream learning, not end-to-end learning.")


def variant_specs(config: int, embedding_sets, classifiers, graphs, tasks) -> list:
    out = []
    for task in tasks:
        if config == 1:
            out += [{"variant": f"c1__{es}__{task}__{clf}", "config": 1, "task": task, "embedding_set": es, "classifier": clf, "graph": None, "sampling": "none"}
                    for es in embedding_sets for clf in classifiers]
        elif config in (2, 3):
            out += [{"variant": f"c{config}__graph{g}__{task}", "config": config, "task": task, "embedding_set": "fused", "classifier": "gcn", "graph": g,
                     "sampling": "graph_smote" if config == 3 else "none"} for g in graphs]
        elif config == 4:
            out += [{"variant": f"c4__graph{g}__{task}__{clf}", "config": 4, "task": task, "embedding_set": "fused", "classifier": clf, "graph": g, "sampling": "none"}
                    for g in graphs for clf in classifiers]
        else:
            raise ValueError(f"config must be one of {CONFIGS}")
    return out


def nonfl_identity(config: int, ca_manifest: dict, ds_cfg: dict, specs: list, max_hospitals, hparams_source=None) -> str:
    rel = {"min_samples": ds_cfg["min_samples"], "focal": ds_cfg["focal"], "seed": ds_cfg["seed"], "operating_points": ds_cfg["operating_points"]}
    if config == 1:
        rel["classifiers"] = ds_cfg["classifiers"]
    else:
        rel.update({"graph": ds_cfg["graph"], "gcn": ds_cfg["gcn"]})
        if config == 3:
            rel["smote"] = ds_cfg["smote"]
        if config == 4:
            rel["classifiers"] = ds_cfg["classifiers"]
    return artifacts.hash_json({"definition": NONFL_DEFINITION_VERSION, "config": config, "cross_attention": [ca_manifest["run_id"], ca_manifest["identity"]],
                                "variants": [s["variant"] for s in specs], "cfg": rel, "max_hospitals": max_hospitals, "hparams_source": hparams_source})


def _base_row(spec: dict, d, seed: int) -> dict:
    return {"variant": spec["variant"], "config": spec["config"], "task": spec["task"], "hospitalid": d.hospitalid, "embedding_set": spec["embedding_set"], "graph": spec["graph"],
            "classifier": spec["classifier"], "sampling": spec["sampling"], "seed": seed, "counts": d.counts, "status": None, "skip_reason": None, "threshold": None,
            "metrics": None, "info": None, "seconds": None}


def _finish(row: dict, d, out: dict, ds_cfg: dict, store: "PredictionStore | None" = None) -> dict:
    yv = d.labels("val")                                                  # validation labels only
    thr = select_threshold(yv, out["p_val"])
    row["threshold"] = thr
    y_test = d.test_labels_for_evaluation()                               # the ONLY use of test labels: after the model and every threshold are fixed
    row["metrics"] = hospital_metrics(y_test, out["p_test"], thr)
    target = float(ds_cfg["operating_points"]["recall_target"])
    thr2 = select_threshold_recall(yv, out["p_val"], target)              # secondary operating point, also chosen on validation only
    m2 = hospital_metrics(y_test, out["p_test"], thr2)
    m2.pop("classification_report", None)
    row["operating_points"] = {"recall_target": {"target": target, "threshold": thr2, "metrics": m2}}
    if store is not None:
        store.add(row["variant"], d.patient_ids("test"), d.hospitalid, y_test, out["p_test"])
    row["info"] = out["info"]
    row["status"] = "trained"
    return row


def _guard(row: dict, fn):
    """Run one hospital; an unexpected error is RECORDED (status 'failed') and counted, never swallowed silently and never allowed to discard the other hospitals' work."""
    t0 = time.time()
    try:
        fn()
    except Exception as exc:          # noqa: BLE001
        row["status"], row["metrics"], row["skip_reason"] = "failed", None, f"error: {type(exc).__name__}: {exc}"
        logger.exception("[NONFL] hospital %s %s failed", row["hospitalid"], row["variant"])
    row["seconds"] = round(time.time() - t0, 2)
    return row


def run_variants(config: int, cohort: Cohort, ds_cfg: dict, specs: list, max_hospitals=None, progress_every: int = 25, store: "PredictionStore | None" = None) -> list:
    """All hospital x variant rows. Hospitals are visited independently; processing order cannot influence any result (seeds derive from hospital/task/variant only)."""
    rows = []
    by_task = {t: plan_hospitals(cohort, t, max_hospitals) for t in sorted({s["task"] for s in specs})}
    gcn_cache = {}
    for task, hosp_list in by_task.items():
        task_specs = [s for s in specs if s["task"] == task]
        for hi, d in enumerate(hosp_list, start=1):
            reason = d.skip_reason(ds_cfg["min_samples"])
            for spec in task_specs:
                seed = stable_seed(ds_cfg["seed"], d.hospitalid, task, spec["variant"])
                row = _base_row(spec, d, seed)
                if reason:
                    row["status"], row["skip_reason"] = "skipped", reason
                    rows.append(row)
                    continue

                def work(spec=spec, d=d, row=row, seed=seed):
                    ytr, yva = d.labels("train"), d.labels("val")
                    if spec["config"] == 1:
                        es = spec["embedding_set"]
                        out = fit_predict(spec["classifier"], d.X("train", es), ytr, d.X("val", es), yva, d.X("test", es), ds_cfg, seed)
                    elif spec["config"] in (2, 3):
                        out = train_gcn(d.X("train"), ytr, d.X("val"), yva, d.X("test"), spec["graph"], ds_cfg, seed, smote=spec["config"] == 3)
                    else:                                              # config 4: ONE Config-2 GCN per (hospital, task, graph), shared by the four classifiers below
                        ck = (d.hospitalid, task, spec["graph"])
                        if ck not in gcn_cache:
                            gseed = stable_seed(ds_cfg["seed"], d.hospitalid, task, f"c4gcn|{spec['graph']}")
                            gcn_cache.clear()
                            gcn_cache[ck] = train_gcn(d.X("train"), ytr, d.X("val"), yva, d.X("test"), spec["graph"], ds_cfg, gseed, smote=False, want_embeddings=True)
                        g = gcn_cache[ck]
                        out = fit_predict(spec["classifier"], g["emb_train"], ytr, g["emb_val"], yva, g["emb_test"], ds_cfg, seed)
                        out["info"]["graph_embedding_source"] = {"gcn_best_epoch": g["info"]["best_epoch"], "graph": g["info"]["graph"], "gcn_trained_without_sampling": True}
                    _finish(row, d, out, ds_cfg, store)

                rows.append(_guard(row, work))
            if progress_every and hi % progress_every == 0:
                logger.info("[NONFL] config %d | %s: %d/%d hospitals visited", config, task, hi, len(hosp_list))
    return rows


_CSV_COLS = ["variant", "config", "task", "embedding_set", "graph", "classifier", "sampling", "hospitalid", "status", "skip_reason", "train_n", "train_pos", "train_neg", "val_n", "val_pos",
             "val_neg", "test_n", "test_pos", "test_neg", "threshold", "auroc", "auprc", "accuracy", "precision", "recall", "f1", "specificity", "tn", "fp", "fn", "tp", "rt_threshold", "rt_precision", "rt_recall", "rt_f1", "rt_accuracy", "rt_specificity", "seconds"]


def _csv(rows: list) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_CSV_COLS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        m = r["metrics"] or {}
        flat = {k: r.get(k) for k in ("variant", "config", "task", "embedding_set", "graph", "classifier", "sampling", "hospitalid", "status", "skip_reason", "threshold", "seconds")}
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


def run_nonfl(config: int, cohort: Cohort, ds_cfg: dict, out_base, *, embedding_sets=("fused",), classifiers=CLASSIFIERS, graphs=("A", "B"), tasks=TASKS, max_hospitals=None,
              run_id: "str | None" = None, force: bool = False, hparams: "dict | None" = None, hparams_source: "str | None" = None) -> dict:
    t0 = time.time()
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {CONFIGS}")
    for es in embedding_sets:
        if es not in EMBEDDING_SETS or es not in cohort.embeddings:
            raise ValueError(f"embedding set {es!r} not loaded in the cohort (loaded: {list(cohort.embeddings)})")
    for c in classifiers:
        if c not in CLASSIFIER_LOSSES:
            raise ValueError(f"unknown classifier {c!r}")
    for g in graphs:
        if g not in GRAPHS:
            raise ValueError(f"unknown graph {g!r}; choose A or B")
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    if config != 1 and tuple(embedding_sets) != ("fused",):
        raise ValueError("Configs 2-4 use the fused embedding only")
    ds_cfg = apply_hparams(ds_cfg, hparams)                               # tuned hyper-parameters (validation-selected) overlay the defaults; eligibility / seeds cannot be overridden
    specs = variant_specs(config, embedding_sets, classifiers, graphs, tasks)
    ident = nonfl_identity(config, cohort.manifest, ds_cfg, specs, max_hospitals, hparams_source)
    if not force:
        hit = artifacts.find_reusable_run(out_base, ident)
        if hit:
            logger.info("[NONFL] cache HIT: reusing verified run %s", hit["run_id"])
            return {"run_dir": hit["run_dir"], "run_id": hit["run_id"], "cache_hit": True, "manifest": hit}
    rid, run_dir = artifacts.new_run_dir(out_base, f"nonfl_c{config}", ident, run_id)
    logger.info("[NONFL] config %d | %d variants | cache MISS (identity %s) -> %s", config, len(specs), ident[:12], run_dir)
    store = PredictionStore()
    rows = run_variants(config, cohort, ds_cfg, specs, max_hospitals, store=store)
    summary, cls_json, md = {}, {}, [f"# Per-class classification reports — Non-FL config {config} (run {rid})", "",
                                     "Primary operating point = the F1-maximising threshold chosen on each hospital's validation patients. Secondary = highest validation threshold reaching the recall target. "
                                     "Test labels were read only after every model and threshold was fixed.", ""]
    for s in specs:
        vr = [r for r in rows if r["variant"] == s["variant"]]
        a = aggregate(vr)
        a["n_failed"] = int(sum(1 for r in vr if r["status"] == "failed"))
        a["spec"] = s
        rt_rows = [{"status": r["status"], "metrics": ((r.get("operating_points") or {}).get("recall_target") or {}).get("metrics"), "skip_reason": r.get("skip_reason")} for r in vr]
        a["recall_target_operating_point"] = {"target": ds_cfg["operating_points"]["recall_target"], **aggregate(rt_rows)["metrics"]}
        summary[s["variant"]] = a
        rep = aggregate_class_reports(vr)
        rep_rt = aggregate_class_reports([{"status": r["status"], "variant": r["variant"], "hospitalid": r["hospitalid"], "metrics": ((r.get("operating_points") or {}).get("recall_target") or {}).get("metrics")} for r in vr])
        cls_json[s["variant"]] = {"spec": s, "f1_max_operating_point": rep, "recall_target_operating_point": {"target": ds_cfg["operating_points"]["recall_target"], **rep_rt}}
        md += [f"### config {config} · {s['task']} · {s['variant']}", "", class_report_markdown(s["variant"], rep, "F1-max threshold"),
               class_report_markdown(s["variant"], rep_rt, f"recall-target {ds_cfg['operating_points']['recall_target']} threshold"), ""]
    n_failed = sum(v["n_failed"] for v in summary.values())
    if n_failed:
        logger.warning("[NONFL] %d hospital/variant runs FAILED with an error (recorded in per_hospital.json as status 'failed')", n_failed)
    run_dir.mkdir(parents=True)
    artifacts.atomic_write(run_dir / "per_hospital.json", json.dumps(rows, indent=1, default=str))
    artifacts.atomic_write(run_dir / "per_hospital.csv", _csv(rows))
    artifacts.atomic_write(run_dir / "summary.json", json.dumps(summary, indent=1, default=str))
    artifacts.atomic_write(run_dir / "classification_reports.json", json.dumps(cls_json, indent=1, default=str))
    artifacts.atomic_write(run_dir / "classification_reports.md", "\n".join(md) + "\n")
    pc = per_hospital_class_rows(rows)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["variant", "hospitalid", "class", "precision", "recall", "f1_score", "support"], lineterminator="\n")
    w.writeheader()
    w.writerows(pc)
    artifacts.atomic_write(run_dir / "per_hospital_classification_reports.csv", buf.getvalue())
    store.save(run_dir / "predictions.npz")
    ca = cohort.manifest
    names = ["per_hospital.json", "per_hospital.csv", "summary.json", "classification_reports.json", "classification_reports.md", "per_hospital_classification_reports.csv", "predictions.npz"]
    manifest = {
        "run_id": rid, "definition_version": NONFL_DEFINITION_VERSION, "identity": ident, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "regime": REGIME,
        "config": config, "variants": [s["variant"] for s in specs], "tasks": list(tasks), "embedding_sets": list(embedding_sets) if config == 1 else ["fused"],
        "classifiers": list(classifiers) if config in (1, 4) else None, "graphs": list(graphs) if config > 1 else None,
        "cross_attention_run": ca["run_id"], "cross_attention_identity": ca["identity"], "encoder_runs": ca["encoder_runs"], "incident_label_run": ca["incident_label_run"],
        "split_identity": ca["split_identity"], "partial": bool(max_hospitals), "max_hospitals": max_hospitals,
        "hparams": {"source_run": hparams_source, "overrides": hparams or {}, "note": "hyper-parameters chosen once from VALIDATION data (nonfl_tune), shared by all hospitals; no test data used"} if hparams else None,
        "eligibility": {"mortality": "strict 24 h landmark with a known hospital-discharge status", "sepsis": "incident cohort (hours 24-96 proxy, alive_discharge_as_negative primary policy) inside the landmark"},
        "selection": "checkpoints / boosting stages / thresholds chosen on each hospital's VALIDATION patients only; test labels used only for the final metrics",
        "classifier_losses": {c: CLASSIFIER_LOSSES[c] for c in classifiers} if config in (1, 4) else None,
        "downstream_config": {k: ds_cfg[k] for k in ("seed", "min_samples", "focal", "graph", "gcn", "smote", "classifiers", "operating_points")},
        "counts": {"hospital_variant_rows": len(rows), "failed": n_failed}, "files": {n: artifacts.sha256_file(run_dir / n) for n in names}, "seconds": round(time.time() - t0, 1),
        "not_directly_comparable": "hospital-wise macro / sample-weighted metrics are not directly comparable with pooled or federated figures from the base papers or the paper track",
    }
    artifacts.atomic_write(run_dir / "manifest.json", json.dumps(manifest, indent=1, default=str))
    logger.info("[NONFL] config %d COMPLETE in %.1fs -> %s", config, time.time() - t0, run_dir)
    return {"run_dir": str(run_dir), "run_id": rid, "cache_hit": False, "manifest": manifest, "summary": summary}
