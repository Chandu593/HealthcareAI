"""Consolidates completed Non-FL runs (Configs 1-4) of ONE cross-attention run into a report: per-variant macro / sample-weighted hospital averages, successful / skipped hospital counts with
reasons, and paired hospital-level comparisons -- Graph A vs Graph B, graph SMOTE vs none (Config 3 vs 2), individual-modality vs fused embeddings (Config 1).

Hospital-level metrics from a handful of test patients are noisy; every comparison therefore reports the number of paired hospitals, the mean paired difference, win/tie/loss counts and a
hospital-bootstrap 95% interval, and says nothing stronger than that. Nothing here is compared with the base papers: protocols and cohorts differ (stated in the report).
"""
from __future__ import annotations

import csv
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.three_modality import artifacts
from src.three_modality.hw_reports import aggregate_class_reports, class_report_markdown, load_predictions, pooled_metrics

REPORT_DEFINITION_VERSION = "nonfl-report-v2"      # v2: aggregated per-class reports, pooled test metrics with bootstrap intervals, recall-target operating point, FLICS reference table
PAIR_METRICS = ("auroc", "auprc", "f1")

FLICS_REFERENCE = {
    "source": "FLICS (Federated Learning for ICU Mortality Prediction: Balancing Accuracy and Privacy in a Multi-Hospital Setting), Tables IV-V; ~14% positive class; thresholds tuned ON THE TEST SET to maximise F1; pooled test set of 1,000 patients",
    "rows": [("3 hospitals", "Local LR A / B / C", "0.87 / 0.89 / 0.90", "0.86 / 0.85 / 0.86", "0.58 / 0.61 / 0.61"), ("3 hospitals", "FL FedAvg LR", "0.90", "0.87", "0.60"),
             ("3 hospitals", "FL FedAvg LSTM", "0.89", "0.87", "0.61"), ("3 hospitals", "Centralized XGBoost (non-private)", "0.91", "0.90", "0.65"),
             ("10 hospitals", "Local LR (avg of 10)", "0.88", "0.83", "0.59"), ("10 hospitals", "FL FedAvg LR", "0.89", "0.83", "0.59"),
             ("10 hospitals", "FL FedAvg LSTM", "0.89", "0.84", "0.61"), ("10 hospitals", "Centralized XGBoost (non-private)", "0.90", "0.88", "0.63")]}


def collect_runs(non_federated_base, cross_attention_run: str, include_partial: bool = False) -> dict:
    """Completed, hash-verified Non-FL runs of this cross-attention run. For a variant present in several runs the NEWEST run wins; partial runs are excluded unless requested."""
    base = Path(non_federated_base)
    chosen, used, skipped_partial = {}, [], []
    for r in sorted(artifacts.list_runs(base), key=lambda r: (r.get("created_utc", ""), r.get("run_id", ""))):      # oldest -> newest by CREATION time (list_runs orders by directory name)
        if r.get("cross_attention_run") != cross_attention_run or not (str(r.get("definition_version", "")).startswith("nonfl-v") and r.get("config") in (1, 2, 3, 4) and "variants" in r):      # by manifest content, never by the run-id spelling
            continue
        if r.get("partial") and not include_partial:
            skipped_partial.append(r["run_id"])
            continue
        d = Path(r["run_dir"])
        if not all((d / n).is_file() and artifacts.sha256_file(d / n) == h for n, h in r["files"].items()):
            raise ValueError(f"run {r['run_id']}: files do not match their recorded hashes; refusing to report on a modified run")
        rows = json.loads((d / "per_hospital.json").read_text(encoding="utf-8"))
        summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        for v in r["variants"]:
            chosen[v] = {"run_id": r["run_id"], "rows": [x for x in rows if x["variant"] == v], "summary": summ[v], "created_utc": r["created_utc"],
                         "pred_file": str(d / "predictions.npz") if (d / "predictions.npz").is_file() else None}
        used.append(r["run_id"])
    return {"variants": chosen, "runs_used": used, "partial_runs_excluded": skipped_partial}


def _by_hospital(rows: list, metric: str) -> dict:
    return {r["hospitalid"]: r["metrics"][metric] for r in rows if r["status"] == "trained" and r["metrics"] and r["metrics"][metric] is not None}


def paired(rows_x: list, rows_y: list, metric: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """delta = y - x over hospitals trained in BOTH with the metric defined in both."""
    a, b = _by_hospital(rows_x, metric), _by_hospital(rows_y, metric)
    common = sorted(set(a) & set(b))
    if not common:
        return {"n_paired_hospitals": 0, "mean_delta": None}
    d = np.array([b[h] - a[h] for h in common])
    rs = np.random.default_rng(seed)
    boots = np.array([d[rs.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"n_paired_hospitals": len(common), "mean_delta": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "y_better": int((d > 1e-9).sum()), "ties": int((np.abs(d) <= 1e-9).sum()), "y_worse": int((d < -1e-9).sum()), "median_delta": float(np.median(d))}


def comparisons(variants: dict) -> list:
    out = []

    def add(kind, x, y, note):
        if x in variants and y in variants:
            out.append({"comparison": kind, "x": x, "y": y, "delta_is": "y minus x", "note": note, **{m: paired(variants[x]["rows"], variants[y]["rows"], m) for m in PAIR_METRICS}})

    for v, rec in variants.items():
        s = rec["summary"]["spec"]
        c, t, g, clf = s["config"], s["task"], s["graph"], s["classifier"]
        if g == "A":
            add("Graph A vs Graph B", v, v.replace("__graphA__", "__graphB__"), f"config {c}, {t}, {clf}")
        if c == 3 and g:
            add("no sampling (Config 2) vs graph SMOTE (Config 3)", v.replace("c3__", "c2__"), v, f"graph {g}, {t}")
        if c == 1 and s["embedding_set"] != "fused":
            add("individual modality vs fused embedding", v, v.replace(f"__{s['embedding_set']}__", "__fused__"), f"{s['embedding_set']} -> fused, {t}, {clf}")
        if c == 4 and g:
            add("Config 2 GCN head vs Config 4 classifier on its embeddings", v.replace("c4__", "c2__").rsplit("__", 1)[0], v, f"graph {g}, {t}, {clf}")
        if c == 4 and clf == "lr":
            pass
    return out


def build_nonfl_report(non_federated_base, reports_base, cross_attention_run: str, include_partial: bool = False, run_id: "str | None" = None,
                       n_bootstrap: int = 500, top_hospitals=(3, 10), seed: int = 0) -> dict:
    t0 = time.time()
    col = collect_runs(non_federated_base, cross_attention_run, include_partial)
    variants = col["variants"]
    if not variants:
        raise ValueError(f"no completed Non-FL runs found for cross-attention run {cross_attention_run} under {non_federated_base}")
    ident = artifacts.hash_json({"definition": REPORT_DEFINITION_VERSION, "cross_attention_run": cross_attention_run, "runs": sorted(col["runs_used"]), "partial": include_partial,
                                 "n_bootstrap": n_bootstrap, "top_hospitals": list(top_hospitals), "seed": seed})
    rid, run_dir = artifacts.new_run_dir(Path(reports_base), "nonfl_report", ident, run_id)
    comp = comparisons(variants)
    header = ["variant", "config", "task", "embedding_set", "graph", "classifier", "sampling", "run_id", "n_hospitals", "n_trained", "n_skipped", "n_failed", "n_low_power_trained",
              "test_patients_in_trained_hospitals"] + [f"{m}_{k}" for m in ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "specificity") for k in ("macro", "weighted", "macro_powered", "n_defined")]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    skips = io.StringIO()
    sw = csv.writer(skips, lineterminator="\n")
    sw.writerow(["variant", "hospitalid", "status", "reason"])
    for v in sorted(variants):
        rec = variants[v]
        s, sm = rec["summary"]["spec"], rec["summary"]
        row = [v, s["config"], s["task"], s["embedding_set"], s["graph"], s["classifier"], s["sampling"], rec["run_id"], sm["n_hospitals"], sm["n_trained"], sm["n_skipped"],
               sm.get("n_failed", 0), sm["n_low_power_trained"], sm["test_patients_in_trained_hospitals"]]
        for m in ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "specificity"):
            x = sm["metrics"][m]
            row += [x["macro_mean"], x["sample_weighted_mean"], x["macro_mean_powered_hospitals"], x["n_defined"]]
        w.writerow(row)
        for r in rec["rows"]:
            if r["status"] != "trained":
                sw.writerow([v, r["hospitalid"], r["status"], r["skip_reason"]])
    run_dir.mkdir(parents=True)
    artifacts.atomic_write(run_dir / "summary_by_variant.csv", buf.getvalue())
    artifacts.atomic_write(run_dir / "skipped_hospitals.csv", skips.getvalue())
    artifacts.atomic_write(run_dir / "comparisons.json", json.dumps(comp, indent=1))
    # ---- aggregated per-class classification reports (all variants, both operating points) ----
    cls_json, cls_md = {}, ["# Per-class classification reports (Non-FL, hospital-wise)", "",
                            "Pooled = counts summed over the hospital models, each patient classified at its own hospital's validation-selected threshold. Test labels were read only after every "
                            "model and threshold was fixed.", ""]
    for v in sorted(variants):
        rows_v = variants[v]["rows"]
        spec = variants[v]["summary"]["spec"]
        rep = aggregate_class_reports(rows_v)
        have_rt = any((r.get("operating_points") or {}).get("recall_target") for r in rows_v)
        rep_rt = aggregate_class_reports([{"status": r["status"], "variant": v, "hospitalid": r["hospitalid"], "metrics": ((r.get("operating_points") or {}).get("recall_target") or {}).get("metrics")}
                                          for r in rows_v]) if have_rt else None
        cls_json[v] = {"spec": spec, "f1_max_operating_point": rep, "recall_target_operating_point": rep_rt}
        cls_md += [f"### config {spec['config']} · {spec['task']} · {v}", "", class_report_markdown(v, rep, "F1-max threshold")]
        if rep_rt:
            cls_md.append(class_report_markdown(v, rep_rt, "recall-target threshold"))
    artifacts.atomic_write(run_dir / "classification_reports.json", json.dumps(cls_json, indent=1, default=str))
    artifacts.atomic_write(run_dir / "classification_reports.md", "\n".join(cls_md) + "\n")
    # ---- pooled test metrics with patient-level bootstrap intervals (needs the saved predictions of a post-fix run) ----
    pooled = {}
    for v in sorted(variants):
        pf = variants[v]["pred_file"]
        if not pf:
            continue
        preds = load_predictions(pf).get(v)
        if not preds or len(preds["y"]) == 0:
            continue
        rows_v = [r for r in variants[v]["rows"] if r["status"] == "trained"]
        thr = {int(r["hospitalid"]): r["threshold"] for r in rows_v}
        size = {int(r["hospitalid"]): sum(r["counts"][k]["n"] for k in ("train", "val", "test")) for r in rows_v}
        ranked = [h for h, _ in sorted(size.items(), key=lambda kv: (-kv[1], kv[0]))]
        pooled[v] = {"all_trained_hospitals": pooled_metrics(preds, thr, None, n_bootstrap, seed)}
        for k in top_hospitals:
            pooled[v][f"top_{k}_largest_hospitals"] = pooled_metrics(preds, thr, ranked[:k], n_bootstrap, seed)
    artifacts.atomic_write(run_dir / "pooled_metrics.json", json.dumps(pooled, indent=1, default=str))
    pbuf = io.StringIO()
    pw = csv.writer(pbuf, lineterminator="\n")
    pw.writerow(["variant", "subset", "n_hospitals", "n_patients", "prevalence", "all_negative_accuracy"] + [f"{m}{sfx}" for m in ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "specificity") for sfx in ("", "_ci_lo", "_ci_hi")])
    for v, subsets in pooled.items():
        for sub, r in subsets.items():
            row = [v, sub, r["n_hospitals"], r["n_patients"], r["prevalence"], r["all_negative_accuracy"]]
            for m in ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "specificity"):
                ci = r["ci95"].get(m)
                row += [r["point"][m], ci[0] if ci else None, ci[1] if ci else None]
            pw.writerow(row)
    artifacts.atomic_write(run_dir / "pooled_metrics.csv", pbuf.getvalue())
    by_cfg = defaultdict(list)
    for v in sorted(variants):
        by_cfg[variants[v]["summary"]["spec"]["config"]].append(v)
    f3 = lambda x: "n/a" if x is None else f"{x:.3f}"       # noqa: E731
    md = ["# Non-FL hospital-wise report", "", f"Cross-attention run: `{cross_attention_run}`  |  Non-FL runs used: {', '.join(col['runs_used'])}", "",
          "**Regime.** One independent model per hospital and task from that hospital's own train/validation/test patients; no parameter sharing or aggregation. The encoders and the "
          "three-token cross-attention are centrally pretrained, so these are hospital-wise downstream results, not end-to-end learning, and no federated claim is made.", "",
          "**Not directly comparable** with pooled, federated or base-paper figures: different protocol (per-hospital models, macro / sample-weighted hospital averages), cohorts and splits.", "",
          "Metrics are hospital macro averages (every hospital equal) and test-sample-weighted averages. 'powered' = hospitals with at least 5 test patients in each class. Hospital-level "
          "metrics from few test patients are noisy.", ""]
    if col["partial_runs_excluded"]:
        md += [f"Partial (--max-hospitals) runs excluded: {', '.join(col['partial_runs_excluded'])}", ""]
    for c in sorted(by_cfg):
        md += [f"## Config {c}", "", "| variant | trained / skipped | AUROC macro (weighted) | AUPRC macro (weighted) | F1 macro | specificity macro |", "|---|---|---|---|---|---|"]
        for v in by_cfg[c]:
            sm = variants[v]["summary"]
            mm = sm["metrics"]
            md.append(f"| {v} | {sm['n_trained']} / {sm['n_skipped']} | {f3(mm['auroc']['macro_mean'])} ({f3(mm['auroc']['sample_weighted_mean'])}) | "
                      f"{f3(mm['auprc']['macro_mean'])} ({f3(mm['auprc']['sample_weighted_mean'])}) | {f3(mm['f1']['macro_mean'])} | {f3(mm['specificity']['macro_mean'])} |")
        md.append("")
    md += ["## Paired comparisons (delta = y minus x, over hospitals trained in both)", "", "| comparison | x | y | pairs | mean dAUROC [95% CI] | y better / tie / worse |", "|---|---|---|---|---|---|"]
    for cmp_ in comp:
        a = cmp_["auroc"]
        if a["n_paired_hospitals"]:
            md.append(f"| {cmp_['comparison']} | {cmp_['x']} | {cmp_['y']} | {a['n_paired_hospitals']} | {a['mean_delta']:+.4f} [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}] | {a['y_better']} / {a['ties']} / {a['y_worse']} |")
    rt_rows = [(v, variants[v]["summary"].get("recall_target_operating_point")) for v in sorted(variants) if variants[v]["summary"].get("recall_target_operating_point")]
    if rt_rows:
        md += ["", "## Secondary operating point: validation-selected threshold reaching the recall target (macro / sample-weighted over hospitals)", "",
               "| variant | target | recall | precision | F1 | accuracy | specificity |", "|---|---|---|---|---|---|---|"]
        for v, o in rt_rows:
            g = lambda m: f"{f3(o[m]['macro_mean'])} ({f3(o[m]['sample_weighted_mean'])})"       # noqa: E731
            md.append(f"| {v} | {o['target']} | {g('recall')} | {g('precision')} | {g('f1')} | {g('accuracy')} | {g('specificity')} |")
    if pooled:
        md += ["", "## Pooled test metrics with 95% patient-bootstrap intervals (each patient at its own hospital's validation threshold)", "",
               "Pooled AUROC mixes the score scales of independently trained hospital models. `prevalence` and the all-negative accuracy are given because accuracy is only informative relative to it.", "",
               "| variant | subset | hospitals | patients | prevalence | AUROC [95% CI] | F1 [95% CI] | recall | accuracy (all-negative) |", "|---|---|---|---|---|---|---|---|---|"]
        for v, subsets in pooled.items():
            for sub, r in subsets.items():
                ci = lambda m: ("" if not r["ci95"].get(m) else f" [{r['ci95'][m][0]:.3f}, {r['ci95'][m][1]:.3f}]")      # noqa: E731
                pt = r["point"]
                md.append(f"| {v} | {sub} | {r['n_hospitals']} | {r['n_patients']} | {f3(r['prevalence'])} | {f3(pt['auroc'])}{ci('auroc')} | {f3(pt['f1'])}{ci('f1')} | {f3(pt['recall'])} | {f3(pt['accuracy'])} ({f3(r['all_negative_accuracy'])}) |")
    md += ["", "## Reference only — NOT directly comparable", "", FLICS_REFERENCE["source"] + ". Different cohort (~14% vs about 9% mortality here), hospitals (their 3 / 10 largest), test set (pooled 1,000 patients) and thresholds (tuned on test).", "",
           "| scale | method | accuracy | AUROC | F1 |", "|---|---|---|---|---|"] + [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in FLICS_REFERENCE["rows"]]
    md += ["", "Per-class reports: `classification_reports.md` / `.json`. Skipped hospitals and reasons: `skipped_hospitals.csv`. Failed runs (unexpected errors) are listed there with status `failed`."]
    artifacts.atomic_write(run_dir / "report.md", "\n".join(md) + "\n")
    names = ["summary_by_variant.csv", "skipped_hospitals.csv", "comparisons.json", "report.md", "classification_reports.json", "classification_reports.md", "pooled_metrics.json", "pooled_metrics.csv"]
    man = {"run_id": rid, "definition_version": REPORT_DEFINITION_VERSION, "identity": ident, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "cross_attention_run": cross_attention_run, "nonfl_runs_used": col["runs_used"], "partial_runs_excluded": col["partial_runs_excluded"], "n_variants": len(variants), "n_pooled_variants": len(pooled),
           "n_comparisons": len(comp), "files": {n: artifacts.sha256_file(run_dir / n) for n in names}, "seconds": round(time.time() - t0, 1)}
    artifacts.atomic_write(run_dir / "manifest.json", json.dumps(man, indent=1))
    return {"run_dir": str(run_dir), "run_id": rid, "manifest": man, "comparisons": comp}
