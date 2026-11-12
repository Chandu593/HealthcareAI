"""FL vs Non-FL consolidated report (`fl_report` stage).

Reads the COMPLETED, hash-verified Non-FL runs (frozen `_v2` runs are only read, never modified) and the completed FL runs of one cross-attention run, and compares each FL arm with its Non-FL
counterpart(s) on the PAIRED set: the hospitals that both regimes evaluated, with the SAME test patients (verified patient-by-patient from the saved predictions; a mismatch is reported as
`identical_test_patients = false` and the pooled comparison for that pair is withheld).

Hospital-macro and pooled results are reported in SEPARATE tables:
  * hospital-macro: per-hospital paired deltas (FL - Non-FL) with a bootstrap-over-hospitals CI and a Wilcoxon signed-rank p-value (the primary comparison);
  * pooled: all paired test patients together, patient-bootstrap CIs, and a paired patient-bootstrap CI of the pooled-AUROC difference. A federated global model has ONE score scale across
    hospitals, whereas independent hospital models do not, so pooled AUROC favours FL structurally; hospital-macro is the comparison to lead with.
Arms are listed with their federated VALIDATION AUROC so any headline arm can be chosen on validation, never on test.

Counterparts: f1 LR/MLP -> c1 LR/MLP; f1/f4 FedGBDT -> c1/c4 LightGBM and GBDT (loss differs: logistic vs focal / standard); f2 -> c2; f3 -> c3; f4 LR/MLP -> c4 LR/MLP;
personalised MLP -> c1 MLP; personalised GCN (graph A) -> c2 graph A.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import wilcoxon

from src.three_modality import artifacts
from src.three_modality import nonfl_report as nr
from src.three_modality.fl_config import FL_DEFINITION_VERSION, FL_REGIME

logger = logging.getLogger("eicu_fl")

FL_REPORT_VERSION = "fl-report-v1"
METRICS = ("auroc", "auprc", "f1")


def collect_fl_runs(federated_base, cross_attention_run: str, include_partial: bool = False) -> dict:
    """Completed, hash-verified FL runs of this cross-attention run; for a variant present in several runs the NEWEST run wins."""
    chosen, used, skipped_partial = {}, [], []
    for r in sorted(artifacts.list_runs(Path(federated_base)), key=lambda r: (r.get("created_utc", ""), r.get("run_id", ""))):
        if r.get("cross_attention_run") != cross_attention_run or not (str(r.get("definition_version", "")).startswith("fl-v") and "variants" in r and r.get("kind") != "fl_tune"):
            continue
        if r.get("partial") and not include_partial:
            skipped_partial.append(r["run_id"])
            continue
        d = Path(r["run_dir"])
        if not all((d / n).is_file() and artifacts.sha256_file(d / n) == h for n, h in r["files"].items()):
            raise ValueError(f"FL run {r['run_id']}: files do not match their recorded hashes; refusing to report on a modified run")
        rows = json.loads((d / "per_hospital.json").read_text(encoding="utf-8"))
        summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        for v in r["variants"]:
            tl = d / "training_logs" / f"{v}.json"
            chosen[v] = {"run_id": r["run_id"], "rows": [x for x in rows if x["variant"] == v], "summary": summ[v], "kind": r["kind"], "pred_file": str(d / "predictions.npz"),
                         "training": json.loads(tl.read_text(encoding="utf-8")) if tl.is_file() else {}, "created_utc": r["created_utc"]}
        used.append(r["run_id"])
    return {"variants": chosen, "runs_used": used, "partial_runs_excluded": skipped_partial}


def counterparts(spec: dict) -> list:
    t = spec["task"]
    c, clf = spec["config"], spec["classifier"]
    g = f"graph{spec['graph']}" if spec.get("graph") else None
    if c == 1:
        base = f"c1__{spec['embedding_set']}__{t}__"
        return [base + "lightgbm", base + "gbdt"] if clf == "gbdt_fed" else [base + clf]
    if c in (2, 3):
        return [f"c{c}__{g}__{t}"]
    if c == 4:
        base = f"c4__{g}__{t}__"
        return [base + "lightgbm", base + "gbdt"] if clf == "gbdt_fed" else [base + clf]
    if c == 5:
        return [f"c1__fused__{t}__mlp"] if clf == "mlp" else [f"c2__graph{spec['graph']}__{t}"]
    return []


def _pred(npz, variant: str) -> "dict | None":
    k = variant + "@pid"
    if k not in npz.files:
        return None
    return {"pid": npz[variant + "@pid"], "hid": npz[variant + "@hid"], "y": npz[variant + "@y"], "p": npz[variant + "@p"]}


def _restrict(pred: dict, hospitals: set) -> dict:
    m = np.isin(pred["hid"], list(hospitals))
    o = np.argsort(pred["pid"][m], kind="stable")
    return {k: v[m][o] for k, v in pred.items()}


def _wilcoxon(a: dict, b: dict, metric: str) -> "float | None":
    common = sorted(set(a) & set(b))
    d = np.array([b[h] - a[h] for h in common], dtype=float)
    if len(d) < 6 or np.allclose(d, 0):
        return None
    try:
        return float(wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        return None


def _paired_pooled_auroc(pa: dict, pb: dict, n_boot: int, seed: int) -> dict:
    """pooled AUROC of Non-FL (pa) and FL (pb) on the same patients + paired patient-bootstrap CI of the difference (FL - Non-FL)."""
    y = pa["y"].astype(int)
    if len(np.unique(y)) < 2:
        return {"pooled_auroc_nonfl": None, "pooled_auroc_fl": None, "delta": None, "delta_ci95": None}
    a, b = roc_auc_score(y, pa["p"]), roc_auc_score(y, pb["p"])
    rs = np.random.default_rng(seed)
    n, ds = len(y), []
    for _ in range(int(n_boot)):
        i = rs.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        ds.append(roc_auc_score(y[i], pb["p"][i]) - roc_auc_score(y[i], pa["p"][i]))
    return {"pooled_auroc_nonfl": float(a), "pooled_auroc_fl": float(b), "delta": float(b - a), "delta_ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))] if ds else None,
            "pooled_auprc_nonfl": float(average_precision_score(y, pa["p"])), "pooled_auprc_fl": float(average_precision_score(y, pb["p"]))}


def compare(nonfl: dict, fl: dict, n_boot: int, pooled_boot: int, seed: int, nonfl_npz: dict, fl_npz: dict) -> list:
    out = []
    for v, f in sorted(fl["variants"].items()):
        spec = f["summary"].get("spec") or {}
        fl_rows = [r for r in f["rows"] if r["status"] == "trained" and r["eval_set"] == "paired"]
        for nv in counterparts(spec):
            n = nonfl["variants"].get(nv)
            if n is None:
                out.append({"fl_variant": v, "nonfl_variant": nv, "task": spec.get("task"), "available": False})
                continue
            nrows = [r for r in n["rows"] if r["status"] == "trained"]
            common = {r["hospitalid"] for r in fl_rows} & {r["hospitalid"] for r in nrows}
            rec = {"fl_variant": v, "nonfl_variant": nv, "task": spec.get("task"), "algorithm": spec.get("algorithm"), "method": spec.get("method"), "available": True, "n_common_hospitals": len(common),
                   "n_nonfl_trained": len(nrows), "n_fl_paired_trained": len(fl_rows)}
            fx = [r for r in nrows if r["hospitalid"] in common]
            fy = [r for r in fl_rows if r["hospitalid"] in common]
            for m in METRICS:
                a, b = nr._by_hospital(fx, m), nr._by_hospital(fy, m)
                p = nr.paired(fx, fy, m, n_boot=n_boot, seed=seed)
                rec[f"macro_{m}_nonfl"] = float(np.mean([a[h] for h in sorted(set(a) & set(b))])) if set(a) & set(b) else None
                rec[f"macro_{m}_fl"] = float(np.mean([b[h] for h in sorted(set(a) & set(b))])) if set(a) & set(b) else None
                rec[f"macro_{m}_delta"], rec[f"macro_{m}_delta_ci95"] = p.get("mean_delta"), p.get("ci95")
                rec[f"macro_{m}_fl_better"], rec[f"macro_{m}_nonfl_better"] = p.get("y_better"), p.get("y_worse")
                rec[f"macro_{m}_wilcoxon_p"] = _wilcoxon(a, b, m)
            pa, pb = _pred(nonfl_npz[n["pred_file"]], nv) if n.get("pred_file") else None, _pred(fl_npz[f["pred_file"]], v)
            rec["identical_test_patients"] = None
            if pa is not None and pb is not None and common:
                ra, rb = _restrict(pa, common), _restrict(pb, common)
                same = bool(len(ra["pid"]) == len(rb["pid"]) and np.array_equal(ra["pid"], rb["pid"]) and np.array_equal(ra["hid"], rb["hid"]) and np.array_equal(ra["y"], rb["y"]))
                rec["identical_test_patients"] = same
                rec["n_pooled_test_patients"] = int(len(ra["pid"]))
                if same:
                    rec.update(_paired_pooled_auroc(ra, rb, pooled_boot, seed))
                else:
                    rec["pooled_withheld"] = "the two regimes' test patients / labels / hospital mapping differ for the common hospitals"
            out.append(rec)
    return out


def _fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def _ci(ci):
    return "" if not ci else f" [{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _md(comp: list, fl: dict, nonfl: dict, manifest: dict) -> str:
    L = ["# FL vs Non-FL (hospital-wise) — consolidated report", "", f"> {FL_REGIME}", "",
         f"Cross-attention run: `{manifest['cross_attention_run']}`. FL runs used: {', '.join(manifest['fl_runs_used']) or 'none'}. Non-FL runs used (read-only): {', '.join(manifest['nonfl_runs_used'])}.", "",
         "delta = FL − Non-FL (positive = FL better). The paired set = hospitals evaluated by BOTH regimes, with identical test patients (checked patient by patient). Hospital-macro and pooled results "
         "are separate tables. Nothing here is directly comparable with base-paper or paper-track figures (different cohorts, windows and protocols).", ""]
    for task in sorted({c["task"] for c in comp if c.get("task")}):
        rows = [c for c in comp if c.get("task") == task and c.get("available")]
        L += [f"## {task}", "", "### Hospital-macro, paired hospitals (primary comparison)", "",
              "| FL arm | Non-FL counterpart | n hosp | macro AUROC Non-FL → FL | Δ AUROC [95% CI] | Wilcoxon p | FL better / worse | Δ AUPRC [95% CI] | Δ F1 [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
        for c in rows:
            L.append(f"| {c['fl_variant']} | {c['nonfl_variant']} | {c['n_common_hospitals']} | {_fmt(c['macro_auroc_nonfl'])} → {_fmt(c['macro_auroc_fl'])} | {_fmt(c['macro_auroc_delta'])}{_ci(c['macro_auroc_delta_ci95'])} | "
                     f"{_fmt(c['macro_auroc_wilcoxon_p'], 4)} | {c['macro_auroc_fl_better']}/{c['macro_auroc_nonfl_better']} | {_fmt(c['macro_auprc_delta'])}{_ci(c['macro_auprc_delta_ci95'])} | "
                     f"{_fmt(c['macro_f1_delta'])}{_ci(c['macro_f1_delta_ci95'])} |")
        L += ["", "### Pooled over the same paired test patients (separate from the table above)", "",
              "| FL arm | Non-FL counterpart | identical test patients | n test patients | pooled AUROC Non-FL → FL | Δ [95% paired patient-bootstrap CI] | pooled AUPRC Non-FL → FL |", "|---|---|---|---|---|---|---|"]
        for c in rows:
            L.append(f"| {c['fl_variant']} | {c['nonfl_variant']} | {c['identical_test_patients']} | {c.get('n_pooled_test_patients', '–')} | {_fmt(c.get('pooled_auroc_nonfl'))} → {_fmt(c.get('pooled_auroc_fl'))} | "
                     f"{_fmt(c.get('delta'))}{_ci(c.get('delta_ci95'))} | {_fmt(c.get('pooled_auprc_nonfl'))} → {_fmt(c.get('pooled_auprc_fl'))} |")
        L += ["", "### FL arms: federated validation AUROC (select the headline arm HERE, on validation), runtime, communication", "",
              "| FL arm | val AUROC (hist) | tie bound | exact diag | best round / iter | rounds run | stop | seconds | comm MB total | paired / FL-only clients evaluated |", "|---|---|---|---|---|---|---|---|---|---|"]
        for v, f in sorted(fl["variants"].items()):
            if (f["summary"].get("spec") or {}).get("task") != task:
                continue
            t, cn = f["training"], f["summary"]["counts"]
            L.append(f"| {v} | {_fmt(t.get('val_auroc_hist_at_best'), 4)} | {_fmt(t.get('val_auroc_tie_bound_at_best'), 4)} | {_fmt(t.get('val_auroc_exact_diag'), 4)} | "
                     f"{t.get('best_round', t.get('best_iteration'))} | {t.get('rounds_run', t.get('n_rounds_run'))} | {t.get('stop_reason', '')} | {t.get('seconds')} | "
                     f"{_fmt(t.get('comm_mb_total', t.get('comm_gb_total_sparse_all_clients')), 2)} | {cn['n_paired_evaluated']} / {cn['n_fl_only_evaluated']} |")
        L.append("")
    L += ["## Notes", "", "* Encoders and cross-attention are centrally pretrained; only the downstream stage is federated.",
          "* FedGBDT uses the logistic loss; Non-FL LightGBM uses a focal objective and Non-FL GBDT the standard loss, so FedGBDT is shown against both.",
          "* `fl_only` clients (hospitals the Non-FL regime had to skip) are evaluated and reported in each run's `summary.json` (`all_evaluated_clients`) but are excluded from every paired comparison here.", ""]
    return "\n".join(L)


def build_fl_report(non_federated_base, federated_base, reports_base, cross_attention_run: str, *, include_partial: bool = False, run_id: "str | None" = None, n_bootstrap: int = 2000,
                    pooled_boot: int = 300, seed: int = 42) -> dict:
    t0 = time.time()
    nonfl = nr.collect_runs(non_federated_base, cross_attention_run, include_partial)
    fl = collect_fl_runs(federated_base, cross_attention_run, include_partial)
    if not fl["variants"]:
        raise ValueError(f"no completed FL runs for cross-attention run {cross_attention_run!r} under {federated_base}")
    if not nonfl["variants"]:
        raise ValueError(f"no completed Non-FL runs for cross-attention run {cross_attention_run!r} under {non_federated_base}")
    nonfl_npz = {p: np.load(p, allow_pickle=True) for p in {v["pred_file"] for v in nonfl["variants"].values() if v.get("pred_file")}}
    fl_npz = {p: np.load(p, allow_pickle=True) for p in {v["pred_file"] for v in fl["variants"].values()}}
    comp = compare(nonfl, fl, n_bootstrap, pooled_boot, seed, nonfl_npz, fl_npz)
    ident = artifacts.hash_json({"definition": FL_REPORT_VERSION, "ca": cross_attention_run, "fl": sorted(fl["runs_used"]), "nonfl": sorted(nonfl["runs_used"]), "n_boot": n_bootstrap,
                                 "pooled_boot": pooled_boot, "seed": seed})
    rid, run_dir = artifacts.new_run_dir(Path(reports_base), "fl_report", ident, run_id)
    run_dir.mkdir(parents=True)
    manifest = {"run_id": rid, "definition_version": FL_REPORT_VERSION, "fl_definition_version": FL_DEFINITION_VERSION, "kind": "fl_report", "identity": ident, "cross_attention_run": cross_attention_run,
                "fl_runs_used": fl["runs_used"], "nonfl_runs_used": nonfl["runs_used"], "partial_runs_excluded": fl["partial_runs_excluded"], "n_comparisons": len(comp),
                "n_identical_test_patient_checks_failed": sum(1 for c in comp if c.get("identical_test_patients") is False), "regime": FL_REGIME,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_bootstrap": n_bootstrap, "pooled_paired_bootstrap": pooled_boot, "seed": seed}
    md = _md(comp, fl, nonfl, manifest)
    flat = sorted({k for c in comp for k in c})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=flat, lineterminator="\n")
    w.writeheader()
    w.writerows([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in c.items()} for c in comp])
    artifacts.atomic_write(run_dir / "report.md", md)
    artifacts.atomic_write(run_dir / "comparison.csv", buf.getvalue())
    artifacts.atomic_write(run_dir / "comparison.json", json.dumps(comp, indent=1, default=str))
    manifest["files"] = {n: artifacts.sha256_file(run_dir / n) for n in ("report.md", "comparison.csv", "comparison.json")}
    manifest["seconds"] = round(time.time() - t0, 1)
    artifacts.atomic_write(run_dir / "manifest.json", json.dumps(manifest, indent=1, default=str))
    return {"run_id": rid, "run_dir": str(run_dir), "manifest": manifest, "comparisons": comp}
