"""Paper-track orchestrator: split reuse -> window-bounded features -> arms (LR / tuned LightGBM /
centralized MLP / FedAvg MLP / FedAvg + Graph none|A|B) x seeds -> calibrated, thresholded, bootstrapped
evaluation -> results.json / results_table.csv / predictions / paper_report.md."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from src.data.splitting.patient_split import build_and_save_split_manifest
from src.paper.config import get_paper_cfg
from src.paper.evaluation import (apply_local_intercepts, auprc, binary_report, fit_local_intercepts, bootstrap_ci, fast_auroc, logit, paired_bootstrap_delta,
                                  per_hospital_summary, platt_fit, sigmoid, threshold_max_f1)
from src.paper.features import FeatureBundle, build_feature_bundle
from src.paper.label_validity import hospital_medication_coverage
from src.paper.gbdt import fit_logreg, fit_xgboost, predict_lightgbm, tune_and_fit_lightgbm
from src.classifiers.imbalance_sampling import hospital_wise_resample
from src.paper.preprocess import prepare_sequence, prepare_tabular
from src.paper.cohort_filters import combine_filters, hospital_size_mask
from src.paper.fed_gbdt import FIXED_DEFAULTS as _GBDT_KEYS, reuse_params_from_run, search_and_train as fed_gbdt_search_and_train
from src.paper.trainers import (NeuralData, build_graphs, embed, pos_weight_value, predict_logits, run_fedavg, run_graph_fedavg,
                                train_central, train_local_only)

logger = logging.getLogger("eicu_fl")
KNOWN_ARMS = ("logreg_central", "lightgbm_central", "xgboost_central", "mlp_central", "mlp_local_only", "fedavg_mlp", "fedavg_logreg", "fedavg_graph", "lightgbm_sampling", "fed_gbdt", "gbdt_central")
GRAPH_METHODS = {"none": "none", "A": "knn_and_threshold", "B": "minibatch_kmeans_and_threshold"}


def _resolve_under_output_dir(cfg, p: str) -> str:
    path = Path(p)
    return str(path) if path.is_absolute() else str(Path(cfg.get("run.output_dir", "outputs")).parent / path)


def load_or_build_split(cfg, partition_result, phase3_result, overwrite=False):
    sc = cfg.get("ablation.split", {}) or {}
    path = _resolve_under_output_dir(cfg, sc.get("manifest_path", "outputs/ablation/split_manifest.json"))
    return build_and_save_split_manifest(
        partition_result, phase3_result["cohort_patients"], phase3_result["mortality"], phase3_result["sepsis_df"], path,
        train_frac=sc.get("train_frac", 0.7), val_frac=sc.get("val_frac", 0.15), test_frac=sc.get("test_frac", 0.15),
        seed=cfg.get("run.seed", 42), stratify_on=sc.get("stratify_on", "mortality"),
        sepsis_cohort=cfg.get("multitask.sepsis_cohort", "broad"), overwrite=overwrite)


def get_bundle(pcfg, cfg, phase1_summary, phase3_result, split_manifest, out_root: Path) -> FeatureBundle:
    train_ids = split_manifest.all_patient_ids("train")
    cp = phase3_result["cohort_patients"]
    key = hashlib.sha1(json.dumps([len(cp), int(np.sum(train_ids) % 10**9), len(train_ids), pcfg["window"], {k: v for k, v in pcfg["features"].items() if k != "cache"},
                                   pcfg["sequence"], cfg.get("multitask.sepsis_cohort", "broad")], sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = out_root.parent / "feature_cache" / f"bundle_{key}.npz"
    if pcfg["features"]["cache"] and path.exists():
        logger.info("[PAPER] reusing cached feature bundle %s", path)
        return FeatureBundle.load(path)
    b = build_feature_bundle(pcfg, phase1_summary, cp, phase3_result["mortality"], phase3_result["sepsis_df"], train_ids,
                             sepsis_cohort=cfg.get("multitask.sepsis_cohort", "broad"))
    if pcfg["features"]["cache"]:
        b.save(path)
    return b


def _edge_homophily(graphs, y):
    agree = tot = 0; pos = 0; n = 0
    for g in graphs.values():
        rows = g["rows"][g["train_local"]]; ei = g["train_ei"].numpy()
        if ei.shape[1]:
            yy = y[rows]; agree += int((yy[ei[0]] == yy[ei[1]]).sum()); tot += ei.shape[1]
        pos += int((y[rows] == 1).sum()); n += len(rows)
    prev = pos / max(n, 1)
    return {"n_train_edges": tot, "edge_label_agreement": (agree / tot) if tot else None, "expected_under_random": prev ** 2 + (1 - prev) ** 2}


def _evaluate_arm(name, seed, score, kind, y, sp, hosp, pids, ecfg, out_dir: Path, tag: str, extra=None) -> dict:
    s = logit(score) if kind == "prob" else np.asarray(score, dtype="float64")
    va, te = sp == "val", sp == "test"
    a, b = platt_fit(s[va], y[va]) if ecfg["calibrate"] else (1.0, 0.0)
    p = sigmoid(a * s + b)
    thr = threshold_max_f1(y[va], p[va])
    rec = {"arm": name, "seed": seed, "calibration": {"a": a, "b": b}, "threshold_rule": ecfg["threshold_rule"], "extra": extra or {}}
    for split in ("train", "val", "test"):
        m = sp == split
        rec[split] = binary_report(y[m], p[m], thr)
    shifts = fit_local_intercepts(s[va], y[va], hosp[va], a, b, ecfg.get("local_calibration_prior_sd", 1.0))
    pl = apply_local_intercepts(s, hosp, a, b, shifts); thr_l = threshold_max_f1(y[va], pl[va])
    rec["local_calibrated"] = {split: binary_report(y[sp == split], pl[sp == split], thr_l) for split in ("train", "val", "test")}
    rec["local_calibrated"]["per_hospital_test"] = per_hospital_summary(y[te], pl[te], hosp[te], ecfg["per_hospital_min_pos"], ecfg["per_hospital_min_neg"])
    tm = sp == "test"
    rec["test_ci"] = {"auroc": bootstrap_ci(y[tm], p[tm], "auroc", ecfg["bootstrap_n"], ecfg["ci_alpha"], seed),
                      "auprc": bootstrap_ci(y[tm], p[tm], "auprc", ecfg["bootstrap_n"], ecfg["ci_alpha"], seed)}
    rec["test_per_hospital"] = per_hospital_summary(y[tm], p[tm], hosp[tm], ecfg["per_hospital_min_pos"], ecfg["per_hospital_min_neg"])
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = np.isin(sp, ("train", "val", "test"))
    np.savez_compressed(out_dir / f"{tag}__{name}__seed{seed}.npz", patient_id=pids[keep], hospital_id=hosp[keep], split=sp[keep].astype(str),
                        y=y[keep], score=s[keep].astype("float32"), p_calibrated=p[keep].astype("float32"), p_local_calibrated=pl[keep].astype("float32"))
    rec["_p"] = p
    logger.info("[PAPER] %-22s seed=%s test AUROC=%.4f [%.4f,%.4f] AUPRC=%.4f F1=%.4f", name, seed, rec["test"]["auroc"],
                rec["test_ci"]["auroc"]["lo"], rec["test_ci"]["auroc"]["hi"], rec["test"]["auprc"], rec["test"]["f1"])
    return rec


def _run_task_featureset(task, fs, bundle, split, pcfg, out_root, arms_filter=None, task_mask=None, filter_info=None):
    ncfg, fcfg, gcfg, ecfg = pcfg["neural"], pcfg["federated"], pcfg["graph"], pcfg["evaluation"]
    y_all = bundle.y_mortality if task == "mortality" else bundle.y_sepsis
    valid = np.isfinite(y_all)
    if task_mask is not None:
        valid = valid & task_mask             # e.g. sepsis: label-valid hospitals only
    sp = np.where(valid, split, "none")
    tr, va, te = (np.where(sp == s)[0] for s in ("train", "val", "test"))
    for nm, idx in (("train", tr), ("val", va), ("test", te)):
        if len(idx) == 0 or len(np.unique(y_all[idx])) < 2:
            raise ValueError(f"[{task}/{fs}] the {nm} split needs both classes (n={len(idx)}).")
    excl = pcfg["feature_sets"][fs]["exclude_groups"]
    cols = bundle.columns_excluding(excl, pcfg["feature_sets"][fs].get("exclude_name_regex"))
    fnames = [bundle.feature_names[i] for i in cols]
    X = bundle.X[:, cols]
    N = X.shape[0]
    logger.info("[PAPER] === task=%s feature_set=%s | %d features | train/val/test = %d/%d/%d | prevalence(train)=%.3f ===",
                task, fs, X.shape[1], len(tr), len(va), len(te), float(y_all[tr].mean()))
    hosp, pids = bundle.hospital_ids, bundle.patient_ids
    seeds = list(pcfg["seeds"]); arms = pcfg["arms"]
    on = lambda a: arms.get(a, False) and (arms_filter is None or a in arms_filter)
    tag = f"{task}__{fs}"; pred_dir = out_root / "predictions"; records = []
    train_by_hosp = {int(h): tr[hosp[tr] == h] for h in np.unique(hosp[tr])}

    Xn, info = prepare_tabular(X, hosp, sp == "train", ncfg)
    seq = None
    if pcfg["sequence"]["enabled"] and ncfg.get("use_sequence", True) and bundle.seq_values is not None and bundle.seq_labs:
        keep_labs = []
        for j, nm in enumerate(bundle.seq_labs):
            g = next((bundle.feature_groups[i] for i, n in enumerate(bundle.feature_names) if n == f"lab_{nm}_mean"), "lab")
            if g not in excl:
                keep_labs.append(j)
        if keep_labs:
            seq = prepare_sequence(bundle.seq_values[:, :, keep_labs], hosp, sp == "train", ncfg)
    import torch
    data = NeuralData(torch.from_numpy(Xn), torch.from_numpy(seq) if seq is not None else None, y_all.astype(np.float32), hosp)
    d_tab, d_seq = Xn.shape[1], (seq.shape[2] if seq is not None else 0)
    logger.info("[PAPER] neural input: tab dim=%d (%s) seq=%s", d_tab, info, None if seq is None else seq.shape)
    all_idx = np.arange(N)

    if on("logreg_central"):
        m = fit_logreg(Xn[tr], y_all[tr], pcfg["logreg"], seeds[0])
        records.append(_evaluate_arm("logreg_central", seeds[0], m.predict_proba(Xn)[:, 1], "prob", y_all, sp, hosp, pids, ecfg, pred_dir, tag))
    fixed = None
    if on("lightgbm_central"):
        for sd in seeds:
            mdl, lg = tune_and_fit_lightgbm(X[tr], y_all[tr], X[va], y_all[va], pcfg["lightgbm"], sd, fixed)
            fixed = fixed or lg["best_params"]
            imp = mdl.booster_.feature_importance(importance_type="gain"); top = np.argsort(-imp)[:30]
            records.append(_evaluate_arm("lightgbm_central", sd, predict_lightgbm(mdl, X), "prob", y_all, sp, hosp, pids, ecfg, pred_dir, tag,
                                         {"best_params": lg["best_params"], "best_val_auroc": lg["best_val_auroc"], "best_iteration": lg["best_iteration"],
                                          "top_features_gain_share": [(fnames[i], float(imp[i] / max(imp.sum(), 1e-12))) for i in top]}))
    if on("lightgbm_sampling"):
        if fixed is None:
            _, lg0 = tune_and_fit_lightgbm(X[tr], y_all[tr], X[va], y_all[va], pcfg["lightgbm"], seeds[0]); fixed = lg0["best_params"]
        fp = dict(fixed, scale_pos_weight_mode="none")           # resampling already rebalances -- do not double-correct
        for strat in pcfg["sampling"]["strategies"]:
            best = None; searched = {}
            for ratio in pcfg["sampling"]["ratios"]:
                Xr, yr = hospital_wise_resample(X[tr], y_all[tr], hosp[tr], strat, ratio, seeds[0], logger=logger)
                m_, _ = tune_and_fit_lightgbm(Xr, yr, X[va], y_all[va], pcfg["lightgbm"], seeds[0], fp)
                pv = predict_lightgbm(m_, X[va]); score = auprc(y_all[va], pv); searched[str(ratio)] = float(score)   # ratio chosen on VAL AUPRC only
                if best is None or score > best[0]:
                    best = (score, ratio, m_)
            records.append(_evaluate_arm(f"lightgbm_{strat}", seeds[0], predict_lightgbm(best[2], X), "prob", y_all, sp, hosp, pids, ecfg, pred_dir, tag,
                                         {"ratio": best[1], "ratio_search_val_auprc": searched}))
    if on("fed_gbdt"):
        fixed_g = None                                   # trial search on the first seed only; later seeds reuse the winner (as lightgbm_central does)
        for sd in seeds:
            mdl_g, gh = fed_gbdt_search_and_train(X, y_all, train_by_hosp, va, pcfg["fed_gbdt"], sd, fixed_g)
            fixed_g = fixed_g or {k: gh["params"][k] for k in _GBDT_KEYS}
            imp = mdl_g.gain_by_feature; top = np.argsort(-imp)[:30]
            gh["top_features_gain_share"] = [(fnames[i], float(imp[i] / max(imp.sum(), 1e-12))) for i in top]
            records.append(_evaluate_arm("fed_gbdt", sd, mdl_g.predict_raw(X), "logit", y_all, sp, hosp, pids, ecfg, pred_dir, tag, gh))
    if on("gbdt_central"):                               # the SAME histogram GBDT code on ONE client that holds every training row (pooled): isolates the federation effect for this model
        reuse = (pcfg.get("gbdt_central") or {}).get("reuse_params_from")
        fixed_c = reuse_params_from_run(reuse, f"{task}/{fs}") if reuse else None      # the federated winner's configuration; None = its own validation search on the first seed
        for sd in seeds:
            mdl_c, ch = fed_gbdt_search_and_train(X, y_all, {0: tr}, va, pcfg["fed_gbdt"], sd, fixed_c)
            fixed_c = fixed_c or {k: ch["params"][k] for k in _GBDT_KEYS}
            if ch["n_clients"] != 1 or ch["n_train"] != len(tr):
                raise RuntimeError("gbdt_central must train on one client holding exactly the training rows")
            ch["pooled"] = True
            records.append(_evaluate_arm("gbdt_central", sd, mdl_c.predict_raw(X), "logit", y_all, sp, hosp, pids, ecfg, pred_dir, tag, ch))
    if on("xgboost_central"):
        mdl = fit_xgboost(X[tr], y_all[tr], X[va], y_all[va], seeds[0])
        records.append(_evaluate_arm("xgboost_central", seeds[0], mdl.predict_proba(X)[:, 1], "prob", y_all, sp, hosp, pids, ecfg, pred_dir, tag))
    for sd in seeds:
        if on("mlp_central"):
            mdl, h = train_central(data, tr, va, d_tab, d_seq, ncfg, sd)
            records.append(_evaluate_arm("mlp_central", sd, predict_logits(mdl, data, all_idx), "logit", y_all, sp, hosp, pids, ecfg, pred_dir, tag, h))
        if on("mlp_local_only"):
            rows = {h: (train_by_hosp.get(h, np.zeros(0, dtype=np.int64)), va[hosp[va] == h], np.where(hosp == h)[0]) for h in np.unique(hosp)}
            sc = train_local_only(data, rows, d_tab, d_seq, ncfg, sd)
            records.append(_evaluate_arm("mlp_local_only", sd, np.nan_to_num(sc), "logit", y_all, sp, hosp, pids, ecfg, pred_dir, tag))
        if on("fedavg_logreg"):
            gl, hl = run_fedavg(data, train_by_hosp, va, d_tab, 0, ncfg, dict(fcfg, **pcfg["federated_logreg"]), sd, kind="linear")
            records.append(_evaluate_arm("fedavg_logreg", sd, predict_logits(gl, data, all_idx), "logit", y_all, sp, hosp, pids, ecfg, pred_dir, tag, hl))
        if on("fedavg_mlp") or on("fedavg_graph"):
            gmodel, h = run_fedavg(data, train_by_hosp, va, d_tab, d_seq, ncfg, fcfg, sd)
            base_logit, emb = embed(gmodel, data, all_idx)
            records.append(_evaluate_arm("fedavg_mlp", sd, base_logit, "logit", y_all, sp, hosp, pids, ecfg, pred_dir, tag, h))
            if on("fedavg_graph"):
                pw = pos_weight_value(y_all[tr], ncfg["pos_weight"])
                for gname, method in GRAPH_METHODS.items():
                    graphs = build_graphs(emb, hosp, pids, sp, gcfg, method)
                    homo = _edge_homophily(graphs, np.nan_to_num(y_all, nan=-1))
                    s_g, gh = run_graph_fedavg(emb, base_logit, np.nan_to_num(y_all, nan=0.0).astype(np.float32), graphs, sp, gcfg, ncfg, sd, pw)
                    gh["homophily"] = homo
                    records.append(_evaluate_arm(f"fedavg_graph_{gname}", sd, s_g, "logit", y_all, sp, hosp, pids, ecfg, pred_dir, tag, gh))
    # paired bootstrap vs the configured comparison arm (first seed), plus APACHE benchmark
    cmp_name = ecfg["compare_to"]; by = {(r["arm"], r["seed"]): r for r in records}
    ref = by.get((cmp_name, seeds[0]))
    te_m = sp == "test"
    comparisons = {}
    if ref is not None:
        for (a, s), r in by.items():
            if s == seeds[0] and a != cmp_name:
                comparisons[a] = {m: paired_bootstrap_delta(y_all[te_m], r["_p"][te_m], ref["_p"][te_m], m, ecfg["bootstrap_n"], ecfg["ci_alpha"], seeds[0]) for m in ("auroc", "auprc")}
    bench = {}
    if task == "mortality":
        k = te_m & np.isfinite(bundle.apache_pred_mortality)
        if k.sum() > 0 and len(np.unique(y_all[k])) == 2:
            bench["apache_iva_predicted_mortality"] = {"n": int(k.sum()), "auroc": bootstrap_ci(y_all[k], bundle.apache_pred_mortality[k], "auroc", ecfg["bootstrap_n"], ecfg["ci_alpha"], 0),
                                                       "auprc": auprc(y_all[k], bundle.apache_pred_mortality[k])}
    for r in records:
        r.pop("_p", None)
    return {"cohort_filter": filter_info, "n_features": int(X.shape[1]), "excluded_groups": list(excl), "records": records, "comparisons_vs_" + cmp_name: comparisons, "benchmarks": bench,
            "split_sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))}}


def _aggregate(records):
    agg = {}
    for r in records:
        agg.setdefault(r["arm"], []).append(r)
    out = {}
    for a, rs in agg.items():
        out[a] = {"n_seeds": len(rs)}
        for m in ("auroc", "auprc", "f1", "brier", "ece"):
            v = np.array([x["test"][m] for x in rs], dtype=float)
            out[a][m] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}
        vl = np.array([x["local_calibrated"]["test"]["auroc"] for x in rs], dtype=float)
        out[a]["auroc_local"] = {"mean": float(np.nanmean(vl)), "std": float(np.nanstd(vl))}
        out[a]["auroc_ci_seed0"] = rs[0]["test_ci"]["auroc"]
    return out


def write_report(results: dict, out_root: Path, pcfg: dict) -> str:
    L = [f"# Fed-THOR paper-track results ({results['run_id']})", "",
         f"Window: first {pcfg['window']['minutes']} min (clipped at ICU discharge); features use only data inside the window. "
         "Splits: the permanent per-hospital 70/15/15 manifest. Thresholds/calibration chosen on validation only.", ""]
    for key, blk in results["experiments"].items():
        cf = blk.get("cohort_filter")
        L += [f"## {key}  (n_features={blk['n_features']}; train/val/test={blk['split_sizes']})", ""]
        if cf:
            L += [f"*Cohort filter:* {cf['rule']} -> {cf['hospitals_kept']}/{cf['hospitals_total']} hospitals, {cf['patients_kept']}/{cf['patients_total']} patients.", ""]
        L += [
              "| arm | test AUROC (mean±sd) | seed-0 95% CI | test AUPRC | F1 | Brier | ECE | per-hosp macro AUROC | AUROC with local calibration |", "|---|---|---|---|---|---|---|---|---|"]
        agg = blk["aggregate"]
        for a in sorted(agg, key=lambda k: -agg[k]["auroc"]["mean"]):
            r0 = next(r for r in blk["records"] if r["arm"] == a)
            ci = agg[a]["auroc_ci_seed0"]; ph = r0["test_per_hospital"].get("macro_auroc")
            L.append(f"| {a} | {agg[a]['auroc']['mean']:.4f} ± {agg[a]['auroc']['std']:.4f} | [{ci['lo']:.4f}, {ci['hi']:.4f}] | {agg[a]['auprc']['mean']:.4f} | "
                     f"{agg[a]['f1']['mean']:.4f} | {agg[a]['brier']['mean']:.4f} | {agg[a]['ece']['mean']:.4f} | {'n/a' if ph is None else f'{ph:.4f}'} | {agg[a]['auroc_local']['mean']:.4f} |")
        for k, v in blk["benchmarks"].items():
            L.append(f"| *{k}* (benchmark, n={v['n']}) | {v['auroc']['point']:.4f} | [{v['auroc']['lo']:.4f}, {v['auroc']['hi']:.4f}] | {v['auprc']:.4f} | | | | | |")
        cmp = next((v for k, v in blk.items() if k.startswith("comparisons_vs_")), {})
        if not cmp and len(agg) > 1 and pcfg["evaluation"]["compare_to"] not in agg:
            L += ["", f"*Paired comparisons skipped: the reference arm `{pcfg['evaluation']['compare_to']}` was not part of this run "
                  "(include it in --arms, or set paper.evaluation.compare_to to an arm that is).*"]
        if cmp:
            L += ["", f"Paired bootstrap vs `{pcfg['evaluation']['compare_to']}` (seed {pcfg['seeds'][0]}):", ""]
            for a, d in cmp.items():
                L.append(f"- {a}: ΔAUROC={d['auroc']['delta']:+.4f} [{d['auroc']['lo']:+.4f},{d['auroc']['hi']:+.4f}] p={d['auroc']['p_value']:.3f}; "
                         f"ΔAUPRC={d['auprc']['delta']:+.4f} [{d['auprc']['lo']:+.4f},{d['auprc']['hi']:+.4f}]")
        tf = next((r["extra"].get("top_features_gain_share") for r in blk["records"] if r["arm"] == "lightgbm_central" and r["extra"].get("top_features_gain_share")), None)
        if tf:
            L += ["", "LightGBM top features (gain share): " + ", ".join(f"{n} ({v:.3f})" for n, v in tf[:15])]
        g = [(r["arm"], r["extra"]) for r in blk["records"] if r["arm"].startswith("fedavg_graph_") and r["seed"] == pcfg["seeds"][0]]
        if g:
            L += ["", "Graph diagnostics (seed 0): " + "; ".join(f"{a}: best_round={e.get('best_round')}, edges={e.get('n_edges')}, "
                                                                f"edge-label agreement={((e.get('homophily') or {}).get('edge_label_agreement'))} vs random {((e.get('homophily') or {}).get('expected_under_random')):.3f}" for a, e in g)]
        L.append("")
    L += ["## Reference results reported by the base papers", "", "| source | reported | value |", "|---|---|---|"]
    L += [f"| {r['name']} ({r['source']}) | {r['metric']} | {r['value']} |" for r in pcfg["reference_results"]]
    L += ["", "*Caveat:* cohorts, windows, feature sets and validation schemes differ between papers; compare headline numbers only with that in mind, "
          "and prefer the internal paired comparisons above (same patients, same split)."]
    text = "\n".join(L)
    (out_root / "paper_report.md").write_text(text, encoding="utf-8")
    return text


def _write_outputs(results: dict, out_root: Path, pcfg: dict) -> None:
    """(Re)writes results.json / results_table.csv / paper_report.md from whatever experiments have finished so far.
    Called after EVERY experiment block, so a crash hours into a run never loses the finished parts."""
    (out_root / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    with open(out_root / "results_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["experiment", "arm", "seed", "split", "auroc", "auprc", "f1", "precision", "recall", "specificity", "brier", "ece", "threshold", "n", "n_positive"])
        for key, blk in results["experiments"].items():
            for r in blk["records"]:
                for sp_, m in (("train", r["train"]), ("val", r["val"]), ("test", r["test"]), ("val_local", r["local_calibrated"]["val"]), ("test_local", r["local_calibrated"]["test"])):
                    w.writerow([key, r["arm"], r["seed"], sp_] + [m[k] for k in ("auroc", "auprc", "f1", "precision", "recall", "specificity", "brier", "ece", "threshold", "n", "n_positive")])
    write_report(results, out_root, pcfg)


def run_paper_track(cfg, phase1_summary, partition_result, phase3_result, run_id: str, tasks=None, arms_filter=None, split_manifest=None, fs_filter=None) -> dict:
    t0 = time.time()
    pcfg = get_paper_cfg(cfg)
    out_root = Path(cfg.get("run.output_dir", "outputs")) / pcfg["output_subdir"] / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    sm = split_manifest or load_or_build_split(cfg, partition_result, phase3_result)
    bundle = get_bundle(pcfg, cfg, phase1_summary, phase3_result, sm, out_root)
    split = np.array([sm.split_of(int(p)) or "none" for p in bundle.patient_ids], dtype="<U5")
    results = {"run_id": run_id, "n_patients": int(len(bundle.patient_ids)), "n_hospitals": int(len(np.unique(bundle.hospital_ids))),
               "n_features_total": int(bundle.X.shape[1]), "experiments": {}}
    sepsis_keep, filter_info = None, None
    hf = pcfg["sepsis_hospital_filter"]
    if hf["enabled"] and pcfg["tasks"].get("sepsis") and (not tasks or "sepsis" in tasks):
        cp = phase3_result["cohort_patients"]
        cov = hospital_medication_coverage(phase1_summary, cp, out_root.parent / "feature_cache" / f"med_coverage_{len(cp)}_{int(cp['patientunitstayid'].sum()) % 10**9}.json")
        keep_h = [int(h) for h in cov.index[cov >= float(hf["min_medication_coverage"])]]
        sepsis_keep = np.isin(bundle.hospital_ids, keep_h)
        filter_info = {"rule": f"hospital medication-record coverage >= {hf['min_medication_coverage']}", "hospitals_total": int(len(np.unique(bundle.hospital_ids))),
                       "hospitals_kept": int(len(set(keep_h) & set(np.unique(bundle.hospital_ids).tolist()))), "patients_kept": int(sepsis_keep.sum()), "patients_total": int(len(sepsis_keep))}
        logger.info("[PAPER] sepsis label-validity filter: %s -> %d/%d hospitals, %d/%d patients kept", filter_info["rule"], filter_info["hospitals_kept"],
                    filter_info["hospitals_total"], filter_info["patients_kept"], filter_info["patients_total"])
    size_filter = hospital_size_mask(bundle.hospital_ids, pcfg["cohort_filter"]["min_hospital_patients"])
    if size_filter is not None:
        si = size_filter[1]
        logger.info("[PAPER] hospital-size filter: %s -> %d/%d hospitals, %d/%d patients kept (kept sizes %s)", si["rule"], si["hospitals_kept"],
                    si["hospitals_total"], si["patients_kept"], si["patients_total"], si["kept_hospital_sizes"])
    for task, fsets in pcfg["tasks"].items():
        if tasks and task not in tasks:
            continue
        for fs in fsets:
            if fs_filter and fs not in fs_filter:
                continue
            tmask, tinfo = combine_filters([(sepsis_keep, filter_info) if (task == "sepsis" and sepsis_keep is not None) else None, size_filter], bundle.hospital_ids)
            blk = _run_task_featureset(task, fs, bundle, split, pcfg, out_root, arms_filter, tmask, tinfo)
            blk["aggregate"] = _aggregate(blk["records"])
            results["experiments"][f"{task}/{fs}"] = blk
            results["elapsed_seconds"] = time.time() - t0
            _write_outputs(results, out_root, pcfg)          # incremental: finished experiments are on disk immediately
    results["elapsed_seconds"] = time.time() - t0
    _write_outputs(results, out_root, pcfg)
    logger.info("[PAPER] done in %.0fs -> %s", results["elapsed_seconds"], out_root)
    return results
