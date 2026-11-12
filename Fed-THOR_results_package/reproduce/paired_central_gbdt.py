"""Federation cost for the GBDT in all six settings: the federated GBDT against the SAME code trained pooled on one client (arm gbdt_central).
Reads saved predictions and results.json only.

  python Fed-THOR_results_package/reproduce/paired_central_gbdt.py [--paper-root outputs/paper] [--out Fed-THOR_results_package/tables/paired_central_gbdt.json]

Per setting (seed 42 for the paired tests, three seeds for the means and the stability counts): federated GBDT, pooled GBDT and pooled LightGBM on IDENTICAL test patients;
federated minus pooled GBDT (the federation effect for this model) and pooled GBDT minus pooled LightGBM (what is implementation and search, not federation).
A setting whose pooled-GBDT run folder does not exist yet is skipped and listed; the script refuses to compare runs whose files hold different patients, splits or labels,
or whose hyperparameters differ (the point of the comparison is same code, same parameters, only federation differs).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score as auc

import paired_flicsscale as pf

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
SEEDS = (42, 43, 44)
SETTINGS = {
    "strict": dict(label="Mortality, strict 24 h landmark", exp="mortality/no_apache_scores", fed="fedgbdt_strict_final", central="gbdtcentral_strict", lgbm="final_mortality_strict"),
    "allstays": dict(label="Mortality, all stays", exp="mortality/no_apache_scores", fed="fedgbdt_allstays_final", central="gbdtcentral_allstays", lgbm="final_mortality_allstays"),
    "sepsis24": dict(label="Sepsis proxy, 24 h", exp="sepsis/sepsis_clean", fed="fedgbdt_sepsis24_final", central="gbdtcentral_sepsis24", lgbm="final_sepsis24"),
    "sepsis6h": dict(label="Sepsis proxy, 6 h", exp="sepsis/sepsis_clean", fed="fedgbdt_sepsis6h_mr1200", central="gbdtcentral_sepsis6h", lgbm="final_sepsis6h"),
    "top3": dict(label="Mortality, 3 largest hospitals", exp="mortality/no_apache_scores", fed="flicsstyle_top3", central="gbdtcentral_top3", lgbm="flicsstyle_top3"),
    "top10": dict(label="Mortality, 10 largest hospitals", exp="mortality/no_apache_scores", fed="flicsstyle_top10", central="gbdtcentral_top10", lgbm="flicsstyle_top10"),
}


def _load(paper, run, exp, arm, seed):
    p = Path(paper) / run / "predictions" / f"{exp.replace('/', '__')}__{arm}__seed{seed}.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    with np.load(p) as z:
        return {k: z[k] for k in ("patient_id", "hospital_id", "split", "y", "score")}


def _params(paper, run, exp, arm):
    blk = json.load(open(Path(paper) / run / "results.json", encoding="utf-8"))["experiments"][exp]
    recs = sorted((r for r in blk["records"] if r["arm"] == arm), key=lambda r: r["seed"])
    if not recs:
        raise ValueError(f"{run}: no {arm} record for {exp}")
    return {k: recs[0]["extra"]["params"][k] for k in sorted(recs[0]["extra"]["params"])}


def analyse(paper, spec):
    exp = spec["exp"]
    fed = _load(paper, spec["fed"], exp, "fed_gbdt", 42)
    cen = _load(paper, spec["central"], exp, "gbdt_central", 42)
    lgb = _load(paper, spec["lgbm"], exp, "lightgbm_central", 42)
    pf.check_same_patients({"federated GBDT": fed, "pooled GBDT": cen, "pooled LightGBM": lgb})
    pf_, pc_ = _params(paper, spec["fed"], exp, "fed_gbdt"), _params(paper, spec["central"], exp, "gbdt_central")
    if pf_ != pc_:
        raise pf.PredictionsMismatch(f"{spec['central']}: the pooled GBDT used different hyperparameters than {spec['fed']}'s federated GBDT ({pc_} against {pf_}); rerun it with paper.gbdt_central.reuse_params_from")
    te = fed["split"].astype(str) == "test"
    y = fed["y"].astype(float)[te]
    s = {"fed": fed["score"].astype(float)[te], "cen": cen["score"].astype(float)[te], "lgb": lgb["score"].astype(float)[te]}
    rng = np.random.default_rng(pf.RNG_PAIRED)
    paired = {"fed - cen": pf.paired_auc_diff(y, s, "fed", "cen", rng), "cen - lgb": pf.paired_auc_diff(y, s, "cen", "lgb", rng)}
    per = {"fed": [], "cen": []}
    for sd in SEEDS:
        for key, run, arm in (("fed", spec["fed"], "fed_gbdt"), ("cen", spec["central"], "gbdt_central")):
            z = _load(paper, run, exp, arm, sd)
            pf.check_same_patients({"seed-42 federated": fed, f"{arm}/seed{sd}": z})
            per[key].append(float(auc(y, z["score"].astype(float)[te])))
    return {"label": spec["label"], "n_test": int(te.sum()), "events": int(y.sum()), "paired": paired,
            "auroc_per_seed": per, "auroc_seed42": {k: float(auc(y, v)) for k, v in s.items()},
            "fed_minus_central_per_seed": [a - b for a, b in zip(per["fed"], per["cen"])], "params": pf_}


def build(paper_root):
    out, skipped = {}, []
    for key, spec in SETTINGS.items():
        if not (Path(paper_root) / spec["central"] / "results.json").exists():
            skipped.append(key)
            continue
        out[key] = analyse(paper_root, spec)
    if not out:
        raise FileNotFoundError("no pooled-GBDT run found (expected folders such as outputs/paper/gbdtcentral_strict); see docs/paper_track_v20.md")
    return {"n_bootstrap": pf.N_BOOT, "seed": 42, "settings": out, "skipped": skipped}


def table_from_json(js):
    """(header, rows, caption, label) for Table 13."""
    fm = lambda d: f"{d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
    ms = lambda v: f"{np.mean(v):.4f} ± {np.std(v):.4f}"
    rows = []
    for key, r in js["settings"].items():
        diffs = r["fed_minus_central_per_seed"]
        rows.append([r["label"], f"{r['n_test']:,} ({r['events']})", ms(r["auroc_per_seed"]["fed"]), ms(r["auroc_per_seed"]["cen"]), fm(r["paired"]["fed - cen"]), f"{sum(d > 0 for d in diffs)}/{len(diffs)}", fm(r["paired"]["cen - lgb"])])
    cap = ("Federation cost for the GBDT. The federated GBDT against the same code trained pooled on one client holding every training row, with the federated winner's hyperparameters, on identical test patients "
           f"(seed 42; {js['n_bootstrap']:,} paired bootstrap resamples; means over three seeds). The only difference between the two GBDTs is that the federated one merges bin edges from client summaries. "
           "The last column is pooled GBDT minus pooled LightGBM: the part of the gap to LightGBM that comes from implementation and search, not from federation."
           + (f" Settings not yet run: {', '.join(js['skipped'])}." if js["skipped"] else ""))
    return (["Setting", "test n (events)", "Federated GBDT AUROC (3 seeds)", "Pooled GBDT, same code (3 seeds)", "federated - pooled [95% CI]", "seeds with federated > pooled", "pooled GBDT - pooled LightGBM [95% CI]"], rows, cap, "tab:gbdtfederationcost")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper-root", default=str(REPO / "outputs" / "paper"))
    ap.add_argument("--out", default=str(PKG / "tables" / "paired_central_gbdt.json"))
    a = ap.parse_args(argv)
    js = build(a.paper_root)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(js, indent=1), encoding="utf-8")
    for key, r in js["settings"].items():
        print(f"{key}: test n={r['n_test']} ({r['events']} events)")
        for k, d in r["paired"].items():
            print(f"   {k:10s} {d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
    if js["skipped"]:
        print("not yet run (skipped):", ", ".join(js["skipped"]))
    print("written:", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
