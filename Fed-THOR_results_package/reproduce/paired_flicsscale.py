"""Like-for-like comparison at FLICS scale (reads saved predictions and results.json only; no data, no training).

  python Fed-THOR_results_package/reproduce/paired_flicsscale.py [--paper-root outputs/paper] [--out Fed-THOR_results_package/tables/paired_flicsscale.json]

Inputs
  outputs/paper/flicsstyle_top3 and outputs/paper/flicsstyle_top10   (configs/paper_flicsstyle.yaml: every model re-trained on ONLY the 3 or the 10 largest
                                                                     hospitals by strict-cohort size; each folder needs results.json and predictions/)
  outputs/paper/fedgbdt_strict_final and outputs/paper/final_mortality_strict   (the all-hospital strict runs, for the training-pool contrast)
Output
  tables/paired_flicsscale.json, read by build_tables.py (Tables 11a, 11b, 11c).

What it computes (seed 42 for paired tests, all three seeds for the stability counts)
  1. Per federation: the saved aggregate metrics (from results.json) and the verdict inputs against FLICS's reported figures.
  2. Paired AUROC differences between methods on IDENTICAL test patients (paired bootstrap over test patients, 2,000 resamples).
  3. Training-pool contrast: the same test patients scored by a model trained on only the federation and by one trained on all hospitals (patients matched by id).

Safety: every comparison refuses to run unless the files it pairs hold the same patients, splits and labels, and each run's predictions match that
run's own results.json split sizes. (A predictions folder copied from another run would otherwise produce plausible-looking, wrong numbers.)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score as auc

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
EXP = "mortality/no_apache_scores"
NAME = "mortality__no_apache_scores__%s__seed%d.npz"
SCALES = {"top3": "flicsstyle_top3", "top10": "flicsstyle_top10"}
SEEDS = (42, 43, 44)
ARMS = ("fed_gbdt", "fedavg_mlp", "fedavg_logreg", "lightgbm_central", "mlp_central", "logreg_central", "mlp_local_only")
FEDERATED = ("fed_gbdt", "fedavg_mlp", "fedavg_logreg")
# the all-hospital strict runs behind the training-pool contrast (same layout as paired_subsets.py)
POOL = {"fed_gbdt": ("fedgbdt_strict_final", "fed_gbdt"), "fedavg_mlp": ("final_mortality_strict", "fedavg_mlp"),
        "fedavg_logreg": ("final_mortality_strict", "fedavg_logreg"), "lightgbm_central": ("final_mortality_strict", "lightgbm_central")}
COMPARISONS = (("fed_gbdt", "fedavg_logreg"), ("fed_gbdt", "fedavg_mlp"), ("fed_gbdt", "lightgbm_central"), ("fedavg_mlp", "fedavg_logreg"),
               ("fedavg_mlp", "mlp_central"), ("fedavg_logreg", "logreg_central"), ("fedavg_mlp", "mlp_local_only"), ("lightgbm_central", "logreg_central"))
PER_SEED = tuple(c for c in COMPARISONS if "logreg_central" not in c)     # every comparison whose two arms were run with all three seeds (central LR has one)
N_BOOT = 2000
RNG_PAIRED = 11
RNG_POOL = 13
NAMES = {"fed_gbdt": "Federated GBDT", "fedavg_mlp": "Federated MLP", "fedavg_logreg": "Federated LR (FLICS's model class)", "lightgbm_central": "Pooled LightGBM",
         "mlp_central": "Central MLP", "logreg_central": "Central LR", "mlp_local_only": "Local-only MLP"}


class PredictionsMismatch(ValueError):
    """Raised when prediction files that must describe the same patients do not (never silently compared)."""


def _load(paper, run, arm, seed):
    p = Path(paper) / run / "predictions" / (NAME % (arm, seed))
    if not p.exists():
        raise FileNotFoundError(f"missing {p}: keep outputs/paper/{run}/predictions from the run that produced it")
    with np.load(p) as z:                                   # arrays copied out so the file handle is closed at once (Windows cannot delete open files)
        return {k: z[k] for k in ("patient_id", "hospital_id", "split", "y", "score")}


def check_same_patients(named):
    """All entries must hold identical patient ids, splits and labels, in the same order."""
    items = list(named.items())
    ref_name, ref = items[0]
    for name, p in items[1:]:
        same = (len(p["patient_id"]) == len(ref["patient_id"]) and bool((p["patient_id"] == ref["patient_id"]).all())
                and bool((p["split"].astype(str) == ref["split"].astype(str)).all()) and bool(np.allclose(p["y"].astype(float), ref["y"].astype(float), equal_nan=True)))
        if not same:
            raise PredictionsMismatch(f"{name} does not hold the same patients, splits and labels as {ref_name}: pairing them would compare different patients")


def check_run_matches_results(arrays, results_path, run):
    """The predictions folder must belong to the run whose results.json sits beside it."""
    sizes = json.load(open(results_path, encoding="utf-8"))["experiments"][EXP]["split_sizes"]
    split = arrays["split"].astype(str)
    got = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    want = {s: int(sizes[s]) for s in ("train", "val", "test")}
    if got != want:
        raise PredictionsMismatch(f"{run}: the predictions have split sizes {got} but its results.json says {want}; this predictions folder belongs to a different run")


def paired_auc_diff(y, s, a, b, rng, n_boot=None):
    """AUROC(a) - AUROC(b) on the same patients with a paired bootstrap over patients (the same resample is used for both)."""
    n_boot = N_BOOT if n_boot is None else n_boot
    n = len(y)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        d.append(auc(y[i], s[a][i]) - auc(y[i], s[b][i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(auc(y, s[a]) - auc(y, s[b])), "lo": float(lo), "hi": float(hi)}


def aggregate_from_results(results):
    """Per-arm metrics exactly as the collector summarises them (3-seed mean, seed-42 interval)."""
    b = results["experiments"][EXP]
    out = {}
    for arm, v in b["aggregate"].items():
        rs = [r for r in b["records"] if r["arm"] == arm]
        ph = [r["test_per_hospital"]["macro_auroc"] for r in rs if r.get("test_per_hospital", {}).get("macro_auroc") is not None]
        out[arm] = {"n_seeds": int(v["n_seeds"]), "auroc": float(v["auroc"]["mean"]), "sd": float(v["auroc"]["std"]), "lo": float(v["auroc_ci_seed0"]["lo"]),
                    "hi": float(v["auroc_ci_seed0"]["hi"]), "auprc": float(v["auprc"]["mean"]), "perhosp": float(np.mean(ph)) if ph else None,
                    "val": float(np.mean([r["val"]["auroc"] for r in rs]))}
    return out


def analyse_scale(paper, run):
    rpath = Path(paper) / run / "results.json"
    if not rpath.exists():
        raise FileNotFoundError(f"missing {rpath}")
    results = json.load(open(rpath, encoding="utf-8"))
    block = results["experiments"][EXP]
    files = {arm: _load(paper, run, arm, 42) for arm in ARMS}
    for arm, z in files.items():
        check_run_matches_results(z, rpath, f"{run}/{arm}")
    check_same_patients(files)
    ref = files[ARMS[0]]
    te = ref["split"].astype(str) == "test"
    y = ref["y"].astype(float)[te]
    s = {a: z["score"].astype(float)[te] for a, z in files.items()}
    rng = np.random.default_rng(RNG_PAIRED)
    paired = {f"{a} - {c}": paired_auc_diff(y, s, a, c, rng) for a, c in COMPARISONS}

    per_seed = {}                                           # point AUROC differences on the same patients, one per training seed
    for a, c in PER_SEED:
        vals = []
        for sd in SEEDS:
            za, zc = _load(paper, run, a, sd), _load(paper, run, c, sd)
            check_same_patients({"seed-42 reference": ref, f"{a}/seed{sd}": za, f"{c}/seed{sd}": zc})
            vals.append(float(auc(y, za["score"].astype(float)[te]) - auc(y, zc["score"].astype(float)[te])))
        per_seed[f"{a} - {c}"] = vals

    pool = {}                                               # same test patients, model trained on the federation vs on all hospitals (matched by patient id)
    pid = ref["patient_id"][te]
    for arm, (prun, parm) in POOL.items():
        z = _load(paper, prun, parm, 42)
        index = {int(p): j for j, p in enumerate(z["patient_id"])}
        if len(index) != len(z["patient_id"]):
            raise PredictionsMismatch(f"{prun}/{parm}: duplicate patient ids, patients cannot be matched")
        absent = [int(p) for p in pid if int(p) not in index]
        if absent:
            raise PredictionsMismatch(f"{prun}/{parm}: {len(absent)} of the federation's test patients are absent (e.g. {absent[:3]}); not the same cohort")
        j = np.array([index[int(p)] for p in pid])
        if not (z["split"].astype(str)[j] == "test").all() or not np.allclose(z["y"].astype(float)[j], y, equal_nan=True):
            raise PredictionsMismatch(f"{prun}/{parm}: matched patients are not all test patients with identical labels; different split or cohort")
        two = {"sub": s[arm], "all": z["score"].astype(float)[j]}
        pool[arm] = {"auroc_trained_on_federation": float(auc(y, two["sub"])), "auroc_trained_on_all": float(auc(y, two["all"])),
                     "all_minus_federation": paired_auc_diff(y, two, "all", "sub", np.random.default_rng(RNG_POOL))}

    cf = block.get("cohort_filter") or {}
    return {"run": run, "cohort_filter_rule": cf.get("rule"), "hospitals_kept": cf.get("hospitals_kept"), "patients_kept": cf.get("patients_kept"),
            "split_sizes": block["split_sizes"], "n_test": int(te.sum()), "events": int(y.sum()), "aggregate": aggregate_from_results(results),
            "paired": paired, "per_seed_diff": per_seed, "training_pool": pool}


def build(paper_root):
    return {"experiment": EXP, "seed": 42, "n_bootstrap": N_BOOT, "scales": {k: analyse_scale(paper_root, run) for k, run in SCALES.items()}}


def _f(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


def tables_from_json(js, refs, verdict):
    """(header, rows, caption, label) for Tables 11a-c. refs: {scale: [(reference name, value), ...]}; verdict(ref, lo, hi) -> 'ahead' | 'level' | 'behind'."""
    sc = js["scales"]
    lab = {"top3": "3 largest", "top10": "10 largest"}
    fm = lambda d: f"{d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
    cap = " and ".join(f"{lab[k]}: {s['hospitals_kept']} hospitals, {s['patients_kept']:,} stays, train/val/test = {s['split_sizes']['train']:,}/{s['split_sizes']['val']:,}/{s['split_sizes']['test']:,}" for k, s in sc.items())
    tst = " and ".join(f"{s['n_test']:,} ({s['events']} events)" for s in sc.values())

    rows = []
    for k, s in sc.items():
        for arm in ARMS:
            r = s["aggregate"][arm]
            a_cell = f"{r['auroc']:.4f} ± {r['sd']:.4f}" if r["n_seeds"] > 1 else f"{r['auroc']:.4f} (1 seed)"
            v_cell = "–"
            if arm in FEDERATED:
                v_cell = "; ".join(f"{verdict(v, r['lo'], r['hi'])} ({r['auroc'] - v:+.3f}) vs {v:.2f} ({n.replace('FLICS, FedAvg ', '')})" for n, v in refs[k])
            rows.append([lab[k] + " hospitals", NAMES[arm], a_cell, f"[{r['lo']:.4f}, {r['hi']:.4f}]", _f(r["auprc"]), _f(r["perhosp"]), v_cell])
    t11a = (["Federation", "Model", "AUROC (mean ± sd, 3 seeds)", "95% CI (seed 42)", "AUPRC", "per-hosp. macro AUROC", "vs FLICS, as reported"], rows,
            "FLICS-scale federations: every model re-trained on only the 3 or the 10 largest hospitals by strict-cohort size (configs/paper_flicsstyle.yaml; " + cap + "). Strict cohort, hospital mortality, "
            "first-24 h features without APACHE scores. 'ahead' = our seed-42 interval lies above the reference, 'level' = it contains it, 'behind' = below (as in Table 3); FLICS's own interval is up to ±0.035, "
            "so these labels are context, not claims. Central logistic regression has one seed. Hospital-level inputs (region, teaching status, bed-size category, log hospital size) identify the site in such small "
            "federations; they are identical for every arm.", "tab:flicsscale")

    rows = []
    for a, c in COMPARISONS:
        key = f"{a} - {c}"
        cells = [NAMES[a] + " - " + NAMES[c]]
        for s in sc.values():
            cells.append(fm(s["paired"][key]))
            cells.append(f"{sum(v > 0 for v in s['per_seed_diff'][key])}/{len(s['per_seed_diff'][key])}" if key in s["per_seed_diff"] else "–")
        rows.append(cells)
    hdr = ["Comparison (A - B)"]
    for k in sc:
        hdr += [f"{lab[k]}: A - B [95% CI]", f"{lab[k]}: seeds with A > B"]
    t11b = (hdr, rows, "Paired AUROC differences on identical test patients inside each FLICS-scale federation (seed 42; " + f"{N_BOOT:,}" + " paired bootstrap resamples over test patients; test n = " + tst +
            "). 'seeds with A > B' counts the three training seeds whose point difference on the same patients is positive ('–' = one of the two arms has a single seed).", "tab:flicsscale_paired")

    rows = []
    for arm in POOL:
        cells = [NAMES[arm]]
        for s in sc.values():
            p = s["training_pool"][arm]
            cells += [f"{p['auroc_trained_on_federation']:.4f} → {p['auroc_trained_on_all']:.4f}", fm(p["all_minus_federation"])]
        rows.append(cells)
    hdr = ["Model"]
    for k in sc:
        hdr += [f"{lab[k]}: AUROC, trained on the federation → on all hospitals", f"{lab[k]}: all - federation [95% CI]"]
    t11c = (hdr, rows, "Does training on more hospitals help? The same test patients scored by a model trained only on the federation (this run) and by one trained on all hospitals (the final strict runs), seed 42, "
            "patients matched by id. Intervals reflect test-patient sampling only, not training randomness.", "tab:flicsscale_pool")
    return {"t11a": t11a, "t11b": t11b, "t11c": t11c}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper-root", default=str(REPO / "outputs" / "paper"))
    ap.add_argument("--out", default=str(PKG / "tables" / "paired_flicsscale.json"))
    a = ap.parse_args(argv)
    js = build(a.paper_root)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(js, indent=1), encoding="utf-8")
    for k, s in js["scales"].items():
        print(f"{k}: test n={s['n_test']} ({s['events']} events)")
        for key, d in s["paired"].items():
            print(f"   {key:36s} {d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
    print("written:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
