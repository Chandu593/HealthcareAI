"""Paired AUROC comparisons for the sepsis 6 h federated GBDT (reads saved predictions only).

  python Fed-THOR_results_package/reproduce/paired_sepsis6h.py [--paper-root outputs/paper] [--out Fed-THOR_results_package/tables/paired_sepsis6h.json]

The first 6 h GBDT run (fedgbdt_sepsis6h_final) hit its 600-round limit in every seed; it was re-run with a 1,200-round limit (fedgbdt_sepsis6h_mr1200). On the identical 22,597 test patients (seed 42,
2,000 paired bootstrap resamples) this reports: rerun - first run, and each GBDT run against the federated MLP and pooled LightGBM of final_sepsis6h. It refuses to run if the files do not hold the same patients, splits and labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score as auc

import paired_flicsscale as pf

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
G = "sepsis__sepsis_clean__%s__seed%d.npz"
RUNS = {"rerun": ("fedgbdt_sepsis6h_mr1200", "fed_gbdt"), "first": ("fedgbdt_sepsis6h_final", "fed_gbdt"),
        "mlp": ("final_sepsis6h", "fedavg_mlp"), "lgbm": ("final_sepsis6h", "lightgbm_central")}
PAIRS = (("rerun", "first"), ("rerun", "mlp"), ("rerun", "lgbm"), ("first", "mlp"), ("first", "lgbm"))
SEEDS = (42, 43, 44)


def _load(paper, run, arm, seed):
    p = Path(paper) / run / "predictions" / (G % (arm, seed))
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    with np.load(p) as z:
        return {k: z[k] for k in ("patient_id", "hospital_id", "split", "y", "score")}


def compute(paper_root):
    files = {k: _load(paper_root, run, arm, 42) for k, (run, arm) in RUNS.items()}
    pf.check_same_patients(files)
    ref = files["rerun"]
    te = ref["split"].astype(str) == "test"
    va = ref["split"].astype(str) == "val"
    y = ref["y"].astype(float)[te]
    s = {k: z["score"].astype(float)[te] for k, z in files.items()}
    rng = np.random.default_rng(pf.RNG_PAIRED)
    out = {"n_test": int(te.sum()), "events": int(y.sum()), "auroc_seed42": {k: float(auc(y, v)) for k, v in s.items()},
           "paired": {f"{a} - {b}": pf.paired_auc_diff(y, s, a, b, rng) for a, b in PAIRS}, "per_seed": {}, "validation_auroc": {}}
    for k in ("rerun", "first"):                                  # all three seeds: the selection rule looks at validation AUROC
        run, arm = RUNS[k]
        per_test, per_val = [], []
        for sd in SEEDS:
            z = _load(paper_root, run, arm, sd)
            pf.check_same_patients({"seed-42 reference": ref, f"{k}/seed{sd}": z})
            per_test.append(float(auc(y, z["score"].astype(float)[te])))
            per_val.append(float(auc(ref["y"].astype(float)[va], z["score"].astype(float)[va])))
        out["per_seed"][k] = per_test
        out["validation_auroc"][k] = per_val
    out["per_seed"]["rerun - first"] = [a - b for a, b in zip(out["per_seed"]["rerun"], out["per_seed"]["first"])]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper-root", default=str(REPO / "outputs" / "paper"))
    ap.add_argument("--out", default=str(PKG / "tables" / "paired_sepsis6h.json"))
    a = ap.parse_args(argv)
    r = compute(a.paper_root)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"sepsis 6 h: test n={r['n_test']} ({r['events']} events)")
    for k, d in r["paired"].items():
        print(f"   {k:16s} {d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
    print("   validation AUROC per seed:", {k: [round(x, 4) for x in v] for k, v in r["validation_auroc"].items()})
    print("written:", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
