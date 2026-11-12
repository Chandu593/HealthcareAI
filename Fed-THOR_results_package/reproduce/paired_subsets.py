"""Same-patient comparisons behind the FLICS discussion (reads saved predictions only; no data, no training).

  python Fed-THOR_results_package/reproduce/paired_subsets.py

1. Evaluation noise: our strict-cohort test predictions scored the way FLICS scored its models (stratified 1,000-patient test sets, about 14% mortality):
   how far does AUROC move just from the draw of patients?
2. Paired method comparisons on identical test patients: all hospitals, the 10 largest and the 3 largest hospitals (by strict-cohort size). The models were
   trained on ALL hospitals and are only EVALUATED on those subsets; this is not a replication of FLICS's 3- and 10-hospital federations.
Seed 42, strict cohort. Writes tables/paired_subsets.json (read by build_tables.py).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score as auc

PKG = Path(__file__).resolve().parents[1]; REPO = PKG.parent; PAPER = REPO / "outputs" / "paper"
NAME = "mortality__no_apache_scores__%s__seed42.npz"
ARMS = {"gbdt": ("fedgbdt_strict_final", "fed_gbdt"), "mlp": ("final_mortality_strict", "fedavg_mlp"),
        "lr": ("final_mortality_strict", "fedavg_logreg"), "lgbm": ("final_mortality_strict", "lightgbm_central")}
LABEL = {"gbdt": "Federated GBDT", "mlp": "Federated MLP", "lr": "Federated LR (FLICS-style model)", "lgbm": "Pooled LightGBM"}


def load():
    P = {k: np.load(PAPER / run / "predictions" / (NAME % arm)) for k, (run, arm) in ARMS.items()}
    ref = P["gbdt"]
    for k, p in P.items():                                    # the same patients, split and labels in every file, or the comparison is meaningless
        assert (p["patient_id"] == ref["patient_id"]).all() and (p["split"] == ref["split"]).all() and np.allclose(p["y"], ref["y"], equal_nan=True), k
    return P, ref


def main():
    P, ref = load()
    split = ref["split"].astype(str); te = split == "test"; y = ref["y"].astype(float); hid = ref["hospital_id"]
    sc = {k: p["score"].astype(float) for k, p in P.items()}
    out = {"seed": 42, "cohort": "strict 24 h landmark", "n_test": int(te.sum()), "events": int(y[te].sum())}

    # 1. FLICS-style evaluation noise
    rng = np.random.default_rng(7); pos, neg = np.flatnonzero(te & (y == 1)), np.flatnonzero(te & (y == 0)); draws = {k: [] for k in sc}
    for _ in range(5000):
        i = np.concatenate([rng.choice(pos, 140, replace=False), rng.choice(neg, 860, replace=False)])
        for k in sc: draws[k].append(auc(y[i], sc[k][i]))
    draws = {k: np.array(v) for k, v in draws.items()}
    out["noise"] = {"n_draws": 5000, "test_size": 1000, "events": 140,
                    "models": {k: {"name": LABEL[k], "mean": float(v.mean()), "sd": float(v.std()), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)),
                                   "p_ge_0.87": float((v >= 0.87).mean())} for k, v in draws.items()},
                    "p_gbdt_beats_mlp": float((draws["gbdt"] > draws["mlp"]).mean()), "p_gbdt_beats_lr": float((draws["gbdt"] > draws["lr"]).mean())}

    # 2. paired comparisons on size-defined subsets (sizes counted over every cohort row in the predictions file)
    hs, cs = np.unique(hid, return_counts=True); order = np.argsort(-cs); brng = np.random.default_rng(11)
    subsets = {"all hospitals": te, "10 largest hospitals": te & np.isin(hid, hs[order[:10]]), "3 largest hospitals": te & np.isin(hid, hs[order[:3]])}
    def paired(idx, a, b, B=1000):
        d = []
        for _ in range(B):
            i = brng.choice(idx, len(idx))
            if y[i].min() == y[i].max(): continue
            d.append(auc(y[i], sc[a][i]) - auc(y[i], sc[b][i]))
        lo, hi = np.percentile(d, [2.5, 97.5]); return {"diff": float(auc(y[idx], sc[a][idx]) - auc(y[idx], sc[b][idx])), "lo": float(lo), "hi": float(hi)}
    out["subsets"] = {}
    for name, mask in subsets.items():
        idx = np.flatnonzero(mask)
        out["subsets"][name] = {"n_test": int(len(idx)), "events": int(y[idx].sum()), "auroc": {k: float(auc(y[idx], sc[k][idx])) for k in sc},
                                "gbdt_minus_lr": paired(idx, "gbdt", "lr"), "gbdt_minus_mlp": paired(idx, "gbdt", "mlp"), "gbdt_minus_lgbm": paired(idx, "gbdt", "lgbm")}
    (PKG / "tables").mkdir(exist_ok=True)
    (PKG / "tables" / "paired_subsets.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    n = out["noise"]["models"]["gbdt"]
    print("noise: federated GBDT on FLICS-style 1,000-patient test sets: %.4f +- %.4f [%.3f, %.3f]" % (n["mean"], n["sd"], n["lo"], n["hi"]))
    for name, s in out["subsets"].items():
        print("%-22s n=%-6d GBDT-LR %+.4f [%+.4f, %+.4f] | GBDT-MLP %+.4f | GBDT-pooled LightGBM %+.4f [%+.4f, %+.4f]" % (name, s["n_test"], s["gbdt_minus_lr"]["diff"], s["gbdt_minus_lr"]["lo"], s["gbdt_minus_lr"]["hi"], s["gbdt_minus_mlp"]["diff"], s["gbdt_minus_lgbm"]["diff"], s["gbdt_minus_lgbm"]["lo"], s["gbdt_minus_lgbm"]["hi"]))


if __name__ == "__main__":
    main()
