"""Table 14: the final comparison in the format of FLICS's Tables IV and V -- Accuracy, AUROC and F1 for every federated and every central model, in each setting.

Reads outputs/paper/<run>/results.json only (per-seed test metrics already stored there): nothing is retrained and no prediction file is opened.

  python Fed-THOR_results_package/reproduce/final_comparison.py [--paper-root outputs/paper]     (prints the tables; build_tables.py writes them into tables.md / tables.tex / tables.json)

Conventions (FLICS, Tables IV and V): bold marks the best FEDERATED result per metric, underline marks the best NON-PRIVATE (central) result per metric. FLICS underlines its single
central oracle; here there are four central models, so the best of them is underlined per metric. Ties (equal after rounding to four decimals) are marked on every tied row.
The mark follows the highest mean; it is not a significance statement (the caption says so).

Accuracy and F1 are the TEST values at the decision threshold that maximises F1 on the VALIDATION set. FLICS tunes its threshold on its test set, so that part of its procedure is not
reproduced here (it would break the validation-only rule); both metrics also depend on prevalence, which differs between cohorts.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
M = "mortality/no_apache_scores"
S = "sepsis/sepsis_clean"
FED = ("fedavg_logreg", "fedavg_mlp", "fed_gbdt")
CEN = ("logreg_central", "mlp_central", "lightgbm_central", "gbdt_central")
OPTIONAL = ("fed_gbdt", "gbdt_central")          # run in their own folders; a row appears once that run exists
NAMES = {"fedavg_logreg": "Federated LR", "fedavg_mlp": "Federated MLP", "fed_gbdt": "Federated GBDT",
         "logreg_central": "Central LR", "mlp_central": "Central MLP", "lightgbm_central": "Central LightGBM", "gbdt_central": "Central GBDT (same code, pooled)"}
METRICS = ("accuracy", "auroc", "f1")
ND = 4                                           # ties are decided on the printed precision

# One block per setting. 'base' holds every arm except those listed in 'over'; each override names the run folder that holds that arm.
BLOCKS = {
    "t14a": dict(key="strict", label="Mortality, strict 24 h landmark cohort", exp=M, base="final_mortality_strict",
                 over={"fed_gbdt": "fedgbdt_strict_final", "gbdt_central": "gbdtcentral_strict"}, flics=None, tag="tab:final_strict"),
    "t14b": dict(key="allstays", label="Mortality, all stays (first-24 h window; includes the short-stay effect)", exp=M, base="final_mortality_allstays",
                 over={"fed_gbdt": "fedgbdt_allstays_final", "gbdt_central": "gbdtcentral_allstays"}, flics=None, tag="tab:final_allstays"),
    "t14c": dict(key="top3", label="Mortality, 3 largest hospitals (FLICS-scale federation)", exp=M, base="flicsstyle_top3",
                 over={"gbdt_central": "gbdtcentral_top3"}, flics="top3", tag="tab:final_top3"),
    "t14d": dict(key="top10", label="Mortality, 10 largest hospitals (FLICS-scale federation)", exp=M, base="flicsstyle_top10",
                 over={"gbdt_central": "gbdtcentral_top10"}, flics="top10", tag="tab:final_top10"),
    "t14e": dict(key="sepsis24", label="Sepsis proxy, 24 h window, clean features, label-valid hospitals", exp=S, base="final_sepsis24",
                 over={"fed_gbdt": "fedgbdt_sepsis24_final", "gbdt_central": "gbdtcentral_sepsis24"}, flics=None, tag="tab:final_sepsis24"),
    "t14f": dict(key="sepsis6h", label="Sepsis proxy, 6 h early-warning window, clean features, label-valid hospitals", exp=S, base="final_sepsis6h",
                 over={"fed_gbdt": "fedgbdt_sepsis6h_mr1200", "gbdt_central": "gbdtcentral_sepsis6h"}, flics=None, tag="tab:final_sepsis6h"),
}

# FLICS's own numbers, transcribed from its Tables IV (3 hospitals, 5,000 stays) and V (10 hospitals, 53,500 stays): accuracy (95% CI), AUC (95% CI), F1. Local-model rows are omitted.
FLICS = {
    "top3": [("FedAvg LR", "0.90 (0.88–0.92)", "0.87 (0.83–0.90)", "0.60"), ("FedAvg LSTM", "0.89 (0.87–0.91)", "0.87 (0.83–0.90)", "0.61"),
             ("Centralized LR", "0.88 (0.86–0.90)", "0.86 (0.83–0.90)", "0.60"), ("Centralized XGBoost", "0.91 (0.89–0.93)", "0.90 (0.87–0.93)", "0.65")],
    "top10": [("FedAvg LR", "0.89 (0.87–0.91)", "0.83 (0.79–0.87)", "0.59"), ("FedAvg LSTM", "0.89 (0.87–0.91)", "0.84 (0.79–0.88)", "0.61"),
              ("Centralized LR", "0.89 (0.87–0.91)", "0.84 (0.79–0.88)", "0.61"), ("Centralized XGBoost", "0.90 (0.88–0.91)", "0.88 (0.85–0.91)", "0.63")],
}
HEAD = ["Model", "Accuracy (mean ± sd, 3 seeds)", "AUROC (mean ± sd, 3 seeds) [95% CI, seed 42]", "F1 (mean ± sd, 3 seeds)"]


def best_arms(means, nd=ND):
    """Arms whose mean equals the highest mean after rounding to nd decimals (ties are all returned)."""
    if not means:
        return set()
    top = max(round(v, nd) for v in means.values())
    return {a for a, v in means.items() if round(v, nd) == top}


def wrap(cell, bold=False, underline=False):
    if bold and underline:
        raise ValueError("a cell is either the best federated or the best central result, never both")
    return f"**{cell}**" if bold else (f"<u>{cell}</u>" if underline else cell)


def to_tex(s):
    """Markdown-style marks to LaTeX (applied by build_tables.py after its own escaping)."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    return re.sub(r"<u>(.+?)</u>", r"\\underline{\1}", s)


def arm_stats(records):
    """Per-metric mean, sd (population, as in results.json's own aggregate), seed count; the seed-42 AUROC interval; test n and prevalence."""
    recs = sorted(records, key=lambda r: r["seed"])
    st = {m: dict(mean=float(np.mean([r["test"][m] for r in recs])), sd=float(np.std([r["test"][m] for r in recs])), n=len(recs)) for m in METRICS}
    r0 = next((r for r in recs if r["seed"] == 42), recs[0])
    ci = r0["test_ci"]["auroc"]
    return dict(stats=st, ci=(float(ci["lo"]), float(ci["hi"])), n_test=int(recs[0]["test"]["n"]), prevalence=float(recs[0]["test"]["prevalence"]))


def _cell(stats, m):
    s = stats[m]
    return f"{s['mean']:.4f} ± {s['sd']:.4f}" if s["n"] > 1 else f"{s['mean']:.4f} (1 seed)"


def build_block(load, spec):
    """(header, rows, caption, label, meta). load(run) -> results.json dict. Required arms must exist; the optional GBDT arms are skipped with a warning."""
    got = {}
    for a in FED + CEN:
        run = spec["over"].get(a, spec["base"])
        try:
            blk = load(run)["experiments"][spec["exp"]]
        except (FileNotFoundError, KeyError):
            if a in OPTIONAL:
                print(f"WARNING: Table 14 ({spec['key']}): no {NAMES[a]} row, run folder '{run}' / {spec['exp']} is not available", file=sys.stderr)
                continue
            raise
        recs = [r for r in blk["records"] if r["arm"] == a]
        if not recs:
            if a in OPTIONAL:
                print(f"WARNING: Table 14 ({spec['key']}): no {NAMES[a]} row, '{run}' holds no {a} records", file=sys.stderr)
                continue
            raise ValueError(f"{run}: no {a} records for {spec['exp']}")
        got[a] = arm_stats(recs)
    ref = next(iter(got.values()))
    for a, g in got.items():
        if (g["n_test"], round(g["prevalence"], 6)) != (ref["n_test"], round(ref["prevalence"], 6)):
            raise ValueError(f"Table 14 ({spec['key']}): {a} was scored on a different test set (n={g['n_test']}) than the other arms (n={ref['n_test']})")
    fed = [a for a in FED if a in got]; cen = [a for a in CEN if a in got]
    bold = {m: best_arms({a: got[a]["stats"][m]["mean"] for a in fed}) for m in METRICS}
    ulin = {m: best_arms({a: got[a]["stats"][m]["mean"] for a in cen}) for m in METRICS}
    rows = []
    for a in fed + cen:
        g = got[a]; st = g["stats"]
        au = _cell(st, "auroc") + f" [{g['ci'][0]:.4f}, {g['ci'][1]:.4f}]"
        cells = {"accuracy": _cell(st, "accuracy"), "auroc": au, "f1": _cell(st, "f1")}
        rows.append([NAMES[a]] + [wrap(cells[m], bold=(a in fed and a in bold[m]), underline=(a in cen and a in ulin[m])) for m in METRICS])
    if spec["flics"]:
        rows += [[f"FLICS, {n} (as reported)", acc, auc, f1] for n, acc, auc, f1 in FLICS[spec["flics"]]]
    prev = ref["prevalence"]
    cap = (f"{spec['label']}: accuracy, AUROC and F1 in the format of FLICS's Tables IV and V (test n = {ref['n_test']:,}; mortality or proxy prevalence {prev:.1%}, so a classifier that predicts nobody positive scores "
           f"accuracy {1 - prev:.3f}). Mean ± sd over the training seeds (same split); the AUROC interval is the seed-42 bootstrap. Bold = highest mean among the federated models, underline = highest mean among the "
           "central (non-private) models, per metric, ties marked on every tied row; the marks follow the point estimates and do not mean the difference is resolved (it is often smaller than the seed sd). "
           "Accuracy and F1 are test values at the threshold that maximises F1 on the validation set (never tuned on the test set; FLICS tunes its threshold on its test set). Both depend on prevalence, so they "
           "are not comparable across cohorts. Central LR has one seed.")
    if spec["flics"]:
        cap += (" The four FLICS rows are its reported numbers (3 hospitals, 5,000 stays; 10 hospitals, 53,500 stays; mortality 12.5-17%; ICU-discharge label; local models omitted) and are not ranked or marked: "
                "cohort, label and threshold procedure differ from ours.")
    meta = dict(n_test=ref["n_test"], prevalence=prev, bold=bold, underline=ulin, arms=sorted(got))
    return HEAD, rows, cap, spec["tag"], meta


def tables(paper_root, load=None):
    """{'t14a': (header, rows, caption, label), ...}; a block whose base run folder is missing is skipped with a warning."""
    cache = {}

    def _load(run):
        if run not in cache:
            cache[run] = json.load(open(Path(paper_root) / run / "results.json", encoding="utf-8"))
        return cache[run]
    out = {}
    for k, spec in BLOCKS.items():
        try:
            h, rows, cap, lab, _ = build_block(load or _load, spec)
        except FileNotFoundError as e:
            print(f"WARNING: Table 14 block {k} ({spec['key']}) skipped: {e}", file=sys.stderr)
            continue
        out[k] = (h, rows, cap, lab)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-root", default=str(REPO / "outputs" / "paper"))
    a = ap.parse_args()
    for k, (h, rows, cap, _) in tables(a.paper_root).items():
        print(f"\n[{k}] {cap}\n| " + " | ".join(h) + " |")
        for r in rows:
            print("| " + " | ".join(r) + " |")
