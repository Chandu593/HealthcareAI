import json, sys
from pathlib import Path
from parse import load
from paired_flicsscale import tables_from_json
from paired_central_gbdt import table_from_json as central_gbdt_table
import final_comparison as fc
PKG = Path(__file__).resolve().parents[1]; REPO = PKG.parent
R = load(); M = "mortality/no_apache_scores"; S = "sepsis/sepsis_clean"
NAMES = {"mlp_local_only": "Local-only MLP (no federation)", "fedavg_logreg": "Federated logistic regression", "fedavg_mlp": "Federated MLP (FedAvg + FedProx)",
         "fedavg_graph_none": "Federated MLP + graph head, no edges", "fedavg_graph_A": "Federated MLP + graph A (kNN + threshold)", "fedavg_graph_B": "Federated MLP + graph B (cluster-restricted)",
         "fed_gbdt": "Federated GBDT (histogram boosting)", "gbdt_central": "Central GBDT (same code, pooled)", "logreg_central": "Central logistic regression", "mlp_central": "Central MLP", "lightgbm_central": "Central LightGBM"}
def f(x, d=4): return "n/a" if x is None else f"{x:.{d}f}"
def row(r): return [f"{r['auroc']:.4f} ± {r['sd']:.4f}", f"[{r['lo']:.4f}, {r['hi']:.4f}]", f(r["auprc"]), f(r["brier"]), f(r["ece"]), f(r["perhosp"]), f(r["local"])]
HEAD = ["Model", "AUROC (mean ± sd, 3 seeds)", "95% CI (seed 42)", "AUPRC", "Brier", "ECE", "per-hosp. macro AUROC", "AUROC, per-hosp. calibration"]
def md(header, rows): return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])
def tex(header, rows, caption, label):
    esc = lambda s: fc.to_tex(s.replace("±", "$\\pm$").replace("%", "\\%").replace("_", "\\_").replace("≥", "$\\geq$").replace("μ", "$\\mu$").replace("→", "$\\rightarrow$"))   # to_tex: **bold** and <u>underline</u> (Table 14)
    return ("\\begin{table}[t]\n\\centering\\small\n\\caption{" + esc(caption) + "}\\label{" + label + "}\n\\begin{tabular}{l" + "c" * (len(header) - 1) + "}\n\\toprule\n" + " & ".join(esc(h) for h in header) + " \\\\\n\\midrule\n" +
            "\n".join(" & ".join(esc(c) for c in r) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
T = {}
# the federated-GBDT arm was run in its own run directories (same split, seeds and cohort as the final runs)
OVR = {"fed_gbdt": {"final_mortality_strict": "fedgbdt_strict_final", "final_mortality_allstays": "fedgbdt_allstays_final",
                "final_sepsis24": "fedgbdt_sepsis24_final", "final_sepsis6h": "fedgbdt_sepsis6h_mr1200"},
       "gbdt_central": {"final_mortality_strict": "gbdtcentral_strict", "final_mortality_allstays": "gbdtcentral_allstays",
                        "final_sepsis24": "gbdtcentral_sepsis24", "final_sepsis6h": "gbdtcentral_sepsis6h"}}
OPTIONAL = ("fed_gbdt", "gbdt_central")      # arms run in their own run directories; a row appears once its run has been collected
def arm_table(run, exp, order, apache):
    key = lambda a: (OVR.get(a, {}).get(run, run), exp, a)
    for a in order:                                   # a mistyped run folder must be loud, never a silently missing row
        if a in OPTIONAL and key(a) not in R: print(f"WARNING: no {NAMES[a]} row for {key(a)[0]} / {exp}: that run is not in the summary (check the folder name under outputs/paper)", file=sys.stderr)
    # the federated-GBDT row appears once its run has been collected; every other arm must exist (a missing one is an error, not a silent skip)
    rows = [[NAMES[a]] + row(R[key(a)]) for a in order if a not in OPTIONAL or key(a) in R]
    if apache: rows.append([f"APACHE IVa predicted mortality (benchmark, n={apache[0]:,})", f"{apache[1]:.4f}", f"[{apache[2]:.4f}, {apache[3]:.4f}]", f(apache[4]), "–", "–", "–", "–"])
    return rows
ORD = ["mlp_local_only", "fedavg_logreg", "fedavg_mlp", "fedavg_graph_none", "fedavg_graph_A", "fedavg_graph_B", "fed_gbdt", "logreg_central", "mlp_central", "lightgbm_central", "gbdt_central"]
T["t1"] = (HEAD, arm_table("final_mortality_strict", M, ORD, (15151, 0.8309, 0.8206, 0.8415, 0.3767)), "Mortality, strict 24h landmark cohort (131,695 stays; 204 hospital clients for the MLP, 206 for the GBDT; test n = 19,802).", "tab:mort_strict")
T["t2"] = (HEAD, arm_table("final_mortality_allstays", M, [a for a in ORD if a in ("fedavg_logreg", "fedavg_mlp", "fed_gbdt", "logreg_central", "mlp_central", "lightgbm_central", "gbdt_central")], (20503, 0.8642, 0.8558, 0.8728, 0.4518)), "Mortality, all stays with the first-24h window (199,108 stays; test n = 29,968). Includes the short-stay effect.", "tab:mort_all")
SORD = ["fedavg_logreg", "fedavg_mlp", "fed_gbdt", "logreg_central", "mlp_central", "lightgbm_central", "gbdt_central"]
T["t4a"] = (HEAD, arm_table("final_sepsis24", S, SORD, None), "Sepsis proxy, 24h window, clean features, label-valid hospitals (151 hospitals; test n = 16,238).", "tab:sep24")
rows_4b = arm_table("final_sepsis6h", S, SORD, None)
first_key = ("fedgbdt_sepsis6h_final", S, "fed_gbdt")           # the first 6 h GBDT run (600-round limit) stays visible next to the rerun
if first_key in R and any(r_[0] == NAMES["fed_gbdt"] for r_ in rows_4b):
    rows_4b.insert(next(n for n, r_ in enumerate(rows_4b) if r_[0] == NAMES["fed_gbdt"]) + 1, ["Federated GBDT, first run (600-round limit)"] + row(R[first_key]))
T["t4b"] = (HEAD, rows_4b, "Sepsis proxy, 6h early-warning window, clean features, label-valid hospitals (test n = 22,597). The federated GBDT row is the rerun with a 1,200-round limit (higher validation AUROC in all three seeds, early-stopped in every seed); the first run hit its 600-round limit in every seed.", "tab:sep6")
sweep = [("fed_v1", "0"), ("fed_v4", "0.01"), ("fed_r2a", "0.05"), ("fed_r2b", "0.1"), ("fed_r3a", "0.3"), ("fed_r3b", "1.0"), ("fed_r4a", "3.0"), ("fed_r4b", "10.0")]
T["t6"] = (["FedProx μ", "validation AUROC", "test AUROC", "per-hosp. macro AUROC", "AUROC, per-hosp. calibration"],
           [[m, f(R[(r, M, 'fedavg_mlp')]['val']), f(R[(r, M, 'fedavg_mlp')]['auroc']), f(R[(r, M, 'fedavg_mlp')]['perhosp']), f(R[(r, M, 'fedavg_mlp')]['local'])] for r, m in sweep],
           "Federated MLP versus the FedProx proximal weight (strict cohort, seed 42, lr 0.001, 2 local epochs). Selection on validation (μ = 1.0 chosen: μ = 3 was not better by 0.002).", "tab:fedprox")
fs, fa = R[("final_mortality_strict", M, "fedavg_mlp")], R[("final_mortality_allstays", M, "fedavg_mlp")]
gs, ga = R[("fedgbdt_strict_final", M, "fed_gbdt")], R[("fedgbdt_allstays_final", M, "fed_gbdt")]
def verdict(ref, ci_lo, ci_hi):
    return "ahead" if ci_lo > ref else ("level" if ci_lo <= ref <= ci_hi else "behind")
def vd(ref, r): return f"{verdict(ref, r['lo'], r['hi'])} ({r['auroc'] - ref:+.3f})"
# Base paper: FLICS. Papers on other or mixed databases are not compared.
FLICS = [("FLICS, FedAvg LR / LSTM", "3 hospitals, 5,000 stays, 1,000-patient test set (CI 0.83-0.90)", 0.87, "0.87"),
         ("FLICS, FedAvg LR", "10 hospitals, 53,500 stays (CI 0.79-0.87)", 0.83, "0.83"), ("FLICS, FedAvg LSTM", "10 hospitals, 53,500 stays (CI 0.79-0.88)", 0.84, "0.84")]
T["t3a"] = (["Reference (as reported)", "Setting", "AUROC", f"ours, strict: federated GBDT {gs['auroc']:.4f}", f"ours, strict: federated MLP {fs['auroc']:.4f}"],
            [[n, s, txt, vd(v, gs), vd(v, fs)] for n, s, v, txt in FLICS],
            "Base paper: FLICS (2026, eICU only). 'ahead' = our 95% CI lies above the reference; 'level' = it contains it; 'behind' = below. This is a comparison of reported numbers across different settings, not a replication: FLICS uses 3 or 10 hospitals, ICU-discharge mortality, whole-stay summaries and a 1,000-patient test set (reference CI up to ±0.035); ours is the strict 24 h cohort of 206 hospitals with hospital mortality (test n = 19,802). Our all-stays figures are not compared with FLICS (different setting and a short-stay effect). Tables 10 and 11 give same-patient comparisons (Table 11: models trained on federations of FLICS's size); Table 12 measures what FLICS's label alone does to our AUROC.", "tab:flics")
PS = PKG / "tables" / "paired_subsets.json"
if PS.exists():
    ps = json.load(open(PS, encoding="utf-8")); rows10 = []
    fm = lambda d: f"{d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
    for nm, s in ps["subsets"].items():
        rows10.append([nm, f"{s['n_test']:,} ({s['events']})", f"{s['auroc']['gbdt']:.4f}", f"{s['auroc']['lr']:.4f}", fm(s["gbdt_minus_lr"]), fm(s["gbdt_minus_mlp"]), fm(s["gbdt_minus_lgbm"])])
    T["t10"] = (["Test patients", "n (events)", "AUROC: federated GBDT", "AUROC: federated LR (FLICS's model class)", "GBDT - federated LR [95% CI]", "GBDT - federated MLP [95% CI]", "GBDT - pooled LightGBM [95% CI]"], rows10,
                "Paired method comparisons on identical strict-cohort test patients (seed 42; 1,000 paired bootstrap resamples; from reproduce/paired_subsets.py). The models were trained on all hospitals and are only evaluated on the 10 and 3 largest hospitals by cohort size: this is not a replication of FLICS's federations (Table 11 trains on federations of FLICS's size).", "tab:paired")
else:
    print("WARNING: tables/paired_subsets.json not found; run reproduce/paired_subsets.py to build Table 10", file=sys.stderr)
# Table 13 (federation cost for the GBDT: federated against the same code pooled, six settings): needs outputs/paper/gbdtcentral_* runs and tables/paired_central_gbdt.json
PCG = PKG / "tables" / "paired_central_gbdt.json"
if PCG.exists():
    _pcg = json.load(open(PCG, encoding="utf-8")); T["t13"] = central_gbdt_table(_pcg)
    if _pcg["skipped"]: print("WARNING: Table 13 lacks the pooled GBDT run for: " + ", ".join(_pcg["skipped"]), file=sys.stderr)
else:
    print("WARNING: tables/paired_central_gbdt.json not found; run reproduce/paired_central_gbdt.py to build Table 13", file=sys.stderr)
# Table 11 (FLICS-scale federations): needs outputs/paper/flicsstyle_top3 and flicsstyle_top10; the references are the same FLICS figures as Table 3a
PFS = PKG / "tables" / "paired_flicsscale.json"
FLICS_BY_SCALE = {"top3": [(FLICS[0][0], FLICS[0][2])], "top10": [(FLICS[1][0], FLICS[1][2]), (FLICS[2][0], FLICS[2][2])]}
if PFS.exists():
    T.update(tables_from_json(json.load(open(PFS, encoding="utf-8")), FLICS_BY_SCALE, verdict))
else:
    print("WARNING: tables/paired_flicsscale.json not found; run reproduce/paired_flicsscale.py to build Table 11", file=sys.stderr)
# Table 14 (final comparison in the format of FLICS Tables IV/V: accuracy, AUROC, F1; bold = best federated, underline = best central): reads outputs/paper/*/results.json only
T.update(fc.tables(REPO / "outputs" / "paper"))
# communication and wall-clock, read from the run files (federated GBDT) and derived from the architecture (federated MLP)
def gb_cost(run):
    e = list(json.load(open(REPO / "outputs" / "paper" / run / "results.json", encoding="utf-8"))["experiments"].values())[0]
    rs = [r["extra"] for r in e["records"] if r["arm"] == "fed_gbdt"]
    el = json.load(open(REPO / "outputs" / "paper" / run / "results.json", encoding="utf-8"))["elapsed_seconds"] / 60
    return (f"{min(r['best_iteration'] for r in rs)}-{max(r['best_iteration'] for r in rs)}", f"{sum(r['comm_mb_per_client_per_round_sparse'] for r in rs) / len(rs):.1f}",
            f"{min(r['comm_gb_total_sparse_all_clients'] for r in rs):.0f}-{max(r['comm_gb_total_sparse_all_clients'] for r in rs):.0f}", f"{el:.0f}")
cs, ca = gb_cost("fedgbdt_strict_final"), gb_cost("fedgbdt_allstays_final")
sep_rows = []                                   # sepsis GBDT rows appear once those runs are collected (folder names must match)
for label, run in (("Sepsis 24 h", "fedgbdt_sepsis24_final"), ("Sepsis 6 h", "fedgbdt_sepsis6h_mr1200")):
    if (REPO / "outputs" / "paper" / run / "results.json").exists():
        c = gb_cost(run); sep_rows.append([label, "Federated GBDT (151 label-valid hospitals)", c[0], c[1], c[2], c[3]])
    else:
        print(f"WARNING: no cost row for {run}: outputs/paper/{run}/results.json not found", file=sys.stderr)
T["t9"] = (["Protocol", "Model", "rounds (trees) kept", "uplink per client per round (MB)", "uplink, all clients, one model (GB)", "wall-clock, 4-trial search + 3 seeds (min)"],
           [["Strict", "Federated GBDT", cs[0], cs[1], cs[2], cs[3]], ["Strict", "Federated MLP (295,937 parameters)", "64-71 rounds run", "1.2", "15-17", "–"],
            ["All stays", "Federated GBDT", ca[0], ca[1], ca[2], ca[3]], ["All stays", "Federated MLP", "75-100 rounds run", "1.2", "18-24", "–"]] + sep_rows,
           "Communication of the two federated models. GBDT figures are sparse upper bounds (a client sends at most its active rows times the selected features per level, 12 bytes per cell) for the winning configuration of each seed; one 'round' is one tree (the 6 h sepsis row is the 1,200-round rerun, which early-stopped in every seed; its first run hit the 600-round limit; the 6 h wall-clock includes one seed that took about three times as long as the other two, probably machine load). The 'rounds kept' column shows the best round per seed. MLP uplink = one float32 copy of the weights per client per round. No compression, differential privacy or secure aggregation in either.", "tab:cost")
T["t7"] = (["Subgroup (strict cohort, seed 42)", "n", "events", "Federated MLP AUROC", "Central LightGBM AUROC"],
           [["overall", "19,802", "1,817", "0.8611", "0.8799"], ["age < 45", "2,658", "117", "0.9318", "0.9341"], ["age 45–64", "6,822", "512", "0.8786", "0.9007"], ["age 65–79", "6,681", "673", "0.8414", "0.8603"],
            ["age 80+", "3,632", "515", "0.7954", "0.8198"], ["Med-Surg ICU", "10,608", "991", "0.8574", "0.8763"], ["MICU", "1,867", "223", "0.8499", "0.8763"], ["CCU-CTICU", "1,661", "152", "0.8709", "0.8839"],
            ["Neuro ICU", "1,486", "111", "0.8685", "0.8816"], ["Cardiac ICU", "1,374", "133", "0.8657", "0.8811"], ["SICU", "1,336", "119", "0.8341", "0.8651"],
            ["same 15,151 patients with APACHE IVa", "15,151", "–", "0.8660 vs APACHE 0.8309 (Δ +0.0351 [+0.0273, +0.0437])", "0.8817 vs APACHE 0.8309 (Δ +0.0508 [+0.0433, +0.0587])"]],
           "Subgroup AUROC on the test split. Per-hospital AUROC over 102 hospitals with at least 5 events and 5 non-events: federated MLP median 0.867 (IQR 0.826–0.898), central LightGBM median 0.881 (IQR 0.839–0.917).", "tab:subgroups")
T["t8"] = (["Model (clean features)", "window", "AUROC vs proxy (target)", "AUROC vs documented sepsis [95% CI]", "AUPRC (prevalence)", "documented rate, lowest → highest risk quintile"],
           [["Central LightGBM", "24h", "0.8621", "0.7721 [0.7626, 0.7827]", "0.409 (0.155)", "5.3% → 40.4%"], ["Federated MLP", "24h", "0.8418", "0.7669 [0.7562, 0.7768]", "0.386 (0.155)", "3.9% → 38.0%"], ["Federated GBDT (seed 42)", "24h", "0.8603", "0.7683 [0.7585, 0.7787]", "0.399 (0.155)", "5.2% → 40.0%"],
            ["Central LightGBM", "6h", "0.8591", "0.7798 [0.7720, 0.7878]", "0.361 (0.137)", "4.4% → 36.0%"], ["Federated MLP", "6h", "0.8390", "0.7769 [0.7690, 0.7852]", "0.345 (0.137)", "3.3% → 34.4%"], ["Federated GBDT (seed 42)", "6h", "0.8574", "0.7796 [0.7717, 0.7873]", "0.359 (0.137)", "4.5% → 35.8%"]],
           "External check: scores trained on the antibiotic-plus-organ-dysfunction proxy also rank clinician-documented sepsis diagnoses (never a training target; infection diagnosis tokens are excluded from the features). The federated GBDT rows are a single seed (42); the 6 h row is the 1,200-round run (fedgbdt_sepsis6h_mr1200), whose five quintile rates, low to high, are 4.5%, 1.9%, 8.5%, 17.7% and 35.8%: the lowest quintile is above the second. Intervals are not paired.", "tab:documented")
# Table 12: the same seed-42 predictions scored against two death labels (values copied from the printed output of scripts/paper_label_variant_check.py --paired-bootstrap 2000, commands in docs/paper_track_v18.md)
LV = [("3 largest hospitals", "Federated LR", 1936, "9.1%", "4.0%", "55.9%", (0.8553, 0.8256, 0.8832), (0.8890, 0.8628, 0.9134), (0.0337, 0.0069, 0.0623)),
      ("3 largest hospitals", "Federated GBDT", 1936, "9.1%", "4.0%", "55.9%", (0.8603, 0.8344, 0.8824), (0.8946, 0.8678, 0.9198), (0.0343, 0.0089, 0.0617)),
      ("10 largest hospitals", "Federated LR", 5002, "9.7%", "5.1%", "47.8%", (0.8528, 0.8353, 0.8701), (0.8718, 0.8495, 0.8895), (0.0190, 0.0030, 0.0362)),
      ("10 largest hospitals", "Federated GBDT", 5002, "9.7%", "5.1%", "47.8%", (0.8675, 0.8521, 0.8827), (0.8889, 0.8710, 0.9055), (0.0214, 0.0062, 0.0365)),
      ("All hospitals (strict cohort)", "Federated GBDT", 19802, "9.2%", "5.2%", "44.1%", (0.8742, 0.8673, 0.8816), (0.8870, 0.8780, 0.8966), (0.0127, 0.0055, 0.0203))]
ci3 = lambda t: f"{t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]"
T["t12"] = (["Test patients (seed 42)", "Model", "test n", "in-hospital mortality", "ICU-discharge mortality", "hospital deaths after ICU discharge", "AUROC vs in-hospital death (our target)", "AUROC vs ICU-discharge death (FLICS's label)", "difference [95% paired CI]"],
            [[a, b, f"{n:,}", c, d, e, ci3(x), ci3(y_), f"{dd[0]:+.4f} [{dd[1]:+.4f}, {dd[2]:+.4f}]"] for a, b, n, c, d, e, x, y_, dd in LV],
            "What FLICS's label does to our AUROC. The same seed-42 predictions are scored against in-hospital death (the training target) and against ICU-discharge death (FLICS's label); no model was trained on ICU-discharge death. Values are copied from the printed output of "
            "scripts/paper_label_variant_check.py with --paired-bootstrap 2000. The two AUROC columns carry marginal bootstrap intervals; the difference column is a paired bootstrap over the test patients (one resample feeds both labels), so its interval is for the difference itself. One seed (42).", "tab:labelvariant")
out_md = []; out_tex = []
for k in ("t1", "t2", "t3a", "t10", "t11a", "t11b", "t11c", "t4a", "t4b", "t9", "t6", "t7", "t8", "t12", "t13", "t14a", "t14b", "t14c", "t14d", "t14e", "t14f"):
    if k.startswith(("t11", "t13", "t14")) and k not in T: continue                 # Table 11 appears once paired_flicsscale.json exists (warned above)
    h, rows, cap, lab = T[k]; out_md.append(f"**{cap}**\n\n{md(h, rows)}\n"); out_tex.append(tex(h, rows, cap, lab))
(PKG / "tables").mkdir(exist_ok=True); (PKG / "tables" / "tables.md").write_text("\n".join(out_md), encoding="utf-8"); open(PKG / "tables" / "tables.tex", "w", encoding="utf-8").write("% requires \\usepackage{booktabs}\n" + "\n".join(out_tex))
json.dump({k: {"header": v[0], "rows": v[1], "caption": v[2]} for k, v in T.items()}, open(PKG / "tables" / "tables.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print((PKG / "tables" / "tables.md").read_text(encoding="utf-8")[:200])
