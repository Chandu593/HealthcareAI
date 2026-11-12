import json, numpy as np, matplotlib
from pathlib import Path
PKG = Path(__file__).resolve().parents[1]; REPO = PKG.parent; FIG = PKG / "figures"; FIG.mkdir(exist_ok=True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from parse import load
R = load()
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7", "grey": "#7f7f7f", "sky": "#56B4E9", "black": "#000000"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
M = "mortality/no_apache_scores"

# ---- Fig 1: FedProx weight sweep (strict cohort, 1 seed) ----
runs = [("fed_v1", 0.0), ("fed_v4", 0.01), ("fed_r2a", 0.05), ("fed_r2b", 0.1), ("fed_r3a", 0.3), ("fed_r3b", 1.0), ("fed_r4a", 3.0), ("fed_r4b", 10.0)]
mu = [m for _, m in runs]; test = [R[(r, M, "fedavg_mlp")]["auroc"] for r, _ in runs]; val = [R[(r, M, "fedavg_mlp")]["val"] for r, _ in runs]
cen = R[("mlp_central_noseq", M, "mlp_central")]
fig, ax = plt.subplots(figsize=(5.4, 3.5))
ax.plot(mu, test, "o-", color=OI["blue"], label="federated MLP, test"); ax.plot(mu, val, "s--", color=OI["orange"], label="federated MLP, validation")
ax.axhline(cen["auroc"], color=OI["black"], ls=":", lw=1); ax.text(13, cen["auroc"] + 0.0006, "central MLP (test)", fontsize=8, ha="right")
ax.axhline(0.84, color=OI["grey"], ls="--", lw=0.8); ax.text(13, 0.8385, "FLICS 10-hospital, 0.84", fontsize=8, color=OI["grey"], ha="right", va="top")
ax.axvline(1.0, color=OI["green"], lw=0.8, alpha=0.7); ax.text(1.12, 0.8315, "chosen on\nvalidation", fontsize=8, color=OI["green"])
ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(-0.002, 14); ax.set_xticks([0, 0.01, 0.1, 0.3, 1, 3, 10]); ax.set_xticklabels(["0", "0.01", "0.1", "0.3", "1", "3", "10"])
ax.set_xlabel("FedProx proximal weight μ (symlog axis; μ=0.05 also run)"); ax.set_ylabel("AUROC (strict 24h cohort, seed 42)"); ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.set_ylim(0.828, 0.872); fig.tight_layout(); fig.savefig(FIG / "fig1_fedprox_sweep.png", dpi=220); plt.close(fig)

# ---- Fig 2: main comparison with reference lines ----
# the federated-GBDT and pooled-GBDT arms were run in their own run directories (same split, seeds and cohort as the final runs)
ARM_RUN = {"fed_gbdt": {"final_mortality_strict": "fedgbdt_strict_final", "final_mortality_allstays": "fedgbdt_allstays_final"},
           "gbdt_central": {"final_mortality_strict": "gbdtcentral_strict", "final_mortality_allstays": "gbdtcentral_allstays"}}
def panel(ax, run, title, include_local):
    arms = [("mlp_local_only", "Local-only\nMLP", OI["grey"]), ("fedavg_logreg", "Federated\nLR", OI["sky"]), ("fedavg_mlp", "Federated\nMLP", OI["blue"]), ("fed_gbdt", "Federated\nGBDT", OI["purple"]),
            ("logreg_central", "Central\nLR", OI["sky"]), ("mlp_central", "Central\nMLP", OI["orange"]), ("lightgbm_central", "Central\nLightGBM", OI["red"]), ("gbdt_central", "Central\nGBDT", OI["green"])]
    arms = [a for a in arms if include_local or a[0] != "mlp_local_only"]
    for a in [a for a in arms if a[0] == "gbdt_central" and (ARM_RUN["gbdt_central"].get(run, run), M, "gbdt_central") not in R]:
        print(f"WARNING: Figure 2 has no Central GBDT bar for {run}: that run is not in the summary (run scripts/paper_collect_results.py)"); arms.remove(a)
    vals, los, his = [], [], []
    for a, _, _ in arms:
        r = R[(ARM_RUN.get(a, {}).get(run, run), M, a)]; vals.append(r["auroc"]); los.append(r["auroc"] - r["lo"]); his.append(r["hi"] - r["auroc"])
    x = np.arange(len(arms)); ax.bar(x, vals, yerr=[los, his], color=[c for *_, c in arms], capsize=2, width=0.65)
    for xi, v in zip(x, vals): ax.text(xi, v + 0.012 if v > 0.8 else v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5)
    refs = [(0.84, "FLICS 10-hosp. 0.84", "--"), (0.87, "FLICS 3-hosp. 0.87", "--")]
    for ref, lab, ls in refs:
        ax.axhline(ref, color=OI["black"], ls=ls, lw=0.7, alpha=0.6); ax.text(len(arms) - 0.45, ref + 0.0015, lab, fontsize=7, ha="left")
    ap = R[(run, M, "fedavg_mlp")]
    ax.set_xticks(x); ax.set_xticklabels([a[1] for a in arms], fontsize=7); ax.set_xlim(-0.6, len(arms) + 1.6); ax.set_ylim(0.70 if include_local else 0.82, 0.93); ax.set_title(title, fontsize=9); ax.set_ylabel("test AUROC")
fig, axs = plt.subplots(1, 2, figsize=(12.4, 3.8)); panel(axs[0], "final_mortality_strict", "Strict 24h cohort (n_test=19,802)", True); panel(axs[1], "final_mortality_allstays", "All stays, first 24h (n_test=29,968)", False)
fig.text(0.5, 0.005, "bars: mean over 3 seeds; whiskers: 95% bootstrap CI (seed 42); dashed: FLICS (base paper), as reported. Reference cohorts, labels and protocols differ; on the all-stays panel the lines are context only, no comparison is claimed", ha="center", fontsize=7.5)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(FIG / "fig2_mortality_main.png", dpi=220); plt.close(fig)

# ---- Fig 3: sepsis ladder ----
def ladder(ax, h, title):
    p = "sepsis24" if h == 24 else "sepsis6h"; lab = [("uncontrolled", (f"{p}_central", "sepsis/sepsis_uncontrolled")), ("controlled", (f"{p}_central", "sepsis/sepsis_controlled")),
                              ("decon-\nfounded", (f"{p}_ladder", "sepsis/sepsis_deconfounded")), ("no hosp.\ncovariates", (f"{p}_nohosp", "sepsis/sepsis_no_hospital")),
                              ("no labs", (f"{p}_ladder", "sepsis/sepsis_no_labs")), ("controlled", (f"{p}_valid", "sepsis/sepsis_controlled")), ("clean", ("final_sepsis24" if h == 24 else "final_sepsis6h", "sepsis/sepsis_clean"))]
    pooled = [R[(a, e, "lightgbm_central")]["auroc"] for _, (a, e) in lab]; ph = [R[(a, e, "lightgbm_central")]["perhosp"] for _, (a, e) in lab]
    x = np.arange(len(lab)); w = 0.38
    ax.bar(x - w / 2, pooled, w, color=OI["blue"], label="pooled AUROC"); ax.bar(x + w / 2, ph, w, color=OI["orange"], label="per-hospital macro AUROC")
    for xi, v in zip(x - w / 2, pooled): ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.5, rotation=90)
    for xi, v in zip(x + w / 2, ph): ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.5, rotation=90)
    ax.axvline(4.5, color=OI["grey"], ls=":"); ax.text(2.0, -0.2, "all 207 hospitals (label artefact present)", ha="center", fontsize=7.5, color=OI["black"], transform=ax.get_xaxis_transform()); ax.text(5.5, -0.2, "label-valid hospitals (151)", ha="center", fontsize=7.5, color=OI["black"], transform=ax.get_xaxis_transform())
    ax.set_xticks(x); ax.set_xticklabels([l for l, _ in lab], fontsize=7); ax.set_ylim(0.70, 1.03); ax.set_title(title, fontsize=9); ax.set_ylabel("AUROC (LightGBM, 1 seed; clean = 3 seeds)", fontsize=8)
fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True); ladder(axs[0], 24, "Sepsis proxy, 24h window"); ladder(axs[1], 6, "Sepsis proxy, 6h window"); axs[0].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "fig3_sepsis_ladder.png", dpi=220); plt.close(fig)

# ---- Fig 4: per-hospital AUROC distribution (strict mortality) from results.json ----
blk = json.load(open(REPO / "outputs" / "paper" / "final_mortality_strict" / "results.json", encoding="utf-8"))["experiments"]["mortality/no_apache_scores"]
blk_gb = json.load(open(REPO / "outputs" / "paper" / "fedgbdt_strict_final" / "results.json", encoding="utf-8"))["experiments"]["mortality/no_apache_scores"]
names = [("mlp_local_only", "Local-only\nMLP"), ("fedavg_mlp", "Federated\nMLP"), ("fed_gbdt", "Federated\nGBDT"), ("mlp_central", "Central\nMLP"), ("lightgbm_central", "Central\nLightGBM")]
fig, ax = plt.subplots(figsize=(5.0, 3.4)); xs = np.arange(len(names))
for i, (a, lab) in enumerate(names):
    rec = [r for r in (blk_gb if a == "fed_gbdt" else blk)["records"] if r["arm"] == a and r["seed"] == 42][0]["test_per_hospital"]
    ax.errorbar(i, rec["median_auroc"], yerr=[[rec["median_auroc"] - rec["q25_auroc"]], [rec["q75_auroc"] - rec["median_auroc"]]], fmt="o", color=OI["blue"], capsize=4)
    ax.text(i + 0.08, rec["median_auroc"], f"{rec['median_auroc']:.3f}", fontsize=7.5, va="center")
ax.set_xticks(xs); ax.set_xticklabels([n[1] for n in names], fontsize=8); ax.set_ylabel("per-hospital test AUROC"); ax.set_title("Median and IQR over hospitals (strict cohort, seed 42)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig4_per_hospital.png", dpi=220); plt.close(fig)
print("figures written")
