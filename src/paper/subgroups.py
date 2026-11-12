"""Post-hoc subgroup / robustness analysis on SAVED patient-level predictions.
unitdischargeoffset is read here ONLY to define evaluation strata (e.g. stays shorter than the window) --
it is never a model input."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.paper.evaluation import auprc, bootstrap_ci, fast_auroc, paired_bootstrap_delta


def _stats(y, p, min_n, min_pos, boot_n=300):
    y = np.asarray(y); n = len(y); npos = int((y == 1).sum())
    if n < min_n or npos < min_pos or npos == n:
        return {"n": n, "n_positive": npos, "skipped": f"needs n>={min_n} and >= {min_pos} positives (and both classes)"}
    ci = bootstrap_ci(y, p, "auroc", boot_n, 0.05, 0)
    return {"n": n, "n_positive": npos, "prevalence": npos / n, "auroc": ci["point"], "auroc_lo": ci["lo"], "auroc_hi": ci["hi"], "auprc": auprc(y, p)}


def subgroup_report(pred, patient_df: pd.DataFrame, apache_df: "pd.DataFrame | None", threshold: float,
                    window_minutes: float = 1440, min_n: int = 300, min_pos: int = 25) -> dict:
    sp = np.asarray(pred["split"]).astype(str); m = sp == "test"
    df = pd.DataFrame({"patientunitstayid": np.asarray(pred["patient_id"])[m].astype("int64"), "hospital": np.asarray(pred["hospital_id"])[m],
                       "y": np.asarray(pred["y"])[m], "p": np.asarray(pred["p_calibrated"], dtype="float64")[m]})
    keep = [c for c in ("patientunitstayid", "unitdischargeoffset", "age", "unittype", "gender") if c in patient_df.columns]
    df = df.merge(patient_df[keep].drop_duplicates("patientunitstayid"), on="patientunitstayid", how="left")
    out = {"overall": _stats(df["y"], df["p"], 1, 1)}
    if "unitdischargeoffset" in df:
        los = pd.to_numeric(df["unitdischargeoffset"], errors="coerce")
        out["by_icu_los"] = {f"LOS >= {int(window_minutes)} min (full window observed)": _stats(df.y[los >= window_minutes], df.p[los >= window_minutes], min_n, min_pos),
                             f"LOS < {int(window_minutes)} min (window truncated)": _stats(df.y[los < window_minutes], df.p[los < window_minutes], min_n, min_pos)}
    if "age" in df:
        age = pd.to_numeric(df["age"].astype(str).str.replace(">", "", regex=False).str.strip(), errors="coerce")
        bins = pd.cut(age, [0, 44.99, 64.99, 79.99, 200], labels=["<45", "45-64", "65-79", "80+"])
        out["by_age"] = {str(b): _stats(df.y[bins == b], df.p[bins == b], min_n, min_pos) for b in bins.cat.categories}
    if "unittype" in df:
        top = df["unittype"].value_counts().index[:6]
        out["by_unit_type"] = {str(u): _stats(df.y[df.unittype == u], df.p[df.unittype == u], min_n, min_pos) for u in top}
    if apache_df is not None and "predictedhospitalmortality" in apache_df.columns:
        ad = apache_df
        if "apacheversion" in ad.columns:      # same rule as the runner: prefer the APACHE IVa row when a stay has several
            ad = ad.assign(_v=(ad["apacheversion"].astype(str).str.upper() != "IVA")).sort_values("_v", kind="stable")
        a = ad.drop_duplicates("patientunitstayid")[["patientunitstayid", "predictedhospitalmortality"]]
        d2 = df.merge(a, on="patientunitstayid", how="left")
        pm = pd.to_numeric(d2["predictedhospitalmortality"], errors="coerce"); k = pm.notna() & (pm >= 0) & (pm <= 1)
        if k.sum() >= min_n and d2.y[k].nunique() == 2:
            ours, ap = d2.p[k].to_numpy(), pm[k].to_numpy()
            out["vs_apache_same_patients"] = {"n": int(k.sum()), "ours_auroc": fast_auroc(d2.y[k], ours), "apache_auroc": fast_auroc(d2.y[k], ap),
                                              "delta_auroc": paired_bootstrap_delta(d2.y[k], ours, ap, "auroc", 300, 0.05, 0)}
    yy = df["y"].to_numpy().astype(int); pred_lab = (df["p"].to_numpy() >= threshold).astype(int)
    out["classification_report"] = classification_report(yy, pred_lab, target_names=["negative (0)", "positive (1)"], digits=3, zero_division=0)
    hs = []
    for h, g in df.groupby("hospital"):
        if (g.y == 1).sum() >= 5 and (g.y == 0).sum() >= 5:
            hs.append({"hospital": int(h), "n": len(g), "n_positive": int((g.y == 1).sum()), "auroc": fast_auroc(g.y, g.p)})
    if hs:
        a = np.array([x["auroc"] for x in hs]); out["per_hospital"] = {"n_hospitals": len(hs), "median_auroc": float(np.median(a)), "q25": float(np.quantile(a, .25)),
                                                                        "q75": float(np.quantile(a, .75)), "min": float(a.min()), "max": float(a.max()),
                                                                        "worst_5": sorted(hs, key=lambda x: x["auroc"])[:5]}
    return out


def render_markdown(title: str, rep: dict) -> str:
    def row(k, v):
        return f"| {k} | {v['n']} | {v['n_positive']} | " + (f"{v['auroc']:.4f} [{v['auroc_lo']:.4f}, {v['auroc_hi']:.4f}] | {v['auprc']:.4f} |" if "auroc" in v else f"{v['skipped']} | |")
    L = [f"# Subgroup analysis -- {title}", "", "Evaluation strata only; unitdischargeoffset is NOT a model input.", "",
         "| group | n | positives | AUROC [95% CI] | AUPRC |", "|---|---|---|---|---|", row("overall (test)", rep["overall"])]
    for sec in ("by_icu_los", "by_age", "by_unit_type"):
        for k, v in rep.get(sec, {}).items():
            L.append(row(f"{sec[3:]}: {k}", v))
    if "vs_apache_same_patients" in rep:
        v = rep["vs_apache_same_patients"]; d = v["delta_auroc"]
        L += ["", f"On the **same {v['n']} test patients that have an APACHE IVa prediction**: ours AUROC {v['ours_auroc']:.4f} vs APACHE {v['apache_auroc']:.4f} "
                  f"(Δ={d['delta']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}], p={d['p_value']:.3f})."]
    if "per_hospital" in rep:
        h = rep["per_hospital"]
        L += ["", f"Per-hospital AUROC over {h['n_hospitals']} hospitals (>=5 events and non-events): median {h['median_auroc']:.4f}, IQR [{h['q25']:.4f}, {h['q75']:.4f}], range [{h['min']:.4f}, {h['max']:.4f}]."]
    L += ["", "```", rep["classification_report"], "```"]
    return "\n".join(L)
