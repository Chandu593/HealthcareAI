"""Per-class classification reports and pooled-prediction metrics for the hospital-wise stages.

* `aggregate_class_reports`: from the per-hospital confusion matrices -- (a) the pooled-over-hospital-models report (counts summed over hospitals; sklearn-style per-class precision /
  recall / F1 / support, accuracy, macro and weighted averages) and (b) the hospital-macro per-class report (each hospital's per-class value averaged over the hospitals where it is
  defined, with the number of hospitals). Nothing is invented: a per-class value that is undefined for a hospital (no positives, nothing predicted positive, ...) is excluded and counted.
* `PredictionStore` / `pooled_metrics`: patient-level TEST predictions (saved after every model, threshold and hyper-parameter has been fixed) give pooled AUROC / AUPRC / accuracy /
  precision / recall / F1 / specificity with patient-level bootstrap intervals, for all trained hospitals and for FLICS-style subsets (the k largest trained hospitals). Each patient is
  classified at its own hospital's VALIDATION-selected threshold; pooled AUROC mixes the score scales of independently trained hospital models and is reported as such.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CLASSES = ("negative", "positive")


def _div(a, b):
    return (a / b) if b else None


def class_report_from_confusion(cm: dict) -> dict:
    tn, fp, fn, tp = (int(cm[k]) for k in ("tn", "fp", "fn", "tp"))
    n = tn + fp + fn + tp
    pos = {"precision": _div(tp, tp + fp), "recall": _div(tp, tp + fn), "f1-score": _div(2 * tp, 2 * tp + fp + fn), "support": tp + fn}
    neg = {"precision": _div(tn, tn + fn), "recall": _div(tn, tn + fp), "f1-score": _div(2 * tn, 2 * tn + fn + fp), "support": tn + fp}
    out = {"negative": neg, "positive": pos, "accuracy": _div(tp + tn, n), "n": n}
    for name, w in (("macro avg", None), ("weighted avg", True)):
        agg = {"support": n}
        for k in ("precision", "recall", "f1-score"):
            v = [neg[k], pos[k]]
            if any(x is None for x in v):
                agg[k] = None
            elif w:
                agg[k] = (neg[k] * neg["support"] + pos[k] * pos["support"]) / n if n else None
            else:
                agg[k] = (neg[k] + pos[k]) / 2
        out[name] = agg
    return out


def _hospital_class_values(cm: dict) -> dict:
    r = class_report_from_confusion(cm)
    return {c: {k: r[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in CLASSES}


def aggregate_class_reports(rows: list, metrics_key: str = "metrics") -> dict:
    trained = [r for r in rows if r.get("status") == "trained" and r.get(metrics_key)]
    cm = {k: sum(int(r[metrics_key]["confusion_matrix"][k]) for r in trained) for k in ("tn", "fp", "fn", "tp")}
    pooled = class_report_from_confusion(cm)
    macro = {}
    for c in CLASSES:
        macro[c] = {}
        for k in ("precision", "recall", "f1-score"):
            v = [_hospital_class_values(r[metrics_key]["confusion_matrix"])[c][k] for r in trained]
            v = [x for x in v if x is not None]
            macro[c][k] = float(np.mean(v)) if v else None
            macro[c][f"n_hospitals_{k.split('-')[0]}"] = len(v)
        macro[c]["support_total"] = pooled[c]["support"]
    return {"n_hospitals_trained": len(trained), "pooled_over_hospital_models": {**pooled, "confusion_matrix": cm}, "hospital_macro_per_class": macro}


def _fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def class_report_markdown(variant: str, rep: dict, title: str = "") -> str:
    p, m = rep["pooled_over_hospital_models"], rep["hospital_macro_per_class"]
    cm = p["confusion_matrix"]
    out = [f"#### {variant}{(' — ' + title) if title else ''}", "",
           f"{rep['n_hospitals_trained']} trained hospitals. Pooled over the hospital models (counts summed over hospitals; each patient classified at its own hospital's validation-selected threshold):", "",
           "| class | precision | recall | f1-score | support |", "|---|---|---|---|---|"]
    for c in CLASSES:
        out.append(f"| {c} | {_fmt(p[c]['precision'])} | {_fmt(p[c]['recall'])} | {_fmt(p[c]['f1-score'])} | {p[c]['support']} |")
    out += [f"| accuracy | | | {_fmt(p['accuracy'])} | {p['n']} |",
            f"| macro avg | {_fmt(p['macro avg']['precision'])} | {_fmt(p['macro avg']['recall'])} | {_fmt(p['macro avg']['f1-score'])} | {p['n']} |",
            f"| weighted avg | {_fmt(p['weighted avg']['precision'])} | {_fmt(p['weighted avg']['recall'])} | {_fmt(p['weighted avg']['f1-score'])} | {p['n']} |", "",
            f"Confusion (tn / fp / fn / tp): {cm['tn']} / {cm['fp']} / {cm['fn']} / {cm['tp']}.", "",
            "Hospital-macro per class (each hospital equal; averaged over the hospitals where the value is defined):", "", "| class | precision | recall | f1-score |", "|---|---|---|---|"]
    for c in CLASSES:
        out.append(f"| {c} | {_fmt(m[c]['precision'])} (n={m[c]['n_hospitals_precision']}) | {_fmt(m[c]['recall'])} (n={m[c]['n_hospitals_recall']}) | {_fmt(m[c]['f1-score'])} (n={m[c]['n_hospitals_f1']}) |")
    out.append("")
    return "\n".join(out)


def per_hospital_class_rows(rows: list, metrics_key: str = "metrics") -> list:
    out = []
    for r in rows:
        if r.get("status") != "trained" or not r.get(metrics_key):
            continue
        vals = _hospital_class_values(r[metrics_key]["confusion_matrix"])
        for c in CLASSES:
            out.append({"variant": r["variant"], "hospitalid": r["hospitalid"], "class": c, **{k.replace("-", "_"): vals[c][k] for k in ("precision", "recall", "f1-score", "support")}})
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# patient-level test predictions and pooled metrics
# ---------------------------------------------------------------------------------------------------------------------------------
class PredictionStore:
    def __init__(self):
        self.data = {}

    def add(self, variant: str, patient_ids, hospitalid: int, y_true, p):
        d = self.data.setdefault(variant, {"pid": [], "hid": [], "y": [], "p": []})
        d["pid"].append(np.asarray(patient_ids, dtype=np.int64))
        d["hid"].append(np.full(len(p), int(hospitalid), dtype=np.int64))
        d["y"].append(np.asarray(y_true, dtype=np.int8))
        d["p"].append(np.asarray(p, dtype=np.float64))

    def arrays(self) -> dict:
        out = {}
        for v, d in self.data.items():
            for k in ("pid", "hid", "y", "p"):
                out[f"{v}@{k}"] = np.concatenate(d[k]) if d[k] else np.empty(0)
        return out

    def save(self, path):
        np.savez_compressed(path, **self.arrays())


def load_predictions(path) -> dict:
    z = np.load(path)
    out = {}
    for key in z.files:
        v, k = key.rsplit("@", 1)
        out.setdefault(v, {})[k] = z[key]
    return out


def _scalar_metrics(y, p, yhat) -> dict:
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    two = (y == 1).any() and (y == 0).any()
    rec, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    return {"auroc": float(roc_auc_score(y, p)) if two else None, "auprc": float(average_precision_score(y, p)) if (y == 1).any() else None,
            "accuracy": _div(tp + tn, len(y)), "precision": _div(tp, tp + fp), "recall": rec, "f1": _div(2 * tp, 2 * tp + fp + fn), "specificity": spec,
            "balanced_accuracy": (rec + spec) / 2 if rec is not None and spec is not None else None}


def pooled_metrics(pred: dict, thresholds: dict, hospitals=None, n_boot: int = 500, seed: int = 0) -> dict:
    """Pooled test metrics over the patients of `hospitals` (None = all with predictions). `thresholds`: hospitalid -> that hospital's validation-selected threshold."""
    hid, y, p = pred["hid"], pred["y"].astype(int), pred["p"]
    keep = np.isin(hid, list(hospitals)) if hospitals is not None else np.ones(len(y), bool)
    hid, y, p = hid[keep], y[keep], p[keep]
    thr = np.array([thresholds[int(h)] for h in hid])
    yhat = (p >= thr).astype(int)
    out = {"n_patients": int(len(y)), "n_hospitals": int(len(np.unique(hid))), "prevalence": float(y.mean()) if len(y) else None,
           "all_negative_accuracy": float(1 - y.mean()) if len(y) else None, "point": _scalar_metrics(y, p, yhat)}
    rs = np.random.default_rng(seed)
    boots = {k: [] for k in out["point"]}
    for _ in range(int(n_boot)):
        i = rs.integers(0, len(y), len(y))
        m = _scalar_metrics(y[i], p[i], yhat[i])
        for k, v in m.items():
            if v is not None:
                boots[k].append(v)
    out["ci95"] = {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 10 else None) for k, v in boots.items()}
    out["ci_note"] = "patient-level bootstrap; pooled AUROC mixes the score scales of independently trained hospital models"
    return out
