"""Evaluation of federated models -- the ONLY FL module that reads test labels.

It runs after every round, checkpoint, hyper-parameter and threshold has been fixed from validation data. Per client:
  * the primary operating point is the F1-maximising threshold on the client's OWN validation patients (identical to the Non-FL regime, `hw_eval.select_threshold`);
  * the secondary operating point is the highest validation threshold reaching the recall target (`select_threshold_recall`);
  * a client whose validation split lacks a class cannot choose a local threshold: it uses the GLOBAL threshold from the federated validation histogram of the final model
    (`threshold_source = global_federated_validation_histogram`), and has no recall-target point.
Every evaluated row is tagged with `eval_set`: "paired" when the hospital satisfies the Non-FL minimum-data rule (the same hospitals and test patients as the Non-FL regime, used for every
paired comparison) or "fl_only" (clients the Non-FL regime could not train).
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from src.three_modality.hw_data import Cohort, hospital_task_data, stable_seed
from src.three_modality.hw_eval import aggregate, hospital_metrics, select_threshold, select_threshold_recall
from src.three_modality.hw_reports import PredictionStore, aggregate_class_reports


def base_row(spec: dict, hospitalid: int, counts: dict, seed: int, client=None) -> dict:
    return {"variant": spec["variant"], "config": spec["config"], "task": spec["task"], "hospitalid": hospitalid, "embedding_set": spec["embedding_set"], "graph": spec["graph"],
            "classifier": spec["classifier"], "sampling": spec["sampling"], "algorithm": spec["algorithm"], "method": spec.get("method", "global"), "seed": seed, "counts": counts,
            "status": None, "skip_reason": None, "threshold": None, "threshold_source": None, "metrics": None, "info": None, "seconds": None, "head_source": None,
            "participated_in_training": client is not None, "nonfl_eligible": bool(client.nonfl_eligible) if client is not None else False,
            "eval_set": None}


def evaluate_variant(spec: dict, cohort: Cohort, task: str, clients: list, skipped: list, scorer, ds_cfg: dict, store: "PredictionStore | None",
                     global_threshold: float = float("nan"), seed: int = 0) -> list:
    """One row per hospital of the task (evaluated, skipped-before-training or skipped-at-evaluation). `scorer(client)` -> {"p_val", "p_test", "head_source"?, "info"?} with probabilities."""
    rows = []
    for c in clients:
        d = hospital_task_data(cohort, c.hospitalid, task)                   # evaluator-side view: full counts and the test labels
        row = base_row(spec, c.hospitalid, d.counts, stable_seed(seed, "fl_eval", c.hospitalid), c)
        row["eval_set"] = "paired" if c.nonfl_eligible else "fl_only"
        if c.n_test == 0:
            row["status"], row["skip_reason"] = "skipped", "no test patients"
            rows.append(row)
            continue
        out = scorer(c)
        yv = c.y["val"]
        has_local = c.n_val > 0 and len(np.unique(yv)) == 2
        if has_local:
            thr, row["threshold_source"] = select_threshold(yv, out["p_val"]), "local_validation"
        elif np.isfinite(global_threshold):
            thr, row["threshold_source"] = float(global_threshold), "global_federated_validation_histogram"
        else:
            row["status"], row["skip_reason"] = "skipped", "validation split lacks a class and no global threshold is available"
            rows.append(row)
            continue
        y_test = d.test_labels_for_evaluation()                              # the ONLY use of test labels: after the model and every threshold are fixed
        row["threshold"] = float(thr)
        row["metrics"] = hospital_metrics(y_test, out["p_test"], thr)
        if has_local:
            target = float(ds_cfg["operating_points"]["recall_target"])
            thr2 = select_threshold_recall(yv, out["p_val"], target)
            m2 = hospital_metrics(y_test, out["p_test"], thr2)
            m2.pop("classification_report", None)
            row["operating_points"] = {"recall_target": {"target": target, "threshold": thr2, "metrics": m2}}
        if store is not None:
            store.add(spec["variant"], c.pid["test"], c.hospitalid, y_test, out["p_test"])
        row["head_source"] = out.get("head_source")
        row["info"] = out.get("info")
        row["status"] = "trained"                                           # "trained" = evaluated (the aggregate() helper's vocabulary)
        rows.append(row)
    for s in skipped:
        d = hospital_task_data(cohort, s["hospitalid"], task)
        row = base_row(spec, s["hospitalid"], d.counts, stable_seed(seed, "fl_eval", s["hospitalid"]), None)
        row["status"], row["skip_reason"], row["eval_set"] = "skipped", s["reason"], "not_a_client"
        rows.append(row)
    return sorted(rows, key=lambda r: r["hospitalid"])


def probs_from_logits(res_logits: dict) -> dict:
    return {k: expit(v) for k, v in res_logits.items()}


def summarise_variant(rows: list, ds_cfg: dict) -> dict:
    """Hospital-macro / sample-weighted aggregates reported separately for the paired set (hospitals the Non-FL regime also trains) and for all evaluated clients."""
    paired = [r for r in rows if r["eval_set"] == "paired"]
    evaluated = [r for r in rows if r["status"] == "trained"]
    out = {"paired_nonfl": aggregate(paired), "all_evaluated_clients": aggregate(rows)}
    out["paired_nonfl"]["definition"] = "hospitals satisfying the Non-FL minimum-data rule; these are the same hospitals and test patients the Non-FL regime evaluates"
    out["all_evaluated_clients"]["definition"] = "every FL client with a usable threshold and test patients (includes hospitals the Non-FL regime had to skip)"
    out["counts"] = {"n_rows": len(rows), "n_evaluated": len(evaluated), "n_paired_evaluated": sum(1 for r in paired if r["status"] == "trained"),
                     "n_fl_only_evaluated": sum(1 for r in evaluated if r["eval_set"] == "fl_only"), "n_not_a_client": sum(1 for r in rows if r["eval_set"] == "not_a_client"),
                     "n_global_threshold": sum(1 for r in evaluated if r["threshold_source"] != "local_validation")}
    rt = [{"status": r["status"], "metrics": ((r.get("operating_points") or {}).get("recall_target") or {}).get("metrics"), "skip_reason": r.get("skip_reason")}
          for r in rows if r["eval_set"] == "paired"]
    out["recall_target_operating_point_paired"] = {"target": ds_cfg["operating_points"]["recall_target"], **aggregate(rt)["metrics"]}
    return out


def class_reports(rows: list, ds_cfg: dict) -> dict:
    paired = [r for r in rows if r["eval_set"] == "paired"]
    rt = lambda rs: [{"status": r["status"], "variant": r["variant"], "hospitalid": r["hospitalid"],
                      "metrics": ((r.get("operating_points") or {}).get("recall_target") or {}).get("metrics")} for r in rs]
    return {"paired_nonfl": {"f1_max_operating_point": aggregate_class_reports(paired), "recall_target_operating_point": {"target": ds_cfg["operating_points"]["recall_target"], **aggregate_class_reports(rt(paired))}},
            "all_evaluated_clients": {"f1_max_operating_point": aggregate_class_reports(rows)}}
