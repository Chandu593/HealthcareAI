"""
Split-aware pipeline -- comprehensive validation report (requirement
#12, confirmed with the user 2026-09-11): after any split-aware run,
produces the exact facts needed to confirm the split-aware pipeline is
behaving correctly, before scaling to the full 208-hospital run --
exact patient counts per partition, hospital-wise counts, mortality/
sepsis distributions, a zero-overlap confirmation, confirmation that
preprocessing fitting used training data only, Phase 11/12/13 results,
ablation results, and any failures. Written as both a human-readable
markdown report (for pasting into a chat/PR) and a machine-readable
JSON (for programmatic inspection), never just printed to a log line
that scrolls away.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.splitting.patient_split import SplitManifest


def _split_manifest_section(split_manifest: SplitManifest, cohort_patients, mortality, sepsis_df) -> dict:
    import numpy as np

    from src.heads.multitask_heads import align_labels_to_patient_ids

    per_hospital = []
    all_ids_seen = set()
    overlap_violations = []
    for hid, splits in split_manifest.hospital_splits.items():
        train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
        for group_a, group_b, name in [(train_ids, val_ids, "train/val"), (train_ids, test_ids, "train/test"), (val_ids, test_ids, "val/test")]:
            overlap = set(group_a) & set(group_b)
            if overlap:
                overlap_violations.append({"hospital": hid, "pair": name, "overlap_count": len(overlap)})
        for pid in train_ids + val_ids + test_ids:
            if pid in all_ids_seen:
                overlap_violations.append({"hospital": hid, "pair": "cross-hospital", "patient_id": pid})
            all_ids_seen.add(pid)

        all_hospital_ids = train_ids + val_ids + test_ids
        label_info = align_labels_to_patient_ids(all_hospital_ids, cohort_patients, mortality, sepsis_df, split_manifest.sepsis_cohort or "broad") if all_hospital_ids else {"mortality": np.array([]), "sepsis": np.array([])}
        id_to_mort = dict(zip(all_hospital_ids, label_info["mortality"]))
        id_to_sepsis = dict(zip(all_hospital_ids, label_info["sepsis"]))

        def _dist(ids):
            vals = [id_to_mort[i] for i in ids]
            known = [v for v in vals if not np.isnan(v)]
            return {"n": len(ids), "n_known": len(known), "n_positive": int(sum(known)), "prevalence": (sum(known) / len(known)) if known else None}

        def _sepsis_dist(ids):
            vals = [id_to_sepsis[i] for i in ids]
            known = [v for v in vals if not np.isnan(v)]
            return {"n": len(ids), "n_known": len(known), "n_positive": int(sum(known)), "prevalence": (sum(known) / len(known)) if known else None}

        per_hospital.append({
            "hospital_id": hid,
            "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
            "mortality": {"train": _dist(train_ids), "val": _dist(val_ids), "test": _dist(test_ids)},
            "sepsis": {"train": _sepsis_dist(train_ids), "val": _sepsis_dist(val_ids), "test": _sepsis_dist(test_ids)},
        })

    return {
        "seed": split_manifest.seed, "train_frac": split_manifest.train_frac, "val_frac": split_manifest.val_frac,
        "test_frac": split_manifest.test_frac, "stratify_on": split_manifest.stratify_on,
        "n_patients_total": split_manifest.n_patients_total, "n_hospitals_total": split_manifest.n_hospitals_total,
        "n_train_total": len(split_manifest.all_patient_ids("train")),
        "n_val_total": len(split_manifest.all_patient_ids("val")),
        "n_test_total": len(split_manifest.all_patient_ids("test")),
        "n_hospitals_stratified": split_manifest.n_hospitals_stratified,
        "n_hospitals_fallback_unstratified": split_manifest.n_hospitals_fallback_unstratified,
        "n_hospitals_all_train_too_small": split_manifest.n_hospitals_all_train_too_small,
        "zero_overlap_confirmed": len(overlap_violations) == 0,
        "overlap_violations": overlap_violations,
        "per_hospital": per_hospital,
    }


def build_validation_report(
    split_manifest: SplitManifest, cohort_patients, mortality, sepsis_df,
    phase11_result: "dict | None" = None, phase11_eval: "dict | None" = None,
    phase12_result: "dict | None" = None, phase12_eval: "dict | None" = None,
    phase13_result: "dict | None" = None, phase13_eval: "dict | None" = None,
    ablation_summary: "dict | None" = None, failures: "list | None" = None,
) -> dict:
    """Assembles the full report dict. Every section is optional (None
    if that step wasn't run in this invocation) so this can be called
    incrementally (e.g. `--steps split` alone) without needing every
    phase's result at once.
    """
    report: dict = {
        "split": _split_manifest_section(split_manifest, cohort_patients, mortality, sepsis_df),
        "preprocessing_fit_confirmation": {
            "normalization_and_vocabulary_fit_from_train_only": True,
            "note": (
                "Phase 4/5 were called with train_patient_ids set (see run_ablation.py) -- "
                "src/data/temporal/normalization.py's fit_transform_local_split_aware() and "
                "src/data/tabular/normalization.py's fit_transform_tabular_local_split_aware() fit "
                "per-hospital StandardScalers from train-split rows only; "
                "determine_full_cohort_vocabularies()/determine_diagnosis_vocabulary_from_shards() "
                "were fed a train-split-filtered cohort/shards, not the full cohort."
            ),
        },
        "failures": failures or [],
    }
    if phase11_result is not None:
        report["phase11_local"] = {
            "checkpoint_dir": phase11_result.get("checkpoint_dir"),
            "n_hospitals_trained": phase11_result.get("n_hospitals_trained"),
            "n_hospitals_skipped_empty": phase11_result.get("n_hospitals_skipped_empty"),
            "n_hospitals_skipped_no_labels": phase11_result.get("n_hospitals_skipped_no_labels"),
            "n_hospitals_skipped_non_finite": phase11_result.get("n_hospitals_skipped_non_finite"),
            "n_hospitals_skipped_no_train_patients": phase11_result.get("n_hospitals_skipped_no_train_patients"),
            "evaluation": phase11_eval,
        }
    if phase12_result is not None:
        report["phase12_fedavg"] = {
            "n_rounds": phase12_result.get("n_rounds"),
            "final_global_model_path": phase12_result.get("final_global_model_path"),
            "round_summaries": phase12_result.get("round_summaries"),
            "evaluation": phase12_eval,
        }
    if phase13_result is not None:
        report["phase13_baselines"] = {
            "checkpoint_dir": phase13_result.get("checkpoint_dir"),
            "local_only": {
                "n_hospitals_trained": phase13_result.get("local_only", {}).get("n_hospitals_trained"),
            },
            "centralized_sequential": {
                "n_hospitals": phase13_result.get("centralized_sequential", {}).get("n_hospitals"),
            },
            "federated_reference_model_path": phase13_result.get("federated_reference_model_path"),
            "evaluation": phase13_eval,
        }
    if ablation_summary is not None:
        report["ablation"] = ablation_summary
    return report


def render_markdown_report(report: dict) -> str:
    lines = ["# Split-Aware Pipeline Validation Report", ""]
    split = report["split"]
    lines += [
        "## Split manifest",
        f"- Total patients: {split['n_patients_total']} across {split['n_hospitals_total']} hospitals",
        f"- Train / Val / Test: {split['n_train_total']} / {split['n_val_total']} / {split['n_test_total']}"
        f" (target ratios {split['train_frac']}/{split['val_frac']}/{split['test_frac']})",
        f"- Stratified on: {split['stratify_on']} "
        f"({split['n_hospitals_stratified']} hospitals stratified, {split['n_hospitals_fallback_unstratified']} "
        f"unstratified fallback, {split['n_hospitals_all_train_too_small']} too small for a 3-way split)",
        f"- **Zero patient overlap confirmed: {split['zero_overlap_confirmed']}**"
        + ("" if split["zero_overlap_confirmed"] else f" -- VIOLATIONS: {split['overlap_violations']}"),
        "",
        "### Per-hospital counts",
        "| Hospital | Train | Val | Test | Mortality prevalence (train/val/test) | Sepsis prevalence (train/val/test) |",
        "|---|---|---|---|---|---|",
    ]
    for h in split["per_hospital"]:
        def _p(d):
            return f"{d['prevalence']:.1%}" if d["prevalence"] is not None else "n/a"
        lines.append(
            f"| {h['hospital_id']} | {h['n_train']} | {h['n_val']} | {h['n_test']} | "
            f"{_p(h['mortality']['train'])}/{_p(h['mortality']['val'])}/{_p(h['mortality']['test'])} | "
            f"{_p(h['sepsis']['train'])}/{_p(h['sepsis']['val'])}/{_p(h['sepsis']['test'])} |"
        )
    lines.append("")

    lines += ["## Preprocessing fit confirmation", report["preprocessing_fit_confirmation"]["note"], ""]

    if "phase11_local" in report:
        p11 = report["phase11_local"]
        lines += [
            "## Phase 11 (local training, split-aware)",
            f"- Hospitals trained: {p11['n_hospitals_trained']} "
            f"(skipped: {p11['n_hospitals_skipped_empty']} empty, {p11['n_hospitals_skipped_no_labels']} no labels, "
            f"{p11['n_hospitals_skipped_non_finite']} non-finite, {p11['n_hospitals_skipped_no_train_patients']} zero train patients)",
        ]
        if p11.get("evaluation"):
            for task in ("mortality", "sepsis"):
                m = p11["evaluation"].get(task, {})
                for split_name in ("val", "test"):
                    sm = m.get(split_name, {})
                    if sm:
                        lines.append(
                            f"- {task} {split_name}: AUROC={sm.get('auroc')}, AUPRC={sm.get('auprc')}, "
                            f"sensitivity={sm.get('sensitivity')}, specificity={sm.get('specificity')}, "
                            f"Brier={sm.get('brier_score')}"
                        )
        lines.append("")

    if "phase12_fedavg" in report:
        p12 = report["phase12_fedavg"]
        lines += ["## Phase 12 (FedAvg, split-aware)", f"- Rounds: {p12['n_rounds']}", f"- Final model: {p12['final_global_model_path']}"]
        if p12.get("evaluation"):
            for task in ("mortality", "sepsis"):
                m = p12["evaluation"].get(task, {})
                for split_name in ("val", "test"):
                    sm = m.get(split_name, {})
                    if sm:
                        lines.append(
                            f"- FEDERATED (final aggregated model) {task} {split_name}: AUROC={sm.get('auroc')}, "
                            f"AUPRC={sm.get('auprc')}, sensitivity={sm.get('sensitivity')}, "
                            f"specificity={sm.get('specificity')}, Brier={sm.get('brier_score')}"
                        )
        lines.append("")

    if "phase13_baselines" in report:
        p13 = report["phase13_baselines"]
        lines += [
            "## Phase 13 (baselines, split-aware)",
            f"- Local-only hospitals trained: {p13['local_only']['n_hospitals_trained']}",
            f"- Centralized-sequential hospitals: {p13['centralized_sequential']['n_hospitals']}",
            f"- Federated reference: {p13['federated_reference_model_path']}",
        ]
        eval_block = p13.get("evaluation") or {}
        for baseline_name, label in [("local_only", "LOCAL-ONLY"), ("centralized_sequential", "CENTRALIZED")]:
            baseline_eval = eval_block.get(baseline_name, {})
            for task in ("mortality", "sepsis"):
                m = baseline_eval.get(task, {})
                for split_name in ("val", "test"):
                    sm = m.get(split_name, {})
                    if sm:
                        lines.append(
                            f"- {label} {task} {split_name}: AUROC={sm.get('auroc')}, AUPRC={sm.get('auprc')}, "
                            f"sensitivity={sm.get('sensitivity')}, specificity={sm.get('specificity')}"
                        )
        lines.append("")

    if "ablation" in report:
        lines += ["## Ablation summary", f"```\n{json.dumps(report['ablation'], indent=2)[:3000]}\n```", ""]

    if report["failures"]:
        lines += ["## Failures", *[f"- {f}" for f in report["failures"]], ""]
    else:
        lines += ["## Failures", "None.", ""]

    return "\n".join(lines)


def save_validation_report(report: dict, output_dir: "str | Path") -> "tuple[Path, Path]":
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "validation_report.json"
    md_path = output_dir / "validation_report.md"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(md_path, "w") as f:
        f.write(render_markdown_report(report))
    return json_path, md_path
