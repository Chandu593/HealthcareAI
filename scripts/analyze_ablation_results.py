#!/usr/bin/env python3
"""
Fed-THOR ablation results analysis script.

Reads master_results.csv (embedding_ablation: TS x TAB x Fusion x
Classifier grid) and, optionally, graph_results.csv (graph_ablation:
GNN embedding x Classifier), and produces a comprehensive, research-
grade analysis:

  - data-integrity checks (skip flags, degenerate precision/recall,
    threshold sanity)
  - overfitting diagnostics (train vs val vs test gaps, by classifier)
  - breakdowns by ts_model / tab_model / fusion_mode / task / classifier
  - runtime analysis (encoder training cost, classifier fit cost)
  - best-performing configuration by AUROC, AUPRC, F1, recall, and
    accuracy -- each computed SEPARATELY, per task, since they can
    disagree (see the printed caveats -- accuracy/recall/F1 are
    threshold-dependent and less trustworthy than AUROC/AUPRC under
    class imbalance; this project's mortality prevalence is ~9%)
  - a flat-vs-graph-embedding comparison, with an explicit check for
    (and warning about) mismatched classifier hyperparameters between
    the two groups being compared -- a real, confirmed issue in this
    project's data as of 2026-09-17 (graph_ablation was re-run with
    tuned classifier params after the embedding grid had already been
    completed with defaults)
  - a per-configuration sklearn-native classification_report (via
    sklearn.metrics.classification_report itself, not a hand-rolled
    imitation of its formatting) for the top 10 configurations by test
    AUPRC, for BOTH mortality and sepsis, to help spot class-imbalance
    effects (e.g. strong majority-class recall/specificity alongside
    weak minority-class precision/recall) worth addressing with
    sampling techniques. Reconstructed from this project's stored
    aggregate stats (n, n_positive, recall, specificity) rather than
    raw predictions, which the CSVs don't carry -- see
    `_reconstruct_confusion_counts()`'s docstring for why this
    reconstruction is exact (a binary confusion matrix has only 4
    cells and is fully determined by n, n_positive, recall, and
    specificity), not an approximation.

LEGACY DATA RECOVERY: earlier runs (before the 2026-09-16
ResultsWriter fix) wrote graph_ablation rows into master_results.csv
itself, under embedding_ablation's column header -- since the two
experiment kinds have different config/training_info shapes, every
value in those rows lands under the wrong column name when read via
the file's header (pandas, Excel, csv.DictReader, etc. all do this).
This script auto-detects those legacy rows (experiment_kind ==
'graph_ablation' present in master_results.csv) and re-decodes them
positionally, using the schema the code actually wrote them with, so
they're usable rather than silently wrong or silently dropped. Runs
using the FIXED ResultsWriter write graph_ablation results to their
own graph_results.csv from the start and never hit this path.

ALL OUTPUT IS PRINTED TO THE TERMINAL ONLY -- nothing is written to
disk by this script.

Usage:
    python scripts/analyze_ablation_results.py --master outputs/ablation/full_run/master_results.csv --graph outputs/ablation/full_run/graph_results.csv --top-n 3

Requires only pandas + numpy + scikit-learn (no project-specific
imports) so it can run standalone on the machine that produced the
CSVs.
"""

from __future__ import annotations

import argparse
import csv as csv_module
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

# Real time-series encoders this project's embedding_ablation grid
# uses -- anything else in config.ts_model (e.g. a GNN type name, from
# a legacy-corrupted graph_ablation row) is not a genuine embedding
#_ablation row and gets excluded/recovered instead.
VALID_TS_MODELS = {"lstm", "rnn", "gru", "transformer"}

# The exact field order graph_ablation records were written with (see
# src/ablation/graph_embedding_ablation_runner.py's ExperimentRecord
# construction) -- used ONLY to recover legacy-corrupted rows found
# embedded in master_results.csv. A clean graph_results.csv (written
# by the fixed ResultsWriter) is read directly via its own real header
# instead and never needs this.
_GRAPH_ABLATION_FIELD_ORDER = (
    ["experiment_id", "experiment_kind", "task", "split_manifest_seed", "timestamp", "runtime_seconds",
     "config.gnn_type", "config.classifier", "config.classifier_params",
     "training.gnn_type", "training.n_hospitals_total", "training.n_hospitals_trained",
     "training.n_hospitals_skipped_empty", "training.n_hospitals_skipped_no_train_patients",
     "training.n_hospitals_skipped_no_labels", "training.n_hospitals_skipped_non_finite",
     "training.n_hospitals_skipped_empty_graph", "training.encoder_train_runtime_seconds",
     "training.classifier_fit_runtime_seconds", "training.classifier_skipped", "training.classifier_skip_reason"]
    + [f"{s}.{m}" for s in ("train", "val", "test")
       for m in ("n", "n_positive", "threshold_used", "accuracy", "precision", "recall", "sensitivity",
                 "f1", "specificity", "brier_score", "auroc", "auprc")]
)

NUMERIC_METRIC_COLS = [
    f"{s}.{m}" for s in ("train", "val", "test")
    for m in ("n", "n_positive", "threshold_used", "accuracy", "precision", "recall", "sensitivity",
              "f1", "specificity", "brier_score", "auroc", "auprc")
]


# --------------------------------------------------------------------------
# Loading + legacy recovery
# --------------------------------------------------------------------------

def _recover_legacy_graph_rows(master_csv_path: Path) -> pd.DataFrame:
    """Re-reads master_results.csv as raw rows (ignoring its header)
    and re-decodes any experiment_kind == 'graph_ablation' row using
    the schema it was ACTUALLY written with. Returns an empty
    DataFrame if none are found."""
    recovered_rows = []
    with open(master_csv_path, newline="") as f:
        reader = csv_module.reader(f)
        header = next(reader, None)
        if header is None:
            return pd.DataFrame()
        try:
            kind_idx = header.index("experiment_kind")
        except ValueError:
            return pd.DataFrame()
        for raw_row in reader:
            if len(raw_row) > kind_idx and raw_row[kind_idx] == "graph_ablation":
                if len(raw_row) != len(_GRAPH_ABLATION_FIELD_ORDER):
                    warnings.warn(
                        f"Found a graph_ablation row in {master_csv_path} with {len(raw_row)} fields, "
                        f"expected {len(_GRAPH_ABLATION_FIELD_ORDER)} -- skipping it (schema may have "
                        f"changed further since this recovery path was written)."
                    )
                    continue
                recovered_rows.append(dict(zip(_GRAPH_ABLATION_FIELD_ORDER, raw_row)))
    return pd.DataFrame(recovered_rows)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col.startswith(("train.", "val.", "test.")) or col in (
            "runtime_seconds", "split_manifest_seed", "training.encoder_train_runtime_seconds",
            "training.classifier_fit_runtime_seconds", "training.n_hospitals_trained",
            "training.n_hospitals_total",
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_results(master_path: str, graph_path: "str | None" = None) -> "tuple[pd.DataFrame, pd.DataFrame, list[str]]":
    """Returns (embedding_df, graph_df, notes) -- notes is a list of
    human-readable strings describing any data-quality issue found
    while loading (legacy rows recovered, files missing, etc.), meant
    to be printed prominently, not buried in a warning stream."""
    notes: list[str] = []
    master_path = Path(master_path)
    raw = pd.read_csv(master_path)

    is_legacy_graph = (raw["experiment_kind"] == "graph_ablation") if "experiment_kind" in raw.columns else pd.Series(False, index=raw.index)
    # A genuine embedding_ablation row always has a real ts_model name;
    # a legacy-corrupted graph_ablation row does not (its "ts_model"
    # column actually holds the GNN type, e.g. "graphsage").
    embedding_df = raw[~is_legacy_graph].copy()
    embedding_df = _coerce_numeric(embedding_df)

    if is_legacy_graph.any():
        n_legacy = int(is_legacy_graph.sum())
        notes.append(
            f"Found {n_legacy} legacy-corrupted graph_ablation row(s) embedded in {master_path.name} "
            f"(pre-2026-09-16 ResultsWriter bug) -- recovering them by re-decoding their actual field "
            f"values rather than trusting the file's header for those rows."
        )
        recovered = _recover_legacy_graph_rows(master_path)
        recovered = _coerce_numeric(recovered)
    else:
        recovered = pd.DataFrame()

    graph_df = recovered
    if graph_path:
        graph_path = Path(graph_path)
        if graph_path.exists():
            clean_graph_df = _coerce_numeric(pd.read_csv(graph_path))
            graph_df = pd.concat([graph_df, clean_graph_df], ignore_index=True) if not graph_df.empty else clean_graph_df
            notes.append(f"Loaded {len(clean_graph_df)} graph_ablation row(s) from {graph_path.name}.")
        else:
            notes.append(f"--graph path {graph_path} does not exist -- skipping (only legacy-recovered rows, if any, will be used).")

    if "config.ts_model" in embedding_df.columns:
        bad_ts = ~embedding_df["config.ts_model"].isin(VALID_TS_MODELS)
        if bad_ts.any():
            notes.append(
                f"WARNING: {int(bad_ts.sum())} embedding_ablation row(s) have an unrecognized "
                f"config.ts_model value ({sorted(embedding_df.loc[bad_ts, 'config.ts_model'].unique())}) -- "
                f"excluding them from analysis. Investigate before trusting counts."
            )
            embedding_df = embedding_df[~bad_ts].copy()

    return embedding_df, graph_df, notes


# --------------------------------------------------------------------------
# Data-integrity checks
# --------------------------------------------------------------------------

def check_data_integrity(df: pd.DataFrame, label: str) -> list[str]:
    """Returns a list of finding strings. Empty list = clean."""
    findings = []
    if df.empty:
        return [f"[{label}] no rows to check."]

    if "training.classifier_skipped" in df.columns:
        skipped = df["training.classifier_skipped"].astype(str).str.lower().eq("true")
        if skipped.any():
            findings.append(f"[{label}] {int(skipped.sum())} row(s) have classifier_skipped=True -- these carry no real metrics.")

    for col in ("training.n_hospitals_skipped_no_labels", "training.n_hospitals_skipped_non_finite"):
        if col in df.columns:
            nonzero = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
            if nonzero.any():
                findings.append(f"[{label}] {int(nonzero.sum())} row(s) have nonzero {col} -- investigate which hospitals and why.")

    if "test.precision" in df.columns and "test.recall" in df.columns:
        degenerate = (pd.to_numeric(df["test.precision"], errors="coerce") == 0) & (pd.to_numeric(df["test.recall"], errors="coerce") == 0)
        if degenerate.any():
            findings.append(
                f"[{label}] {int(degenerate.sum())} row(s) have test precision=recall=0 -- the classifier "
                f"never predicts positive on test at its chosen threshold. Check test.threshold_used and "
                f"test.n_positive for these rows."
            )

    if "val.threshold_used" in df.columns and "test.threshold_used" in df.columns:
        v = pd.to_numeric(df["val.threshold_used"], errors="coerce")
        t = pd.to_numeric(df["test.threshold_used"], errors="coerce")
        mismatched = (v != t) & v.notna() & t.notna()
        if mismatched.any():
            findings.append(
                f"[{label}] {int(mismatched.sum())} row(s) have val.threshold_used != test.threshold_used -- "
                f"this should never happen (the val-selected threshold is meant to be reused verbatim for "
                f"test); if it does, the threshold-selection code path needs re-checking."
            )

    if not findings:
        findings.append(f"[{label}] no data-integrity issues found ({len(df)} rows checked).")
    return findings


# --------------------------------------------------------------------------
# Overfitting diagnostics
# --------------------------------------------------------------------------

def overfitting_by_classifier(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "config.classifier" not in df.columns:
        return pd.DataFrame()
    out = df.groupby("config.classifier")[["train.auroc", "val.auroc", "test.auroc"]].mean()
    out["train_minus_test_gap"] = out["train.auroc"] - out["test.auroc"]
    return out.sort_values("train_minus_test_gap", ascending=False)


# --------------------------------------------------------------------------
# Breakdowns
# --------------------------------------------------------------------------

def breakdown_by(df: pd.DataFrame, group_col: str, metrics=("train.auroc", "val.auroc", "test.auroc", "test.auprc")) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    cols = [c for c in metrics if c in df.columns]
    return df.groupby(group_col)[cols].agg(["mean", "std", "count"])


def runtime_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    out = {}
    if "training.encoder_train_runtime_seconds" in df.columns and "config.ts_model" in df.columns:
        # Encoder training time is paid ONCE per (ts_model, tab_model,
        # fusion_mode) combo and repeated across the 6 classifier x
        # task rows that share it -- de-duplicate before summarizing,
        # or per-classifier totals will look 6x too large.
        combo_cols = [c for c in ("config.ts_model", "config.tab_model", "config.fusion_mode") if c in df.columns]
        per_combo = df.drop_duplicates(subset=combo_cols)[combo_cols + ["training.encoder_train_runtime_seconds"]]
        out["encoder_training_minutes_by_ts_model"] = (
            per_combo.groupby("config.ts_model")["training.encoder_train_runtime_seconds"].sum() / 60
        ).round(1).to_dict()
        out["total_encoder_training_minutes"] = round(per_combo["training.encoder_train_runtime_seconds"].sum() / 60, 1)
    if "training.classifier_fit_runtime_seconds" in df.columns and "config.classifier" in df.columns:
        out["mean_classifier_fit_seconds"] = df.groupby("config.classifier")["training.classifier_fit_runtime_seconds"].mean().round(1).to_dict()
    return out


# --------------------------------------------------------------------------
# Best-performing configuration, by metric, computed SEPARATELY
# --------------------------------------------------------------------------

def best_by_metric(df: pd.DataFrame, metric: str, split: str = "test", per_task: bool = True) -> pd.DataFrame:
    """Returns the single best row (or one per task, if per_task=True)
    for `{split}.{metric}`. Caller decides which metrics matter --
    see the caveat printed in main() about accuracy/recall/F1 being
    threshold-dependent and less trustworthy than AUROC/AUPRC under
    class imbalance.
    """
    col = f"{split}.{metric}"
    if df.empty or col not in df.columns:
        return pd.DataFrame()
    id_cols = [c for c in ("experiment_id", "config.ts_model", "config.tab_model", "config.fusion_mode",
                            "config.gnn_type", "config.classifier", "task") if c in df.columns]
    show_cols = id_cols + [c for c in (f"{split}.auroc", f"{split}.auprc", col) if c not in id_cols]
    if per_task and "task" in df.columns:
        return df.loc[df.groupby("task")[col].idxmax()][show_cols]
    return df.nlargest(1, col)[show_cols]


# --------------------------------------------------------------------------
# Flat (embedding_ablation) vs Graph (graph_ablation) comparison
# --------------------------------------------------------------------------

def compare_flat_vs_graph(embedding_df: pd.DataFrame, graph_df: pd.DataFrame) -> "tuple[pd.DataFrame, list[str]]":
    """Compares each classifier's mean flat-grid performance against
    its graph-embedding performance, per task. Returns (table, notes)
    -- notes flags a classifier-hyperparameter mismatch explicitly if
    one is found, since a lower/higher graph number could then be an
    artifact of different regularization, not of the representation
    itself (a real, confirmed issue in this project as of 2026-09-17).
    """
    notes: list[str] = []
    if embedding_df.empty or graph_df.empty:
        notes.append("Flat-vs-graph comparison skipped: one of the two result sets is empty.")
        return pd.DataFrame(), notes

    flat_agg = embedding_df.groupby(["config.classifier", "task"])[["test.auroc", "test.auprc", "test.f1", "test.recall", "test.accuracy"]].mean()
    flat_agg.columns = [f"flat_mean_{c.split('.')[-1]}" for c in flat_agg.columns]
    graph_agg = graph_df.groupby(["config.classifier", "task"])[["test.auroc", "test.auprc", "test.f1", "test.recall", "test.accuracy"]].mean()
    graph_agg.columns = [f"graph_{c.split('.')[-1]}" for c in graph_agg.columns]
    combined = flat_agg.join(graph_agg, how="outer")
    combined["auroc_gap_graph_minus_flat"] = combined["graph_auroc"] - combined["flat_mean_auroc"]

    # Confound check: do the classifiers being compared actually share
    # the same hyperparameters across the two groups?
    if "config.classifier_params" in embedding_df.columns and "config.classifier_params" in graph_df.columns:
        for clf in sorted(set(embedding_df["config.classifier"]) & set(graph_df["config.classifier"])):
            flat_params = set(embedding_df.loc[embedding_df["config.classifier"] == clf, "config.classifier_params"].astype(str).unique())
            graph_params = set(graph_df.loc[graph_df["config.classifier"] == clf, "config.classifier_params"].astype(str).unique())
            if flat_params != graph_params:
                notes.append(
                    f"CONFOUND WARNING: '{clf}' uses DIFFERENT hyperparameters in the flat grid vs. graph_ablation "
                    f"(flat={flat_params}, graph={graph_params}). Any performance difference below may reflect "
                    f"the hyperparameter change, not the representation (flat vs graph) itself. Re-run one side "
                    f"with matching params before drawing a conclusion from this comparison."
                )

    return combined.round(4), notes


# --------------------------------------------------------------------------
# Sklearn-native classification_report, reconstructed from aggregate stats
# --------------------------------------------------------------------------

def _reconstruct_confusion_counts(n: int, n_positive: int, recall: float, specificity: float) -> "tuple[int, int, int, int]":
    """Returns (TP, FN, TN, FP).

    This reconstruction is EXACT, not an approximation: a binary
    confusion matrix has exactly 4 cells, and they're fully determined
    by 4 independent quantities -- n_positive (= TP+FN), n_negative
    (= TN+FP, = n - n_positive), recall (= TP/n_positive), and
    specificity (= TN/n_negative). All four of those are already
    stored per-row in this project's results, so no raw predictions
    are needed to get the real counts back (up to integer rounding,
    which is negligible at the row counts involved here -- thousands
    to tens of thousands of patients per split).
    """
    n_negative = n - n_positive
    tp = int(round(recall * n_positive)) if n_positive > 0 else 0
    tp = min(max(tp, 0), n_positive)
    fn = n_positive - tp
    tn = int(round(specificity * n_negative)) if n_negative > 0 else 0
    tn = min(max(tn, 0), n_negative)
    fp = n_negative - tn
    return tp, fn, tn, fp


def _synthetic_arrays_from_confusion(tp: int, fn: int, tn: int, fp: int) -> "tuple[np.ndarray, np.ndarray]":
    """Builds the SMALLEST y_true/y_pred pair consistent with the given
    confusion-matrix counts -- one row per (actual, predicted) pair,
    not one row per real patient (that would need the same counts
    anyway, so it produces an identical classification_report either
    way, just without needing the original tens-of-thousands of rows)."""
    y_true = np.array([1] * (tp + fn) + [0] * (tn + fp))
    y_pred = np.array([1] * tp + [0] * fn + [1] * fp + [0] * tn)
    return y_true, y_pred


def print_classification_report_for_row(row: pd.Series, split: str = "test") -> None:
    """Prints a REAL sklearn.metrics.classification_report (not a
    hand-rolled imitation of its formatting) for one result row, at
    the given split, reconstructed from that row's own stored (n,
    n_positive, recall, specificity)."""
    n = row.get(f"{split}.n")
    n_positive = row.get(f"{split}.n_positive")
    recall = row.get(f"{split}.recall")
    specificity = row.get(f"{split}.specificity")
    if pd.isna(n) or pd.isna(n_positive) or pd.isna(recall) or pd.isna(specificity):
        print(f"  [{split}] missing stats needed to reconstruct a classification report for this row -- skipping.")
        return

    tp, fn, tn, fp = _reconstruct_confusion_counts(int(n), int(n_positive), float(recall), float(specificity))
    y_true, y_pred = _synthetic_arrays_from_confusion(tp, fn, tn, fp)
    print(classification_report(y_true, y_pred, target_names=["negative (0)", "positive (1)"], digits=3, zero_division=0))

    stored_precision = row.get(f"{split}.precision")
    reconstructed_precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    if pd.notna(stored_precision) and abs(reconstructed_precision - float(stored_precision)) > 0.01:
        print(
            f"  [reconstruction note: derived positive-class precision {reconstructed_precision:.3f} vs "
            f"stored {float(stored_precision):.3f} -- larger than expected rounding drift, worth a second look]"
        )


def top_n_by_auprc(df: pd.DataFrame, task: str, n: int = 10, split: str = "test") -> pd.DataFrame:
    col = f"{split}.auprc"
    if df.empty or col not in df.columns or "task" not in df.columns:
        return pd.DataFrame()
    task_df = df[df["task"] == task]
    return task_df.nlargest(n, col)


def print_classification_reports_section(df: pd.DataFrame, label: str, split: str = "test", top_n: int = 10) -> None:
    if df.empty:
        print(f"  [{label}] no rows available.")
        return
    id_cols = [c for c in ("config.ts_model", "config.tab_model", "config.fusion_mode",
                            "config.gnn_type", "config.classifier") if c in df.columns]
    for task in ("mortality", "sepsis"):
        top = top_n_by_auprc(df, task, n=top_n, split=split)
        if top.empty:
            print(f"\n--- {label}: task={task} -- no rows for this task ---")
            continue
        print(f"\n--- {label}: task={task}, top {len(top)} by {split}.auprc ---")
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            combo_desc = ", ".join(f"{c.split('.')[-1]}={row[c]}" for c in id_cols)
            print(f"\n  #{rank}  {row.get('experiment_id', '?')}  ({combo_desc})")
            print(f"      {split}.auroc={row.get(f'{split}.auroc')}  {split}.auprc={row.get(f'{split}.auprc')}  "
                  f"{split}.threshold_used={row.get(f'{split}.threshold_used')}")
            print_classification_report_for_row(row, split=split)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master", required=True, help="Path to master_results.csv")
    parser.add_argument("--graph", default=None, help="Path to graph_results.csv (optional; new dedicated file from the fixed ResultsWriter)")
    parser.add_argument("--top-n", type=int, default=10, help="How many top-by-AUPRC configs to show classification reports for, per task (default 10)")
    args = parser.parse_args()

    print("=" * 80)
    print("LOADING")
    print("=" * 80)
    embedding_df, graph_df, notes = load_results(args.master, args.graph)
    for n in notes:
        print(f"  NOTE: {n}")
    print(f"\n  embedding_ablation rows: {len(embedding_df)}")
    print(f"  graph_ablation rows:     {len(graph_df)}")

    print("\n" + "=" * 80)
    print("DATA INTEGRITY")
    print("=" * 80)
    for f in check_data_integrity(embedding_df, "embedding_ablation"):
        print(" ", f)
    for f in check_data_integrity(graph_df, "graph_ablation"):
        print(" ", f)

    print("\n" + "=" * 80)
    print("OVERFITTING DIAGNOSTICS (train vs test AUROC, by classifier)")
    print("=" * 80)
    print(overfitting_by_classifier(embedding_df))

    print("\n" + "=" * 80)
    print("BREAKDOWNS")
    print("=" * 80)
    for col in ("task", "config.ts_model", "config.tab_model", "config.fusion_mode", "config.classifier"):
        if col not in embedding_df.columns:
            continue
        print(f"\n--- by {col} ---")
        print(breakdown_by(embedding_df, col))

    print("\n" + "=" * 80)
    print("RUNTIME SUMMARY")
    print("=" * 80)
    rt = runtime_summary(embedding_df)
    for k, v in rt.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print("BEST-PERFORMING CONFIGURATION, BY METRIC (test split, per task)")
    print("=" * 80)
    print("  CAVEAT: accuracy/recall/F1 are computed at a single decision threshold")
    print("  (selected on validation data). Under class imbalance -- mortality ~9%")
    print("  prevalence here -- accuracy in particular can be misleadingly high for a")
    print("  model that barely discriminates at all. AUROC/AUPRC are threshold-free")
    print("  and are the more trustworthy ranking metrics; the others are included")
    print("  because they were explicitly requested, not because they're the best")
    print("  basis for picking a final model.")
    for metric in ("auroc", "auprc", "f1", "recall", "accuracy"):
        print(f"\n--- best by test.{metric} ---")
        print(best_by_metric(embedding_df, metric))

    print("\n" + "=" * 80)
    print("FLAT (embedding_ablation) vs GRAPH (graph_ablation) COMPARISON")
    print("=" * 80)
    comparison, comparison_notes = compare_flat_vs_graph(embedding_df, graph_df)
    for n in comparison_notes:
        print(f"  {n}")
    print(comparison)

    print("\n" + "=" * 80)
    print(f"CLASSIFICATION REPORTS -- top {args.top_n} by test AUPRC, per task")
    print("=" * 80)
    print("  For spotting class-imbalance effects (e.g. high negative-class recall")
    print("  alongside low positive-class recall/precision) worth addressing with")
    print("  sampling techniques (class weighting, oversampling the minority class,")
    print("  undersampling the majority class, etc.). Reconstructed exactly from each")
    print("  row's own stored (n, n_positive, recall, specificity) -- see")
    print("  _reconstruct_confusion_counts()'s docstring for why this is exact, not")
    print("  approximate. Support counts below are per-split (test), not the full cohort.")
    print_classification_reports_section(embedding_df, "embedding_ablation", top_n=args.top_n)
    print_classification_reports_section(graph_df, "graph_ablation", top_n=args.top_n)


if __name__ == "__main__":
    sys.exit(main())