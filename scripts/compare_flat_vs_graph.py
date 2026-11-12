"""
Compares the FLAT no_sampling baseline (from sampling_results.csv)
against Graph A and Graph B (from graph_comparison_results.csv) for
one winning (ts_model, tab_model, fusion_mode) combo -- confirmed with
the user 2026-10-02 as the comparison needed once both the sampling
comparison and the graph comparison have real results.

Terminal-only, no file writes (matching the convention of this
project's other analysis scripts, e.g. check_local_training_quality.py).

Both CSVs share the exact same metric keys for train/val/test
(n, n_positive, threshold_used, accuracy, precision, recall,
sensitivity, f1, specificity, brier_score, auroc, auprc) because both
are written by the SAME compute_binary_classification_metrics()
(src/classifiers/boosting_classifiers.py) -- the flat classifier path
and the GNN+heads path both call it. What differs is only the `config`
columns: sampling_results.csv has config.classifier/sampling_strategy/
sampling_ratio (no graph), graph_comparison_results.csv has
config.graph/method/threshold_used (no classifier) -- this script
reads each file's own config shape rather than assuming the other
applies.

Usage:
    python scripts/compare_flat_vs_graph.py \\
        --sampling-results outputs/ablation/<run_id>/sampling_results.csv \\
        --graph-results outputs/ablation/<run_id>/graph_comparison_results.csv \\
        --combo ts_model=gru,tab_model=autoencoder,fusion_mode=cross_attention

`--combo` is optional if graph-results has exactly one combo_id in it
(the normal case -- graph_comparison always runs for ONE selected
combo at a time); otherwise it's required, since sampling_results.csv
covers all 48 combos and this script needs to know which one to pull
the flat baseline from. `--classifier` narrows the flat baseline to
one classifier's no_sampling row; omitted, every classifier's
no_sampling row for the combo is shown so you can see all of them next
to Graph A/B.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ablation.config_grid import EmbeddingComboSpec  # noqa: E402

METRIC_ORDER = ("n", "n_positive", "threshold_used", "accuracy", "precision", "recall", "specificity", "f1", "auroc", "auprc", "brier_score")
SPLITS = ("train", "val", "test")


def _read_csv(path: str) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _as_float(value) -> "float | None":
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_metrics(row: dict) -> dict:
    out: dict = {}
    for split in SPLITS:
        out[split] = {}
        for metric in METRIC_ORDER:
            out[split][metric] = _as_float(row.get(f"{split}.{metric}"))
    return out


def _parse_combo_arg(combo_arg: str) -> dict:
    parts = dict(p.split("=", 1) for p in combo_arg.split(","))
    missing = [k for k in ("ts_model", "tab_model", "fusion_mode") if k not in parts]
    if missing:
        raise ValueError(f"--combo is missing required field(s): {missing}. Format: ts_model=X,tab_model=Y,fusion_mode=Z")
    return parts


def _detect_combo_id(graph_rows: list) -> str:
    combo_ids = {r.get("config.combo_id") for r in graph_rows if r.get("config.combo_id")}
    if len(combo_ids) == 1:
        return next(iter(combo_ids))
    if not combo_ids:
        raise ValueError("graph-results file has no rows with a config.combo_id column -- is this really a graph_comparison_results.csv?")
    raise ValueError(
        f"graph-results file contains {len(combo_ids)} different combos ({sorted(combo_ids)}) -- pass --combo to pick one."
    )


def _print_metrics_table(title: str, rows_by_label: dict, task: str) -> None:
    print(f"\n=== {title} (task={task}) ===")
    labels = list(rows_by_label.keys())
    header = f"{'metric':<16}" + "".join(f"{label:>18}" for label in labels)
    print(header)
    print("-" * len(header))
    for split in SPLITS:
        print(f"-- {split} --")
        for metric in METRIC_ORDER:
            values = []
            for label in labels:
                m = rows_by_label[label]
                v = m[split].get(metric) if m else None
                values.append("n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) and metric != "n" and metric != "n_positive" else f"{v:.0f}"))
            print(f"{metric:<16}" + "".join(f"{v:>18}" for v in values))


def _reconstruct_classification_report(metrics: dict, split: str = "test") -> "dict | None":
    """Approximate sklearn-style confusion-matrix counts from the
    aggregate metrics stored in the CSV (n, n_positive, recall,
    specificity) -- NOT a real per-sample classification_report, since
    neither sampling_results.csv nor graph_comparison_results.csv store
    per-patient predictions (that layer was never built in this
    engagement). Rounds to the nearest integer count; clearly labeled
    as reconstructed/approximate, never presented as exact."""
    m = metrics[split]
    n, n_pos, recall, specificity = m.get("n"), m.get("n_positive"), m.get("recall"), m.get("specificity")
    if n is None or n_pos is None or recall is None or specificity is None:
        return None
    n_neg = n - n_pos
    tp = round(recall * n_pos)
    fn = round(n_pos - tp)
    tn = round(specificity * n_neg)
    fp = round(n_neg - tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "confusion_matrix_approx": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "class_1_precision_approx": precision, "class_1_recall": recall, "class_1_f1_approx": f1,
        "class_0_recall_aka_specificity": specificity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sampling-results", required=True, help="Path to sampling_results.csv")
    parser.add_argument("--graph-results", required=True, help="Path to graph_comparison_results.csv")
    parser.add_argument("--combo", default=None, help="ts_model=X,tab_model=Y,fusion_mode=Z -- optional if graph-results has exactly one combo")
    parser.add_argument("--classifier", default=None, help="Narrow the flat no_sampling baseline to one classifier (default: show all)")
    parser.add_argument("--task", default=None, help="Restrict to one task (default: every task present in graph-results)")
    args = parser.parse_args()

    sampling_rows = _read_csv(args.sampling_results)
    graph_rows = _read_csv(args.graph_results)

    if args.combo:
        combo_fields = _parse_combo_arg(args.combo)
        combo_id = EmbeddingComboSpec(**combo_fields).combo_id
    else:
        combo_id = _detect_combo_id(graph_rows)
    print(f"Comparing combo_id={combo_id!r}")

    graph_rows_for_combo = [r for r in graph_rows if r.get("config.combo_id") == combo_id]
    if not graph_rows_for_combo:
        print(f"No rows in {args.graph_results} match combo_id={combo_id!r} -- check --combo or the file.")
        return

    sampling_rows_for_combo = [
        r for r in sampling_rows
        if r.get("config.sampling_strategy") == "no_sampling"
        and EmbeddingComboSpec(
            ts_model=r.get("config.ts_model"), tab_model=r.get("config.tab_model"),
            fusion_mode=r.get("config.fusion_mode"),
        ).combo_id == combo_id
    ]
    if args.classifier:
        sampling_rows_for_combo = [r for r in sampling_rows_for_combo if r.get("config.classifier") == args.classifier]

    tasks = [args.task] if args.task else sorted({r.get("task") for r in graph_rows_for_combo if r.get("task")})

    for task in tasks:
        rows_by_label: dict = {}
        for r in sampling_rows_for_combo:
            if r.get("task") != task:
                continue
            label = f"flat/{r.get('config.classifier')}"
            rows_by_label[label] = _row_metrics(r)
        for graph_name in ("graph_a", "graph_b"):
            match = next((r for r in graph_rows_for_combo if r.get("config.graph") == graph_name and r.get("task") == task), None)
            if match:
                rows_by_label[graph_name] = _row_metrics(match)

        if not rows_by_label:
            print(f"\n(no rows found for task={task!r})")
            continue

        _print_metrics_table(f"Flat (no_sampling) vs Graph A vs Graph B -- combo {combo_id}", rows_by_label, task)

        print(f"\n-- reconstructed test-split classification detail (approximate -- see script docstring) --")
        for label, m in rows_by_label.items():
            report = _reconstruct_classification_report(m)
            print(f"  {label}: {report if report else 'not enough data to reconstruct'}")

        graph_stats_printed = False
        for graph_name in ("graph_a", "graph_b"):
            match = next((r for r in graph_rows_for_combo if r.get("config.graph") == graph_name and r.get("task") == task), None)
            if match and match.get("training.graph_stats"):
                if not graph_stats_printed:
                    print("\n-- graph structure --")
                    graph_stats_printed = True
                print(f"  {graph_name}: {match.get('training.graph_stats')}")


if __name__ == "__main__":
    main()
