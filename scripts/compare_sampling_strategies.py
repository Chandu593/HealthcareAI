"""Compares every class-imbalance sampling strategy actually run
(no_sampling, undersample, oversample, and -- since 2026-10-03 -- any
of adasyn/enn/adasyn_enn that were enabled via
`ablation.sampling.strategies`) against each other, per
(ts_model, tab_model, fusion_mode, classifier, task) group.

Reads directly from `sampling_comparison`'s own per-experiment JSON
records under `{run_dir}/records/*.json` -- NEVER from
`sampling_results.csv`. This is deliberate: `sampling_results.csv` is a
CSV MIRROR of those same records, kept purely for easy loading into
pandas/Excel (see results_writer.py's module docstring); the JSON
records are the source of truth ResultsWriter actually writes
(ExperimentRecord, one complete, self-contained, correctly-labeled
dict per file). A CSV read by its header is one extra step the JSON
records don't need, and this project has already hit one real,
confirmed case (2026-10-01) of that extra step silently misaligning
columns for exactly this experiment kind -- so this script sidesteps
that whole class of risk entirely, not just the one instance found so
far.

Ranking uses the project's OWN established selection rule
(`metric_sort_key()` from `src/classifiers/imbalance_sampling.py`:
F1 -> recall -> precision -> AUROC -> AUPRC -> balanced_accuracy ->
specificity -> Brier, lower-is-better only for Brier), applied here to
TEST metrics -- not validation. This is intentionally different from
`search_sampling_ratio()`'s own internal ratio selection (which is,
correctly, validation-only, never touching test): once a ratio has
already been chosen per strategy and the run is complete, comparing
STRATEGIES against each other on the held-out test split is the
actual question this script answers, and reusing the exact same
tie-break ordering keeps "best ratio within a strategy" and "best
strategy overall" consistent with each other rather than two
different, driftable rules.

Usage:
    python scripts/compare_sampling_strategies.py \\
        --records-dir outputs/ablation/full_run/records

    # Narrow to one combo/classifier/task:
    python scripts/compare_sampling_strategies.py \\
        --records-dir outputs/ablation/full_run/records \\
        --ts-model transformer --tab-model autoencoder --fusion-mode cross_attention \\
        --classifier xgboost --task sepsis

Terminal output only -- no file writes, matching this project's
existing compare_flat_vs_graph.py precedent for this kind of ad hoc,
re-run-whenever-you-like comparison script.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.classifiers.imbalance_sampling import metric_sort_key  # noqa: E402

METRIC_COLUMNS = ["f1", "recall", "precision", "auroc", "auprc", "specificity", "brier_score"]
# Display order only -- ranking itself always uses metric_sort_key(),
# never re-derived here, so this script can never silently drift from
# the project's own established selection rule.
STRATEGY_DISPLAY_ORDER = ["no_sampling", "undersample", "oversample", "adasyn", "enn", "adasyn_enn"]


def load_sampling_records(records_dir: Path) -> "list[dict]":
    records = []
    for path in sorted(records_dir.glob("*.json")):
        with open(path) as f:
            record = json.load(f)
        if record.get("experiment_kind") == "sampling_comparison":
            records.append(record)
    return records


def group_key(record: dict) -> tuple:
    c = record["config"]
    return (c["ts_model"], c["tab_model"], c["fusion_mode"], c["classifier"], record["task"])


def passes_filters(record: dict, args: argparse.Namespace) -> bool:
    c = record["config"]
    if args.ts_model and c["ts_model"] != args.ts_model:
        return False
    if args.tab_model and c["tab_model"] != args.tab_model:
        return False
    if args.fusion_mode and c["fusion_mode"] != args.fusion_mode:
        return False
    if args.classifier and c["classifier"] != args.classifier:
        return False
    if args.task and record["task"] != args.task:
        return False
    return True


def fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records-dir", required=True, help="e.g. outputs/ablation/full_run/records")
    parser.add_argument("--ts-model", default=None)
    parser.add_argument("--tab-model", default=None)
    parser.add_argument("--fusion-mode", default=None, help="concatenation | cross_attention")
    parser.add_argument("--classifier", default=None, help="adaboost | xgboost")
    parser.add_argument("--task", default=None, help="mortality | sepsis")
    args = parser.parse_args()

    records_dir = Path(args.records_dir)
    if not records_dir.is_dir():
        print(f"ERROR: {records_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    all_records = load_sampling_records(records_dir)
    if not all_records:
        print(f"No sampling_comparison records found under {records_dir}.", file=sys.stderr)
        sys.exit(1)
    records = [r for r in all_records if passes_filters(r, args)]
    if not records:
        print("No records match the given filters.", file=sys.stderr)
        sys.exit(1)

    groups: "dict[tuple, dict[str, dict]]" = defaultdict(dict)
    for r in records:
        groups[group_key(r)][r["config"]["sampling_strategy"]] = r

    print(f"Loaded {len(records)} sampling_comparison record(s) across {len(groups)} "
          f"(ts_model, tab_model, fusion_mode, classifier, task) group(s).\n")

    strategy_wins = defaultdict(int)
    strategy_test_metrics = defaultdict(lambda: defaultdict(list))

    for key in sorted(groups):
        ts_model, tab_model, fusion_mode, classifier, task = key
        entries = groups[key]
        print(f"=== {ts_model} / {tab_model} / {fusion_mode} / {classifier} / {task} ===")
        ranked = sorted(entries.items(), key=lambda kv: metric_sort_key(kv[1]["metrics"]["test"]), reverse=True)
        header = f"  {'strategy':<14}" + "".join(f"{m:>12}" for m in METRIC_COLUMNS)
        print(header)
        for strategy, record in ranked:
            test = record["metrics"]["test"]
            row = f"  {strategy:<14}" + "".join(f"{fmt(test.get(m)):>12}" for m in METRIC_COLUMNS)
            print(row)
            for m in METRIC_COLUMNS:
                if isinstance(test.get(m), (int, float)):
                    strategy_test_metrics[strategy][m].append(test[m])
        winner = ranked[0][0]
        strategy_wins[winner] += 1
        print(f"  -> winner (by this project's F1->recall->...->Brier rule): {winner}\n")

    print("=" * 70)
    print(f"Winning strategy (by test metrics) across all {len(groups)} group(s):")
    for strategy in STRATEGY_DISPLAY_ORDER:
        if strategy in strategy_wins or strategy in strategy_test_metrics:
            n_groups_with_strategy = len(strategy_test_metrics.get(strategy, {}).get("f1", []))
            print(f"  {strategy:<14} won {strategy_wins.get(strategy, 0):>3} / {len(groups)} "
                  f"(present in {n_groups_with_strategy} group(s))")

    print()
    print("Mean test metric by strategy, across every group it appears in:")
    header = f"  {'strategy':<14}" + "".join(f"{m:>12}" for m in METRIC_COLUMNS)
    print(header)
    for strategy in STRATEGY_DISPLAY_ORDER:
        if strategy not in strategy_test_metrics:
            continue
        row = f"  {strategy:<14}"
        for m in METRIC_COLUMNS:
            vals = strategy_test_metrics[strategy][m]
            mean_val = sum(vals) / len(vals) if vals else None
            row += f"{fmt(mean_val):>12}"
        print(row)


if __name__ == "__main__":
    main()
