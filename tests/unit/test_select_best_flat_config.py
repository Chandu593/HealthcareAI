"""Tests for src/ablation/graph_comparison_runner.py's
select_best_flat_config_from_sampling_results() -- reuses the SAME
F1-primary/recall-secondary/... selection priority as the sampling
comparison itself, restricted to one task, over sampling_results.csv
rows (flat CSV or nested JSON shape)."""

import unittest

from src.ablation.graph_comparison_runner import select_best_flat_config_from_sampling_results


def _flat_row(ts, tab, fusion, clf, strategy, task, val_f1, val_recall=0.5, experiment_id=None):
    return {
        "experiment_id": experiment_id or f"ts-{ts}_tab-{tab}_fusion-{fusion}_clf-{clf}_task-{task}_samp-{strategy}",
        "task": task,
        "config.ts_model": ts, "config.tab_model": tab, "config.fusion_mode": fusion,
        "config.classifier": clf, "config.sampling_strategy": strategy,
        "val.f1": str(val_f1), "val.recall": str(val_recall), "val.auroc": "0.7",
        "test.f1": str(val_f1), "test.auroc": "0.7",
    }


def _nested_row(ts, tab, fusion, clf, strategy, task, val_f1, val_recall=0.5):
    return {
        "experiment_id": f"ts-{ts}_tab-{tab}_fusion-{fusion}_clf-{clf}_task-{task}_samp-{strategy}",
        "task": task,
        "config": {"ts_model": ts, "tab_model": tab, "fusion_mode": fusion, "classifier": clf, "sampling_strategy": strategy},
        "metrics": {"val": {"f1": val_f1, "recall": val_recall, "auroc": 0.7}, "test": {"f1": val_f1, "auroc": 0.7}},
    }


class TestSelectBestFlatConfig(unittest.TestCase):
    def test_picks_the_row_with_highest_val_f1(self):
        rows = [
            _flat_row("lstm", "mlp", "concatenation", "adaboost", "no_sampling", "mortality", val_f1=0.5),
            _flat_row("gru", "autoencoder", "cross_attention", "lightgbm", "undersample", "mortality", val_f1=0.8),
            _flat_row("rnn", "mlp", "concatenation", "xgboost", "oversample", "mortality", val_f1=0.6),
        ]
        best = select_best_flat_config_from_sampling_results(rows, task="mortality")
        self.assertEqual(best["ts_model"], "gru")
        self.assertEqual(best["tab_model"], "autoencoder")
        self.assertEqual(best["fusion_mode"], "cross_attention")
        self.assertEqual(best["source_classifier"], "lightgbm")
        self.assertEqual(best["source_sampling_strategy"], "undersample")

    def test_restricted_to_the_requested_task_only(self):
        rows = [
            _flat_row("lstm", "mlp", "concatenation", "adaboost", "no_sampling", "sepsis", val_f1=0.99),
            _flat_row("gru", "mlp", "concatenation", "lightgbm", "no_sampling", "mortality", val_f1=0.5),
        ]
        best = select_best_flat_config_from_sampling_results(rows, task="mortality")
        self.assertEqual(best["ts_model"], "gru")  # the higher-F1 sepsis row must never be picked for a mortality request

    def test_works_with_nested_json_shape_too(self):
        rows = [
            _nested_row("lstm", "mlp", "concatenation", "adaboost", "no_sampling", "mortality", val_f1=0.4),
            _nested_row("transformer", "autoencoder", "concatenation", "xgboost", "oversample", "mortality", val_f1=0.9),
        ]
        best = select_best_flat_config_from_sampling_results(rows, task="mortality")
        self.assertEqual(best["ts_model"], "transformer")

    def test_recall_breaks_f1_ties(self):
        rows = [
            _flat_row("lstm", "mlp", "concatenation", "adaboost", "no_sampling", "mortality", val_f1=0.6, val_recall=0.3),
            _flat_row("gru", "mlp", "concatenation", "lightgbm", "undersample", "mortality", val_f1=0.6, val_recall=0.9),
        ]
        best = select_best_flat_config_from_sampling_results(rows, task="mortality")
        self.assertEqual(best["ts_model"], "gru")

    def test_no_usable_rows_raises(self):
        with self.assertRaises(ValueError):
            select_best_flat_config_from_sampling_results([], task="mortality")
        with self.assertRaises(ValueError):
            select_best_flat_config_from_sampling_results(
                [_flat_row("lstm", "mlp", "concatenation", "adaboost", "no_sampling", "sepsis", val_f1=0.9)],
                task="mortality",
            )

    def test_returns_flat_metrics_for_the_eventual_comparison_report(self):
        rows = [_flat_row("lstm", "mlp", "concatenation", "adaboost", "no_sampling", "mortality", val_f1=0.7)]
        best = select_best_flat_config_from_sampling_results(rows, task="mortality")
        self.assertIn("flat_val_metrics", best)
        self.assertIn("flat_test_metrics", best)
        self.assertAlmostEqual(best["flat_val_metrics"]["f1"], 0.7)


if __name__ == "__main__":
    unittest.main()
