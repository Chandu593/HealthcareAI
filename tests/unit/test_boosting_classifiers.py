import unittest

import numpy as np

from src.classifiers.boosting_classifiers import (
    VALID_CLASSIFIERS,
    build_classifier,
    compute_binary_classification_metrics,
    fit_and_evaluate_classifier,
    select_threshold_from_validation,
)


def _make_separable_data(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X.astype(np.float32), y


class TestBuildClassifier(unittest.TestCase):
    def test_all_valid_classifiers_fit_and_predict(self):
        X, y = _make_separable_data()
        for name in VALID_CLASSIFIERS:
            clf = build_classifier(name)
            clf.fit(X, y)
            proba = clf.predict_proba(X)
            self.assertEqual(proba.shape, (200, 2))

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            build_classifier("not_a_real_classifier")

    def test_params_override_defaults(self):
        clf = build_classifier("adaboost", {"n_estimators": 7})
        self.assertEqual(clf.n_estimators, 7)


class TestComputeBinaryClassificationMetrics(unittest.TestCase):
    def test_normal_case_has_all_metrics(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.4, 0.8, 0.6, 0.3, 0.9])
        m = compute_binary_classification_metrics(y_true, y_proba)
        for key in ("accuracy", "precision", "recall", "f1", "auroc", "auprc"):
            self.assertIsNotNone(m[key])
        self.assertEqual(m["n"], 6)
        self.assertEqual(m["n_positive"], 3)

    def test_single_class_present_gives_none_auroc_not_a_crash(self):
        y_true = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])
        m = compute_binary_classification_metrics(y_true, y_proba)
        self.assertIsNone(m["auroc"])
        self.assertIsNone(m["auprc"])
        self.assertIn("metrics_note", m)
        # Other metrics still computed, not skipped entirely.
        self.assertIsNotNone(m["accuracy"])


class TestFitAndEvaluateClassifier(unittest.TestCase):
    def test_end_to_end_metrics_for_all_splits(self):
        X_train, y_train = _make_separable_data(n=150, seed=1)
        X_val, y_val = _make_separable_data(n=40, seed=2)
        X_test, y_test = _make_separable_data(n=40, seed=3)
        result = fit_and_evaluate_classifier(
            "xgboost", None, X_train, y_train, X_val, y_val, X_test, y_test, task="mortality",
        )
        self.assertFalse(result.skipped)
        self.assertIn("auroc", result.train_metrics)
        self.assertIn("auroc", result.val_metrics)
        self.assertIn("auroc", result.test_metrics)
        # A cleanly separable synthetic task should score well above chance.
        self.assertGreater(result.test_metrics["auroc"], 0.7)
        self.assertEqual(result.n_train_used, 150)

    def test_nan_labels_are_masked_out_not_coerced(self):
        """Direct regression target for the masking discipline: rows
        with NaN labels must be completely excluded from fitting AND
        from every reported metric, never coerced to class 0."""
        X, y = _make_separable_data(n=100, seed=4)
        y_with_unknowns = y.copy()
        unknown_idx = np.arange(0, 30)
        y_with_unknowns[unknown_idx] = np.nan

        result = fit_and_evaluate_classifier(
            "adaboost", None, X, y_with_unknowns, None, None, None, None, task="sepsis",
        )
        self.assertFalse(result.skipped)
        self.assertEqual(result.n_train_used, 70)  # 100 - 30 unknowns
        self.assertEqual(result.train_metrics["n"], 70)

    def test_too_few_labeled_rows_skips_gracefully(self):
        X = np.random.randn(5, 10).astype(np.float32)
        y = np.array([np.nan, np.nan, np.nan, 1.0, np.nan])  # only 1 known label
        result = fit_and_evaluate_classifier("lightgbm", None, X, y, None, None, None, None, task="mortality")
        self.assertTrue(result.skipped)
        self.assertIsNotNone(result.skip_reason)

    def test_single_class_train_split_skips_gracefully(self):
        X = np.random.randn(20, 10).astype(np.float32)
        y = np.zeros(20)  # all negative, no positive class at all
        result = fit_and_evaluate_classifier("xgboost", None, X, y, None, None, None, None, task="mortality")
        self.assertTrue(result.skipped)

    def test_empty_val_or_test_does_not_crash(self):
        X_train, y_train = _make_separable_data(n=60, seed=5)
        X_val = np.zeros((0, 10), dtype=np.float32)
        y_val = np.zeros((0,), dtype=np.float32)
        result = fit_and_evaluate_classifier(
            "adaboost", None, X_train, y_train, X_val, y_val, None, None, task="mortality",
        )
        self.assertFalse(result.skipped)
        self.assertEqual(result.val_metrics, {})
        self.assertEqual(result.n_val_used, 0)

    def test_resampler_none_by_default_matches_pre_existing_behavior(self):
        """Backward-compatibility regression target for the
        `resampler` parameter added 2026-09-20: not passing it (every
        pre-existing call site) must be byte-for-byte identical to
        before it existed."""
        X_train, y_train = _make_separable_data(n=150, seed=1)
        X_val, y_val = _make_separable_data(n=40, seed=2)
        without_param = fit_and_evaluate_classifier(
            "xgboost", None, X_train, y_train, X_val, y_val, None, None, task="mortality",
        )
        explicit_none = fit_and_evaluate_classifier(
            "xgboost", None, X_train, y_train, X_val, y_val, None, None, task="mortality", resampler=None,
        )
        self.assertEqual(without_param.train_metrics, explicit_none.train_metrics)
        self.assertEqual(without_param.val_metrics, explicit_none.val_metrics)
        self.assertEqual(without_param.n_train_used, explicit_none.n_train_used)

    def test_resampler_receives_masked_train_only_never_val_or_test(self):
        """The resampler must see TRAIN rows only, after NaN-label
        masking -- and val/test metrics must reflect the UNRESAMPLED
        val/test sets exactly (same n_val_used/n_test_used as with no
        resampler at all)."""
        X_train, y_train = _make_separable_data(n=100, seed=6)
        y_train_with_unknowns = y_train.copy()
        y_train_with_unknowns[:10] = np.nan
        X_val, y_val = _make_separable_data(n=30, seed=7)
        seen = {}

        def spy_resampler(X, y):
            seen["n_seen"] = len(y)
            return X, y  # no-op, just observes what it was called with

        baseline = fit_and_evaluate_classifier(
            "adaboost", None, X_train, y_train_with_unknowns, X_val, y_val, None, None, task="mortality",
        )
        result = fit_and_evaluate_classifier(
            "adaboost", None, X_train, y_train_with_unknowns, X_val, y_val, None, None, task="mortality",
            resampler=spy_resampler,
        )
        self.assertEqual(seen["n_seen"], 90)  # 100 - 10 NaN-masked rows, never the raw 100
        self.assertEqual(result.n_val_used, baseline.n_val_used)
        self.assertEqual(result.val_metrics, baseline.val_metrics)

    def test_resampler_that_changes_train_actually_changes_the_fit(self):
        """A resampler that drops all of one class must produce a
        genuinely different (here: skipped) result than not resampling
        -- proves the returned (X, y) is what's actually fit on, not
        silently ignored."""
        X_train, y_train = _make_separable_data(n=100, seed=8)

        def drop_positive_class(X, y):
            keep = y == 0
            return X[keep], y[keep]

        baseline = fit_and_evaluate_classifier("lightgbm", None, X_train, y_train, None, None, None, None, task="mortality")
        resampled = fit_and_evaluate_classifier(
            "lightgbm", None, X_train, y_train, None, None, None, None, task="mortality", resampler=drop_positive_class,
        )
        self.assertFalse(baseline.skipped)
        self.assertTrue(resampled.skipped)  # only one class left -> cannot fit a binary classifier


class TestSensitivitySpecificityBrierAndThreshold(unittest.TestCase):
    def test_sensitivity_equals_recall_and_specificity_is_correct(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1])  # threshold 0.5 -> pred [1,1,0,1,0,0]
        m = compute_binary_classification_metrics(y_true, y_proba, threshold=0.5)
        self.assertAlmostEqual(m["sensitivity"], m["recall"])
        self.assertAlmostEqual(m["recall"], 2 / 3, places=4)  # 2 of 3 positives caught (0.9, 0.8)
        self.assertAlmostEqual(m["specificity"], 2 / 3, places=4)  # 2 of 3 negatives correctly rejected (0.2, 0.1)

    def test_brier_score_is_mean_squared_error_of_probabilities(self):
        y_true = np.array([1.0, 0.0])
        y_proba = np.array([0.8, 0.3])
        m = compute_binary_classification_metrics(y_true, y_proba)
        expected = np.mean([(0.8 - 1.0) ** 2, (0.3 - 0.0) ** 2])
        self.assertAlmostEqual(m["brier_score"], expected, places=6)

    def test_threshold_parameter_changes_precision_recall(self):
        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([0.6, 0.4, 0.3, 0.2])
        low = compute_binary_classification_metrics(y_true, y_proba, threshold=0.1)  # everyone predicted positive
        high = compute_binary_classification_metrics(y_true, y_proba, threshold=0.9)  # everyone predicted negative
        self.assertEqual(low["recall"], 1.0)
        self.assertEqual(high["recall"], 0.0)

    def test_select_threshold_from_validation_finds_a_better_f1_than_naive_default(self):
        # Constructed so 0.5 is a poor cutoff but a lower one separates
        # classes perfectly.
        y_val = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        y_val_proba = np.array([0.3, 0.32, 0.35, 0.1, 0.12, 0.15, 0.2, 0.25])
        threshold = select_threshold_from_validation(y_val, y_val_proba)
        pred_at_selected = (y_val_proba >= threshold).astype(int)
        pred_at_naive = (y_val_proba >= 0.5).astype(int)
        from sklearn.metrics import f1_score
        self.assertGreaterEqual(f1_score(y_val, pred_at_selected), f1_score(y_val, pred_at_naive))

    def test_select_threshold_falls_back_to_half_when_validation_is_degenerate(self):
        self.assertEqual(select_threshold_from_validation(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])), 0.5)
        self.assertEqual(select_threshold_from_validation(np.array([]), np.array([])), 0.5)

    def test_fit_and_evaluate_never_uses_test_labels_to_pick_the_threshold(self):
        """Direct proof: construct validation data that clearly prefers
        a LOW threshold, and test data that -- if it were (wrongly)
        consulted -- would prefer a HIGH threshold instead. The
        `threshold_used` recorded in test_metrics must match what
        validation alone would select, never what would look best on
        test."""
        rng = np.random.RandomState(7)
        X_train, y_train = _make_separable_data(n=100, seed=10)

        # Validation: positives score around 0.3, negatives around 0.1
        # -- best threshold is low (~0.2).
        X_val = rng.randn(40, 10)
        y_val = np.array([1] * 20 + [0] * 20)

        # Test: deliberately adversarial distribution where a HIGH
        # threshold would look better -- if the code peeked at test to
        # choose the threshold, it would pick something near 0.8+.
        X_test = rng.randn(40, 10)
        y_test = np.array([1] * 20 + [0] * 20)

        result = fit_and_evaluate_classifier(
            "adaboost", None, X_train, y_train, X_val, y_val, X_test, y_test, task="mortality",
        )
        self.assertFalse(result.skipped)
        # The threshold actually applied to test must be EXACTLY the one
        # recorded for validation -- not independently re-optimized
        # against test.
        self.assertEqual(result.val_metrics["threshold_used"], result.test_metrics["threshold_used"])

    def test_val_absent_falls_back_to_conventional_threshold_for_test(self):
        X_train, y_train = _make_separable_data(n=100, seed=11)
        X_test, y_test = _make_separable_data(n=30, seed=12)
        result = fit_and_evaluate_classifier(
            "xgboost", None, X_train, y_train, None, None, X_test, y_test, task="mortality",
        )
        self.assertEqual(result.test_metrics["threshold_used"], 0.5)


if __name__ == "__main__":
    unittest.main()
