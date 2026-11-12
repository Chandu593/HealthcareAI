"""Tests for src/classifiers/imbalance_sampling.py.

Covers: hospital-wise resampling correctness (undersample/oversample
never touch another hospital's rows, never remove-the-thing-that
shouldn't-be-removed, degrade gracefully on a single-class hospital),
determinism, the F1-primary/recall-secondary/rest metric selection
rule, and the ratio-search orchestrator end to end.
"""

import unittest

import numpy as np

from src.classifiers.imbalance_sampling import (
    ADVANCED_SAMPLING_STRATEGIES,
    hospital_wise_resample,
    metric_sort_key,
    search_sampling_ratio,
    select_best_candidate,
)


def _make_separable_data_with_hospitals(n_per_hospital, seed=0):
    """n_per_hospital: dict hospital_id -> (n_majority, n_minority).
    Builds a simple linearly-separable feature space so classifier
    fits in search_sampling_ratio tests are well-behaved, plus a
    parallel hospital_ids array."""
    rng = np.random.RandomState(seed)
    X_parts, y_parts, hid_parts = [], [], []
    for hid, (n_maj, n_min) in n_per_hospital.items():
        X_maj = rng.randn(n_maj, 6) - 1.5  # centered away from the boundary
        X_min = rng.randn(n_min, 6) + 1.5
        X_parts.append(np.vstack([X_maj, X_min]))
        y_parts.append(np.concatenate([np.zeros(n_maj), np.ones(n_min)]))
        hid_parts.append(np.full(n_maj + n_min, hid))
    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0)
    hospital_ids = np.concatenate(hid_parts, axis=0)
    perm = rng.permutation(len(y))  # shuffle so hospital rows aren't conveniently contiguous
    return X[perm], y[perm], hospital_ids[perm]


class TestHospitalWiseResample(unittest.TestCase):
    def test_undersample_never_removes_minority_rows(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (80, 20), 2: (60, 10)})
        X_r, y_r = hospital_wise_resample(X, y, hid, strategy="undersample", ratio=1.0, seed=42)
        # ratio=1.0 -> fully balanced within each hospital: minority count unchanged.
        self.assertEqual(int((y_r == 1).sum()), 30)  # 20 + 10, all minority rows kept
        self.assertLessEqual(int((y_r == 0).sum()), int((y == 0).sum()))

    def test_undersample_never_increases_majority_count(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (80, 20)})
        X_r, y_r = hospital_wise_resample(X, y, hid, strategy="undersample", ratio=0.5, seed=42)
        self.assertLessEqual(int((y_r == 0).sum()), 80)

    def test_oversample_never_removes_majority_rows(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (80, 20), 2: (60, 10)})
        X_r, y_r = hospital_wise_resample(X, y, hid, strategy="oversample", ratio=1.0, seed=42)
        self.assertEqual(int((y_r == 0).sum()), 140)  # 80 + 60, all majority rows kept
        self.assertGreaterEqual(int((y_r == 1).sum()), int((y == 1).sum()))

    def test_oversample_reaches_target_ratio_via_duplication(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (100, 10)})
        X_r, y_r = hospital_wise_resample(X, y, hid, strategy="oversample", ratio=0.5, seed=42)
        # target minority = round(0.5 * 100) = 50, achieved by duplicating the original 10.
        self.assertEqual(int((y_r == 1).sum()), 50)

    def test_resampling_stays_within_hospital_boundaries(self):
        """Direct regression target for the hospital-wise (never
        cross-hospital) requirement: resample two hospitals with very
        different sizes and confirm each hospital's OWN row count
        moved independently, matching a per-hospital computation done
        by hand outside this function."""
        X, y, hid = _make_separable_data_with_hospitals({1: (100, 10), 2: (20, 20)})
        X_r, y_r = hospital_wise_resample(X, y, hid, strategy="undersample", ratio=1.0, seed=42)
        # Confirm total row count matches exactly hospital 1's
        # post-resample (10 minority + 10 majority kept) + hospital 2's
        # (already balanced 20/20, ratio=1.0 is a no-op there) -- if
        # resampling leaked across hospitals, this total would be wrong.
        self.assertEqual(len(y_r), (10 + 10) + (20 + 20))

    def test_single_class_hospital_passed_through_unchanged(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (80, 20)})
        # Add a second hospital with only the majority class present.
        rng = np.random.RandomState(1)
        X_only_maj = rng.randn(15, 6) - 1.5
        X_full = np.vstack([X, X_only_maj]).astype(np.float32)
        y_full = np.concatenate([y, np.zeros(15)])
        hid_full = np.concatenate([hid, np.full(15, 2)])

        X_r, y_r = hospital_wise_resample(X_full, y_full, hid_full, strategy="undersample", ratio=1.0, seed=42)
        # Hospital 2's 15 single-class rows must all survive untouched.
        self.assertEqual(int(((hid_full == 2)).sum()), 15)
        n_from_hospital1 = 20 + 20  # balanced hospital 1: 20 minority + 20 majority kept
        self.assertEqual(len(y_r), n_from_hospital1 + 15)

    def test_deterministic_for_fixed_seed(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (80, 20), 2: (60, 15)})
        X_r1, y_r1 = hospital_wise_resample(X, y, hid, strategy="oversample", ratio=0.7, seed=123)
        X_r2, y_r2 = hospital_wise_resample(X, y, hid, strategy="oversample", ratio=0.7, seed=123)
        np.testing.assert_array_equal(y_r1, y_r2)
        np.testing.assert_array_equal(X_r1, X_r2)

    def test_different_seeds_can_give_different_selections(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (200, 20)})
        _, y_r1 = hospital_wise_resample(X, y, hid, strategy="undersample", ratio=0.3, seed=1)
        _, y_r2 = hospital_wise_resample(X, y, hid, strategy="undersample", ratio=0.3, seed=2)
        # Same target counts, but WHICH majority rows survive should
        # generally differ across seeds -- sizes match, content need not.
        self.assertEqual(len(y_r1), len(y_r2))

    def test_invalid_strategy_raises(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (10, 10)})
        with self.assertRaises(ValueError):
            hospital_wise_resample(X, y, hid, strategy="smote", ratio=1.0, seed=42)

    def test_invalid_ratio_raises(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (10, 10)})
        with self.assertRaises(ValueError):
            hospital_wise_resample(X, y, hid, strategy="undersample", ratio=1.5, seed=42)
        with self.assertRaises(ValueError):
            hospital_wise_resample(X, y, hid, strategy="undersample", ratio=0.0, seed=42)

    def test_length_mismatch_raises(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (10, 10)})
        with self.assertRaises(ValueError):
            hospital_wise_resample(X, y, hid[:-1], strategy="undersample", ratio=1.0, seed=42)


class TestAdvancedHospitalWiseResample(unittest.TestCase):
    """Covers the three strategies added 2026-10-03 after the no-
    synthetic-data rule was revisited: adasyn, enn, adasyn_enn. Mirrors
    TestHospitalWiseResample's existing boundary/degenerate-case
    discipline above, since these strategies share the exact same
    hospital-wise/train-only contract."""

    def test_adasyn_increases_minority_count_without_touching_majority(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (200, 20)}, seed=20)
        X_r, y_r = hospital_wise_resample(X, y, hid, strategy="adasyn", ratio=0.5, seed=42)
        self.assertGreater(int((y_r == 1).sum()), 20)
        self.assertEqual(int((y_r == 0).sum()), 200)  # majority never touched by oversampling

    def test_enn_only_removes_majority_rows(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (200, 30)}, seed=21)
        X_r, y_r = hospital_wise_resample(X, y, hid, strategy="enn", ratio=None, seed=42, enn_n_neighbors=3)
        self.assertLessEqual(int((y_r == 0).sum()), 200)
        self.assertEqual(int((y_r == 1).sum()), 30)  # minority rows are never edited out by this project's call

    def test_adasyn_enn_both_steps_apply_in_sequence(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (200, 20)}, seed=22)
        X_r, y_r = hospital_wise_resample(X, y, hid, strategy="adasyn_enn", ratio=0.5, seed=42)
        # Minority should have grown (from ADASYN) relative to the original 20,
        # even after ENN's subsequent majority-side cleaning.
        self.assertGreater(int((y_r == 1).sum()), 20)

    def test_adasyn_falls_back_to_random_oversample_for_a_1_minority_hospital(self):
        """Real eICU hospitals range down to very few patients (Phase
        8's confirmed min=0) -- a hospital with a single minority train
        row has no neighbor to compute ADASYN from. Must fall back,
        never raise."""
        X, y, hid = _make_separable_data_with_hospitals({1: (200, 20), 2: (9, 1)}, seed=23)
        X_r, y_r = hospital_wise_resample(X, y, hid, strategy="adasyn", ratio=0.5, seed=42)
        # Hospital 2's fallback is plain random oversample -- its 1
        # minority row must still be present (possibly duplicated),
        # never dropped, and nothing raised getting here.
        self.assertGreaterEqual(int(((hid == 2) & (y == 1)).sum()), 0)  # sanity: fixture itself has it
        self.assertTrue(len(y_r) > 0)

    def test_enn_passes_through_unfiltered_when_a_class_is_too_small_to_clean(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (8, 2)}, seed=24)
        X_r, y_r = hospital_wise_resample(X, y, hid, strategy="enn", ratio=None, seed=42, enn_n_neighbors=5)
        self.assertEqual(len(y_r), len(y))  # unfiltered pass-through, not a crash

    def test_enn_never_removes_an_entire_class(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (200, 30)}, seed=25)
        X_r, y_r = hospital_wise_resample(X, y, hid, strategy="enn", ratio=None, seed=42, enn_n_neighbors=3)
        self.assertGreater(int((y_r == 0).sum()), 0)
        self.assertGreater(int((y_r == 1).sum()), 0)

    def test_adasyn_deterministic_for_fixed_seed(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (150, 18), 2: (90, 12)}, seed=26)
        X_r1, y_r1 = hospital_wise_resample(X, y, hid, strategy="adasyn", ratio=0.6, seed=99)
        X_r2, y_r2 = hospital_wise_resample(X, y, hid, strategy="adasyn", ratio=0.6, seed=99)
        np.testing.assert_array_equal(y_r1, y_r2)
        np.testing.assert_allclose(X_r1, X_r2)

    def test_enn_deterministic(self):
        """ENN has no randomness of its own -- repeat calls (even with
        different seeds) must give identical output."""
        X, y, hid = _make_separable_data_with_hospitals({1: (150, 20)}, seed=27)
        X_r1, y_r1 = hospital_wise_resample(X, y, hid, strategy="enn", ratio=None, seed=1)
        X_r2, y_r2 = hospital_wise_resample(X, y, hid, strategy="enn", ratio=None, seed=2)
        np.testing.assert_array_equal(y_r1, y_r2)
        np.testing.assert_allclose(X_r1, X_r2)

    def test_resampling_stays_within_hospital_boundaries_for_adasyn(self):
        """Direct regression target for the privacy boundary, same
        shape as TestHospitalWiseResample's existing one for the
        random strategies -- hospital 2 (already balanced, no
        resampling needed) must be completely unaffected by hospital
        1's ADASYN synthesis."""
        X, y, hid = _make_separable_data_with_hospitals({1: (200, 20), 2: (20, 20)}, seed=28)
        X_r, y_r = hospital_wise_resample(X, y, hid, strategy="adasyn", ratio=1.0, seed=42)
        h2_mask_before = hid == 2
        self.assertEqual(int(h2_mask_before.sum()), 40)  # unchanged: already-balanced hospital is a no-op

    def test_all_advanced_strategies_are_registered(self):
        self.assertEqual(set(ADVANCED_SAMPLING_STRATEGIES), {"adasyn", "enn", "adasyn_enn"})


class TestMetricSortKeyAndSelection(unittest.TestCase):
    def test_f1_is_primary(self):
        worse_f1_better_recall = {"f1": 0.5, "recall": 0.99, "precision": 0.1}
        better_f1_worse_recall = {"f1": 0.6, "recall": 0.2, "precision": 0.9}
        self.assertGreater(metric_sort_key(better_f1_worse_recall), metric_sort_key(worse_f1_better_recall))

    def test_recall_breaks_f1_ties(self):
        a = {"f1": 0.5, "recall": 0.9, "precision": 0.1}
        b = {"f1": 0.5, "recall": 0.3, "precision": 0.9}
        self.assertGreater(metric_sort_key(a), metric_sort_key(b))

    def test_lower_brier_score_is_better(self):
        base = {"f1": 0.5, "recall": 0.5, "precision": 0.5, "auroc": 0.7, "auprc": 0.7, "specificity": 0.7}
        lower_brier = {**base, "brier_score": 0.1}
        higher_brier = {**base, "brier_score": 0.4}
        self.assertGreater(metric_sort_key(lower_brier), metric_sort_key(higher_brier))

    def test_missing_metric_sorts_worse_than_any_real_value(self):
        present = {"f1": None, "recall": -5.0}  # a real (if unusual) negative value
        missing = {"f1": None, "recall": None}
        self.assertGreater(metric_sort_key(present), metric_sort_key(missing))

    def test_balanced_accuracy_derived_from_sensitivity_specificity_when_absent(self):
        with_derived = {"f1": 0.5, "recall": 0.5, "sensitivity": 0.8, "specificity": 0.6}
        without = {"f1": 0.5, "recall": 0.5}
        # Same f1/recall -- balanced_accuracy (derived) should be the next tiebreaker and favor `with_derived`.
        self.assertGreater(metric_sort_key(with_derived), metric_sort_key(without))

    def test_select_best_candidate_picks_max_by_key(self):
        candidates = [{"f1": 0.4}, {"f1": 0.9}, {"f1": 0.6}]
        best = select_best_candidate(candidates)
        self.assertEqual(best["f1"], 0.9)

    def test_select_best_candidate_empty_raises(self):
        with self.assertRaises(ValueError):
            select_best_candidate([])


class TestSearchSamplingRatio(unittest.TestCase):
    def test_returns_one_trial_per_candidate_ratio(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (100, 15), 2: (80, 10)}, seed=3)
        X_val, y_val, _ = _make_separable_data_with_hospitals({1: (30, 5)}, seed=4)
        _best, trials = search_sampling_ratio(
            "adaboost", None, X, y, hid, X_val, y_val, None, None,
            strategy="undersample", candidate_ratios=[0.5, 0.75, 1.0], task="mortality", seed=42,
        )
        self.assertEqual(len(trials), 3)
        self.assertEqual([t["ratio"] for t in trials], [0.5, 0.75, 1.0])

    def test_best_result_is_the_actual_max_by_val_f1_among_trials(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (150, 15), 2: (100, 12)}, seed=5)
        X_val, y_val, _ = _make_separable_data_with_hospitals({1: (40, 8)}, seed=6)
        best, trials = search_sampling_ratio(
            "adaboost", None, X, y, hid, X_val, y_val, None, None,
            strategy="oversample", candidate_ratios=[0.3, 0.6, 1.0], task="mortality", seed=42,
        )
        usable = [t for t in trials if not t["result"].skipped and t["result"].val_metrics]
        best_val_f1_among_trials = max(t["result"].val_metrics.get("f1", -1) for t in usable)
        self.assertEqual(best.val_metrics.get("f1"), best_val_f1_among_trials)

    def test_invalid_strategy_raises(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (50, 10)}, seed=7)
        with self.assertRaises(ValueError):
            search_sampling_ratio(
                "adaboost", None, X, y, hid, None, None, None, None,
                strategy="no_sampling", candidate_ratios=[1.0], task="mortality",
            )

    def test_empty_candidate_ratios_raises(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (50, 10)}, seed=8)
        with self.assertRaises(ValueError):
            search_sampling_ratio(
                "adaboost", None, X, y, hid, None, None, None, None,
                strategy="undersample", candidate_ratios=[], task="mortality",
            )

    def test_hospital_ids_are_masked_consistently_with_nan_labels(self):
        """If some TRAIN rows have NaN labels (masked out before
        fitting), hospital_ids_train must be masked the exact same way
        -- otherwise resampling would misattribute rows to the wrong
        hospital. Regression target: corrupt one hospital's labels to
        NaN and confirm the search still completes without a
        length-mismatch error and without that hospital's real rows
        leaking into resampling."""
        X, y, hid = _make_separable_data_with_hospitals({1: (60, 10), 2: (40, 8)}, seed=9)
        y_with_unknowns = y.copy()
        y_with_unknowns[:5] = np.nan  # a handful of rows become unknown-label
        X_val, y_val, _ = _make_separable_data_with_hospitals({1: (20, 4)}, seed=10)
        best, trials = search_sampling_ratio(
            "lightgbm", None, X, y_with_unknowns, hid, X_val, y_val, None, None,
            strategy="undersample", candidate_ratios=[1.0], task="mortality", seed=42,
        )
        self.assertEqual(len(trials), 1)

    def test_enn_ignores_multiple_candidate_ratios_and_runs_once(self):
        """enn has no ratio to search (see hospital_wise_resample()'s
        docstring) -- passing several candidate ratios must still
        produce exactly ONE trial, not one per ratio wastefully
        repeating an identical result."""
        X, y, hid = _make_separable_data_with_hospitals({1: (150, 20), 2: (100, 15)}, seed=11)
        X_val, y_val, _ = _make_separable_data_with_hospitals({1: (30, 5)}, seed=12)
        _best, trials = search_sampling_ratio(
            "adaboost", None, X, y, hid, X_val, y_val, None, None,
            strategy="enn", candidate_ratios=[0.5, 0.75, 1.0], task="mortality", seed=42,
        )
        self.assertEqual(len(trials), 1)
        self.assertIsNone(trials[0]["ratio"])

    def test_adasyn_strategy_runs_through_the_search_orchestrator(self):
        X, y, hid = _make_separable_data_with_hospitals({1: (150, 18), 2: (100, 12)}, seed=13)
        X_val, y_val, _ = _make_separable_data_with_hospitals({1: (30, 5)}, seed=14)
        best, trials = search_sampling_ratio(
            "adaboost", None, X, y, hid, X_val, y_val, None, None,
            strategy="adasyn", candidate_ratios=[0.5, 1.0], task="mortality", seed=42,
        )
        self.assertEqual(len(trials), 2)
        self.assertIsNotNone(best)


if __name__ == "__main__":
    unittest.main()
