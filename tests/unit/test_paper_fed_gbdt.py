"""Federated histogram GBDT: exactness of the federated protocol, train-only fitting, missing values, early stopping, selection."""

import unittest

import numpy as np

from src.paper.fed_gbdt import (FIXED_DEFAULTS, bin_features, client_histograms, federated_bin_edges, merge_quantile_summaries,
                                local_quantile_summary, search_and_train, train_fed_gbdt, _auroc)


def make_data(n=6000, f=12, n_hosp=15, seed=0, miss=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype("float32")
    logit = 1.3 * X[:, 0] - 1.0 * X[:, 1] * X[:, 2] + 0.9 * np.sin(2 * X[:, 3]) + 0.8 * (X[:, 4] > 0.4) - 1.8
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("float32")
    X[rng.random((n, f)) < miss] = np.nan
    hosp = rng.integers(0, n_hosp, n)
    idx = rng.permutation(n)
    tr, va, te = idx[: int(.65 * n)], idx[int(.65 * n): int(.82 * n)], idx[int(.82 * n):]
    return X, y, hosp, tr, va, te


def clients(tr, hosp, k=None):
    if k is None:
        return {int(h): tr[hosp[tr] == h] for h in np.unique(hosp[tr])}
    parts = np.array_split(tr, k)                           # an arbitrary re-partition of the SAME train rows
    return {i: p for i, p in enumerate(parts)}


FAST = dict(n_bins=32, max_rounds=25, early_stopping_rounds=None, log_every=0)


class TestFedGBDT(unittest.TestCase):
    def test_client_histograms_sum_to_the_pooled_histogram(self):
        rng = np.random.default_rng(3)
        n, K, nodes, B1 = 500, 7, 4, 9
        bins = rng.integers(0, B1, n).astype(np.uint8); g = rng.normal(size=n); h = rng.random(n)
        cid = rng.integers(0, K, n); node = rng.integers(0, nodes, n)
        cg, ch = client_histograms(bins, g, h, cid, node, K, nodes, B1)
        idx = node * B1 + bins.astype(int)
        pg = np.bincount(idx, weights=g, minlength=nodes * B1).reshape(nodes, B1)
        ph = np.bincount(idx, weights=h, minlength=nodes * B1).reshape(nodes, B1)
        np.testing.assert_allclose(cg.sum(0), pg, atol=1e-10); np.testing.assert_allclose(ch.sum(0), ph, atol=1e-10)
        self.assertEqual(cg.shape, (K, nodes, B1))                                   # one slab per client

    def test_explicit_per_client_protocol_matches_the_aggregate_path(self):
        X, y, hosp, tr, va, te = make_data(2500, 8, 9)
        cr = clients(tr, hosp)
        m1, _ = train_fed_gbdt(X, y, cr, va, {}, 0, **FAST)
        m2, _ = train_fed_gbdt(X, y, cr, va, {}, 0, explicit_client_histograms=True, **FAST)
        np.testing.assert_allclose(m1.predict_raw(X[te]), m2.predict_raw(X[te]), atol=1e-8)

    def test_partition_invariance_same_trees_for_any_client_split(self):
        """The federated model must equal the pooled model: re-partitioning the same train rows across 1, 5 or 60 clients (same bin edges)
        may not change a single prediction. This is what makes 'federated' mean 'no pooled rows' without costing accuracy."""
        X, y, hosp, tr, va, te = make_data(3000, 10, 12)
        edges = federated_bin_edges(X, clients(tr, hosp), 32)
        preds = []
        for k in (1, 5, 60):
            m, _ = train_fed_gbdt(X, y, clients(tr, hosp, k), va, {}, 0, edges=edges, **FAST)
            preds.append(m.predict_raw(X[te]))
        np.testing.assert_allclose(preds[0], preds[1], atol=1e-8); np.testing.assert_allclose(preds[0], preds[2], atol=1e-8)

    def test_validation_and_test_rows_never_influence_training(self):
        """Edges, base score, splits and leaf values come from TRAIN rows only. Scrambling the val/test features AND the val labels may change
        only WHICH round is kept (round selection is what validation is for) -- never the content of any tree that was grown."""
        X, y, hosp, tr, va, te = make_data(3000, 10, 12)
        X2, y2 = X.copy(), y.copy()
        other = np.concatenate([va, te]); rng = np.random.default_rng(9)
        X2[other] = rng.normal(loc=50.0, scale=30.0, size=X2[other].shape); y2[va] = 1 - y2[va]
        cr = clients(tr, hosp)
        a, ha = train_fed_gbdt(X, y, cr, va, {}, 0, **FAST)
        b, hb = train_fed_gbdt(X2, y2, cr, va, {}, 0, **FAST)
        self.assertNotEqual(ha["best_iteration"], hb["best_iteration"])               # the premise: validation really did change the kept round
        for ta, tb in zip(a.trees, b.trees):                                          # ... but every tree both models share is identical
            np.testing.assert_array_equal(ta.feature, tb.feature); np.testing.assert_array_equal(ta.thr, tb.thr)
            np.testing.assert_array_equal(ta.left, tb.left); np.testing.assert_allclose(ta.value, tb.value, atol=1e-12)
        self.assertGreaterEqual(min(len(a.trees), len(b.trees)), 1)
        for ea, eb in zip(a.edges, b.edges):
            np.testing.assert_array_equal(ea, eb)
        self.assertEqual(a.base_score, b.base_score)

    def test_merged_quantile_edges_track_the_pooled_quantiles(self):
        rng = np.random.default_rng(5)
        X = np.concatenate([rng.normal(m, 1.0, size=(400, 3)) for m in rng.uniform(-1, 1, 20)]).astype("float32")      # 20 shifted clients
        cr = {i: np.arange(i * 400, (i + 1) * 400) for i in range(20)}
        edges = federated_bin_edges(X, cr, 16, 128)
        for j in range(3):
            e = edges[j]; frac = np.array([(X[:, j] <= v).mean() for v in e])
            expected = np.arange(1, len(e) + 1) / 16.0
            self.assertLess(np.max(np.abs(frac - expected[: len(frac)])), 0.03)

    def test_binary_constant_and_all_missing_features_are_safe(self):
        X = np.zeros((200, 3), dtype="float32"); X[:, 0] = (np.arange(200) % 2); X[:, 1] = 7.0; X[:, 2] = np.nan
        e = federated_bin_edges(X, {0: np.arange(100), 1: np.arange(100, 200)}, 8)
        b = bin_features(X, e, 8)
        self.assertTrue(np.all(b[:, 2] == 8))                                        # all-missing -> dedicated missing bin
        self.assertEqual(len(np.unique(b[:, 1])), 1); self.assertEqual(len(np.unique(b[:, 0])), 2)

    def test_learns_nonlinear_signal_close_to_sklearn_histgbdt_and_beats_logreg(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        X, y, hosp, tr, va, te = make_data(9000, 14, 20, seed=2)
        m, h = train_fed_gbdt(X, y, clients(tr, hosp), va, {}, 0, n_bins=32, max_rounds=200, early_stopping_rounds=25, log_every=0)
        a_fed = _auroc(y[te], m.predict_raw(X[te]))
        hgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=200, early_stopping=False, random_state=0).fit(X[tr], y[tr])
        a_hgb = _auroc(y[te], hgb.predict_proba(X[te])[:, 1])
        lr = LogisticRegression(max_iter=500).fit(np.nan_to_num(X[tr]), y[tr])
        a_lr = _auroc(y[te], lr.predict_proba(np.nan_to_num(X[te]))[:, 1])
        self.assertGreater(a_fed, a_lr + 0.02); self.assertGreater(a_fed, a_hgb - 0.02)
        self.assertEqual(h["best_iteration"], len(m.trees))                           # model truncated to the best round

    def test_informative_missingness_is_learned_through_the_default_direction(self):
        rng = np.random.default_rng(11)
        n = 6000; y = (rng.random(n) < 0.3).astype("float32")
        X = rng.normal(size=(n, 4)).astype("float32")                                # features carry NO signal
        X[:, 0] = np.where(rng.random(n) < np.where(y > 0, 0.85, 0.10), np.nan, X[:, 0])   # but missingness of feature 0 does
        idx = rng.permutation(n); tr, va, te = idx[:4000], idx[4000:5000], idx[5000:]
        hosp = rng.integers(0, 10, n)
        m, _ = train_fed_gbdt(X, y, clients(tr, hosp), va, {}, 0, n_bins=16, max_rounds=60, early_stopping_rounds=15, log_every=0)
        self.assertGreater(_auroc(y[te], m.predict_raw(X[te])), 0.80)

    def test_default_direction_sends_missing_with_the_class_it_resembles(self):
        """Missing values behave like HIGH values here (positives = x > 0 OR missing). Depth-1 stumps can only be perfect if the learned default
        direction sends missing RIGHT; a splitter that always sends missing left cannot separate this with stumps."""
        rng = np.random.default_rng(13)
        n = 6000; x = rng.normal(size=n).astype("float32"); miss = rng.random(n) < 0.3
        y = ((x > 0) | miss).astype("float32"); X = np.where(miss, np.nan, x).astype("float32")[:, None]
        idx = rng.permutation(n); tr, va, te = idx[:4000], idx[4000:5000], idx[5000:]
        hosp = rng.integers(0, 8, n)
        m, _ = train_fed_gbdt(X, y, clients(tr, hosp), va, {"max_depth": 1, "learning_rate": 0.5, "min_child_weight": 0.0}, 0, n_bins=32, max_rounds=8,
                              early_stopping_rounds=None, log_every=0)
        self.assertGreater(_auroc(y[te], m.predict_raw(X[te])), 0.97)
        self.assertTrue(any(bool((t.feature >= 0).any() and not t.miss_left[0]) for t in m.trees))     # at least one stump routes missing right

    def test_early_stopping_picks_best_validation_round(self):
        X, y, hosp, tr, va, te = make_data(4000, 10, 10, seed=4)
        m, h = train_fed_gbdt(X, y, clients(tr, hosp), va, {"learning_rate": 0.3}, 0, n_bins=32, max_rounds=300, early_stopping_rounds=10, log_every=0)
        self.assertLess(h["n_rounds_run"], 300); self.assertLessEqual(h["best_iteration"], h["n_rounds_run"])
        self.assertAlmostEqual(_auroc(y[va], m.predict_raw(X[va])), h["best_val_auroc"], places=9)

    def test_deterministic_for_a_seed_and_colsample_depends_on_it(self):
        X, y, hosp, tr, va, te = make_data(2500, 10, 8)
        cr = clients(tr, hosp); p = {"colsample_bytree": 0.5}
        a, _ = train_fed_gbdt(X, y, cr, va, p, 1, **FAST); b, _ = train_fed_gbdt(X, y, cr, va, p, 1, **FAST); c, _ = train_fed_gbdt(X, y, cr, va, p, 2, **FAST)
        np.testing.assert_array_equal(a.predict_raw(X[te]), b.predict_raw(X[te]))
        self.assertFalse(np.allclose(a.predict_raw(X[te]), c.predict_raw(X[te])))

    def test_communication_accounting_is_reported_and_sparse_is_below_dense(self):
        X, y, hosp, tr, va, te = make_data(2500, 10, 8)
        _, h = train_fed_gbdt(X, y, clients(tr, hosp), va, {}, 0, **FAST)
        self.assertGreater(h["comm_mb_per_client_per_round_sparse"], 0)
        self.assertLessEqual(h["comm_mb_per_client_per_round_sparse"], h["comm_mb_per_client_per_round_dense"])

    def test_guards(self):
        X, y, hosp, tr, va, te = make_data(800, 5, 4)
        with self.assertRaises(ValueError):
            train_fed_gbdt(X, y, clients(tr, hosp), va, {}, 0, n_bins=300, **{k: v for k, v in FAST.items() if k != "n_bins"})
        with self.assertRaises(ValueError):
            train_fed_gbdt(X, np.zeros_like(y), clients(tr, hosp), va, {}, 0, **FAST)

    def test_search_selects_on_validation_and_reuses_winner(self):
        X, y, hosp, tr, va, te = make_data(3000, 10, 10, seed=6)
        cfg = {"n_bins": 32, "max_rounds": 30, "early_stopping_rounds": 8, "n_trials": 3,
               "fixed": dict(FIXED_DEFAULTS), "search": {"max_depth": [2, 3], "learning_rate": [0.05, 0.2]}}
        m, h = search_and_train(X, y, clients(tr, hosp), va, cfg, 0)
        self.assertEqual(len(h["trials"]), 3)
        self.assertAlmostEqual(h["best_val_auroc"], max(t["val_auroc"] for t in h["trials"]), places=12)
        win = {k: h["params"][k] for k in FIXED_DEFAULTS}
        m2, h2 = search_and_train(X, y, clients(tr, hosp), va, cfg, 1, win)
        self.assertEqual(len(h2["trials"]), 1)                                       # later seeds do not re-search


if __name__ == "__main__":
    unittest.main()
