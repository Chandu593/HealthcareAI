import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.paper.config import get_paper_cfg  # noqa: E402
from src.paper.evaluation import (binary_report, bootstrap_ci, ece, fast_auroc, paired_bootstrap_delta, per_hospital_summary,  # noqa: E402
                                  platt_fit, sigmoid, threshold_max_f1)
from src.paper.preprocess import prepare_sequence, prepare_tabular  # noqa: E402

NC = get_paper_cfg({})["neural"]


class TestPreprocess(unittest.TestCase):
    def _data(self, seed=0):
        rng = np.random.default_rng(seed); n = 400
        hosp = np.repeat([1, 2], n // 2)
        X = np.column_stack([rng.normal(10, 3, n), rng.normal(100, 20, n), (rng.random(n) > 0.5).astype(float), rng.normal(0, 1, n)]).astype(np.float32)
        X[rng.random(n) < 0.4, 1] = np.nan
        train = rng.random(n) < 0.7
        return X, hosp, train

    def test_statistics_use_train_rows_only(self):
        X, hosp, train = self._data()
        a, _ = prepare_tabular(X, hosp, train, NC)
        X2 = X.copy(); X2[~train] = X2[~train] * 1000 + 5e5     # corrupt every non-train row
        b, _ = prepare_tabular(X2, hosp, train, NC)
        self.assertTrue(np.array_equal(a[train], b[train]), "train-row outputs changed when only non-train rows changed")

    def test_per_hospital_standardization_and_clipping(self):
        X, hosp, train = self._data()
        X[hosp == 2, 0] += 50                       # hospital 2 has a shifted distribution
        Xn, info = prepare_tabular(X, hosp, train, dict(NC, min_hospital_n_for_local_stats=10))
        self.assertFalse(np.isnan(Xn).any())
        self.assertLessEqual(np.abs(Xn).max(), NC["clip"] + 1e-6)
        for h in (1, 2):
            m = (hosp == h) & train
            self.assertAlmostEqual(float(Xn[m, 0].mean()), 0.0, places=1)     # each hospital centred on ITS OWN train rows
        self.assertGreater(info["n_missing_indicators"], 0)                   # 40%-missing column gets an indicator

    def test_small_hospital_fallback_uses_train_only_global_stats(self):
        rng = np.random.default_rng(5); n = 300
        hosp = np.concatenate([np.repeat([1, 2], 120), np.repeat([3], 60)])
        X = rng.normal(5, 2, size=(n, 3)).astype(np.float32)
        train = np.zeros(n, dtype=bool); train[:90] = True; train[120:210] = True; train[240:246] = True   # hospital 3: only 6 train rows (< min_n)
        a, _ = prepare_tabular(X, hosp, train, NC)
        X2 = X.copy(); X2[~train] = X2[~train] * 500 + 1e6
        b, _ = prepare_tabular(X2, hosp, train, NC)
        self.assertTrue(np.array_equal(a[train], b[train]), "fallback (global) statistics were influenced by non-train rows")

    def test_sequence_shape_and_mask(self):
        rng = np.random.default_rng(1); seq = rng.normal(size=(60, 24, 3)).astype(np.float32); seq[rng.random(seq.shape) < 0.7] = np.nan
        out = prepare_sequence(seq, np.repeat([1, 2], 30), rng.random(60) < 0.7, dict(NC, min_hospital_n_for_local_stats=5))
        self.assertEqual(out.shape, (60, 24, 6)); self.assertFalse(np.isnan(out).any())
        self.assertTrue(np.array_equal(out[:, :, 3:], (~np.isnan(seq)).astype(np.float32)))


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3); self.y = (rng.random(3000) < 0.12).astype(int)
        self.p = np.clip(0.1 + 0.25 * self.y + rng.normal(0, 0.15, 3000), 0.001, 0.999)

    def test_metrics_match_sklearn(self):
        self.assertAlmostEqual(fast_auroc(self.y, self.p), roc_auc_score(self.y, self.p), places=10)
        r = binary_report(self.y, self.p, 0.3)
        self.assertAlmostEqual(r["auprc"], average_precision_score(self.y, self.p), places=10)
        self.assertEqual(r["tp"] + r["fn"], int(self.y.sum()))

    def test_ties_handled(self):
        y = np.array([0, 1, 0, 1]); p = np.array([0.5, 0.5, 0.5, 0.5]); self.assertAlmostEqual(fast_auroc(y, p), 0.5)

    def test_single_threshold_rule_and_platt_monotone(self):
        a, b = platt_fit(np.log(self.p / (1 - self.p)), self.y)
        self.assertGreater(a, 0)
        thr = threshold_max_f1(self.y, self.p)
        self.assertTrue(0 < thr < 1)
        self.assertLess(ece(self.y, sigmoid(a * np.log(self.p / (1 - self.p)) + b)), ece(self.y, self.p) + 1e-9)

    def test_bootstrap_ci_and_paired_delta(self):
        ci = bootstrap_ci(self.y, self.p, "auroc", n=200, seed=1)
        self.assertLessEqual(ci["lo"], ci["point"]); self.assertGreaterEqual(ci["hi"], ci["point"])
        rng = np.random.default_rng(9); worse = np.clip(self.p + rng.normal(0, 0.4, len(self.p)), 0, 1)
        d = paired_bootstrap_delta(self.y, self.p, worse, "auroc", n=200, seed=1)
        self.assertGreater(d["delta"], 0); self.assertLess(d["p_value"], 0.05)
        same = paired_bootstrap_delta(self.y, self.p, self.p, "auroc", n=50, seed=1)
        self.assertEqual(same["delta"], 0.0)

    def test_per_hospital_skips_tiny_hospitals(self):
        h = np.repeat([1, 2, 3], 1000); h[:3] = 9
        s = per_hospital_summary(self.y, self.p, h, min_pos=5, min_neg=5)
        self.assertEqual(s["n_hospitals_evaluated"], 3)


if __name__ == "__main__":
    unittest.main()
