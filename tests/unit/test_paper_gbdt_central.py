"""Pooled run of the federated GBDT code (`gbdt_central`): parameter reuse from a finished federated run, and the one-client training path. numpy only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.paper.config import get_paper_cfg  # noqa: E402
from src.paper.evaluation import fast_auroc  # noqa: E402
from src.paper.fed_gbdt import FIXED_DEFAULTS, reuse_params_from_run, search_and_train  # noqa: E402

EXP = "mortality/no_apache_scores"


def write_run(root, records, exp=EXP):
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "results.json").write_text(json.dumps({"experiments": {exp: {"records": records}}}), encoding="utf-8")


def rec(seed, arm="fed_gbdt", **over):
    return {"arm": arm, "seed": seed, "extra": {"params": dict(FIXED_DEFAULTS, **over)}}


class TestDefaults(unittest.TestCase):
    def test_pooled_arm_is_off_by_default_and_the_override_path_exists(self):
        p = get_paper_cfg({})
        self.assertFalse(p["arms"]["gbdt_central"]); self.assertIsNone(p["gbdt_central"]["reuse_params_from"])
        q = get_paper_cfg({"paper": {"gbdt_central": {"reuse_params_from": "outputs/paper/x"}, "arms": {"gbdt_central": True}}})
        self.assertTrue(q["arms"]["gbdt_central"]); self.assertEqual(q["gbdt_central"]["reuse_params_from"], "outputs/paper/x")
        self.assertTrue(q["arms"]["fed_gbdt"] is False and q["arms"]["lightgbm_central"] is True)       # nothing else changes


class TestReuseParams(unittest.TestCase):
    def test_returns_exactly_the_fixed_keys_from_the_lowest_seed_whatever_the_record_order(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d, [rec(44, max_depth=2), rec(42, max_depth=7), rec(43, max_depth=3), rec(42, arm="lightgbm_central", max_depth=99)])
            p = reuse_params_from_run(d, EXP)
        self.assertEqual(set(p), set(FIXED_DEFAULTS)); self.assertEqual(p["max_depth"], 7)

    def test_missing_pieces_are_loud_never_a_silent_search(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError): reuse_params_from_run(Path(d) / "nope", EXP)
            write_run(d, [rec(42)])
            with self.assertRaises(KeyError): reuse_params_from_run(d, "sepsis/sepsis_clean")
            write_run(d, [rec(42, arm="lightgbm_central")])
            with self.assertRaises(ValueError): reuse_params_from_run(d, EXP)
            r = rec(42); r["extra"]["params"].pop("reg_lambda"); write_run(d, [r])
            with self.assertRaises(ValueError): reuse_params_from_run(d, EXP)


class TestOneClientTraining(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4); n = 900
        self.X = rng.normal(size=(n, 6)); self.y = ((self.X[:, 0] + 0.7 * self.X[:, 1] + rng.normal(0, 1, n)) > 0.8).astype(float)
        idx = rng.permutation(n); self.tr, self.va = idx[:600], idx[600:]
        self.cfg = {"n_bins": 16, "quantile_grid": 32, "max_rounds": 25, "early_stopping_rounds": 8}
        self.params = dict(FIXED_DEFAULTS, max_depth=3, learning_rate=0.2)

    def test_one_client_holding_every_training_row_matches_the_federated_model(self):
        fed_rows = {h: self.tr[i::5] for h, i in zip(range(5), range(5))}
        m_f, h_f = search_and_train(self.X, self.y, fed_rows, self.va, self.cfg, 42, self.params)
        m_c, h_c = search_and_train(self.X, self.y, {0: self.tr}, self.va, self.cfg, 42, self.params)
        self.assertEqual((h_f["n_clients"], h_c["n_clients"]), (5, 1)); self.assertEqual(h_c["n_train"], len(self.tr))
        a_f = fast_auroc(self.y[self.va], m_f.predict_raw(self.X)[self.va]); a_c = fast_auroc(self.y[self.va], m_c.predict_raw(self.X)[self.va])
        self.assertLess(abs(a_f - a_c), 0.02)                                # same code and parameters: only the merged bin edges differ
        self.assertEqual(h_c["params"]["max_depth"], 3)                      # the fixed parameters were used, no search

    def test_fixed_parameters_skip_the_search(self):
        _, h = search_and_train(self.X, self.y, {0: self.tr}, self.va, dict(self.cfg, n_trials=4), 42, self.params)
        self.assertEqual(len(h["trials"]), 1)


if __name__ == "__main__":
    unittest.main()
