"""Fixes after the first real Non-FL runs: stabilised LightGBM focal objective, selectable checkpoint metric / min_epochs, recall-target operating point, aggregated per-class reports,
pooled prediction metrics, validation-only tuning helpers and the hyper-parameter overlay (synthetic data only)."""
import copy
import unittest
import warnings

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from src.three_modality import hw_classifiers as hc
from src.three_modality import hw_eval, hw_reports, hw_tune
from src.three_modality.downstream_config import DOWNSTREAM_DEFAULTS, apply_hparams, get_downstream_cfg, get_tuning_cfg


def cfg():
    return copy.deepcopy(DOWNSTREAM_DEFAULTS)


def toy(n=260, d=8, seed=0, pos=0.25, sig=2.5):
    rs = np.random.default_rng(seed)
    y = (rs.random(n) < pos).astype(int)
    Z = rs.normal(size=(n, d))
    Z[:, 0] += sig * y
    return Z, y


OLD_LGB = {"hessian_floor": 1e-9, "lambda_l2": 0.0, "max_delta_step": 0.0, "min_sum_hessian_in_leaf": 1e-3}


class TestLightGBMStability(unittest.TestCase):
    def _runs(self, over):
        c = cfg()
        c["classifiers"]["lightgbm"].update(over)
        out = []
        for n in (60, 120, 250):
            for s in range(3):
                rs = np.random.default_rng(1000 * n + s)
                N = n + 80
                y = (rs.random(N) < 0.09).astype(int)
                if y[:n].sum() < 3 or y[n:].sum() < 1:
                    continue
                X = rs.normal(size=(N, 128))
                X[:, :6] += 1.2 * y[:, None]                                 # 128-d, ~9% prevalence, tiny hospital: the regime of the real runs
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)           # an overflow in exp is an error here
                    r = hc.fit_predict("lightgbm", X[:n], y[:n], X[n:n + 40], y[n:n + 40], X[n + 40:], c, seed=s)
                out.append((r["info"]["max_abs_raw_score"], r["info"]["best_iteration"]))
        return np.array(out)

    def test_stabilised_objective_keeps_raw_scores_bounded_where_the_old_settings_explode(self):
        new, old = self._runs({}), self._runs(OLD_LGB)
        self.assertGreater(len(new), 5)
        self.assertLess(new[:, 0].max(), 10.0, "stabilised LightGBM produced exploding raw scores")
        self.assertGreater(old[:, 0].max(), 30.0, "fixture no longer reproduces the instability of the old settings (test would be vacuous)")
        self.assertLessEqual((new[:, 1] <= 2).mean(), (old[:, 1] <= 2).mean())

    def test_numerics_are_overflow_free_at_extreme_scores(self):
        z = np.array([-800.0, -50.0, 0.0, 50.0, 800.0])
        y = np.array([1, 0, 1, 0, 1])
        with np.errstate(all="raise"):
            g, h = hc.focal_grad_hess(z, y, 0.5, 2.0)
            loss = hc.focal_loss_np(z, y, 0.5, 2.0)
        self.assertTrue(np.isfinite(g).all() and np.isfinite(h).all() and np.isfinite(loss))

    def test_hessian_floor_scales_with_the_class_weight_and_validation_rejects_bad_values(self):
        c = get_downstream_cfg({})
        self.assertGreater(c["classifiers"]["lightgbm"]["hessian_floor"], 1e-3)
        with self.assertRaises(ValueError):
            get_downstream_cfg({"three_modality": {"downstream": {"classifiers": {"lightgbm": {"hessian_floor": 0}}}}})


class TestSelectionMetrics(unittest.TestCase):
    def setUp(self):
        Z, y = toy(n=300, seed=3, sig=1.0)
        self.tr, self.va, self.te = slice(0, 170), slice(170, 240), slice(240, 300)
        self.Z, self.y = Z, y

    def _fit(self, name, over, **kw):
        c = cfg()
        c["classifiers"][name].update(over)
        return hc.fit_predict(name, self.Z[self.tr], self.y[self.tr], self.Z[self.va], self.y[self.va], self.Z[self.te], c, seed=7, **kw)

    def test_min_epochs_is_a_hard_floor_for_the_selected_checkpoint(self):
        for name in ("lr", "mlp"):
            r = self._fit(name, {"min_epochs": 15, "selection_metric": "auroc"})
            self.assertGreaterEqual(r["info"]["best_epoch"], 15, name)
            self.assertEqual(r["info"]["min_epochs"], 15)

    def test_auroc_selection_never_returns_a_worse_validation_auroc_than_a_shorter_budget(self):
        for name in ("lr", "mlp"):
            full = self._fit(name, {"selection_metric": "auroc", "patience": 1000, "max_epochs": 40})
            a_full = roc_auc_score(self.y[self.va], full["p_val"])
            for e in (1, 3, 10, 20):
                capped = self._fit(name, {"selection_metric": "auroc", "patience": 1000, "max_epochs": e})
                self.assertGreaterEqual(a_full + 1e-12, roc_auc_score(self.y[self.va], capped["p_val"]), f"{name}: epoch budget {e} beats the selected checkpoint")

    def test_loss_selection_never_returns_a_worse_validation_loss_than_a_shorter_budget(self):
        full = self._fit("mlp", {"selection_metric": "loss", "patience": 1000, "max_epochs": 40})
        for e in (1, 5, 15):
            capped = self._fit("mlp", {"selection_metric": "loss", "patience": 1000, "max_epochs": e})
            self.assertLessEqual(full["info"]["best_val_focal_loss"], capped["info"]["best_val_focal_loss"] + 1e-9)

    def test_gbdt_stage_choice_follows_the_chosen_validation_metric(self):
        full = self._fit("gbdt", {"selection_metric": "auroc", "n_estimators": 60})
        for k in (5, 20):
            capped = self._fit("gbdt", {"selection_metric": "auroc", "n_estimators": k})
            self.assertGreaterEqual(roc_auc_score(self.y[self.va], full["p_val"]) + 1e-12, roc_auc_score(self.y[self.va], capped["p_val"]))
        self.assertEqual(full["info"]["selection_metric"], "auroc")

    def test_lightgbm_auroc_selection_runs_and_reports_the_metric(self):
        r = self._fit("lightgbm", {"selection_metric": "auroc", "n_estimators": 60})
        self.assertEqual(r["info"]["selection_metric"], "auroc")
        self.assertAlmostEqual(r["info"]["best_val_score"], roc_auc_score(self.y[self.va], r["p_val"]), places=6)

    def test_per_family_focal_override_changes_alpha_and_gamma_only_for_that_family(self):
        a = self._fit("lr", {"gamma": 0.0, "alpha": 0.3})
        self.assertEqual((a["info"]["gamma"], a["info"]["alpha"]), (0.0, 0.3))
        b = self._fit("mlp", {})
        self.assertEqual(b["info"]["gamma"], 2.0)

    def test_validation_with_one_class_falls_back_to_loss(self):
        self.assertEqual(hc.selection_score("auroc", 0.7, np.zeros(5, int), np.linspace(0, 1, 5)), 0.7)


class TestRecallTargetThreshold(unittest.TestCase):
    def test_highest_threshold_reaching_the_target_and_validation_only(self):
        rs = np.random.default_rng(0)
        y = (rs.random(120) < 0.3).astype(int)
        p = np.clip(0.3 + 0.35 * y + rs.normal(0, 0.2, 120), 0, 1)
        t = hw_eval.select_threshold_recall(y, p, 0.65)
        rec = lambda th: ((p >= th) & (y == 1)).sum() / y.sum()     # noqa: E731
        self.assertGreaterEqual(rec(t), 0.65)
        higher = np.unique(p)[np.unique(p) > t]
        self.assertTrue(all(rec(h) < 0.65 for h in higher), "a higher threshold would also reach the target")
        self.assertEqual(hw_eval.select_threshold_recall(np.zeros(5, int), np.linspace(0, 1, 5), 0.65), 0.5)
        self.assertEqual(list(__import__("inspect").signature(hw_eval.select_threshold_recall).parameters), ["y_val", "p_val", "target"])


class TestClassReports(unittest.TestCase):
    def test_confusion_report_matches_sklearn(self):
        rs = np.random.default_rng(1)
        y = (rs.random(300) < 0.3).astype(int)
        pred = np.where(rs.random(300) < 0.8, y, 1 - y)
        cm = {"tn": int(((pred == 0) & (y == 0)).sum()), "fp": int(((pred == 1) & (y == 0)).sum()), "fn": int(((pred == 0) & (y == 1)).sum()), "tp": int(((pred == 1) & (y == 1)).sum())}
        mine = hw_reports.class_report_from_confusion(cm)
        ref = classification_report(y, pred, labels=[0, 1], target_names=["negative", "positive"], output_dict=True)
        for c in ("negative", "positive"):
            for k in ("precision", "recall", "f1-score", "support"):
                self.assertAlmostEqual(mine[c][k], ref[c][k], places=9)
        self.assertAlmostEqual(mine["accuracy"], ref["accuracy"], places=9)
        for avg in ("macro avg", "weighted avg"):
            for k in ("precision", "recall", "f1-score"):
                self.assertAlmostEqual(mine[avg][k], ref[avg][k], places=9)

    def test_aggregate_sums_confusions_and_excludes_undefined_per_class_values(self):
        def row(h, tn, fp, fn, tp, status="trained"):
            return {"status": status, "variant": "v", "hospitalid": h, "metrics": {"confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}} if status == "trained" else None}
        rows = [row(1, 50, 5, 3, 7), row(2, 30, 0, 0, 0), row(3, 0, 0, 0, 0, "skipped"), row(4, 40, 10, 4, 6)]
        rep = hw_reports.aggregate_class_reports(rows)
        self.assertEqual(rep["n_hospitals_trained"], 3)
        self.assertEqual(rep["pooled_over_hospital_models"]["confusion_matrix"], {"tn": 120, "fp": 15, "fn": 7, "tp": 13})
        pos = rep["pooled_over_hospital_models"]["positive"]
        self.assertAlmostEqual(pos["precision"], 13 / 28)
        self.assertAlmostEqual(pos["recall"], 13 / 20)
        self.assertEqual(pos["support"], 20)
        m = rep["hospital_macro_per_class"]["positive"]
        self.assertEqual(m["n_hospitals_recall"], 2)                       # hospital 2 has no positives -> recall undefined, excluded and counted
        self.assertAlmostEqual(m["recall"], (7 / 10 + 6 / 10) / 2)
        self.assertEqual(len(hw_reports.per_hospital_class_rows(rows)), 6)
        self.assertIn("| positive |", hw_reports.class_report_markdown("v", rep))


class TestPooledMetrics(unittest.TestCase):
    def _preds(self):
        rs = np.random.default_rng(2)
        store = hw_reports.PredictionStore()
        thr = {}
        for h, n in ((11, 80), (12, 120), (13, 60)):
            y = (rs.random(n) < 0.2).astype(int)
            p = np.clip(0.25 + 0.4 * y + rs.normal(0, 0.2, n), 0, 1)
            store.add("v", np.arange(n) + 1000 * h, h, y, p)
            thr[h] = 0.3 + 0.02 * (h - 11)
        return store, thr

    def test_store_roundtrip_and_pooled_values_match_manual(self):
        import tempfile, os
        store, thr = self._preds()
        with tempfile.TemporaryDirectory() as d:
            store.save(os.path.join(d, "p.npz"))
            pred = hw_reports.load_predictions(os.path.join(d, "p.npz"))["v"]
        res = hw_reports.pooled_metrics(pred, thr, None, n_boot=50, seed=1)
        y, p, hid = pred["y"].astype(int), pred["p"], pred["hid"]
        yhat = (p >= np.array([thr[int(h)] for h in hid])).astype(int)
        self.assertAlmostEqual(res["point"]["auroc"], roc_auc_score(y, p))
        self.assertAlmostEqual(res["point"]["recall"], ((yhat == 1) & (y == 1)).sum() / (y == 1).sum())
        self.assertAlmostEqual(res["point"]["accuracy"], (yhat == y).mean())
        self.assertEqual((res["n_patients"], res["n_hospitals"]), (len(y), 3))
        self.assertAlmostEqual(res["all_negative_accuracy"], 1 - y.mean())
        lo, hi = res["ci95"]["auroc"]
        self.assertLessEqual(lo, res["point"]["auroc"] + 0.05)
        self.assertGreaterEqual(hi, res["point"]["auroc"] - 0.05)
        again = hw_reports.pooled_metrics(pred, thr, None, n_boot=50, seed=1)
        self.assertEqual(res["ci95"], again["ci95"])                          # reproducible

    def test_subset_restricts_to_the_chosen_hospitals(self):
        store, thr = self._preds()
        pred = {k: np.concatenate(v) for k, v in store.data["v"].items()}
        sub = hw_reports.pooled_metrics(pred, thr, [12], n_boot=10, seed=0)
        self.assertEqual((sub["n_hospitals"], sub["n_patients"]), (1, 120))


class TestTuningHelpers(unittest.TestCase):
    def test_grids_start_with_the_defaults_and_every_recipe_validates(self):
        g = hw_tune.default_grids()
        base = cfg()
        for fam, recipes in g.items():
            self.assertEqual(recipes[0], {}, f"{fam}: recipe 0 must be the current defaults")
            for r in recipes:
                if fam == "gcn":
                    apply_hparams(base, {"gcn": r})
                elif fam == "graph":
                    apply_hparams(base, {"graph": r})
                else:
                    apply_hparams(base, {"classifiers": {fam: r}})

    def test_argbest_rules(self):
        res = [{"auroc": 0.7, "auprc": 0.2}, {"auroc": 0.7, "auprc": 0.3}, {"auroc": None, "auprc": None}, {"auroc": 0.7, "auprc": 0.3}]
        self.assertEqual(hw_tune._argbest(res), 1)                              # tie on AUROC -> AUPRC; remaining tie -> the earlier recipe
        self.assertEqual(hw_tune._argbest([{"auroc": None, "auprc": None}, {"auroc": 0.5, "auprc": 0.1}]), 1)
        self.assertEqual(hw_tune._argbest([{"auroc": 0.6, "auprc": 0.1}, {"auroc": 0.6, "auprc": 0.1}]), 0)

    def test_overlay_cannot_change_eligibility_seed_or_reporting(self):
        for bad in ({"min_samples": {"train": 1}}, {"seed": 1}, {"reporting": {"n_bootstrap": 1}}, {"operating_points": {"recall_target": 0.1}}):
            with self.assertRaises(ValueError):
                apply_hparams(cfg(), bad)
        out = apply_hparams(cfg(), {"classifiers": {"lr": {"weight_decay": 0.1}}, "gcn": {"hidden": 32}})
        self.assertEqual((out["classifiers"]["lr"]["weight_decay"], out["gcn"]["hidden"]), (0.1, 32))
        self.assertEqual(cfg()["classifiers"]["lr"]["weight_decay"], DOWNSTREAM_DEFAULTS["classifiers"]["lr"]["weight_decay"])      # the input is not mutated
        with self.assertRaises(ValueError):
            apply_hparams(cfg(), {"classifiers": {"lr": {"selection_metric": "accuracy"}}})
        with self.assertRaises(ValueError):
            apply_hparams(cfg(), {"classifiers": {"lr": {"min_epochs": 1000}}})

    def test_tuning_config_defaults(self):
        t = get_tuning_cfg({})
        self.assertEqual((t["n_hospitals"], t["quick"]), (60, False))
        with self.assertRaises(ValueError):
            get_tuning_cfg({"three_modality": {"tuning": {"n_hospitals": 1}}})


class TestLightGBMTinyHospitals(unittest.TestCase):
    """The first real tuning run crashed with LightGBM's fatal 'num_features() > 0' check: min_child_samples=20 on a ~14-row fitting slice leaves no splittable feature."""

    def test_leaf_minimum_larger_than_the_training_set_no_longer_crashes(self):
        c = cfg()
        c["classifiers"]["lightgbm"].update(min_child_samples=20, num_leaves=7)
        rs = np.random.default_rng(0)
        for n in (8, 14, 40):
            X = rs.normal(size=(n + 30, 128))
            y = (rs.random(n + 30) < 0.3).astype(int)
            y[:3], y[3:6] = 1, 0
            for sel in ("loss", "auroc"):
                c["classifiers"]["lightgbm"]["selection_metric"] = sel
                r = hc.fit_predict("lightgbm", X[:n], y[:n], X[n:n + 15], y[n:n + 15], X[n + 15:], c, seed=1)
                self.assertTrue(np.isfinite(r["p_test"]).all() and (r["p_test"] >= 0).all() and (r["p_test"] <= 1).all(), (n, sel))
                self.assertLessEqual(r["info"]["min_data_in_leaf_effective"], max(1, n // 4))
                self.assertIn("best_val_score", r["info"])

    def test_argbest_never_picks_a_disqualified_recipe(self):
        res = [{"auroc": 0.70, "auprc": 0.2}, {"auroc": 0.95, "auprc": 0.9, "disqualified": True}, {"auroc": 0.72, "auprc": 0.2}]
        self.assertEqual(hw_tune._argbest(res), 2)
        self.assertEqual(hw_tune._argbest([{"auroc": 0.9, "auprc": 0.9, "disqualified": True}, {"auroc": None, "auprc": None}]), 0)


if __name__ == "__main__":
    unittest.main()
