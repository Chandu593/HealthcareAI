"""Hospital-wise Non-FL building blocks (synthetic data only): split / test isolation, eligibility and skip rules, validation-only thresholds, metric definitions, focal gradient / Hessian
validation, the four classifiers, hospital-local Graph A/B, inductive attachment, GCN message-passing boundaries and graph-compatible SMOTE."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.classifiers.boosting_classifiers import select_threshold_from_validation
from src.graph.patient_graph import build_hospital_graph
from src.three_modality import cross_attention3 as ca
from src.three_modality import hw_classifiers as hc
from src.three_modality import hw_data, hw_eval, hw_graph
from src.three_modality.downstream_config import DOWNSTREAM_DEFAULTS, GRAPHS, get_cross_attention_cfg
from tests.unit._xattn_synth import build_synth_upstream


def cfg(**over):
    c = copy.deepcopy(DOWNSTREAM_DEFAULTS)
    c["gcn"].update(max_epochs=12, patience=6)
    for k, v in over.items():
        c[k].update(v)
    return c


def toy(n=240, d=8, seed=0, pos=0.25):
    rs = np.random.default_rng(seed)
    y = (rs.random(n) < pos).astype(int)
    Z = rs.normal(size=(n, d))
    Z[:, 0] += 2.5 * y
    return Z, y


# ======================================================================================================================================
class TestDataIsolationAndEligibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.up = build_synth_upstream(cls.root, per=[150, 150, 24], seed=3)
        tcfg = {"window_minutes": 1440}
        cls.res = ca.build_cross_attention(cls.up["runs"], cls.up["bases"], cls.up["patient_table"], get_cross_attention_cfg({"three_modality": {"cross_attention": {
            "model": {"attn_dim": 8, "num_heads": 2, "output_dim": 8}, "training": {"max_epochs": 2, "batch_size": 64}}}}), tcfg, cls.root / "cross_attention")
        cls.cohort = hw_data.load_cohort(cls.res["run_dir"], cls.root, ("fused", "gru", "tabular", "tcd"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_test_labels_are_not_reachable_for_training(self):
        d = hw_data.hospital_task_data(self.cohort, 11, "mortality")
        for s in ("test", "other"):
            with self.assertRaises(PermissionError):
                d.labels(s)
        self.assertEqual(len(d.test_labels_for_evaluation()), d.counts["test"]["n"])

    def test_eligibility_is_landmark_and_known_label(self):
        up = self.up
        for task, truth in (("mortality", up["landmark"] & ~np.isnan(up["mort"])), ("sepsis", up["landmark"] & ~np.isnan(up["sep"]))):
            got = np.zeros(len(up["ids"]), bool)
            for h in self.cohort.hospitals():
                d = hw_data.hospital_task_data(self.cohort, int(h), task)
                for s in hw_data.SPLITS:
                    rows = d.rows[s]
                    self.assertTrue((self.cohort.hospitalid[rows] == h).all(), "a row from another hospital leaked in")
                    self.assertTrue((self.cohort.split[rows] == s).all())
                    got[rows] = True
            self.assertTrue(np.array_equal(got, truth), f"{task}: eligible set differs from landmark & known-label ground truth")

    def test_patient_with_invalid_exit_time_is_not_landmark(self):
        # the real data has stays whose ICU-exit time is missing/invalid but whose window_end_minutes is still 1440 -- they must not count as landmark
        up = self.up
        pt = up["patient_table"].copy()
        pt.loc[0, "unitdischargeoffset"] = np.nan
        run = ca.build_cross_attention(up["runs"], up["bases"], pt, get_cross_attention_cfg({"three_modality": {"cross_attention": {
            "model": {"attn_dim": 8, "num_heads": 2, "output_dim": 8}, "training": {"max_epochs": 1, "batch_size": 64}}}}), {"window_minutes": 1440}, self.root / "ca2")
        lab = np.load(Path(run["run_dir"]) / "task_labels.npz")
        self.assertFalse(bool(lab["landmark"][0]))

    def test_counts_match_ground_truth_and_skip_reasons_cover_every_criterion(self):
        ms = DOWNSTREAM_DEFAULTS["min_samples"]
        tiny = hw_data.hospital_task_data(self.cohort, 13, "mortality")           # the 24-patient hospital
        reason = tiny.skip_reason(ms)
        self.assertIsNotNone(reason)
        self.assertIn("train patients", reason)
        d = hw_data.hospital_task_data(self.cohort, 11, "sepsis")
        for s in ("train", "val", "test"):
            rows = d.rows[s]
            y = self.up["sep"][rows]
            self.assertEqual(d.counts[s], {"n": len(rows), "pos": int((y == 1).sum()), "neg": int((y == 0).sum())})
        zero = hw_data.HospitalTaskData(hospitalid=1, task="x", rows={}, counts={"train": {"n": 30, "pos": 0, "neg": 30}, "val": {"n": 5, "pos": 0, "neg": 0}, "test": {"n": 0, "pos": 0, "neg": 0}})
        r = zero.skip_reason(ms)
        for needle in ("train positives", "validation positives", "validation negatives"):
            self.assertIn(needle, r)
        self.assertIsNone(hw_data.HospitalTaskData(1, "x", {}, {"train": {"n": 30, "pos": 5, "neg": 25}, "val": {"n": 8, "pos": 2, "neg": 6}, "test": {"n": 8, "pos": 0, "neg": 8}}).skip_reason(ms))

    def test_hospitals_with_no_eligible_patient_for_the_task_are_absent_but_small_ones_are_planned_and_reported(self):
        plan = hw_data.plan_hospitals(self.cohort, "mortality")
        self.assertEqual([p.hospitalid for p in plan], [11, 12, 13])
        self.assertEqual([p.hospitalid for p in hw_data.plan_hospitals(self.cohort, "mortality", max_hospitals=2)], [11, 12])

    def test_individual_embedding_sets_are_aligned_by_patient_id(self):
        for k in ("gru", "tabular", "tcd"):
            np.testing.assert_allclose(self.cohort.embeddings[k], self.up["emb"][k], atol=1e-6)    # synth order is the TCD order; stored shuffled for none of them here
        up2 = build_synth_upstream(self.root / "u2", per=[150, 150, 24], seed=3, shuffle_gru=True, shuffle_tab=True)
        r2 = ca.build_cross_attention(up2["runs"], up2["bases"], up2["patient_table"], get_cross_attention_cfg({"three_modality": {"cross_attention": {
            "model": {"attn_dim": 8, "num_heads": 2, "output_dim": 8}, "training": {"max_epochs": 1, "batch_size": 64}}}}), {"window_minutes": 1440}, self.root / "ca3")
        c2 = hw_data.load_cohort(r2["run_dir"], self.root / "u2", ("gru", "tabular"))
        np.testing.assert_allclose(c2.embeddings["gru"], up2["emb"]["gru"], atol=1e-6)
        np.testing.assert_allclose(c2.embeddings["tabular"], up2["emb"]["tabular"], atol=1e-6)

    def test_stable_seed_is_order_independent_and_distinct(self):
        self.assertEqual(hw_data.stable_seed(42, 11, "mortality", "c1"), hw_data.stable_seed(42, 11, "mortality", "c1"))
        self.assertNotEqual(hw_data.stable_seed(42, 11, "mortality", "c1"), hw_data.stable_seed(42, 12, "mortality", "c1"))


# ======================================================================================================================================
class TestMetricsAndThresholds(unittest.TestCase):
    def test_threshold_uses_validation_only_and_matches_the_legacy_rule(self):
        rs = np.random.default_rng(0)
        y = (rs.random(80) < 0.3).astype(int)
        p = np.clip(0.3 + 0.35 * y + rs.normal(0, 0.2, 80), 0, 1)
        t = hw_eval.select_threshold(y, p)
        self.assertAlmostEqual(t, select_threshold_from_validation(y, p), places=12)
        self.assertEqual(hw_eval.select_threshold(np.zeros(5, int), np.linspace(0, 1, 5)), 0.5)   # single-class validation: nothing to learn

    def test_metrics_match_sklearn_and_confusion(self):
        rs = np.random.default_rng(1)
        y = (rs.random(60) < 0.4).astype(int)
        p = np.clip(0.4 + 0.3 * y + rs.normal(0, 0.25, 60), 0, 1)
        m = hw_eval.hospital_metrics(y, p, 0.5)
        self.assertAlmostEqual(m["auroc"], roc_auc_score(y, p))
        self.assertAlmostEqual(m["auprc"], average_precision_score(y, p))
        cm = m["confusion_matrix"]
        pred = p >= 0.5
        self.assertEqual((cm["tp"], cm["fp"], cm["tn"], cm["fn"]), (int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())))
        self.assertAlmostEqual(m["specificity"], cm["tn"] / (cm["tn"] + cm["fp"]))
        self.assertAlmostEqual(m["recall"], cm["tp"] / (cm["tp"] + cm["fn"]))
        self.assertAlmostEqual(m["accuracy"], (cm["tp"] + cm["tn"]) / 60)
        self.assertEqual(set(m["classification_report"]) >= {"negative", "positive"}, True)

    def test_undefined_metrics_are_none_not_zero(self):
        m = hw_eval.hospital_metrics(np.zeros(6, int), np.linspace(0, 1, 6), 0.5)
        self.assertIsNone(m["auroc"])
        self.assertIsNone(m["auprc"])
        self.assertIsNone(m["recall"])
        self.assertIsNotNone(m["specificity"])
        m1 = hw_eval.hospital_metrics(np.ones(4, int), np.array([.9, .8, .7, .6]), 0.95)        # nothing predicted positive
        self.assertIsNone(m1["precision"])
        self.assertIsNone(m1["specificity"])

    def test_aggregation_macro_weighted_none_handling_and_skip_accounting(self):
        def row(h, n, pos, auroc, status="trained", reason=None):
            if status != "trained":
                return {"status": status, "hospitalid": h, "skip_reason": reason, "metrics": None}
            m = {"n": n, "n_pos": pos, "n_neg": n - pos}
            m.update({k: None for k in hw_eval.METRICS})
            m["auroc"] = auroc
            return {"status": "trained", "hospitalid": h, "skip_reason": None, "metrics": m}
        rows = [row(1, 100, 20, 0.9), row(2, 20, 2, 0.5), row(3, 50, 10, None), row(4, 0, 0, None, "skipped", "train positives 1 < 3; validation negatives 0 < 1"),
                row(5, 0, 0, None, "skipped", "train patients 5 < 20")]
        a = hw_eval.aggregate(rows)
        self.assertEqual((a["n_hospitals"], a["n_trained"], a["n_skipped"]), (5, 3, 2))
        au = a["metrics"]["auroc"]
        self.assertAlmostEqual(au["macro_mean"], 0.7)
        self.assertAlmostEqual(au["sample_weighted_mean"], (0.9 * 100 + 0.5 * 20) / 120)
        self.assertEqual((au["n_defined"], au["n_undefined"]), (2, 1))
        self.assertAlmostEqual(au["macro_mean_powered_hospitals"], 0.9)          # hospital 2 has only 2 positives -> not 'powered'
        self.assertEqual(a["n_low_power_trained"], 1)
        self.assertEqual(a["skip_reason_counts"]["too few training positives"], 1)
        self.assertEqual(a["skip_reason_counts"]["no/too few validation negatives"], 1)
        self.assertEqual(a["skip_reason_counts"]["too few training patients"], 1)


# ======================================================================================================================================
class TestFocalAndClassifiers(unittest.TestCase):
    def test_focal_gradient_and_hessian_match_finite_differences(self):
        z = np.linspace(-6, 6, 41)
        for gamma in (0.0, 0.5, 1.0, 2.0, 3.0):
            for alpha in (0.1, 0.25, 0.5, 0.75):
                for y in (0, 1):
                    yy = np.full_like(z, y)
                    g, h = hc.focal_grad_hess(z, yy, alpha, gamma)
                    L = lambda zz: np.array([hc.focal_loss_np(np.array([a]), np.array([y]), alpha, gamma) for a in zz])   # noqa: E731
                    eps = 1e-4
                    gn = (L(z + eps) - L(z - eps)) / (2 * eps)
                    hn = (L(z + eps) - 2 * L(z) + L(z - eps)) / eps ** 2
                    np.testing.assert_allclose(g, gn, atol=1e-7, err_msg=f"gradient gamma={gamma} alpha={alpha} y={y}")
                    np.testing.assert_allclose(h, hn, atol=2e-4 * (1 + np.abs(hn).max()), err_msg=f"hessian gamma={gamma} alpha={alpha} y={y}")

    def test_gamma_zero_reduces_to_alpha_weighted_cross_entropy(self):
        z = np.linspace(-4, 4, 9)
        for y in (0, 1):
            g, h = hc.focal_grad_hess(z, np.full_like(z, y), 0.3, 0.0)
            p = 1 / (1 + np.exp(-z))
            w = 0.3 if y == 1 else 0.7
            np.testing.assert_allclose(g, w * (p - y), atol=1e-9)
            np.testing.assert_allclose(h, w * p * (1 - p), atol=1e-9)

    def test_torch_and_numpy_focal_loss_agree(self):
        rs = np.random.default_rng(2)
        z, y = rs.normal(0, 2, 50), (rs.random(50) < 0.3).astype(int)
        for alpha, gamma in ((0.25, 2.0), (0.5, 0.0), (0.7, 1.0)):
            t = hc.focal_loss_torch(torch.from_numpy(z), torch.from_numpy(y), alpha, gamma).item()
            self.assertAlmostEqual(t, hc.focal_loss_np(z, y, alpha, gamma), places=9)

    def test_alpha_resolution(self):
        f = DOWNSTREAM_DEFAULTS["focal"]
        y = np.array([0] * 80 + [1] * 20)
        self.assertAlmostEqual(hc.resolve_alpha(f, y, False), 0.8)               # balanced = fraction of negatives
        self.assertEqual(hc.resolve_alpha(f, y, True), 0.5)                      # SMOTE: no second class re-weighting
        self.assertEqual(hc.resolve_alpha({**f, "alpha": 0.3}, y, False), 0.3)

    def test_all_four_classifiers_learn_the_signal_and_return_probabilities(self):
        Z, y = toy(n=300, seed=1)
        tr, va, te = np.arange(0, 180), np.arange(180, 240), np.arange(240, 300)
        c = cfg()
        c["classifiers"]["lightgbm"]["n_estimators"] = 60
        c["classifiers"]["gbdt"]["n_estimators"] = 60
        for name in hc.CLASSIFIER_LOSSES:
            r = hc.fit_predict(name, Z[tr], y[tr], Z[va], y[va], Z[te], c, seed=5)
            for k in ("p_val", "p_test"):
                self.assertTrue(np.isfinite(r[k]).all() and (r[k] >= 0).all() and (r[k] <= 1).all(), name)
            self.assertGreater(roc_auc_score(y[te], r["p_test"]), 0.8, f"{name} did not learn the signal")
            self.assertEqual(len(r["p_val"]), 60)

    def test_gbdt_uses_the_standard_loss_and_documents_it(self):
        Z, y = toy(seed=2)
        r = hc.fit_predict("gbdt", Z[:150], y[:150], Z[150:200], y[150:200], Z[200:], cfg(), seed=1)
        self.assertIsNone(r["info"]["alpha"])
        self.assertIn("NO focal", r["info"]["loss"])
        self.assertIn("focal", hc.fit_predict("lightgbm", Z[:150], y[:150], Z[150:200], y[150:200], Z[200:], cfg(), seed=1)["info"]["loss"])

    def test_lightgbm_validation_loss_matches_the_loss_of_its_own_predictions(self):
        # guards the init-score bookkeeping of the custom objective: best_val_focal_loss must equal the focal loss of predict(raw) + init
        Z, y = toy(seed=3)
        tr, va, te = np.arange(0, 150), np.arange(150, 210), np.arange(210, 240)
        c = cfg()
        c["classifiers"]["lightgbm"]["n_estimators"] = 40
        r = hc.fit_predict("lightgbm", Z[tr], y[tr], Z[va], y[va], Z[te], c, seed=2)
        alpha = r["info"]["alpha"]
        z = np.log(r["p_val"] / (1 - r["p_val"]))
        self.assertAlmostEqual(hc.focal_loss_np(z, y[va], alpha, 2.0), r["info"]["best_val_focal_loss"], places=6)

    def test_classifiers_are_deterministic_and_never_read_test_labels_or_use_test_rows_to_fit(self):
        Z, y = toy(seed=4)
        tr, va, te = np.arange(0, 150), np.arange(150, 210), np.arange(210, 240)
        c = cfg()
        for name in ("lr", "mlp", "lightgbm", "gbdt"):
            a = hc.fit_predict(name, Z[tr], y[tr], Z[va], y[va], Z[te], c, seed=9)
            b = hc.fit_predict(name, Z[tr], y[tr], Z[va], y[va], Z[te], c, seed=9)
            np.testing.assert_array_equal(a["p_test"], b["p_test"], err_msg=f"{name} is not reproducible")
            Zt = Z.copy()
            Zt[te] += 25.0                                                      # corrupt ONLY the test features
            d = hc.fit_predict(name, Z[tr], y[tr], Z[va], y[va], Zt[te], c, seed=9)
            np.testing.assert_array_equal(a["p_val"], d["p_val"], err_msg=f"{name}: test features influenced the fitted model / validation predictions")
            self.assertEqual({k: v for k, v in a["info"].items() if k.startswith("best")}, {k: v for k, v in d["info"].items() if k.startswith("best")})

    def test_scaler_statistics_come_from_training_rows_only(self):
        from unittest import mock
        from sklearn.preprocessing import StandardScaler
        Z, y = toy(seed=5)
        tr, va, te = np.arange(0, 150), np.arange(150, 210), np.arange(210, 240)
        seen = []

        class Spy(StandardScaler):
            def fit(self, X, *a, **k):
                seen.append(np.array(X, copy=True))
                return super().fit(X, *a, **k)
        for mod, call in ((hc, lambda: hc.fit_predict("lr", Z[tr], y[tr], Z[va], y[va], Z[te], cfg(), seed=1)),
                          (hw_graph, lambda: hw_graph.train_gcn(Z[tr], y[tr], Z[va], y[va], Z[te], "A", cfg(), seed=1))):
            seen.clear()
            with mock.patch.object(mod, "StandardScaler", Spy):
                call()
            self.assertEqual(len(seen), 1)
            np.testing.assert_array_equal(seen[0], Z[tr], err_msg=f"{mod.__name__}: scaling was fitted on something other than the training rows")

    def test_single_class_training_is_refused(self):
        Z, y = toy()
        with self.assertRaisesRegex(ValueError, "single class"):
            hc.fit_predict("lr", Z[:50], np.zeros(50, int), Z[50:80], y[50:80], Z[80:100], cfg(), seed=1)


# ======================================================================================================================================
class TestGraphs(unittest.TestCase):
    def _gr(self, **kw):
        g = copy.deepcopy(DOWNSTREAM_DEFAULTS["graph"])
        g.update(kw)
        return g

    def test_train_graph_edges_are_identical_to_the_repo_builder(self):
        Z, _ = toy(n=150, d=6, seed=7)
        for key in ("A", "B"):
            gr = self._gr(cosine_threshold=0.0)
            tg = hw_graph.fit_train_graph(Z, key, gr)
            ref = build_hospital_graph(list(range(150)), Z, method=GRAPHS[key], k=gr["k"], cosine_threshold=gr["cosine_threshold"], n_clusters=gr["n_clusters"], cluster_seed=gr["cluster_seed"])
            np.testing.assert_array_equal(tg.edge_index, ref["edge_index"])
            np.testing.assert_allclose(tg.edge_weight, ref["edge_weight"], atol=1e-6)
        self.assertIsNotNone(hw_graph.fit_train_graph(Z, "B", self._gr()).kmeans)

    def test_graph_b_edges_stay_inside_clusters(self):
        Z, _ = toy(n=200, d=6, seed=8)
        tg = hw_graph.fit_train_graph(Z, "B", self._gr(cosine_threshold=-1.0))
        cl = tg.cluster_labels
        self.assertTrue((cl[tg.edge_index[0]] == cl[tg.edge_index[1]]).all())
        self.assertGreater(len(np.unique(cl)), 1)

    def test_attachment_is_directed_train_to_query_with_exact_cosine_top_k(self):
        Z, _ = toy(n=120, d=6, seed=9)
        Zq = np.random.default_rng(1).normal(size=(15, 6))
        for key in ("A", "B"):
            tg = hw_graph.fit_train_graph(Z, key, self._gr(cosine_threshold=0.2))
            s, d, w = hw_graph.attach_edges(tg, Zq)
            self.assertTrue(((s >= 0) & (s < 120)).all() and ((d >= 0) & (d < 15)).all())         # sources are TRAIN nodes, targets are QUERY nodes; no query->query possible
            Qn, Tn = Zq / np.linalg.norm(Zq, axis=1, keepdims=True), Z / np.linalg.norm(Z, axis=1, keepdims=True)
            np.testing.assert_allclose(w, (Qn[d] * Tn[s]).sum(1), atol=1e-5)
            self.assertTrue((w >= 0.2 - 1e-6).all())
            self.assertLessEqual(np.bincount(d, minlength=15).max(), 10)
            if key == "B":
                qcl = tg.kmeans.predict(Qn)
                self.assertTrue((tg.cluster_labels[s] == qcl[d]).all(), "Graph B attached a query to a training node of another cluster")

    def test_attachment_with_unreachable_threshold_gives_no_edges_not_an_error(self):
        Z, _ = toy(n=60, d=5, seed=10)
        tg = hw_graph.fit_train_graph(Z, "A", self._gr(cosine_threshold=1.01))
        s, d, w = hw_graph.attach_edges(tg, Z[:5] + 1.0)
        self.assertEqual((len(s), tg.edge_index.shape[1]), (0, 0))

    def test_labels_never_enter_graph_construction(self):
        import inspect
        self.assertEqual(list(inspect.signature(hw_graph.fit_train_graph).parameters), ["Z_train", "key", "gr"])
        self.assertEqual(list(inspect.signature(hw_graph.attach_edges).parameters), ["tg", "Zq", "chunk"])


# ======================================================================================================================================
class TestGCNBoundariesAndSmote(unittest.TestCase):
    def _setup(self, key="A", smote=False, seed=0):
        Z, y = toy(n=220, d=8, seed=seed)
        return Z, y, np.arange(0, 140), np.arange(140, 180), np.arange(180, 220)

    def _forward(self, tg, X_train, Xq, Zq, model):
        s, d, w = hw_graph.attach_edges(tg, Zq)
        ei, ew = hw_graph._graph_tensors(tg, s, d, w, len(Zq), torch)
        with torch.no_grad():
            return model(torch.cat([X_train, Xq]), ei, ew)

    def test_train_nodes_and_other_query_nodes_are_unaffected_by_a_query_node(self):
        Z, y, tr, va, te = self._setup()
        gr = copy.deepcopy(DOWNSTREAM_DEFAULTS["graph"]); gr["cosine_threshold"] = 0.0
        for key in ("A", "B"):
            tg = hw_graph.fit_train_graph(Z[tr], key, gr)
            torch.manual_seed(0)
            model = hw_graph.make_gcn(8, DOWNSTREAM_DEFAULTS["gcn"]).eval()
            Xtr = torch.from_numpy(Z[tr].astype(np.float32))
            Zq = Z[va][:10].copy()
            e0, l0 = self._forward(tg, Xtr, torch.from_numpy(Zq.astype(np.float32)), Zq, model)
            Zq2 = Zq.copy(); Zq2[3] += 10.0                                    # perturb ONE query patient
            e1, l1 = self._forward(tg, Xtr, torch.from_numpy(Zq2.astype(np.float32)), Zq2, model)
            n = len(tr)
            self.assertTrue(torch.equal(e0[:n], e1[:n]), "a query node changed a TRAIN node's representation (message passing crossed the boundary)")
            others = [i for i in range(10) if i != 3]
            self.assertTrue(torch.equal(e0[n:][others], e1[n:][others]), "a query node influenced ANOTHER query node (validation/test dependence)")
            self.assertGreater((e0[n + 3] - e1[n + 3]).abs().max().item(), 0.0)
            # and the train-node representation equals a forward on the train-only graph
            ei, ew = torch.from_numpy(tg.edge_index), torch.from_numpy(tg.edge_weight)
            with torch.no_grad():
                etr, _ = model(Xtr, ei, ew)
            self.assertTrue(torch.equal(e0[:n], etr))

    def test_val_and_test_are_attached_separately_and_do_not_see_each_other(self):
        Z, y, tr, va, te = self._setup(seed=1)
        c = cfg()
        r = hw_graph.train_gcn(Z[tr], y[tr], Z[va], y[va], Z[te], "A", c, seed=3)
        Zt2 = Z[te].copy(); Zt2[:] += 5.0                                      # change every test node's features
        r2 = hw_graph.train_gcn(Z[tr], y[tr], Z[va], y[va], Zt2, "A", c, seed=3)
        np.testing.assert_array_equal(r["p_val"], r2["p_val"])                 # validation predictions cannot depend on test nodes
        self.assertEqual(r["info"]["best_epoch"], r2["info"]["best_epoch"])
        Zv2 = Z[va].copy(); Zv2 += 5.0
        r3 = hw_graph.train_gcn(Z[tr], y[tr], Zv2, y[va], Z[te], "A", c, seed=3)
        # test predictions use the model chosen on validation; with a different validation set the chosen epoch may differ, so compare at a fixed one-epoch budget
        c1 = cfg(); c1["gcn"].update(max_epochs=1)
        a = hw_graph.train_gcn(Z[tr], y[tr], Z[va], y[va], Z[te], "A", c1, seed=3)
        b = hw_graph.train_gcn(Z[tr], y[tr], Zv2, y[va], Z[te], "A", c1, seed=3)
        np.testing.assert_array_equal(a["p_test"], b["p_test"])
        self.assertEqual(len(r3["p_val"]), len(va))

    def test_validation_labels_only_select_the_checkpoint(self):
        Z, y, tr, va, te = self._setup(seed=2)
        c1 = cfg(); c1["gcn"].update(max_epochs=1)
        a = hw_graph.train_gcn(Z[tr], y[tr], Z[va], y[va], Z[te], "B", c1, seed=4)
        b = hw_graph.train_gcn(Z[tr], y[tr], Z[va], 1 - y[va], Z[te], "B", c1, seed=4)
        np.testing.assert_array_equal(a["p_test"], b["p_test"])                # epoch-1 weights do not depend on validation labels

    def test_gcn_is_reproducible_and_returns_node_embeddings(self):
        Z, y, tr, va, te = self._setup(seed=3)
        c = cfg()
        a = hw_graph.train_gcn(Z[tr], y[tr], Z[va], y[va], Z[te], "A", c, seed=8, want_embeddings=True)
        b = hw_graph.train_gcn(Z[tr], y[tr], Z[va], y[va], Z[te], "A", c, seed=8, want_embeddings=True)
        np.testing.assert_array_equal(a["p_test"], b["p_test"])
        self.assertEqual((a["emb_train"].shape, a["emb_val"].shape, a["emb_test"].shape), ((140, 64), (40, 64), (40, 64)))

    # ---- graph-compatible SMOTE ----
    def test_synthetic_nodes_are_interpolations_of_same_hospital_training_positives(self):
        Z, y = toy(n=160, d=5, seed=11, pos=0.2)
        sc = copy.deepcopy(DOWNSTREAM_DEFAULTS["smote"])
        Zs, info = hw_graph.generate_synthetic_positives(Z, y, sc, seed=2)
        P = Z[y == 1]
        n_pos, n_neg = len(P), int((y == 0).sum())
        self.assertEqual(info["n_synthetic"], min(max(0, n_neg - n_pos), 20 * n_pos))
        self.assertEqual(Zs.shape, (info["n_synthetic"], 5))
        for z in Zs[:25]:                                                       # each synthetic point lies on a segment between two training positives
            best = np.inf
            for i in range(n_pos):
                for j in range(i + 1, n_pos):
                    a, b = P[i], P[j]
                    t = np.clip(np.dot(z - a, b - a) / np.dot(b - a, b - a), 0, 1)
                    best = min(best, np.linalg.norm(z - (a + t * (b - a))))
            self.assertLess(best, 1e-8)
        lo, hi = P.min(0), P.max(0)
        self.assertTrue(((Zs >= lo - 1e-9) & (Zs <= hi + 1e-9)).all(), "synthetic nodes left the hull of the training positives")

    def test_smote_skips_hospitals_with_too_few_positives_with_a_reason(self):
        Z, y = toy(n=60, seed=12)
        y1 = np.zeros(60, int); y1[0] = 1
        Zs, info = hw_graph.generate_synthetic_positives(Z, y1, DOWNSTREAM_DEFAULTS["smote"], seed=1)
        self.assertIsNone(Zs)
        self.assertIn("fewer than 2 training positives", info["skipped"])
        yb = np.array([1] * 30 + [0] * 10)
        Zs2, info2 = hw_graph.generate_synthetic_positives(np.random.default_rng(0).normal(size=(40, 4)), yb, DOWNSTREAM_DEFAULTS["smote"], seed=1)
        self.assertIsNone(Zs2)
        self.assertIn("already at/above", info2["note"])

    def test_smote_only_touches_training_data_and_is_hospital_local(self):
        import inspect
        self.assertEqual(list(inspect.signature(hw_graph.generate_synthetic_positives).parameters), ["Z_train", "y_train", "scfg", "seed"])
        Z, y, tr, va, te = self._setup(seed=5)
        c = cfg()
        a = hw_graph.train_gcn(Z[tr], y[tr], Z[va], y[va], Z[te], "A", c, seed=6, smote=True)
        Zv, Zt = Z[va] + 3.0, Z[te] - 3.0
        b = hw_graph.train_gcn(Z[tr], y[tr], Zv, 1 - y[va], Zt, "A", c, seed=6, smote=True)
        self.assertEqual(a["info"]["smote"]["n_synthetic"], b["info"]["smote"]["n_synthetic"])
        self.assertEqual(a["info"]["smote"]["synthetic_edges"], b["info"]["smote"]["synthetic_edges"])
        self.assertEqual((len(a["p_val"]), len(a["p_test"])), (len(va), len(te)))           # nothing synthetic is ever added to validation / test

    def test_synthetic_nodes_never_change_real_nodes_and_receive_edges_from_real_train_nodes_only(self):
        Z, y, tr, va, te = self._setup(seed=6)
        gr = copy.deepcopy(DOWNSTREAM_DEFAULTS["graph"]); gr["cosine_threshold"] = 0.0
        tg = hw_graph.fit_train_graph(Z[tr], "A", gr)
        Zs, _ = hw_graph.generate_synthetic_positives(Z[tr], y[tr], DOWNSTREAM_DEFAULTS["smote"], seed=1)
        s, d, w = hw_graph.attach_edges(tg, Zs)
        self.assertTrue((s < len(tr)).all() and (d < len(Zs)).all())                                  # sources: real training nodes; targets: synthetic nodes
        n, m = len(tr), len(Zs)
        ei, ew = hw_graph._graph_tensors(tg, s, d, w, m, torch)
        self.assertTrue((ei[1][tg.edge_index.shape[1]:] >= n).all() and (ei[0][tg.edge_index.shape[1]:] < n).all())   # no synthetic->real edge, no synthetic->synthetic edge
        torch.manual_seed(0)
        model = hw_graph.make_gcn(8, DOWNSTREAM_DEFAULTS["gcn"]).eval()
        Xtr = torch.from_numpy(Z[tr].astype(np.float32))
        with torch.no_grad():
            e_with, _ = model(torch.cat([Xtr, torch.from_numpy(Zs.astype(np.float32))]), ei, ew)
            e_without, _ = model(Xtr, torch.from_numpy(tg.edge_index), torch.from_numpy(tg.edge_weight))
        self.assertTrue(torch.equal(e_with[:n], e_without), "synthetic nodes altered a real training node's representation")


if __name__ == "__main__":
    unittest.main()
