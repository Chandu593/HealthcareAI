"""End-to-end checks for the post-run fixes on synthetic upstream runs: saved patient-level test predictions, aggregated per-class classification reports, the recall-target operating
point, pooled metrics with intervals in the report, the validation-only tuning stage (never loads test data) and runs that apply tuned hyper-parameters."""
import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.three_modality import cross_attention3 as ca
from src.three_modality import hw_data, hw_reports, hw_tune, non_federated as nf, nonfl_report as nr
from src.three_modality.downstream_config import DOWNSTREAM_DEFAULTS, get_cross_attention_cfg
from tests.unit._xattn_synth import build_synth_upstream

ROOT = Path(__file__).resolve().parents[2]
SIZES = [220, 220, 200, 24]


def fast_cfg():
    c = copy.deepcopy(DOWNSTREAM_DEFAULTS)
    c["gcn"].update(max_epochs=8, patience=4)
    c["classifiers"]["lr"].update(max_epochs=25, patience=5)
    c["classifiers"]["mlp"].update(max_epochs=25, patience=5)
    c["classifiers"]["lightgbm"].update(n_estimators=15, early_stopping_rounds=5)
    c["classifiers"]["gbdt"].update(n_estimators=10)
    return c


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "out"
        cls.root = cls.out / "three_modality"
        cls.up = build_synth_upstream(cls.root, per=SIZES, seed=5)
        xcfg = get_cross_attention_cfg({"three_modality": {"cross_attention": {"model": {"attn_dim": 8, "num_heads": 2, "output_dim": 8}, "training": {"max_epochs": 2, "batch_size": 64}}}})
        cls.ca_res = ca.build_cross_attention(cls.up["runs"], cls.up["bases"], cls.up["patient_table"], xcfg, {"window_minutes": 1440}, cls.root / "cross_attention")
        cls.cohort = hw_data.load_cohort(cls.ca_res["run_dir"], cls.root, ("fused", "gru", "tabular", "tcd"))
        cls.cfg = fast_cfg()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def clone(self):
        c = copy.copy(self.cohort)
        c.labels = {k: v.copy() for k, v in self.cohort.labels.items()}
        c.embeddings = {k: v.copy() for k, v in self.cohort.embeddings.items()}
        return c


class TestPredictionsAndReports(Base):
    def test_predictions_reports_and_operating_points(self):
        base = self.root / "non_federated"
        res = nf.run_nonfl(1, self.cohort, self.cfg, base, embedding_sets=("fused",), classifiers=("lr", "gbdt"), tasks=("mortality", "sepsis"))
        d = Path(res["run_dir"])
        for n in ("predictions.npz", "classification_reports.json", "classification_reports.md", "per_hospital_classification_reports.csv"):
            self.assertTrue((d / n).is_file(), n)
            self.assertIn(n, res["manifest"]["files"])
        rows = json.loads((d / "per_hospital.json").read_text())
        preds = hw_reports.load_predictions(d / "predictions.npz")
        self.assertEqual(set(preds), {"c1__fused__mortality__lr", "c1__fused__mortality__gbdt", "c1__fused__sepsis__lr", "c1__fused__sepsis__gbdt"})
        trained = [r for r in rows if r["status"] == "trained"]
        self.assertTrue(trained)
        for r in trained:
            P = preds[r["variant"]]
            m = P["hid"] == r["hospitalid"]
            task = r["task"]
            dd = hw_data.hospital_task_data(self.cohort, r["hospitalid"], task)
            self.assertEqual(sorted(P["pid"][m].tolist()), sorted(dd.patient_ids("test").tolist()), "saved predictions are not exactly this hospital's test patients")
            order = {int(p): i for i, p in enumerate(dd.patient_ids("test"))}
            ytrue = dd.test_labels_for_evaluation()
            self.assertTrue(all(int(y) == int(ytrue[order[int(p)]]) for p, y in zip(P["pid"][m], P["y"][m])))
            pred = (P["p"][m] >= r["threshold"]).astype(int)
            y = P["y"][m].astype(int)
            cm = r["metrics"]["confusion_matrix"]
            self.assertEqual((cm["tp"], cm["fp"], cm["tn"], cm["fn"]), (int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())),
                             "re-applying the saved threshold to the saved predictions does not reproduce the recorded confusion matrix")
            rt = r["operating_points"]["recall_target"]
            self.assertEqual(rt["target"], 0.65)
            self.assertTrue(0.0 <= rt["threshold"] <= 1.0)
        rep = json.loads((d / "classification_reports.json").read_text())
        self.assertEqual(set(rep), set(preds))
        for v, body in rep.items():
            tv = [r for r in rows if r["variant"] == v and r["status"] == "trained"]
            self.assertEqual(body["f1_max_operating_point"]["n_hospitals_trained"], len(tv))
            self.assertEqual(body["f1_max_operating_point"]["pooled_over_hospital_models"]["n"], sum(r["metrics"]["n"] for r in tv))
            self.assertIn("recall_target_operating_point", body)
        with open(d / "per_hospital_classification_reports.csv", newline="") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 2 * len(trained))
        self.assertIn("### config 1 · mortality · c1__fused__mortality__lr", (d / "classification_reports.md").read_text())
        summ = json.loads((d / "summary.json").read_text())
        self.assertTrue(all("recall_target_operating_point" in s for s in summ.values()))

    def test_recall_target_threshold_is_validation_only(self):
        specs = nf.variant_specs(1, ("fused",), ("lr",), ("A",), ("mortality",))
        a = nf.run_variants(1, self.cohort, self.cfg, specs, progress_every=0)
        c = self.clone()
        te = c.split == "test"
        c.labels["mortality"][te] = np.where(np.isnan(c.labels["mortality"][te]), np.nan, 1 - c.labels["mortality"][te])
        b = nf.run_variants(1, c, self.cfg, specs, progress_every=0)
        tr = [i for i, r in enumerate(a) if r["status"] == "trained"]
        self.assertTrue(tr)
        for i in tr:
            self.assertEqual(a[i]["operating_points"]["recall_target"]["threshold"], b[i]["operating_points"]["recall_target"]["threshold"])
        self.assertTrue(any(a[i]["operating_points"]["recall_target"]["metrics"] != b[i]["operating_points"]["recall_target"]["metrics"] for i in tr))

    def test_report_has_pooled_metrics_per_class_reports_and_the_flics_reference_labelled_not_comparable(self):
        base = self.root / "non_federated_rep"
        nf.run_nonfl(1, self.cohort, self.cfg, base, classifiers=("lr",), tasks=("mortality",))
        nf.run_nonfl(2, self.cohort, self.cfg, base, graphs=("A",), tasks=("mortality",))
        rep = nr.build_nonfl_report(base, self.root / "reports_x", self.cohort.manifest["run_id"], n_bootstrap=25, top_hospitals=(2,))
        d = Path(rep["run_dir"])
        pooled = json.loads((d / "pooled_metrics.json").read_text())
        self.assertEqual(set(pooled), {"c1__fused__mortality__lr", "c2__graphA__mortality"})
        v = "c1__fused__mortality__lr"
        rows = [r for r in json.loads((Path(rep["manifest"]["nonfl_runs_used"] and base / rep["manifest"]["nonfl_runs_used"][0]) / "per_hospital.json").read_text()) if r["variant"] == v and r["status"] == "trained"]
        self.assertEqual(pooled[v]["all_trained_hospitals"]["n_patients"], sum(r["metrics"]["n"] for r in rows))
        self.assertLessEqual(pooled[v]["top_2_largest_hospitals"]["n_hospitals"], 2)
        self.assertIn("auroc", pooled[v]["all_trained_hospitals"]["ci95"])
        text = (d / "report.md").read_text(encoding="utf-8")
        self.assertIn("Reference only", text)
        self.assertIn("NOT directly comparable", text)
        self.assertIn("thresholds tuned ON THE TEST SET", text)
        self.assertIn("Pooled test metrics", text)
        self.assertIn("Secondary operating point", text)
        self.assertTrue((d / "classification_reports.md").is_file() and (d / "pooled_metrics.csv").is_file())


class TestTuning(Base):
    GRIDS = {"lr": [{}, {"weight_decay": 0.1, "selection_metric": "auroc", "min_epochs": 5}], "mlp": [{}, {"hidden": 32, "selection_metric": "auroc", "min_epochs": 5}],
             "lightgbm": [{}, {"num_leaves": 7, "selection_metric": "auroc"}], "gbdt": [{}, {"max_depth": 2, "selection_metric": "auroc"}],
             "gcn": [{}, {"hidden": 32, "selection_metric": "auroc", "min_epochs": 3}], "graph": [{}, {"k": 5}]}

    def tune(self, cohort=None, out="tuning", **kw):
        return hw_tune.run_tuning(cohort or self.cohort, self.cfg, self.root / out, tasks=("mortality",), n_hospitals=3, grids=self.GRIDS, **kw)

    def test_tuning_outputs_best_is_the_argmax_and_recipe_zero_is_the_defaults(self):
        res = self.tune(out="t_a")
        table = res["table"]
        self.assertTrue(all(t["overrides"] == {} for t in table if t["recipe"].endswith("_00")))
        for fam in ("lr", "mlp", "lightgbm", "gbdt"):
            rs = [t for t in table if t["family"] == fam]
            self.assertEqual(len(rs), 2)
            win = res["best_hparams"]["classifiers"][fam]
            scores = [t["auroc"] for t in rs]
            self.assertEqual(win, rs[hw_tune._argbest(rs)]["overrides"])
        for fam in ("gcn", "graph"):                                               # an empty override (the defaults won) is simply omitted from best_hparams
            rs = [t for t in table if t["family"] == fam]
            self.assertEqual(len(rs), 2)
            self.assertEqual(res["best_hparams"].get(fam, {}), rs[hw_tune._argbest(rs)]["overrides"])
        m = res["manifest"]
        self.assertIn("never loaded", m["test_data"])
        self.assertIn("TRAINING patients", m["method"])
        again = self.tune(out="t_a")
        self.assertTrue(again["cache_hit"])
        loaded = hw_tune.load_best_hparams(self.root / "t_a", res["run_id"])
        self.assertEqual(loaded, res["best_hparams"])
        self.assertTrue((Path(res["run_dir"]) / "tuning_table.csv").is_file())

    def test_tuning_never_loads_test_features_or_labels_and_scores_on_held_out_train(self):
        real_X = hw_data.HospitalTaskData.X
        seen_xte = []

        def guarded_X(self_, split, name="fused"):
            if split == "test":
                raise AssertionError("tuning loaded TEST features")
            return real_X(self_, split, name)
        import src.three_modality.hw_tune as ht
        real_fp = ht.fit_predict

        def spy(name, Xtr, ytr, Xva, yva, Xte, cfg, seed, smote=False):
            seen_xte.append((Xte.shape, Xtr.shape))
            return real_fp(name, Xtr, ytr, Xva, yva, Xte, cfg, seed, smote)
        with mock.patch.object(hw_data.HospitalTaskData, "X", guarded_X), \
             mock.patch.object(hw_data.HospitalTaskData, "test_labels_for_evaluation", side_effect=AssertionError("tuning loaded TEST labels")), \
             mock.patch.object(ht, "fit_predict", spy):
            self.tune(out="t_guard")
        self.assertTrue(seen_xte)
        for (xte_shape, xtr_shape) in seen_xte:                                # the scored part is a held-out slice of TRAIN, ~30% of it, disjoint from the fitting part
            self.assertAlmostEqual(xte_shape[0] / (xte_shape[0] + xtr_shape[0]), hw_tune.HOLDOUT_FRACTION, delta=0.06)

    def test_tuning_result_is_independent_of_test_labels_and_features(self):
        a = self.tune(out="t_i1")
        c = self.clone()
        te = c.split == "test"
        for t in ("mortality", "sepsis"):
            c.labels[t][te] = np.where(np.isnan(c.labels[t][te]), np.nan, 1 - c.labels[t][te])
        rs = np.random.default_rng(3)
        for k in c.embeddings:
            c.embeddings[k][te] = rs.normal(0, 3, c.embeddings[k][te].shape).astype(np.float32)
        b = self.tune(cohort=c, out="t_i2", force=True)
        self.assertEqual(a["best_hparams"], b["best_hparams"])
        self.assertEqual([(t["recipe"], t["auroc"], t["auprc"]) for t in a["table"]], [(t["recipe"], t["auroc"], t["auprc"]) for t in b["table"]])

    def test_a_recipe_that_errors_is_counted_disqualified_and_never_fatal(self):
        import src.three_modality.hw_tune as ht
        real = ht.fit_predict

        def flaky(name, Xtr, ytr, Xva, yva, Xte, cfg, seed, smote=False):
            if name == "lr" and cfg["classifiers"]["lr"]["weight_decay"] == 0.1:
                raise RuntimeError("synthetic recipe failure")
            return real(name, Xtr, ytr, Xva, yva, Xte, cfg, seed, smote)
        grids = {"lr": [{}, {"weight_decay": 0.1}], "graph": [{}], "gcn": [{}]}
        with mock.patch.object(ht, "fit_predict", flaky):
            res = hw_tune.run_tuning(self.cohort, self.cfg, self.root / "t_flaky", families=("lr",), tasks=("mortality",), n_hospitals=3, grids=grids)
        rows = {t["recipe"]: t for t in res["table"]}
        self.assertEqual(rows["lr_00"]["n_failed"], 0)
        self.assertGreater(rows["lr_01"]["n_failed"], 0)
        self.assertTrue(rows["lr_01"]["disqualified"])
        self.assertIn("RuntimeError", rows["lr_01"]["first_error"])
        self.assertEqual(res["best_hparams"]["classifiers"]["lr"], {})

    def test_interrupted_tuning_resumes_from_the_finished_recipes_and_matches_an_uninterrupted_run(self):
        import src.three_modality.hw_tune as ht
        real = ht._evaluate_recipe
        calls = {"n": 0}

        def dies_on_fourth(*a, **k):
            calls["n"] += 1
            if calls["n"] == 4:
                raise KeyboardInterrupt("simulated crash / Ctrl-C")
            return real(*a, **k)
        out = self.root / "t_resume"
        with mock.patch.object(ht, "_evaluate_recipe", dies_on_fourth):
            with self.assertRaises(KeyboardInterrupt):
                hw_tune.run_tuning(self.cohort, self.cfg, out, tasks=("mortality",), n_hospitals=3, grids=self.GRIDS)
        self.assertEqual(len(list(out.glob("_partial_*.json"))), 1)
        done = {"n": 0}

        def counting(*a, **k):
            done["n"] += 1
            return real(*a, **k)
        with mock.patch.object(ht, "_evaluate_recipe", counting):
            resumed = hw_tune.run_tuning(self.cohort, self.cfg, out, tasks=("mortality",), n_hospitals=3, grids=self.GRIDS)
        total = sum(len(v) for v in self.GRIDS.values())
        self.assertEqual(done["n"], total - 3, "finished recipes were recomputed instead of reused")
        self.assertEqual(list(out.glob("_partial_*.json")), [])
        fresh = hw_tune.run_tuning(self.cohort, self.cfg, self.root / "t_resume_fresh", tasks=("mortality",), n_hospitals=3, grids=self.GRIDS)
        self.assertEqual(resumed["best_hparams"], fresh["best_hparams"])
        self.assertEqual([(t["recipe"], t["auroc"]) for t in resumed["table"]], [(t["recipe"], t["auroc"]) for t in fresh["table"]])

    def test_holdout_split_is_stratified_disjoint_train_only_and_deterministic(self):
        d = hw_data.plan_hospitals(self.cohort, "mortality")[0]
        fit, ho = hw_tune.split_train_holdout(d, 1)
        self.assertEqual(len(set(fit) & set(ho)), 0)
        self.assertEqual(len(fit) + len(ho), d.counts["train"]["n"])
        y = d.labels("train")
        self.assertAlmostEqual(y[ho].mean(), y.mean(), delta=0.05)
        f2, h2 = hw_tune.split_train_holdout(d, 1)
        self.assertTrue(np.array_equal(fit, f2) and np.array_equal(ho, h2))

    def test_tuned_hparams_are_applied_recorded_and_change_the_cache_identity(self):
        base = self.root / "non_federated_hp"
        plain = nf.run_nonfl(1, self.cohort, self.cfg, base, classifiers=("lr",), tasks=("mortality",))
        hp = {"classifiers": {"lr": {"min_epochs": 7, "selection_metric": "auroc"}}}
        tuned = nf.run_nonfl(1, self.cohort, self.cfg, base, classifiers=("lr",), tasks=("mortality",), hparams=hp, hparams_source="tune_unit")
        self.assertFalse(tuned["cache_hit"])
        self.assertNotEqual(plain["run_id"], tuned["run_id"])
        self.assertEqual(tuned["manifest"]["hparams"]["source_run"], "tune_unit")
        rows = json.loads((Path(tuned["run_dir"]) / "per_hospital.json").read_text())
        tr = [r for r in rows if r["status"] == "trained"]
        self.assertTrue(tr and all(r["info"]["min_epochs"] == 7 and r["info"]["best_epoch"] >= 7 and r["info"]["selection_metric"] == "auroc" for r in tr))
        with self.assertRaises(ValueError):
            nf.run_nonfl(1, self.cohort, self.cfg, base, classifiers=("lr",), tasks=("mortality",), hparams={"min_samples": {"train": 1}}, hparams_source="x")


class TestNewestRunWins(Base):
    def test_report_uses_the_most_recently_created_run_for_a_variant_not_the_alphabetically_last(self):
        base = self.root / "non_federated_order"
        old = nf.run_nonfl(1, self.cohort, self.cfg, base, classifiers=("lr",), tasks=("mortality",), run_id="nonfl_zzz_older")
        new = nf.run_nonfl(1, self.cohort, self.cfg, base, classifiers=("lr",), tasks=("mortality",), run_id="nonfl_aaa_newer", force=True, hparams={"classifiers": {"lr": {"min_epochs": 3}}}, hparams_source="x")
        mp = Path(old["run_dir"]) / "manifest.json"
        m = json.loads(mp.read_text())
        m["created_utc"] = "2020-01-01T00:00:00Z"                              # the alphabetically-last run is the OLDER one
        mp.write_text(json.dumps(m))
        col = nr.collect_runs(base, self.cohort.manifest["run_id"])
        self.assertEqual(col["variants"]["c1__fused__mortality__lr"]["run_id"], "nonfl_aaa_newer")
        self.assertEqual(new["run_id"], "nonfl_aaa_newer")
        odd = nf.run_nonfl(2, self.cohort, self.cfg, base, graphs=("A",), tasks=("mortality",), run_id="my_custom_name")              # a run id that does not start with 'nonfl_c'
        self.assertEqual(nr.collect_runs(base, self.cohort.manifest["run_id"])["variants"]["c2__graphA__mortality"]["run_id"], odd["run_id"], "runs are selected by manifest content, not by run-id spelling")


class TestCLITuning(Base):
    def cli(self, *args):
        sets = ["--set", f"run.output_dir={self.out}", "--set", "three_modality.downstream.classifiers.lr.max_epochs=20", "--set", "three_modality.downstream.classifiers.lr.patience=5",
                "--set", "three_modality.downstream.classifiers.gbdt.n_estimators=8"]
        return subprocess.run([sys.executable, "run_three_modality.py", "--config", "configs/three_modality.yaml", *args, *sets], cwd=ROOT, capture_output=True, text=True)

    def test_tune_then_apply(self):
        r = self.cli("--stage", "nonfl_tune", "--tasks", "mortality", "--tune-families", "lr,gbdt", "--tune-hospitals", "3", "--set", "three_modality.tuning.quick=true", "--run-id", "tune_cli")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:] + r.stdout[-1500:])
        s = json.loads(r.stdout[r.stdout.rindex("\n{") + 1: r.stdout.rindex("}") + 1])
        self.assertEqual(s["run_id"], "tune_cli")
        self.assertIn("lr", s["best_hparams"]["classifiers"])
        r2 = self.cli("--stage", "nonfl_config1", "--classifiers", "lr", "--tasks", "mortality", "--max-hospitals", "3", "--hparams-run-id", "tune_cli", "--run-id", "c1_tuned")
        self.assertEqual(r2.returncode, 0, r2.stderr[-2000:] + r2.stdout[-1500:])
        man = json.loads((self.root / "non_federated" / "c1_tuned" / "manifest.json").read_text())
        self.assertEqual(man["hparams"]["source_run"], "tune_cli")
        bad = self.cli("--stage", "nonfl_config1", "--hparams-run-id", "does_not_exist")
        self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
