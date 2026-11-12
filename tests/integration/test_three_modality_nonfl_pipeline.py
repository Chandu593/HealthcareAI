"""Non-FL Configs 1-4 end to end on synthetic upstream runs: per-hospital independence, test-set isolation, validation-only selection, reproducibility, hospital-local graph construction,
skip reporting, caching / manifests, the consolidated report, and the CLI (subprocess)."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.three_modality import cross_attention3 as ca
from src.three_modality import hw_data, hw_graph, non_federated as nf, nonfl_report as nr
from src.three_modality.downstream_config import DOWNSTREAM_DEFAULTS, get_cross_attention_cfg
from tests.unit._xattn_synth import build_synth_upstream

ROOT = Path(__file__).resolve().parents[2]
SIZES = [220, 220, 200, 24]                        # the last hospital is far too small to train: it must be REPORTED as skipped, never silently dropped


def fast_cfg():
    c = copy.deepcopy(DOWNSTREAM_DEFAULTS)
    c["gcn"].update(max_epochs=8, patience=4)
    c["classifiers"]["lr"].update(max_epochs=25, patience=5)
    c["classifiers"]["mlp"].update(max_epochs=25, patience=5)
    c["classifiers"]["lightgbm"].update(n_estimators=15, early_stopping_rounds=5)
    c["classifiers"]["gbdt"].update(n_estimators=10)
    return c


def strip(rows):
    return [{k: v for k, v in r.items() if k != "seconds"} for r in rows]


class NonFLBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "out"
        cls.root = cls.out / "three_modality"
        cls.up = build_synth_upstream(cls.root, per=SIZES, seed=5)
        xcfg = get_cross_attention_cfg({"three_modality": {"cross_attention": {"model": {"attn_dim": 8, "num_heads": 2, "output_dim": 8}, "training": {"max_epochs": 2, "batch_size": 64}}}})
        cls.ca_res = ca.build_cross_attention(cls.up["runs"], cls.up["bases"], cls.up["patient_table"], xcfg, {"window_minutes": 1440}, cls.root / "cross_attention")
        cls.sets = ("fused", "gru", "tabular", "tcd")
        cls.cohort = hw_data.load_cohort(cls.ca_res["run_dir"], cls.root, cls.sets)
        cls.cfg = fast_cfg()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def clone_cohort(self, **changes):
        c = copy.copy(self.cohort)
        c.labels = {k: v.copy() for k, v in self.cohort.labels.items()}
        c.embeddings = {k: v.copy() for k, v in self.cohort.embeddings.items()}
        return c

    def run_cfg(self, config, cohort=None, tasks=("mortality",), classifiers=("lr", "mlp", "lightgbm", "gbdt"), graphs=("A", "B"), sets=("fused",)):
        specs = nf.variant_specs(config, sets, classifiers, graphs, tasks)
        return nf.run_variants(config, cohort or self.cohort, self.cfg, specs, progress_every=0)


class TestNonFLRowsAndSkips(NonFLBase):
    def test_config1_counts_skip_reasons_and_metric_fields(self):
        rows = self.run_cfg(1, tasks=("mortality", "sepsis"), classifiers=("lr", "gbdt"))
        up = self.up
        self.assertEqual({r["hospitalid"] for r in rows}, {11, 12, 13, 14})
        for r in rows:
            h, task = r["hospitalid"], r["task"]
            truth = up["mort"] if task == "mortality" else up["sep"]
            for s in ("train", "val", "test"):
                m = (up["hosp"] == h) & (up["split"] == s) & up["landmark"] & ~np.isnan(truth)
                self.assertEqual(r["counts"][s], {"n": int(m.sum()), "pos": int((truth[m] == 1).sum()), "neg": int((truth[m] == 0).sum())})
            if r["status"] == "skipped":
                self.assertTrue(r["skip_reason"])
                self.assertIsNone(r["metrics"])
            else:
                self.assertEqual(r["status"], "trained")
                self.assertTrue(0.0 <= r["threshold"] <= 1.0)
                for k in ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "specificity", "confusion_matrix", "classification_report"):
                    self.assertIn(k, r["metrics"])
                self.assertEqual(r["metrics"]["n"], r["counts"]["test"]["n"])
        tiny = [r for r in rows if r["hospitalid"] == 14]
        self.assertTrue(tiny and all(r["status"] == "skipped" and "train patients" in r["skip_reason"] for r in tiny))
        self.assertTrue(any(r["status"] == "trained" for r in rows if r["hospitalid"] != 14))
        self.assertFalse(any(r["status"] == "failed" for r in rows), [r["skip_reason"] for r in rows if r["status"] == "failed"])


class TestIndependenceAndIsolation(NonFLBase):
    def test_hospitals_are_independent_and_order_does_not_matter(self):
        base = self.run_cfg(1)
        c2 = self.clone_cohort()
        sel = c2.hospitalid == 12
        rs = np.random.default_rng(0)
        c2.embeddings["fused"][sel] = rs.normal(size=c2.embeddings["fused"][sel].shape).astype(np.float32)      # replace ONE hospital's data entirely...
        for t in ("mortality", "sepsis"):
            c2.labels[t][sel] = np.where(np.isnan(c2.labels[t][sel]), np.nan, 1 - c2.labels[t][sel])           # ...and flip its labels
        again = self.run_cfg(1, cohort=c2)
        key = lambda r: (r["variant"], r["hospitalid"])                                                         # noqa: E731
        a, b = {key(r): r for r in strip(base)}, {key(r): r for r in strip(again)}
        self.assertEqual(set(a), set(b))
        other = [k for k in a if k[1] != 12]
        self.assertTrue(other)
        for k in other:
            self.assertEqual(json.dumps(a[k], sort_keys=True, default=str), json.dumps(b[k], sort_keys=True, default=str), f"hospital {k[1]} changed when ONLY hospital 12's data changed")
        self.assertTrue(any(json.dumps(a[k], default=str) != json.dumps(b[k], default=str) for k in a if k[1] == 12 and a[k]["status"] == "trained"))
        with mock.patch.object(nf, "plan_hospitals", side_effect=lambda c, t, m=None: list(reversed(hw_data.plan_hospitals(c, t, m)))):
            rev = self.run_cfg(1)
        self.assertEqual(sorted(json.dumps(r, sort_keys=True, default=str) for r in strip(base)), sorted(json.dumps(r, sort_keys=True, default=str) for r in strip(rev)))

    def _flip_test_labels(self):
        c = self.clone_cohort()
        te = c.split == "test"
        for t in ("mortality", "sepsis"):
            c.labels[t][te] = np.where(np.isnan(c.labels[t][te]), np.nan, 1 - c.labels[t][te])
        return c

    def _noise_test_features(self):
        c = self.clone_cohort()
        te = c.split == "test"
        rs = np.random.default_rng(1)
        for k in c.embeddings:
            c.embeddings[k][te] = rs.normal(0, 3, c.embeddings[k][te].shape).astype(np.float32)
        return c

    def test_test_set_isolation_for_every_config(self):
        for config in (1, 2, 3, 4):
            kw = dict(classifiers=("lr", "gbdt")) if config in (1, 4) else {}
            base = self.run_cfg(config, **kw)
            flipped = self.run_cfg(config, cohort=self._flip_test_labels(), **kw)
            noisy = self.run_cfg(config, cohort=self._noise_test_features(), **kw)
            trained = [i for i, r in enumerate(base) if r["status"] == "trained"]
            self.assertTrue(trained, f"config {config}: nothing trained")
            for other, what in ((flipped, "test LABELS"), (noisy, "test FEATURES")):
                for i in trained:
                    self.assertEqual(base[i]["threshold"], other[i]["threshold"], f"config {config}: {what} changed a validation-selected threshold")
                    drop = lambda d: {k: v for k, v in d.items() if k != "test_attach_edges"}          # noqa: E731  descriptive statistic OF the test graph itself, not a training/selection quantity
                    self.assertEqual(drop(base[i]["info"]), drop(other[i]["info"]), f"config {config}: {what} changed model selection / training")
            self.assertTrue(any(base[i]["metrics"] != flipped[i]["metrics"] for i in trained), "flipping test labels must change the test metrics (sanity)")
            self.assertTrue(any(base[i]["metrics"] != noisy[i]["metrics"] for i in trained))

    def test_validation_labels_drive_selection_but_test_does_not(self):
        base = self.run_cfg(1, classifiers=("lr",))
        c = self.clone_cohort()
        va = c.split == "val"
        for t in ("mortality", "sepsis"):
            c.labels[t][va] = np.where(np.isnan(c.labels[t][va]), np.nan, 1 - c.labels[t][va])
        flipped = self.run_cfg(1, cohort=c, classifiers=("lr",))
        trained = [i for i, r in enumerate(base) if r["status"] == "trained"]
        self.assertTrue(any(base[i]["threshold"] != flipped[i]["threshold"] or base[i]["info"] != flipped[i]["info"] for i in trained), "validation labels had no effect: selection is not using them")

    def test_runs_are_reproducible(self):
        for config in (1, 3):
            kw = dict(classifiers=("mlp", "lightgbm")) if config == 1 else {}
            a, b = self.run_cfg(config, **kw), self.run_cfg(config, **kw)
            self.assertEqual(json.dumps(strip(a), sort_keys=True, default=str), json.dumps(strip(b), sort_keys=True, default=str), f"config {config} is not reproducible")


class TestGraphConfigs(NonFLBase):
    def test_graph_construction_uses_only_that_hospitals_own_training_rows(self):
        seen = []
        real = hw_graph.fit_train_graph

        def spy(Z, key, gr):
            seen.append((np.array(Z, copy=True), key))
            return real(Z, key, gr)
        with mock.patch.object(hw_graph, "fit_train_graph", spy):
            self.run_cfg(2, tasks=("mortality",))
        self.assertTrue(seen)
        plan = {d.hospitalid: d for d in hw_data.plan_hospitals(self.cohort, "mortality")}
        for Z, key in seen:
            matches = [h for h, d in plan.items() if d.counts["train"]["n"] == len(Z) and np.allclose(d.X("train").astype(np.float64), Z)]
            self.assertEqual(len(matches), 1, "a graph was built from rows that are not exactly one hospital's training patients")
            h = matches[0]
            rows = plan[h].rows["train"]
            self.assertTrue((self.cohort.hospitalid[rows] == h).all() and (self.cohort.split[rows] == "train").all())

    def test_config3_graph_smote_reports_synthetic_nodes_and_config4_shares_one_gcn(self):
        r3 = [r for r in self.run_cfg(3) if r["status"] == "trained"]
        self.assertTrue(r3)
        for r in r3:
            sm = r["info"]["smote"]
            self.assertGreater(sm["n_synthetic"], 0)
            self.assertEqual(r["sampling"], "graph_smote")
            self.assertEqual(r["info"]["alpha"], 0.5)
        r4 = [r for r in self.run_cfg(4, classifiers=("lr", "gbdt")) if r["status"] == "trained"]
        by = {}
        for r in r4:
            by.setdefault((r["hospitalid"], r["graph"]), set()).add(r["info"]["graph_embedding_source"]["gcn_best_epoch"])
            self.assertTrue(r["info"]["graph_embedding_source"]["gcn_trained_without_sampling"])
        self.assertTrue(all(len(v) == 1 for v in by.values()), "the classifiers of one hospital/graph were not fed by the same GCN")


class TestRunsManifestsAndReport(NonFLBase):
    def test_manifest_cache_files_and_report(self):
        base = self.root / "non_federated"
        r1 = nf.run_nonfl(1, self.cohort, self.cfg, base, embedding_sets=self.sets, classifiers=("lr", "gbdt"), tasks=("mortality",))
        m = r1["manifest"]
        self.assertIn("independent model per hospital", m["regime"])
        self.assertEqual(m["cross_attention_run"], self.cohort.manifest["run_id"])
        self.assertEqual(m["split_identity"], self.cohort.manifest["split_identity"])
        self.assertFalse(m["partial"])
        self.assertIn("not directly comparable", m["not_directly_comparable"])
        for n in ("per_hospital.json", "per_hospital.csv", "summary.json", "manifest.json"):
            self.assertTrue((Path(r1["run_dir"]) / n).is_file())
        again = nf.run_nonfl(1, self.cohort, self.cfg, base, embedding_sets=self.sets, classifiers=("lr", "gbdt"), tasks=("mortality",))
        self.assertTrue(again["cache_hit"] and again["run_id"] == r1["run_id"])
        with self.assertRaises(FileExistsError):
            nf.run_nonfl(1, self.cohort, self.cfg, base, embedding_sets=self.sets, classifiers=("lr", "gbdt"), tasks=("mortality",), run_id=r1["run_id"], force=True)
        for c in (2, 3):
            nf.run_nonfl(c, self.cohort, self.cfg, base, tasks=("mortality",))
        nf.run_nonfl(1, self.cohort, self.cfg, base, embedding_sets=("fused",), classifiers=("lr",), tasks=("mortality",), max_hospitals=2)       # a PARTIAL run
        rep = nr.build_nonfl_report(base, self.root / "reports", self.cohort.manifest["run_id"])
        kinds = {c["comparison"] for c in rep["comparisons"]}
        self.assertTrue({"Graph A vs Graph B", "no sampling (Config 2) vs graph SMOTE (Config 3)", "individual modality vs fused embedding"} <= kinds, kinds)
        self.assertTrue(rep["manifest"]["partial_runs_excluded"])
        d = Path(rep["run_dir"])
        text = (d / "report.md").read_text(encoding="utf-8")
        self.assertIn("Not directly comparable", text)
        self.assertIn("centrally pretrained", text)
        self.assertTrue((d / "skipped_hospitals.csv").read_text().count("\n") > 1)
        for c in rep["comparisons"]:
            self.assertLessEqual(c["auroc"]["n_paired_hospitals"], 3)                                          # never more pairs than trained hospitals (the tiny one is skipped)
        # a modified run is refused, not reported
        p = Path(r1["run_dir"]) / "per_hospital.csv"
        p.write_text(p.read_text() + "tamper\n")
        with self.assertRaisesRegex(ValueError, "do not match their recorded hashes"):
            nr.build_nonfl_report(base, self.root / "reports2", self.cohort.manifest["run_id"])


class TestCLI(NonFLBase):
    def cli(self, *args):
        sets = ["--set", f"run.output_dir={self.out}", "--set", "three_modality.downstream.classifiers.lr.max_epochs=20", "--set", "three_modality.downstream.classifiers.lr.patience=5"]
        return subprocess.run([sys.executable, "run_three_modality.py", "--config", "configs/three_modality.yaml", *args, *sets], cwd=ROOT, capture_output=True, text=True)

    @staticmethod
    def summary(stdout):
        return json.loads(stdout[stdout.rindex("\n{") + 1: stdout.rindex("}") + 1])

    def test_nonfl_stage_report_and_argument_errors(self):
        r = self.cli("--stage", "nonfl_config1", "--classifiers", "lr", "--tasks", "mortality", "--max-hospitals", "2")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:] + r.stdout[-2000:])
        s = self.summary(r.stdout)
        self.assertTrue(s["partial"] and not s["cache_hit"])
        self.assertEqual(list(s["variants"]), ["c1__fused__mortality__lr"])
        self.assertTrue((Path(s["run_dir"]) / "manifest.json").is_file())
        again = self.summary(self.cli("--stage", "nonfl_config1", "--classifiers", "lr", "--tasks", "mortality", "--max-hospitals", "2").stdout)
        self.assertTrue(again["cache_hit"])
        bad = self.cli("--stage", "nonfl_config2", "--embedding-set", "gru")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("nonfl_config1 only", bad.stderr + bad.stdout)
        worse = self.cli("--stage", "nonfl_config1", "--classifiers", "xgboost")
        self.assertNotEqual(worse.returncode, 0)
        self.assertIn("unknown classifier", worse.stderr + worse.stdout)
        rep = self.cli("--stage", "nonfl_report")                                    # only a partial run exists -> excluded -> nothing to report
        self.assertNotEqual(rep.returncode, 0)
        self.assertIn("no completed Non-FL runs", rep.stderr + rep.stdout)


class TestCLICrossAttention(unittest.TestCase):
    """The cross_attention stage through the real CLI (subprocess): explicit encoder run selection, verified-run reuse, --inspect-cache, and the permanent split manifest only LOADED."""

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        from src.data.splitting.patient_split import SplitManifest, save_split_manifest
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        cls.out = cls.base / "out"
        cls.up = build_synth_upstream(cls.out / "three_modality", per=[120, 120], seed=9)
        cls.data = cls.base / "eicu"
        cls.data.mkdir()
        cls.up["patient_table"].to_csv(cls.data / "patient.csv", index=False)
        save_split_manifest(SplitManifest(seed=1, train_frac=.7, val_frac=.15, test_frac=.15, stratify_on="none", sepsis_cohort=None,
                                          patient_to_split={str(int(i)): str(sp) for i, sp in zip(cls.up["ids"], cls.up["split"])}), cls.base / "split_manifest.json")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cli(self, *args):
        sets = ["--set", f"data.root={self.data}", "--set", f"run.output_dir={self.out}", "--set", f"ablation.split.manifest_path={self.base / 'split_manifest.json'}",
                "--set", "three_modality.cross_attention.model.attn_dim=8", "--set", "three_modality.cross_attention.model.num_heads=2", "--set", "three_modality.cross_attention.model.output_dim=8",
                "--set", "three_modality.cross_attention.training.max_epochs=2", "--set", "three_modality.cross_attention.training.batch_size=64"]
        return subprocess.run([sys.executable, "run_three_modality.py", "--config", "configs/three_modality.yaml", "--stage", "cross_attention", *args, *sets], cwd=ROOT, capture_output=True, text=True)

    def test_cross_attention_stage_run_reuse_inspect(self):
        ids = ["--tcd-encoder-run-id", "tcde_synth", "--gru-encoder-run-id", "grue_synth", "--tabular-ae-run-id", "tabae_synth"]
        r = self.cli(*ids)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:] + r.stdout[-2000:])
        s = json.loads(r.stdout[r.stdout.rindex("\n{") + 1: r.stdout.rindex("}") + 1])
        self.assertFalse(s["cache_hit"])
        self.assertEqual(s["embedding_shape"], [240, 8])
        self.assertEqual(s["token_order"], ["gru", "tabular", "tcd"])
        self.assertEqual(s["encoder_runs"], {"gru": "grue_synth", "tabular": "tabae_synth", "tcd": "tcde_synth"})
        z = np.load(Path(s["run_dir"]) / "fused_embeddings.npz")
        self.assertTrue(np.array_equal(z["patient_ids"], self.up["ids"]))
        again = self.cli(*ids)
        self.assertEqual(again.returncode, 0)
        self.assertIn('"cache_hit": true', again.stdout)
        insp = self.cli(*ids, "--inspect-cache")
        self.assertEqual(insp.returncode, 0, insp.stderr[-1500:])
        self.assertIn(s["run_id"], insp.stdout)


if __name__ == "__main__":
    unittest.main()
