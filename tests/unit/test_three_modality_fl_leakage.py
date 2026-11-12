"""FL leakage tests: test labels are architecturally unreachable from training / tuning / round selection / thresholds, graph and SMOTE stay hospital-local, and the frozen cohort is never modified.

The decisive test is the POISON test: every test label is flipped and every training artefact (global parameters, selected round, validation history, thresholds, predictions) must be
BIT-IDENTICAL while the test metrics change -- so labels can only reach the metric computation."""
import dataclasses
import hashlib
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.three_modality import federated as fed, fl_algorithms, fl_clients, fl_eval, fl_gbdt_adapter, fl_server, fl_tune
from src.three_modality import hw_data
from tests.unit._fl_cohort import build_fl_cohort, fast_ds, fast_fl, poison_test_labels

SRC = Path(fl_clients.__file__).parent


def digest(cohort):
    h = hashlib.sha256()
    for k in sorted(cohort.labels):
        h.update(np.ascontiguousarray(cohort.labels[k]).tobytes())
    for k in sorted(cohort.embeddings):
        h.update(np.ascontiguousarray(cohort.embeddings[k]).tobytes())
    h.update(np.ascontiguousarray(cohort.patient_ids).tobytes())
    return h.hexdigest()


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "out" / "three_modality"
        cls.up, cls.cohort, cls.ca = build_fl_cohort(cls.root)
        cls.ds, cls.fl = fast_ds(), fast_fl()
        cls.poisoned = poison_test_labels(cls.cohort)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestStructuralIsolation(Base):
    def test_clients_and_problems_expose_no_test_label_anywhere(self):
        for task in ("mortality", "sepsis"):
            clients, _ = fl_clients.build_clients(self.cohort, task, "fused", self.ds, "all")
            self.assertTrue(clients)
            for c in clients:
                self.assertEqual(set(c.y), {"train", "val"})
                self.assertEqual(set(c.counts["test"]), {"n"})                    # not even the test class counts
                fields = {f.name for f in dataclasses.fields(c)} | set(vars(c))
                self.assertFalse([f for f in fields if "test" in f.lower() and "label" in f.lower()])
                pr = fl_algorithms.ClientProblem(c, "lr", self.ds)
                with self.assertRaises(PermissionError):
                    pr.labels("test")
                for name, val in vars(pr).items():
                    self.assertFalse("y_test" in name or "ytest" in name or name == "yte")
                    if isinstance(val, torch.Tensor) and val.dtype in (torch.int64, torch.int32):
                        self.assertLessEqual(val.numel(), c.n_train + c.n_val + 1000)       # no int tensor sized like a test label vector beyond train/val

    def test_only_the_evaluator_reads_test_labels(self):
        for mod in (fl_clients, fl_algorithms, fl_server, fl_gbdt_adapter, fl_tune, fed):
            src = inspect.getsource(mod)
            self.assertNotIn("test_labels_for_evaluation", src, f"{mod.__name__} must not read test labels")
            self.assertNotIn("._y", src, f"{mod.__name__} must not touch the cohort's raw label store")
        self.assertIn("test_labels_for_evaluation", inspect.getsource(fl_eval))
        self.assertNotIn("fl_eval", inspect.getsource(fl_tune).replace("evaluator", ""), "the tuning module must not import the evaluator")

    def test_hw_data_still_refuses_test_label_access_through_the_public_api(self):
        d = hw_data.hospital_task_data(self.cohort, 11, "mortality")
        with self.assertRaises(PermissionError):
            d.labels("test")


class TestPoisonedTestLabels(Base):
    """Flip every test label: nothing a model, round selection, threshold or checkpoint depends on may change."""

    def run_both(self, fn):
        n = len(list(Path(self.tmp.name).glob("pair*")))
        a, b = Path(self.tmp.name) / f"pair{n}_clean", Path(self.tmp.name) / f"pair{n}_poisoned"
        ra, rb = fn(self.cohort, a), fn(self.poisoned, b)
        return Path(ra["run_dir"]), Path(rb["run_dir"])

    @staticmethod
    def tensors_equal(x, y):
        if isinstance(x, (list, tuple)):
            return len(x) == len(y) and all(TestPoisonedTestLabels.tensors_equal(p, q) for p, q in zip(x, y))
        if isinstance(x, dict):
            return set(x) == set(y) and all(TestPoisonedTestLabels.tensors_equal(x[k], y[k]) for k in x)
        if isinstance(x, torch.Tensor):
            return torch.equal(x, y)
        return x == y

    def assert_training_artifacts_identical(self, da, db):
        import json
        ra = json.loads((da / "per_hospital.json").read_text())
        rb = json.loads((db / "per_hospital.json").read_text())
        self.assertEqual(len(ra), len(rb))
        changed = 0
        for x, y in zip(ra, rb):
            for k in ("variant", "hospitalid", "status", "skip_reason", "threshold", "threshold_source", "eval_set", "head_source", "participated_in_training"):
                self.assertEqual(x[k], y[k], f"{k} depends on test labels ({x['variant']} / {x['hospitalid']})")
            self.assertEqual(x["operating_points"]["recall_target"]["threshold"] if x.get("operating_points") else None,
                             y["operating_points"]["recall_target"]["threshold"] if y.get("operating_points") else None)
            if x["metrics"] and x["metrics"].get("auroc") is not None and abs(x["metrics"]["auroc"] - y["metrics"]["auroc"]) > 1e-9:
                changed += 1
        self.assertGreater(changed, 0, "the poison did not change any test metric -- the test would be vacuous")
        for ckpt in sorted((da / "checkpoints").glob("*/final.*")):
            other = db / "checkpoints" / ckpt.parent.name / ckpt.name
            self.assertTrue(other.is_file())
            if ckpt.suffix == ".pt":
                a, b = torch.load(ckpt, weights_only=False)["result"], torch.load(other, weights_only=False)["result"]
                self.assertTrue(self.tensors_equal(a["w"], b["w"]), f"global parameters differ: {ckpt.parent.name}")
                self.assertEqual(a["best_round"], b["best_round"])
                self.assertEqual([h["val_auroc_hist"] for h in a["history"]], [h["val_auroc_hist"] for h in b["history"]])
                self.assertTrue(self.tensors_equal(a["heads"], b["heads"]))
            else:
                import pickle
                a, b = pickle.load(open(ckpt, "rb"))["result"], pickle.load(open(other, "rb"))["result"]
                self.assertEqual(a["best_iteration"], b["best_iteration"])
                for h in a["margins"]:
                    np.testing.assert_array_equal(a["margins"][h]["test"], b["margins"][h]["test"])
        pa, pb = np.load(da / "predictions.npz", allow_pickle=True), np.load(db / "predictions.npz", allow_pickle=True)
        for k in pa.files:
            if k.endswith("@p") or k.endswith("@pid") or k.endswith("@hid"):
                np.testing.assert_array_equal(pa[k], pb[k], f"predictions {k} depend on test labels")

    def test_config1_lr_mlp_and_fedgbdt(self):
        da, db = self.run_both(lambda c, d: fed.run_fl(1, c, self.ds, self.fl, d, classifiers=("lr", "mlp", "gbdt_fed"), algorithms=("fedavg", "fedprox", "fedadam", "scaffold"),
                                                       tasks=("mortality", "sepsis"), run_id="p1"))
        self.assert_training_artifacts_identical(da, db)

    def test_config2_and_config3_graph_models(self):
        for cfgn in (2, 3):
            da, db = self.run_both(lambda c, d, n=cfgn: fed.run_fl(n, c, self.ds, self.fl, d, graphs=("A", "B"), algorithms=("fedavg", "scaffold"), tasks=("mortality",), run_id=f"p{n}"))
            self.assert_training_artifacts_identical(da, db)

    def test_config4_two_stage_pipeline(self):
        da, db = self.run_both(lambda c, d: fed.run_fl(4, c, self.ds, self.fl, d, graphs=("A",), classifiers=("lr", "mlp", "gbdt_fed"), algorithms=("fedavg",), tasks=("mortality", "sepsis"),
                                                       run_id="p4"))
        self.assert_training_artifacts_identical(da, db)
        self.assertTrue((da / "checkpoints" / "c4s1__graphA__mortality__fedavg" / "final.pt").is_file())

    def test_personalised_methods(self):
        da, db = self.run_both(lambda c, d: fed.run_fl_personalized(c, self.ds, self.fl, d, tasks=("mortality",), run_id="pp"))
        self.assert_training_artifacts_identical(da, db)

    def test_tuning_never_sees_test_labels(self):
        from src.three_modality.fl_tune import run_fl_tuning
        import json
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            kw = dict(families=("lr", "gbdt"), algorithms=("fedavg", "fedprox"), tasks=("mortality",), quick=True)
            ra = run_fl_tuning(self.cohort, self.ds, self.fl, a, run_id="t", **kw)
            rb = run_fl_tuning(self.poisoned, self.ds, self.fl, b, run_id="t", **kw)
            self.assertEqual(ra["best_hparams"], rb["best_hparams"])
            ta = (Path(ra["run_dir"]) / "tuning_table.csv").read_text()
            tb = (Path(rb["run_dir"]) / "tuning_table.csv").read_text()
            strip = lambda t: [",".join(l.split(",")[:6]) for l in t.splitlines()]
            self.assertEqual(strip(ta), strip(tb))
            self.assertIn("val_auroc_hist", ta)


class TestFrozenInputsAreNeverModified(Base):
    def test_cohort_arrays_are_unchanged_by_training_and_evaluation(self):
        before = digest(self.cohort)
        with tempfile.TemporaryDirectory() as d:
            fed.run_fl(1, self.cohort, self.ds, self.fl, d, classifiers=("lr",), algorithms=("fedavg",), tasks=("mortality",), run_id="x")
            fed.run_fl(2, self.cohort, self.ds, self.fl, d, graphs=("A",), algorithms=("fedavg",), tasks=("mortality",), run_id="y")
        self.assertEqual(digest(self.cohort), before)


if __name__ == "__main__":
    unittest.main()
