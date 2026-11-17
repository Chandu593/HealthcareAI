"""Pipeline regression for hospitals with NO test or NO validation patients (real-data case): every FL config completes, the hospital still contributes its training data, is reported with the right
status / reason, and every saved prediction equals the (modified) frozen cohort's test patients. Each scenario uses its own output base: a run's identity does not include in-memory test edits."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.three_modality import federated as fed
from tests.unit._fl_cohort import build_fl_cohort, clone_cohort, fast_ds, fast_fl


def reassign(cohort, hospital, src, dst):
    c = clone_cohort(cohort)
    c.split = c.split.copy()
    c.split[(c.hospitalid == hospital) & (c.split == src)] = dst
    return c


def expected_from_cohort(c, hospital, task):
    m = (c.hospitalid == hospital) & (c.split == "test") & c.eligible(task)
    return c.patient_ids[m], c.labels[task][m]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "out" / "three_modality"
        cls.up, cls.cohort, _ = build_fl_cohort(cls.root)
        cls.ds, cls.fl = fast_ds(), fast_fl()
        # hospital 12: no TEST patients; hospital 13: no VALIDATION patients (their patients move to the neighbouring split)
        c = reassign(cls.cohort, 12, "test", "val")
        cls.coh = reassign(c, 13, "val", "test")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, res):
        return json.loads((Path(res["run_dir"]) / "per_hospital.json").read_text())

    def check(self, res, task="mortality"):
        rows = self.rows(res)
        for v in sorted({r["variant"] for r in rows}):
            vr = {r["hospitalid"]: r for r in rows if r["variant"] == v}
            self.assertEqual(set(vr), {11, 12, 13, 14}, f"{v}: every hospital must appear exactly once")
            self.assertTrue(all(r["participated_in_training"] for r in vr.values()), f"{v}: a hospital with an empty split must still be a federation client")
            self.assertEqual((vr[12]["status"], vr[12]["skip_reason"]), ("skipped", "no test patients"))
            self.assertEqual(vr[13]["status"], "trained")
            self.assertEqual(vr[13]["threshold_source"], "global_federated_validation_histogram", "no validation patients -> the global federated-histogram threshold")
            self.assertFalse(vr[13].get("operating_points"), "no recall-target point without validation patients")
        npz = np.load(Path(res["run_dir"]) / "predictions.npz", allow_pickle=True)
        names = sorted({k.rsplit("@", 1)[0] for k in npz.files})
        self.assertTrue(names)
        for v in names:
            pid, hid, y = npz[v + "@pid"], npz[v + "@hid"], npz[v + "@y"]
            self.assertNotIn(12, set(hid.tolist()), "no predictions may exist for a hospital without test patients")
            for h in np.unique(hid):
                ids, truth = expected_from_cohort(self.coh, h, task)
                m = hid == h
                np.testing.assert_array_equal(np.sort(pid[m]), np.sort(ids), f"{v} hospital {h}: test patients differ from the cohort")
                np.testing.assert_array_equal(y[m][np.argsort(pid[m])], truth[np.argsort(ids)])


class TestAllConfigsWithEmptySplits(Base):
    def test_config1_lr_mlp_gbdt(self):
        self.check(fed.run_fl(1, self.coh, self.ds, self.fl, self.root / "b1", classifiers=("lr", "mlp", "gbdt_fed"), algorithms=("fedavg", "scaffold"), tasks=("mortality",), run_id="e1"))

    def test_config2_and_config3_graphs(self):
        for n in (2, 3):
            self.check(fed.run_fl(n, self.coh, self.ds, self.fl, self.root / f"b{n}", graphs=("A", "B"), algorithms=("fedavg", "fedadam"), tasks=("mortality",), run_id=f"e{n}"))

    def test_config4_two_stage_reaches_stage_two_for_every_client(self):
        res = fed.run_fl(4, self.coh, self.ds, self.fl, self.root / "b4", graphs=("A", "B"), classifiers=("lr", "mlp", "gbdt_fed"), algorithms=("fedavg",), tasks=("mortality",),
                         c4_gcn_algorithm="fedadam", run_id="e4")
        self.check(res)
        d = Path(res["run_dir"])
        lg = json.loads((d / "training_logs" / "c4s1__graphA__mortality__fedadam.json").read_text())
        self.assertEqual(lg["n_clients"], 4)
        self.assertEqual(lg["failed_clients"], [])
        self.assertEqual(json.loads((d / "training_logs" / "f4__graphA__mortality__lr__fedavg.json").read_text())["n_clients"], 4)

    def test_personalised_methods(self):
        self.check(fed.run_fl_personalized(self.coh, self.ds, self.fl, self.root / "bp", tasks=("mortality",), run_id="ep"))

    def test_resuming_after_the_crash_point_reuses_the_finished_stage_one(self):
        """Config 4 crashed AFTER its (66 min on the real data) stage-1 GCN had finished: the stage-1 checkpoint must be reloaded, not retrained, on resume."""
        class Boom(Exception):
            pass
        calls = {"n": 0}
        base = self.root / "bres"
        kw = dict(graphs=("A",), classifiers=("lr",), algorithms=("fedavg",), tasks=("mortality",), c4_gcn_algorithm="fedadam")

        def hook(r):                                    # interrupt during the stage-2 LR training (after stage 1 completed and was checkpointed)
            calls["n"] += 1
            if calls["n"] > self.fl["rounds"]["max_rounds"] + 1:
                raise Boom()
        with self.assertRaises(Boom):
            fed.run_fl(4, self.coh, self.ds, self.fl, base, run_id="e4res", round_hook=hook, **kw)
        ck = base / "e4res" / "checkpoints" / "c4s1__graphA__mortality__fedadam" / "final.pt"
        self.assertTrue(ck.is_file())
        stamp = ck.stat().st_mtime_ns
        res = fed.run_fl(4, self.coh, self.ds, self.fl, base, run_id="e4res", resume=True, **kw)
        self.assertEqual(ck.stat().st_mtime_ns, stamp, "the finished stage-1 checkpoint must not be rewritten")
        lg = json.loads((Path(res["run_dir"]) / "training_logs" / "c4s1__graphA__mortality__fedadam.json").read_text())
        self.assertEqual(lg["resumed_from"], "final")
        self.check(res)


if __name__ == "__main__":
    unittest.main()
