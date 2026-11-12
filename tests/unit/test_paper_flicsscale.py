"""Tests for Fed-THOR_results_package/reproduce/paired_flicsscale.py (FLICS-scale paired comparisons).

The script reads saved predictions only, so the tests build small synthetic prediction folders with the real file layout. The guards are the point: each one
stops a comparison that would otherwise print plausible, wrong numbers (a predictions folder copied from another run was the real case that motivated them).
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Fed-THOR_results_package" / "reproduce"))
import paired_flicsscale as pf  # noqa: E402

N_HOSP, PER_HOSP = 6, 200
FEDERATIONS = {"flicsstyle_top3": (0, 1, 2), "flicsstyle_top10": (0, 1, 2, 3, 4)}     # folder names are fixed by the script; sizes here are small on purpose
SIGNAL = {"fed_gbdt": 1.4, "fedavg_mlp": 1.2, "fedavg_logreg": 1.1, "lightgbm_central": 1.5, "mlp_central": 1.2, "logreg_central": 1.0, "mlp_local_only": 0.8}


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def _save(folder, arm, seed, d):
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / (pf.NAME % (arm, seed)), patient_id=d["patient_id"], hospital_id=d["hospital_id"], split=d["split"].astype("U5"), y=d["y"], score=d["score"])


def make_world(root, seed=0):
    """root/outputs-paper layout: two federation runs (subsets of hospitals), plus the all-hospital runs the training-pool contrast reads."""
    rng = np.random.default_rng(seed)
    n = N_HOSP * PER_HOSP
    pid = np.arange(5000, 5000 + n)
    hosp = np.repeat(np.arange(N_HOSP), PER_HOSP)
    split = rng.choice(np.array(["train", "val", "test"]), size=n, p=[0.5, 0.2, 0.3])
    y = (rng.random(n) < 0.2).astype(float)
    base = {"patient_id": pid, "hospital_id": hosp, "split": split, "y": y}
    noise = {(arm, sd): rng.normal(size=n) for arm in pf.ARMS for sd in pf.SEEDS}
    sc = lambda arm, sd, gain=0.0: SIGNAL[arm] * y + gain * y + noise[(arm, sd)]
    for run, hs in FEDERATIONS.items():
        keep = np.isin(hosp, hs)
        sub = {k: v[keep] for k, v in base.items()}
        recs, agg = [], {}
        for arm in pf.ARMS:
            seeds = (42,) if arm == "logreg_central" else pf.SEEDS
            vals = []
            for sd in seeds:
                _save(root / run / "predictions", arm, sd, dict(sub, score=sc(arm, sd)[keep]))
                te = sub["split"] == "test"
                a = _auc(sub["y"][te], sc(arm, sd)[keep][te]); vals.append(a)
                recs.append({"arm": arm, "seed": sd, "val": {"auroc": a - 0.01}, "test_per_hospital": {"macro_auroc": a - 0.005}})
            agg[arm] = {"n_seeds": len(seeds), "auroc": {"mean": float(np.mean(vals)), "std": float(np.std(vals))}, "auprc": {"mean": 0.4},
                        "auroc_ci_seed0": {"lo": vals[0] - 0.03, "hi": vals[0] + 0.03}}
        sizes = {s: int((sub["split"] == s).sum()) for s in ("train", "val", "test")}
        block = {"split_sizes": sizes, "cohort_filter": {"rule": f"hospital size >= x -> {len(hs)} hospitals", "hospitals_kept": len(hs), "patients_kept": int(keep.sum())},
                 "aggregate": agg, "records": recs}
        (root / run).mkdir(parents=True, exist_ok=True)
        (root / run / "results.json").write_text(json.dumps({"experiments": {pf.EXP: block}}), encoding="utf-8")
    for arm, (run, parm) in pf.POOL.items():                       # trained on every hospital: a little better than the federation-only model
        _save(root / run / "predictions", parm, 42, dict(base, score=sc(arm, 42, gain=0.4)))
    return base


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = make_world(self.root)
        self._n_boot = pf.N_BOOT
        pf.N_BOOT = 60                                             # small, the tests check logic not interval width

    def tearDown(self):
        pf.N_BOOT = self._n_boot
        self.tmp.cleanup()


class TestGuards(Base):
    def test_clean_world_runs_and_pairs_the_right_patients(self):
        r = pf.analyse_scale(self.root, "flicsstyle_top3")
        te = (self.base["split"] == "test") & np.isin(self.base["hospital_id"], FEDERATIONS["flicsstyle_top3"])
        self.assertEqual(r["n_test"], int(te.sum()))
        self.assertEqual(r["events"], int(self.base["y"][te].sum()))

    def test_copied_predictions_folder_is_rejected(self):
        """The real incident: the top-3 folder held the top-10 run's predictions. Split sizes disagree with its own results.json."""
        shutil.rmtree(self.root / "flicsstyle_top3" / "predictions")
        shutil.copytree(self.root / "flicsstyle_top10" / "predictions", self.root / "flicsstyle_top3" / "predictions")
        with self.assertRaises(pf.PredictionsMismatch) as cm:
            pf.analyse_scale(self.root, "flicsstyle_top3")
        self.assertIn("split sizes", str(cm.exception))

    def test_arm_with_different_patient_order_is_rejected(self):
        f = self.root / "flicsstyle_top3" / "predictions" / (pf.NAME % ("fedavg_mlp", 42))
        z = {k: v for k, v in np.load(f).items()}
        order = np.random.default_rng(1).permutation(len(z["y"]))
        _save(self.root / "flicsstyle_top3" / "predictions", "fedavg_mlp", 42, {k: v[order] for k, v in z.items()})
        with self.assertRaises(pf.PredictionsMismatch) as cm:
            pf.analyse_scale(self.root, "flicsstyle_top3")
        self.assertIn("same patients", str(cm.exception))

    def test_arm_with_different_labels_is_rejected(self):
        f = self.root / "flicsstyle_top3" / "predictions" / (pf.NAME % ("lightgbm_central", 42))
        z = {k: v for k, v in np.load(f).items()}
        z["y"] = z["y"].copy(); z["y"][0] = 1.0 - z["y"][0]
        _save(self.root / "flicsstyle_top3" / "predictions", "lightgbm_central", 42, z)
        with self.assertRaises(pf.PredictionsMismatch):
            pf.analyse_scale(self.root, "flicsstyle_top3")

    def test_missing_predictions_file_is_loud(self):
        (self.root / "flicsstyle_top10" / "predictions" / (pf.NAME % ("fed_gbdt", 43))).unlink()
        with self.assertRaises(FileNotFoundError):
            pf.analyse_scale(self.root, "flicsstyle_top10")

    def test_missing_results_json_is_loud(self):
        (self.root / "flicsstyle_top10" / "results.json").unlink()
        with self.assertRaises(FileNotFoundError):
            pf.analyse_scale(self.root, "flicsstyle_top10")


class TestTrainingPool(Base):
    def _shuffle_pool_file(self, run, arm):
        f = self.root / run / "predictions" / (pf.NAME % (arm, 42))
        z = {k: v for k, v in np.load(f).items()}
        order = np.random.default_rng(7).permutation(len(z["y"]))
        _save(self.root / run / "predictions", arm, 42, {k: v[order] for k, v in z.items()})

    def test_matching_is_by_patient_id_not_row_order(self):
        before = pf.analyse_scale(self.root, "flicsstyle_top3")["training_pool"]
        for arm, (run, parm) in pf.POOL.items():
            self._shuffle_pool_file(run, parm)
        after = pf.analyse_scale(self.root, "flicsstyle_top3")["training_pool"]
        self.assertEqual(before, after)
        self.assertGreater(before["fed_gbdt"]["all_minus_federation"]["diff"], 0)     # the all-hospital model was built a little better

    def test_patient_missing_from_the_all_hospital_run_is_rejected(self):
        run, parm = pf.POOL["fed_gbdt"]
        f = self.root / run / "predictions" / (pf.NAME % (parm, 42))
        z = {k: v for k, v in np.load(f).items()}
        te_in_fed = np.flatnonzero((z["split"] == "test") & np.isin(z["hospital_id"], FEDERATIONS["flicsstyle_top3"]))[0]
        keep = np.arange(len(z["y"])) != te_in_fed
        _save(self.root / run / "predictions", parm, 42, {k: v[keep] for k, v in z.items()})
        with self.assertRaises(pf.PredictionsMismatch) as cm:
            pf.analyse_scale(self.root, "flicsstyle_top3")
        self.assertIn("absent", str(cm.exception))

    def test_label_disagreement_with_the_all_hospital_run_is_rejected(self):
        run, parm = pf.POOL["fedavg_mlp"]
        f = self.root / run / "predictions" / (pf.NAME % (parm, 42))
        z = {k: v for k, v in np.load(f).items()}
        j = np.flatnonzero((z["split"] == "test") & np.isin(z["hospital_id"], FEDERATIONS["flicsstyle_top3"]))[0]
        z["y"] = z["y"].copy(); z["y"][j] = 1.0 - z["y"][j]
        _save(self.root / run / "predictions", parm, 42, z)
        with self.assertRaises(pf.PredictionsMismatch):
            pf.analyse_scale(self.root, "flicsstyle_top3")

    def test_duplicate_patient_ids_are_rejected(self):
        run, parm = pf.POOL["lightgbm_central"]
        f = self.root / run / "predictions" / (pf.NAME % (parm, 42))
        z = {k: v for k, v in np.load(f).items()}
        z["patient_id"] = z["patient_id"].copy(); z["patient_id"][1] = z["patient_id"][0]
        _save(self.root / run / "predictions", parm, 42, z)
        with self.assertRaises(pf.PredictionsMismatch):
            pf.analyse_scale(self.root, "flicsstyle_top3")


class TestPairedDifference(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.y = (rng.random(400) < 0.25).astype(float)
        self.s = {"good": 1.5 * self.y + rng.normal(size=400), "weak": 0.5 * self.y + rng.normal(size=400)}
        self.s["same"] = self.s["good"].copy()

    def test_identical_scores_give_exactly_zero(self):
        d = pf.paired_auc_diff(self.y, self.s, "good", "same", np.random.default_rng(0), n_boot=50)
        self.assertEqual((d["diff"], d["lo"], d["hi"]), (0.0, 0.0, 0.0))

    def test_point_difference_is_the_difference_of_aurocs_and_is_antisymmetric(self):
        d = pf.paired_auc_diff(self.y, self.s, "good", "weak", np.random.default_rng(0), n_boot=200)
        self.assertAlmostEqual(d["diff"], _auc(self.y, self.s["good"]) - _auc(self.y, self.s["weak"]), places=12)
        self.assertGreater(d["diff"], 0)
        self.assertLess(d["lo"], d["diff"] + 1e-9)
        self.assertGreater(d["hi"], d["diff"] - 1e-9)
        e = pf.paired_auc_diff(self.y, self.s, "weak", "good", np.random.default_rng(0), n_boot=200)
        self.assertAlmostEqual(e["diff"], -d["diff"], places=12)

    def test_same_seed_same_interval(self):
        a = pf.paired_auc_diff(self.y, self.s, "good", "weak", np.random.default_rng(5), n_boot=100)
        b = pf.paired_auc_diff(self.y, self.s, "good", "weak", np.random.default_rng(5), n_boot=100)
        self.assertEqual(a, b)


class TestBuildAndTables(Base):
    def test_build_is_deterministic_and_has_the_expected_structure(self):
        a = pf.build(self.root)
        b = pf.build(self.root)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertEqual(set(a["scales"]), {"top3", "top10"})
        s = a["scales"]["top3"]
        self.assertEqual(set(s["paired"]), {f"{x} - {y}" for x, y in pf.COMPARISONS})
        self.assertTrue(all(len(v) == 3 for v in s["per_seed_diff"].values()))
        self.assertNotIn("fedavg_logreg - logreg_central", s["per_seed_diff"])              # central LR has a single seed
        self.assertEqual(set(s["training_pool"]), set(pf.POOL))

    def test_tables_have_consistent_shapes_and_verdicts(self):
        js = pf.build(self.root)
        calls = []
        def verdict(ref, lo, hi):
            calls.append((ref, lo, hi)); return "level"
        refs = {"top3": [("FLICS, FedAvg LR / LSTM", 0.87)], "top10": [("FLICS, FedAvg LR", 0.83), ("FLICS, FedAvg LSTM", 0.84)]}
        t = pf.tables_from_json(js, refs, verdict)
        self.assertEqual(set(t), {"t11a", "t11b", "t11c"})
        for key, (header, rows, caption, label) in t.items():
            self.assertTrue(rows and caption and label, key)
            self.assertTrue(all(len(r) == len(header) for r in rows), key)
        self.assertEqual(len(t["t11a"][1]), 2 * len(pf.ARMS))
        self.assertEqual(len(calls), 3 * 1 + 3 * 2)                                       # 3 federated arms x (1 reference at 3 hospitals + 2 at 10)
        seeds_col = {r[0]: r[2] for r in t["t11b"][1]}
        self.assertEqual(seeds_col["Federated LR (FLICS's model class) - Central LR"], "–")
        self.assertRegex(seeds_col["Federated GBDT - Federated LR (FLICS's model class)"], r"^[0-3]/3$")

    def test_main_writes_the_json(self):
        out = self.root / "out" / "paired_flicsscale.json"
        self.assertEqual(pf.main(["--paper-root", str(self.root), "--out", str(out)]), 0)
        js = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(js["n_bootstrap"], 60)
        self.assertEqual(set(js["scales"]), {"top3", "top10"})


if __name__ == "__main__":
    unittest.main()
