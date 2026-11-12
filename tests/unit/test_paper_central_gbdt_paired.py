"""Tests for Fed-THOR_results_package/reproduce/paired_central_gbdt.py (federated GBDT against the same code pooled)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Fed-THOR_results_package" / "reproduce"))
import paired_central_gbdt as pcg  # noqa: E402
import paired_flicsscale as pf  # noqa: E402

EXP = "mortality/no_apache_scores"
PARAMS = {"max_depth": 6, "learning_rate": 0.08, "reg_lambda": 10.0}


def _save(root, run, arm, seed, d):
    f = Path(root) / run / "predictions"; f.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f / f"{EXP.replace('/', '__')}__{arm}__seed{seed}.npz", patient_id=d["patient_id"], hospital_id=d["hospital_id"], split=d["split"].astype("U5"), y=d["y"], score=d["score"])


def _results(root, run, arm, params):
    (Path(root) / run).mkdir(parents=True, exist_ok=True)
    (Path(root) / run / "results.json").write_text(json.dumps({"experiments": {EXP: {"records": [{"arm": arm, "seed": sd, "extra": {"params": params}} for sd in (42, 43, 44)]}}}), encoding="utf-8")


def make_world(root, central_params=None, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    base = {"patient_id": np.arange(100, 100 + n), "hospital_id": np.repeat(np.arange(5), n // 5), "split": rng.choice(np.array(["train", "val", "test"]), size=n, p=[0.5, 0.2, 0.3]), "y": (rng.random(n) < 0.2).astype(float)}
    noise = {k: rng.normal(size=n) for k in ("fed", "cen", "lgb")}
    gain = {"fed": 1.30, "cen": 1.30, "lgb": 1.38}
    for sd in (42, 43, 44):
        _save(root, "fedrun", "fed_gbdt", sd, dict(base, score=gain["fed"] * base["y"] + noise["fed"] * (1 + 0.01 * (sd - 42))))
        _save(root, "cenrun", "gbdt_central", sd, dict(base, score=gain["cen"] * base["y"] + noise["cen"] * (1 + 0.01 * (sd - 42))))
    _save(root, "lgbrun", "lightgbm_central", 42, dict(base, score=gain["lgb"] * base["y"] + noise["lgb"]))
    _results(root, "fedrun", "fed_gbdt", PARAMS); _results(root, "cenrun", "gbdt_central", central_params or PARAMS)
    return base


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self._settings, self._n = pcg.SETTINGS, pf.N_BOOT
        pcg.SETTINGS = {"one": dict(label="Setting one", exp=EXP, fed="fedrun", central="cenrun", lgbm="lgbrun"),
                        "two": dict(label="Setting two", exp=EXP, fed="fedrun", central="missing_central_run", lgbm="lgbrun")}
        pf.N_BOOT = 50

    def tearDown(self):
        pcg.SETTINGS, pf.N_BOOT = self._settings, self._n
        self.tmp.cleanup()


class TestPairedCentralGbdt(Base):
    def test_structure_values_and_a_missing_setting_is_listed_not_invented(self):
        base = make_world(self.root); js = pcg.build(self.root)
        self.assertEqual(list(js["settings"]), ["one"]); self.assertEqual(js["skipped"], ["two"])
        r = js["settings"]["one"]; te = base["split"] == "test"
        self.assertEqual(r["n_test"], int(te.sum())); self.assertEqual(len(r["fed_minus_central_per_seed"]), 3)
        for key, d in r["paired"].items():
            a, b = key.split(" - ")
            self.assertAlmostEqual(d["diff"], r["auroc_seed42"][a] - r["auroc_seed42"][b], places=12)

    def test_identical_predictions_give_exactly_zero_federation_effect(self):
        make_world(self.root)
        for sd in (42, 43, 44):                                              # make the pooled model's scores identical to the federated one's
            f = np.load(self.root / "fedrun" / "predictions" / f"{EXP.replace('/', '__')}__fed_gbdt__seed{sd}.npz"); d = {k: f[k] for k in f.files}
            _save(self.root, "cenrun", "gbdt_central", sd, d)
        r = pcg.build(self.root)["settings"]["one"]
        d = r["paired"]["fed - cen"]; self.assertEqual((d["diff"], d["lo"], d["hi"]), (0.0, 0.0, 0.0)); self.assertEqual(r["fed_minus_central_per_seed"], [0.0, 0.0, 0.0])

    def test_deterministic(self):
        make_world(self.root)
        self.assertEqual(json.dumps(pcg.build(self.root), sort_keys=True), json.dumps(pcg.build(self.root), sort_keys=True))

    def test_different_hyperparameters_are_refused(self):
        make_world(self.root, central_params=dict(PARAMS, max_depth=8))
        with self.assertRaises(pf.PredictionsMismatch) as cm: pcg.build(self.root)
        self.assertIn("hyperparameters", str(cm.exception))

    def test_other_patients_are_refused(self):
        make_world(self.root)
        f = self.root / "lgbrun" / "predictions" / f"{EXP.replace('/', '__')}__lightgbm_central__seed42.npz"
        z = {k: v for k, v in np.load(f).items()}; order = np.random.default_rng(3).permutation(len(z["y"]))
        _save(self.root, "lgbrun", "lightgbm_central", 42, {k: v[order] for k, v in z.items()})
        with self.assertRaises(pf.PredictionsMismatch): pcg.build(self.root)

    def test_no_pooled_run_at_all_is_an_error(self):
        make_world(self.root); (self.root / "cenrun" / "results.json").unlink()
        with self.assertRaises(FileNotFoundError): pcg.build(self.root)

    def test_table_shape_and_main(self):
        make_world(self.root)
        h, rows, cap, label = pcg.table_from_json(pcg.build(self.root))
        self.assertTrue(all(len(r) == len(h) for r in rows)); self.assertIn("Settings not yet run: two", cap)
        out = self.root / "o" / "x.json"; self.assertEqual(pcg.main(["--paper-root", str(self.root), "--out", str(out)]), 0)
        self.assertEqual(list(json.loads(out.read_text(encoding="utf-8"))["settings"]), ["one"])


if __name__ == "__main__":
    unittest.main()
