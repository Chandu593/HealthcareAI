"""Tests for Fed-THOR_results_package/reproduce/paired_sepsis6h.py (paired 6 h GBDT comparisons from saved predictions)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Fed-THOR_results_package" / "reproduce"))
import paired_flicsscale as pf  # noqa: E402
import paired_sepsis6h as ps  # noqa: E402


def _save(root, run, arm, seed, d):
    f = Path(root) / run / "predictions"
    f.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f / (ps.G % (arm, seed)), patient_id=d["patient_id"], hospital_id=d["hospital_id"], split=d["split"].astype("U5"), y=d["y"], score=d["score"])


def make_world(root, seed=0, n=1500):
    rng = np.random.default_rng(seed)
    base = {"patient_id": np.arange(900, 900 + n), "hospital_id": np.repeat(np.arange(5), n // 5), "split": rng.choice(np.array(["train", "val", "test"]), size=n, p=[0.5, 0.25, 0.25]),
            "y": (rng.random(n) < 0.3).astype(float)}
    gain = {"rerun": 1.30, "first": 1.25, "mlp": 1.0, "lgbm": 1.35}
    for k, (run, arm) in ps.RUNS.items():
        for sd in ps.SEEDS if k in ("rerun", "first") else (42,):
            _save(root, run, arm, sd, dict(base, score=gain[k] * base["y"] + np.random.default_rng(100 * sd + len(k)).normal(size=n)))
    return base


class TestPairedSepsis6h(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = make_world(self.root)
        self._n = pf.N_BOOT
        pf.N_BOOT = 50

    def tearDown(self):
        pf.N_BOOT = self._n
        self.tmp.cleanup()

    def test_structure_and_point_differences(self):
        r = ps.compute(self.root)
        self.assertEqual(set(r["paired"]), {f"{a} - {b}" for a, b in ps.PAIRS})
        te = self.base["split"] == "test"
        self.assertEqual(r["n_test"], int(te.sum()))
        for key, d in r["paired"].items():
            a, b = key.split(" - ")
            self.assertAlmostEqual(d["diff"], r["auroc_seed42"][a] - r["auroc_seed42"][b], places=12)
        self.assertEqual(len(r["per_seed"]["rerun - first"]), 3)
        self.assertEqual(len(r["validation_auroc"]["rerun"]), 3)

    def test_deterministic(self):
        self.assertEqual(json.dumps(ps.compute(self.root), sort_keys=True), json.dumps(ps.compute(self.root), sort_keys=True))

    def test_run_with_other_patients_is_rejected(self):
        run, arm = ps.RUNS["mlp"]
        f = self.root / run / "predictions" / (ps.G % (arm, 42))
        z = {k: v for k, v in np.load(f).items()}
        order = np.random.default_rng(2).permutation(len(z["y"]))
        _save(self.root, run, arm, 42, {k: v[order] for k, v in z.items()})
        with self.assertRaises(pf.PredictionsMismatch):
            ps.compute(self.root)

    def test_seed_file_with_other_labels_is_rejected(self):
        run, arm = ps.RUNS["rerun"]
        f = self.root / run / "predictions" / (ps.G % (arm, 44))
        z = {k: v for k, v in np.load(f).items()}
        z["y"] = z["y"].copy(); z["y"][3] = 1.0 - z["y"][3]
        _save(self.root, run, arm, 44, z)
        with self.assertRaises(pf.PredictionsMismatch):
            ps.compute(self.root)

    def test_missing_file_is_loud(self):
        run, arm = ps.RUNS["first"]
        (self.root / run / "predictions" / (ps.G % (arm, 43))).unlink()
        with self.assertRaises(FileNotFoundError):
            ps.compute(self.root)


if __name__ == "__main__":
    unittest.main()
