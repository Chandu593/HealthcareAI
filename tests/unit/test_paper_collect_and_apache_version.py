import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))

from paper_collect_results import collect, to_markdown  # noqa: E402
from src.paper.subgroups import subgroup_report  # noqa: E402


def rec(arm, seed, auroc, local=True):
    m = lambda a: {"auroc": a, "auprc": 0.4, "brier": 0.07, "ece": 0.005, "n": 100}
    r = {"arm": arm, "seed": seed, "test": m(auroc), "val": m(auroc - 0.003), "test_ci": {"auroc": {"lo": auroc - 0.01, "hi": auroc + 0.01}}, "test_per_hospital": {"macro_auroc": auroc - 0.002}}
    if local:
        r["local_calibrated"] = {"test": m(auroc + 0.007)}
    return r


class TestCollectAndApacheVersion(unittest.TestCase):
    def test_collect_handles_old_and_new_runs_and_multiple_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            for run, local in (("old_run", False), ("new_run", True)):
                (Path(d) / run).mkdir()
                blk = {"n_features": 696, "records": [rec("fedavg_mlp", 42, 0.84, local), rec("fedavg_mlp", 43, 0.85, local), rec("lightgbm_central", 42, 0.88, local)]}
                (Path(d) / run / "results.json").write_text(json.dumps({"experiments": {"mortality/no_apache_scores": blk}}), encoding="utf-8")
            rows = collect(d); self.assertEqual(len(rows), 4)
            fm = [r for r in rows if r["run"] == "new_run" and r["arm"] == "fedavg_mlp"][0]
            self.assertEqual(fm["seeds"], 2); self.assertAlmostEqual(fm["test_auroc"], 0.845); self.assertAlmostEqual(fm["auroc_local_cal"], 0.852)
            self.assertIsNone([r for r in rows if r["run"] == "old_run"][0]["auroc_local_cal"])
            md = to_markdown(rows); self.assertIn("n/a", md); self.assertIn("0.8450±0.0050", md)

    def test_apache_benchmark_prefers_iva_row(self):
        rng = np.random.default_rng(0); n = 2000; y = (rng.random(n) < 0.1).astype(float); p = np.clip(0.1 + 0.3 * y + rng.normal(0, .15, n), .001, .999)
        pred = {"split": np.array(["test"] * n), "patient_id": np.arange(n) + 1, "hospital_id": np.ones(n, int), "y": y, "p_calibrated": p}
        pat = pd.DataFrame({"patientunitstayid": np.arange(n) + 1, "unitdischargeoffset": 3000, "age": "60", "unittype": "MICU"})
        good = np.clip(y * 0.5 + rng.normal(0, .2, n), 0, 1); bad = rng.random(n)                       # IVa is informative; the other version is noise
        apache = pd.concat([pd.DataFrame({"patientunitstayid": np.arange(n) + 1, "predictedhospitalmortality": bad, "apacheversion": "IV"}),    # listed FIRST
                            pd.DataFrame({"patientunitstayid": np.arange(n) + 1, "predictedhospitalmortality": good, "apacheversion": "IVa"})], ignore_index=True)
        rep = subgroup_report(pred, pat, apache, 0.2, 1440, min_n=50, min_pos=5)
        self.assertGreater(rep["vs_apache_same_patients"]["apache_auroc"], 0.8)                          # would be ~0.5 if the first row were taken


if __name__ == "__main__":
    unittest.main()
