import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))

from paper_documented_sepsis_check import check, documented_ids  # noqa: E402


class TestDocumentedSepsisCheck(unittest.TestCase):
    def test_documented_ids_text_and_icd(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "diagnosis.csv"
            pd.DataFrame({"patientunitstayid": [1, 2, 3, 4, 5], "diagnosisstring": ["infectious|sepsis", "cardiac|chf", "x", "SEPTIC shock", "y"],
                          "icd9code": ["", "", "995.91, A41.9", "", "428.0"]}).to_csv(p, index=False)
            self.assertEqual(documented_ids(p), {1, 3, 4})

    def test_informative_score_ranks_documented_cases_and_only_test_rows_count(self):
        rng = np.random.default_rng(0); n = 6000
        doc = (rng.random(n) < 0.15).astype(int); proxy = np.where(rng.random(n) < 0.5, doc, (rng.random(n) < 0.15).astype(int))
        p = np.clip(0.2 + 0.35 * doc + rng.normal(0, 0.15, n), 0.001, 0.999)
        split = np.array(["train"] * 3000 + ["test"] * 3000)
        pred = {"split": split, "patient_id": np.arange(n) + 1, "y": proxy, "p_calibrated": p}
        r = check(pred, set((np.where(doc == 1)[0] + 1).tolist()), boot_n=50)
        self.assertEqual(r["n_test"], 3000); self.assertGreater(r["auroc_vs_documented"]["point"], 0.8)
        q = r["documented_rate_by_risk_quintile"]; self.assertGreater(q[-1], q[0] + 0.1)            # documented rate rises with predicted risk
        pred2 = dict(pred, p_calibrated=np.random.default_rng(1).random(n))                          # a useless score must sit near 0.5
        self.assertLess(abs(check(pred2, set((np.where(doc == 1)[0] + 1).tolist()), boot_n=50)["auroc_vs_documented"]["point"] - 0.5), 0.06)


if __name__ == "__main__":
    unittest.main()
