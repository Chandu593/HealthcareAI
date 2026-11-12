import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))

from paper_label_variant_check import check, death_ids, paired_label_difference  # noqa: E402
from src.paper.evaluation import fast_auroc  # noqa: E402


class TestLabelVariantCheck(unittest.TestCase):
    def test_death_ids_reads_both_status_columns_case_insensitively(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "patient.csv"
            pd.DataFrame({"patientunitstayid": [1, 2, 3, 4], "unitdischargestatus": ["Expired", "Alive", "alive", " EXPIRED "],
                          "hospitaldischargestatus": ["Expired", "Expired", "Alive", "Expired"]}).to_csv(p, index=False)
            icu, hosp = death_ids(p)
            self.assertEqual(icu, {1, 4}); self.assertEqual(hosp, {1, 2, 4})

    def test_icu_label_is_a_different_label_and_only_test_rows_count(self):
        rng = np.random.default_rng(0); n = 8000
        hosp = (rng.random(n) < 0.10).astype(int)
        icu = hosp * (rng.random(n) < 0.6)                                        # ICU deaths are a subset of hospital deaths
        s = 2.0 * hosp + rng.normal(0, 1.0, n)                                    # a score informative about hospital death
        split = np.array(["train"] * 4000 + ["test"] * 4000)
        pred = {"split": split, "patient_id": np.arange(n) + 1, "score": s}
        r = check(pred, set((np.where(icu == 1)[0] + 1).tolist()), set((np.where(hosp == 1)[0] + 1).tolist()), boot_n=50)
        self.assertEqual(r["n_test"], 4000)
        self.assertLess(r["icu_prevalence"], r["hospital_prevalence"])           # a subset can never be more prevalent
        self.assertAlmostEqual(r["share_of_hospital_deaths_after_icu_discharge"], 0.4, delta=0.06)
        self.assertGreater(r["auroc_vs_hospital_death"]["point"], 0.85)
        # the ICU label must really be used: its AUROC differs from the hospital one (the score only carries ICU deaths through the hospital label)
        self.assertGreater(abs(r["auroc_vs_hospital_death"]["point"] - r["auroc_vs_icu_death"]["point"]), 0.01)
        self.assertAlmostEqual(r["auroc_vs_hospital_death"]["point"], r["auroc_vs_hospital_death_check"], places=3)

    def test_cli_end_to_end_prints_both_labels(self):
        rng = np.random.default_rng(1); n = 3000
        hosp = (rng.random(n) < 0.12).astype(int); icu = hosp * (rng.random(n) < 0.7)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d / "run" / "predictions").mkdir(parents=True)
            pd.DataFrame({"patientunitstayid": np.arange(n) + 1, "unitdischargestatus": np.where(icu == 1, "Expired", "Alive"),
                          "hospitaldischargestatus": np.where(hosp == 1, "Expired", "Alive")}).to_csv(d / "patient.csv", index=False)
            np.savez(d / "run" / "predictions" / "mortality__no_apache_scores__fed_gbdt__seed42.npz", split=np.array(["test"] * n), patient_id=np.arange(n) + 1,
                     score=2.0 * hosp + rng.normal(0, 1, n), y=hosp)
            r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "paper_label_variant_check.py"), "--run-dir", str(d / "run"), "--experiment", "mortality__no_apache_scores",
                                "--arm", "fed_gbdt", "--data-root", str(d)], capture_output=True, text=True, cwd=str(REPO_ROOT))
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        self.assertIn("ICU-discharge mortality", r.stdout); self.assertIn("AFTER ICU discharge", r.stdout); self.assertIn("FLICS's label", r.stdout)


def _world(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    hosp = (rng.random(n) < 0.10).astype(int); icu = hosp * (rng.random(n) < 0.55)
    s = 1.6 * hosp + rng.normal(0, 1.0, n)
    return hosp, icu, s


class TestPairedLabelDifference(unittest.TestCase):
    def test_off_by_default_and_the_existing_result_is_untouched(self):
        hosp, icu, s = _world(); n = len(s)
        pred = {"split": np.array(["test"] * n), "patient_id": np.arange(n) + 1, "score": s}
        ids = lambda v: set((np.where(v == 1)[0] + 1).tolist())
        r = check(pred, ids(icu), ids(hosp), boot_n=40)
        self.assertNotIn("auroc_difference_icu_minus_hospital", r)
        r2 = check(pred, ids(icu), ids(hosp), boot_n=40, paired_boot_n=40)
        for k in r: self.assertEqual(r[k], r2[k], k)                               # asking for the paired interval changes nothing else

    def test_point_is_the_difference_of_the_two_aurocs_and_the_interval_is_deterministic(self):
        hosp, icu, s = _world()
        d = paired_label_difference(hosp, icu, s, 120)
        self.assertAlmostEqual(d["point"], fast_auroc(icu, s) - fast_auroc(hosp, s), places=12)
        self.assertGreater(abs(d["point"]), 0.01)                                  # the two labels really are different (the sign is a property of the data, not asserted)
        self.assertLess(d["lo"], d["hi"]); self.assertEqual(d["n_resamples"], 120)
        self.assertEqual(d, paired_label_difference(hosp, icu, s, 120))

    def test_identical_labels_give_exactly_zero_which_needs_one_shared_resample(self):
        hosp, _, s = _world()
        d = paired_label_difference(hosp, hosp.copy(), s, 80)
        self.assertEqual((d["point"], d["lo"], d["hi"]), (0.0, 0.0, 0.0))

    def test_degenerate_label_returns_nan_not_an_error(self):
        hosp, _, s = _world(n=400)
        d = paired_label_difference(hosp, np.zeros_like(hosp), s, 30)               # no ICU deaths at all
        self.assertTrue(np.isnan(d["point"]) and d["n_resamples"] == 0)

    def test_cli_prints_the_paired_line_only_when_asked(self):
        hosp, icu, s = _world(n=2500, seed=5); n = len(s)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d / "run" / "predictions").mkdir(parents=True)
            pd.DataFrame({"patientunitstayid": np.arange(n) + 1, "unitdischargestatus": np.where(icu == 1, "Expired", "Alive"),
                          "hospitaldischargestatus": np.where(hosp == 1, "Expired", "Alive")}).to_csv(d / "patient.csv", index=False)
            np.savez(d / "run" / "predictions" / "mortality__no_apache_scores__fed_gbdt__seed42.npz", split=np.array(["test"] * n), patient_id=np.arange(n) + 1, score=s, y=hosp)
            cmd = [sys.executable, str(REPO_ROOT / "scripts" / "paper_label_variant_check.py"), "--run-dir", str(d / "run"), "--experiment", "mortality__no_apache_scores", "--arm", "fed_gbdt", "--data-root", str(d)]
            plain = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
            paired = subprocess.run(cmd + ["--paired-bootstrap", "40"], capture_output=True, text=True, cwd=str(REPO_ROOT))
        self.assertEqual((plain.returncode, paired.returncode), (0, 0), plain.stderr[-300:] + paired.stderr[-300:])
        self.assertNotIn("paired bootstrap", plain.stdout); self.assertIn("paired bootstrap over test patients, 40 resamples", paired.stdout)
        self.assertEqual(plain.stdout.strip().split("\n"), paired.stdout.strip().split("\n")[:4])   # the four original lines are identical either way


if __name__ == "__main__":
    unittest.main()
