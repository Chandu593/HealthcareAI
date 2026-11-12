"""Hospital-size cohort filter, reference-row corrections, and default-off wiring for the cohort-filter and fed_gbdt additions."""

import re
import unittest
from pathlib import Path

import numpy as np

from src.paper.cohort_filters import combine_filters, hospital_size_mask, hospital_size_profile
from src.paper.config import PAPER_DEFAULTS, get_paper_cfg

ROOT = Path(__file__).resolve().parents[2]


class TestHospitalSizeFilter(unittest.TestCase):
    def test_off_by_default_and_for_nonpositive_thresholds(self):
        self.assertEqual(PAPER_DEFAULTS["cohort_filter"]["min_hospital_patients"], 0)
        self.assertIsNone(hospital_size_mask([1, 1, 2], 0)); self.assertIsNone(hospital_size_mask([1, 1, 2], None)); self.assertIsNone(hospital_size_mask([1, 1, 2], -3))

    def test_keeps_exactly_hospitals_at_or_above_the_threshold(self):
        hid = np.array([10] * 5 + [20] * 3 + [30] * 8 + [40] * 1)
        mask, info = hospital_size_mask(hid, 5)
        self.assertEqual(sorted(np.unique(hid[mask]).tolist()), [10, 30])                  # >= is inclusive: the 5-patient hospital stays
        self.assertEqual((info["hospitals_total"], info["hospitals_kept"], info["patients_kept"], info["patients_total"]), (4, 2, 13, 17))
        self.assertEqual(info["kept_hospital_sizes"], {"min": 5, "median": 6.5, "max": 8})

    def test_threshold_above_every_hospital_keeps_nothing_without_crashing(self):
        mask, info = hospital_size_mask(np.array([1, 1, 2]), 99)
        self.assertFalse(mask.any()); self.assertIsNone(info["kept_hospital_sizes"])

    def test_single_filter_passes_through_unchanged_and_two_filters_are_anded(self):
        hid = np.array([1, 1, 1, 2, 2, 3])
        a = hospital_size_mask(hid, 2)
        m1, i1 = combine_filters([None, a], hid); self.assertIs(m1, a[0]); self.assertIs(i1, a[1])     # old sepsis-only behaviour is untouched
        other = (np.array([True, True, False, True, True, True]), {"rule": "x", "hospitals_total": 3, "hospitals_kept": 3, "patients_kept": 5, "patients_total": 6})
        m, i = combine_filters([other, a], hid)
        np.testing.assert_array_equal(m, np.array([True, True, False, True, True, False]))
        self.assertEqual((i["hospitals_kept"], i["patients_kept"]), (2, 4)); self.assertIn("hospital size >= 2", i["rule"]); self.assertIn("x", i["rule"])
        self.assertEqual(combine_filters([None, None], hid), (None, None))


class TestSizeProfile(unittest.TestCase):
    def test_profile_counts_and_target_threshold(self):
        hid = np.array([1] * 900 + [2] * 700 + [3] * 650 + [4] * 300 + [5] * 100)
        pr = hospital_size_profile(hid, [200, 600, 800], target_hospitals=3)
        self.assertEqual((pr["hospitals_total"], pr["patients_total"]), (5, 2650))
        self.assertEqual([(r["threshold"], r["hospitals"], r["patients"]) for r in pr["rows"]], [(200, 4, 2550), (600, 3, 2250), (800, 1, 900)])
        self.assertEqual(pr["target"], {"hospitals": 3, "threshold_for_target": 650, "patients_at_target": 2250})
        # the advertised threshold really keeps exactly that many hospitals
        self.assertEqual(hospital_size_mask(hid, 650)[1]["hospitals_kept"], 3)
        self.assertEqual(hospital_size_mask(hid, 651)[1]["hospitals_kept"], 2)

    def test_target_larger_than_available_is_reported_not_crashing(self):
        pr = hospital_size_profile(np.array([1, 1, 2]), [1], target_hospitals=5)
        self.assertIsNone(pr["target"]["threshold_for_target"])


class TestWiring(unittest.TestCase):
    def test_new_arm_is_off_by_default_and_registered(self):
        self.assertFalse(get_paper_cfg({})["arms"]["fed_gbdt"])
        src = (ROOT / "src" / "paper" / "runner.py").read_text(encoding="utf-8")
        known = re.search(r"KNOWN_ARMS = \((.*?)\)\n", src, re.S).group(1)
        self.assertIn('"fed_gbdt"', known)

    def test_user_config_overrides_deep_merge_into_the_new_stanzas(self):
        p = get_paper_cfg({"paper": {"cohort_filter": {"min_hospital_patients": 600}, "fed_gbdt": {"n_trials": 1}}})
        self.assertEqual(p["cohort_filter"]["min_hospital_patients"], 600)
        self.assertEqual(p["fed_gbdt"]["n_trials"], 1); self.assertEqual(p["fed_gbdt"]["n_bins"], 64)       # untouched keys keep their defaults

    def test_reference_rows_are_eicu_only_and_flics_only(self):
        """Only papers that use eICU alone may be compared. DP-IG-FL (MIMIC-III + eICU) and Kerkouche et al. (Premier Healthcare Database) are not."""
        refs = PAPER_DEFAULTS["reference_results"]
        self.assertTrue(all(r["eicu_only"] for r in refs))
        self.assertTrue(all(r["name"].startswith("FLICS") for r in refs))
        by = {r["name"]: r["value"] for r in refs}
        self.assertEqual(sorted(v for n, v in by.items() if n.startswith("FLICS")), [0.84, 0.87])
        blob = " ".join(r["name"] + " " + r["source"] for r in refs).lower()
        for banned in ("dp-ig-fl", "differentially", "kerkouche", "premier", "mimic", "bandwidth"):
            self.assertNotIn(banned, blob)
        self.assertFalse(any("(eICU)" in r["name"] for r in refs))


if __name__ == "__main__":
    unittest.main()
