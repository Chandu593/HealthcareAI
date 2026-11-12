import unittest

import pandas as pd

from src.data.cohort.infection_proxy import InfectionProxyConfig, combine_medication_sources, compute_infection_proxy


class TestInfectionProxy(unittest.TestCase):
    def setUp(self):
        self.config = InfectionProxyConfig(
            corroboration_window_minutes=1440, lookback_minutes=1440, lookahead_minutes=2880,
        )

    def test_antibiotic_alone_is_broad_but_not_strict(self):
        medication = pd.DataFrame([
            {"patientunitstayid": 1, "drugstartoffset": 100, "drugname": "VANCOMYCIN 1000 MG IV SOLN"},
        ])
        micro_lab = pd.DataFrame(columns=["patientunitstayid", "culturetakenoffset"])

        result = compute_infection_proxy(medication, micro_lab, self.config)
        row = result[result["patientunitstayid"] == 1].iloc[0]
        self.assertTrue(row["broad_infection"])
        self.assertFalse(row["strict_infection"])

    def test_antibiotic_with_nearby_culture_is_strict(self):
        medication = pd.DataFrame([
            {"patientunitstayid": 1, "drugstartoffset": 100, "drugname": "VANCOMYCIN 1000 MG IV SOLN"},
        ])
        micro_lab = pd.DataFrame([
            {"patientunitstayid": 1, "culturetakenoffset": 90},  # within 1440 min window
        ])

        result = compute_infection_proxy(medication, micro_lab, self.config)
        row = result[result["patientunitstayid"] == 1].iloc[0]
        self.assertTrue(row["broad_infection"])
        self.assertTrue(row["strict_infection"])

    def test_culture_far_outside_window_does_not_corroborate(self):
        medication = pd.DataFrame([
            {"patientunitstayid": 1, "drugstartoffset": 100, "drugname": "VANCOMYCIN 1000 MG IV SOLN"},
        ])
        micro_lab = pd.DataFrame([
            {"patientunitstayid": 1, "culturetakenoffset": 100 + 10000},  # way outside window
        ])
        result = compute_infection_proxy(medication, micro_lab, self.config)
        row = result[result["patientunitstayid"] == 1].iloc[0]
        self.assertTrue(row["broad_infection"])
        self.assertFalse(row["strict_infection"])

    def test_non_antibiotic_medication_does_not_trigger_infection(self):
        medication = pd.DataFrame([
            {"patientunitstayid": 1, "drugstartoffset": 100, "drugname": "METOPROLOL TARTRATE 25 MG PO TABS"},
        ])
        micro_lab = pd.DataFrame(columns=["patientunitstayid", "culturetakenoffset"])
        result = compute_infection_proxy(medication, micro_lab, self.config)
        row = result[result["patientunitstayid"] == 1].iloc[0]
        self.assertFalse(row["broad_infection"])
        self.assertFalse(row["strict_infection"])

    def test_strict_is_always_subset_of_broad(self):
        medication = pd.DataFrame([
            {"patientunitstayid": i, "drugstartoffset": 100, "drugname": "MEROPENEM 500 MG IV SOLN"}
            for i in range(1, 6)
        ])
        micro_lab = pd.DataFrame([
            {"patientunitstayid": 1, "culturetakenoffset": 100},
            {"patientunitstayid": 3, "culturetakenoffset": 50000},  # outside window
        ])
        result = compute_infection_proxy(medication, micro_lab, self.config)
        strict_patients = set(result[result["strict_infection"]]["patientunitstayid"])
        broad_patients = set(result[result["broad_infection"]]["patientunitstayid"])
        self.assertTrue(strict_patients.issubset(broad_patients))


class TestCombineMedicationSources(unittest.TestCase):
    def test_exact_duplicate_across_tables_is_deduped(self):
        medication = pd.DataFrame([
            {"patientunitstayid": 1, "drugname": "VANCOMYCIN 1000 MG IV SOLN",
             "drugstartoffset": 100, "drugstopoffset": 500},
        ])
        medication_imputed = pd.DataFrame([
            {"patientunitstayid": 1, "drugname": "VANCOMYCIN 1000 MG IV SOLN",
             "drugstartoffset": 100, "drugstopoffset": 500},  # exact duplicate of the above
        ])
        combined = combine_medication_sources({"medication": medication, "medication_imputed": medication_imputed})
        self.assertEqual(len(combined), 1)

    def test_distinct_rows_across_tables_are_both_kept(self):
        medication = pd.DataFrame([
            {"patientunitstayid": 1, "drugname": "VANCOMYCIN 1000 MG IV SOLN",
             "drugstartoffset": 100, "drugstopoffset": 500},
        ])
        medication_imputed1 = pd.DataFrame([
            {"patientunitstayid": 1, "drugname": "MEROPENEM 500 MG IV SOLN",
             "drugstartoffset": 300, "drugstopoffset": 700},  # different drug/time -- genuinely distinct
        ])
        combined = combine_medication_sources({"medication": medication, "medication_imputed1": medication_imputed1})
        self.assertEqual(len(combined), 2)

    def test_missing_tables_are_skipped_gracefully(self):
        combined = combine_medication_sources({"medication": pd.DataFrame(
            [{"patientunitstayid": 1, "drugname": "x", "drugstartoffset": 1, "drugstopoffset": 2}]
        ), "medication_imputed": None, "medication_imputed1": pd.DataFrame()})
        self.assertEqual(len(combined), 1)

    def test_empty_input_returns_empty_frame_not_error(self):
        combined = combine_medication_sources({})
        self.assertEqual(len(combined), 0)


if __name__ == "__main__":
    unittest.main()
