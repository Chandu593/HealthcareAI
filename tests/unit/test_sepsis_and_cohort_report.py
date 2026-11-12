import unittest

import pandas as pd

from src.data.cohort.cohort_report import build_cohort_report
from src.data.cohort.sepsis_label import compute_sepsis_labels


class TestSepsisLabel(unittest.TestCase):
    def test_sepsis_requires_both_infection_and_organ_dysfunction(self):
        infection_df = pd.DataFrame([
            {"patientunitstayid": 1, "broad_infection": True, "broad_infection_offset": 100,
             "strict_infection": True, "strict_infection_offset": 100,
             "n_antibiotic_orders": 1, "n_cultures_ordered": 1},
            {"patientunitstayid": 2, "broad_infection": True, "broad_infection_offset": 200,
             "strict_infection": False, "strict_infection_offset": None,
             "n_antibiotic_orders": 1, "n_cultures_ordered": 0},
            {"patientunitstayid": 3, "broad_infection": False, "broad_infection_offset": None,
             "strict_infection": False, "strict_infection_offset": None,
             "n_antibiotic_orders": 0, "n_cultures_ordered": 0},
        ])
        sofa_df = pd.DataFrame([
            {"patientunitstayid": 1, "sofa_like_total": 4, "n_domains_observed": 6,
             "max_possible_score": 22, "organ_dysfunction": True},
            {"patientunitstayid": 2, "sofa_like_total": 0, "n_domains_observed": 6,
             "max_possible_score": 22, "organ_dysfunction": False},
            {"patientunitstayid": 3, "sofa_like_total": 5, "n_domains_observed": 6,
             "max_possible_score": 22, "organ_dysfunction": True},
        ])

        result = compute_sepsis_labels(infection_df, sofa_df)
        by_id = result.set_index("patientunitstayid")

        self.assertTrue(by_id.loc[1, "sepsis_broad"])
        self.assertTrue(by_id.loc[1, "sepsis_strict"])
        # patient 2: infected but no organ dysfunction -> not septic
        self.assertFalse(by_id.loc[2, "sepsis_broad"])
        # patient 3: organ dysfunction but no infection -> not septic
        self.assertFalse(by_id.loc[3, "sepsis_broad"])
        self.assertFalse(by_id.loc[3, "sepsis_strict"])

    def test_onset_offset_only_populated_for_septic_patients(self):
        infection_df = pd.DataFrame([
            {"patientunitstayid": 1, "broad_infection": True, "broad_infection_offset": 150,
             "strict_infection": False, "strict_infection_offset": None,
             "n_antibiotic_orders": 1, "n_cultures_ordered": 0},
        ])
        sofa_df = pd.DataFrame([
            {"patientunitstayid": 1, "sofa_like_total": 3, "n_domains_observed": 4,
             "max_possible_score": 16, "organ_dysfunction": True},
        ])
        result = compute_sepsis_labels(infection_df, sofa_df)
        row = result.iloc[0]
        self.assertEqual(row["sepsis_broad_onset_offset_approx"], 150)
        self.assertTrue(pd.isna(row["sepsis_strict_onset_offset_approx"]))

    def test_partial_coverage_regression_no_dtype_corruption(self):
        """Regression test for a real crash found running against real
        eICU data: patients present in infection_df (they have
        medication/microLab data) but ABSENT from sofa_df (no
        apachePredVar row at all -- a real, common situation; ~468/4064
        patients in one real run) must not corrupt organ_dysfunction's
        dtype. Before the fix, the outer merge introduced NaN into what
        was a bool column, pandas silently upcast it to `object` dtype,
        and .fillna(False) did not restore bool dtype -- `~` on the
        resulting object-dtype Python-bool column then computed bitwise
        invert (True->-2, False->-1) instead of logical NOT, which
        crashed downstream in build_cohort_report with a KeyError."""
        infection_df = pd.DataFrame([
            # patient 1: has infection data AND sofa data (normal case)
            {"patientunitstayid": 1, "broad_infection": True, "broad_infection_offset": 100,
             "strict_infection": True, "strict_infection_offset": 100,
             "n_antibiotic_orders": 1, "n_cultures_ordered": 1},
            # patient 2: has infection data but NO apachePredVar row at
            # all -> absent from sofa_df entirely (not merely NaN scores)
            {"patientunitstayid": 2, "broad_infection": True, "broad_infection_offset": 50,
             "strict_infection": False, "strict_infection_offset": None,
             "n_antibiotic_orders": 1, "n_cultures_ordered": 0},
        ])
        sofa_df = pd.DataFrame([
            {"patientunitstayid": 1, "sofa_like_total": 4, "n_domains_observed": 6,
             "max_possible_score": 22, "organ_dysfunction": True},
            # patient 3: has sofa data but no medication/microLab at all
            # -> absent from infection_df entirely
            {"patientunitstayid": 3, "sofa_like_total": 5, "n_domains_observed": 6,
             "max_possible_score": 22, "organ_dysfunction": True},
        ])

        result = compute_sepsis_labels(infection_df, sofa_df)

        # dtype must be genuine bool, not object -- this is the actual regression check
        self.assertEqual(result["organ_dysfunction"].dtype, bool)
        self.assertEqual(result["sepsis_broad"].dtype, bool)
        self.assertEqual(result["sepsis_strict"].dtype, bool)

        by_id = result.set_index("patientunitstayid")
        # patient 2 (no sofa data at all) must resolve to organ_dysfunction=False, not crash/NaN
        self.assertEqual(by_id.loc[2, "organ_dysfunction"], False)
        self.assertFalse(by_id.loc[2, "sepsis_broad"])
        # patient 3 (no infection data at all) must resolve to infection=False, not crash/NaN
        self.assertEqual(by_id.loc[3, "broad_infection"], False)
        self.assertFalse(by_id.loc[3, "sepsis_broad"])

        # The actual failure mode: ~ on the resulting columns must behave
        # as logical NOT (True/False), never bitwise invert (-1/-2).
        inverted = ~result["sepsis_broad"]
        self.assertTrue(set(inverted.unique()).issubset({True, False}))


class TestCohortReport(unittest.TestCase):
    def test_report_counts_and_overlap(self):
        patient_df = pd.DataFrame([
            {"patientunitstayid": i, "hospitalid": 1 if i <= 2 else 2} for i in range(1, 5)
        ])
        mortality = pd.Series([1, 0, 0, 0], index=patient_df.index)
        sepsis_df = pd.DataFrame([
            {"patientunitstayid": 1, "sepsis_broad": True, "sepsis_strict": True,
             "sepsis_broad_onset_offset_approx": 10, "sepsis_strict_onset_offset_approx": 10},
            {"patientunitstayid": 2, "sepsis_broad": True, "sepsis_strict": False,
             "sepsis_broad_onset_offset_approx": 20, "sepsis_strict_onset_offset_approx": None},
            {"patientunitstayid": 3, "sepsis_broad": False, "sepsis_strict": False,
             "sepsis_broad_onset_offset_approx": None, "sepsis_strict_onset_offset_approx": None},
            {"patientunitstayid": 4, "sepsis_broad": False, "sepsis_strict": False,
             "sepsis_broad_onset_offset_approx": None, "sepsis_strict_onset_offset_approx": None},
        ])

        report = build_cohort_report(patient_df, mortality, sepsis_df)

        self.assertEqual(report.total_patients, 4)
        self.assertEqual(report.mortality_positive, 1)
        self.assertEqual(report.sepsis_broad_positive, 2)
        self.assertEqual(report.sepsis_strict_positive, 1)
        self.assertEqual(report.cohort_overlap["both_strict_and_broad"], 1)
        self.assertEqual(report.cohort_overlap["broad_only"], 1)
        self.assertEqual(report.cohort_overlap["strict_only"], 0)
        self.assertEqual(report.cohort_overlap["neither"], 2)
        self.assertEqual(len(report.hospital_distribution), 2)
        self.assertIn("N/A", report.downstream_model_performance)

    def test_patients_missing_from_sepsis_df_do_not_crash_or_corrupt_overlap(self):
        """Regression test for the exact real-data crash: build_cohort_report
        left-merges the full patient cohort onto sepsis_df, which only
        covers patients with at least some medication/microLab/apachePredVar
        data. Patients entirely absent from sepsis_df (real, common --
        eICU coverage is incomplete per table) must resolve to
        sepsis_broad=False/sepsis_strict=False cleanly, not crash inside
        the cohort_overlap or mortality_among_septic computations."""
        patient_df = pd.DataFrame([
            {"patientunitstayid": i, "hospitalid": 1} for i in range(1, 6)
        ])
        mortality = pd.Series([1, 0, 0, 1, 0], index=patient_df.index)
        # sepsis_df covers only patients 1 and 2 -- patients 3, 4, 5 are
        # entirely absent (not even a False row), exactly like a patient
        # with zero medication/microLab/apachePredVar rows in real data.
        sepsis_df = pd.DataFrame([
            {"patientunitstayid": 1, "sepsis_broad": True, "sepsis_strict": True,
             "sepsis_broad_onset_offset_approx": 10, "sepsis_strict_onset_offset_approx": 10},
            {"patientunitstayid": 2, "sepsis_broad": False, "sepsis_strict": False,
             "sepsis_broad_onset_offset_approx": None, "sepsis_strict_onset_offset_approx": None},
        ])

        report = build_cohort_report(patient_df, mortality, sepsis_df)  # must not raise

        self.assertEqual(report.total_patients, 5)
        self.assertEqual(report.sepsis_broad_positive, 1)
        self.assertEqual(report.sepsis_strict_positive, 1)
        # patients 3,4,5 (missing from sepsis_df) must land in "neither"
        self.assertEqual(report.cohort_overlap["neither"], 4)  # patients 2,3,4,5
        self.assertEqual(report.cohort_overlap["both_strict_and_broad"], 1)
        # mortality_among_septic must compute cleanly for the non-septic group too
        self.assertIsNotNone(report.mortality_among_septic["broad_sepsis_negative"])


if __name__ == "__main__":
    unittest.main()
