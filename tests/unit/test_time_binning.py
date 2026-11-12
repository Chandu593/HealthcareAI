import unittest

import numpy as np
import pandas as pd

from src.data.temporal.time_binning import (
    assign_bin,
    bin_infusion_features,
    bin_lab_features,
    bin_medication_features,
    select_lab_features,
)


class TestAssignBin(unittest.TestCase):
    def test_basic_binning(self):
        offsets = pd.Series([0, 239, 240, 479, 480])
        bins = assign_bin(offsets, bin_minutes=240, max_bins=10)
        self.assertEqual(list(bins), [0, 0, 1, 1, 2])

    def test_negative_offset_excluded(self):
        offsets = pd.Series([-100, 0, 100])
        bins = assign_bin(offsets, bin_minutes=240, max_bins=10)
        self.assertTrue(np.isnan(bins.iloc[0]))
        self.assertFalse(np.isnan(bins.iloc[1]))

    def test_beyond_max_bins_excluded(self):
        offsets = pd.Series([0, 100000])
        bins = assign_bin(offsets, bin_minutes=240, max_bins=5)
        self.assertFalse(np.isnan(bins.iloc[0]))
        self.assertTrue(np.isnan(bins.iloc[1]))


class TestSelectLabFeatures(unittest.TestCase):
    def test_configured_features_matched(self):
        lab_df = pd.DataFrame({"labname": ["platelets", "platelets", "total bilirubin"]})
        result = select_lab_features(lab_df, tracked_labnames=["platelet", "bilirubin"])
        self.assertEqual(set(result), {"platelet", "bilirubin"})

    def test_falls_back_to_top_k_when_no_match(self):
        lab_df = pd.DataFrame({"labname": ["glucose"] * 5 + ["sodium"] * 3 + ["potassium"] * 1})
        result = select_lab_features(lab_df, tracked_labnames=["nonexistent_labname_xyz"], top_k_fallback=2)
        self.assertEqual(result, ["glucose", "sodium"])

    def test_empty_lab_df_returns_empty(self):
        result = select_lab_features(pd.DataFrame(columns=["labname"]), tracked_labnames=["platelet"])
        self.assertEqual(result, [])


class TestBinLabFeatures(unittest.TestCase):
    def test_mean_aggregation_within_bin(self):
        lab_df = pd.DataFrame([
            {"patientunitstayid": 1, "labresultoffset": 10, "labname": "platelets", "labresult": 100},
            {"patientunitstayid": 1, "labresultoffset": 200, "labname": "platelets", "labresult": 200},
            {"patientunitstayid": 1, "labresultoffset": 250, "labname": "platelets", "labresult": 50},  # bin 1
        ])
        result = bin_lab_features(lab_df, feature_names=["platelet"], bin_minutes=240, max_bins=10)
        bin0 = result[(result["bin_idx"] == 0)]
        self.assertAlmostEqual(bin0["value"].iloc[0], 150.0)  # mean(100,200)
        bin1 = result[(result["bin_idx"] == 1)]
        self.assertAlmostEqual(bin1["value"].iloc[0], 50.0)

    def test_feature_name_prefixed(self):
        lab_df = pd.DataFrame([
            {"patientunitstayid": 1, "labresultoffset": 10, "labname": "total bilirubin", "labresult": 1.5},
        ])
        result = bin_lab_features(lab_df, feature_names=["bilirubin"], bin_minutes=240, max_bins=10)
        self.assertEqual(result["feature"].iloc[0], "lab_bilirubin")


class TestBinMedicationFeatures(unittest.TestCase):
    def test_drug_spanning_multiple_bins_is_active_in_all(self):
        med_df = pd.DataFrame([
            {"patientunitstayid": 1, "drugstartoffset": 10, "drugstopoffset": 700, "drugname": "VANCOMYCIN 1000 MG IV SOLN"},
        ])
        # bin_minutes=240 -> bins 0,1,2 (0-239, 240-479, 480-719) all overlap [10,700]
        result = bin_medication_features(med_df, bin_minutes=240, max_bins=10)
        abx = result[result["feature"] == "medication_antibiotic_active"]
        self.assertEqual(sorted(abx["bin_idx"].tolist()), [0, 1, 2])
        self.assertTrue((abx["value"] == 1).all())

    def test_non_antibiotic_flagged_zero(self):
        med_df = pd.DataFrame([
            {"patientunitstayid": 1, "drugstartoffset": 10, "drugstopoffset": 50, "drugname": "ASPIRIN EC 81 MG PO TBEC"},
        ])
        result = bin_medication_features(med_df, bin_minutes=240, max_bins=10)
        abx = result[result["feature"] == "medication_antibiotic_active"]
        self.assertEqual(abx["value"].iloc[0], 0)

    def test_missing_stop_offset_defaults_to_one_bin(self):
        med_df = pd.DataFrame([
            {"patientunitstayid": 1, "drugstartoffset": 10, "drugstopoffset": None, "drugname": "x"},
        ])
        result = bin_medication_features(med_df, bin_minutes=240, max_bins=10)
        n_active = result[result["feature"] == "medication_n_active"]
        self.assertEqual(n_active["bin_idx"].tolist(), [0])


class TestBinInfusionFeatures(unittest.TestCase):
    def test_vasopressor_rate_averaged(self):
        inf_df = pd.DataFrame([
            {"patientunitstayid": 1, "infusionoffset": 10, "drugname": "Norepinephrine (mcg/min)", "infusionrate": 10.0},
            {"patientunitstayid": 1, "infusionoffset": 20, "drugname": "Norepinephrine (mcg/min)", "infusionrate": 20.0},
        ])
        result = bin_infusion_features(inf_df, bin_minutes=240, max_bins=10)
        rate = result[result["feature"] == "infusion_vasopressor_rate_mean"]
        self.assertAlmostEqual(rate["value"].iloc[0], 15.0)

    def test_non_vasopressor_not_counted_in_rate(self):
        inf_df = pd.DataFrame([
            {"patientunitstayid": 1, "infusionoffset": 10, "drugname": "Propofol (mcg/kg/min)", "infusionrate": 50.0},
        ])
        result = bin_infusion_features(inf_df, bin_minutes=240, max_bins=10)
        rate = result[result["feature"] == "infusion_vasopressor_rate_mean"]
        self.assertEqual(len(rate), 0)


if __name__ == "__main__":
    unittest.main()
