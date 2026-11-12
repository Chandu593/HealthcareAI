import unittest

import pandas as pd

from src.data.cohort.sofa_like_scoring import SofaLikeConfig, compute_sofa_like_scores


class TestSofaLikeScoring(unittest.TestCase):
    def setUp(self):
        self.config = SofaLikeConfig(threshold=2, observation_window_minutes=1440)

    def test_healthy_patient_scores_zero(self):
        apv = pd.DataFrame([{
            "patientunitstayid": 1, "day1pao2": 95.0, "day1fio2": 21.0,
            "day1verbal": 5, "day1motor": 6, "day1eyes": 4,
            "creatinine": 0.9, "ventday1": 0,
        }])
        lab = pd.DataFrame([
            {"patientunitstayid": 1, "labresultoffset": 100, "labname": "platelets", "labresult": 250},
            {"patientunitstayid": 1, "labresultoffset": 100, "labname": "total bilirubin", "labresult": 0.6},
        ])
        infusion = pd.DataFrame(columns=["patientunitstayid", "infusionoffset", "drugname"])

        result = compute_sofa_like_scores(apv, lab, infusion, self.config)
        row = result.iloc[0]
        self.assertEqual(row["sofa_like_total"], 0)
        self.assertFalse(row["organ_dysfunction"])

    def test_severe_patient_scores_high_and_flagged(self):
        apv = pd.DataFrame([{
            "patientunitstayid": 2, "day1pao2": 55.0, "day1fio2": 100.0,
            "day1verbal": 1, "day1motor": 2, "day1eyes": 1,
            "creatinine": 5.5, "ventday1": 1,
        }])
        lab = pd.DataFrame([
            {"patientunitstayid": 2, "labresultoffset": 100, "labname": "platelets", "labresult": 15},
            {"patientunitstayid": 2, "labresultoffset": 100, "labname": "total bilirubin", "labresult": 13.0},
        ])
        infusion = pd.DataFrame([
            {"patientunitstayid": 2, "infusionoffset": 50, "drugname": "Norepinephrine (mcg/min)"},
        ])

        result = compute_sofa_like_scores(apv, lab, infusion, self.config)
        row = result.iloc[0]
        self.assertEqual(row["respiratory_score"], 4)
        self.assertEqual(row["coagulation_score"], 4)
        self.assertEqual(row["liver_score"], 4)
        self.assertEqual(row["renal_score"], 4)
        self.assertEqual(row["cns_score"], 4)
        self.assertEqual(row["cardiovascular_score"], 2)
        self.assertTrue(row["organ_dysfunction"])

    def test_sentinel_negative_one_is_treated_as_missing_not_a_real_value(self):
        apv = pd.DataFrame([{
            "patientunitstayid": 3, "day1pao2": -1.0, "day1fio2": -1.0,
            "day1verbal": -1, "day1motor": -1, "day1eyes": -1,
            "creatinine": -1.0, "ventday1": 0,
        }])
        lab = pd.DataFrame(columns=["patientunitstayid", "labresultoffset", "labname", "labresult"])
        infusion = pd.DataFrame(columns=["patientunitstayid", "infusionoffset", "drugname"])

        result = compute_sofa_like_scores(apv, lab, infusion, self.config)
        row = result.iloc[0]
        self.assertTrue(pd.isna(row["respiratory_score"]))
        self.assertTrue(pd.isna(row["renal_score"]))
        self.assertTrue(pd.isna(row["cns_score"]))
        # With everything missing, max_possible_score should be 0, not silently treated as "healthy"
        self.assertEqual(row["max_possible_score"], 0)

    def test_lab_outside_observation_window_excluded(self):
        apv = pd.DataFrame([{
            "patientunitstayid": 4, "day1pao2": 95.0, "day1fio2": 21.0,
            "day1verbal": 5, "day1motor": 6, "day1eyes": 4,
            "creatinine": 0.9, "ventday1": 0,
        }])
        # Bilirubin reading is way outside the 24h window (offset=5000 > 1440)
        lab = pd.DataFrame([
            {"patientunitstayid": 4, "labresultoffset": 5000, "labname": "total bilirubin", "labresult": 20.0},
        ])
        infusion = pd.DataFrame(columns=["patientunitstayid", "infusionoffset", "drugname"])

        result = compute_sofa_like_scores(apv, lab, infusion, self.config)
        row = result.iloc[0]
        self.assertTrue(pd.isna(row["liver_score"]))  # excluded, not scored as abnormal


if __name__ == "__main__":
    unittest.main()
