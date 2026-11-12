"""Leakage / window / vocabulary guarantees of src/paper/features.py, on a tiny hand-built eICU-shaped dataset."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.paper.config import SEPSIS_INGREDIENT_GROUPS, get_paper_cfg  # noqa: E402
from src.paper.features import build_feature_bundle  # noqa: E402


def make_dataset(root: Path, flip_outcomes=False, los_p1=3000):
    ids = list(range(1, 9)); hosp = [10] * 4 + [20] * 4
    los = {1: los_p1, 2: 300}
    patient = pd.DataFrame({
        "patientunitstayid": ids, "hospitalid": hosp, "age": ["65", "> 89", "40", "50", "70", "30", "55", "60"], "gender": ["Male", "Female"] * 4,
        "ethnicity": ["Caucasian"] * 8, "admissionheight": [170] * 8, "admissionweight": [80] * 8, "hospitaladmitoffset": [-60] * 8,
        "unitvisitnumber": [1] * 8, "unittype": ["MICU"] * 8, "unitadmitsource": ["ED"] * 8, "hospitaladmitsource": ["ED"] * 8,
        "unitstaytype": ["admit"] * 8, "apacheadmissiondx": ["sepsis"] * 8,
        "unitdischargeoffset": [los.get(i, 4000) for i in ids],
        "hospitaldischargestatus": (["Alive", "Expired"] * 4) if not flip_outcomes else (["Expired", "Alive"] * 4)})
    patient.to_csv(root / "patient.csv", index=False)
    pd.DataFrame({"hospitalid": [10, 20], "numbedscategory": ["100-249", ">= 500"], "teachingstatus": ["f", "t"], "region": ["South", "West"]}).to_csv(root / "hospital.csv", index=False)
    labs = [(1, -100, "creatinine", 1.0), (1, 60, "creatinine", 2.0), (1, 700, "creatinine", 3.0), (1, 1500, "creatinine", 99.0), (1, 5000, "creatinine", 99.0),
            (2, 200, "creatinine", 5.0), (2, 400, "creatinine", 77.0),        # 400 > discharge at 300 -> excluded
            (3, -2000, "creatinine", 42.0),                                    # beyond lookback -> excluded
            (4, 100, "WBC x 1000", 11.0), (5, 100, "creatinine", 1.5), (6, 50, "WBC x 1000", 9.0),
            (8, 100, "rare_lab", 1.0), (7, 100, "rare_lab", 2.0)]              # rare_lab only in non-train patients
    pd.DataFrame(labs, columns=["patientunitstayid", "labresultoffset", "labname", "labresult"]).assign(labid=range(len(labs))).to_csv(root / "lab.csv", index=False)
    apv = pd.DataFrame({"patientunitstayid": ids, "verbal": [5] * 8, "motor": [6] * 8, "eyes": [4] * 8, "pao2": [90] * 8, "fio2": [40] * 8,
                        "diabetes": [0, 1] * 4, "diedinhospital": ([0, 1] * 4) if not flip_outcomes else ([1, 0] * 4), "dischargelocation": ["Home"] * 8})
    apv.to_csv(root / "apachePredVar.csv", index=False)
    pd.DataFrame({"patientunitstayid": ids, "acutephysiologyscore": [30] * 8, "apachescore": [50] * 8, "predictedhospitalmortality": [0.2] * 8,
                  "actualhospitalmortality": (["ALIVE", "EXPIRED"] * 4) if not flip_outcomes else (["EXPIRED", "ALIVE"] * 4), "apacheversion": ["IVa"] * 8}).to_csv(root / "apachePatientResult.csv", index=False)
    return patient


def build(root, flip=False, los_p1=3000, **fover):
    patient = make_dataset(root, flip, los_p1)
    cfg = get_paper_cfg({"paper": {"features": {"min_vocab_patients": 1, "top_labs": 5, "chunksize": 4, **fover}, "sequence": {"n_labs": 2}}})
    p1 = {"table_paths": {n: str(root / f"{n}.csv") for n in ("patient", "hospital", "lab", "apachePredVar", "apachePatientResult")}, "data_root": str(root)}
    mortality = pd.Series((patient["hospitaldischargestatus"] == "Expired").astype(int).to_numpy(), index=patient.index)
    return build_feature_bundle(cfg, p1, patient, mortality, None, train_patient_ids=[1, 2, 3, 5, 6]), cfg


def col(b, name):
    return b.X[:, b.feature_names.index(name)]


class TestWindowAndLeakage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_lab_window_clipping_lookback_and_discharge(self):
        b, _ = build(self.root)
        row = {int(p): i for i, p in enumerate(b.patient_ids)}
        cnt, first, last = col(b, "lab_creatinine_count"), col(b, "lab_creatinine_first"), col(b, "lab_creatinine_last")
        # patient 1: -100 (lookback ok), 60, 700 kept; 1500/5000 are after the 1440-min window
        self.assertEqual(cnt[row[1]], 3); self.assertEqual(first[row[1]], 1.0); self.assertEqual(last[row[1]], 3.0)
        # patient 2 discharged at minute 300: the value at 400 must NOT be visible
        self.assertEqual(cnt[row[2]], 1); self.assertEqual(last[row[2]], 5.0)
        # patient 3: only a lab 2000 min before ICU admission (beyond 1440 lookback)
        self.assertEqual(cnt[row[3]], 0); self.assertTrue(np.isnan(first[row[3]]))

    def test_features_do_not_depend_on_outcome_columns_or_los_beyond_window(self):
        a, _ = build(self.root)
        b, _ = build(self.root, flip=True, los_p1=9000)     # flips every outcome column AND changes LOS beyond the window
        self.assertEqual(a.X.shape, b.X.shape)
        self.assertTrue(np.array_equal(a.X, b.X, equal_nan=True), "X changed when only outcome columns / post-window LOS changed")
        self.assertFalse(np.array_equal(a.y_mortality, b.y_mortality, equal_nan=True), "sanity: labels must have flipped")

    def test_vocabulary_is_train_only(self):
        b, _ = build(self.root)
        self.assertFalse(any("rare_lab" in n for n in b.feature_names), "a lab seen only in val/test patients leaked into the vocabulary")
        self.assertTrue(any("creatinine" in n for n in b.feature_names))

    def test_no_forbidden_feature_names_and_benchmark_kept_separate(self):
        b, _ = build(self.root)
        for n, g in zip(b.feature_names, b.feature_groups):
            if g in ("demographics", "hospital", "apache_physiology", "apache_comorbidity", "apache_score", "lab", "lab_sofa"):
                for bad in ("discharge", "died", "actual", "predicted", "mortality"):
                    self.assertNotIn(bad, n.lower())
        self.assertTrue(np.allclose(b.apache_pred_mortality, 0.2))   # kept only as a benchmark vector
        self.assertNotIn("predictedhospitalmortality", " ".join(b.feature_names))

    def test_apache_score_toggle_and_sepsis_controlled_columns(self):
        off, _ = build(self.root, include_apache_scores=False)
        self.assertNotIn("apache_score", off.feature_groups)
        b, _ = build(self.root)
        cols = b.columns_excluding(SEPSIS_INGREDIENT_GROUPS)
        kept = [b.feature_names[i] for i in cols]
        for bad in ("creatinine", "pao2", "fio2", "verbal", "motor", "eyes", "apr_apachescore", "med_antibiotic", "vasopressor"):
            self.assertFalse(any(bad in n for n in kept), f"{bad} survived in the sepsis-controlled feature set")

    def test_age_parsing_and_missing_tables_do_not_crash(self):
        b, _ = build(self.root)
        age = col(b, "age"); row = {int(p): i for i, p in enumerate(b.patient_ids)}
        self.assertEqual(age[row[2]], 89.0)     # '> 89' -> 89 after stripping '>' (documented eICU top-coding)


if __name__ == "__main__":
    unittest.main()


class TestSepsisLadderFeatureSets(unittest.TestCase):
    def test_ladder_drops_what_it_claims_and_is_strictly_nested(self):
        from src.paper.features import FeatureBundle
        names = ["cat_apacheadmissiondx__sepsis_pulmonary", "cat_apacheadmissiondx__cabg", "lab_pao2_mean", "lab_paco2_mean", "lab_ph_last", "lab_wbc_x_1000_mean",
                 "lab_bun_last", "dx_renal_acute_renal_failure", "dx_neurologic_seizures", "apv_diabetes", "med_antibiotic_any", "hosp_log_n_patients", "hosp_region__midwest"]
        groups = ["demographics", "demographics", "lab", "lab", "lab", "lab", "lab", "diagnosis", "diagnosis", "apache_comorbidity", "medication_antibiotic", "hospital", "hospital"]
        b = FeatureBundle(np.arange(2), np.zeros(2, int), np.zeros((2, len(names)), "float32"), names, groups, np.zeros(2), np.zeros(2), np.zeros(2))
        cfg = get_paper_cfg({})["feature_sets"]
        kept = lambda fs: {b.feature_names[i] for i in b.columns_excluding(cfg[fs]["exclude_groups"], cfg[fs].get("exclude_name_regex"))}
        ctrl, dec, nohosp, nolab = kept("sepsis_controlled"), kept("sepsis_deconfounded"), kept("sepsis_no_hospital"), kept("sepsis_no_labs")
        self.assertIn("lab_paco2_mean", ctrl); self.assertNotIn("lab_paco2_mean", dec)             # the #1 feature of the controlled model on real data
        for abg in ("lab_pao2_mean", "lab_ph_last"):
            self.assertNotIn(abg, dec)
        self.assertIn("cat_apacheadmissiondx__sepsis_pulmonary", ctrl); self.assertNotIn("cat_apacheadmissiondx__sepsis_pulmonary", dec)
        self.assertIn("dx_renal_acute_renal_failure", ctrl); self.assertNotIn("dx_renal_acute_renal_failure", dec)
        self.assertIn("hosp_log_n_patients", dec); self.assertNotIn("hosp_log_n_patients", nohosp); self.assertNotIn("hosp_region__midwest", nohosp)
        self.assertIn("lab_wbc_x_1000_mean", nohosp); self.assertNotIn("lab_wbc_x_1000_mean", nolab); self.assertNotIn("lab_bun_last", nolab)
        self.assertIn("cat_apacheadmissiondx__cabg", nolab); self.assertIn("dx_neurologic_seizures", nolab); self.assertNotIn("med_antibiotic_any", ctrl)
        self.assertTrue(nolab < nohosp < dec < ctrl)


if __name__ == "__main__":
    unittest.main()


class TestSepsisCleanSet(unittest.TestCase):
    def test_clean_drops_activetx_and_is_nested_in_no_hospital(self):
        from src.paper.features import FeatureBundle
        names = ["apv_activetx", "apv_aids", "hosp_log_n_patients", "lab_bun_last", "lab_paco2_mean", "age"]
        groups = ["apache_comorbidity", "apache_comorbidity", "hospital", "lab", "lab", "demographics"]
        b = FeatureBundle(np.arange(2), np.zeros(2, int), np.zeros((2, len(names)), "float32"), names, groups, np.zeros(2), np.zeros(2), np.zeros(2))
        cfg = get_paper_cfg({})["feature_sets"]
        kept = lambda fs: {b.feature_names[i] for i in b.columns_excluding(cfg[fs]["exclude_groups"], cfg[fs].get("exclude_name_regex"))}
        self.assertIn("apv_activetx", kept("sepsis_no_hospital")); self.assertNotIn("apv_activetx", kept("sepsis_clean"))
        self.assertEqual(kept("sepsis_clean"), {"apv_aids", "lab_bun_last", "age"}); self.assertTrue(kept("sepsis_clean") < kept("sepsis_no_hospital"))
