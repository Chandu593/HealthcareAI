import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))

from paper_local_calibration import rescore  # noqa: E402
from paper_sepsis_label_audit import audit  # noqa: E402
from src.paper.evaluation import apply_local_intercepts, fast_auroc, fit_local_intercepts, per_hospital_summary, platt_fit  # noqa: E402


def hospital_offset_data(seed=0, n_h=30, per_h=400):
    rng = np.random.default_rng(seed); h = np.repeat(np.arange(1, n_h + 1), per_h); off = rng.normal(0, 1.2, n_h)
    x = rng.normal(size=len(h)); y = (rng.random(len(h)) < 1 / (1 + np.exp(-(1.5 * x + off[h - 1] - 1.0)))).astype(int)
    s = 1.5 * x - 1.0           # a model that knows the within-hospital signal but NOT the hospital offset
    sp = rng.choice(["train", "val", "test"], len(h), p=[.5, .25, .25])
    return s, y, h, sp, off


class TestLocalCalibration(unittest.TestCase):
    def test_recovers_offsets_improves_pooled_auroc_and_keeps_within_hospital_ranking(self):
        s, y, h, sp, off = hospital_offset_data(); va, te = sp == "val", sp == "test"
        a, b = platt_fit(s[va], y[va]); sh = fit_local_intercepts(s[va], y[va], h[va], a, b, 1.5)
        est = np.array([sh[k] for k in sorted(sh)]); self.assertGreater(np.corrcoef(est, off)[0, 1], 0.9)
        pl = apply_local_intercepts(s, h, a, b, sh); pg = 1 / (1 + np.exp(-(a * s + b)))
        self.assertGreater(fast_auroc(y[te], pl[te]), fast_auroc(y[te], pg[te]) + 0.02)
        self.assertAlmostEqual(per_hospital_summary(y[te], pl[te], h[te])["macro_auroc"], per_hospital_summary(y[te], pg[te], h[te])["macro_auroc"], places=9)

    def test_fitted_on_validation_only_and_shrinks_tiny_hospitals(self):
        s, y, h, sp, off = hospital_offset_data(1); va = sp == "val"; a, b = platt_fit(s[va], y[va])
        base = fit_local_intercepts(s[va], y[va], h[va], a, b, 1.0)
        y2 = y.copy(); y2[sp == "test"] = 1 - y2[sp == "test"]                    # corrupt every TEST label: shifts must not move
        self.assertEqual(base, fit_local_intercepts(s[va], y2[va], h[va], a, b, 1.0))
        k = np.where(va & (h == 1))[0][:3]                                       # a hospital with 3 validation rows
        tiny = fit_local_intercepts(s[k], y[k], h[k], a, b, 0.3)[1]; loose = fit_local_intercepts(s[k], y[k], h[k], a, b, 10.0)[1]
        self.assertLess(abs(tiny), abs(loose) + 1e-12)
        self.assertEqual(apply_local_intercepts(np.array([0.0]), np.array([999]), 1.0, 0.0, base)[0], 0.5)     # unseen hospital -> no shift

    def test_offline_script_rescoring_on_saved_predictions(self):
        s, y, h, sp, off = hospital_offset_data(2)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "mortality__full__lightgbm_central__seed42.npz"
            np.savez_compressed(f, patient_id=np.arange(len(y)), hospital_id=h, split=sp, y=y, score=s)
            r = rescore(f, 1.5); self.assertGreater(r["local"]["auroc"], r["global"]["auroc"] + 0.02)
            self.assertAlmostEqual(r["local"]["per_hosp_macro_auroc"], r["global"]["per_hosp_macro_auroc"], places=9)
            out = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "paper_local_calibration.py"), "--run-dir", d.replace("\\", "/")], capture_output=True, text=True)
            self.assertEqual(out.returncode, 1)                                   # no predictions/ dir -> clean message, not a crash


class TestSepsisLabelAudit(unittest.TestCase):
    def test_unknown_labels_excluded_documented_overlap_and_coverage_bins(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "data"; out = Path(d) / "outputs" / "cohort_reports"; root.mkdir(); out.mkdir(parents=True)
            ids = list(range(1, 11)); hosp = [10] * 5 + [20] * 5
            pd.DataFrame({"patientunitstayid": ids, "hospitalid": hosp, "unittype": ["CCU-CTICU"] * 5 + ["MICU"] * 5, "unitdischargeoffset": [3000] * 10}).to_csv(out / "cohort_patients.csv", index=False)
            # patient 5 and 10 have NO label row (unknown); hospital 20 has no medication rows at all and no positives
            pd.DataFrame({"patientunitstayid": [1, 2, 3, 4, 6, 7, 8, 9], "sepsis_broad": [1, 1, 0, 0, 0, 0, 0, 0], "broad_infection": [1, 1, 1, 0, 0, 0, 0, 0],
                          "organ_dysfunction": [1, 1, 0, 0, 1, 1, 0, 0]}).to_csv(out / "sepsis_labels.csv", index=False)
            pd.DataFrame({"patientunitstayid": ids, "graftcount": [3, 2, 0, 0, 0, 0, 0, 0, 0, 0]}).to_csv(root / "apachePredVar.csv", index=False)
            pd.DataFrame({"patientunitstayid": [1, 6, 7], "diagnosisstring": ["sepsis, pulmonary", "cardiac|chf", "infectious|sepsis"]}).to_csv(root / "diagnosis.csv", index=False)
            pd.DataFrame({"patientunitstayid": [1, 2, 3, 4, 5]}).to_csv(root / "medication.csv", index=False)
            r = audit(root, Path(d) / "outputs", thresholds=(0.5,), min_hospital_n=3)
            self.assertEqual((r["n"], r["n_unknown_label"]), (10, 2)); self.assertAlmostEqual(r["proxy_prevalence_known"], 2 / 8)          # unknown is NOT treated as negative
            self.assertEqual(r["documented"]["n"], 2); self.assertAlmostEqual(r["documented"]["proxy_recall_of_documented"], 0.5)           # patients 1 (proxy) and 7 (not)
            self.assertAlmostEqual(r["documented"]["documented_among_proxy_positives"], 0.5)
            self.assertIn("0", r["graftcount_value_counts"])
            self.assertEqual(r["threshold_options"][0]["hospitals_kept"], 1); self.assertAlmostEqual(r["threshold_options"][0]["proxy_rate_kept"], 2 / 4)   # only hospital 10 is covered
            low = [b for b in r["coverage_bins"] if b["coverage"].startswith("[0.00")][0]; self.assertEqual(low["hospitals"], 1)         # hospital 20: no medication rows at all, zero positives


if __name__ == "__main__":
    unittest.main()
