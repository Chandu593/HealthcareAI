import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.splitting.patient_split import SplitManifest
from src.split_aware.reporting import build_validation_report, render_markdown_report, save_validation_report


def _make_manifest():
    hospital_splits = {
        "1": {"train": [1, 2, 3], "val": [4], "test": [5]},
        "2": {"train": [6, 7], "val": [8], "test": [9]},
    }
    patient_to_split = {}
    for h in hospital_splits.values():
        for split_name in ("train", "val", "test"):
            for pid in h[split_name]:
                patient_to_split[str(pid)] = split_name
    return SplitManifest(
        seed=42, train_frac=0.7, val_frac=0.15, test_frac=0.15, stratify_on="mortality", sepsis_cohort="broad",
        hospital_splits=hospital_splits, patient_to_split=patient_to_split, n_patients_total=9, n_hospitals_total=2,
        n_hospitals_stratified=2, n_hospitals_fallback_unstratified=0, n_hospitals_all_train_too_small=0,
    )


def _make_cohort():
    cohort_patients = pd.DataFrame({"patientunitstayid": list(range(1, 10)), "hospitalid": [1, 1, 1, 1, 1, 2, 2, 2, 2]})
    mortality = pd.Series([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], index=cohort_patients.index)
    sepsis_df = pd.DataFrame({
        "patientunitstayid": list(range(1, 10)),
        "sepsis_broad": [False] * 9, "sepsis_strict": [False] * 9,
    })
    return cohort_patients, mortality, sepsis_df


class TestBuildValidationReport(unittest.TestCase):
    def test_split_section_reports_correct_counts_and_no_overlap(self):
        manifest = _make_manifest()
        cohort_patients, mortality, sepsis_df = _make_cohort()
        report = build_validation_report(manifest, cohort_patients, mortality, sepsis_df)
        self.assertEqual(report["split"]["n_train_total"], 5)
        self.assertEqual(report["split"]["n_val_total"], 2)
        self.assertEqual(report["split"]["n_test_total"], 2)
        self.assertTrue(report["split"]["zero_overlap_confirmed"])
        self.assertEqual(len(report["split"]["per_hospital"]), 2)

    def test_detects_injected_overlap_violation(self):
        """Revert->fail->restore->pass-style proof: a hand-corrupted
        manifest with a patient in two partitions must be caught."""
        manifest = _make_manifest()
        manifest.hospital_splits["1"]["val"].append(1)  # patient 1 is ALSO in train
        cohort_patients, mortality, sepsis_df = _make_cohort()
        report = build_validation_report(manifest, cohort_patients, mortality, sepsis_df)
        self.assertFalse(report["split"]["zero_overlap_confirmed"])
        self.assertTrue(report["split"]["overlap_violations"])

    def test_optional_sections_included_only_when_given(self):
        manifest = _make_manifest()
        cohort_patients, mortality, sepsis_df = _make_cohort()
        report_minimal = build_validation_report(manifest, cohort_patients, mortality, sepsis_df)
        self.assertNotIn("phase11_local", report_minimal)
        self.assertNotIn("ablation", report_minimal)

        report_full = build_validation_report(
            manifest, cohort_patients, mortality, sepsis_df,
            phase11_result={"checkpoint_dir": "x", "n_hospitals_trained": 2, "n_hospitals_skipped_empty": 0,
                             "n_hospitals_skipped_no_labels": 0, "n_hospitals_skipped_non_finite": 0,
                             "n_hospitals_skipped_no_train_patients": 0},
            ablation_summary={"n_combos": 24},
            failures=["something went wrong"],
        )
        self.assertIn("phase11_local", report_full)
        self.assertIn("ablation", report_full)
        self.assertEqual(report_full["failures"], ["something went wrong"])

    def test_markdown_renders_without_crashing_and_contains_key_facts(self):
        manifest = _make_manifest()
        cohort_patients, mortality, sepsis_df = _make_cohort()
        report = build_validation_report(manifest, cohort_patients, mortality, sepsis_df)
        md = render_markdown_report(report)
        self.assertIn("Zero patient overlap confirmed: True", md)
        self.assertIn("9", md)  # total patients or similar appears somewhere reasonable

    def test_save_writes_both_json_and_markdown(self):
        manifest = _make_manifest()
        cohort_patients, mortality, sepsis_df = _make_cohort()
        report = build_validation_report(manifest, cohort_patients, mortality, sepsis_df)
        with tempfile.TemporaryDirectory() as d:
            json_path, md_path = save_validation_report(report, d)
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            with open(json_path) as f:
                reloaded = json.load(f)
            self.assertEqual(reloaded["split"]["n_train_total"], 5)


if __name__ == "__main__":
    unittest.main()
