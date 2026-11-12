import unittest

import numpy as np

from src.data.splitting.patient_split import SplitManifest
from src.split_aware.evaluation import (
    aggregate_predictions_by_hospital,
    evaluate_hospital_results,
    evaluate_model_predictions,
)


def _make_manifest(hospital_splits: dict) -> SplitManifest:
    patient_to_split = {}
    for hid_splits in hospital_splits.values():
        for split_name in ("train", "val", "test"):
            for pid in hid_splits[split_name]:
                patient_to_split[str(pid)] = split_name
    return SplitManifest(
        seed=1, train_frac=0.7, val_frac=0.15, test_frac=0.15, stratify_on="mortality", sepsis_cohort="broad",
        hospital_splits={str(k): v for k, v in hospital_splits.items()}, patient_to_split=patient_to_split,
    )


class TestAggregatePredictions(unittest.TestCase):
    def test_concatenates_across_hospitals_in_sorted_order(self):
        hospital_predictions = {
            "2": {"patient_ids": [10, 11], "mortality_logits": [1.0, -1.0], "sepsis_logits": [0.5, 0.5],
                  "mortality_labels": [1.0, 0.0], "sepsis_labels": [0.0, 0.0], "is_train_row": ["train", "not_train"]},
            "1": {"patient_ids": [20], "mortality_logits": [2.0], "sepsis_logits": [0.1],
                  "mortality_labels": [1.0], "sepsis_labels": [1.0], "is_train_row": ["train"]},
        }
        agg = aggregate_predictions_by_hospital(hospital_predictions)
        # hospital "1" sorts before "2" (string sort) -> patient 20 first
        np.testing.assert_array_equal(agg["patient_ids"], [20, 10, 11])
        self.assertEqual(len(agg["mortality_proba"]), 3)

    def test_logits_are_converted_to_probabilities(self):
        hospital_predictions = {
            "1": {"patient_ids": [1], "mortality_logits": [0.0], "sepsis_logits": [0.0],
                  "mortality_labels": [1.0], "sepsis_labels": [0.0], "is_train_row": ["train"]},
        }
        agg = aggregate_predictions_by_hospital(hospital_predictions)
        self.assertAlmostEqual(agg["mortality_proba"][0], 0.5, places=6)


class TestEvaluateModelPredictions(unittest.TestCase):
    def test_end_to_end_train_val_test_split_with_threshold_from_val(self):
        rng = np.random.RandomState(0)
        n = 60
        patient_ids = list(range(n))
        # Separable-ish synthetic logits: positives skew positive logit.
        labels = (rng.rand(n) > 0.5).astype(float)
        logits = labels * 2.0 - 1.0 + rng.randn(n) * 0.3

        hospital_predictions = {
            "1": {
                "patient_ids": patient_ids,
                "mortality_logits": logits.tolist(),
                "sepsis_logits": logits.tolist(),
                "mortality_labels": labels.tolist(),
                "sepsis_labels": labels.tolist(),
                "is_train_row": ["train"] * n,
            }
        }
        hospital_splits = {
            1: {"train": patient_ids[:40], "val": patient_ids[40:50], "test": patient_ids[50:60]},
        }
        manifest = _make_manifest(hospital_splits)

        metrics = evaluate_model_predictions(hospital_predictions, manifest, "mortality")
        for split in ("train", "val", "test"):
            self.assertIn("auroc", metrics[split])
            self.assertGreater(metrics[split]["auroc"], 0.5)
        # threshold used for val and test must match (selected on val only)
        self.assertEqual(metrics["val"]["threshold_used"], metrics["test"]["threshold_used"])

    def test_invalid_task_raises(self):
        with self.assertRaises(ValueError):
            evaluate_model_predictions({}, _make_manifest({}), "not_a_task")

    def test_empty_split_returns_empty_dict_not_crash(self):
        hospital_predictions = {
            "1": {"patient_ids": [1, 2], "mortality_logits": [1.0, -1.0], "sepsis_logits": [1.0, -1.0],
                  "mortality_labels": [1.0, 0.0], "sepsis_labels": [1.0, 0.0], "is_train_row": ["train", "train"]},
        }
        manifest = _make_manifest({1: {"train": [1, 2], "val": [], "test": []}})
        metrics = evaluate_model_predictions(hospital_predictions, manifest, "mortality")
        self.assertEqual(metrics["val"], {})
        self.assertEqual(metrics["test"], {})
        self.assertIn("auroc", metrics["train"])


class TestEvaluateHospitalResults(unittest.TestCase):
    def test_wraps_phase11_style_hospital_results_shape(self):
        hospital_results = {
            "1": {
                "n_patients": 2,
                "predictions": {
                    "patient_ids": [1, 2], "mortality_logits": [1.0, -1.0], "sepsis_logits": [1.0, -1.0],
                    "mortality_labels": [1.0, 0.0], "sepsis_labels": [1.0, 0.0], "is_train_row": ["train", "train"],
                },
            },
        }
        manifest = _make_manifest({1: {"train": [1, 2], "val": [], "test": []}})
        result = evaluate_hospital_results(hospital_results, manifest)
        self.assertIn("mortality", result)
        self.assertIn("sepsis", result)

    def test_no_predictions_present_returns_empty_shape(self):
        hospital_results = {"1": {"n_patients": 2}}  # non-split-aware run, no "predictions" key
        manifest = _make_manifest({1: {"train": [1, 2], "val": [], "test": []}})
        result = evaluate_hospital_results(hospital_results, manifest)
        self.assertEqual(result, {"mortality": {}, "sepsis": {}})


if __name__ == "__main__":
    unittest.main()
