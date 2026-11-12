import unittest

import numpy as np
import pandas as pd

from src.data.temporal.sequence_builder import (
    attach_sequence_length,
    build_sequence_tensor,
    compute_sequence_length,
)


class TestComputeSequenceLength(unittest.TestCase):
    def test_short_stay_gives_few_bins(self):
        patient_df = pd.DataFrame([
            {"patientunitstayid": 1, "unitdischargeoffset": 100},  # < 1 bin of 240
        ])
        lengths = compute_sequence_length(patient_df, bin_minutes=240, max_bins=48)
        self.assertEqual(lengths[1], 1)

    def test_long_stay_capped_at_max_bins(self):
        patient_df = pd.DataFrame([
            {"patientunitstayid": 1, "unitdischargeoffset": 999999},
        ])
        lengths = compute_sequence_length(patient_df, bin_minutes=240, max_bins=48)
        self.assertEqual(lengths[1], 48)

    def test_missing_offset_falls_back_to_hospital_discharge(self):
        patient_df = pd.DataFrame([
            {"patientunitstayid": 1, "unitdischargeoffset": np.nan, "hospitaldischargeoffset": 480},
        ])
        lengths = compute_sequence_length(patient_df, bin_minutes=240, max_bins=48)
        self.assertEqual(lengths[1], 2)


class TestBuildSequenceTensor(unittest.TestCase):
    def test_shape_and_feature_alignment(self):
        binned = pd.DataFrame([
            {"patientunitstayid": 1, "bin_idx": 0, "feature": "lab_platelet", "value": 200.0},
            {"patientunitstayid": 1, "bin_idx": 2, "feature": "medication_n_active", "value": 3.0},
            {"patientunitstayid": 2, "bin_idx": 0, "feature": "lab_platelet", "value": 150.0},
        ])
        tensor = build_sequence_tensor(
            [binned], patient_ids=[1, 2], feature_names=["lab_platelet", "medication_n_active"], max_bins=5,
        )
        self.assertEqual(tensor.array.shape, (2, 5, 2))
        self.assertEqual(tensor.array[0, 0, 0], 200.0)
        self.assertEqual(tensor.array[0, 2, 1], 3.0)
        self.assertEqual(tensor.array[1, 0, 0], 150.0)
        self.assertTrue(np.isnan(tensor.array[1, 0, 1]))  # patient 2 has no medication data at all

    def test_observed_mask_matches_nan_pattern(self):
        binned = pd.DataFrame([
            {"patientunitstayid": 1, "bin_idx": 0, "feature": "f1", "value": 5.0},
        ])
        tensor = build_sequence_tensor([binned], patient_ids=[1], feature_names=["f1"], max_bins=3)
        self.assertTrue(tensor.observed_mask[0, 0, 0])
        self.assertFalse(tensor.observed_mask[0, 1, 0])

    def test_unknown_patient_in_binned_frame_is_ignored_not_crashed(self):
        binned = pd.DataFrame([
            {"patientunitstayid": 999, "bin_idx": 0, "feature": "f1", "value": 5.0},  # not in patient_ids
        ])
        tensor = build_sequence_tensor([binned], patient_ids=[1], feature_names=["f1"], max_bins=3)
        self.assertTrue(np.all(np.isnan(tensor.array)))

    def test_attach_sequence_length(self):
        tensor = build_sequence_tensor([], patient_ids=[1, 2], feature_names=["f1"], max_bins=5)
        lengths = pd.Series({1: 3, 2: 5})
        tensor = attach_sequence_length(tensor, lengths)
        self.assertEqual(list(tensor.sequence_length), [3, 5])

    def test_attach_sequence_length_missing_patient_defaults_to_max_bins(self):
        tensor = build_sequence_tensor([], patient_ids=[1], feature_names=["f1"], max_bins=5)
        tensor = attach_sequence_length(tensor, pd.Series(dtype=int))
        self.assertEqual(tensor.sequence_length[0], 5)


if __name__ == "__main__":
    unittest.main()
