import unittest

import numpy as np

from src.data.temporal.normalization import fit_transform_local
from src.data.temporal.sequence_builder import SequenceTensor


class TestFitTransformLocal(unittest.TestCase):
    def test_hospitals_normalized_independently(self):
        # Hospital A's feature values center around 100, hospital B's
        # around 1000 -- if normalization were global, B's low-end
        # values would look artificially "normal" relative to A's scale.
        # With correct per-hospital fitting, each hospital's OWN real
        # values should end up roughly mean-0 after transform.
        n_patients = 6
        max_bins = 4
        n_features = 1
        array = np.full((n_patients, max_bins, n_features), np.nan, dtype=np.float32)

        # Patients 0,1,2 -> hospital A, values ~100
        array[0, 0, 0] = 95
        array[1, 0, 0] = 100
        array[2, 0, 0] = 105
        # Patients 3,4,5 -> hospital B, values ~1000
        array[3, 0, 0] = 950
        array[4, 0, 0] = 1000
        array[5, 0, 0] = 1050

        tensor = SequenceTensor(
            patient_ids=[0, 1, 2, 3, 4, 5],
            feature_names=["f1"],
            array=array,
            observed_mask=~np.isnan(array),
            sequence_length=np.array([1, 1, 1, 1, 1, 1]),
        )
        hospital_map = {0: "A", 1: "A", 2: "A", 3: "B", 4: "B", 5: "B"}

        result = fit_transform_local(tensor, hospital_map)

        hosp_a_values = result.array[[0, 1, 2], 0, 0]
        hosp_b_values = result.array[[3, 4, 5], 0, 0]
        self.assertAlmostEqual(float(np.mean(hosp_a_values)), 0.0, places=4)
        self.assertAlmostEqual(float(np.mean(hosp_b_values)), 0.0, places=4)
        self.assertIn("A", result.scalers_by_hospital)
        self.assertIn("B", result.scalers_by_hospital)

    def test_padding_bins_are_zeroed(self):
        array = np.full((1, 4, 1), np.nan, dtype=np.float32)
        array[0, 0, 0] = 50.0  # only bin 0 is "real"
        tensor = SequenceTensor(
            patient_ids=[0], feature_names=["f1"], array=array,
            observed_mask=~np.isnan(array), sequence_length=np.array([1]),
        )
        result = fit_transform_local(tensor, {0: "A"})
        # bins 1,2,3 are padding (sequence_length=1) -> must be exactly 0
        self.assertTrue(np.all(result.array[0, 1:, 0] == 0.0))

    def test_missing_within_real_bins_imputed_not_left_nan(self):
        array = np.full((1, 3, 2), np.nan, dtype=np.float32)
        array[0, 0, 0] = 100.0  # feature 0 observed in bin 0
        array[0, 1, 0] = 100.0  # feature 0 observed in bin 1
        # feature 1 never observed for this patient/hospital at all
        tensor = SequenceTensor(
            patient_ids=[0], feature_names=["f1", "f2"], array=array,
            observed_mask=~np.isnan(array), sequence_length=np.array([2]),
        )
        result = fit_transform_local(tensor, {0: "A"})
        self.assertFalse(np.any(np.isnan(result.array)))
        self.assertIn("A", result.hospitals_with_insufficient_data)

    def test_single_hospital_all_patients(self):
        array = np.random.default_rng(1).normal(50, 10, size=(4, 2, 1)).astype(np.float32)
        tensor = SequenceTensor(
            patient_ids=[0, 1, 2, 3], feature_names=["f1"], array=array,
            observed_mask=np.ones_like(array, dtype=bool), sequence_length=np.array([2, 2, 2, 2]),
        )
        result = fit_transform_local(tensor, {0: "A", 1: "A", 2: "A", 3: "A"})
        self.assertEqual(result.array.shape, array.shape)
        self.assertEqual(list(result.scalers_by_hospital.keys()), ["A"])


if __name__ == "__main__":
    unittest.main()
