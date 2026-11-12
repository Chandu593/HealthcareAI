import unittest

import numpy as np

from src.data.tabular.normalization import fit_transform_tabular_local


class TestFitTransformTabularLocal(unittest.TestCase):
    def test_hospitals_normalized_independently(self):
        # column 0 numeric, column 1 categorical (one-hot, pass-through)
        array = np.array([
            [95.0, 1.0],
            [100.0, 0.0],
            [105.0, 1.0],
            [950.0, 0.0],
            [1000.0, 1.0],
            [1050.0, 0.0],
        ], dtype=np.float32)
        patient_ids = [0, 1, 2, 3, 4, 5]
        hospital_map = {0: "A", 1: "A", 2: "A", 3: "B", 4: "B", 5: "B"}
        mask = np.array([True, False])

        result = fit_transform_tabular_local(array, patient_ids, hospital_map, mask)

        hosp_a_numeric = result.array[[0, 1, 2], 0]
        hosp_b_numeric = result.array[[3, 4, 5], 0]
        self.assertAlmostEqual(float(np.mean(hosp_a_numeric)), 0.0, places=4)
        self.assertAlmostEqual(float(np.mean(hosp_b_numeric)), 0.0, places=4)

    def test_categorical_column_passed_through_unscaled(self):
        array = np.array([[50.0, 1.0], [60.0, 0.0]], dtype=np.float32)
        mask = np.array([True, False])
        result = fit_transform_tabular_local(array, [0, 1], {0: "A", 1: "A"}, mask)
        self.assertEqual(sorted(result.array[:, 1].tolist()), [0.0, 1.0])

    def test_missing_numeric_imputed_not_left_nan(self):
        array = np.array([[np.nan, 1.0], [50.0, 0.0]], dtype=np.float32)
        mask = np.array([True, False])
        result = fit_transform_tabular_local(array, [0, 1], {0: "A", 1: "A"}, mask)
        self.assertFalse(np.any(np.isnan(result.array)))

    def test_all_categorical_no_numeric_columns(self):
        array = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        mask = np.array([False, False])
        result = fit_transform_tabular_local(array, [0, 1], {0: "A", 1: "A"}, mask)
        np.testing.assert_array_equal(result.array, array)

    def test_hospital_with_entirely_missing_column_flagged(self):
        array = np.array([[np.nan, 1.0], [np.nan, 0.0]], dtype=np.float32)
        mask = np.array([True, False])
        result = fit_transform_tabular_local(array, [0, 1], {0: "A", 1: "A"}, mask)
        self.assertIn("A", result.hospitals_with_insufficient_data)
        self.assertFalse(np.any(np.isnan(result.array)))


if __name__ == "__main__":
    unittest.main()
