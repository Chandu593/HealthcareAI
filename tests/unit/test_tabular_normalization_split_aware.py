import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.tabular.normalization import fit_transform_tabular_local, fit_transform_tabular_local_split_aware


class TestFitTransformTabularLocalSplitAware(unittest.TestCase):
    def test_scaler_matches_a_direct_fit_on_train_rows_only(self):
        array = np.array([[90.0], [100.0], [110.0], [120.0], [5000.0], [9000.0]], dtype=np.float32)
        patient_ids = [1, 2, 3, 4, 5, 6]
        hospital_map = {pid: "A" for pid in patient_ids}
        mask = np.array([True])
        train_ids = {1, 2, 3, 4}

        result = fit_transform_tabular_local_split_aware(array, patient_ids, hospital_map, mask, train_ids)

        reference = StandardScaler()
        reference.fit(np.array([[90.0], [100.0], [110.0], [120.0]]))
        np.testing.assert_allclose(result.scalers_by_hospital["A"].mean_, reference.mean_)
        np.testing.assert_allclose(result.scalers_by_hospital["A"].scale_, reference.scale_)

    def test_changing_val_test_rows_never_changes_the_fitted_scaler_or_train_output(self):
        patient_ids = [1, 2, 3, 4]
        hospital_map = {pid: "A" for pid in patient_ids}
        mask = np.array([True])
        train_ids = {1, 2, 3}

        array_before = np.array([[90.0], [100.0], [110.0], [200.0]], dtype=np.float32)
        result_before = fit_transform_tabular_local_split_aware(array_before, patient_ids, hospital_map, mask, train_ids)

        array_after = np.array([[90.0], [100.0], [110.0], [999999.0]], dtype=np.float32)
        result_after = fit_transform_tabular_local_split_aware(array_after, patient_ids, hospital_map, mask, train_ids)

        np.testing.assert_allclose(result_before.scalers_by_hospital["A"].mean_, result_after.scalers_by_hospital["A"].mean_)
        np.testing.assert_allclose(result_before.array[:3], result_after.array[:3])

    def test_categorical_columns_still_pass_through_unscaled(self):
        array = np.array([[90.0, 1.0], [100.0, 0.0], [110.0, 1.0], [5000.0, 0.0]], dtype=np.float32)
        patient_ids = [1, 2, 3, 4]
        hospital_map = {pid: "A" for pid in patient_ids}
        mask = np.array([True, False])
        train_ids = {1, 2, 3}

        result = fit_transform_tabular_local_split_aware(array, patient_ids, hospital_map, mask, train_ids)
        np.testing.assert_array_equal(result.array[:, 1], [1.0, 0.0, 1.0, 0.0])

    def test_original_function_is_untouched(self):
        array = np.array([[90.0], [100.0], [110.0]], dtype=np.float32)
        patient_ids = [1, 2, 3]
        hospital_map = {pid: "A" for pid in patient_ids}
        mask = np.array([True])
        result = fit_transform_tabular_local(array, patient_ids, hospital_map, mask)
        reference = StandardScaler()
        reference.fit(array)
        np.testing.assert_allclose(result.scalers_by_hospital["A"].mean_, reference.mean_)

    def test_hospital_with_zero_train_patients_flagged_insufficient_and_does_not_crash(self):
        array = np.array([[90.0], [100.0]], dtype=np.float32)
        patient_ids = [1, 2]
        hospital_map = {pid: "A" for pid in patient_ids}
        mask = np.array([True])
        result = fit_transform_tabular_local_split_aware(array, patient_ids, hospital_map, mask, set())
        self.assertIn("A", result.hospitals_with_insufficient_data)
        self.assertTrue(np.all(np.isfinite(result.array)))


if __name__ == "__main__":
    unittest.main()
