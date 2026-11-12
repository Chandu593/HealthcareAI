import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.temporal.normalization import fit_transform_local, fit_transform_local_split_aware
from src.data.temporal.sequence_builder import SequenceTensor


def _make_tensor(patient_ids, values_1d, seq_len=1, max_bins=1):
    n = len(patient_ids)
    array = np.zeros((n, max_bins, 1), dtype=np.float32)
    for i, v in enumerate(values_1d):
        array[i, 0, 0] = v
    return SequenceTensor(
        patient_ids=patient_ids, feature_names=["f1"], array=array,
        observed_mask=~np.isnan(array), sequence_length=np.array([seq_len] * n),
    )


class TestFitTransformLocalSplitAware(unittest.TestCase):
    def test_scaler_matches_a_direct_fit_on_train_rows_only(self):
        # Hospital A: 6 patients, values 90..140 in steps of 10; first
        # 4 are "train", last 2 are "val"/"test" and have values far
        # outside the train range -- if they leaked into fitting, the
        # scaler's mean/scale would be pulled toward them.
        patient_ids = [1, 2, 3, 4, 5, 6]
        values = [90, 100, 110, 120, 5000, 9000]
        tensor = _make_tensor(patient_ids, values)
        hospital_map = {pid: "A" for pid in patient_ids}
        train_ids = {1, 2, 3, 4}

        result = fit_transform_local_split_aware(tensor, hospital_map, train_ids)

        reference = StandardScaler()
        reference.fit(np.array([[90.0], [100.0], [110.0], [120.0]]))
        np.testing.assert_allclose(result.scalers_by_hospital["A"].mean_, reference.mean_)
        np.testing.assert_allclose(result.scalers_by_hospital["A"].scale_, reference.scale_)

    def test_val_test_rows_are_still_transformed_not_left_untouched(self):
        patient_ids = [1, 2, 3, 4]
        values = [90, 100, 110, 5000]
        tensor = _make_tensor(patient_ids, values)
        hospital_map = {pid: "A" for pid in patient_ids}
        train_ids = {1, 2, 3}

        result = fit_transform_local_split_aware(tensor, hospital_map, train_ids)
        # Patient 4 (val/test, value 5000) must be transformed using the
        # train-fit scaler, not passed through raw and not zeroed.
        transformed_val_value = result.array[3, 0, 0]
        self.assertNotEqual(transformed_val_value, 5000)
        self.assertNotEqual(transformed_val_value, 0.0)
        reference = StandardScaler()
        reference.fit(np.array([[90.0], [100.0], [110.0]]))
        expected = reference.transform(np.array([[5000.0]]))[0, 0]
        self.assertAlmostEqual(transformed_val_value, expected, places=4)

    def test_changing_val_test_values_never_changes_the_fitted_scaler(self):
        """Direct leakage-isolation proof, mirroring the PCA encoder's
        hospital-isolation test: corrupting only the val/test patients'
        raw values must leave the fitted scaler (and therefore the
        TRAIN patients' own transformed output) byte-identical."""
        patient_ids = [1, 2, 3, 4]
        hospital_map = {pid: "A" for pid in patient_ids}
        train_ids = {1, 2, 3}

        tensor_before = _make_tensor(patient_ids, [90, 100, 110, 200])
        result_before = fit_transform_local_split_aware(tensor_before, hospital_map, train_ids)

        tensor_after = _make_tensor(patient_ids, [90, 100, 110, 999999])  # corrupt only patient 4 (val/test)
        result_after = fit_transform_local_split_aware(tensor_after, hospital_map, train_ids)

        np.testing.assert_allclose(result_before.scalers_by_hospital["A"].mean_, result_after.scalers_by_hospital["A"].mean_)
        np.testing.assert_allclose(result_before.scalers_by_hospital["A"].scale_, result_after.scalers_by_hospital["A"].scale_)
        # Train patients' (0,1,2) transformed output is also unaffected.
        np.testing.assert_allclose(result_before.array[:3], result_after.array[:3])

    def test_original_fit_transform_local_is_untouched(self):
        """fit_transform_local() (the original, full-cohort-fit
        function) must produce EXACTLY the same result it always has --
        this new split-aware sibling must not have altered it."""
        patient_ids = [1, 2, 3]
        tensor = _make_tensor(patient_ids, [90, 100, 110])
        hospital_map = {pid: "A" for pid in patient_ids}
        result = fit_transform_local(tensor, hospital_map)
        reference = StandardScaler()
        reference.fit(np.array([[90.0], [100.0], [110.0]]))
        np.testing.assert_allclose(result.scalers_by_hospital["A"].mean_, reference.mean_)

    def test_hospital_with_zero_train_patients_flagged_insufficient(self):
        patient_ids = [1, 2]
        tensor = _make_tensor(patient_ids, [90, 100])
        hospital_map = {pid: "A" for pid in patient_ids}
        train_ids = set()  # nobody in this hospital is train

        result = fit_transform_local_split_aware(tensor, hospital_map, train_ids)
        self.assertIn("A", result.hospitals_with_insufficient_data)
        # Must not crash -- falls back to zero-mean imputation.
        self.assertTrue(np.all(np.isfinite(result.array)))


if __name__ == "__main__":
    unittest.main()
