import unittest

import pandas as pd

from src.data.cohort.hospital_partition import partition_by_hospital


def _patient_df(n_per_hospital: dict, expired_frac: float = 0.2) -> pd.DataFrame:
    rows = []
    pid = 1
    for hid, n in n_per_hospital.items():
        for i in range(n):
            status = "Expired" if i < int(n * expired_frac) else "Alive"
            rows.append({"patientunitstayid": pid, "hospitalid": hid, "hospitaldischargestatus": status})
            pid += 1
    return pd.DataFrame(rows)


class TestHospitalPartition(unittest.TestCase):
    def test_subset_mode_filters_to_requested_hospitals(self):
        patient_df = _patient_df({1: 10, 2: 10, 3: 10})
        hospital_df = pd.DataFrame({"hospitalid": [1, 2, 3]})

        result = partition_by_hospital(patient_df, hospital_df, mode="subset", requested_hospitals=[1, 2])
        self.assertEqual(result.eligible_hospitals, [1, 2])
        self.assertNotIn(3, result.partitions)

    def test_all_mode_uses_every_available_hospital(self):
        patient_df = _patient_df({1: 5, 2: 5})
        hospital_df = pd.DataFrame({"hospitalid": [1, 2]})
        result = partition_by_hospital(patient_df, hospital_df, mode="all")
        self.assertEqual(result.eligible_hospitals, [1, 2])

    def test_requested_hospital_not_in_data_is_excluded_not_crashed(self):
        patient_df = _patient_df({1: 5})
        hospital_df = pd.DataFrame({"hospitalid": [1]})
        result = partition_by_hospital(patient_df, hospital_df, mode="subset", requested_hospitals=[1, 99])
        self.assertEqual(result.eligible_hospitals, [1])
        self.assertEqual(result.excluded_hospitals, [99])

    def test_undersampling_does_not_crash_and_is_flagged(self):
        """Regression test for the bug in the user's original prototype:
        groupby().sample(n=N) raises if a hospital has fewer than N
        patients. This must instead sample min(N, available) and flag it."""
        patient_df = _patient_df({1: 3, 2: 100})  # hospital 1 has only 3 patients
        hospital_df = pd.DataFrame({"hospitalid": [1, 2]})

        result = partition_by_hospital(
            patient_df, hospital_df, mode="all", patient_subset=50, seed=42,
        )
        self.assertEqual(result.partitions[1].sampled_patient_count, 3)
        self.assertTrue(result.partitions[1].was_undersampled)
        self.assertEqual(result.partitions[2].sampled_patient_count, 50)
        self.assertFalse(result.partitions[2].was_undersampled)

    def test_sampling_is_deterministic_given_seed(self):
        patient_df = _patient_df({1: 100})
        hospital_df = pd.DataFrame({"hospitalid": [1]})
        r1 = partition_by_hospital(patient_df, hospital_df, mode="all", patient_subset=10, seed=7)
        r2 = partition_by_hospital(patient_df, hospital_df, mode="all", patient_subset=10, seed=7)
        self.assertEqual(sorted(r1.partitions[1].patient_ids), sorted(r2.partitions[1].patient_ids))

    def test_mortality_counts_are_correct(self):
        patient_df = _patient_df({1: 10}, expired_frac=0.3)  # 3 expired, 7 alive
        hospital_df = pd.DataFrame({"hospitalid": [1]})
        result = partition_by_hospital(patient_df, hospital_df, mode="all")
        self.assertEqual(result.partitions[1].mortality_positive_count, 3)
        self.assertEqual(result.partitions[1].mortality_negative_count, 7)

    def test_unknown_mode_raises(self):
        patient_df = _patient_df({1: 5})
        hospital_df = pd.DataFrame({"hospitalid": [1]})
        with self.assertRaises(ValueError):
            partition_by_hospital(patient_df, hospital_df, mode="bogus_mode")


if __name__ == "__main__":
    unittest.main()
