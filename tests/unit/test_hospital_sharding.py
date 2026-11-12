import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.ingestion.hospital_sharding import (
    load_hospital_shard,
    shard_table_by_hospital,
)


class TestHospitalSharding(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)
        self.cache_dir = str(self.dir / "cache")

        # 3 patients across 2 hospitals, with extra columns that should
        # be dropped by column projection.
        self.lab_df = pd.DataFrame([
            {"labid": 1, "patientunitstayid": 1, "labresultoffset": 10, "labname": "platelets",
             "labresult": 200.0, "labtypeid": 1, "labresulttext": "200"},
            {"labid": 2, "patientunitstayid": 1, "labresultoffset": 300, "labname": "creatinine",
             "labresult": 1.1, "labtypeid": 1, "labresulttext": "1.1"},
            {"labid": 3, "patientunitstayid": 2, "labresultoffset": 20, "labname": "platelets",
             "labresult": 150.0, "labtypeid": 1, "labresulttext": "150"},
            {"labid": 4, "patientunitstayid": 3, "labresultoffset": 5, "labname": "pH",
             "labresult": 7.3, "labtypeid": 1, "labresulttext": "7.3"},
            # patient 99 is NOT in the cohort -- must be excluded entirely
            {"labid": 5, "patientunitstayid": 99, "labresultoffset": 5, "labname": "pH",
             "labresult": 7.1, "labtypeid": 1, "labresulttext": "7.1"},
        ])
        self.lab_path = self.dir / "lab.csv"
        self.lab_df.to_csv(self.lab_path, index=False)

        # patients 1,2 -> hospital 10; patient 3 -> hospital 20
        self.patient_hospital_map = {1: 10, 2: 10, 3: 20}

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_shard_routes_rows_to_correct_hospital(self):
        manifest = shard_table_by_hospital(
            str(self.lab_path), "lab", self.patient_hospital_map, self.cache_dir, chunksize=2,
        )
        hosp10 = load_hospital_shard(manifest, 10)
        hosp20 = load_hospital_shard(manifest, 20)

        self.assertEqual(set(hosp10["patientunitstayid"]), {1, 2})
        self.assertEqual(set(hosp20["patientunitstayid"]), {3})
        # No cross-hospital leakage
        self.assertNotIn(3, hosp10["patientunitstayid"].values)
        self.assertNotIn(1, hosp20["patientunitstayid"].values)

    def test_column_projection_drops_unused_columns(self):
        manifest = shard_table_by_hospital(
            str(self.lab_path), "lab", self.patient_hospital_map, self.cache_dir, chunksize=2,
        )
        hosp10 = load_hospital_shard(manifest, 10)
        self.assertNotIn("labid", hosp10.columns)
        self.assertNotIn("labtypeid", hosp10.columns)
        self.assertNotIn("labresulttext", hosp10.columns)
        self.assertIn("labname", hosp10.columns)
        self.assertIn("labresult", hosp10.columns)

    def test_patient_not_in_cohort_is_excluded(self):
        manifest = shard_table_by_hospital(
            str(self.lab_path), "lab", self.patient_hospital_map, self.cache_dir, chunksize=2,
        )
        all_patient_ids = set()
        for hid in (10, 20):
            all_patient_ids.update(load_hospital_shard(manifest, hid)["patientunitstayid"])
        self.assertNotIn(99, all_patient_ids)

    def test_hospital_with_zero_rows_returns_empty_frame_not_error(self):
        manifest = shard_table_by_hospital(
            str(self.lab_path), "lab", self.patient_hospital_map, self.cache_dir, chunksize=2,
        )
        # hospital 30 has no patients in patient_hospital_map at all
        result = load_hospital_shard(manifest, 30)
        self.assertEqual(len(result), 0)
        self.assertIn("patientunitstayid", result.columns)

    def test_data_correctness_survives_small_chunksize(self):
        """chunksize=2 forces multiple chunks per hospital -- verifies
        the persistent-file-handle-across-chunks logic doesn't drop or
        duplicate rows, and headers aren't written more than once."""
        manifest = shard_table_by_hospital(
            str(self.lab_path), "lab", self.patient_hospital_map, self.cache_dir, chunksize=2,
        )
        hosp10 = load_hospital_shard(manifest, 10)
        # patient 1: 2 rows (labid 1,2), patient 2: 1 row (labid 3) -> hospital 10 total = 3
        self.assertEqual(len(hosp10), 3)

    def test_cache_reused_on_second_call(self):
        manifest1 = shard_table_by_hospital(
            str(self.lab_path), "lab", self.patient_hospital_map, self.cache_dir, chunksize=2,
        )
        # Mutate the shard file directly to prove the SECOND call doesn't re-shard
        shard_path = Path(manifest1.hospital_shard_paths["10"])
        original_content = shard_path.read_text()
        shard_path.write_text(original_content + "\n# tampered marker, should survive if cache is reused")

        manifest2 = shard_table_by_hospital(
            str(self.lab_path), "lab", self.patient_hospital_map, self.cache_dir, chunksize=2,
        )
        self.assertEqual(manifest1.cache_key, manifest2.cache_key)
        self.assertIn("tampered marker", shard_path.read_text())

    def test_cache_invalidated_when_cohort_changes(self):
        manifest1 = shard_table_by_hospital(
            str(self.lab_path), "lab", self.patient_hospital_map, self.cache_dir, chunksize=2,
        )
        different_cohort = {1: 10}  # only patient 1 now
        manifest2 = shard_table_by_hospital(
            str(self.lab_path), "lab", different_cohort, self.cache_dir, chunksize=2,
        )
        self.assertNotEqual(manifest1.cache_key, manifest2.cache_key)
        hosp10_v2 = load_hospital_shard(manifest2, 10)
        self.assertEqual(set(hosp10_v2["patientunitstayid"]), {1})

    def test_unregistered_table_name_raises(self):
        with self.assertRaises(ValueError):
            shard_table_by_hospital(
                str(self.lab_path), "not_a_registered_table", self.patient_hospital_map, self.cache_dir,
            )


if __name__ == "__main__":
    unittest.main()
