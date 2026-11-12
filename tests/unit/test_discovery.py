import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.ingestion.discovery import discover_tables


def _write(dir_: Path, name: str, cols: dict) -> None:
    pd.DataFrame(cols).to_csv(dir_ / f"{name}.csv", index=False)


class TestDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_finds_confirmed_tables_case_insensitively(self):
        _write(self.dir, "Patient", {"patientunitstayid": [1], "patienthealthsystemstayid": [1],
                                       "hospitalid": [1], "unitdischargestatus": ["Alive"],
                                       "hospitaldischargestatus": ["Alive"]})
        _write(self.dir, "HOSPITAL", {"hospitalid": [1]})

        result = discover_tables(str(self.dir))
        self.assertIn("patient", result.found)
        self.assertIn("hospital", result.found)
        self.assertTrue(result.is_usable())

    def test_missing_confirmed_table_is_reported(self):
        _write(self.dir, "hospital", {"hospitalid": [1]})
        result = discover_tables(str(self.dir))
        self.assertIn("patient", result.missing_confirmed)
        self.assertFalse(result.is_usable())

    def test_all_13_tables_now_required_and_absence_flagged(self):
        # All 13 tables are now confirmed (2026-08-08) -- absence of any
        # of them (beyond patient+hospital, tested separately) shows up
        # in missing_confirmed, never missing_unconfirmed.
        _write(self.dir, "patient", {"patientunitstayid": [1], "patienthealthsystemstayid": [1],
                                       "hospitalid": [1], "unitdischargestatus": ["Alive"],
                                       "hospitaldischargestatus": ["Alive"]})
        _write(self.dir, "hospital", {"hospitalid": [1]})
        result = discover_tables(str(self.dir))
        self.assertTrue(result.is_usable())
        for required in ("treatment", "medication_imputed", "medication_imputed1"):
            self.assertIn(required, result.missing_confirmed)
        self.assertEqual(result.missing_unconfirmed, [])

    def test_medication_imputed_found_is_reported(self):
        _write(self.dir, "patient", {"patientunitstayid": [1], "patienthealthsystemstayid": [1],
                                       "hospitalid": [1], "unitdischargestatus": ["Alive"],
                                       "hospitaldischargestatus": ["Alive"]})
        _write(self.dir, "hospital", {"hospitalid": [1]})
        _write(self.dir, "medication_imputed", {"medicationid": [1], "patientunitstayid": [1],
                                                   "drugstartoffset": [10], "drugstopoffset": [20]})
        result = discover_tables(str(self.dir))
        self.assertIn("medication_imputed", result.found)

    def test_no_unconfirmed_tables_remain_in_registry(self):
        # All 13 tables the user has confirmed should be registered as
        # "confirmed" in the schema registry itself, independent of what
        # happens to be on disk in any given test.
        from src.data.ingestion.schema_registry import TABLE_REGISTRY
        unconfirmed = [t.name for t in TABLE_REGISTRY.values() if t.availability != "confirmed"]
        self.assertEqual(unconfirmed, [])

    def test_known_absent_table_found_is_flagged(self):
        _write(self.dir, "patient", {"patientunitstayid": [1], "patienthealthsystemstayid": [1],
                                       "hospitalid": [1], "unitdischargestatus": ["Alive"],
                                       "hospitaldischargestatus": ["Alive"]})
        _write(self.dir, "hospital", {"hospitalid": [1]})
        _write(self.dir, "vitalPeriodic", {"observationoffset": [1], "patientunitstayid": [1]})
        result = discover_tables(str(self.dir))
        self.assertIn("vitalPeriodic", result.known_absent_present)

    def test_missing_data_root_raises(self):
        with self.assertRaises(FileNotFoundError):
            discover_tables("/path/does/not/exist_xyz")


if __name__ == "__main__":
    unittest.main()
