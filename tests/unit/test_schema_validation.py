import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.validation.schema_validator import validate_table


class TestSchemaValidation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_valid_patient_table_passes(self):
        df = pd.DataFrame({
            "patientunitstayid": [1, 2],
            "patienthealthsystemstayid": [10, 20],
            "hospitalid": [1, 1],
            "unitdischargestatus": ["Alive", "Expired"],
            "hospitaldischargestatus": ["Alive", "Expired"],
            "age": ["65", "70"],
        })
        path = self.dir / "patient.csv"
        df.to_csv(path, index=False)

        result = validate_table("patient", str(path))
        self.assertTrue(result.passed)
        self.assertEqual(result.missing_required_columns, [])

    def test_missing_required_column_fails(self):
        df = pd.DataFrame({
            "patientunitstayid": [1, 2],
            "hospitalid": [1, 1],
            # missing patienthealthsystemstayid, unitdischargestatus, hospitaldischargestatus
        })
        path = self.dir / "patient.csv"
        df.to_csv(path, index=False)

        result = validate_table("patient", str(path))
        self.assertFalse(result.passed)
        self.assertIn("unitdischargestatus", result.missing_required_columns)
        self.assertIn("hospitaldischargestatus", result.missing_required_columns)

    def test_unregistered_table_name_fails_explicitly(self):
        path = self.dir / "mystery.csv"
        pd.DataFrame({"a": [1]}).to_csv(path, index=False)
        result = validate_table("mystery_table_not_in_registry", str(path))
        self.assertFalse(result.passed)
        self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main()
