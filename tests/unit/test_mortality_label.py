import unittest

import pandas as pd

from src.data.cohort.mortality_label import mortality_label


class TestMortalityLabel(unittest.TestCase):
    def test_expired_and_alive_map_correctly(self):
        df = pd.DataFrame({"hospitaldischargestatus": ["Expired", "Alive", "Alive"]})
        labels = mortality_label(df)
        self.assertEqual(list(labels), [1, 0, 0])

    def test_unrecognized_status_becomes_nan_not_zero(self):
        df = pd.DataFrame({"hospitaldischargestatus": ["Expired", "Transferred", None]})
        labels = mortality_label(df)
        self.assertEqual(labels.iloc[0], 1)
        self.assertTrue(pd.isna(labels.iloc[1]))
        self.assertTrue(pd.isna(labels.iloc[2]))

    def test_missing_column_raises(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        with self.assertRaises(ValueError):
            mortality_label(df)


if __name__ == "__main__":
    unittest.main()
