import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))

from paper_eol_audit import audit  # noqa: E402
from src.paper.gbdt import tune_and_fit_lightgbm  # noqa: E402
from src.paper.overrides import apply_set_overrides  # noqa: E402


class TestOverrides(unittest.TestCase):
    def test_nested_keys_yaml_values_and_errors(self):
        raw = {"paper": {"window": {"minutes": 1440}}}
        apply_set_overrides(raw, ["paper.window.min_icu_los_minutes=1440", "paper.seeds=[1, 2]", "paper.arms.mlp_local_only=true", "new.branch.x=0.5"])
        self.assertEqual(raw["paper"]["window"], {"minutes": 1440, "min_icu_los_minutes": 1440})
        self.assertEqual(raw["paper"]["seeds"], [1, 2]); self.assertIs(raw["paper"]["arms"]["mlp_local_only"], True); self.assertEqual(raw["new"]["branch"]["x"], 0.5)
        with self.assertRaises(ValueError):
            apply_set_overrides(raw, ["no_equals_sign"])


class TestEolAudit(unittest.TestCase):
    def test_counts_only_inside_window_and_reports_mortality(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pd.DataFrame({"patientunitstayid": [1, 2, 3, 4], "unitdischargeoffset": [3000, 3000, 3000, 200],
                          "hospitaldischargestatus": ["Expired", "Alive", "Alive", "Expired"]}).to_csv(root / "patient.csv", index=False)
            pd.DataFrame({"patientunitstayid": [1, 2, 3, 4], "treatmentoffset": [100, 100, 2000, 500],      # 3: after window; 4: after its own discharge
                          "treatmentstring": ["palliative care|comfort", "ventilation", "palliative care", "palliative care"]}).to_csv(root / "treatment.csv", index=False)
            r = audit(root, window=1440, min_los=0)
            self.assertEqual((r["n_patients"], r["n_flagged"]), (4, 1))
            self.assertAlmostEqual(r["mortality_flagged"], 1.0); self.assertAlmostEqual(r["share_of_all_deaths_flagged"], 0.5)
            self.assertEqual(audit(root, window=1440, min_los=1440)["n_patients"], 3)         # strict cohort drops the 200-min stay


class TestLightGBMNoDeprecation(unittest.TestCase):
    def test_no_eval_set_deprecation_warning(self):
        rng = np.random.default_rng(0); X = rng.normal(size=(400, 6)); y = (X[:, 0] + rng.normal(size=400) > 1).astype(int)
        lcfg = {"n_trials": 1, "n_estimators": 20, "early_stopping_rounds": 5, "n_jobs": 1,
                "search": {"learning_rate": [0.1], "num_leaves": [7], "min_child_samples": [5], "feature_fraction": [1.0], "bagging_fraction": [1.0], "lambda_l2": [0.0], "scale_pos_weight_mode": ["none"]}}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m, info = tune_and_fit_lightgbm(X[:300], y[:300], X[300:], y[300:], lcfg, 1)
        self.assertFalse([x for x in w if "eval_set" in str(x.message)])
        self.assertGreater(info["best_val_auroc"], 0.5)


if __name__ == "__main__":
    unittest.main()
