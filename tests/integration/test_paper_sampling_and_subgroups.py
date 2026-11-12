"""Sampling arm on strong features, LightGBM importance export, name-regex feature set, and the subgroup-analysis script."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_mock_eicu import generate  # noqa: E402
from run_pipeline import run_phase1_ingestion, run_phase2_hospital_partition, run_phase3_cohort_construction  # noqa: E402
from src.paper.features import FeatureBundle  # noqa: E402
from src.paper.runner import run_paper_track  # noqa: E402
from src.paper.subgroups import render_markdown, subgroup_report  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402

PAPER = {"features": {"min_vocab_patients": 2, "chunksize": 500}, "seeds": [42],
         "tasks": {"mortality": ["no_apache_no_eol"], "sepsis": []},
         "arms": {"logreg_central": False, "lightgbm_central": True, "lightgbm_sampling": True, "mlp_central": False, "fedavg_mlp": False, "fedavg_graph": False},
         "lightgbm": {"n_trials": 2, "n_estimators": 50, "early_stopping_rounds": 10, "n_jobs": 1},
         "sampling": {"strategies": ["undersample", "oversample"], "ratios": [0.5, 1.0]}, "evaluation": {"bootstrap_n": 20}}


class TestSamplingAndSubgroups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); root = Path(cls.tmp.name); cls.out = root / "outputs"; cls.data = root / "mock"
        generate(str(cls.data), n_hospitals=6, n_patients_per_hospital=150, seed=21)
        cls._old = os.environ.get("EICU_DATA_ROOT"); os.environ["EICU_DATA_ROOT"] = str(cls.data)
        cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        cfg.raw["run"]["output_dir"] = str(cls.out); cfg.raw["run"]["log_dir"] = str(cls.out / "logs"); cfg.raw["cohort"]["patient_subset"] = 150
        cfg.raw["ablation"]["split"]["manifest_path"] = str(cls.out / "ablation" / "split_manifest.json"); cfg.raw["paper"] = PAPER
        p1 = run_phase1_ingestion(cfg); part, pdf, _ = run_phase2_hospital_partition(cfg, p1); p3 = run_phase3_cohort_construction(cfg, p1, part, pdf)
        cls.res = run_paper_track(cfg, p1, part, p3, "s1"); cls.run_dir = cls.out / "paper" / "s1"
        cls.cfg, cls.p1, cls.part, cls.p3 = cfg, p1, part, p3

    @classmethod
    def tearDownClass(cls):
        close_logging()
        if cls._old is None:
            os.environ.pop("EICU_DATA_ROOT", None)
        else:
            os.environ["EICU_DATA_ROOT"] = cls._old
        cls.tmp.cleanup()

    def test_sampling_arms_have_ratio_chosen_on_validation_and_importance_exported(self):
        blk = self.res["experiments"]["mortality/no_apache_no_eol"]
        arms = {r["arm"]: r for r in blk["records"]}
        self.assertEqual(set(arms), {"lightgbm_central", "lightgbm_undersample", "lightgbm_oversample"})
        for a in ("lightgbm_undersample", "lightgbm_oversample"):
            self.assertIn(arms[a]["extra"]["ratio"], (0.5, 1.0)); self.assertEqual(set(arms[a]["extra"]["ratio_search_val_auprc"]), {"0.5", "1.0"})
        top = arms["lightgbm_central"]["extra"]["top_features_gain_share"]
        self.assertGreater(len(top), 0); self.assertAlmostEqual(sum(v for _, v in top) <= 1.0 + 1e-6, True)
        self.assertIn("LightGBM top features", (self.run_dir / "paper_report.md").read_text(encoding="utf-8"))

    def test_strict_landmark_cohort_keeps_only_full_window_stays(self):
        cp = self.p3["cohort_patients"]; ok = set(cp.loc[pd.to_numeric(cp["unitdischargeoffset"], errors="coerce") >= 1440, "patientunitstayid"].astype(int))
        self.cfg.raw["paper"] = dict(PAPER, window={"min_icu_los_minutes": 1440}, arms={"logreg_central": True, "lightgbm_central": False, "lightgbm_sampling": False,
                                                                                           "mlp_central": False, "fedavg_mlp": False, "fedavg_graph": False})
        res = run_paper_track(self.cfg, self.p1, self.part, self.p3, "lm")
        sizes = res["experiments"]["mortality/no_apache_no_eol"]["split_sizes"]
        self.assertLess(res["n_patients"], len(cp)); self.assertEqual(res["n_patients"], len(ok))
        self.assertLessEqual(sum(sizes.values()), len(ok))

    def test_name_regex_feature_set_drops_matching_columns(self):
        b = FeatureBundle(np.arange(3), np.zeros(3, int), np.zeros((3, 4), "float32"), ["tx_palliative_care", "tx_vent", "lab_a_mean", "dx_comfort_measures"],
                          ["treatment", "treatment", "lab", "diagnosis"], np.zeros(3), np.zeros(3), np.zeros(3))
        kept = [b.feature_names[i] for i in b.columns_excluding([], ["palliat", "comfort"])]
        self.assertEqual(kept, ["tx_vent", "lab_a_mean"])
        self.assertEqual(len(b.columns_excluding(["lab"], None)), 3)

    def test_subgroup_script_end_to_end(self):
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "paper_subgroup_analysis.py"), "--run-dir", str(self.run_dir), "--experiment", "mortality__no_apache_no_eol",
               "--arm", "lightgbm_central", "--data-root", str(self.data)]
        r = subprocess.run(cmd, capture_output=True, text=True); self.assertEqual(r.returncode, 0, r.stderr[-800:])
        md = (self.run_dir / "subgroups__mortality__no_apache_no_eol__lightgbm_central.md").read_text(encoding="utf-8")
        self.assertIn("overall (test)", md); self.assertIn("classification", md.lower().replace("_", " ") + "classification")

    def test_subgroup_strata_partition_the_test_set_and_match_apache_rows(self):
        rng = np.random.default_rng(0); n = 4000
        y = (rng.random(n) < 0.1).astype(float); p = np.clip(0.08 + 0.3 * y + rng.normal(0, 0.12, n), 0.001, 0.999)
        pred = {"split": np.array(["test"] * n), "patient_id": np.arange(n) + 1, "hospital_id": rng.integers(1, 5, n), "y": y, "p_calibrated": p}
        pat = pd.DataFrame({"patientunitstayid": np.arange(n) + 1, "unitdischargeoffset": rng.integers(100, 6000, n), "age": rng.integers(20, 95, n).astype(str),
                            "unittype": rng.choice(["MICU", "SICU"], n)})
        apache = pd.DataFrame({"patientunitstayid": np.arange(n) + 1, "predictedhospitalmortality": np.where(rng.random(n) < 0.7, np.clip(p + rng.normal(0, 0.2, n), 0, 1), -1)})
        rep = subgroup_report(pred, pat, apache, 0.2, 1440, min_n=50, min_pos=5)
        los = list(rep["by_icu_los"].values()); self.assertEqual(sum(v["n"] for v in los), n)
        self.assertEqual(sum(v["n"] for v in rep["by_age"].values()), n)
        va = rep["vs_apache_same_patients"]; self.assertLess(va["n"], n); self.assertGreater(va["n"], 0.6 * n)
        self.assertIn("negative (0)", rep["classification_report"]); self.assertIn("overall", render_markdown("t", rep))


if __name__ == "__main__":
    unittest.main()


class TestRunControls(unittest.TestCase):
    """--feature-sets filter, incremental result writing, and the skipped-comparison note."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); root = Path(cls.tmp.name); cls.out = root / "outputs"; data = root / "mock"
        generate(str(data), n_hospitals=5, n_patients_per_hospital=120, seed=4)
        cls._old = os.environ.get("EICU_DATA_ROOT"); os.environ["EICU_DATA_ROOT"] = str(data)
        cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        cfg.raw["run"]["output_dir"] = str(cls.out); cfg.raw["run"]["log_dir"] = str(cls.out / "logs"); cfg.raw["cohort"]["patient_subset"] = 120
        cfg.raw["ablation"]["split"]["manifest_path"] = str(cls.out / "ablation" / "split_manifest.json")
        cls.paper = dict(PAPER, tasks={"mortality": ["full", "no_apache_no_eol"], "sepsis": []},
                         arms={"logreg_central": True, "lightgbm_central": False, "lightgbm_sampling": False, "mlp_central": False, "fedavg_mlp": False, "fedavg_graph": False})
        cfg.raw["paper"] = cls.paper; cls.cfg = cfg
        cls.p1 = run_phase1_ingestion(cfg); cls.part, pdf, _ = run_phase2_hospital_partition(cfg, cls.p1); cls.p3 = run_phase3_cohort_construction(cfg, cls.p1, cls.part, pdf)

    @classmethod
    def tearDownClass(cls):
        close_logging()
        if cls._old is None:
            os.environ.pop("EICU_DATA_ROOT", None)
        else:
            os.environ["EICU_DATA_ROOT"] = cls._old
        cls.tmp.cleanup()

    def test_feature_set_filter(self):
        res = run_paper_track(self.cfg, self.p1, self.part, self.p3, "fsf", fs_filter={"no_apache_no_eol"})
        self.assertEqual(list(res["experiments"]), ["mortality/no_apache_no_eol"])

    def test_results_are_on_disk_even_if_a_later_experiment_crashes(self):
        import src.paper.runner as R
        real = R._run_task_featureset; calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated crash in the second experiment")
            return real(*a, **k)
        R._run_task_featureset = flaky
        try:
            with self.assertRaises(RuntimeError):
                run_paper_track(self.cfg, self.p1, self.part, self.p3, "crash")
        finally:
            R._run_task_featureset = real
        import json
        saved = json.loads((self.out / "paper" / "crash" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(list(saved["experiments"]), ["mortality/full"])
        self.assertTrue((self.out / "paper" / "crash" / "paper_report.md").exists())

    def test_report_says_when_reference_arm_missing(self):
        self.cfg.raw["paper"] = dict(self.paper, arms=dict(self.paper["arms"], logreg_central=True, mlp_central=True), neural={"central_epochs": 2, "patience": 2, "hidden": [16, 8], "emb_dim": 8, "seq_hidden": 8, "min_hospital_n_for_local_stats": 10})
        res = run_paper_track(self.cfg, self.p1, self.part, self.p3, "nocmp", fs_filter={"full"})
        self.assertIn("Paired comparisons skipped", (self.out / "paper" / "nocmp" / "paper_report.md").read_text(encoding="utf-8"))
        self.cfg.raw["paper"] = self.paper


class TestLogregArmAndSequenceSwitch(TestRunControls):
    def test_fedavg_logreg_arm_and_use_sequence_switch(self):
        arms = {"logreg_central": False, "lightgbm_central": False, "lightgbm_sampling": False, "mlp_central": True, "fedavg_mlp": False, "fedavg_graph": False, "fedavg_logreg": True}
        neural = {"central_epochs": 2, "patience": 2, "hidden": [16, 8], "emb_dim": 8, "seq_hidden": 8, "min_hospital_n_for_local_stats": 10, "use_sequence": False}
        self.cfg.raw["paper"] = dict(self.paper, arms=arms, neural=neural, federated={"rounds": 2, "local_epochs": 1, "patience": 2},
                                     federated_logreg={"rounds": 3, "local_epochs": 1, "lr": 0.02, "batch_size": 32, "patience": 3}, tasks={"mortality": ["full"], "sepsis": []})
        res = run_paper_track(self.cfg, self.p1, self.part, self.p3, "lr1")
        agg = res["experiments"]["mortality/full"]["aggregate"]
        self.assertEqual(set(agg), {"mlp_central", "fedavg_logreg"}); self.assertTrue(np.isfinite(agg["fedavg_logreg"]["auroc"]["mean"]))
        self.cfg.raw["paper"] = self.paper


class TestSepsisHospitalFilter(TestRunControls):
    def _run(self, hf, rid):
        arms = {"logreg_central": True, "lightgbm_central": False, "lightgbm_sampling": False, "mlp_central": False, "fedavg_mlp": False, "fedavg_graph": False, "fedavg_logreg": False}
        self.cfg.raw["paper"] = dict(self.paper, arms=arms, tasks={"mortality": [], "sepsis": ["sepsis_clean"]}, sepsis_hospital_filter=hf)
        res = run_paper_track(self.cfg, self.p1, self.part, self.p3, rid); self.cfg.raw["paper"] = self.paper
        return res

    def test_filter_restricts_sepsis_to_covered_hospitals_and_is_reported(self):
        import pandas as pd
        from src.paper.label_validity import hospital_medication_coverage
        cov = hospital_medication_coverage(self.p1, self.p3["cohort_patients"])
        thr = float(np.sort(cov.to_numpy())[len(cov) // 2])                         # a threshold that splits the mock hospitals
        off = self._run({"enabled": False, "min_medication_coverage": thr}, "hf_off")["experiments"]["sepsis/sepsis_clean"]
        on_res = self._run({"enabled": True, "min_medication_coverage": thr}, "hf_on")["experiments"]["sepsis/sepsis_clean"]
        self.assertIsNone(off["cohort_filter"]); cf = on_res["cohort_filter"]
        self.assertLessEqual(cf["hospitals_kept"], cf["hospitals_total"]); self.assertEqual(cf["hospitals_kept"], int((cov >= thr).sum()))
        self.assertLessEqual(sum(on_res["split_sizes"].values()), sum(off["split_sizes"].values()))
        self.assertIn("Cohort filter", (self.out / "paper" / "hf_on" / "paper_report.md").read_text(encoding="utf-8"))
        pred = np.load(self.out / "paper" / "hf_on" / "predictions" / "sepsis__sepsis_clean__logreg_central__seed42.npz")
        self.assertTrue(set(pred["hospital_id"].tolist()) <= {int(h) for h in cov.index[cov >= thr]})      # no patient from a dropped hospital is in any split
