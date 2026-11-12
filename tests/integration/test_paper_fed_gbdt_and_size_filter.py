"""End-to-end through run_paper_track on mock eICU data: the federated-GBDT arm and the hospital-size cohort filter."""

import json
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
from src.paper.runner import run_paper_track  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402

ARMS_OFF = {"logreg_central": False, "lightgbm_central": False, "lightgbm_sampling": False, "mlp_central": False, "mlp_local_only": False,
            "fedavg_mlp": False, "fedavg_graph": False, "fedavg_logreg": False, "fed_gbdt": False}
GB = {"n_trials": 2, "max_rounds": 30, "early_stopping_rounds": 8, "n_bins": 16, "quantile_grid": 32,
      "search": {"max_depth": [3, 4], "learning_rate": [0.1, 0.2]}}
PAPER = {"features": {"min_vocab_patients": 2, "chunksize": 500}, "seeds": [42, 43], "sequence": {"enabled": False},
         "tasks": {"mortality": ["no_apache_scores"], "sepsis": []}, "arms": dict(ARMS_OFF, lightgbm_central=True, fed_gbdt=True), "fed_gbdt": GB,
         "lightgbm": {"n_trials": 1, "n_estimators": 30, "early_stopping_rounds": 5, "n_jobs": 1}, "evaluation": {"bootstrap_n": 20}}


class TestFedGBDTAndSizeFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); root = Path(cls.tmp.name); cls.out = root / "outputs"; cls.data = root / "mock"
        generate(str(cls.data), n_hospitals=6, n_patients_per_hospital=150, seed=21)
        cls._old = os.environ.get("EICU_DATA_ROOT"); os.environ["EICU_DATA_ROOT"] = str(cls.data)
        cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        cfg.raw["run"]["output_dir"] = str(cls.out); cfg.raw["run"]["log_dir"] = str(cls.out / "logs"); cfg.raw["cohort"]["patient_subset"] = 150
        cfg.raw["ablation"]["split"]["manifest_path"] = str(cls.out / "ablation" / "split_manifest.json"); cfg.raw["paper"] = PAPER
        cls.p1 = run_phase1_ingestion(cfg); cls.part, pdf, _ = run_phase2_hospital_partition(cfg, cls.p1)
        cls.p3 = run_phase3_cohort_construction(cfg, cls.p1, cls.part, pdf); cls.cfg = cfg
        cls.res = run_paper_track(cfg, cls.p1, cls.part, cls.p3, "gb1"); cls.run_dir = cls.out / "paper" / "gb1"

    @classmethod
    def tearDownClass(cls):
        close_logging()
        if cls._old is None:
            os.environ.pop("EICU_DATA_ROOT", None)
        else:
            os.environ["EICU_DATA_ROOT"] = cls._old
        cls.tmp.cleanup()

    def test_fed_gbdt_arm_runs_reports_and_reuses_the_winning_params(self):
        blk = self.res["experiments"]["mortality/no_apache_scores"]
        recs = [r for r in blk["records"] if r["arm"] == "fed_gbdt"]
        self.assertEqual([r["seed"] for r in recs], [42, 43])
        for r in recs:
            self.assertEqual(r["extra"]["n_train"], blk["split_sizes"]["train"])          # clients hold exactly the TRAIN rows: no val/test row is ever trained on
            self.assertTrue(np.isfinite(r["test"]["auroc"])); self.assertGreaterEqual(r["extra"]["best_iteration"], 1)
            self.assertGreater(r["extra"]["comm_mb_per_client_per_round_sparse"], 0)
            self.assertGreater(len(r["extra"]["top_features_gain_share"]), 0)
        self.assertEqual(len(recs[0]["extra"]["trials"]), 2)                              # the search ran once, on the first seed ...
        self.assertEqual(len(recs[1]["extra"]["trials"]), 1)                              # ... and later seeds reuse its winner
        for k in ("max_depth", "learning_rate", "reg_lambda", "colsample_bytree"):
            self.assertEqual(recs[0]["extra"]["params"][k], recs[1]["extra"]["params"][k])
        self.assertIn("fed_gbdt", blk["aggregate"]); self.assertIn("fed_gbdt", (self.run_dir / "paper_report.md").read_text(encoding="utf-8"))
        pred = np.load(self.run_dir / "predictions" / "mortality__no_apache_scores__fed_gbdt__seed42.npz")
        self.assertEqual(set(np.unique(pred["split"]).tolist()) - {"train", "val", "test"}, set())
        json.dumps(blk["records"], default=str)                                             # results.json stays serialisable

    def test_fed_gbdt_is_off_unless_asked(self):
        arms = {r["arm"] for r in self.res["experiments"]["mortality/no_apache_scores"]["records"]}
        self.assertEqual(arms, {"lightgbm_central", "fed_gbdt"})                             # only what the config switched on

    def test_hospital_size_filter_restricts_every_split_to_large_hospitals(self):
        cfg = self.cfg
        cp = self.p3["cohort_patients"]
        los = pd.to_numeric(cp["unitdischargeoffset"], errors="coerce")
        elig = cp[los >= 1440]; counts = elig.groupby("hospitalid").size().sort_values()
        self.assertGreater(counts.nunique(), 1, "premise: the landmark cohort must make hospital sizes uneven")
        thr = int(counts.iloc[len(counts) // 2])                                             # a threshold that splits the hospitals
        cfg.raw["paper"] = dict(PAPER, window={"min_icu_los_minutes": 1440}, arms=dict(ARMS_OFF, lightgbm_central=True), seeds=[42],
                                cohort_filter={"min_hospital_patients": thr})
        res = run_paper_track(cfg, self.p1, self.part, self.p3, "sz_on")
        cfg.raw["paper"] = dict(PAPER, window={"min_icu_los_minutes": 1440}, arms=dict(ARMS_OFF, lightgbm_central=True), seeds=[42])
        off = run_paper_track(cfg, self.p1, self.part, self.p3, "sz_off")
        keep = {int(h) for h in counts.index[counts >= thr]}
        b_on, b_off = res["experiments"]["mortality/no_apache_scores"], off["experiments"]["mortality/no_apache_scores"]
        self.assertIsNone(b_off["cohort_filter"]); cf = b_on["cohort_filter"]
        self.assertEqual(cf["hospitals_kept"], len(keep)); self.assertLess(cf["hospitals_kept"], cf["hospitals_total"])
        self.assertEqual(cf["patients_kept"], int(counts[counts >= thr].sum()))
        self.assertLess(sum(b_on["split_sizes"].values()), sum(b_off["split_sizes"].values()))
        pred = np.load(self.out / "paper" / "sz_on" / "predictions" / "mortality__no_apache_scores__lightgbm_central__seed42.npz")
        self.assertTrue(set(pred["hospital_id"].tolist()) <= keep)                           # no patient from a dropped hospital is in any split
        self.assertIn("hospital size >=", (self.out / "paper" / "sz_on" / "paper_report.md").read_text(encoding="utf-8"))

    def test_probe_script_end_to_end_matches_the_cohort(self):
        cp = self.p3["cohort_patients"]
        los = pd.to_numeric(cp["unitdischargeoffset"], errors="coerce"); elig = cp[los >= 1440]
        counts = elig.groupby("hospitalid").size().sort_values(ascending=False)
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "paper_hospital_size_probe.py"), "--config", str(REPO_ROOT / "configs" / "debug.yaml"),
               "--set", f"run.output_dir={self.out}", "--set", "cohort.patient_subset=150", "--set", "paper.window.min_icu_los_minutes=1440",
               "--thresholds", f"{int(counts.min())},{int(counts.max()) + 1}", "--target-hospitals", "3"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), env=dict(os.environ, EICU_DATA_ROOT=str(self.data)))
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertIn(f"eligible cohort: {len(elig):,} stays in {counts.shape[0]} hospitals", r.stdout)
        self.assertIn(f"threshold that keeps exactly the 3 largest hospitals: {int(counts.iloc[2])}", r.stdout)

    def test_fed_gbdt_on_the_sepsis_task_uses_only_label_valid_hospitals_and_known_labels(self):
        """Sepsis has two things mortality does not: a label-validity hospital filter and unknown labels (NaN). The GBDT must train, select and score
        on known labels only, and no patient from a filtered-out hospital may appear in any split."""
        from src.paper.label_validity import hospital_medication_coverage
        cov = hospital_medication_coverage(self.p1, self.p3["cohort_patients"])
        thr = float(np.sort(cov.to_numpy())[len(cov) // 2])
        keep = {int(h) for h in cov.index[cov >= thr]}
        self.cfg.raw["paper"] = dict(PAPER, arms=dict(ARMS_OFF, fed_gbdt=True), tasks={"mortality": [], "sepsis": ["sepsis_clean"]}, seeds=[42],
                                     sepsis_hospital_filter={"enabled": True, "min_medication_coverage": thr})
        try:
            res = run_paper_track(self.cfg, self.p1, self.part, self.p3, "gb_sepsis")
        finally:
            self.cfg.raw["paper"] = PAPER
        blk = res["experiments"]["sepsis/sepsis_clean"]
        rec = [r for r in blk["records"] if r["arm"] == "fed_gbdt"][0]
        self.assertTrue(np.isfinite(rec["test"]["auroc"])); self.assertTrue(np.isfinite(rec["val"]["auroc"])); self.assertGreaterEqual(rec["extra"]["best_iteration"], 1)
        self.assertEqual(rec["extra"]["n_train"], blk["split_sizes"]["train"])             # exactly the label-valid TRAIN rows
        self.assertLessEqual(rec["extra"]["n_clients"], len(keep))                          # clients are drawn from the kept hospitals only
        pred = np.load(self.out / "paper" / "gb_sepsis" / "predictions" / "sepsis__sepsis_clean__fed_gbdt__seed42.npz")
        self.assertTrue(set(pred["hospital_id"].tolist()) <= keep)
        sp = pred["split"].astype(str)
        for s in ("train", "val", "test"):
            self.assertTrue(np.all(np.isfinite(pred["y"][sp == s])), f"unknown (NaN) sepsis labels leaked into the {s} split")
        self.assertEqual(blk["cohort_filter"]["hospitals_kept"], len(keep))


    def _central(self, run_id, reuse):
        self.cfg.raw["paper"] = dict(PAPER, arms=dict(ARMS_OFF, gbdt_central=True), gbdt_central={"reuse_params_from": reuse})
        return run_paper_track(self.cfg, self.p1, self.part, self.p3, run_id)

    def test_gbdt_central_is_off_unless_asked(self):
        arms = {r["arm"] for r in self.res["experiments"]["mortality/no_apache_scores"]["records"]}
        self.assertNotIn("gbdt_central", arms)

    def test_gbdt_central_trains_on_one_client_with_the_federated_winners_parameters(self):
        res = self._central("gb_central", str(self.run_dir))                                   # reuse the winner of the federated run gb1
        blk = res["experiments"]["mortality/no_apache_scores"]
        recs = [r for r in blk["records"] if r["arm"] == "gbdt_central"]
        self.assertEqual([r["seed"] for r in recs], [42, 43]); self.assertEqual({r["arm"] for r in blk["records"]}, {"gbdt_central"})
        fed = [r for r in self.res["experiments"]["mortality/no_apache_scores"]["records"] if r["arm"] == "fed_gbdt"]
        for r in recs:
            self.assertEqual(r["extra"]["n_clients"], 1); self.assertEqual(r["extra"]["n_train"], blk["split_sizes"]["train"]); self.assertTrue(r["extra"]["pooled"])
            self.assertEqual(len(r["extra"]["trials"]), 1)                                      # no search: the federated winner is reused as given
            self.assertEqual(r["extra"]["params"], fed[0]["extra"]["params"])
            self.assertTrue(np.isfinite(r["test"]["auroc"]))
        self.assertLess(abs(recs[0]["test"]["auroc"] - fed[0]["test"]["auroc"]), 0.1)           # same code and parameters: a sanity bound on mock data, not a claim
        pred = np.load(self.out / "paper" / "gb_central" / "predictions" / "mortality__no_apache_scores__gbdt_central__seed42.npz")
        ref = np.load(self.run_dir / "predictions" / "mortality__no_apache_scores__fed_gbdt__seed42.npz")
        np.testing.assert_array_equal(pred["patient_id"], ref["patient_id"]); np.testing.assert_array_equal(pred["split"], ref["split"])    # the very same patients and splits
        self.assertIn("gbdt_central", (self.out / "paper" / "gb_central" / "paper_report.md").read_text(encoding="utf-8"))

    def test_gbdt_central_refuses_a_missing_parameter_source(self):
        with self.assertRaises(FileNotFoundError):
            self._central("gb_central_bad", str(self.out / "paper" / "no_such_run"))


if __name__ == "__main__":
    unittest.main()
