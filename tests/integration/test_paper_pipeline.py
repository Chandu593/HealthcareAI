"""End-to-end test of the paper track on mock eICU data: split reuse, window-bounded features,
every arm (incl. FedAvg + Graph none/A/B), one consistent threshold rule, patient-level predictions,
and the report."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_mock_eicu import generate  # noqa: E402
from run_pipeline import run_phase1_ingestion, run_phase2_hospital_partition, run_phase3_cohort_construction  # noqa: E402
from src.paper.runner import load_or_build_split, run_paper_track  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402

PAPER = {"features": {"min_vocab_patients": 2, "chunksize": 500}, "arms": {"mlp_local_only": True}, "seeds": [42, 43], "sepsis_hospital_filter": {"enabled": False},
         "tasks": {"mortality": ["full", "no_apache_scores"], "sepsis": ["sepsis_controlled", "sepsis_uncontrolled"]},
         "lightgbm": {"n_trials": 2, "n_estimators": 50, "early_stopping_rounds": 10, "n_jobs": 1},
         "neural": {"central_epochs": 3, "patience": 3, "hidden": [16, 8], "emb_dim": 8, "seq_hidden": 8, "local_epochs_local_only": 2,
                    "min_hospital_n_for_local_stats": 10},
         "federated": {"rounds": 3, "local_epochs": 1, "patience": 3},
         "graph": {"k": 3, "cosine_threshold": 0.0, "n_clusters": 2, "hidden": 8, "rounds": 3, "local_epochs": 2},
         "evaluation": {"bootstrap_n": 20}}


class TestPaperPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); root = Path(cls.tmp.name)
        cls.out = root / "outputs"; data = root / "mock"
        generate(str(data), n_hospitals=6, n_patients_per_hospital=120, seed=11)
        cls._old = os.environ.get("EICU_DATA_ROOT"); os.environ["EICU_DATA_ROOT"] = str(data)
        cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        cfg.raw["run"]["output_dir"] = str(cls.out); cfg.raw["run"]["log_dir"] = str(cls.out / "logs"); cfg.raw["cohort"]["patient_subset"] = 120
        cfg.raw["ablation"]["split"]["manifest_path"] = str(cls.out / "ablation" / "split_manifest.json")
        cfg.raw["paper"] = PAPER; cls.cfg = cfg
        p1 = run_phase1_ingestion(cfg); part, pdf, _ = run_phase2_hospital_partition(cfg, p1); p3 = run_phase3_cohort_construction(cfg, p1, part, pdf)
        cls.sm = load_or_build_split(cfg, part, p3)
        cls.manifest_path = Path(cfg.raw["ablation"]["split"]["manifest_path"]); cls.manifest_bytes = cls.manifest_path.read_bytes()
        cls.res = run_paper_track(cfg, p1, part, p3, "t1", split_manifest=None)
        cls.p3 = p3

    @classmethod
    def tearDownClass(cls):
        close_logging()
        if cls._old is None:
            os.environ.pop("EICU_DATA_ROOT", None)
        else:
            os.environ["EICU_DATA_ROOT"] = cls._old
        cls.tmp.cleanup()

    def test_all_experiments_and_arms_ran(self):
        self.assertEqual(set(self.res["experiments"]), {"mortality/full", "mortality/no_apache_scores", "sepsis/sepsis_controlled", "sepsis/sepsis_uncontrolled"})
        want = {"logreg_central", "lightgbm_central", "mlp_central", "mlp_local_only", "fedavg_mlp", "fedavg_graph_none", "fedavg_graph_A", "fedavg_graph_B"}
        for k, blk in self.res["experiments"].items():
            self.assertEqual(set(blk["aggregate"]), want, k)
            for a, v in blk["aggregate"].items():
                self.assertTrue(np.isfinite(v["auroc"]["mean"]), f"{k}/{a} test AUROC not finite")

    def test_existing_split_manifest_was_reused_untouched(self):
        self.assertEqual(self.manifest_path.read_bytes(), self.manifest_bytes)

    def test_one_threshold_rule_across_splits_and_calibration_fit_on_val(self):
        for blk in self.res["experiments"].values():
            for r in blk["records"]:
                self.assertEqual(r["train"]["threshold"], r["val"]["threshold"]); self.assertEqual(r["val"]["threshold"], r["test"]["threshold"])

    def test_predictions_are_patient_level_and_split_clean(self):
        d = self.out / "paper" / "t1" / "predictions"; files = sorted(d.glob("mortality__full__*seed42.npz")); self.assertGreaterEqual(len(files), 7)
        manifest_test = set(self.sm.all_patient_ids("test")); manifest_train = set(self.sm.all_patient_ids("train"))
        for f in files:
            z = np.load(f, allow_pickle=False); sp = z["split"]; pid = z["patient_id"]
            self.assertTrue(set(sp) <= {"train", "val", "test"})
            self.assertTrue(set(pid[sp == "test"].tolist()) <= manifest_test); self.assertTrue(set(pid[sp == "train"].tolist()) <= manifest_train)
            self.assertEqual(len(pid), len(set(pid.tolist())), "a patient appears twice")
            for key in ("hospital_id", "y", "score", "p_calibrated"):
                self.assertEqual(len(z[key]), len(pid))

    def test_sepsis_controlled_has_fewer_features_than_uncontrolled(self):
        e = self.res["experiments"]
        self.assertLess(e["sepsis/sepsis_controlled"]["n_features"], e["sepsis/sepsis_uncontrolled"]["n_features"])
        self.assertLess(e["mortality/no_apache_scores"]["n_features"], e["mortality/full"]["n_features"])

    def test_graph_arms_report_diagnostics_and_benchmark_present(self):
        blk = self.res["experiments"]["mortality/full"]
        g = [r for r in blk["records"] if r["arm"] == "fedavg_graph_A"][0]
        self.assertIn("homophily", g["extra"]); self.assertIn("best_round", g["extra"])
        self.assertIn("apache_iva_predicted_mortality", blk["benchmarks"])
        none = [r for r in blk["records"] if r["arm"] == "fedavg_graph_none"][0]
        self.assertEqual(none["extra"]["n_edges"], 0)

    def test_local_calibration_in_runner_matches_validation_only_offline_rescoring(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from paper_local_calibration import rescore
        blk = self.res["experiments"]["mortality/full"]; rec = [r for r in blk["records"] if r["arm"] == "lightgbm_central" and r["seed"] == 42][0]
        off = rescore(self.out / "paper" / "t1" / "predictions" / "mortality__full__lightgbm_central__seed42.npz")
        self.assertAlmostEqual(rec["local_calibrated"]["test"]["auroc"], off["local"]["auroc"], places=3)
        self.assertAlmostEqual(rec["test"]["auroc"], off["global"]["auroc"], places=3)
        self.assertIn("per_hospital_test", rec["local_calibrated"])
        self.assertIn("AUROC with local calibration", (self.out / "paper" / "t1" / "paper_report.md").read_text(encoding="utf-8"))

    def test_outputs_written(self):
        base = self.out / "paper" / "t1"
        for n in ("results.json", "results_table.csv", "paper_report.md"):
            self.assertTrue((base / n).exists(), n)
        self.assertIn("Reference results reported by the base papers", (base / "paper_report.md").read_text(encoding="utf-8"))
        json.loads((base / "results.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
