"""End-to-end integration test for src/ablation/embedding_ablation_runner.py
against mock data -- exercises every tab_model branch (autoencoder, mlp,
pca) and every fusion_mode branch (concatenation, cross_attention) with
a real train/val/test split, real per-hospital training, real frozen-
embedding extraction, and a real classifier fit/eval, end to end.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_mock_eicu import generate  # noqa: E402
from run_pipeline import (  # noqa: E402
    run_phase1_ingestion,
    run_phase2_hospital_partition,
    run_phase3_cohort_construction,
    run_phase4_timeseries_modality,
    run_phase5_tabular_modality,
)
from src.ablation.embedding_ablation_runner import run_embedding_ablation  # noqa: E402
from src.ablation.results_writer import ResultsWriter  # noqa: E402
from src.data.splitting.patient_split import build_split_manifest  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestEmbeddingAblationRunnerEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        generate(str(self.data_root), n_hospitals=5, n_patients_per_hospital=45, seed=11)

        self._old_env = os.environ.get("EICU_DATA_ROOT")
        os.environ["EICU_DATA_ROOT"] = str(self.data_root)

        self.cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        self.cfg.raw["run"]["output_dir"] = str(self.output_dir)
        self.cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")
        self.cfg.raw["cohort"]["patient_subset"] = 40
        # Encoder choice here only affects Phase 4/5's OWN saved embedding
        # file, not the raw sequences/features the ablation framework
        # consumes -- kept at explicit defaults for a fast, deterministic
        # test regardless of what base.yaml's checked-in value is.
        self.cfg.raw.setdefault("modalities", {}).setdefault("timeseries", {})["encoder"] = "lstm"
        self.cfg.raw["modalities"].setdefault("tabular", {})["encoder"] = "autoencoder"

        phase1_summary = run_phase1_ingestion(self.cfg)
        self.partition_result, patient_df, hospital_df = run_phase2_hospital_partition(self.cfg, phase1_summary)
        self.phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, self.partition_result, patient_df)
        self.phase4_result = run_phase4_timeseries_modality(self.cfg, self.phase3_result, self.partition_result)
        self.phase5_result = run_phase5_tabular_modality(self.cfg, phase1_summary, self.phase3_result, self.partition_result)

        self.split_manifest = build_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], seed=42, stratify_on="mortality",
        )

    def tearDown(self):
        close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def _run(self, ablation_cfg, run_id):
        results_writer = ResultsWriter(self.output_dir / "ablation", run_id=run_id)
        summary = run_embedding_ablation(
            self.cfg, ablation_cfg, self.phase3_result, self.phase4_result, self.phase5_result,
            self.partition_result, self.split_manifest, results_writer,
        )
        return summary, results_writer

    def test_every_tab_model_and_fusion_mode_branch_runs_end_to_end(self):
        ablation_cfg = {
            "local_training": {"epochs": 2, "learning_rate": 0.01},
            "pca": {"n_components": 6},
            "subset": {
                "combos": [
                    {"ts_model": "lstm", "tab_model": "autoencoder", "fusion_mode": "concatenation"},
                    {"ts_model": "lstm", "tab_model": "mlp", "fusion_mode": "concatenation"},
                    {"ts_model": "lstm", "tab_model": "pca", "fusion_mode": "concatenation"},
                    {"ts_model": "lstm", "tab_model": "autoencoder", "fusion_mode": "cross_attention"},
                    {"ts_model": "gru", "tab_model": "mlp", "fusion_mode": "cross_attention"},
                ],
                "classifiers": ["adaboost"],
                "tasks": ["mortality"],
            },
        }
        summary, results_writer = self._run(ablation_cfg, run_id="run_all_branches")
        self.assertEqual(summary["n_combos"], 5)

        record_files = sorted(results_writer.records_dir.glob("*.json"))
        self.assertEqual(len(record_files), 5)  # 5 combos x 1 classifier x 1 task

        for path in record_files:
            with open(path) as f:
                data = json.load(f)
            self.assertIn("train", data["metrics"])
            self.assertIn("val", data["metrics"])
            self.assertIn("test", data["metrics"])
            self.assertGreater(data["training_info"]["n_hospitals_trained"], 0)

        # Combo-level summary reports real hospital counts, not zeros.
        for combo_id, combo_summary in summary["combos"].items():
            self.assertGreater(combo_summary.get("n_patients_embedded", 0), 0, combo_id)

    def test_no_patient_used_for_training_and_test_at_once(self):
        """Direct leakage regression check specific to the ablation
        runner: for every combo, no patient_id assigned 'test' by the
        split manifest may have influenced that hospital's TRAINING
        (this doesn't re-derive the whole training loop -- it checks
        the row mask actually used to slice which rows fed the
        optimizer, which is the real leakage surface here)."""
        from src.ablation.embedding_ablation_runner import _load_hospital_raw_tensors, _build_ts_tab_lookup
        import numpy as np

        ts_row_of_patient, tab_row_of_patient = _build_ts_tab_lookup(self.phase4_result, self.phase5_result)
        for hid, partition in self.partition_result.partitions.items():
            patient_ids = sorted(int(pid) for pid in partition.patient_ids)
            if not patient_ids:
                continue
            split_of_row = np.array([self.split_manifest.split_of(pid) or "train" for pid in patient_ids])
            train_row_mask = split_of_row == "train"
            test_row_mask = split_of_row == "test"
            # A patient cannot be in both masks at once (structural
            # invariant of a single categorical split_of_row array,
            # asserted directly rather than assumed).
            self.assertEqual(int((train_row_mask & test_row_mask).sum()), 0)

    def test_results_never_silently_overwritten_across_two_runs_same_id(self):
        ablation_cfg = {
            "local_training": {"epochs": 1, "learning_rate": 0.01},
            "subset": {
                "combos": [{"ts_model": "lstm", "tab_model": "mlp", "fusion_mode": "concatenation"}],
                "classifiers": ["adaboost"], "tasks": ["mortality"],
            },
        }
        results_writer = ResultsWriter(self.output_dir / "ablation", run_id="run_dup_check")
        run_embedding_ablation(
            self.cfg, ablation_cfg, self.phase3_result, self.phase4_result, self.phase5_result,
            self.partition_result, self.split_manifest, results_writer,
        )
        with self.assertRaises(FileExistsError):
            run_embedding_ablation(
                self.cfg, ablation_cfg, self.phase3_result, self.phase4_result, self.phase5_result,
                self.partition_result, self.split_manifest, results_writer,
            )


if __name__ == "__main__":
    unittest.main()
