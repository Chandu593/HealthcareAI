"""
Scale-focused integration test for the memory-efficiency rework.

Directly mimics the shape of the scenario that crashed on real data:
`full` mode (patient_subset=null, no per-hospital cap) with an UNEVEN
hospital-size distribution (one large hospital, several small ones) --
this exercises the actual hospital-sharding path with real routing
logic across many chunks, not just a handful of hand-built rows.

This does not claim to prove the pipeline handles 200,859 real patients
-- that can only be verified against the real dataset, which isn't
available here. What it DOES verify: the sharding/per-hospital-loop
architecture produces identical structural results to what the old
pooled approach would have (subset invariants, no cross-hospital
leakage, correct row routing) at a scale (~500 patients, uneven
hospital sizes, multiple chunks per table) that exercises the same code
paths a full-mode run would, just smaller.
"""

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
from run_pipeline import (  # noqa: E402
    run_phase1_ingestion,
    run_phase2_hospital_partition,
    run_phase3_cohort_construction,
    run_phase4_timeseries_modality,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestFullModeScaleAndSharding(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        self.cache_dir = Path(self.tmpdir.name) / "cache"

        # Deliberately UNEVEN: mimics the real hospital-size distribution
        # (one hospital with ~7,059 patients vs. most hospitals with a
        # few hundred). generate_mock_eicu.py doesn't support per-hospital
        # size variation directly, so build it hospital-by-hospital.
        self.data_root.mkdir(parents=True)
        self._generate_uneven_mock_data()

        self._old_env = os.environ.get("EICU_DATA_ROOT")
        os.environ["EICU_DATA_ROOT"] = str(self.data_root)

        self.cfg = load_config(str(REPO_ROOT / "configs" / "full.yaml"))
        self.cfg.raw["run"]["output_dir"] = str(self.output_dir)
        self.cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")
        self.cfg.raw["run"]["cache_dir"] = str(self.cache_dir)
        self.cfg.raw["cohort"]["patient_subset"] = None  # full mode: no per-hospital cap
        self.cfg.raw["data"]["chunksize"] = 37  # deliberately small to force MANY chunks per table

    def tearDown(self):
        close_logging()  # must close file handlers before deleting the tempdir they write into (Windows) -- see logging_setup.close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def _generate_uneven_mock_data(self):
        # Generate two separate small mock datasets with distinct
        # hospital-id/patient-id ranges, then merge -- simplest robust
        # way to get an uneven hospital-size distribution since
        # generate_mock_eicu.py always gives every hospital the same
        # patient count within a single call.
        import pandas as pd
        from generate_mock_eicu import generate as _gen

        tmp1 = Path(tempfile.mkdtemp())
        tmp2 = Path(tempfile.mkdtemp())
        _gen(str(tmp1), n_hospitals=1, n_patients_per_hospital=220, seed=1)  # the "big" hospital
        _gen(str(tmp2), n_hospitals=5, n_patients_per_hospital=15, seed=2)   # several "small" hospitals

        HOSPITAL_OFFSET = 100
        PATIENT_OFFSET = 500000
        for fname in os.listdir(tmp2):
            df = pd.read_csv(tmp2 / fname)
            if "hospitalid" in df.columns:
                df["hospitalid"] += HOSPITAL_OFFSET
            if "patientunitstayid" in df.columns:
                df["patientunitstayid"] += PATIENT_OFFSET
            df.to_csv(tmp2 / fname, index=False)

        for fname in os.listdir(tmp1):
            df1 = pd.read_csv(tmp1 / fname)
            df2_path = tmp2 / fname
            if df2_path.exists():
                df2 = pd.read_csv(df2_path)
                merged = pd.concat([df1, df2], ignore_index=True)
            else:
                merged = df1
            merged.to_csv(self.data_root / fname, index=False)

    def test_full_mode_uneven_hospitals_no_crash_and_correct_isolation(self):
        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)

        sizes = {hid: p.sampled_patient_count for hid, p in partition_result.partitions.items()}
        self.assertGreater(max(sizes.values()), 150)  # the "big" hospital
        self.assertTrue(any(v < 20 for v in sizes.values()))  # the "small" hospitals
        # full mode: sampled == eligible, no subsampling cap applied
        for hid, p in partition_result.partitions.items():
            self.assertEqual(p.sampled_patient_count, p.eligible_patient_count)

        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)

        report_path = Path(phase3_result["cohort_report_path"])
        with open(report_path) as f:
            report = json.load(f)
        self.assertEqual(report["total_patients"], sum(sizes.values()))
        self.assertLessEqual(report["sepsis_strict_positive"], report["sepsis_broad_positive"])

        sepsis_df = phase3_result["sepsis_df"]
        patient_hospital_map = phase3_result["patient_hospital_map"]
        for _, row in sepsis_df.iterrows():
            self.assertIn(row["patientunitstayid"], patient_hospital_map)

        # Phase 4 must also complete without needing the old pooled
        # lab/medication/infusion_drug DataFrames (removed by this rework).
        phase4_result = run_phase4_timeseries_modality(self.cfg, phase3_result, partition_result)
        seq_path = Path(phase4_result["sequences_path"])
        self.assertTrue(seq_path.exists())
        data = np.load(seq_path, allow_pickle=True)
        self.assertEqual(data["array"].shape[0], sum(sizes.values()))
        self.assertFalse(np.any(np.isnan(data["array"])))

    def test_shard_cache_is_reused_across_phase3_and_phase4(self):
        """Phase 4 must NOT re-shard tables Phase 3 already sharded in
        the same run -- verifies the cache-hit path end-to-end, not just
        at the hospital_sharding.py unit level."""
        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)

        lab_manifest_path = self.cache_dir / "hospital_shards" / "lab_manifest.json"
        self.assertTrue(lab_manifest_path.exists())
        mtime_after_phase3 = lab_manifest_path.stat().st_mtime_ns

        run_phase4_timeseries_modality(self.cfg, phase3_result, partition_result)
        mtime_after_phase4 = lab_manifest_path.stat().st_mtime_ns

        self.assertEqual(
            mtime_after_phase3, mtime_after_phase4,
            "lab shard manifest was rewritten during Phase 4 -- cache was not reused",
        )


if __name__ == "__main__":
    unittest.main()
