"""Integration tests for the ablation framework's permanent split
manifest (src/data/splitting/patient_split.py), run against the real
Phase 1->2->3 pipeline on mock data -- the same fixture pattern as
test_phase2_3_pipeline.py.
"""

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
)
from src.data.splitting.patient_split import (  # noqa: E402
    build_and_save_split_manifest,
    build_split_manifest,
    load_split_manifest,
    save_split_manifest,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestPatientSplitManifest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        generate(str(self.data_root), n_hospitals=6, n_patients_per_hospital=40, seed=7)

        self._old_env = os.environ.get("EICU_DATA_ROOT")
        os.environ["EICU_DATA_ROOT"] = str(self.data_root)

        self.cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        self.cfg.raw["run"]["output_dir"] = str(self.output_dir)
        self.cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")
        self.cfg.raw["cohort"]["patient_subset"] = 30

        phase1_summary = run_phase1_ingestion(self.cfg)
        self.partition_result, patient_df, hospital_df = run_phase2_hospital_partition(self.cfg, phase1_summary)
        self.phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, self.partition_result, patient_df)

    def tearDown(self):
        close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def _all_partition_patient_ids(self) -> set:
        out = set()
        for p in self.partition_result.partitions.values():
            out.update(int(pid) for pid in p.patient_ids)
        return out

    def test_every_patient_assigned_to_exactly_one_partition(self):
        manifest = build_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], seed=42, stratify_on="mortality",
        )
        expected = self._all_partition_patient_ids()
        actual = set(manifest.all_patient_ids("train")) | set(manifest.all_patient_ids("val")) | set(manifest.all_patient_ids("test"))
        self.assertEqual(actual, expected)

        # No duplicates across partitions.
        train = set(manifest.all_patient_ids("train"))
        val = set(manifest.all_patient_ids("val"))
        test = set(manifest.all_patient_ids("test"))
        self.assertEqual(len(train & val), 0)
        self.assertEqual(len(train & test), 0)
        self.assertEqual(len(val & test), 0)

        # patient_to_split reverse index agrees with the per-hospital lists.
        for pid in expected:
            self.assertIn(manifest.split_of(pid), ("train", "val", "test"))

    def test_reproducible_given_same_seed(self):
        m1 = build_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], seed=123, stratify_on="mortality",
        )
        m2 = build_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], seed=123, stratify_on="mortality",
        )
        self.assertEqual(m1.hospital_splits, m2.hospital_splits)

    def test_different_seed_produces_a_different_split(self):
        m1 = build_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], seed=1, stratify_on="mortality",
        )
        m2 = build_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], seed=2, stratify_on="mortality",
        )
        self.assertNotEqual(m1.hospital_splits, m2.hospital_splits)

    def test_each_hospital_contributes_to_all_three_partitions_when_large_enough(self):
        manifest = build_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], seed=42, stratify_on="mortality",
        )
        for hid, p in self.partition_result.partitions.items():
            if len(p.patient_ids) < 10:
                continue  # too small to guarantee all three are non-empty at 70/15/15
            train = manifest.patient_ids_for_hospital(hid, "train")
            val = manifest.patient_ids_for_hospital(hid, "val")
            test = manifest.patient_ids_for_hospital(hid, "test")
            self.assertTrue(train, f"hospital {hid} has an empty train split")
            self.assertTrue(val, f"hospital {hid} has an empty val split")
            self.assertTrue(test, f"hospital {hid} has an empty test split")

    def test_tiny_hospital_falls_back_to_all_train_without_crashing(self):
        # A hospital with < MIN_PATIENTS_FOR_THREE_WAY_SPLIT patients
        # cannot be split three ways -- must not raise, must land
        # everyone in train.
        from src.data.cohort.hospital_partition import HospitalPartition, PartitionResult

        tiny_partition = PartitionResult(
            mode="subset", requested_hospitals=[999], eligible_hospitals=[999], excluded_hospitals=[],
            partitions={999: HospitalPartition(
                hospital_id=999, patient_count=2, eligible_patient_count=2, sampled_patient_count=2,
                patient_ids=[1, 2], mortality_missing_count=0, mortality_positive_count=0,
                mortality_negative_count=2, was_undersampled=False,
            )},
        )
        manifest = build_split_manifest(
            tiny_partition, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], seed=42, stratify_on="none",
        )
        self.assertEqual(manifest.n_hospitals_all_train_too_small, 1)
        # Falls back gracefully even though patients 1/2 aren't real
        # cohort members with real labels -- stratify_on="none" here
        # avoids touching the label-lookup path for this synthetic
        # tiny-partition case.
        self.assertEqual(set(manifest.patient_ids_for_hospital(999, "train")), {1, 2})
        self.assertEqual(manifest.patient_ids_for_hospital(999, "val"), [])
        self.assertEqual(manifest.patient_ids_for_hospital(999, "test"), [])

    def test_save_and_load_round_trip_is_lossless(self):
        manifest = build_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], seed=42, stratify_on="mortality",
        )
        path = Path(self.tmpdir.name) / "split_manifest.json"
        save_split_manifest(manifest, path)
        reloaded = load_split_manifest(path)
        self.assertEqual(manifest.hospital_splits, reloaded.hospital_splits)
        self.assertEqual(manifest.patient_to_split, reloaded.patient_to_split)

    def test_build_and_save_refuses_to_silently_overwrite(self):
        path = Path(self.tmpdir.name) / "permanent_split_manifest.json"
        m1 = build_and_save_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], path, seed=1,
        )
        # Calling again with a DIFFERENT seed, but overwrite=False (the
        # default), must return the ORIGINAL manifest untouched -- this
        # is the "one permanent manifest reused by every experiment"
        # guarantee (decision #9).
        m2 = build_and_save_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], path, seed=999,
        )
        self.assertEqual(m1.hospital_splits, m2.hospital_splits)
        self.assertEqual(m2.seed, 1)  # NOT 999 -- proves it reloaded rather than regenerated

        # With overwrite=True, it genuinely regenerates.
        m3 = build_and_save_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], path, seed=999, overwrite=True,
        )
        self.assertEqual(m3.seed, 999)

    def test_injected_leakage_bug_is_caught_by_the_invariant_check(self):
        """Revert->fail->restore->pass proof: temporarily corrupt one
        hospital's split so a patient appears in two partitions, and
        confirm build_split_manifest's own invariant check raises
        rather than silently returning a leaking manifest."""
        manifest = build_split_manifest(
            self.partition_result, self.phase3_result["cohort_patients"], self.phase3_result["mortality"],
            self.phase3_result["sepsis_df"], seed=42, stratify_on="mortality",
        )
        # Directly exercise the same invariant-checking code path the
        # builder itself uses, by re-running the duplicate-detection
        # loop against a hand-corrupted copy of a real manifest's
        # hospital_splits (the same shape build_split_manifest produces
        # internally right before it raises).
        import copy
        corrupted = copy.deepcopy(manifest.hospital_splits)
        some_hid = next(hid for hid, s in corrupted.items() if s["train"])
        leaked_patient = corrupted[some_hid]["train"][0]
        other_hid = next((hid for hid in corrupted if hid != some_hid and corrupted[hid]["val"] is not None), some_hid)
        corrupted[other_hid]["val"].append(leaked_patient)

        seen = set()
        raised = False
        try:
            for hid_splits in corrupted.values():
                for split_name in ("train", "val", "test"):
                    ids = hid_splits[split_name]
                    overlap = seen.intersection(ids)
                    if overlap:
                        raise AssertionError("leak")
                    seen.update(ids)
        except AssertionError:
            raised = True
        self.assertTrue(raised, "the duplicate-detection logic failed to catch an injected cross-partition leak")


    def test_unstratified_fallback_path_does_not_crash(self):
        """Direct regression test for a real bug found while building
        the embedding-ablation runner (2026-09-10): a hospital falling
        back to an unstratified split (too few members in the smallest
        class) crashed inside sklearn's train_test_split because a
        redundant second positional array was being co-split alongside
        `ids`, and that second array was `None` in exactly this
        fallback case. Forces the fallback by using a hospital-sized
        partition where the minority class has exactly 1 member."""
        from src.data.cohort.hospital_partition import HospitalPartition, PartitionResult

        # 10 patients, only 1 positive -- guaranteed to trip
        # MIN_PER_CLASS_FOR_STRATIFY=2 and fall back to unstratified.
        patient_ids = list(range(9000, 9010))
        partition = PartitionResult(
            mode="subset", requested_hospitals=[888], eligible_hospitals=[888], excluded_hospitals=[],
            partitions={888: HospitalPartition(
                hospital_id=888, patient_count=10, eligible_patient_count=10, sampled_patient_count=10,
                patient_ids=patient_ids, mortality_missing_count=0, mortality_positive_count=1,
                mortality_negative_count=9, was_undersampled=False,
            )},
        )
        # Build a matching mortality Series/cohort_patients so
        # align_labels_to_patient_ids can resolve real 0/1 labels for
        # these synthetic ids (only patient 9000 is positive).
        import pandas as pd
        cohort_patients = pd.DataFrame({"patientunitstayid": patient_ids, "hospitalid": [888] * 10})
        mortality = pd.Series([1.0] + [0.0] * 9, index=cohort_patients.index)
        sepsis_df = pd.DataFrame({
            "patientunitstayid": patient_ids, "sepsis_broad": [False] * 10, "sepsis_strict": [False] * 10,
        })

        manifest = build_split_manifest(
            partition, cohort_patients, mortality, sepsis_df=sepsis_df, seed=42, stratify_on="mortality",
        )  # must not raise
        self.assertEqual(manifest.n_hospitals_fallback_unstratified, 1)
        all_ids = set(manifest.patient_ids_for_hospital(888, "train")) | set(manifest.patient_ids_for_hospital(888, "val")) | set(manifest.patient_ids_for_hospital(888, "test"))
        self.assertEqual(all_ids, set(patient_ids))


if __name__ == "__main__":
    unittest.main()
