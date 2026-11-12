"""End-to-end regression test for the Phase 5 vocabulary-sampling-bias bug.

Real bug (see IMPLEMENTATION_TRACKER.md and docs/tabular_modality.md):
Phase 5's categorical vocabulary determination used to sample only the
10 hospitals with the SMALLEST hospital IDs
(`list(partition_result.partitions.keys())[:10]`, since
hospital_partition.py builds `partitions` via
`sorted(available_hospitals)`) -- not a representative sample. On the
real 199,108-patient / 208-hospital run this collapsed several
categorical vocabularies below their real diversity, most visibly
apachePredVar.teachtype/region down to 1 category each.

That specific field turned out to have a SECOND, independent cause on
top of the sampling bug: apachePredVar.teachtype/region are genuinely
CONSTANT across the entire real apachePredVar table in this eICU
export (confirmed directly against the raw CSV) -- no sampling
strategy could have surfaced diversity that isn't there. Fixed
2026-08-24 by sourcing hospital_region/hospital_teachingstatus from
hospital.csv instead (which does have real diversity), via
attach_hospital_features(). This file's tests below target that fixed,
hospital.csv-sourced path (`patient_hospital_region_*` feature
columns), not the old `apv_region_*` one.

Unit-level coverage of the fixed helper functions -- including a direct
reproduction of the exact collapse-to-1 symptom, and a literal
revert -> fail -> restore -> pass proof performed by hand -- lives in
tests/unit/test_tabular_feature_engineering.py
(TestAttachHospitalFeatures, TestDetermineFullCohortVocabularies,
TestDetermineDiagnosisVocabularyFromShards).

THIS file instead proves the fix holds through the actual
Phase 1 -> 2 -> 3 -> 5 pipeline path, against generator-produced (not
hand-constructed) mock data, so a future refactor of how Phase 5 wires
its inputs together would still be caught even if the isolated helper
functions above kept passing.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_mock_eicu import generate  # noqa: E402
from run_pipeline import (  # noqa: E402
    run_phase1_ingestion,
    run_phase2_hospital_partition,
    run_phase3_cohort_construction,
    run_phase5_tabular_modality,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402

N_HOSPITALS = 15
# Chosen empirically: with this seed, hospitals 1-10 (sorted by ID --
# exactly what the old buggy sample would have picked) never draw the
# "Midwest" region, which only appears among hospitals 11-15.
# test_premise_* below verifies this property directly against the
# freshly-generated data rather than just assuming the seed still
# behaves this way (protects against generate_mock_eicu.py drifting
# out from under this test silently).
SEED = 3


class TestPhase5VocabularyBiasRegression(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        generate(str(self.data_root), n_hospitals=N_HOSPITALS, n_patients_per_hospital=5, seed=SEED)

        self._old_env = os.environ.get("EICU_DATA_ROOT")
        os.environ["EICU_DATA_ROOT"] = str(self.data_root)

        self.cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        self.cfg.raw["run"]["output_dir"] = str(self.output_dir)
        self.cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")
        self.cfg.raw["run"]["cache_dir"] = str(self.output_dir / "cache")
        # debug.yaml's federated.hospitals defaults to [], which
        # (per hospital_partition.partition_by_hospital) means "every
        # discovered hospital", regardless of federated.mode -- no
        # override needed to get all N_HOSPITALS included.

        self._skew_diagnosis_by_hospital()

        self.hospital_df = pd.read_csv(self.data_root / "hospital.csv")
        self.diagnosis_df = pd.read_csv(self.data_root / "diagnosis.csv")

    def _skew_diagnosis_by_hospital(self):
        """generate()'s diagnosis assignment is a uniform per-patient
        `rng.choice(DIAGNOSES)` draw, NOT hospital-correlated (unlike
        hospital.csv's teachingstatus/region, which run_pipeline's real
        bug actually collapsed) -- with only 4 DIAGNOSES categories and
        ~10 draws per hospital, the odds of the first 10 hospitals
        missing a whole category by chance are negligible, so no seed
        would reliably exercise determine_diagnosis_vocabulary_from_shards()'s
        full-cohort-coverage fix. Post-processing the generated
        diagnosis.csv here (rather than changing generate_mock_eicu.py
        itself) gets a deterministic version of the same regression
        shape used for region/teachtype above, without touching
        diagnosis-string generation shared by sepsis-labeling tests
        elsewhere."""
        diag_path = self.data_root / "diagnosis.csv"
        patient_df = pd.read_csv(self.data_root / "patient.csv")
        hosp_of_patient = dict(zip(patient_df["patientunitstayid"], patient_df["hospitalid"]))

        extra_categories = ["renal|acute renal failure|acute renal failure due to sepsis",
                             "pulmonary|disorders of the airways|COPD",
                             "infectious diseases|sepsis|sepsis due to urinary tract infection"]

        diag_df = pd.read_csv(diag_path)

        def assign(row):
            hid = hosp_of_patient.get(row["patientunitstayid"])
            if hid is not None and hid <= 10:
                return "cardiovascular|chest pain / ASHD|coronary artery disease"
            return extra_categories[int(hid) % len(extra_categories)] if hid is not None else row["diagnosisstring"]

        diag_df["diagnosisstring"] = diag_df.apply(assign, axis=1)
        diag_df.to_csv(diag_path, index=False)

    def tearDown(self):
        close_logging()  # must close file handlers before deleting the tempdir they write into (Windows) -- see logging_setup.close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def test_premise_first_10_hospitals_lack_a_region_present_later(self):
        """Sanity check on the FIXTURE, not the fix: confirms this seed
        actually reproduces the bug's shape. If this starts failing
        (e.g. after touching generate_mock_eicu.py's RNG usage), the
        two tests below would stop meaningfully exercising the fix --
        better to find out loudly here than have them pass for the
        wrong reason."""
        first_10_regions = set(self.hospital_df[self.hospital_df["hospitalid"] <= 10]["region"])
        all_regions = set(self.hospital_df["region"])
        self.assertLess(
            len(first_10_regions), len(all_regions),
            f"fixture (seed={SEED}) no longer reproduces the bias shape "
            f"(first_10={first_10_regions}, all={all_regions}) -- pick a new SEED",
        )

    def test_full_pipeline_region_vocabulary_not_missing_a_real_category(self):
        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        self.assertEqual(len(partition_result.partitions), N_HOSPITALS)

        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)
        phase5_result = run_phase5_tabular_modality(self.cfg, phase1_summary, phase3_result, partition_result)

        data = np.load(Path(phase5_result["features_path"]), allow_pickle=True)
        feature_names = [str(f) for f in data["feature_names"]]
        region_columns = [f for f in feature_names if f.startswith("patient_hospital_region_")]

        # 4 real regions -> 4 named columns once the "_other" bucket
        # column is excluded by the startswith filter matching named
        # values too, so just require strictly more than what the old
        # first-10-sorted-hospitals sample could see (verified < 4 by
        # test_premise_* above). FIXED 2026-08-24: region is sourced
        # from hospital.csv now, not apachePredVar (real apachePredVar.
        # region turned out to be genuinely constant in the real eICU
        # export -- see attach_hospital_features()'s docstring), hence
        # the "patient_hospital_region_" prefix rather than "apv_region_".
        self.assertGreaterEqual(
            len(region_columns), 4,
            f"region vocabulary looks collapsed -- only got columns: {region_columns}",
        )

    def test_premise_first_10_hospitals_lack_a_diagnosis_category_present_later(self):
        """Sanity check on the FIXTURE (see _skew_diagnosis_by_hospital
        in setUp), analogous to the region premise check above."""
        patient_df = pd.read_csv(self.data_root / "patient.csv")
        diag = self.diagnosis_df.merge(patient_df[["patientunitstayid", "hospitalid"]], on="patientunitstayid")
        diag["top"] = diag["diagnosisstring"].str.split("|").str[0]
        first_10_categories = set(diag[diag["hospitalid"] <= 10]["top"])
        all_categories = set(diag["top"])
        self.assertLess(
            len(first_10_categories), len(all_categories),
            f"diagnosis fixture no longer reproduces the bias shape "
            f"(first_10={first_10_categories}, all={all_categories})",
        )

    def test_full_pipeline_diagnosis_vocabulary_covers_all_hospitals_categories(self):
        real_top_level_categories = set(self.diagnosis_df["diagnosisstring"].str.split("|").str[0])

        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)
        phase5_result = run_phase5_tabular_modality(self.cfg, phase1_summary, phase3_result, partition_result)

        data = np.load(Path(phase5_result["features_path"]), allow_pickle=True)
        feature_names = [str(f) for f in data["feature_names"]]
        diagnosis_columns = [f for f in feature_names if f.startswith("diag_")]

        # top_k_categorical_overrides.diagnosis defaults to 20 (see
        # configs/base.yaml) -- comfortably above the mock generator's
        # small, fixed DIAGNOSES list, so every real category should
        # survive into the vocabulary (plus one "_other" bucket).
        self.assertGreaterEqual(
            len(diagnosis_columns), len(real_top_level_categories),
            f"expected >= {len(real_top_level_categories)} diagnosis columns "
            f"for real categories {real_top_level_categories}, got {diagnosis_columns}",
        )


if __name__ == "__main__":
    unittest.main()
