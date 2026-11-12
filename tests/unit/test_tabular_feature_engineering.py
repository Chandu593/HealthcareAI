import unittest

import numpy as np
import pandas as pd

from src.data.tabular.feature_engineering import (
    APACHEPREDVAR_NUMERIC_FEATURES,
    PATIENT_NUMERIC_FEATURES,
    accumulate_value_counts,
    assemble_tabular_features,
    attach_hospital_features,
    build_apachepredvar_features,
    build_diagnosis_features,
    build_patient_features,
    clean_eicu_sentinels,
    determine_categorical_vocabulary,
    determine_categorical_vocabulary_from_value_counts,
    determine_diagnosis_vocabulary_from_shards,
    determine_full_cohort_vocabularies,
    one_hot_with_vocabulary,
    parse_age,
)


class TestParseAge(unittest.TestCase):
    def test_plain_integer_string(self):
        result = parse_age(pd.Series(["70", "52", "18"]))
        self.assertEqual(list(result), [70.0, 52.0, 18.0])

    def test_hipaa_top_coded_band(self):
        result = parse_age(pd.Series(["> 89", ">89"]))
        self.assertEqual(list(result), [90.0, 90.0])

    def test_missing_values_become_nan(self):
        result = parse_age(pd.Series([None, np.nan, ""]))
        self.assertTrue(all(pd.isna(v) for v in result))

    def test_unparseable_string_becomes_nan_not_zero(self):
        result = parse_age(pd.Series(["unknown"]))
        self.assertTrue(pd.isna(result.iloc[0]))


class TestCleanEicuSentinels(unittest.TestCase):
    def test_negative_one_becomes_nan(self):
        result = clean_eicu_sentinels(pd.Series([-1, 5, -1.0, 10]))
        self.assertTrue(pd.isna(result.iloc[0]))
        self.assertEqual(result.iloc[1], 5)
        self.assertTrue(pd.isna(result.iloc[2]))
        self.assertEqual(result.iloc[3], 10)


class TestDetermineCategoricalVocabulary(unittest.TestCase):
    def test_returns_most_frequent(self):
        values = pd.Series(["A"] * 10 + ["B"] * 5 + ["C"] * 1)
        vocab = determine_categorical_vocabulary(values, top_k=2)
        self.assertEqual(vocab, ["A", "B"])

    def test_ignores_nan(self):
        values = pd.Series(["A", "A", np.nan, np.nan, np.nan])
        vocab = determine_categorical_vocabulary(values, top_k=5)
        self.assertEqual(vocab, ["A"])


class TestAccumulateValueCounts(unittest.TestCase):
    def test_first_call_with_none_running_returns_counts(self):
        result = accumulate_value_counts(None, pd.Series(["A", "A", "B"]))
        self.assertEqual(result["A"], 2)
        self.assertEqual(result["B"], 1)

    def test_folds_counts_across_multiple_calls(self):
        running = accumulate_value_counts(None, pd.Series(["A", "A", "B"]))
        running = accumulate_value_counts(running, pd.Series(["B", "C"]))
        running = accumulate_value_counts(running, pd.Series(["A"]))
        self.assertEqual(running["A"], 3)
        self.assertEqual(running["B"], 2)
        self.assertEqual(running["C"], 1)

    def test_never_sees_more_than_one_call_worth_of_raw_values(self):
        # Each call only ever receives ITS OWN Series -- this test just
        # documents/asserts the calling contract (one hospital's worth
        # of raw values per call), which is what determine_diagnosis_
        # vocabulary_from_shards() relies on to avoid pooling raw rows.
        seen_lengths = []
        running = None
        for batch in [pd.Series(["A"] * 100), pd.Series(["B"] * 3), pd.Series(["C"])]:
            seen_lengths.append(len(batch))
            running = accumulate_value_counts(running, batch)
        self.assertEqual(seen_lengths, [100, 3, 1])  # never concatenated before counting
        self.assertEqual(running["A"], 100)

    def test_nan_ignored_in_each_batch(self):
        running = accumulate_value_counts(None, pd.Series(["A", np.nan]))
        running = accumulate_value_counts(running, pd.Series([np.nan, np.nan]))
        self.assertEqual(dict(running), {"A": 1})


class TestDetermineCategoricalVocabularyFromValueCounts(unittest.TestCase):
    def test_ranks_by_aggregated_count(self):
        counts = pd.Series({"A": 3, "B": 10, "C": 1})
        vocab = determine_categorical_vocabulary_from_value_counts(counts, top_k=2)
        self.assertEqual(vocab, ["B", "A"])

    def test_none_counts_returns_empty_list(self):
        self.assertEqual(determine_categorical_vocabulary_from_value_counts(None, top_k=5), [])

    def test_empty_counts_returns_empty_list(self):
        self.assertEqual(determine_categorical_vocabulary_from_value_counts(pd.Series(dtype=float), top_k=5), [])

    def test_matches_determine_categorical_vocabulary_on_equivalent_data(self):
        # Aggregating in two batches then ranking should give the same
        # answer as ranking the pooled raw values directly -- proves
        # the streaming/aggregate path is equivalent to the original.
        raw = pd.Series(["A"] * 10 + ["B"] * 5 + ["C"] * 1)
        direct = determine_categorical_vocabulary(raw, top_k=2)

        running = accumulate_value_counts(None, pd.Series(["A"] * 10 + ["B"] * 2))
        running = accumulate_value_counts(running, pd.Series(["B"] * 3 + ["C"] * 1))
        streamed = determine_categorical_vocabulary_from_value_counts(running, top_k=2)

        self.assertEqual(direct, streamed)


class TestAttachHospitalFeatures(unittest.TestCase):
    def test_merges_region_and_teachingstatus_via_hospitalid(self):
        cohort_patients = pd.DataFrame({
            "patientunitstayid": [1, 2, 3],
            "hospitalid": [10, 20, 10],
        })
        hospital_df = pd.DataFrame({
            "hospitalid": [10, 20],
            "region": ["Midwest", "South"],
            "teachingstatus": ["t", "f"],
        })

        result = attach_hospital_features(cohort_patients, hospital_df)

        self.assertEqual(list(result["hospital_region"]), ["Midwest", "South", "Midwest"])
        self.assertEqual(list(result["hospital_teachingstatus"]), ["t", "f", "t"])

    def test_row_count_unchanged(self):
        cohort_patients = pd.DataFrame({"patientunitstayid": [1, 2, 3], "hospitalid": [10, 20, 10]})
        hospital_df = pd.DataFrame({"hospitalid": [10, 20], "region": ["Midwest", "South"]})
        result = attach_hospital_features(cohort_patients, hospital_df)
        self.assertEqual(len(result), len(cohort_patients))

    def test_duplicate_hospitalid_in_hospital_df_does_not_multiply_rows(self):
        # Defensive guard: hospital.csv should be one row per hospital,
        # but a left-merge against a duplicated key would otherwise
        # silently multiply patient rows -- exactly the kind of silent
        # data corruption this project treats as unacceptable.
        cohort_patients = pd.DataFrame({"patientunitstayid": [1, 2], "hospitalid": [10, 20]})
        hospital_df = pd.DataFrame({
            "hospitalid": [10, 10, 20],  # 10 appears twice
            "region": ["Midwest", "Midwest", "South"],
        })
        result = attach_hospital_features(cohort_patients, hospital_df)
        self.assertEqual(len(result), 2)

    def test_missing_hospital_row_becomes_nan_not_a_crash(self):
        cohort_patients = pd.DataFrame({"patientunitstayid": [1, 2], "hospitalid": [10, 999]})
        hospital_df = pd.DataFrame({"hospitalid": [10], "region": ["Midwest"], "teachingstatus": ["t"]})
        result = attach_hospital_features(cohort_patients, hospital_df)
        self.assertEqual(result["hospital_region"].iloc[0], "Midwest")
        self.assertTrue(pd.isna(result["hospital_region"].iloc[1]))

    def test_missing_optional_columns_in_hospital_df_becomes_nan_not_a_crash(self):
        # region/teachingstatus are OPTIONAL columns per hospital's
        # schema (schema_registry.py) -- some real exports might lack
        # one or both.
        cohort_patients = pd.DataFrame({"patientunitstayid": [1], "hospitalid": [10]})
        hospital_df = pd.DataFrame({"hospitalid": [10]})  # no region/teachingstatus at all
        result = attach_hospital_features(cohort_patients, hospital_df)
        self.assertIn("hospital_region", result.columns)
        self.assertIn("hospital_teachingstatus", result.columns)
        self.assertTrue(pd.isna(result["hospital_region"].iloc[0]))


class TestDetermineFullCohortVocabularies(unittest.TestCase):
    def test_uses_every_hospital_not_just_the_smallest_ids(self):
        # Regression scenario for the vocabulary-sampling-bias bug:
        # hospitals 1-10 (the smallest IDs, sorted -- exactly what the
        # old `[:10]` sample would have picked) all share ONE admitsource
        # value; hospitals 11-15 introduce three more. The old
        # sorted-first-10 sample would see only "solo" and collapse the
        # vocabulary to size 1.
        rows = []
        for hid in range(1, 11):
            rows.append({"patientunitstayid": hid, "admitsource": "solo"})
        for i, hid in enumerate(range(11, 16)):
            rows.append({"patientunitstayid": hid, "admitsource": ["team_a", "team_b", "team_c"][i % 3]})
        apv = pd.DataFrame(rows)
        cohort_patients = pd.DataFrame({"patientunitstayid": range(1, 16), "gender": "Female"})

        vocab = determine_full_cohort_vocabularies(cohort_patients, apv, top_k=10)

        self.assertIn("solo", vocab["admitsource"])
        self.assertGreater(len(vocab["admitsource"]), 1, "full-cohort vocab must not collapse like the old 10-hospital sample")
        self.assertIn("team_a", vocab["admitsource"])

    def test_uses_every_hospital_not_just_the_smallest_ids_for_hospital_sourced_columns(self):
        # Same regression shape as above, but for hospital_region --
        # the column that turned out to matter most in practice: see
        # attach_hospital_features()'s docstring. teachtype/region were
        # REMOVED from APACHEPREDVAR_CATEGORICAL_FEATURES entirely
        # (real apachePredVar.teachtype/region are genuinely constant
        # in the real eICU export -- not a sampling artifact at all),
        # replaced by hospital_region/hospital_teachingstatus sourced
        # from hospital.csv instead, which DOES have real diversity.
        # This test proves the full-cohort fix still applies correctly
        # to that new column, same as any other PATIENT_CATEGORICAL_FEATURES
        # entry.
        rows = []
        for hid in range(1, 11):
            rows.append({"patientunitstayid": hid, "hospital_region": "solo"})
        for i, hid in enumerate(range(11, 16)):
            rows.append({"patientunitstayid": hid, "hospital_region": ["team_a", "team_b", "team_c"][i % 3]})
        cohort_patients = pd.DataFrame(rows)
        apv = pd.DataFrame({"patientunitstayid": range(1, 16)})

        vocab = determine_full_cohort_vocabularies(cohort_patients, apv, top_k=10)

        self.assertIn("solo", vocab["hospital_region"])
        self.assertGreater(len(vocab["hospital_region"]), 1, "full-cohort vocab must not collapse like the old 10-hospital sample")
        self.assertIn("team_a", vocab["hospital_region"])

    def test_top_k_override_applied_per_column(self):
        cohort_patients = pd.DataFrame({
            "patientunitstayid": range(1, 21),
            "apacheadmissiondx": [f"dx_{i}" for i in range(20)],
        })
        apv = pd.DataFrame({"patientunitstayid": range(1, 21)})

        default_vocab = determine_full_cohort_vocabularies(cohort_patients, apv, top_k=5)
        self.assertEqual(len(default_vocab["apacheadmissiondx"]), 5)

        overridden_vocab = determine_full_cohort_vocabularies(
            cohort_patients, apv, top_k=5, top_k_overrides={"apacheadmissiondx": 15},
        )
        self.assertEqual(len(overridden_vocab["apacheadmissiondx"]), 15)


class TestDetermineDiagnosisVocabularyFromShards(unittest.TestCase):
    def test_aggregates_across_all_hospitals_not_just_a_sample(self):
        # Same regression shape as the teachtype/region scenario above,
        # but for the genuinely sharded `diagnosis` table: hospitals
        # 1-10 all share one top-level diagnosis category, 11-15 add
        # three more. A load_shard_fn that only ever gets called for
        # ALL 15 hospitals (not just the first 10) is what fixes this.
        shards = {}
        for hid in range(1, 11):
            shards[hid] = pd.DataFrame({
                "patientunitstayid": [hid], "diagnosisoffset": [0], "diagnosisstring": ["cardiovascular|chest pain"],
            })
        extra_cats = ["renal|AKI", "neurologic|stroke", "pulmonary|pneumonia"]
        for i, hid in enumerate(range(11, 16)):
            shards[hid] = pd.DataFrame({
                "patientunitstayid": [hid], "diagnosisoffset": [0], "diagnosisstring": [extra_cats[i % 3]],
            })

        def load_shard(hid):
            return shards.get(hid, pd.DataFrame(columns=["patientunitstayid", "diagnosisoffset", "diagnosisstring"]))

        vocab = determine_diagnosis_vocabulary_from_shards(load_shard, list(range(1, 16)), top_k=10)

        self.assertIn("cardiovascular", vocab)
        self.assertGreater(len(vocab), 1, "full-shard diagnosis vocab must not collapse like the old 10-hospital sample")
        self.assertIn("renal", vocab)

    def test_hospitals_with_no_shard_are_skipped_not_errored(self):
        def load_shard(hid):
            return pd.DataFrame(columns=["patientunitstayid", "diagnosisoffset", "diagnosisstring"])

        vocab = determine_diagnosis_vocabulary_from_shards(load_shard, [1, 2, 3], top_k=10)
        self.assertEqual(vocab, [])

    def test_empty_hospital_list_returns_empty_vocabulary(self):
        vocab = determine_diagnosis_vocabulary_from_shards(lambda hid: pd.DataFrame(), [], top_k=10)
        self.assertEqual(vocab, [])


class TestOneHotWithVocabulary(unittest.TestCase):
    def test_known_values_encoded_correctly(self):
        series = pd.Series(["Male", "Female", "Male"])
        result = one_hot_with_vocabulary(series, "gender", ["Male", "Female"])
        self.assertEqual(list(result["gender_male"]), [1.0, 0.0, 1.0])
        self.assertEqual(list(result["gender_female"]), [0.0, 1.0, 0.0])
        self.assertEqual(list(result["gender_other"]), [0.0, 0.0, 0.0])

    def test_unknown_value_goes_to_other_bucket(self):
        series = pd.Series(["Male", "Nonbinary"])
        result = one_hot_with_vocabulary(series, "gender", ["Male", "Female"])
        self.assertEqual(list(result["gender_other"]), [0.0, 1.0])

    def test_missing_value_not_counted_anywhere(self):
        series = pd.Series(["Male", np.nan])
        result = one_hot_with_vocabulary(series, "gender", ["Male", "Female"])
        row = result.iloc[1]
        self.assertEqual(row["gender_male"], 0.0)
        self.assertEqual(row["gender_female"], 0.0)
        self.assertEqual(row["gender_other"], 0.0)  # NaN is not "other" -- it's genuinely unknown


class TestBuildPatientFeatures(unittest.TestCase):
    def test_leakage_columns_never_appear(self):
        patient_df = pd.DataFrame([
            {"patientunitstayid": 1, "age": "70", "gender": "Male", "ethnicity": "Caucasian",
             "admissionheight": 170.0, "admissionweight": 80.0,
             "hospitaladmitsource": "Emergency Department", "unitadmitsource": "Emergency Department",
             "unitstaytype": "admit", "apacheadmissiondx": "Sepsis",
             "hospitaldischargestatus": "Expired", "unitdischargestatus": "Expired",
             "hospitaldischargelocation": "Death", "unitdischargelocation": "Death",
             "dischargeweight": 78.0, "hospitaldischargeoffset": 5000, "unitdischargeoffset": 3000},
        ])
        vocabularies = {
            "gender": ["Male", "Female"], "ethnicity": ["Caucasian"],
            "hospitaladmitsource": ["Emergency Department"], "unitadmitsource": ["Emergency Department"],
            "unitstaytype": ["admit"], "apacheadmissiondx": ["Sepsis"],
        }
        result = build_patient_features(patient_df, vocabularies)
        for leaky_term in ["dischargestatus", "dischargelocation", "dischargeweight", "dischargeoffset"]:
            matches = [c for c in result.columns if leaky_term in c.lower()]
            self.assertEqual(matches, [], f"leakage column pattern '{leaky_term}' found in {matches}")

    def test_age_parsed_within_patient_features(self):
        patient_df = pd.DataFrame([{"patientunitstayid": 1, "age": "> 89"}])
        result = build_patient_features(patient_df, {})
        self.assertEqual(result["age"].iloc[0], 90.0)


class TestBuildApachePredVarFeatures(unittest.TestCase):
    def test_diedinhospital_never_appears(self):
        apv_df = pd.DataFrame([
            {"patientunitstayid": 1, "diedinhospital": 1, "diabetes": 0, "aids": 0,
             "hepaticfailure": 0, "lymphoma": 0, "metastaticcancer": 0, "leukemia": 0,
             "immunosuppression": 0, "cirrhosis": 0, "electivesurgery": 1, "thrombolytics": 0,
             "bedcount": 20, "readmit": 0, "visitnumber": 1,
             "admitsource": 1, "teachtype": 0, "region": 3,
             "pao2": 90.0, "fio2": 21.0, "creatinine": 1.0},  # SOFA-consumed, must also be absent
        ])
        result = build_apachepredvar_features(apv_df, {})
        for leaky_col in ["diedinhospital", "pao2", "fio2", "creatinine", "verbal", "motor", "eyes", "ventday1"]:
            matches = [c for c in result.columns if leaky_col in c.lower()]
            self.assertEqual(matches, [], f"leaked column '{leaky_col}' found in {matches}")

    def test_sentinel_negative_one_cleaned(self):
        apv_df = pd.DataFrame([{"patientunitstayid": 1, "bedcount": -1, "readmit": 0}])
        result = build_apachepredvar_features(apv_df, {})
        self.assertTrue(pd.isna(result["bedcount"].iloc[0]))

    def test_comorbidity_flags_present(self):
        apv_df = pd.DataFrame([{"patientunitstayid": 1, "diabetes": 1, "aids": 0}])
        result = build_apachepredvar_features(apv_df, {})
        self.assertEqual(result["diabetes"].iloc[0], 1)


class TestBuildDiagnosisFeatures(unittest.TestCase):
    def test_top_level_category_extracted_and_windowed(self):
        diagnosis_df = pd.DataFrame([
            {"patientunitstayid": 1, "diagnosisoffset": 100, "diagnosisstring": "cardiovascular|chest pain / ASHD|CAD"},
            {"patientunitstayid": 1, "diagnosisoffset": 5000, "diagnosisstring": "infectious diseases|sepsis|severe sepsis"},  # OUTSIDE window
            {"patientunitstayid": 2, "diagnosisoffset": 50, "diagnosisstring": "pulmonary|COPD|COPD exacerbation"},
        ])
        result = build_diagnosis_features(
            diagnosis_df, patient_ids=[1, 2], vocabulary=["cardiovascular", "pulmonary", "infectious diseases"],
            lookback_minutes=1440, cutoff_minutes=1440,
        )
        self.assertEqual(result.loc[1, "diag_cardiovascular"], 1.0)
        self.assertEqual(result.loc[1, "diag_infectious_diseases"], 0.0)  # windowed out
        self.assertEqual(result.loc[2, "diag_pulmonary"], 1.0)

    def test_unknown_category_goes_to_other(self):
        diagnosis_df = pd.DataFrame([
            {"patientunitstayid": 1, "diagnosisoffset": 10, "diagnosisstring": "obscure_category|x|y"},
        ])
        result = build_diagnosis_features(
            diagnosis_df, patient_ids=[1], vocabulary=["cardiovascular"],
            lookback_minutes=1440, cutoff_minutes=1440,
        )
        self.assertEqual(result.loc[1, "diag_other"], 1.0)
        self.assertEqual(result.loc[1, "diag_cardiovascular"], 0.0)

    def test_patient_with_no_diagnosis_rows_gets_all_zeros(self):
        diagnosis_df = pd.DataFrame([
            {"patientunitstayid": 1, "diagnosisoffset": 10, "diagnosisstring": "cardiovascular|x|y"},
        ])
        result = build_diagnosis_features(
            diagnosis_df, patient_ids=[1, 2], vocabulary=["cardiovascular"],
            lookback_minutes=1440, cutoff_minutes=1440,
        )
        self.assertEqual(result.loc[2, "diag_cardiovascular"], 0.0)
        self.assertEqual(result.loc[2, "diag_other"], 0.0)

    def test_empty_diagnosis_df_does_not_crash(self):
        result = build_diagnosis_features(
            pd.DataFrame(columns=["patientunitstayid", "diagnosisoffset", "diagnosisstring"]),
            patient_ids=[1, 2], vocabulary=["cardiovascular"],
            lookback_minutes=1440, cutoff_minutes=1440,
        )
        self.assertEqual(len(result), 2)
        self.assertTrue((result == 0.0).all().all())


class TestAssembleTabularFeatures(unittest.TestCase):
    def test_columns_identical_regardless_of_missing_source_data(self):
        """Two calls with different (even empty) source data must
        produce the SAME column set, since hospitals' feature blocks
        get concatenated later and must align."""
        patient_df = pd.DataFrame([{"patientunitstayid": 1, "age": "70", "gender": "Male"}])
        apv_df = pd.DataFrame([{"patientunitstayid": 1, "diabetes": 1}])
        diagnosis_df = pd.DataFrame(columns=["patientunitstayid", "diagnosisoffset", "diagnosisstring"])
        vocabularies = {"gender": ["Male", "Female"]}

        result_a = assemble_tabular_features(
            patient_df, apv_df, diagnosis_df, patient_ids=[1],
            vocabularies=vocabularies, diagnosis_vocabulary=["cardiovascular"],
            diagnosis_lookback_minutes=1440, diagnosis_cutoff_minutes=1440,
        )

        # Second hospital has NO apachePredVar or diagnosis data at all for its patient
        empty_apv = pd.DataFrame(columns=["patientunitstayid", "diabetes"])
        result_b = assemble_tabular_features(
            patient_df, empty_apv, diagnosis_df, patient_ids=[1],
            vocabularies=vocabularies, diagnosis_vocabulary=["cardiovascular"],
            diagnosis_lookback_minutes=1440, diagnosis_cutoff_minutes=1440,
        )
        self.assertEqual(list(result_a.columns), list(result_b.columns))

    def test_row_order_matches_patient_ids(self):
        patient_df = pd.DataFrame([
            {"patientunitstayid": 2, "age": "50"},
            {"patientunitstayid": 1, "age": "70"},
        ])
        apv_df = pd.DataFrame(columns=["patientunitstayid"])
        diagnosis_df = pd.DataFrame(columns=["patientunitstayid", "diagnosisoffset", "diagnosisstring"])
        result = assemble_tabular_features(
            patient_df, apv_df, diagnosis_df, patient_ids=[1, 2],
            vocabularies={}, diagnosis_vocabulary=[],
            diagnosis_lookback_minutes=1440, diagnosis_cutoff_minutes=1440,
        )
        self.assertEqual(list(result.index), [1, 2])
        self.assertEqual(result.loc[1, "age"], 70.0)
        self.assertEqual(result.loc[2, "age"], 50.0)


if __name__ == "__main__":
    unittest.main()
