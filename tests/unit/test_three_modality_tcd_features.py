"""TCD feature extraction (src/three_modality/tcd_features.py): exact feature values, equivalence with an independent loop-based reference, explicit missingness,
the 24-hour temporal cutoff, train-only fitting, variable selection on training patients only, and cache behaviour."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _tcd_reference import ref_features          # noqa: E402
from _tcd_synth import make_dataset, make_manifest  # noqa: E402

from src.data.splitting.patient_split import SplitManifest  # noqa: E402
from src.three_modality import artifacts  # noqa: E402
from src.three_modality import tcd_features as tf  # noqa: E402
from src.three_modality.config import THREE_MODALITY_DEFAULTS, get_tcd_cfg, validate_tcd_cfg  # noqa: E402


def tcfg(**over):
    c = copy.deepcopy(THREE_MODALITY_DEFAULTS["tcd"])
    for k, v in over.items():
        a = c
        parts = k.split(".")
        for p in parts[:-1]:
            a = a[p]
        a[parts[-1]] = v
    c["variables"]["chunksize"] = 4000
    return validate_tcd_cfg(c, 24)


def series_of(**kw):
    """{key: (times_min, values)} for ONE patient (pidx 0) -> series dict."""
    return {k: (np.zeros(len(t), np.int64), np.array(t, float), np.array(v, float)) for k, (t, v) in kw.items()}


def matrix(series, n=1, we=None, cfg=None, kinds=None):
    cfg = cfg or tcfg()
    we = np.full(n, 1440.0) if we is None else np.asarray(we, float)
    X, names, entries, un = tf.compute_tcd_feature_matrix(series, kinds or {k: "lab" for k in series}, n, we, cfg)
    return pd.DataFrame(X, columns=names), entries, un


class TestHandCheckedValues(unittest.TestCase):
    def test_every_documented_value_for_a_small_series(self):
        t = [60, 180, 600, 900, 1380]; v = [1.0, 3.0, 2.5, 1.5, 1.0]
        df, _, _ = matrix(series_of(lactate=(t, v)))
        r = df.iloc[0]
        exp = {"lactate__mean": 1.8, "lactate__median": 1.5, "lactate__min": 1.0, "lactate__max": 3.0, "lactate__first": 1.0, "lactate__last": 1.0,
               "lactate__std": np.std(v, ddof=1), "lactate__early_mean": (1 + 3 + 2.5) / 3, "lactate__late_mean": 1.25,
               "lactate__late_minus_early": 1.25 - (1 + 3 + 2.5) / 3, "lactate__slope_per_h": np.polyfit(np.array(t) / 60, v, 1)[0],
               "lactate__rate_first_last_per_h": 0.0, "lactate__median_abs_change": 0.75, "lactate__max_abs_change_6h": 2.0,
               "lactate__n_meas": 5, "lactate__any_meas": 1, "lactate__mean_interval_h": 5.5, "lactate__max_gap_h": 8.0, "lactate__hours_since_last": 1.0,
               "lactate__first_meas_h": 1.0, "lactate__hourly_coverage": 5 / 24,
               # reference range for lactate is [None, 2.0]: abnormal = [F, T, T, F, F]
               "lactate__n_abnormal": 2, "lactate__frac_abnormal": 0.4, "lactate__n_abn_episodes": 1, "lactate__first_abnormal": 0, "lactate__last_abnormal": 0,
               "lactate__deteriorated": 0.0, "lactate__abn_time_frac": 12 / 22, "lactate__longest_abn_h": 12.0}
        for k, e in exp.items():
            self.assertAlmostEqual(float(r[k]), float(e), places=4, msg=k)
        self.assertTrue(np.isnan(r["lactate__recovered"]))                  # first measurement normal -> 'recovered' is not applicable (NaN, not 0)

    def test_recovery_and_persistence_semantics(self):
        s = series_of(lactate=([60, 600], [4.0, 1.0]), creatinine=([60, 600], [2.5, 2.4]), bun=([60, 600], [10, 40]))
        df, _, _ = matrix(s)
        r = df.iloc[0]
        self.assertEqual((r["lactate__recovered"], r["creatinine__recovered"]), (1.0, 0.0))       # abnormal->normal vs abnormal->still abnormal
        self.assertEqual((r["bun__deteriorated"], r["bun__recovered"] if not np.isnan(r["bun__recovered"]) else -1), (1.0, -1))


class TestAgainstIndependentReference(unittest.TestCase):
    def test_vectorised_equals_loop_based_reference(self):
        cfg = tcfg(); total = bad = 0
        for seed in range(3):
            rs = np.random.default_rng(seed); n_pat = 250
            P, T, V = [], [], []
            for p in range(n_pat):
                k = rs.choice([0, 1, 2, 3, 4, 5, 9, 20, 60], p=[.12, .12, .1, .12, .1, .14, .12, .1, .08])
                if k == 0:
                    continue
                ts = np.sort(rs.choice(np.arange(0, 1441, 5), size=min(k, 289), replace=False)).astype(float)
                P += [p] * len(ts); T += list(ts); V += list(np.round(rs.normal(2.0, 1.5, len(ts)), 2))
            P, T, V = np.array(P), np.array(T), np.array(V)
            for rng in (None, [None, 2.0], [1.0, 3.0], [0.5, None]):
                F = tf.variable_features(P, T, V, n_pat, cfg, rng)
                for p in range(n_pat):
                    m = np.flatnonzero(P == p)
                    for k, val in ref_features(T[m].tolist(), V[m].tolist(), cfg, rng).items():
                        got = float(F[k][p]); total += 1
                        if not (np.isnan(val) and np.isnan(got)) and not np.isclose(got, val, rtol=1e-4, atol=1e-4):
                            bad += 1
        self.assertGreater(total, 90000)      # guard: the comparison really covered a large number of feature values
        self.assertEqual(bad, 0)

    def test_inclusive_boundaries_are_exact_in_whole_minutes(self):
        cfg = tcfg()
        def mac(t):                          # max_abs_change_6h for values 0,1 at the two times
            return tf.variable_features(np.zeros(2, np.int64), np.array(t, float), np.array([0.0, 1.0]), 1, cfg, None)["max_abs_change_6h"][0]
        self.assertEqual(mac([0, 30]), 1.0); self.assertTrue(np.isnan(mac([0, 29])))        # 0.5 h inclusive
        self.assertEqual(mac([0, 360]), 1.0); self.assertTrue(np.isnan(mac([0, 361])))      # 6 h inclusive
        def gate(t, name):
            return tf.variable_features(np.zeros(len(t), np.int64), np.array(t, float), np.arange(len(t), dtype=float), 1, cfg, None)[name][0]
        self.assertFalse(np.isnan(gate([0, 120], "rate_first_last_per_h"))); self.assertTrue(np.isnan(gate([0, 119], "rate_first_last_per_h")))   # span >= 2 h
        self.assertFalse(np.isnan(gate([0, 120, 240], "slope_per_h"))); self.assertTrue(np.isnan(gate([0, 120, 239], "slope_per_h")))              # n>=3 and span >= 4 h


class TestMissingness(unittest.TestCase):
    def test_unmeasured_patients_get_counts_zero_and_everything_else_nan(self):
        s = {"creatinine": (np.array([1], np.int64), np.array([100.0]), np.array([1.1]))}     # patient 1 measured, patients 0 and 2 never
        X, names, _, _ = tf.compute_tcd_feature_matrix(s, {"creatinine": "lab"}, 3, np.full(3, 1440.0), tcfg())
        df = pd.DataFrame(X, columns=names)
        for p in (0, 2):
            self.assertEqual((df.loc[p, "creatinine__n_meas"], df.loc[p, "creatinine__any_meas"], df.loc[p, "creatinine__hourly_coverage"]), (0.0, 0.0, 0.0))
            for c in ("mean", "median", "last", "std", "slope_per_h", "frac_abnormal", "hours_since_last", "first_meas_h"):
                self.assertTrue(np.isnan(df.loc[p, f"creatinine__{c}"]), c)       # never fabricated
        self.assertEqual(df.loc[1, "creatinine__mean"], np.float32(1.1))

    def test_statistics_below_their_minimum_counts_are_nan_not_zero(self):
        df, _, _ = matrix(series_of(creatinine=([100, 200], [1.0, 2.0])))
        r = df.iloc[0]
        for c in ("iqr", "p10", "p90", "mad", "slope_per_h", "median_abs_change"):
            self.assertTrue(np.isnan(r[f"creatinine__{c}"]), c)                    # need n>=4 / 3 / 3
        self.assertFalse(np.isnan(r["creatinine__std"]))                           # n>=2 is enough

    def test_unavailable_derived_and_interactions_are_declared_never_imputed(self):
        df, entries, un = matrix(series_of(creatinine=([100], [1.0]), bun=([100], [20.0])))
        names = {u["name"]: u["reason"] for u in un}
        for n in ("shock_index", "pulse_pressure", "map", "pf_ratio", "anion_gap_calc", "ix__shock_joint", "ix__hr_map_joint", "ix__resp_vitals_joint", "ix__coag_joint"):
            self.assertIn(n, names)
        self.assertIn("hr", names["shock_index"])
        self.assertNotIn("shock_index__mean", df.columns)                           # no column at all for an unavailable indicator
        self.assertIn("bun_creat_ratio__mean", df.columns)                          # available one is present


class TestDerivedAndInteractions(unittest.TestCase):
    def test_identical_offset_alignment_and_fio2_normalisation(self):
        s = series_of(pao2=([100, 200, 300], [80.0, 60.0, 90.0]), fio2=([100, 250, 300], [40.0, 50.0, 0.5]))   # aligned at 100 (percent) and 300 (fraction); 200 vs 250 not aligned
        df, _, un = matrix(s)
        r = df.iloc[0]
        self.assertEqual(r["pf_ratio__n_meas"], 2)
        self.assertAlmostEqual(float(r["pf_ratio__min"]), 90 / 0.5, places=3); self.assertAlmostEqual(float(r["pf_ratio__max"]), 80 / 0.4, places=3)   # 180 and 200
        self.assertAlmostEqual(float(r["pf_ratio__frac_abnormal"]), 1.0, places=3)                  # range [300, None]: both 200 and 180 are below it

    def test_joint_abnormal_counts(self):
        s = series_of(creatinine=([100, 200, 300], [2.0, 1.0, 3.0]), bun=([100, 200, 300, 400], [30.0, 40.0, 10.0, 50.0]))   # aligned at 100/200/300
        df, _, _ = matrix(s)
        r = df.iloc[0]
        self.assertEqual((r["ix__renal_joint__n_aligned"], r["ix__renal_joint__n_joint"]), (3, 1))   # only t=100 has creatinine>1.3 AND bun>25
        self.assertAlmostEqual(float(r["ix__renal_joint__frac_joint"]), 1 / 3, places=5)

    def test_patient_level_draw_and_joint_abnormality_features(self):
        s = series_of(creatinine=([100, 400], [2.0, 1.0]), bun=([100, 400], [40.0, 10.0]), sodium=([400], [150.0]))
        df, _, _ = matrix(s)
        r = df.iloc[0]
        self.assertEqual((r["pt__n_draws"], r["pt__n_meas_total"], r["pt__n_vars_measured"]), (2, 5, 3))
        self.assertEqual(r["pt__max_abn_per_draw"], 2)                              # draw at 100: creatinine & bun abnormal
        self.assertEqual(r["pt__last_draw_n_abn"], 1)                               # draw at 400: only sodium (150>145)
        self.assertAlmostEqual(float(r["pt__mean_abn_frac_per_draw"]), (1.0 + 1 / 3) / 2, places=5)
        self.assertAlmostEqual(float(r["pt__mean_draw_interval_h"]), 5.0, places=5)

    def test_optional_vitals_path_computes_shock_index_pulse_pressure_and_map_when_the_table_exists(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pd.DataFrame({"patientunitstayid": [1, 1, 1, 1], "observationoffset": [10, 20, 70, 100], "heartrate": [100, 110, 90, 95], "respiration": [20, 22, 18, 19],
                          "sao2": [97, 96, 98, 99], "temperature": [37, 37, 37, 37], "systemicsystolic": [100, np.nan, 120, 110], "systemicdiastolic": [60, np.nan, 80, 70],
                          "systemicmean": [73, np.nan, 93, np.nan], "noninvasivesystolic": [np.nan, 90, np.nan, np.nan], "noninvasivediastolic": [np.nan, 50, np.nan, np.nan],
                          "noninvasivemean": [np.nan, 63, np.nan, 83]}).to_csv(root / "vitalPeriodic.csv", index=False)
            v, info = tf.read_vitals({"data_root": d}, np.array([1]), np.array([1440.0]), tcfg())
            self.assertTrue(info["available"]); self.assertIn("hr", v)
            s = {k: (a, b, c) for k, (a, b, c) in v.items()}
            X, names, _, un = tf.compute_tcd_feature_matrix(s, {k: "vital" for k in s}, 1, np.array([1440.0]), tcfg())
            df = pd.DataFrame(X, columns=names).iloc[0]
            # hour bin 1 = offsets (0,60]: hr mean 105, sbp = (100 [systemic], 90 [non-invasive fallback]) -> mean 95 ; bin 2: hr 92.5, sbp 115
            self.assertAlmostEqual(float(df["shock_index__max"]), max(105 / 95, 92.5 / 115), places=4)
            self.assertIn("pulse_pressure__mean", df.index); self.assertNotIn("shock_index", [u["name"] for u in un])
            self.assertEqual(df["map__n_meas"], 2)                                   # MAP was MEASURED (systemic/non-invasive) -> not re-derived from sbp/dbp

    def test_vitals_table_absent_is_reported_not_faked(self):
        with tempfile.TemporaryDirectory() as d:
            v, info = tf.read_vitals({"data_root": d}, np.array([1]), np.array([1440.0]), tcfg())
        self.assertEqual((v, info["available"]), ({}, False)); self.assertIn("not present", info["reason"])


class TestTemporalCutoff(unittest.TestCase):
    def base(self, extra_t=(), extra_v=()):
        t = [100, 700, 1440] + list(extra_t); v = [1.0, 2.0, 3.0] + list(extra_v)
        return series_of(creatinine=(t, v))

    def test_rows_after_the_window_or_before_icu_cannot_change_any_feature(self):
        clean, _, _ = matrix(self.base())
        dirty, _, _ = matrix(self.base([1441, 2000, 5000, -30, -500], [99.0, -7.0, 1e6, 55.0, 1e3]))
        pd.testing.assert_frame_equal(clean, dirty)                                  # bit-identical

    def test_minute_1440_is_inside_and_1441_is_outside(self):
        inside, _, _ = matrix(series_of(creatinine=([100, 1440], [1.0, 9.0])))
        outside, _, _ = matrix(series_of(creatinine=([100, 1441], [1.0, 9.0])))
        self.assertEqual(inside["creatinine__last"][0], 9.0); self.assertEqual(outside["creatinine__last"][0], 1.0)
        self.assertEqual(outside["creatinine__n_meas"][0], 1)

    def test_a_patient_discharged_early_only_contributes_rows_up_to_exit(self):
        s = series_of(creatinine=([100, 500, 700], [1.0, 2.0, 50.0]))
        df, _, _ = matrix(s, we=[600.0])                                              # ICU exit at minute 600 -> the 700 row is after the window
        self.assertEqual((df["creatinine__n_meas"][0], df["creatinine__last"][0]), (2, 2.0))

    def test_reader_enforces_the_window_and_reports_what_it_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({"patientunitstayid": [1] * 6, "labresultoffset": [-60, 0, 1440, 1441, 3000, 90], "labname": ["creatinine"] * 6,
                          "labresult": [1, 2, 3, 4, 5, 6]}).to_csv(Path(d) / "lab.csv", index=False)
            cfg = tcfg()
            ids = np.array([1]); we = np.array([1440.0])
            cov = {"key_of_name": {"creatinine": "creatinine"}}
            s, st = tf.read_lab_series({"data_root": d}, ids, we, ["creatinine"], cov, cfg)
            p, t, v = s["creatinine"]
            self.assertEqual(sorted(t.tolist()), [0.0, 90.0, 1440.0])
            self.assertEqual((st["rows_dropped_before_icu"], st["rows_dropped_after_window"], st["max_offset_used_minutes"]), (1, 2, 1440.0))

    def test_cutoff_audit_poisons_late_rows_and_proves_features_are_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pats, _ = make_dataset(root / "data", n_pat=120, seed=3)
            man = make_manifest(pats)
            cfg = tcfg(**{"audit.n_patients": 40})
            res = tf.build_tcd_features({"data_root": str(root / "data")}, pats, man, cfg, root / "out")
            a = json.loads((Path(res["run_dir"]) / "cutoff_audit.json").read_text())
            self.assertTrue(a["passed"]); self.assertGreater(a["rows_after_window_present_and_corrupted"], 0)      # the audit had teeth
            self.assertTrue(a["features_identical_with_and_without_corruption"]); self.assertTrue(a["reader_and_clipping_agree"]); self.assertTrue(a["features_match_main_run"])
            self.assertLessEqual(a["reader_max_offset_used_minutes"], 1440)

    def test_audit_detects_a_clipping_regression(self):
        """If the in-function clipping were removed, the poison audit must FAIL (otherwise it proves nothing)."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pats, _ = make_dataset(root / "data", n_pat=120, seed=3)
            man = make_manifest(pats)
            cfg = tcfg(**{"audit.n_patients": 40})
            orig = tf.clip_series
            tf.clip_series = lambda series, we, w: series                          # simulate the regression
            try:
                with self.assertRaises(AssertionError):
                    tf.build_tcd_features({"data_root": str(root / "data")}, pats, man, cfg, root / "out")
            finally:
                tf.clip_series = orig

    def test_window_must_equal_the_incident_cutoff(self):
        with self.assertRaises(ValueError):
            get_tcd_cfg({"three_modality": {"incident_sepsis": {"cutoff_hours": 48}}})
        self.assertEqual(get_tcd_cfg({})["window_minutes"], 1440)


class TestSelectionAndPreprocessingAreFittedOnTrainOnly(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.pats, self.lab = make_dataset(self.root / "data", n_pat=260, seed=0)
        self.man = make_manifest(self.pats)
        self.cfg = tcfg(**{"audit.n_patients": 0})
        self.src = {"data_root": str(self.root / "data")}

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_lab_measured_only_in_validation_and_test_patients_is_not_selected(self):
        sp = {int(k): v for k, v in self.man.patient_to_split.items()}
        non_train = [i for i, s in sp.items() if s != "train"][:30]
        extra = pd.DataFrame({"patientunitstayid": non_train, "labresultoffset": 100, "labname": "val_only_lab", "labresult": 1.0})
        pd.concat([self.lab, extra]).to_csv(self.root / "data" / "lab.csv", index=False)
        ids = np.array(sorted(sp)); split = np.array([sp[i] for i in ids]); we = np.full(len(ids), 1440.0)
        cov = tf.lab_coverage(self.src, ids, split == "train", we, self.cfg)
        self.assertNotIn("val_only_lab", cov["counts"])                                 # never even counted: coverage is a TRAIN statistic
        sel, rows = tf.select_variables(cov, self.cfg)
        self.assertNotIn("val_only_lab", sel)
        self.assertIn("creatinine", sel); self.assertEqual(cov["n_train"], int((split == "train").sum()))

    def test_preprocessing_state_is_identical_when_validation_and_test_rows_are_corrupted(self):
        ids = np.arange(300); split = np.array(["train"] * 200 + ["val"] * 50 + ["test"] * 50)
        rs = np.random.default_rng(0); raw = rs.normal(size=(300, 12)).astype(np.float32)
        raw[rs.random((300, 12)) < 0.2] = np.nan; raw[:, 3] = np.nan                   # a fully missing feature -> dropped
        names = [f"f{i}" for i in range(12)]
        pp = self.cfg["preprocessing"]
        a = tf.fit_preprocessing(raw, names, split == "train", pp)
        bad = raw.copy(); bad[split != "train"] = 1e9; bad[(split != "train")][:, :4] = np.nan
        b = tf.fit_preprocessing(bad, names, split == "train", pp)
        for k in ("kept_idx", "lo", "hi", "median", "mean", "std"):
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        self.assertEqual(a["kept_names"], b["kept_names"]); self.assertEqual(a["indicator_groups"], b["indicator_groups"])

    def test_transform_standardises_train_clips_outliers_imputes_and_adds_deduplicated_indicators(self):
        rs = np.random.default_rng(1); n = 400
        split = np.array(["train"] * 300 + ["val"] * 50 + ["test"] * 50)
        raw = np.column_stack([rs.normal(5, 2, n), rs.normal(0, 1, n), np.full(n, 3.0), rs.normal(0, 1, n), rs.normal(0, 1, n)]).astype(np.float32)
        miss = rs.random(n) < 0.3; raw[miss, 1] = np.nan; raw[miss, 3] = np.nan            # features 1 and 3 share ONE missingness pattern
        raw[5, 0] = 1e6                                                                  # an extreme outlier in the training rows
        names = ["a", "b", "const", "c", "d"]
        st = tf.fit_preprocessing(raw, names, split == "train", self.cfg["preprocessing"])
        self.assertEqual(st["kept_names"], ["a", "b", "c", "d"])                         # 'const' dropped
        self.assertEqual([d["code"] for d in st["dropped"]], ["constant"])
        self.assertEqual(len(st["indicator_groups"]), 1)                                 # features b and c share one indicator column
        X = tf.transform_features(raw, st)
        self.assertEqual(X.shape, (n, 4 + 1)); self.assertTrue(np.isfinite(X).all()); self.assertEqual(X.dtype, np.float32)
        tr = X[:300, :4]
        np.testing.assert_allclose(tr[~miss[:300]].mean(axis=0)[[0, 2, 3]], 0, atol=0.15)  # standardised on the train rows
        self.assertLess(np.abs(X[5, 0]), 50)                                              # the 1e6 outlier was winsorised, not propagated
        self.assertEqual(X[miss, 4].min(), 1.0); self.assertEqual(X[~miss, 4].max(), 0.0)  # indicator = missing flag
        names_t = tf.transformed_names(st); self.assertEqual(names_t[-1], "b__missing")

    def test_features_missing_in_nearly_all_training_patients_are_dropped_with_a_reason(self):
        raw = np.full((100, 2), np.nan, np.float32); raw[:, 0] = np.arange(100); raw[:2, 1] = [1.0, 2.0]
        st = tf.fit_preprocessing(raw, ["ok", "rare"], np.ones(100, bool), self.cfg["preprocessing"])
        self.assertEqual(st["kept_names"], ["ok"]); self.assertEqual(st["dropped"][0]["code"], "too_missing")


class TestPipelineLevelLeakageGuards(unittest.TestCase):
    """Through build_tcd_features itself (not helper calls): what the validation/test patients and post-exit rows are allowed to influence."""

    def _build(self, root, pats, man, sub):
        cfg = tcfg(**{"audit.n_patients": 0})
        return tf.build_tcd_features({"data_root": str(root / sub)}, pats, man, cfg, root / ("out_" + sub))

    def test_validation_and_test_patients_cannot_influence_selection_or_preprocessing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pats, lab = make_dataset(root / "A", n_pat=300, seed=4)
            man = make_manifest(pats)
            sp = {int(k): v for k, v in man.patient_to_split.items()}
            nontrain = pats["patientunitstayid"].map(lambda i: sp[int(i)] != "train")
            # B = same data, but val/test patients' lab values are wrecked AND they alone carry an extra, very common lab
            labB = lab.copy()
            m = labB["patientunitstayid"].map(lambda i: sp[int(i)] != "train")
            labB.loc[m, "labresult"] = pd.to_numeric(labB.loc[m, "labresult"], errors="coerce") * 1000 + 777
            extra = pd.DataFrame({"patientunitstayid": pats.loc[nontrain, "patientunitstayid"], "labresultoffset": 50, "labname": "val_test_only_lab", "labresult": 3.0})
            (root / "B").mkdir(); pats.to_csv(root / "B" / "patient.csv", index=False)
            pd.concat([labB, extra]).to_csv(root / "B" / "lab.csv", index=False)
            ra, rb = self._build(root, pats, man, "A"), self._build(root, pats, man, "B")
            fa, fb = tf.load_tcd_features(ra["run_dir"]), tf.load_tcd_features(rb["run_dir"])
            self.assertEqual(fa["schema"]["selected_variables"], fb["schema"]["selected_variables"])     # selection used TRAIN coverage only
            self.assertNotIn("val_test_only_lab", fb["schema"]["selected_variables"])
            self.assertEqual(fa["state"]["kept_names"], fb["state"]["kept_names"])
            for k in ("lo", "hi", "median", "mean", "std", "kept_idx"):
                np.testing.assert_array_equal(fa["state"][k], fb["state"][k], err_msg=k)           # preprocessing fitted on TRAIN only
            tr = (fa["patients"]["split"] == "train").to_numpy()
            np.testing.assert_array_equal(np.asarray(fa["X"])[tr], np.asarray(fb["X"])[tr])          # every TRAIN row bit-identical
            self.assertFalse(np.array_equal(np.asarray(fa["X"])[~tr], np.asarray(fb["X"])[~tr]))     # (sanity) the corruption did reach val/test rows

    def test_rows_after_a_patients_icu_exit_never_enter_the_features(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pats, _ = make_dataset(root / "P", n_pat=300, seed=5, post_exit_poison=True)
            man = make_manifest(pats)
            r = self._build(root, pats, man, "P")
            f = tf.load_tcd_features(r["run_dir"])
            names = [e["name"] for e in f["schema"]["features"]]
            short = (f["patients"]["window_end_minutes"] < 1440).to_numpy()
            self.assertGreater(short.sum(), 10)
            raw = np.asarray(f["raw"])
            for col in ("creatinine__max", "bun__max"):
                v = raw[short, names.index(col)]
                self.assertLess(np.nanmax(v), 900, col)                                              # the 999 planted after exit is nowhere
            rep = json.loads((Path(r["run_dir"]) / "availability_report.json").read_text())
            self.assertGreater(rep["read_statistics_lab"]["rows_dropped_after_window"], 2 * short.sum() - 1)   # ...and the reader reported dropping them


class TestVentilationSupportOverlap(unittest.TestCase):
    def test_overlap_classes(self):
        self.assertEqual(tf.overlap_class("creatinine")[0], "direct"); self.assertEqual(tf.overlap_class("peep")[0], "direct")
        self.assertIn("ventilator", tf.overlap_class("peep")[1]); self.assertEqual(tf.overlap_class("sodium"), ("none", ""))
        a = tf.overlap_audit(["peep", "fio2", "sodium", "tv", "pf_ratio", "x&y", "patient"])
        self.assertEqual(a, {"lab_ingredients_of_the_label": ["fio2", "pf_ratio"], "ventilation_support_settings": ["peep", "tv"]})

    def test_a_peep_variable_is_kept_as_a_feature_and_flagged_in_the_schema(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pats, lab = make_dataset(root / "data", n_pat=200, seed=8)
            ids = pats["patientunitstayid"].to_numpy()
            extra = pd.DataFrame({"patientunitstayid": np.repeat(ids[:120], 3), "labresultoffset": np.tile([60, 300, 700], 120), "labname": "PEEP", "labresult": np.random.default_rng(0).choice([5.0, 8.0, 10.0, 12.0], 360)})        # varying values (a constant would be dropped by the training-only filter)
            pd.concat([lab, extra]).to_csv(root / "data" / "lab.csv", index=False)
            man = make_manifest(pats)
            r = tf.build_tcd_features({"data_root": str(root / "data")}, pats, man, tcfg(**{"audit.n_patients": 0}), root / "out")
            f = tf.load_tcd_features(r["run_dir"])
            ent = {e["name"]: e for e in f["schema"]["features"]}
            self.assertIn("peep__mean", ent)                                                               # KEPT as a feature
            self.assertEqual(ent["peep__mean"]["overlap_with_incident_label"], "direct"); self.assertIn("ventilator", ent["peep__mean"]["overlap_basis"])
            self.assertEqual(ent["sodium__mean"]["overlap_with_incident_label"], "none")
            self.assertIn("peep__mean", f["state"]["kept_names"])
            self.assertIn("(ventilator", (Path(r["run_dir"]) / "feature_definitions.md").read_text())


class TestOrchestrationAndCaching(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.pats, _ = make_dataset(self.root / "data", n_pat=200, seed=2)
        self.man = make_manifest(self.pats)
        self.cfg = tcfg(**{"audit.n_patients": 30})
        self.src = {"data_root": str(self.root / "data")}
        self.out = self.root / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, cfg=None, man=None, **kw):
        return tf.build_tcd_features(self.src, self.pats, man or self.man, cfg or self.cfg, self.out, **kw)

    def test_artifacts_schema_patient_order_and_split_counts(self):
        r = self.build(); d = Path(r["run_dir"])
        for n in tf.FEATURE_FILES + ["manifest.json"]:
            self.assertTrue((d / n).is_file(), n)
        f = tf.load_tcd_features(d)
        ids = f["patients"]["patientunitstayid"].to_numpy()
        self.assertTrue((np.diff(ids) > 0).all()); self.assertEqual(len(ids), len(self.pats))                    # sorted, every manifest patient exactly once
        sp = {int(k): v for k, v in self.man.patient_to_split.items()}
        self.assertEqual(f["patients"]["split"].tolist(), [sp[i] for i in ids])                                      # the PERMANENT manifest's assignment, untouched
        self.assertEqual(f["raw"].shape[0], len(ids)); self.assertEqual(f["X"].shape[0], len(ids))
        names = [e["name"] for e in f["schema"]["features"]]
        self.assertEqual(len(names), len(set(names))); self.assertEqual(f["raw"].shape[1], len(names))
        tn = f["schema"]["transformed_names"]; self.assertEqual(f["X"].shape[1], len(tn))
        self.assertTrue(set(f["state"]["kept_names"]) <= set(names))
        self.assertTrue(np.isfinite(np.asarray(f["X"])).all()); self.assertTrue(f["manifest"]["cutoff_audit_passed"])
        fams = {e["family"] for e in f["schema"]["features"]}
        self.assertTrue({"A", "B", "C", "F", "G"} <= fams)
        ov = {e["name"]: e["overlap_with_incident_label"] for e in f["schema"]["features"]}
        self.assertEqual(ov["creatinine__mean"], "direct"); self.assertEqual(ov["pt__n_draws"], "partial"); self.assertEqual(ov["sodium__mean"], "none")
        self.assertIn("pf_ratio__mean", names)                                                                       # family D present (aligned paO2/FiO2)
        self.assertNotIn("shock_index__mean", names)                                                                 # ...and the vitals-based ones are absent, declared unavailable
        self.assertIn("shock_index", {u["name"] for u in f["schema"]["unavailable"]})
        rep = json.loads((d / "availability_report.json").read_text())
        self.assertEqual(sum(rep["patients"][k] for k in ("train", "val", "test")), rep["patients"]["total"])

    def test_the_encoder_input_is_one_vector_per_patient_with_no_per_family_blocks(self):
        f = tf.load_tcd_features(self.build()["run_dir"])
        self.assertEqual(f["X"].ndim, 2); self.assertEqual(f["X"].shape[0], len(self.pats))

    def test_cache_reuse_new_versions_and_never_overwrite(self):
        r1 = self.build(); self.assertFalse(r1["cache_hit"])
        d1 = Path(r1["run_dir"]); h1 = artifacts.sha256_file(d1 / "tcd_features_raw.npy")
        r2 = self.build(); self.assertTrue(r2["cache_hit"]); self.assertEqual(r2["run_id"], r1["run_id"])
        r3 = self.build(force=True, run_id="forced"); self.assertFalse(r3["cache_hit"]); self.assertNotEqual(r3["run_dir"], r1["run_dir"])
        self.assertEqual(artifacts.sha256_file(Path(r3["run_dir"]) / "tcd_features_raw.npy"), h1)                    # deterministic recompute
        self.assertEqual(artifacts.sha256_file(d1 / "tcd_features_raw.npy"), h1)                                     # old run untouched
        with self.assertRaises(FileExistsError):
            self.build(force=True, run_id="forced")
        self.assertEqual(len(list(self.out.glob("**/*.tmp"))), 0)

    def test_changing_anything_that_affects_a_feature_changes_the_identity(self):
        r1 = self.build()
        r2 = self.build(tcfg(**{"audit.n_patients": 30, "half_split_minutes": 600})); self.assertFalse(r2["cache_hit"])
        c3 = tcfg(**{"audit.n_patients": 30}); c3["variables"]["ranges"]["creatinine"] = [None, 1.5]
        r3 = self.build(c3); self.assertFalse(r3["cache_hit"])
        m4 = SplitManifest(**{**self.man.to_dict(), "patient_to_split": {**self.man.patient_to_split, str(int(self.pats.patientunitstayid[0])): "test"}})
        r4 = self.build(man=m4); self.assertFalse(r4["cache_hit"])                                                  # selection/preprocessing depend on the manifest's train set
        self.assertEqual(len({r1["run_id"], r2["run_id"], r3["run_id"], r4["run_id"]}), 4)
        c5 = tcfg(**{"audit.n_patients": 30}); c5["training"]["gamma"] = 3.0; c5["encoder"]["embedding_dim"] = 8
        self.assertTrue(self.build(c5)["cache_hit"])                                                                # training/encoder settings do NOT invalidate the features

    def test_a_tampered_artifact_is_not_reused(self):
        r1 = self.build()
        p = Path(r1["run_dir"]) / "tcd_features_transformed.npy"; b = bytearray(p.read_bytes()); b[len(b) // 2] ^= 0xFF; p.write_bytes(bytes(b))     # flip one byte (overwriting with zeros could change nothing)
        r2 = self.build(); self.assertFalse(r2["cache_hit"])

    def test_changing_the_lab_table_invalidates_the_cache(self):
        r1 = self.build()
        with open(self.root / "data" / "lab.csv", "a") as f:
            f.write("1000,50,creatinine,1.0\n")
        self.assertFalse(self.build()["cache_hit"])

    def test_no_training_patients_is_a_clear_error(self):
        m = SplitManifest(**{**self.man.to_dict(), "patient_to_split": {k: "test" for k in self.man.patient_to_split}})
        with self.assertRaises(ValueError):
            self.build(man=m)


if __name__ == "__main__":
    unittest.main()
