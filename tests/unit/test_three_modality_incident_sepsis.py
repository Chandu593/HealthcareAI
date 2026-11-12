"""Incident sepsis-proxy label (src/three_modality/incident_sepsis.py): window boundaries, prevalent exclusion, co-occurrence, censoring policies,
evidence cap at ICU exit, cohort-flow reconciliation, run identity / never-overwrite, and separation from the legacy label files."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splitting.patient_split import SplitManifest
from src.three_modality import incident_sepsis as inc
from src.three_modality.config import INCIDENT_LABEL_DEFINITION_VERSION, THREE_MODALITY_DEFAULTS, get_incident_sepsis_cfg, validate_incident_sepsis_cfg


def cfg(**over):
    c = validate_incident_sepsis_cfg(copy.deepcopy(THREE_MODALITY_DEFAULTS["incident_sepsis"]))
    c["followup"]["censoring_policy"] = "exclude_censored"        # the fixtures below were hand-labelled for this policy; the REAL default is tested separately
    for k, v in over.items():
        if "." in k:
            a, b = k.split(".")
            c[a][b] = v
        else:
            c[k] = v
    return validate_incident_sepsis_cfg(c)


def ev(abx=(), creat=(), plt_=(), pf=(), vent=(), vaso=(), culture=()):
    """Evidence already in (pidx, hour[, value]) form."""
    def mk(rows, value):
        cols = ["pidx", "hour"] + (["value"] if value else [])
        return pd.DataFrame(list(rows), columns=cols) if rows else inc._empty(*cols)
    return inc.EvidenceEvents(abx=mk(abx, False), culture=mk(culture, False), creat=mk(creat, True), plt=mk(plt_, True), pf=mk(pf, True),
                              vent=mk(vent, False), vaso=mk(vaso, False))


def inc_eval(e, n=1, c=None):
    c = c or cfg()
    return inc.evaluate_window(e, n, c["cutoff_hours"] + 1, c["horizon_end_hours"], c), inc.evaluate_window(e, n, -c["prevalent_lookback_hours"], c["cutoff_hours"], c)


class TestConfig(unittest.TestCase):
    def test_defaults_are_24_to_96(self):
        c = get_incident_sepsis_cfg({})
        self.assertEqual((c["cutoff_hours"], c["horizon_end_hours"]), (24, 96))
        self.assertEqual(c["followup"]["censoring_policy"], "alive_discharge_as_negative")       # PRIMARY analysis

    def test_all_three_policies_remain_selectable(self):
        for pol in ("alive_discharge_as_negative", "require_full_followup_all", "exclude_censored"):
            c = get_incident_sepsis_cfg({"three_modality": {"incident_sepsis": {"followup": {"censoring_policy": pol}}}})
            self.assertEqual(c["followup"]["censoring_policy"], pol)

    def test_overrides_merge_and_validate(self):
        c = get_incident_sepsis_cfg({"three_modality": {"incident_sepsis": {"horizon_end_hours": 72, "organ_dysfunction": {"sofa_threshold": 3}}}})
        self.assertEqual((c["horizon_end_hours"], c["organ_dysfunction"]["sofa_threshold"]), (72, 3))
        self.assertIn("platelets", c["organ_dysfunction"]["lab_names"])           # untouched siblings survive the deep merge
        for bad in ({"horizon_end_hours": 24}, {"cutoff_hours": 24.5}, {"followup": {"censoring_policy": "nope"}}, {"infection": {"evidence": "x"}},
                    {"label_validity": {"min_medication_coverage": 1.5}}, {"co_occurrence_window_hours": 0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                get_incident_sepsis_cfg({"three_modality": {"incident_sepsis": bad}})


class TestBinning(unittest.TestCase):
    def test_hour_24_is_observation_side_and_hour_25_is_event_side(self):
        self.assertEqual(inc.hour_bin([-1440, -1, 0, 1, 60, 61, 1440, 1441, 5760]).tolist(), [-24, 0, 0, 1, 1, 2, 24, 25, 96])


class TestEvaluateWindow(unittest.TestCase):
    def test_incident_positive_when_both_elements_fall_after_the_cutoff(self):
        i, p = inc_eval(ev(abx=[(0, 30)], creat=[(0, 40, 2.5)]))
        self.assertTrue(i["met"][0]); self.assertEqual(i["first_hour"][0], 40.0); self.assertFalse(p["met"][0])

    def test_order_does_not_matter_event_hour_is_when_the_second_element_arrives(self):
        i, _ = inc_eval(ev(creat=[(0, 30, 2.5)], abx=[(0, 45)]))
        self.assertTrue(i["met"][0]); self.assertEqual(i["first_hour"][0], 45.0)

    def test_elements_too_far_apart_do_not_meet_criteria_but_diagnostics_still_see_both(self):
        i, _ = inc_eval(ev(abx=[(0, 26)], creat=[(0, 60, 2.5)]))
        self.assertFalse(i["met"][0]); self.assertTrue(i["org_any"][0]); self.assertTrue(i["infection_any"][0])

    def test_co_occurrence_window_boundary_is_w_minus_one_bins(self):
        self.assertTrue(inc_eval(ev(creat=[(0, 25, 2.5)], abx=[(0, 48)]))[0]["met"][0])     # bins differ by 23 -> inside the 24-bin trailing window
        self.assertFalse(inc_eval(ev(creat=[(0, 25, 2.5)], abx=[(0, 49)]))[0]["met"][0])    # bins differ by 24 -> outside

    def test_pre_cutoff_evidence_never_contributes_to_the_incident_event(self):
        i, p = inc_eval(ev(abx=[(0, 20)], creat=[(0, 40, 2.5)]))                          # infection evidence only BEFORE the cutoff
        self.assertFalse(i["met"][0]); self.assertFalse(i["infection_any"][0]); self.assertFalse(p["met"][0])
        i2, _ = inc_eval(ev(abx=[(0, 40)], creat=[(0, 20, 2.5)]))                          # organ dysfunction only before the cutoff
        self.assertFalse(i2["met"][0]); self.assertFalse(i2["org_any"][0])

    def test_trailing_window_is_clipped_at_the_cutoff_so_observation_window_values_cannot_leak_in(self):
        i, _ = inc_eval(ev(abx=[(0, 25)], creat=[(0, 24, 4.0)]))                           # severe creatinine at hour 24 (prevalent side)
        self.assertFalse(i["met"][0]); self.assertEqual(i["max_sofa"][0], 0)

    def test_evidence_exactly_at_hour_24_counts_as_prevalent(self):
        i, p = inc_eval(ev(abx=[(0, 24)], creat=[(0, 24, 2.5)]))
        self.assertTrue(p["met"][0]); self.assertFalse(i["met"][0])

    def test_pre_icu_negative_hour_evidence_counts_for_prevalence(self):
        self.assertTrue(inc_eval(ev(abx=[(0, -5)], creat=[(0, 3, 2.5)]))[1]["met"][0])
        self.assertFalse(inc_eval(ev(abx=[(0, -20)], creat=[(0, 10, 2.5)]))[1]["met"][0])  # 30 bins apart > window

    def test_events_after_the_horizon_are_ignored_and_hour_96_is_included(self):
        self.assertTrue(inc_eval(ev(abx=[(0, 96)], creat=[(0, 96, 2.5)]))[0]["met"][0])
        self.assertFalse(inc_eval(ev(abx=[(0, 97)], creat=[(0, 97, 2.5)]))[0]["met"][0])

    def test_respiratory_bands_three_and_four_need_ventilation_support(self):
        c = cfg(**{"organ_dysfunction.sofa_threshold": 3})
        pf = [(0, 30, 150.0)]
        self.assertFalse(inc_eval(ev(abx=[(0, 30)], pf=pf), c=c)[0]["met"][0])                          # P/F<200 without support is capped at 2
        self.assertTrue(inc_eval(ev(abx=[(0, 30)], pf=pf, vent=[(0, 29)]), c=c)[0]["met"][0])
        c2 = cfg(**{"organ_dysfunction.sofa_threshold": 3, "organ_dysfunction.use_ventilation_support": False})
        self.assertTrue(inc_eval(ev(abx=[(0, 30)], pf=pf), c=c2)[0]["met"][0])

    def test_other_domains_and_vasopressor_presence(self):
        self.assertTrue(inc_eval(ev(abx=[(0, 30)], vaso=[(0, 31)]))[0]["met"][0])                        # vasopressor present = 2
        i, _ = inc_eval(ev(abx=[(0, 30)], plt_=[(0, 31, 40.0)]))                                          # platelets 40 -> band 3
        self.assertTrue(i["met"][0]); self.assertEqual(i["max_sofa"][0], 3)
        self.assertFalse(inc_eval(ev(abx=[(0, 30)], creat=[(0, 31, 1.5)]))[0]["met"][0])                  # creatinine 1.5 -> band 1 < threshold 2

    def test_culture_corroboration_mode_requires_both_inside_the_window(self):
        c = cfg(**{"infection.evidence": "antibiotic_plus_culture"})
        base = dict(abx=[(0, 30)], creat=[(0, 31, 2.5)])
        self.assertFalse(inc_eval(ev(**base), c=c)[0]["met"][0])                                          # no culture
        i, _ = inc_eval(ev(culture=[(0, 35)], **base), c=c)
        self.assertTrue(i["met"][0]); self.assertEqual(i["first_hour"][0], 35.0)                          # evidence hour = later of abx/culture
        self.assertFalse(inc_eval(ev(culture=[(0, 20)], **base), c=c)[0]["met"][0])                       # culture before the cutoff does not qualify
        self.assertFalse(inc_eval(ev(culture=[(0, 80)], **base), c=c)[0]["met"][0])                       # culture too far from the antibiotic

    def test_chunking_over_patients_does_not_change_the_result(self):
        rows = dict(abx=[(0, 30), (2, 50), (3, 26), (4, 90)], creat=[(0, 31, 2.5), (1, 40, 5.0), (2, 52, 2.2), (3, 80, 2.5), (4, 91, 3.6)])
        e, c = ev(**rows), cfg()
        a = inc.evaluate_window(e, 6, 25, 96, c, chunk_patients=20000)
        b = inc.evaluate_window(e, 6, 25, 96, c, chunk_patients=2)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a["met"].tolist(), [True, False, True, False, True, False])


class TestClassification(unittest.TestCase):
    def table(self, policy="exclude_censored", min_cov=0.0, rows=None):
        # rows: (hospital, exit_hours, died, prevalent, event)
        rows = rows or []
        e = pd.DataFrame({"patientunitstayid": range(1, len(rows) + 1), "hospitalid": [r[0] for r in rows], "split": "train",
                          "icu_exit_minutes": [r[1] * 60 for r in rows], "died_in_icu": [r[2] for r in rows], "pre_status": np.nan})
        prev = pd.DataFrame({"met": [r[3] for r in rows], "first_hour": np.nan, "max_sofa": 0, "org_any": False, "infection_any": False})
        inc_ = pd.DataFrame({"met": [r[4] for r in rows], "first_hour": [30.0 if r[4] else np.nan for r in rows], "max_sofa": 0, "org_any": False, "infection_any": False})
        c = cfg(**{"followup.censoring_policy": policy, "label_validity.min_medication_coverage": min_cov})
        cov = pd.Series({1: 1.0, 2: 0.1})
        return inc.classify_patients(e, prev, inc_, np.zeros(len(rows), dtype=bool), cov, c)

    ROWS = [(1, 120, 0, False, False),    # complete follow-up, no event      -> negative
            (1, 50, 0, False, False),     # alive early discharge, no event   -> censored
            (1, 50, 1, False, False),     # ICU death, no event               -> competing risk
            (1, 50, 0, False, True),      # event, then left early            -> positive (kept) / excluded under require_full
            (1, 120, 0, False, True),     # event, full follow-up             -> positive
            (1, 120, 0, True, True),      # prevalent                         -> excluded
            (2, 120, 0, False, True),     # label-invalid hospital            -> excluded
            (1, 50, np.nan, False, False)]  # unknown exit status             -> censored (never negative)

    def test_default_policy_never_turns_early_exits_into_negatives(self):
        t = self.table(rows=self.ROWS, min_cov=0.8)
        self.assertEqual(t["status"].tolist(), [inc.NEG_COMPLETE, inc.CENS_ALIVE, inc.CENS_DEATH, inc.POSITIVE, inc.POSITIVE, inc.EXCL_PREVALENT,
                                                inc.EXCL_LABEL_INVALID_HOSPITAL, inc.CENS_UNKNOWN])
        self.assertEqual(int((t["label"] == 1).sum()), 2); self.assertEqual(int((t["label"] == 0).sum()), 1)
        self.assertEqual(int(t["included"].sum()), 3)

    def test_alive_discharge_as_negative_policy(self):
        t = self.table("alive_discharge_as_negative", 0.8, self.ROWS)
        self.assertEqual(t["status"][1], inc.NEG_ALIVE_DISCHARGE); self.assertEqual(t["label"][1], 0.0)
        self.assertEqual(t["status"][2], inc.CENS_DEATH); self.assertEqual(t["status"][7], inc.CENS_UNKNOWN)   # deaths / unknowns still excluded

    def test_require_full_followup_all_also_drops_early_positives(self):
        t = self.table("require_full_followup_all", 0.8, self.ROWS)
        self.assertEqual(t["status"][3], inc.EXCL_POS_INCOMPLETE); self.assertTrue(np.isnan(t["label"][3]))
        self.assertEqual(int((t["label"] == 1).sum()), 1)

    def test_label_validity_filter_can_be_disabled(self):
        self.assertEqual(self.table(rows=self.ROWS, min_cov=0.0)["status"][6], inc.POSITIVE)

    def test_exit_exactly_at_the_horizon_counts_as_complete_follow_up(self):
        t = self.table(rows=[(1, 96, 0, False, False), (1, 95.99, 0, False, False)])
        self.assertEqual(t["status"].tolist(), [inc.NEG_COMPLETE, inc.CENS_ALIVE])


class TestUniverse(unittest.TestCase):
    def test_landmark_boundary_and_invalid_rows(self):
        p = pd.DataFrame({"patientunitstayid": [1, 2, 3, 4, 5], "hospitalid": [7] * 5, "unitdischargeoffset": [1440, 1441, np.nan, -5, 5000],
                          "unitdischargestatus": ["Alive", "Expired", "Alive", "Alive", None]})
        m = SplitManifest(seed=1, train_frac=.7, val_frac=.15, test_frac=.15, stratify_on="none", sepsis_cohort=None,
                          patient_to_split={str(i): "train" for i in (1, 2, 3, 4, 5, 6)})
        u = inc.build_universe(p, m, cfg()).set_index("patientunitstayid")
        self.assertEqual(u.loc[1, "pre_status"], inc.PRE_ENDED_BY_CUTOFF)                  # exactly hour 24 -> no time after the cutoff
        self.assertTrue(pd.isna(u.loc[2, "pre_status"])); self.assertEqual(u.loc[2, "died_in_icu"], 1.0)
        self.assertEqual(u.loc[3, "pre_status"], inc.PRE_INVALID_EXIT); self.assertEqual(u.loc[4, "pre_status"], inc.PRE_INVALID_EXIT)
        self.assertTrue(np.isnan(u.loc[5, "died_in_icu"]))                                  # unknown exit status stays unknown
        self.assertEqual(u.loc[6, "pre_status"], inc.PRE_NOT_IN_PATIENT_TABLE)


# ---------------------------------------------------------------------------------------------------------------------------------------
# end-to-end through the chunked CSV reader
# ---------------------------------------------------------------------------------------------------------------------------------------
def write_csv(path, header, rows):
    pd.DataFrame(rows, columns=header).to_csv(path, index=False)


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        d = self.root / "data"; d.mkdir()
        # (id, hospital, exit_minutes, status)
        pats = [(1, 1, 7000, "Alive"), (2, 1, 7000, "Alive"), (3, 1, 1000, "Alive"), (4, 1, 3000, "Alive"), (5, 1, 3000, "Expired"),
                (6, 1, 7000, "Alive"), (7, 1, 3000, "Alive"), (8, 2, 7000, "Alive"), (9, 2, 7000, "Alive"), (11, 1, 2000, "Alive"),
                (12, 1, None, "Alive"), (13, 1, 7000, "Alive"), (14, 1, 7000, "Alive"),
                (15, 1, 7000, "Alive"), (16, 1, 7000, "Alive"), (17, 1, 7000, "Alive"), (18, 1, 7000, "Alive"), (19, 1, 7000, "Alive")]
        write_csv(d / "patient.csv", ["patientunitstayid", "hospitalid", "unitdischargeoffset", "unitdischargestatus"], pats)
        h = ["patientunitstayid", "drugname", "drugstartoffset"]
        meds = [(i, "Acetaminophen", 100) for i in (1, 2, 4, 5, 6, 7, 11, 3, 12, 13, 14, 15, 16, 17, 18, 19)]                   # every hospital-1 patient has a medication row
        meds += [(1, "Vancomycin", 1600), (6, "Vancomycin", 600), (7, "Vancomycin", 1600), (11, "Vancomycin", 2400),
                 (13, "Vancomycin", 1200),    # 13: antibiotic at hour 20 (observation window) ...
                 (14, "Vancomycin", 2400),    # 14: antibiotic at hour 40 ...
                 (15, "Vancomycin", 1440),    # 15: BOTH elements exactly at minute 1440 (hour 24) -> prevalent ("before or at hour 24")
                 (16, "Vancomycin", 5760),    # 16: BOTH elements exactly at minute 5760 (hour 96) -> still inside the event window
                 (17, "Vancomycin", 5761),    # 17: BOTH elements one minute past the horizon -> dropped
                 (18, "Vancomycin", 1440),    # 18: antibiotic AT hour 24 (not "after") + organ dysfunction at hour 40 -> neither prevalent nor incident
                 (19, "Vancomycin", 2400)]    # 19: organ dysfunction AT hour 24 + antibiotic at hour 40 -> neither prevalent nor incident
        write_csv(d / "medication.csv", h, meds)
        write_csv(d / "medication_imputed.csv", h, [(1, "Vancomycin", 1600)])                      # duplicate of patient 1's order: harmless
        write_csv(d / "medication_imputed1.csv", h, [(2, "Acetaminophen", 50)])
        write_csv(d / "lab.csv", ["patientunitstayid", "labresultoffset", "labname", "labresult"],
                  [(1, 1700, "creatinine", 2.5), (6, 700, "creatinine", 2.5), (7, 1700, "creatinine", 2.5), (11, 2500, "creatinine", 2.5),
                   (13, 2400, "creatinine", 2.5),   # ... 13: organ dysfunction only at hour 40 -> not prevalent, and no qualifying post-cutoff infection evidence
                   (14, 1200, "creatinine", 2.5),   # ... 14: organ dysfunction only at hour 20 -> not prevalent, and no qualifying post-cutoff organ dysfunction
                   (15, 1440, "creatinine", 2.5), (16, 5760, "creatinine", 2.5), (17, 5761, "creatinine", 2.5),
                   (18, 2400, "creatinine", 2.5), (19, 1440, "creatinine", 2.5),
                  
                   (2, 2000, "urinary creatinine", 300.0), (2, 2000, "direct bilirubin", 9.0), (2, 2100, "platelets x 1000", 250.0)])
        write_csv(d / "infusionDrug.csv", ["patientunitstayid", "infusionoffset", "drugname"], [(2, 100, "NS (ml/hr)")])
        write_csv(d / "treatment.csv", ["patientunitstayid", "treatmentoffset", "treatmentstring"], [(2, 100, "pulmonary|ventilation and oxygenation|mechanical ventilation")])
        write_csv(d / "microLab.csv", ["patientunitstayid", "culturetakenoffset"], [(1, 1650)])
        self.src = {"data_root": str(d)}
        self.patients = pd.read_csv(d / "patient.csv")
        ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 99]
        self.manifest = SplitManifest(seed=1, train_frac=.7, val_frac=.15, test_frac=.15, stratify_on="none", sepsis_cohort=None,
                                      patient_to_split={str(i): ("train", "val", "test")[k % 3] for k, i in enumerate(ids)})
        self.cfg = cfg(chunksize=3)                                                                 # tiny chunks -> many read passes
        self.out = self.root / "outputs" / "three_modality" / "incident_sepsis"

    def tearDown(self):
        self.tmp.cleanup()

    def run_build(self, c=None, **kw):
        return inc.build_incident_sepsis_label(self.src, self.patients, self.manifest, c or self.cfg, self.out, **kw)

    def status(self, res):
        return res["labels"].set_index("patientunitstayid")["status"].to_dict()

    def test_statuses_counts_and_reconciliation(self):
        r = self.run_build(); s = self.status(r)
        self.assertEqual(s, {1: inc.POSITIVE, 2: inc.NEG_COMPLETE, 3: inc.PRE_ENDED_BY_CUTOFF, 4: inc.CENS_ALIVE, 5: inc.CENS_DEATH, 6: inc.EXCL_PREVALENT,
                             7: inc.POSITIVE, 8: inc.EXCL_LABEL_INVALID_HOSPITAL, 9: inc.EXCL_LABEL_INVALID_HOSPITAL, 11: inc.CENS_ALIVE,
                             12: inc.PRE_INVALID_EXIT, 13: inc.NEG_COMPLETE, 14: inc.NEG_COMPLETE, 15: inc.EXCL_PREVALENT, 16: inc.POSITIVE, 17: inc.NEG_COMPLETE, 18: inc.NEG_COMPLETE, 19: inc.NEG_COMPLETE,
                             99: inc.PRE_NOT_IN_PATIENT_TABLE})
        f = r["report"]["flow"]
        self.assertEqual((f["manifest_patients"], f["initial_eligible_cohort"], f["excluded_prevalent"], f["incident_positive"], f["incident_negative"]), (19, 16, 2, 3, 6))
        self.assertEqual((f["other_exclusions"]["label_invalid_hospital_patients"], f["other_exclusions"]["label_invalid_hospitals"]), (2, 1))
        self.assertEqual((f["followup"]["censored_alive_discharge"], f["followup"]["censored_death_no_event"], f["followup"]["positives_with_incomplete_followup_kept"]), (2, 1, 1))
        self.assertTrue(f["reconciliation"]["ok"])
        lab = r["labels"].set_index("patientunitstayid")
        self.assertEqual((lab.loc[1, "event_hour"], lab.loc[16, "event_hour"]), (29.0, 96.0))                       # abx hour 27, creatinine hour 29 -> met when the second element arrives
        cov = r["report"]["evidence_coverage"]["patients_with_evidence_in_incident_window"]
        self.assertEqual((cov["creatinine"], cov["antibiotic_start"]), (5, 5))      # creatinine: 1, 7, 13, 16, 18   antibiotic: 1, 7, 14, 16, 19   (elements at or before hour 24 do not count; 11 and 17 are past their cap; 2's labs are decoy names)
        dg = r["report"]["diagnostics"]                                              # the limitations of the definition are quantified, not hidden
        self.assertEqual(dg["negatives_with_antibiotic_start_before_or_at_cutoff"], 2)                                                       # patients 13, 18
        self.assertEqual(dg["negatives_with_organ_dysfunction_in_event_window_but_no_qualifying_infection_evidence"], 2)                    # patients 13, 18
        self.assertEqual(dg["negatives_with_infection_evidence_but_no_organ_dysfunction"], 2)                                                # patients 14, 19

    def test_evidence_after_icu_exit_and_decoy_lab_names_never_count(self):
        s = self.status(self.run_build())
        self.assertEqual(s[11], inc.CENS_ALIVE)       # patient 11 leaves the ICU at 33 h; its antibiotic/creatinine after exit are dropped
        self.assertEqual(s[2], inc.NEG_COMPLETE)      # 'urinary creatinine' 300 and 'direct bilirubin' 9 are not matched (exact names)

    def test_policy_variants_and_sensitivity_table(self):
        r = self.run_build()
        sens = r["report"]["policy_sensitivity"]
        self.assertEqual(sens["exclude_censored"], {"positive": 3, "negative": 6, "prevalence": 3 / 9})
        self.assertEqual((sens["alive_discharge_as_negative"]["positive"], sens["alive_discharge_as_negative"]["negative"]), (3, 8))
        self.assertEqual((sens["require_full_followup_all"]["positive"], sens["require_full_followup_all"]["negative"]), (2, 6))
        c = cfg(chunksize=3, **{"followup.censoring_policy": "alive_discharge_as_negative"})
        r2 = self.run_build(c); self.assertEqual(r2["report"]["flow"]["incident_negative"], 8)
        c3 = cfg(chunksize=3, **{"followup.censoring_policy": "require_full_followup_all"})
        r3 = self.run_build(c3); self.assertEqual((r3["report"]["flow"]["incident_positive"], r3["report"]["flow"]["followup"]["excluded_positive_incomplete_followup"]), (2, 1))
        for a, b in ((r2, "alive_discharge_as_negative"), (r3, "require_full_followup_all")):             # the table predicted what the re-run produced
            self.assertEqual(a["report"]["flow"]["incident_negative"], sens[b]["negative"]); self.assertEqual(a["report"]["flow"]["incident_positive"], sens[b]["positive"])

    def test_culture_mode_changes_the_label_through_config_only(self):
        r = self.run_build(cfg(chunksize=3, **{"infection.evidence": "antibiotic_plus_culture"}))
        s = self.status(r)
        self.assertEqual(s[1], inc.POSITIVE)          # patient 1 has a culture at 1650 min (hour 28), antibiotic hour 27, creatinine hour 29
        self.assertEqual(s[7], inc.CENS_ALIVE)        # patient 7 has no culture -> no qualifying infection evidence -> censored (left early, no event)

    def test_never_overwrites_reuses_verified_runs_and_forces_new_versions(self):
        r1 = self.run_build(); self.assertFalse(r1["cache_hit"])
        d1 = Path(r1["run_dir"]); h1 = hashlib.sha256((d1 / "incident_sepsis_labels.csv").read_bytes()).hexdigest()
        r2 = self.run_build(); self.assertTrue(r2["cache_hit"]); self.assertEqual(r2["run_id"], r1["run_id"])
        r3 = self.run_build(force=True, run_id="forced_v2"); self.assertFalse(r3["cache_hit"]); self.assertNotEqual(r3["run_dir"], r1["run_dir"])
        self.assertEqual(hashlib.sha256((d1 / "incident_sepsis_labels.csv").read_bytes()).hexdigest(), h1)          # old run untouched
        self.assertEqual(hashlib.sha256((Path(r3["run_dir"]) / "incident_sepsis_labels.csv").read_bytes()).hexdigest(), h1)   # deterministic recompute
        with self.assertRaises(FileExistsError):
            self.run_build(force=True, run_id="forced_v2")
        (d1 / "incident_sepsis_labels.csv").write_text("tampered")                                                  # provenance check: a modified artifact is not reused ...
        r_mid = self.run_build(); self.assertTrue(r_mid["cache_hit"]); self.assertEqual(r_mid["run_id"], r3["run_id"])   # ... but the other, intact identical run still is
        (Path(r3["run_dir"]) / "incident_sepsis_labels.csv").write_text("tampered")
        r4 = self.run_build(); self.assertFalse(r4["cache_hit"]); self.assertNotIn(r4["run_id"], (r1["run_id"], r3["run_id"]))
        self.assertEqual(len(list(self.out.glob("**/*.tmp"))), 0)
        self.assertTrue((Path(r4["run_dir"]) / "manifest.json").is_file())

    def test_changing_a_parameter_or_the_split_creates_a_new_run(self):
        r1 = self.run_build()
        r2 = self.run_build(cfg(chunksize=3, horizon_end_hours=72)); self.assertFalse(r2["cache_hit"])
        m2 = SplitManifest(**{**self.manifest.to_dict(), "patient_to_split": {**self.manifest.patient_to_split, "1": "test"}})
        r3 = inc.build_incident_sepsis_label(self.src, self.patients, m2, self.cfg, self.out); self.assertFalse(r3["cache_hit"])
        self.assertEqual(len({r1["run_id"], r2["run_id"], r3["run_id"]}), 3)

    def test_legacy_label_files_and_other_output_trees_are_untouched(self):
        legacy = self.root / "outputs" / "sepsis_labels.csv"; paper = self.root / "outputs" / "paper" / "results.json"
        legacy.parent.mkdir(parents=True); paper.parent.mkdir(parents=True)
        legacy.write_text("patientunitstayid,sepsis_broad\n1,True\n"); paper.write_text("{}")
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (legacy, paper)}
        self.run_build()
        self.assertEqual({p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (legacy, paper)}, before)
        written = {p.relative_to(self.root / "outputs").parts[0:2] for p in (self.root / "outputs").rglob("*") if p.is_file()}
        self.assertEqual(written, {("sepsis_labels.csv",), ("paper", "results.json"), ("three_modality", "incident_sepsis")})

    def test_artifacts_and_documentation_are_written(self):
        d = Path(self.run_build()["run_dir"])
        for n in ("incident_sepsis_labels.csv", "hospital_summary.csv", "cohort_report.json", "cohort_report.md", "label_definition.md", "config_snapshot.json", "manifest.json"):
            self.assertTrue((d / n).is_file(), n)
        m = json.loads((d / "manifest.json").read_text())
        self.assertEqual(m["definition_version"], INCIDENT_LABEL_DEFINITION_VERSION); self.assertEqual(set(m["files"]), {"incident_sepsis_labels.csv", "hospital_summary.csv", "cohort_report.json", "cohort_report.md", "label_definition.md", "config_snapshot.json"})
        md = (d / "cohort_report.md").read_text()
        for needle in ("Initial eligible cohort", "PREVALENT", "Incident positive", "Incident negative", "competing risk", "Evidence coverage"):
            self.assertIn(needle, md)
        self.assertIn("ceil(minutes/60)", (d / "label_definition.md").read_text())

    def test_missing_required_tables_fail_loudly(self):
        (Path(self.src["data_root"]) / "lab.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_build()


if __name__ == "__main__":
    unittest.main()
