"""
Generate a tiny, schema-compatible SYNTHETIC eICU dataset for testing.

This does NOT contain, resemble, or derive from any real patient data --
values are drawn from plausible ranges purely to exercise the pipeline's
parsing / validation / chunking / cohort logic. Per Section 34 ("tests
must not require the complete eICU dataset") and Section 43 (do not
fabricate results, but synthetic schema-compatible fixtures are fine).

Usage:
    python scripts/generate_mock_eicu.py --out /path/to/mock_data --n-hospitals 3 --n-patients-per-hospital 20
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

ANTIBIOTICS = [
    "VANCOMYCIN 1000 MG IV SOLN", "PIPERACILLIN-TAZOBACTAM 3.375 GM IV SOLN",
    "MEROPENEM 500 MG IV SOLN", "CEFTRIAXONE 1 GM IV SOLN",
    "CEFEPIME 1 GM IV SOLN", "LINEZOLID 600 MG IV SOLN",
]
NON_ANTIBIOTICS = [
    "METOPROLOL TARTRATE 25 MG PO TABS", "ASPIRIN EC 81 MG PO TBEC",
    "INSULIN-GLARGINE *UNIT* INJ", "OMEPRAZOLE 20 MG PO CAPS",
    "ENOXAPARIN SODIUM 40 MG/0.4ML SC SOLN",
]
VASOPRESSORS = ["Norepinephrine (mcg/min)", "Epinephrine (mcg/min)", "Vasopressin (units/min)"]
NON_VASOPRESSORS = ["NS (ml/hr)", "Propofol (mcg/kg/min)", "Insulin (units/hr)"]
LAB_NAMES = ["platelets", "total bilirubin", "creatinine", "pH", "fibrinogen", "PT - INR"]
DIAGNOSES = [
    "cardiovascular|chest pain / ASHD|coronary artery disease",
    "pulmonary|disorders of the airways|COPD",
    "infectious diseases|sepsis|sepsis due to urinary tract infection",
    "renal|acute renal failure|acute renal failure due to sepsis",
]
TREATMENTS = [
    "cardiovascular|hypertension|angiotensin II receptor blocker",
    "cardiovascular|shock|vasopressors|norepinephrine",
    "infectious diseases|medications|therapeutic antibacterials|vancomycin",
    "pulmonary|ventilation and oxygenation|mechanical ventilation",
    "renal|dialysis|hemodialysis",
]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _draw_lab_value(rng: np.random.Generator, labname: str) -> float:
    """Per-labname realistic-ish distributions -- a single generic
    Normal(100, 40) across every labname (the original approach) made
    e.g. 'total bilirubin' (normal range < 1.2 mg/dL) draw values around
    100, which trivially maxes out the liver SOFA-like domain for nearly
    every mock patient. This does not need to be clinically precise, just
    in the right order of magnitude so downstream scoring logic gets a
    realistic mix of normal/abnormal rather than an artifact of the
    generator."""
    if labname == "platelets":
        return max(5.0, float(rng.normal(190, 85)))
    if labname == "total bilirubin":
        return max(0.1, float(rng.gamma(1.6, 0.55)))
    if labname == "creatinine":
        return max(0.2, float(rng.gamma(2.2, 0.42)))
    if labname == "pH":
        return float(rng.normal(7.38, 0.06))
    if labname == "fibrinogen":
        return max(50.0, float(rng.normal(320, 90)))
    if labname == "PT - INR":
        return max(0.6, float(rng.normal(1.1, 0.3)))
    return float(rng.normal(100, 40))


def generate(out_dir: str, n_hospitals: int = 3, n_patients_per_hospital: int = 20, seed: int = 42) -> None:
    rng = _rng(seed)
    random.seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hospital_ids = list(range(1, n_hospitals + 1))
    hospital_df = pd.DataFrame({
        "hospitalid": hospital_ids,
        "numbedscategory": rng.choice(["<100", "100 - 249", "250 - 499", ">=500"], size=n_hospitals),
        "teachingstatus": rng.choice(["t", "f"], size=n_hospitals),
        "region": rng.choice(["Midwest", "South", "West", "Northeast"], size=n_hospitals),
    })
    hospital_df.to_csv(out / "hospital.csv", index=False)

    # apachePredVar.teachtype/region are constant 0/3 for EVERY row, not
    # varied per-hospital -- matching a real, confirmed finding, not a
    # simplification. CORRECTED 2026-08-24: an earlier version of this
    # generator (2026-08-2x) derived these per-hospital from
    # hospital_df above, on the assumption that apachePredVar.teachtype/
    # region should mirror each patient's own hospital's characteristics.
    # That assumption was WRONG: direct inspection of the real eICU
    # export's raw apachePredVar.csv showed teachtype=0 and region=3 for
    # literally all 171,177 rows, with zero variation -- a genuine
    # property of this data, not a sampling artifact (see
    # feature_engineering.APACHEPREDVAR_CATEGORICAL_FEATURES' comment).
    # The real per-hospital diversity instead lives in hospital.csv's
    # own region/teachingstatus columns above, which this generator
    # already correctly varies and always did -- see
    # attach_hospital_features() in feature_engineering.py, which is
    # what Phase 5 now uses instead of apachePredVar for this.

    patient_rows, apr_rows, apv_rows, diag_rows, lab_rows = [], [], [], [], []
    med_rows, admdrug_rows, inf_rows, micro_rows, treat_rows = [], [], [], [], []
    medimp_rows, medimp1_rows = [], []

    stay_id = 141000
    apache_id = 1
    diag_id = 1
    lab_id = 1
    med_id = 1
    admdrug_id = 1
    inf_id = 1
    micro_id = 1
    apv_id = 1
    treat_id = 1
    medimp_id = 1
    medimp1_id = 1

    for hid in hospital_ids:
        for _ in range(n_patients_per_hospital):
            stay_id += 1
            expired = bool(rng.random() < 0.15)
            age = int(rng.integers(20, 95))
            gender = rng.choice(["Male", "Female"])
            hs_stay = stay_id + 100000

            patient_rows.append({
                "patientunitstayid": stay_id,
                "patienthealthsystemstayid": hs_stay,
                "gender": gender,
                "age": str(age),
                "ethnicity": rng.choice(["Caucasian", "African American", "Hispanic", "Asian", "Other"]),
                "hospitalid": hid,
                "wardid": int(rng.integers(50, 150)),
                "apacheadmissiondx": rng.choice(DIAGNOSES),
                "admissionheight": round(float(rng.normal(168, 10)), 1),
                "hospitaladmittime24": "10:00:00",
                "hospitaladmitoffset": 0,
                "hospitaladmitsource": rng.choice(["Emergency Department", "Direct Admit", "Floor"]),
                "hospitaldischargeyear": 2020,
                "hospitaldischargetime24": "12:00:00",
                "hospitaldischargeoffset": int(rng.integers(500, 10000)),
                "hospitaldischargelocation": "Home" if not expired else "Death",
                "hospitaldischargestatus": "Expired" if expired else "Alive",
                "unittype": "Med-Surg ICU",
                "unitadmittime24": "10:05:00",
                "unitadmitsource": "Emergency Department",
                "unitvisitnumber": 1,
                "unitstaytype": "admit",
                "admissionweight": round(float(rng.normal(80, 15)), 1),
                "dischargeweight": round(float(rng.normal(80, 15)), 1),
                "unitdischargetime24": "09:00:00",
                "unitdischargeoffset": int(rng.integers(200, 5000)),
                "unitdischargelocation": "Death" if expired else "Floor",
                "unitdischargestatus": "Expired" if expired else "Alive",
                "uniquepid": f"999-{stay_id}",
            })

            apache_id += 1
            apr_rows.append({
                "apachepatientresultsid": apache_id,
                "patientunitstayid": stay_id,
                "physicianspeciality": "critical care medicine (CCM)",
                "physicianinterventioncategory": "Unknown",
                "acutephysiologyscore": int(rng.integers(20, 90)),
                "apachescore": int(rng.integers(30, 100)),
                "apacheversion": "IVa",
                "predictedicumortality": round(float(rng.random() * 0.5), 4),
                "actualicumortality": "EXPIRED" if expired else "ALIVE",
                "predictediculos": round(float(rng.normal(4, 2)), 2),
                "actualiculos": round(float(rng.normal(4, 2)), 2),
                "predictedhospitalmortality": round(float(rng.random() * 0.5), 4),
                "actualhospitalmortality": "EXPIRED" if expired else "ALIVE",
                "predictedhospitallos": round(float(rng.normal(8, 3)), 2),
                "actualhospitallos": round(float(rng.normal(8, 3)), 2),
                "preopmi": 0, "preopcardiaccath": 0, "ptcawithin24h": 0,
                "unabridgedunitlos": round(float(rng.normal(4, 2)), 2),
                "unabridgedhosplos": round(float(rng.normal(8, 3)), 2),
                "actualventdays": float(rng.integers(0, 5)) if rng.random() < 0.3 else np.nan,
                "predventdays": float(rng.normal(3, 1)) if rng.random() < 0.3 else np.nan,
                "unabridgedactualventdays": float(rng.integers(0, 5)) if rng.random() < 0.3 else np.nan,
            })

            # ~85% of real eICU patients have an apachePredVar row
            # (171,177 / 200,859 in the actual dataset) -- the other
            # ~15% have NONE at all. This is exactly the scenario that
            # exposed a real dtype-corruption bug when this project was
            # first run against real data (see
            # tests/unit/test_sepsis_and_cohort_report.py's
            # partial-coverage regression tests): SOFA scoring silently
            # omits these patients from sofa_df entirely, and downstream
            # merges must handle their absence cleanly. Simulating this
            # here means the integration tests exercise that path too,
            # not just carefully hand-built unit tests.
            has_apache_predvar = rng.random() < 0.85
            apv_id += 1
            # Correlate respiratory severity so PaO2/FiO2 ratios aren't
            # artificially abnormal for most patients: most ICU stays are
            # mild (room air or low-flow O2), a minority are more severe.
            severity_roll = rng.random()
            if severity_roll < 0.55:
                fio2_val = 21.0
                pao2_val = float(rng.normal(92, 12))
            elif severity_roll < 0.80:
                fio2_val = float(rng.choice([30, 40]))
                pao2_val = float(rng.normal(78, 12))
            else:
                fio2_val = float(rng.choice([60, 100]))
                pao2_val = float(rng.normal(80, 20))
            pao2_val = max(35.0, pao2_val)
            creatinine_val = round(float(rng.gamma(2.2, 0.42)), 2)  # mean ~0.9, right-skewed tail for AKI cases

            if has_apache_predvar:
                apv_rows.append({
                "apachepredvarid": apv_id,
                "patientunitstayid": stay_id,
                "sicuday": 1,
                "saps3day1": 0, "saps3today": 0, "saps3yesterday": 0,
                "gender": 1 if gender == "Male" else 0,
                "teachtype": 0, "region": 3,  # constant in real data -- see comment above
                "bedcount": int(rng.integers(10, 40)),
                "admitsource": int(rng.integers(1, 10)),
                "graftcount": 0,
                "meds": int(rng.integers(0, 10)),
                "verbal": int(rng.integers(1, 6)),
                "motor": int(rng.integers(1, 7)),
                "eyes": int(rng.integers(1, 5)),
                "age": float(age),
                "admitdiagnosis": rng.choice(DIAGNOSES),
                "thrombolytics": 0, "diedinhospital": int(expired),
                "aids": 0, "hepaticfailure": 0, "lymphoma": 0,
                "metastaticcancer": 0, "leukemia": 0, "immunosuppression": 0,
                "cirrhosis": 0,
                "electivesurgery": float(rng.integers(0, 2)) if rng.random() < 0.2 else np.nan,
                "activetx": 1, "readmit": 0, "ima": 0, "midur": 0,
                "ventday1": int(rng.integers(0, 2)),
                "oobventday1": 0, "oobintubday1": 0,
                "diabetes": int(rng.integers(0, 2)),
                "managementsystem": 0, "var03hspxlos": 0,
                "pao2": round(pao2_val, 1),
                "fio2": round(fio2_val, 1),
                "ejectfx": -1,
                "creatinine": creatinine_val,
                "dischargelocation": 4,
                "visitnumber": 1, "amilocation": -1,
                "day1meds": int(rng.integers(0, 10)),
                "day1verbal": int(rng.integers(1, 6)) if severity_roll > 0.75 else 5,
                "day1motor": int(rng.integers(1, 7)) if severity_roll > 0.75 else 6,
                "day1eyes": int(rng.integers(1, 5)) if severity_roll > 0.75 else 4,
                "day1pao2": round(pao2_val, 1),
                "day1fio2": round(fio2_val, 1),
                })

            # Diagnoses: 1-3 per patient
            for _ in range(int(rng.integers(1, 4))):
                diag_id += 1
                diag_rows.append({
                    "diagnosisid": diag_id,
                    "patientunitstayid": stay_id,
                    "activeupondischarge": bool(rng.random() < 0.5),
                    "diagnosisoffset": int(rng.integers(0, 500)),
                    "diagnosisstring": rng.choice(DIAGNOSES),
                    "icd9code": "038.9, A41.9",
                    "diagnosispriority": "Primary" if rng.random() < 0.3 else "Other",
                })

            # Labs: 5-15 per patient across a few labnames
            for _ in range(int(rng.integers(5, 16))):
                lab_id += 1
                chosen_labname = str(rng.choice(LAB_NAMES))
                lab_rows.append({
                    "labid": lab_id,
                    "patientunitstayid": stay_id,
                    "labresultoffset": int(rng.integers(0, 2000)),
                    "labtypeid": 1,
                    "labname": chosen_labname,
                    "labresult": round(_draw_lab_value(rng, chosen_labname), 2),
                    "labresulttext": "",
                    "labmeasurenamesystem": "mg/dL",
                    "labmeasurenameinterface": "mg/dL",
                    "labresultrevisedoffset": int(rng.integers(0, 2000)),
                })

            # Medications: mix of antibiotics and non-antibiotics
            # Some patients genuinely have zero medication orders in the
            # sampled window (or, in real eICU, could belong to a
            # hospital/timeframe with sparse medication charting) --
            # allow 0, not just >=3, so infection_proxy's "patient absent
            # from medication entirely" path gets exercised too.
            n_meds = int(rng.integers(0, 10))
            for _ in range(n_meds):
                med_id += 1
                is_abx = rng.random() < 0.25
                start = int(rng.integers(0, 1500))
                med_rows.append({
                    "medicationid": med_id,
                    "patientunitstayid": stay_id,
                    "drugorderoffset": start,
                    "drugstartoffset": start,
                    "drugivadmixture": "No",
                    "drugordercancelled": "No",
                    "drugname": rng.choice(ANTIBIOTICS) if is_abx else rng.choice(NON_ANTIBIOTICS),
                    "drughiclseqno": float(rng.integers(1000, 40000)),
                    "dosage": "1 3",
                    "routeadmin": "IV" if is_abx else "PO",
                    "frequency": "Q8H SCH",
                    "loadingdose": np.nan,
                    "prn": "No",
                    "drugstopoffset": start + int(rng.integers(60, 1000)),
                    "gtc": 0,
                })
                # ~30% of orders also appear in medication_imputed, as an
                # imputed re-statement of the SAME clinical event (tests
                # the dedup path in combine_medication_sources) -- and a
                # further independent set only appears in
                # medication_imputed1 (tests the union/additional-signal
                # path), modeling the genuine uncertainty about how these
                # tables actually relate on the real dataset.
                if rng.random() < 0.30:
                    medimp_id += 1
                    medimp_rows.append({
                        "medicationid": medimp_id,
                        "patientunitstayid": stay_id,
                        "drugorderoffset": start,
                        "drugstartoffset": start,
                        "drugivadmixture": "No",
                        "drugordercancelled": "No",
                        "drugname": med_rows[-1]["drugname"],
                        "drughiclseqno": med_rows[-1]["drughiclseqno"],
                        "dosage": "1 3",
                        "routeadmin": med_rows[-1]["routeadmin"],
                        "frequency": "Q8H SCH",
                        "loadingdose": np.nan,
                        "prn": "No",
                        "drugstopoffset": med_rows[-1]["drugstopoffset"],
                        "gtc": 0,
                    })
                if rng.random() < 0.10:
                    medimp1_id += 1
                    imp1_is_abx = rng.random() < 0.25
                    imp1_start = int(rng.integers(0, 1500))
                    medimp1_rows.append({
                        "medicationid": medimp1_id,
                        "patientunitstayid": stay_id,
                        "drugorderoffset": imp1_start,
                        "drugstartoffset": imp1_start,
                        "drugivadmixture": "No",
                        "drugordercancelled": "No",
                        "drugname": rng.choice(ANTIBIOTICS) if imp1_is_abx else rng.choice(NON_ANTIBIOTICS),
                        "drughiclseqno": float(rng.integers(1000, 40000)),
                        "dosage": "1 3",
                        "routeadmin": "IV" if imp1_is_abx else "PO",
                        "frequency": "Q8H SCH",
                        "loadingdose": np.nan,  # medication_imputed1 -- loadingdose fully null, matching real sample
                        "prn": "No",
                        "drugstopoffset": imp1_start + int(rng.integers(60, 1000)),
                        "gtc": 0,
                    })

            # Admission drugs: 0-3
            for _ in range(int(rng.integers(0, 4))):
                admdrug_id += 1
                admdrug_rows.append({
                    "admissiondrugid": admdrug_id,
                    "patientunitstayid": stay_id,
                    "drugoffset": int(rng.integers(-500, 100)),
                    "drugenteredoffset": int(rng.integers(-500, 100)),
                    "drugnotetype": "Admission",
                    "specialtytype": "eCM Primary",
                    "usertype": "THC Nurse",
                    "rxincluded": bool(rng.random() < 0.5),
                    "writtenineicu": True,
                    "drugname": rng.choice(NON_ANTIBIOTICS),
                    "drugdosage": 0.0,
                    "drugunit": "",
                    "drugadmitfrequency": "",
                    "drughiclseqno": int(rng.integers(1000, 40000)),
                })

            # Infusion drugs: vasopressor exposure for a subset (correlated with expired)
            n_inf = int(rng.integers(0, 6)) + (2 if expired and rng.random() < 0.6 else 0)
            for _ in range(n_inf):
                inf_id += 1
                is_vaso = rng.random() < (0.4 if expired else 0.15)
                inf_rows.append({
                    "infusiondrugid": inf_id,
                    "patientunitstayid": stay_id,
                    "infusionoffset": int(rng.integers(0, 2000)),
                    "drugname": rng.choice(VASOPRESSORS) if is_vaso else rng.choice(NON_VASOPRESSORS),
                    "drugrate": str(round(float(rng.random() * 20), 2)),
                    "infusionrate": round(float(rng.random() * 40), 1),
                    "drugamount": round(float(rng.choice([4.0, 20.0, 250.0])), 1),
                    "volumeoffluid": round(float(rng.choice([100.0, 250.0, np.nan])), 1) if rng.random() < 0.7 else np.nan,
                    "patientweight": round(float(rng.normal(80, 15)), 1) if rng.random() < 0.3 else np.nan,
                })

            # MicroLab: real eICU has only ~2,923 unique patients out of
            # 200,859 with ANY microLab entry (~1.5% coverage) --
            # 16,996 rows total across the whole database. The previous
            # version of this generator gave every patient an
            # independent rng.integers(0,3) roll, which meant ~67% of
            # mock patients got at least one entry -- ~45x denser than
            # real eICU. That mismatch is exactly why an earlier mock
            # run showed a strict-sepsis-cohort rate of ~33% while the
            # user's real-data run showed ~0.05% (2 patients out of
            # 4064) -- not a bug, just an unrepresentative mock. Using
            # 15% coverage here as a deliberate compromise: still ~10x
            # denser than real eICU (so small debug/subset mock cohorts
            # reliably contain at least a few strict-cohort-eligible
            # patients, keeping that code path testable), but close
            # enough to convey that microLab is genuinely sparse, not a
            # routine per-patient table like lab or medication.
            if rng.random() < 0.15:
                for _ in range(int(rng.integers(1, 3))):
                    micro_id += 1
                    micro_rows.append({
                        "microlabid": micro_id,
                        "patientunitstayid": stay_id,
                        "culturetakenoffset": int(rng.integers(-1000, 500)),
                        "culturesite": rng.choice(["Blood, Venipuncture", "Urine", "Sputum", "Blood, Central Line"]),
                        "organism": rng.choice(["no growth", "gram negative rods", "gram positive cocci"]),
                        "antibiotic": np.nan,
                        "sensitivitylevel": np.nan,
                    })

            # Treatment: 1-4 per patient
            for _ in range(int(rng.integers(1, 5))):
                treat_id += 1
                treat_rows.append({
                    "treatmentid": treat_id,
                    "patientunitstayid": stay_id,
                    "treatmentoffset": int(rng.integers(0, 1500)),
                    "treatmentstring": rng.choice(TREATMENTS),
                    "activeupondischarge": bool(rng.random() < 0.4),
                })

    pd.DataFrame(patient_rows).to_csv(out / "patient.csv", index=False)
    pd.DataFrame(apr_rows).to_csv(out / "apachePatientResult.csv", index=False)
    pd.DataFrame(apv_rows).to_csv(out / "apachePredVar.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(out / "diagnosis.csv", index=False)
    pd.DataFrame(lab_rows).to_csv(out / "lab.csv", index=False)
    pd.DataFrame(med_rows).to_csv(out / "medication.csv", index=False)
    pd.DataFrame(medimp_rows).to_csv(out / "medication_imputed.csv", index=False)
    pd.DataFrame(medimp1_rows).to_csv(out / "medication_imputed1.csv", index=False)
    pd.DataFrame(admdrug_rows).to_csv(out / "admissionDrug.csv", index=False)
    pd.DataFrame(inf_rows).to_csv(out / "infusionDrug.csv", index=False)
    pd.DataFrame(micro_rows).to_csv(out / "microLab.csv", index=False)
    pd.DataFrame(treat_rows).to_csv(out / "treatment.csv", index=False)

    print(f"Mock eICU data written to {out}")
    print(f"  hospitals={n_hospitals}, patients={n_hospitals * n_patients_per_hospital}")
    for fname in sorted(out.glob("*.csv")):
        with open(fname) as fh:
            n_rows = sum(1 for _ in fh) - 1
        print(f"  {fname.name}: {n_rows} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output directory for mock CSVs")
    parser.add_argument("--n-hospitals", type=int, default=3)
    parser.add_argument("--n-patients-per-hospital", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.out, args.n_hospitals, args.n_patients_per_hospital, args.seed)
