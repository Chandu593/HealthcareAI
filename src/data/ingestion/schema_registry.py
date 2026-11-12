"""
Registry of eICU tables this project knows how to ingest.

IMPORTANT -- read before editing:

This registry was built from the actual table samples the user shared
(row counts, dtypes, sample rows), not from the full eICU documentation,
because only a subset of eICU tables is available in this project
(no vitalPeriodic, vitalAperiodic, nurseCharting, respiratoryCare,
intakeOutput, note, pastHistory, allergy, etc.). All 13 tables the user
has confirmed present (2026-08-08) are registered as required. One
provenance question remains open even though availability is resolved
-- flagged below rather than silently assumed:

  1. `treatment` -- CONFIRMED present (2026-08-08). Registered as
     required like the other 12.

  2. `medication_imputed` / `medication_imputed1` -- CONFIRMED present
     (2026-08-08, listed explicitly among the user's 13 tables).
     Registered as required. Their exact relationship to `medication`
     is still unknown: are they distinct imputation passes over the
     same underlying orders (which would mean unioning them for
     antibiotic detection double-counts the same clinical event), or
     genuinely independent order sets? Only `medication_imputed1`'s
     shape was shown in the original sample dump (1,674,261 rows,
     `loadingdose` entirely null -- clearly an imputed variant of
     something). Because the relationship is unconfirmed,
     `infection_proxy.py`'s `combine_medication_sources()` unions all
     three tables' antibiotic-flagged rows and deduplicates exact
     repeated (patient, drugname, start, stop) rows as a defensive
     heuristic -- see that function's docstring and
     `docs/sepsis_label_methodology.md` for the caveat this doesn't
     fully resolve.

Tables NOT registered because they were never shown as available:
vitalPeriodic, vitalAperiodic, nurseCharting, respiratoryCharting,
respiratoryCare, intakeOutput, note, pastHistory, allergy, carePlan*,
customLab, apacheApsVar. This is why the time-series modality (Phase 4)
cannot use "vital signs" as the architecture doc suggests, and why SOFA
/ qSOFA computation for sepsis labeling is necessarily an approximation
built from `apachePredVar` + `lab` rather than a continuous trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableSchema:
    name: str
    # Column names that MUST be present for the table to be usable.
    required_columns: tuple[str, ...]
    # Columns that are expected but tolerable if missing (schema drift).
    optional_columns: tuple[str, ...] = field(default_factory=tuple)
    # Patient-level join key, if this table has one.
    patient_key: str | None = "patientunitstayid"
    # Whether this table must be read in chunks / with column projection
    # rather than pd.read_csv() in one shot (Section 3: large-data rule).
    large_table: bool = False
    # Whether the table is confirmed available vs. flagged ambiguous.
    availability: str = "confirmed"  # "confirmed" | "unconfirmed"
    # Free-text note surfaced in the data report for humans reviewing it.
    note: str = ""
    # Flags fields that are derived from / correlated with the outcome
    # and therefore must be excluded from model INPUT features to avoid
    # leakage (Section 9). They remain valid for baseline comparison
    # (e.g. reporting APACHE-predicted mortality as a benchmark), just
    # not as a feature.
    leakage_risk_columns: tuple[str, ...] = field(default_factory=tuple)


TABLE_REGISTRY: dict[str, TableSchema] = {
    "patient": TableSchema(
        name="patient",
        required_columns=(
            "patientunitstayid", "patienthealthsystemstayid", "hospitalid",
            "unitdischargestatus", "hospitaldischargestatus",
        ),
        optional_columns=(
            "gender", "age", "ethnicity", "wardid", "apacheadmissiondx",
            "admissionheight", "admissionweight", "dischargeweight",
            "hospitaladmitoffset", "hospitaldischargeoffset",
            "unitdischargeoffset", "unitdischargelocation",
            "unitadmitsource", "unitstaytype", "unitvisitnumber",
            "uniquepid", "hospitaladmitsource", "hospitaldischargelocation",
        ),
        note=(
            "Cohort base table. unitdischargestatus/hospitaldischargestatus "
            "are the mortality label source. hospitalid is the federated "
            "client key. uniquepid/patienthealthsystemstayid identify the "
            "same real person across multiple stays -- needed to prevent "
            "patient-level leakage across train/test splits."
        ),
    ),
    "hospital": TableSchema(
        name="hospital",
        required_columns=("hospitalid",),
        optional_columns=("numbedscategory", "teachingstatus", "region"),
        patient_key=None,
        note="Site metadata. Useful for characterizing non-IID FL clients.",
    ),
    "apachePatientResult": TableSchema(
        name="apachePatientResult",
        required_columns=(
            "apachepatientresultsid", "patientunitstayid", "apacheversion",
            "actualicumortality", "actualhospitalmortality",
        ),
        optional_columns=(
            "physicianspeciality", "physicianinterventioncategory",
            "acutephysiologyscore", "apachescore", "predictedicumortality",
            "predictediculos", "actualiculos", "predictedhospitalmortality",
            "predictedhospitallos", "actualhospitallos", "preopmi",
            "preopcardiaccath", "ptcawithin24h", "unabridgedunitlos",
            "unabridgedhosplos", "actualventdays", "predventdays",
        ),
        note=(
            "Multiple rows per stay (one per APACHE version scored). Use "
            "for benchmarking APACHE's own predicted mortality against "
            "this project's model -- NOT as an input feature (see "
            "leakage_risk_columns)."
        ),
        leakage_risk_columns=(
            "predictedicumortality", "actualicumortality",
            "predictedhospitalmortality", "actualhospitalmortality",
            "apachescore", "acutephysiologyscore",
            "predictediculos", "actualiculos",
            "predictedhospitallos", "actualhospitallos",
        ),
    ),
    "apachePredVar": TableSchema(
        name="apachePredVar",
        required_columns=("apachepredvarid", "patientunitstayid"),
        optional_columns=(
            "age", "gender", "admitdiagnosis", "verbal", "motor", "eyes",
            "day1verbal", "day1motor", "day1eyes", "pao2", "fio2",
            "day1pao2", "day1fio2", "creatinine", "diabetes",
            "electivesurgery", "thrombolytics", "aids", "hepaticfailure",
            "lymphoma", "metastaticcancer", "leukemia",
            "immunosuppression", "cirrhosis", "ventday1", "dischargelocation",
            "diedinhospital", "admitsource", "teachtype", "region",
            "bedcount", "readmit", "visitnumber",
        ),
        note=(
            "Raw physiologic inputs behind the APACHE score -- NOT the "
            "same as apachePatientResult's derived score/outcome fields, "
            "so most of this table is safe to use as an input feature "
            "source (tabular modality, and the main source for SOFA-like "
            "organ dysfunction proxies: pao2/fio2, creatinine, GCS "
            "components). EXCEPTION: `diedinhospital` is the mortality "
            "outcome itself, present directly in this table -- flagged "
            "below, never usable as a feature. Added to the registry "
            "2026-08-11 when Phase 5 (tabular modality) was built -- it "
            "existed in the real table all along but had not been "
            "registered, so nothing was flagging it before now."
        ),
        leakage_risk_columns=("diedinhospital",),
    ),
    "diagnosis": TableSchema(
        name="diagnosis",
        required_columns=(
            "diagnosisid", "patientunitstayid", "diagnosisoffset",
            "diagnosisstring",
        ),
        optional_columns=("activeupondischarge", "icd9code", "diagnosispriority"),
        large_table=True,
        note=(
            "Time-stamped, hierarchical (pipe-delimited) diagnosis "
            "strings. Candidate source for tabular multi-hot features "
            "and for the optional diagnosis-string pseudo-text mode "
            "(see text modality decision in Phase 6)."
        ),
    ),
    "lab": TableSchema(
        name="lab",
        required_columns=(
            "labid", "patientunitstayid", "labresultoffset", "labname",
            "labresult",
        ),
        optional_columns=(
            "labtypeid", "labresulttext", "labmeasurenamesystem",
            "labmeasurenameinterface", "labresultrevisedoffset",
        ),
        large_table=True,
        note=(
            "~39.1M rows / ~2.9GB. Primary source for the time-series "
            "modality (no vitalPeriodic available) AND for SOFA-like "
            "components (platelets, bilirubin, creatinine). Must be "
            "read with chunking + patient/hospital filtering during "
            "ingestion, never in one pd.read_csv() call."
        ),
    ),
    "medication": TableSchema(
        name="medication",
        required_columns=(
            "medicationid", "patientunitstayid", "drugstartoffset",
            "drugstopoffset", "drugname",
        ),
        optional_columns=(
            "drugorderoffset", "drugivadmixture", "drugordercancelled",
            "drughiclseqno", "dosage", "routeadmin", "frequency",
            "loadingdose", "prn", "gtc",
        ),
        large_table=True,
        note="~7.3M rows / ~836MB. Antibiotic-flag source for the sepsis infection proxy.",
    ),
    "medication_imputed": TableSchema(
        name="medication_imputed",
        required_columns=(
            "medicationid", "patientunitstayid", "drugstartoffset",
            "drugstopoffset",
        ),
        optional_columns=(
            "drugorderoffset", "drugivadmixture", "drugordercancelled",
            "drugname", "drughiclseqno", "dosage", "routeadmin",
            "frequency", "loadingdose", "prn", "gtc",
        ),
        large_table=True,
        availability="confirmed",
        note=(
            "CONFIRMED present 2026-08-08 -- see module docstring point 2 "
            "for the still-open provenance question re: overlap with "
            "`medication`/`medication_imputed1`. Unioned (with dedup) via "
            "combine_medication_sources() in infection_proxy.py."
        ),
    ),
    "medication_imputed1": TableSchema(
        name="medication_imputed1",
        required_columns=(
            "medicationid", "patientunitstayid", "drugstartoffset",
            "drugstopoffset",
        ),
        optional_columns=(
            "drugorderoffset", "drugivadmixture", "drugordercancelled",
            "drugname", "drughiclseqno", "dosage", "routeadmin",
            "frequency", "loadingdose", "prn", "gtc",
        ),
        large_table=True,
        availability="confirmed",
        note=(
            "CORRECTED 2026-08-09: confirmed via the user's real Phase 1 "
            "run to be ~7.3M rows / ~3.27GB -- the SAME size as raw "
            "`medication`, not the smaller ~1.67M-row table as originally "
            "guessed from an ambiguous early sample dump (that smaller "
            "table is actually `medication_imputed`). Was previously "
            "marked large_table=False, which made Phase 1's data_report.py "
            "do a risky full (non-chunked) read of a 3.27GB file -- fixed "
            "here. This mislabeling never affected Phase 3/4's correctness "
            "or memory safety: hospital_sharding.py always chunks every "
            "table it shards regardless of this flag, which is why the "
            "full-mode Phase 3 run succeeded even before this fix."
        ),
    ),
    "admissionDrug": TableSchema(
        name="admissionDrug",
        required_columns=("admissiondrugid", "patientunitstayid", "drugname"),
        optional_columns=(
            "drugoffset", "drugenteredoffset", "drugnotetype",
            "specialtytype", "usertype", "rxincluded", "writtenineicu",
            "drugdosage", "drugunit", "drugadmitfrequency", "drughiclseqno",
        ),
        note="Home/admission medications. Lower priority; sparse free-text drug names.",
    ),
    "infusionDrug": TableSchema(
        name="infusionDrug",
        required_columns=("infusiondrugid", "patientunitstayid", "infusionoffset", "drugname"),
        optional_columns=("drugrate", "infusionrate", "drugamount", "volumeoffluid", "patientweight"),
        large_table=True,
        note=(
            "~4.8M rows / ~330MB. Continuous drip data -- primary source "
            "for the vasopressor-exposure component of the sepsis "
            "cardiovascular-dysfunction proxy (no MAP available)."
        ),
    ),
    "microLab": TableSchema(
        name="microLab",
        required_columns=("microlabid", "patientunitstayid", "culturetakenoffset", "culturesite", "organism"),
        optional_columns=("antibiotic", "sensitivitylevel"),
        note=(
            "Only ~17k rows total across all of eICU -- sparse. Use as "
            "corroborating evidence for suspected infection, not a "
            "required condition."
        ),
    ),
    "treatment": TableSchema(
        name="treatment",
        required_columns=("treatmentid", "patientunitstayid", "treatmentoffset", "treatmentstring"),
        optional_columns=("activeupondischarge",),
        large_table=True,
        availability="confirmed",
        note=(
            "Confirmed present by the user 2026-08-08. Hierarchical "
            "(pipe-delimited) treatment strings, same structure as "
            "diagnosis. Candidate corroborating signal for organ-support "
            "interventions (e.g. vasopressor/ventilation treatment "
            "entries) alongside infusionDrug/apachePredVar."
        ),
    ),
}


# Tables the architecture doc's "vital signs" / "physiological signals"
# language implies but that are NOT registered because they were never
# shown as available in this project's eICU export. Kept here (not just
# in the docstring) so the data-report generator can explicitly report
# them as "expected by the architecture but absent" rather than the
# absence being silently invisible.
KNOWN_ABSENT_TABLES: tuple[str, ...] = (
    "vitalPeriodic", "vitalAperiodic", "nurseCharting", "respiratoryCharting",
    "respiratoryCare", "intakeOutput", "note", "pastHistory", "allergy",
    "customLab", "apacheApsVar",
)


def required_tables() -> list[str]:
    return [t.name for t in TABLE_REGISTRY.values() if t.availability == "confirmed"]


def optional_tables() -> list[str]:
    return [t.name for t in TABLE_REGISTRY.values() if t.availability == "unconfirmed"]
