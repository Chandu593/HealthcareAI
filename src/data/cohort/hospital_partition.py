"""
Phase 2 -- Hospital-wise partitioning.

Splits the patient population by `hospitalid` (the federated client key)
and optionally subsamples patients per hospital for debug/subset runs.

Fixes a real bug from the user's original prototype bucketing code:
`groupby("hospitalid").apply(lambda x: x.sample(n=N_PATIENTS, ...))`
raises if any hospital has fewer than N_PATIENTS rows. Here we sample
min(N, available) per hospital instead, and report which hospitals were
undersized rather than crashing or silently guaranteeing every hospital
has exactly N patients.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.cohort.mortality_label import mortality_label


@dataclass
class HospitalPartition:
    hospital_id: int
    patient_count: int
    eligible_patient_count: int
    sampled_patient_count: int
    patient_ids: list[int]
    mortality_missing_count: int
    mortality_positive_count: int
    mortality_negative_count: int
    was_undersampled: bool  # True if patient_subset was requested but hospital had fewer patients


@dataclass
class PartitionResult:
    mode: str  # "subset" | "all"
    requested_hospitals: list[int]
    eligible_hospitals: list[int]
    excluded_hospitals: list[int]  # requested but not found in patient/hospital intersection
    partitions: dict[int, HospitalPartition] = field(default_factory=dict)

    def total_patients(self) -> int:
        return sum(p.sampled_patient_count for p in self.partitions.values())


def partition_by_hospital(
    patient_df: pd.DataFrame,
    hospital_df: pd.DataFrame,
    mode: str = "subset",
    requested_hospitals: list[int] | None = None,
    patient_subset: int | None = None,
    seed: int = 42,
) -> PartitionResult:
    if "hospitalid" not in patient_df.columns:
        raise ValueError("patient_df missing required column 'hospitalid'")
    if "hospitalid" not in hospital_df.columns:
        raise ValueError("hospital_df missing required column 'hospitalid'")

    if mode not in ("all", "subset"):
        raise ValueError(f"Unknown federated.mode: {mode!r} (expected 'subset' or 'all')")

    available_hospitals = set(patient_df["hospitalid"].unique()) & set(hospital_df["hospitalid"].unique())

    if mode == "all" or not requested_hospitals:
        eligible_hospitals = sorted(available_hospitals)
        excluded_hospitals: list[int] = []
        requested_hospitals = eligible_hospitals
    else:  # mode == "subset" and requested_hospitals is non-empty
        eligible_hospitals = sorted(set(requested_hospitals) & available_hospitals)
        excluded_hospitals = sorted(set(requested_hospitals) - available_hospitals)

    partitions: dict[int, HospitalPartition] = {}
    for hid in eligible_hospitals:
        hosp_patients = patient_df[patient_df["hospitalid"] == hid].copy()
        mortality = mortality_label(hosp_patients)
        eligible_mask = mortality.notna()

        eligible_df = hosp_patients[eligible_mask]
        patient_count = len(hosp_patients)
        eligible_count = len(eligible_df)

        was_undersampled = False
        if patient_subset is not None and eligible_count > 0:
            n = min(patient_subset, eligible_count)
            was_undersampled = eligible_count < patient_subset
            sampled_df = eligible_df.sample(n=n, random_state=seed)
        else:
            sampled_df = eligible_df

        sampled_mortality = mortality_label(sampled_df)

        partitions[hid] = HospitalPartition(
            hospital_id=hid,
            patient_count=patient_count,
            eligible_patient_count=eligible_count,
            sampled_patient_count=len(sampled_df),
            patient_ids=sampled_df["patientunitstayid"].tolist(),
            mortality_missing_count=int((~eligible_mask).sum()),
            mortality_positive_count=int((sampled_mortality == 1).sum()),
            mortality_negative_count=int((sampled_mortality == 0).sum()),
            was_undersampled=was_undersampled,
        )

    return PartitionResult(
        mode=mode,
        requested_hospitals=list(requested_hospitals),
        eligible_hospitals=eligible_hospitals,
        excluded_hospitals=excluded_hospitals,
        partitions=partitions,
    )
