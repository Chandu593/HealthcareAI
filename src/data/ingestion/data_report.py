"""
Generate per-table data-quality reports (Section 7 deliverable).

For tables flagged `large_table=True` in the registry, this reads in
chunks and accumulates statistics incrementally -- it never calls
pd.read_csv() without a chunksize on a large table, per Section 3.

Output: one JSON file per table under outputs/data_reports/, plus a
combined summary written by the caller (run_pipeline.py).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.data.ingestion.schema_registry import TABLE_REGISTRY

logger = logging.getLogger("eicu_fl")

DEFAULT_CHUNKSIZE = 200_000


@dataclass
class TableReport:
    table_name: str
    path: str
    row_count: int
    column_count: int
    columns: list[str]
    required_columns_present: bool
    missing_pct_by_column: dict[str, float] = field(default_factory=dict)
    unique_patient_count: int | None = None
    unique_hospital_count: int | None = None
    estimated_memory_mb: float = 0.0
    read_strategy: str = "full"  # "full" | "chunked"
    leakage_risk_columns_present: list[str] = field(default_factory=list)


def _report_full_read(table_name: str, path: str) -> TableReport:
    df = pd.read_csv(path)
    schema = TABLE_REGISTRY[table_name]
    missing_pct = (df.isna().mean() * 100).round(3).to_dict()

    unique_patients = None
    if "patientunitstayid" in df.columns:
        unique_patients = int(df["patientunitstayid"].nunique())
    unique_hospitals = None
    if "hospitalid" in df.columns:
        unique_hospitals = int(df["hospitalid"].nunique())

    required_present = all(c in df.columns for c in schema.required_columns)
    leakage_cols = [c for c in schema.leakage_risk_columns if c in df.columns]

    return TableReport(
        table_name=table_name,
        path=path,
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        required_columns_present=required_present,
        missing_pct_by_column={k: float(v) for k, v in missing_pct.items()},
        unique_patient_count=unique_patients,
        unique_hospital_count=unique_hospitals,
        estimated_memory_mb=round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
        read_strategy="full",
        leakage_risk_columns_present=leakage_cols,
    )


def _report_chunked_read(
    table_name: str, path: str, chunksize: int = DEFAULT_CHUNKSIZE
) -> TableReport:
    schema = TABLE_REGISTRY[table_name]

    row_count = 0
    columns: list[str] = []
    na_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    patient_ids: set = set()
    hospital_ids: set = set()
    est_memory_mb = 0.0
    required_present = True

    reader = pd.read_csv(path, chunksize=chunksize, low_memory=False)
    for i, chunk in enumerate(reader):
        if i == 0:
            columns = list(chunk.columns)
            required_present = all(c in chunk.columns for c in schema.required_columns)
            na_counts = {c: 0 for c in columns}
            total_counts = {c: 0 for c in columns}

        row_count += len(chunk)
        est_memory_mb += chunk.memory_usage(deep=True).sum() / (1024 ** 2)

        for c in columns:
            na_counts[c] += int(chunk[c].isna().sum())
            total_counts[c] += len(chunk)

        if "patientunitstayid" in chunk.columns:
            # NOTE: for very large hospital-scale runs, tracking the full
            # unique-id set in memory is still far cheaper than holding
            # the raw rows, but a HyperLogLog-style estimator would be
            # preferable at true full-eICU scale. Documented as a known
            # scaling limitation rather than silently approximated.
            patient_ids.update(chunk["patientunitstayid"].unique().tolist())
        if "hospitalid" in chunk.columns:
            hospital_ids.update(chunk["hospitalid"].unique().tolist())

    missing_pct = {
        c: round(100.0 * na_counts[c] / total_counts[c], 3) if total_counts[c] else 0.0
        for c in columns
    }
    leakage_cols = [c for c in schema.leakage_risk_columns if c in columns]

    return TableReport(
        table_name=table_name,
        path=path,
        row_count=row_count,
        column_count=len(columns),
        columns=columns,
        required_columns_present=required_present,
        missing_pct_by_column=missing_pct,
        unique_patient_count=len(patient_ids) if patient_ids else None,
        unique_hospital_count=len(hospital_ids) if hospital_ids else None,
        estimated_memory_mb=round(est_memory_mb, 2),
        read_strategy="chunked",
        leakage_risk_columns_present=leakage_cols,
    )


def generate_table_report(table_name: str, path: str, chunksize: int = DEFAULT_CHUNKSIZE) -> TableReport:
    schema = TABLE_REGISTRY.get(table_name)
    if schema is None:
        raise ValueError(f"No schema registered for table '{table_name}'.")

    if schema.large_table:
        logger.info(f"[DATA_REPORT] {table_name}: chunked read (large_table=True, chunksize={chunksize})")
        return _report_chunked_read(table_name, path, chunksize=chunksize)

    logger.info(f"[DATA_REPORT] {table_name}: full read")
    return _report_full_read(table_name, path)


def write_report(report: TableReport, output_dir: str) -> str:
    out_path = Path(output_dir) / f"{report.table_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    return str(out_path)
