"""
Validate discovered tables against the schema registry.

Reads only the header (nrows=0) for the required-column check, so this
is cheap even for the multi-GB tables (lab, medication, diagnosis,
infusionDrug, treatment) -- per Section 3, we must not load a full large
table just to check its columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.ingestion.schema_registry import TABLE_REGISTRY


@dataclass
class TableValidationResult:
    table_name: str
    path: str
    passed: bool
    missing_required_columns: list[str] = field(default_factory=list)
    missing_optional_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    error: str | None = None


def validate_table(table_name: str, path: str) -> TableValidationResult:
    schema = TABLE_REGISTRY.get(table_name)
    if schema is None:
        return TableValidationResult(
            table_name=table_name, path=path, passed=False,
            error=f"No schema registered for table '{table_name}'.",
        )

    try:
        header_df = pd.read_csv(path, nrows=0)
    except Exception as exc:  # noqa: BLE001 - surface any parse error to caller
        return TableValidationResult(
            table_name=table_name, path=path, passed=False,
            error=f"Failed to read header: {exc}",
        )

    columns = {c.strip() for c in header_df.columns}
    required = set(schema.required_columns)
    optional = set(schema.optional_columns)
    known = required | optional

    missing_required = sorted(required - columns)
    missing_optional = sorted(optional - columns)
    extra = sorted(columns - known)

    return TableValidationResult(
        table_name=table_name,
        path=path,
        passed=len(missing_required) == 0,
        missing_required_columns=missing_required,
        missing_optional_columns=missing_optional,
        extra_columns=extra,
    )


def validate_all(found_tables: dict[str, str]) -> dict[str, TableValidationResult]:
    return {name: validate_table(name, path) for name, path in found_tables.items()}
