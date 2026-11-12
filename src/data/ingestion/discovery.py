"""
Discover eICU table files under a configured data root.

Per Section 3 (DO NOT hardcode paths) and Section 7 (automatic file
discovery), this module scans EICU_DATA_ROOT and matches files to the
TABLE_REGISTRY by name, tolerant of:
  - case differences (eICU exports vary: "patient.csv" vs "Patient.csv")
  - .csv / .csv.gz
  - an underscore/hyphen variant of the table name

It does NOT assume every registered table exists -- it reports found vs.
missing separately for confirmed vs. unconfirmed-availability tables, so
Phase 1 output directly resolves the ambiguities flagged in
schema_registry.py instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.data.ingestion.schema_registry import TABLE_REGISTRY, KNOWN_ABSENT_TABLES


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass
class DiscoveryResult:
    data_root: str
    found: dict[str, str]              # table_name -> resolved file path
    missing_confirmed: list[str]       # required-availability tables not found
    missing_unconfirmed: list[str]     # ambiguous-availability tables not found
    unrecognized_files: list[str]      # files present but not in the registry
    known_absent_present: list[str]    # e.g. vitalPeriodic actually found (surprise!)

    def is_usable(self) -> bool:
        """Minimum bar to proceed to Phase 2: patient + hospital present."""
        return "patient" in self.found and "hospital" in self.found

    def summary_lines(self) -> list[str]:
        lines = [f"Data root: {self.data_root}"]
        lines.append(f"Found ({len(self.found)}): {sorted(self.found.keys())}")
        if self.missing_confirmed:
            lines.append(f"MISSING required tables: {sorted(self.missing_confirmed)}")
        if self.missing_unconfirmed:
            lines.append(f"Not found (unconfirmed-availability, OK if truly absent): {sorted(self.missing_unconfirmed)}")
        if self.known_absent_present:
            lines.append(
                f"NOTE: found tables the project assumed absent: {sorted(self.known_absent_present)} "
                f"-- re-run cohort/label design, these unlock real vitals/SOFA if genuine."
            )
        if self.unrecognized_files:
            lines.append(f"Unrecognized files in data root (ignored): {self.unrecognized_files}")
        return lines


def discover_tables(data_root: str) -> DiscoveryResult:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"data_root does not exist: {data_root}\n"
            f"Set it via: export EICU_DATA_ROOT=/path/to/eicu"
        )

    candidate_files = [
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in (".csv", ".gz")
    ]

    normalized_to_path: dict[str, Path] = {}
    for p in candidate_files:
        stem = p.name
        if stem.lower().endswith(".csv.gz"):
            stem = stem[: -len(".csv.gz")]
        else:
            stem = p.stem
        normalized_to_path[_normalize(stem)] = p

    found: dict[str, str] = {}
    for table_name in TABLE_REGISTRY:
        key = _normalize(table_name)
        if key in normalized_to_path:
            found[table_name] = str(normalized_to_path.pop(key))

    known_absent_present: list[str] = []
    for table_name in KNOWN_ABSENT_TABLES:
        key = _normalize(table_name)
        if key in normalized_to_path:
            known_absent_present.append(table_name)
            found[table_name] = str(normalized_to_path.pop(key))

    missing_confirmed = [
        t.name for t in TABLE_REGISTRY.values()
        if t.availability == "confirmed" and t.name not in found
    ]
    missing_unconfirmed = [
        t.name for t in TABLE_REGISTRY.values()
        if t.availability == "unconfirmed" and t.name not in found
    ]
    unrecognized_files = [str(p) for p in normalized_to_path.values()]

    return DiscoveryResult(
        data_root=str(root),
        found=found,
        missing_confirmed=missing_confirmed,
        missing_unconfirmed=missing_unconfirmed,
        unrecognized_files=unrecognized_files,
        known_absent_present=known_absent_present,
    )
