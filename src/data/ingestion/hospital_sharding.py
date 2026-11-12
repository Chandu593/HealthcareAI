"""
Hospital-wise sharding for large eICU tables (Section 3: large-data
engineering requirement).

WHY THIS EXISTS
----------------
Phase 3 originally loaded each large table (lab: ~39M rows/9.7GB,
medication + medication_imputed + medication_imputed1 combined: ~16.3M
rows) filtered to "all sampled patients," then held the result as one
pooled in-memory DataFrame. In `debug`/`subset` mode that filter is
meaningful (a few thousand patients out of 200K+). In `full` mode it
barely filters anything -- nearly every patient is eligible -- so the
pooled approach doesn't scale. This is exactly what crashed on a real
199,108-patient full-mode run:

    numpy._core._exceptions._ArrayMemoryError: Unable to allocate
    123. MiB for an array with shape (16124697,)

inside `combine_medication_sources()`'s `drop_duplicates()`, on a
concatenation of all three medication tables for the ENTIRE dataset.

THE FIX
-------
Read each large source table ONCE, in chunks (never held fully in
memory), projecting only the columns actually needed downstream
(SHARD_COLUMNS -- e.g. `lab` only needs 4 of its 10 columns for SOFA
scoring), and route each chunk's rows into small per-hospital shard
files on disk (plain CSV -- see "Why CSV, not Parquet" below).
Downstream phases then loop hospital-by-hospital, loading only one
hospital's already-small shard at a time. Peak memory becomes bounded
by the LARGEST SINGLE HOSPITAL's data, not the whole dataset -- on the
real eICU hospital-size distribution the user's own log showed, the
largest hospital (hospitalid=73) has 7,059 patients vs. 199,108 total,
a ~28x reduction, before column projection is even counted.

This also happens to be a much closer match to the federated
architecture's own "Hospital-wise Data Separation" step than pooling
ever was.

Shards are CACHED (Section 26): re-running the pipeline with the same
source file (checked via mtime + size), the same column projection, and
the same cohort patient set reuses existing shards instead of re-
reading a 9.7GB CSV again. The cache key changes (forcing a re-shard)
if any of those three things change -- Section 26's explicit
"do not silently reuse incompatible caches" requirement.

WHY CSV, NOT PARQUET
--------------------
Parquet would very likely shrink shard files and speed up re-reads
further. It is NOT implemented here because this project has no way to
verify it: neither this development sandbox nor (as of this writing)
the user's machine has confirmed `pyarrow` installed, and a subtle
schema-inference bug in an untested Parquet writer path could silently
corrupt data on a 39M-row real file in a way that's far worse than the
original OOM crash (which at least failed loudly). CSV sharding is
fully tested here and is the safer choice until Parquet's real-data
behavior can actually be verified. Documented as a genuine follow-up
opportunity, not a silent gap -- see docs/memory_efficiency.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger("eicu_fl")


# Column projection: only what Phase 3 (SOFA scoring, infection proxy)
# and Phase 4 (time binning) actually consume. Reading fewer columns
# per chunk directly reduces peak memory independent of sharding itself
# -- e.g. `medication`'s 15 columns become 4.
SHARD_COLUMNS: dict[str, list[str]] = {
    "lab": ["patientunitstayid", "labresultoffset", "labname", "labresult"],
    "medication": ["patientunitstayid", "drugname", "drugstartoffset", "drugstopoffset"],
    "medication_imputed": ["patientunitstayid", "drugname", "drugstartoffset", "drugstopoffset"],
    "medication_imputed1": ["patientunitstayid", "drugname", "drugstartoffset", "drugstopoffset"],
    "infusionDrug": ["patientunitstayid", "infusionoffset", "drugname", "infusionrate"],
    "microLab": ["patientunitstayid", "culturetakenoffset"],
    "diagnosis": ["patientunitstayid", "diagnosisoffset", "diagnosisstring"],
}


@dataclass
class ShardManifest:
    table_name: str
    source_path: str
    hospital_shard_paths: dict = field(default_factory=dict)  # {hospital_id(str): path}
    row_counts: dict = field(default_factory=dict)            # {hospital_id(str): n_rows}
    cache_key: str = ""


def _cache_key(source_path: Path, columns: list[str], patient_ids: set[int]) -> str:
    """Deterministic key over (source file identity, columns, cohort).
    Using mtime+size rather than a full file hash -- hashing a 9.7GB
    file on every run would itself be a meaningful cost; mtime+size is
    the standard cheap proxy for "has this file changed," same
    trade-off pandas/make/etc. make."""
    h = hashlib.sha256()
    stat = source_path.stat()
    h.update(str(stat.st_mtime_ns).encode())
    h.update(str(stat.st_size).encode())
    h.update(",".join(sorted(columns)).encode())
    h.update(",".join(str(p) for p in sorted(patient_ids)).encode())
    return h.hexdigest()[:16]


def _manifest_path(cache_dir: Path, table_name: str) -> Path:
    return cache_dir / "hospital_shards" / f"{table_name}_manifest.json"


def _load_cached_manifest(cache_dir: str, table_name: str, expected_cache_key: str) -> ShardManifest | None:
    path = _manifest_path(Path(cache_dir), table_name)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("cache_key") != expected_cache_key:
        return None
    manifest = ShardManifest(**data)
    for p in manifest.hospital_shard_paths.values():
        if not Path(p).exists():
            return None  # cache/ was partially cleared by hand -- don't trust a partial manifest
    return manifest


def shard_table_by_hospital(
    source_path: str,
    table_name: str,
    patient_hospital_map: dict[int, int],
    cache_dir: str,
    chunksize: int = 200_000,
) -> ShardManifest:
    """Single chunked pass over source_path, routing rows into
    per-hospital CSV shard files. Reuses a cached manifest if the
    source file, columns, and patient cohort are unchanged."""
    columns = SHARD_COLUMNS.get(table_name)
    if columns is None:
        raise ValueError(f"No column projection registered for '{table_name}' in SHARD_COLUMNS")

    src = Path(source_path)
    patient_ids = set(patient_hospital_map.keys())
    key = _cache_key(src, columns, patient_ids)

    cached = _load_cached_manifest(cache_dir, table_name, key)
    if cached is not None:
        logger.info(f"[SHARD] {table_name}: reusing cached shards (cache_key={key})")
        return cached

    shard_dir = Path(cache_dir) / "hospital_shards" / table_name
    if shard_dir.exists():
        for f in shard_dir.glob("*"):
            f.unlink()
    shard_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[SHARD] {table_name}: sharding by hospital (chunksize={chunksize})")

    row_counts: dict[str, int] = {}
    hospital_shard_paths: dict[str, str] = {}
    # Keep one open file handle per hospital for the whole pass rather
    # than re-opening/appending per chunk -- with ~200 chunks x up to
    # 208 hospitals, repeated open+append+close would be a meaningful
    # I/O cost on its own, especially on Windows.
    open_handles: dict[int, object] = {}
    wrote_header: set[int] = set()

    try:
        reader = pd.read_csv(source_path, usecols=columns, chunksize=chunksize, low_memory=False)
        for chunk in reader:
            chunk = chunk[chunk["patientunitstayid"].isin(patient_ids)]
            if chunk.empty:
                continue
            chunk = chunk.copy()
            chunk["_hospitalid"] = chunk["patientunitstayid"].map(patient_hospital_map)

            for hid, group in chunk.groupby("_hospitalid"):
                hid_int = int(hid)
                group = group.drop(columns=["_hospitalid"])
                hid_str = str(hid_int)
                row_counts[hid_str] = row_counts.get(hid_str, 0) + len(group)

                if hid_int not in open_handles:
                    path = shard_dir / f"hospital_{hid_str}.csv"
                    hospital_shard_paths[hid_str] = str(path)
                    open_handles[hid_int] = open(path, "w", newline="", encoding="utf-8")

                handle = open_handles[hid_int]
                group.to_csv(handle, header=hid_int not in wrote_header, index=False)
                wrote_header.add(hid_int)
    finally:
        for handle in open_handles.values():
            handle.close()

    manifest = ShardManifest(
        table_name=table_name, source_path=str(src),
        hospital_shard_paths=hospital_shard_paths, row_counts=row_counts, cache_key=key,
    )
    manifest_path = _manifest_path(Path(cache_dir), table_name)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(asdict(manifest), f, indent=2)

    total_rows = sum(row_counts.values())
    logger.info(
        f"[SHARD] {table_name}: {total_rows:,} rows routed across {len(hospital_shard_paths)} hospitals"
    )
    return manifest


def load_hospital_shard(manifest: ShardManifest, hospital_id: int) -> pd.DataFrame:
    """Load one hospital's already-small shard. A hospital absent from
    the manifest had ZERO matching rows in the source table for this
    cohort (e.g. no medication orders at all for any sampled patient at
    that hospital) -- a valid, common state, not an error, so this
    returns an empty frame with the right columns rather than raising."""
    hid_str = str(int(hospital_id))
    path = manifest.hospital_shard_paths.get(hid_str)
    if path is None:
        return pd.DataFrame(columns=SHARD_COLUMNS[manifest.table_name])
    return pd.read_csv(path)
