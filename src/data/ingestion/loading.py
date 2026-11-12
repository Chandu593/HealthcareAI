"""
Load a (possibly large) eICU CSV filtered down to a known set of
patientunitstayid values, using chunked reading so we never hold the
full unfiltered table in memory (Section 3: large-data engineering
requirement). Small tables (patient, hospital, apachePredVar) are cheap
enough to just pd.read_csv() directly and are not routed through here.
"""

from __future__ import annotations

import pandas as pd


def load_filtered(
    path: str, patient_ids: set[int], chunksize: int = 200_000, usecols: list[str] | None = None,
) -> pd.DataFrame:
    if not patient_ids:
        # Read header only to preserve column structure in the empty result.
        return pd.read_csv(path, nrows=0, usecols=usecols)

    chunks = []
    reader = pd.read_csv(path, chunksize=chunksize, low_memory=False, usecols=usecols)
    for chunk in reader:
        filtered = chunk[chunk["patientunitstayid"].isin(patient_ids)]
        if len(filtered):
            chunks.append(filtered)
    if not chunks:
        return pd.read_csv(path, nrows=0, usecols=usecols)
    return pd.concat(chunks, ignore_index=True)
