"""
Phase 3a -- Mortality label construction.

Kept as the single source of truth for the mortality mapping (Section 9:
"label-generation logic must be clearly separated from feature
generation") so hospital_partition.py and any later feature-building code
import this rather than each re-deriving the mapping.
"""

from __future__ import annotations

import pandas as pd

# hospitaldischargestatus is the target -- NOT unitdischargestatus, which
# can be "Alive" for a patient who is discharged from the ICU to the
# floor and then dies later in the same hospitalization. Using
# unitdischargestatus would silently mislabel those cases as survivors.
LABEL_SOURCE_COLUMN = "hospitaldischargestatus"
KNOWN_STATUS_MAP = {"Expired": 1, "Alive": 0}


def mortality_label(patient_df: pd.DataFrame, source_column: str = LABEL_SOURCE_COLUMN) -> pd.Series:
    """Map `source_column` -> {1: died, 0: survived, NaN: unknown/exclude}.

    Any status string outside KNOWN_STATUS_MAP becomes NaN (excluded)
    rather than silently defaulting to 0 -- eICU's discharge-status field
    occasionally has extra categories, and treating an unrecognized value
    as "survived" would be a silent labeling error, not a safe default.
    """
    if source_column not in patient_df.columns:
        raise ValueError(f"patient_df missing required column '{source_column}'")
    return patient_df[source_column].map(KNOWN_STATUS_MAP)
