"""
eICU commonly encodes "not recorded" as -1 (seen in apachePredVar:
verbal/motor/eyes, pao2/fio2, creatinine, electivesurgery, etc.) rather
than a true NaN. Silently computing on -1 as if it were a real
physiological value would corrupt SOFA-like scoring (e.g. a -1 "GCS
component" would look like severe dysfunction when it actually means
"unknown"). This module centralizes that cleanup so every consumer
handles it the same way instead of each script reinventing it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SENTINELS = (-1, -1.0)


def clean_sentinels(series: pd.Series, sentinels=DEFAULT_SENTINELS) -> pd.Series:
    """Replace eICU's -1 'not recorded' sentinel with NaN. Does not touch
    genuine NaNs already present, and does not touch values that are
    negative for a legitimate clinical reason (e.g. offsets, which are
    handled separately and are NOT passed through this function)."""
    out = series.copy()
    mask = out.isin(sentinels)
    out[mask] = np.nan
    return out


def clean_columns(df: pd.DataFrame, columns: list[str], sentinels=DEFAULT_SENTINELS) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = clean_sentinels(out[col], sentinels=sentinels)
    return out
