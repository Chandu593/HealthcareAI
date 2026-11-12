"""Screening audit: how common are end-of-life / comfort-care strings in the first-24h treatment & diagnosis
records, and how strongly do they relate to in-hospital death? If they touch few patients they cannot drive the
model's AUROC; if they are common and tightly tied to death, run the model with them removed.

  python scripts/paper_eol_audit.py --data-root "%EICU_DATA_ROOT%" [--min-los 1440] [--window 1440]
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATTERN = r"palliat|comfort|hospice|dnr|do.?not.?resus|withdraw|end.?of.?life|terminal"


def audit(data_root, pattern=DEFAULT_PATTERN, window=1440.0, min_los=0.0, chunksize=1_000_000) -> dict:
    root = Path(data_root); rx = re.compile(pattern, re.I)
    pat = pd.read_csv(root / "patient.csv", usecols=["patientunitstayid", "unitdischargeoffset", "hospitaldischargestatus"])
    pat = pat[pat["hospitaldischargestatus"].isin(["Alive", "Expired"])]
    pat = pat[pd.to_numeric(pat["unitdischargeoffset"], errors="coerce") >= min_los]
    died = pat.set_index("patientunitstayid")["hospitaldischargestatus"].eq("Expired")
    wend = pd.Series(np.minimum(pd.to_numeric(pat["unitdischargeoffset"], errors="coerce").fillna(window).clip(lower=0), window).to_numpy(), index=pat["patientunitstayid"].to_numpy())
    flagged = set(); strings = Counter(); deaths_by_string = Counter()
    for table, col, off in (("treatment", "treatmentstring", "treatmentoffset"), ("diagnosis", "diagnosisstring", "diagnosisoffset")):
        path = root / f"{table}.csv"
        if not path.exists():
            continue
        for ch in pd.read_csv(path, usecols=["patientunitstayid", col, off], chunksize=chunksize, low_memory=False):
            ch = ch[ch["patientunitstayid"].isin(wend.index)]
            ch = ch[ch[col].astype(str).str.contains(rx, na=False)]
            if ch.empty:
                continue
            ch = ch[pd.to_numeric(ch[off], errors="coerce") <= ch["patientunitstayid"].map(wend)]
            for pid, txt in zip(ch["patientunitstayid"].to_numpy(), ch[col].astype(str).str.lower().to_numpy()):
                flagged.add(int(pid)); strings[f"{table}: {txt}"] += 1; deaths_by_string[f"{table}: {txt}"] += int(died.get(pid, False))
    fl = pd.Series(died.index.isin(list(flagged)), index=died.index)
    n, nf = len(died), int(fl.sum())
    return {"n_patients": n, "n_flagged": nf, "share_flagged": nf / max(n, 1), "mortality_flagged": float(died[fl].mean()) if nf else float("nan"),
            "mortality_unflagged": float(died[~fl].mean()) if n - nf else float("nan"), "share_of_all_deaths_flagged": float(died[fl].sum() / max(died.sum(), 1)),
            "top_strings": [(k, c, deaths_by_string[k]) for k, c in strings.most_common(15)]}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--data-root", required=True); ap.add_argument("--window", type=float, default=1440)
    ap.add_argument("--min-los", type=float, default=0); ap.add_argument("--pattern", default=DEFAULT_PATTERN)
    a = ap.parse_args(); r = audit(a.data_root, a.pattern, a.window, a.min_los)
    print(f"patients: {r['n_patients']}; with an end-of-life/comfort string inside the window: {r['n_flagged']} ({100 * r['share_flagged']:.2f}%)")
    print(f"in-hospital mortality: flagged {r['mortality_flagged']:.3f} vs unflagged {r['mortality_unflagged']:.3f}; flagged patients hold {100 * r['share_of_all_deaths_flagged']:.1f}% of all deaths")
    print("most common matching strings (count, deaths):")
    for k, c, d in r["top_strings"]:
        print(f"  {c:6d} {d:6d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
