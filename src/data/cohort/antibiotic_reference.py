"""
Curated keyword lists for identifying antibiotics and vasopressors from
free-text `drugname` fields.

WHY KEYWORD MATCHING (not drughiclseqno):
eICU's HICL sequence numbers are a real, more reliable identifier, but
this project's tables do not include a drug dictionary/formulary table
mapping HICL codes to drug classes -- only the raw `drughiclseqno`
integer sits in `medication`/`admissionDrug`, with no lookup table to
resolve it to "this is an antibiotic." Keyword matching against
`drugname` is therefore the only option available with the tables this
project actually has. This is a known precision/recall tradeoff:
- False negatives: brand names or formulations not in the list.
- False positives: a drug name containing a keyword for an unrelated
  reason (rare, but e.g. combination products).
If the user later obtains an eICU HICL formulary reference, this module
is the single place to swap in ID-based matching instead.

Lists are deliberately NOT exhaustive of every antibiotic/vasopressor in
existence -- they cover the broad-spectrum agents most relevant to
Sepsis-3 "suspected infection" (per Seymour et al. 2017's supplementary
antibiotic list and standard ICU practice) and the vasopressors used in
Sepsis-3 cardiovascular-SOFA scoring.
"""

from __future__ import annotations

import re

import pandas as pd

# Broad-spectrum / commonly used IV antibiotics in ICU sepsis workups.
ANTIBIOTIC_KEYWORDS: tuple[str, ...] = (
    "vancomycin", "piperacillin", "tazobactam", "zosyn", "meropenem",
    "imipenem", "ertapenem", "cefepime", "ceftriaxone", "cefotaxime",
    "ceftazidime", "cefazolin", "ampicillin", "sulbactam", "unasyn",
    "linezolid", "daptomycin", "levofloxacin", "ciprofloxacin",
    "moxifloxacin", "azithromycin", "metronidazole", "clindamycin",
    "gentamicin", "tobramycin", "amikacin", "aztreonam", "doripenem",
    "colistin", "polymyxin", "tigecycline", "nafcillin", "oxacillin",
    "penicillin", "amoxicillin", "trimethoprim", "sulfamethoxazole",
    "bactrim", "doxycycline", "minocycline",
)

# Vasopressors used for the (partial) cardiovascular SOFA-like component.
VASOPRESSOR_KEYWORDS: tuple[str, ...] = (
    "norepinephrine", "levophed", "epinephrine", "vasopressin",
    "dopamine", "phenylephrine", "neosynephrine", "angiotensin ii",
)

_antibiotic_pattern = re.compile("|".join(re.escape(k) for k in ANTIBIOTIC_KEYWORDS), re.IGNORECASE)
_vasopressor_pattern = re.compile("|".join(re.escape(k) for k in VASOPRESSOR_KEYWORDS), re.IGNORECASE)


def is_antibiotic(drugname: pd.Series) -> pd.Series:
    return drugname.fillna("").str.contains(_antibiotic_pattern)


def is_vasopressor(drugname: pd.Series) -> pd.Series:
    return drugname.fillna("").str.contains(_vasopressor_pattern)
