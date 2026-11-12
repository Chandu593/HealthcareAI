# Sepsis Label Methodology (Phase 3 — implemented, dual-cohort)

**Status: implemented** in `src/data/cohort/{infection_proxy,sofa_like_scoring,sepsis_label,cohort_report}.py`,
verified against synthetic data (41/41 tests pass; see IMPLEMENTATION_TRACKER.md).
Design locked 2026-08-08 per explicit user decision after reviewing this
document's first draft — see "Dual-cohort design" below for the
correction that changed the approach.

## Why this document exists

eICU has no `sepsis = 0/1` field. None of the project's reference papers
(checked: all 10 PDFs in the project) describe their Sepsis-3 extraction
algorithm in implementable detail — they cite Singer et al. 2016 and move
on. So this project's sepsis label is a **derived phenotype we must
define and justify ourselves**, not something copied from a source.

## Constraint that shapes everything below

This project's eICU export does **not** include `vitalPeriodic`,
`vitalAperiodic`, `nurseCharting`, `respiratoryCare`, or `intakeOutput`.
That rules out:
- A continuous, hour-by-hour SOFA trajectory (no repeated vitals).
- True qSOFA (needs respiratory rate + systolic BP, both vitals-only).
- Renal SOFA's urine-output component.
- Cardiovascular SOFA's MAP component.

What remains is a **single-timepoint, structured-data-only organ
dysfunction score**, built from `apachePredVar`, `lab`, and
`infusionDrug`, combined with an infection proxy built from `medication`
and `microLab`. This is a documented limitation of the label, not a
hidden one.

## Important correction (2026-08-08)

An earlier draft of this document (and an earlier chat response) framed
"antibiotic administration alone" as though it were a direct
implementation of Sepsis-3's "suspected infection" criterion. That
overclaimed. **None of this project's reference papers specify their
eICU extraction logic at implementation resolution** — the FRL paper,
for example, states only that patients were extracted "according to the
International Consensus Sepsis-3 criteria" and moves straight to
modeling. There is no published ground truth to match against, and no
basis for asserting that one specific antibiotic/culture rule *is* the
correct Sepsis-3 implementation on this dataset.

The correct response to that gap is not to pick one rule and present it
as authoritative — it's to build **both** plausible definitions,
label them for what they are (an approximation with a name, not "the"
definition), and report them side by side so the choice is visible and
auditable rather than buried.

## Dual-cohort design (implemented)

Two cohorts are always computed together — never just one:

| Cohort | Suspected-infection rule | Role |
|---|---|---|
| **Strict** | Antibiotic administration **+** a `microLab` culture order within `corroboration_window_minutes` (default 1440 = 24h) of the antibiotic start | **Primary analysis** — higher specificity |
| **Broad** | Antibiotic administration alone, bounded by `lookback_minutes`/`lookahead_minutes` relative to ICU admission | **Sensitivity analysis** — higher recall |

Both share the same organ-dysfunction score (below). By construction,
`sepsis_strict` is always a subset of `sepsis_broad` — enforced as a
tested invariant in `tests/unit/test_infection_proxy.py` and
`tests/integration/test_phase2_3_pipeline.py`, not just asserted here.

```
# src/data/cohort/infection_proxy.py
broad_infection  = antibiotic_administered(medication, curated keyword list)
strict_infection = broad_infection AND culture_within_window(microLab, corroboration_window_minutes)

# src/data/cohort/sofa_like_scoring.py
organ_dysfunction = sofa_like_score(apachePredVar, lab, infusionDrug) >= sofa_like_threshold   # default 2

# src/data/cohort/sepsis_label.py
sepsis_broad  = broad_infection  AND organ_dysfunction
sepsis_strict = strict_infection AND organ_dysfunction
```

`sofa_like_score` sums points across the 6 SOFA domains, of which 4 are
fully computable and 2 (renal, cardiovascular) are partial-credit:

| Domain | Source | Not available |
|---|---|---|
| Respiratory (PaO2/FiO2) | `apachePredVar.pao2`/`fio2` (or `day1pao2`/`day1fio2`) | — |
| Coagulation (platelets) | `lab` (labname containing "platelet") | — |
| Liver (bilirubin) | `lab` (labname containing "bilirubin") | — |
| Renal (creatinine) | `apachePredVar.creatinine` | urine output |
| CNS (GCS) | `apachePredVar.verbal+motor+eyes` (or `day1` variants) | — |
| Cardiovascular | vasopressor presence in `infusionDrug` (flat 2pt, no dose tiering) | MAP, dose-based 3/4 tiers |

Renal and cardiovascular are intentionally **partial-credit** domains
(config: `renal_urine_output_available: false`,
`cardiovascular_map_available: false`) — the scoring function does not
silently treat a missing urine-output/MAP component as "0 points = no
dysfunction." A patient's `max_possible_score` (computed per patient in
`sofa_like_scoring.py`, based on how many of the 6 domains actually had
data) is tracked alongside the raw `sofa_like_total` sum specifically so
that isn't hidden — see "Known gap" below for where this still needs to
be surfaced in the cohort report.

### Cohort report — what gets compared, per the user's explicit request

`src/data/cohort/cohort_report.py` (Phase 3e) produces, for both
cohorts side by side:

- number of ICU stays / eligible patients / excluded patients
- number of septic patients and sepsis prevalence (strict, broad)
- sepsis onset-offset distribution (mean/median/min/max, in minutes —
  labeled `*_onset_offset_approx` because it's the infection-proxy
  timestamp, not a true SOFA-trajectory onset; see "Explicitly rejected
  shortcuts" below)
- hospital-wise cohort sizes (patient count, mortality, sepsis-strict,
  sepsis-broad per hospital)
- mortality rate among septic vs. non-septic patients, per cohort
- overlap between the two cohorts (`both_strict_and_broad`,
  `broad_only`, `strict_only`, `neither`)
- `downstream_model_performance`: explicitly `"N/A"` until Phases
  11-14 (training/evaluation) exist — not fabricated

## Explicitly rejected shortcuts

- **Using `diagnosis.diagnosisstring` alone to define sepsis.** This
  would make the label circular with a feature source. Diagnosis
  strings are for cohort *validation* (sanity-checking the derived rate
  against clinician-coded rate), never for *defining* the label. Not
  yet wired into the cohort report as a validation check — candidate
  future addition, see IMPLEMENTATION_TRACKER.md.
- **Using `apachePatientResult.actualhospitalmortality` or
  `apachescore` in any part of the sepsis definition.** These are
  leakage-risk columns (see `schema_registry.py`) — outcome-adjacent by
  construction.
- **Treating "SOFA >= 2" as equivalent to Sepsis-3's "SOFA increase >= 2
  from baseline."** Without a trajectory there is no baseline to
  subtract. The config key is named `sofa_like_threshold`, not
  `sofa_delta_threshold`, and every output field derived from it keeps
  the `_like_`/`_approx` naming, specifically to keep this distinction
  visible wherever it's referenced downstream (feature tables, figures,
  a thesis write-up) rather than only living in this doc.

## Known gap (tracked, not silent)

`n_domains_observed` and `max_possible_score` are computed per patient
but not yet propagated into `cohort_report.py`'s output. Before treating
`sofa_like_total` as comparable across two patients in any analysis or
figure, check that both had a similar `max_possible_score` — a patient
scoring 3 out of a possible 22 is not equivalent to one scoring 3 out of
a possible 8 because half their domains were unrecorded.

## Real-data finding: the strict cohort is currently near-empty (2026-08-09)

Running against the actual eICU dataset (200,859 patients, 208
hospitals), a 4,064-patient subset run (20/hospital) produced
`sepsis_broad_positive=1331` but `sepsis_strict_positive=2`. This is a
direct, expected consequence of `microLab`'s extreme sparsity (2,923
patients with any culture order out of 200,859 total, ~1.5% dataset-wide)
combined with per-hospital random patient sampling that isn't stratified
toward microLab coverage — not a code bug. (A separate real code bug
this same run surfaced, unrelated to cohort size, is documented in
`IMPLEMENTATION_TRACKER.md`'s "Bug found and fixed via the user's
real-data execution" section.)

Before drawing any conclusion from the strict cohort at small-subset
scale, consider: increasing `cohort.patient_subset`, switching
`federated.mode` to `all`, or explicitly reporting the strict cohort's
near-emptiness as a methodology limitation rather than treating its
current numbers as representative.

