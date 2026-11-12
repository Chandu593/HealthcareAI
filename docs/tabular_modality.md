# Tabular Modality (Phase 5)

**Status: implemented and verified at real full scale.** Feature
engineering and per-hospital normalization (`src/data/tabular/`) are
fully tested (40 unit tests in `test_tabular_feature_engineering.py` +
5 in `test_tabular_normalization.py`). The autoencoder
(`src/modalities/tabular/`) has been genuinely executed and verified
on the user's own machine at real full scale (199,108 patients, 208
hospitals, 2026-08-13) — produced real `(199108, 64)` embeddings; it
has since also been genuinely executed in a sandbox with torch
installed (2026-08-23), not just written/skip-marked as before.

## Source tables and why

`patient`, `apachePredVar`, `diagnosis` — declared in
`configs/base.yaml`'s `modalities.tabular.source_tables`. These three
tables characterize a patient largely at a single point in time
(admission demographics, day-1 severity/comorbidity flags, diagnosis
codes), in contrast to Phase 4's time-series sources which have
genuinely repeated, timestamped observations. `hospital` is also read
(small, one row per hospital) — not for per-patient features directly,
but merged in via `hospitalid` for two hospital-level characterization
fields; see "Categorical encoding" below.

## Leakage prevention: allowlist, not blocklist

Every feature this phase produces is explicitly named in
`feature_engineering.py`'s `PATIENT_NUMERIC_FEATURES` /
`PATIENT_CATEGORICAL_FEATURES` / `APACHEPREDVAR_NUMERIC_FEATURES` /
`APACHEPREDVAR_CATEGORICAL_FEATURES` lists — an allowlist, not "exclude
known-bad columns, use everything else." A blocklist would silently
pull in any new column added to `patient`/`apachePredVar` later without
anyone deciding it belongs; an allowlist means every feature has a
positive, documented reason for being there (Section 9's explicit
requirement).

**Excluded, with reasons:**

| Excluded | Why |
|---|---|
| `hospitaldischargestatus`, `unitdischargestatus` | The labels themselves |
| `hospitaldischargelocation`, `unitdischargelocation` | "Death" is a literal possible value — post-outcome |
| `dischargeweight` | Measured at discharge, unavailable at prediction time |
| `hospitaldischargeoffset`, `unitdischargeoffset` | Length-of-stay — a downstream consequence of severity/outcome, not an admission-time predictor |
| `apachePredVar.diedinhospital` | The mortality outcome itself, present directly in this table. Was not even registered in `schema_registry.py` before this phase was built — nothing was flagging it. Now registered as a `leakage_risk_column`. |
| `apachePredVar.dischargelocation` | Same discharge-outcome concern |
| `pao2`, `fio2`, `day1pao2`, `day1fio2`, `creatinine`, `verbal`, `motor`, `eyes`, `day1verbal`, `day1motor`, `day1eyes`, `ventday1` | **Consumed to construct the sepsis label** in Phase 3 (`sofa_like_scoring.py`). This is a subtler leakage than the above: it's not that these values reveal the future, it's that the sepsis label was mechanically computed FROM them via a threshold rule (`sofa_like_total >= 2`). Including them as input features would let the model partially reconstruct that rule rather than learn a genuinely predictive pattern. Since Phase 10's multi-task heads will share ONE fused embedding for both mortality and sepsis (no task-specific feature subsets designed yet), the conservative choice is to exclude them from the shared representation entirely — at some cost to the mortality task's potential signal from admission physiology. **Revisit if/when task-specific feature masking exists.** |

**Verified in code, not just documentation:** both the unit tests
(`test_tabular_feature_engineering.py`) and the integration test
(`test_phase5_pipeline.py`) explicitly assert these terms never appear
in the assembled feature matrix's column names — a real check against
the actual pipeline output, not just a promise in this doc.

## Categorical encoding: fixed vocabulary, full cohort / all shards

Each categorical feature (gender, ethnicity, admission source, etc.)
is one-hot encoded against a **fixed vocabulary** determined once, up
front. Using a fixed, externally-supplied vocabulary (rather than each
hospital deriving its own) is what keeps every hospital's one-hot
columns identical and alignable into one tensor — if each hospital
picked its own top-K, hospital A's and hospital B's feature matrices
could have entirely different columns.

**FIXED 2026-08-23 (vocabulary-sampling-bias bug):** this vocabulary
used to be determined from a *bounded sample* of the first 10
hospitals by sorted hospital ID
(`modalities.tabular.vocabulary_sample_hospitals`, default 10) — the
same pattern Phase 4's `select_lab_features()` still uses. Because
`hospital_partition.py` builds its partitions dict via
`sorted(available_hospitals)`, `[:10]` always picked the 10
**smallest**-ID hospitals, not a representative sample. On the real
199,108-patient / 208-hospital run this collapsed several vocabularies
below their real diversity — confirmed genuinely fixed for
`hospitaladmitsource`, `apacheadmissiondx`, and `diagnosis`, which all
now show real diversity beyond the old `top_k=10` ceiling.

**A second, independent bug (FIXED 2026-08-24):** `apachePredVar.teachtype`
and `apachePredVar.region` — this project's ORIGINAL source for
hospital-characterization features — kept showing vocab=1 even after
the fix above, because it wasn't (solely) the sampling bug. Direct
inspection of the real, raw `apachePredVar.csv` (bypassing this
pipeline entirely) showed `teachtype=0` and `region=3` for **literally
all 171,177 rows** — genuinely constant, not a sampling artifact. No
vocabulary-determination strategy, however correct, could have
surfaced diversity that isn't in the source data.

The real per-hospital diversity instead lives in `hospital.csv`'s
`region`/`teachingstatus` columns (confirmed: 189 non-teaching / 19
teaching, 4 real regions + some missing, across the real 208
hospitals) — exactly the kind of "site metadata... useful for
characterizing non-IID FL clients" `schema_registry.py` already
described that table as, before this bug was ever found. Fixed by
`feature_engineering.attach_hospital_features()`, which merges these
two columns onto `cohort_patients` via `hospitalid` (as
`hospital_region`/`hospital_teachingstatus`, now in
`PATIENT_CATEGORICAL_FEATURES`) — `apachePredVar.teachtype`/`region`
were removed from `APACHEPREDVAR_CATEGORICAL_FEATURES` entirely, since
a constant column contributes zero information as a feature.

The fix treats the different source tables differently, because they
have different memory characteristics:

- **`patient` / `apachePredVar` / `hospital`** vocabularies now come
  from the **full cohort**
  (`feature_engineering.determine_full_cohort_vocabularies()`).
  This is safe re: the project's never-pool-raw-data-across-hospitals
  rule specifically because all three tables are already loaded once
  for the whole cohort upstream in Phase 5 — none of them were ever
  hospital-sharded in the first place (all small: `hospital` is ~208
  rows total, `patient`/`apachePredVar` up to ~200K rows, one row per
  patient). Using the full data that's already in memory introduces no
  NEW cross-hospital pooling; it just stops throwing away all but 10
  hospitals' worth of it.
- **`diagnosis`** is genuinely large and hospital-sharded (too big to
  load once for the whole cohort like the two tables above), so its
  vocabulary is instead built by **aggregating running value-counts
  hospital-by-hospital across every shard**
  (`feature_engineering.accumulate_value_counts()` +
  `determine_diagnosis_vocabulary_from_shards()`) — never holding more
  than one hospital's raw diagnosis rows in memory at once, preserving
  the project's hospital-sharding architecture for the one Phase 5
  source table that actually needs it.
- Phase 4's `select_lab_features()` (`time_binning.py`) still uses a
  genuinely **bounded, seeded-random** sample (fixed from
  sorted-first-10 to `random.Random(cfg.get("run.seed", 42)).sample(...)`,
  same seed convention as `partition_by_hospital`) for its own,
  different purpose: detecting which labname substrings exist at all
  in this cohort's `lab` data. `lab` is tens of millions of rows —
  genuinely too large to load in full just to check for substring
  matches, so a bounded sample stays necessary there; making it
  *random* rather than *sorted-first-N* removes the systematic bias
  without reintroducing the OOM risk hospital-sharding was built to
  avoid.

`modalities.tabular.vocabulary_sample_hospitals` has been **removed**
from `configs/base.yaml` (no longer used by anything). A new
`modalities.tabular.top_k_categorical_overrides` config lets specific
high-cardinality columns use a larger `top_k` than the shared default
— seeded with `apacheadmissiondx: 30`, `diagnosis: 20`,
`hospitaladmitsource: 15`, informed by which columns hit the old
`top_k=10` ceiling on the real run above. **These are a starting
point, not confirmed true category counts** from eICU's published data
dictionary — worth revisiting if/when those figures are available.

Diagnosis strings (`diagnosisstring`, e.g. `"cardiovascular|chest pain
/ ASHD|CAD"`) are reduced to their **top-level category** (the segment
before the first `|`) and encoded the same way, **time-windowed** to
`[-diagnosis_lookback_minutes, +diagnosis_cutoff_minutes]` around ICU
admission (default: the same 24h window as SOFA scoring) — a diagnosis
entered on day 5 of a long stay reflects clinical knowledge gained
after the observation window this project predicts from, and would
leak that later knowledge back into an "admission-time" feature. Note
this time-windowing applies only to the per-patient presence
*features*, not to vocabulary determination — the vocabulary itself
intentionally looks at every recorded diagnosis for a hospital
(regardless of offset) to decide which categories are common enough to
deserve their own column; a category should get one whether or not
this particular window is what surfaced it for a given patient.

## Per-hospital local normalization

Numeric columns are standardized per hospital, exactly like Phase 4's
time-series normalization and for the same reason (statistical
validity + the federated privacy boundary — see
`docs/data_pipeline_methodology.md` Section 7 for the full rationale,
which applies identically here). One-hot/categorical columns are
passed through unscaled — standardizing a 0/1 indicator would destroy
its interpretability for no benefit.

## Architecture options (added 2026-09-06, by user request)

`modalities.tabular.encoder` selects which architecture Phase 5 uses
-- this config key already existed as an unused stub (already
commented `# autoencoder | mlp`, but nothing in the code ever read it)
before this change; it now actually does something.

| `encoder` value | Output file | Notes |
|---|---|---|
| `autoencoder` (default) | `tabular_embeddings.npz` | **Unchanged code path.** `TabularAutoencoder` itself was never modified by this addition. |
| `mlp` | `tabular_embeddings_mlp.npz` | Plain feedforward projection (`input_dim -> hidden_dim -> embedding_dim`), no decoder, no reconstruction objective -- for users who want to skip the unsupervised reconstruction-based approach entirely. Returns a single embedding tensor, not the `(embedding, reconstruction)` tuple `TabularAutoencoder` returns. |

**The canonical `tabular_embeddings.npz` is never touched by `mlp`**
-- proven the same way as Phase 4's LSTM guarantee above: a dedicated
integration test confirms the file's bytes AND mtime are unchanged
after running `mlp`, with its own hand-performed revert→fail→restore→
pass proof.

Same "additive, standalone output" scope as Phase 4's new encoders:
Phase 11/12/13's actual training code never reads Phase 5's
`embeddings_path` at all, so selecting `mlp` here does not change what
those phases train with.

## What this does NOT yet do

- **No training loop.** Same caveat as the LSTM: the autoencoder's
  weights are randomly initialized. This phase verifies the
  architecture and forward-pass mechanics, not a learned representation.
- **`admissionDrug` and `treatment` are not consumed.** Both are
  registered, confirmed-available tables, but no feature engineering
  reads them yet — candidate future additions.
- **`top_k_categorical_overrides`' defaults are a starting estimate,
  not confirmed cardinalities** (see above) — a category still falls
  into `_other` once a column's real cardinality exceeds its `top_k`,
  same mechanism as before the fix, just with the vocabulary itself no
  longer artificially restricted to 10 hospitals' worth of data first.
- **`apachePredVar.teachtype`/`region` are still read from the raw
  table columns (only excluded from the FEATURE allowlist, not deleted
  from the source) — worth periodically re-checking whether a future
  eICU export or a different cohort selection populates them
  meaningfully, in case the constant-value finding above turns out to
  be specific to this particular extract rather than the whole
  eICU-CRD.**

## Verification status

| Component | Status |
|---|---|
| `feature_engineering.py` (age parsing, sentinel cleaning, one-hot encoding, diagnosis windowing, leakage exclusion, full-cohort/all-shard vocabulary determination, hospital.csv attachment) | Verified — 40 unit tests (`test_tabular_feature_engineering.py`), including direct reproductions of both the sampling-bias collapse and the hospital.csv-vs-apachePredVar source bug, and hand-performed revert→fail→restore→pass proofs for both |
| `normalization.py` (per-hospital isolation, categorical pass-through, missing-value imputation) | Verified — 5 unit tests |
| Full Phase 1→5 pipeline on mock data, including the leakage assertion at the pipeline-output level | Verified — 2 integration tests (`test_phase5_pipeline.py`) |
| Full Phase 1→5 pipeline, vocabulary-sampling-bias + hospital-source regression specifically | Verified — 4 integration tests (`test_phase5_vocabulary_bias_regression.py`), including two "premise" tests confirming the mock fixture still reproduces the bug's shape, and hand-performed revert→fail→restore→pass proofs at the full-pipeline level for both fixes |
| `autoencoder.py` (forward pass mechanics) | Verified — 2 of 3 unit tests in `test_tabular_autoencoder.py` genuinely execute (forward-pass shape + eval-mode determinism) in a torch-enabled sandbox as of 2026-08-23; the 3rd is correctly `unittest.skip`-marked (it only applies to the no-torch fallback path). More importantly, verified on the user's own machine at real full scale (199,108 patients, 2026-08-13), producing real `(199108, 64)` embeddings. |
| `mlp_encoder.py` (added 2026-09-06) | Verified — 9 unit tests (shape, single-patient, empty-input, NaN/Inf, gradient flow, single-tensor-not-tuple contract) + shared integration coverage in `test_modality_architecture_options.py` (correct output filename, canonical-file-untouched guarantee with its own revert→fail→restore→pass proof, matching `embedding_dim` for direct comparability with the autoencoder). Full existing suite re-run and passing throughout. **Not yet run against real full-scale data.** |

**CONFIRMED on the real 199,108-patient / 208-hospital dataset
(2026-08-24):** the vocabulary-sampling-bias fix was verified end-to-end
against real full-scale data before the hospital-source bug above was
found (`hospitaladmitsource=15, apacheadmissiondx=30, diagnosis=17` —
all showing real diversity beyond the old cap). The hospital-source fix
(`teachtype`/`region` → `hospital_region`/`hospital_teachingstatus`)
is sandbox-verified only so far — **still needs one more real full-scale
run to confirm `hospital_region`/`hospital_teachingstatus` show real
diversity (not vocab=1) against the actual 208-hospital dataset.**
