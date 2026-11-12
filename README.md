# Federated Multimodal Graph Learning for Mortality and Sepsis Prediction on eICU

Research-grade, reproducible implementation of a federated learning
framework that predicts **in-hospital mortality** and **sepsis** from
multimodal eICU patient data, where each hospital is an independent
federated client and raw data never leaves the hospital.

**Research use only.** Retrospective research system. Not medical
advice. Not clinically validated. Not approved for diagnosis or
treatment. Predictions must not be interpreted as clinical
recommendations.

## Paper track (current)

The results in the paper come from the **paper track**, not from the phase pipeline described further down. Entry point `run_paper.py`; code in `src/paper/`; configs `configs/paper*.yaml`; documentation in
`docs/paper_track.md` (design, leakage rules, run table) and `docs/paper_track_v10.md`, `v14`, `v15`, `v16`, `v20`, `v21` (federated GBDT, FLICS comparison, FLICS-scale federations, sepsis 6 h rerun and label check, pooled-GBDT control, final comparison in FLICS's format);
the results text is `Fed-THOR_results_section.md` (abstract and discussion draft: `Fed-THOR_abstract_and_discussion.md`) and the tables, figures and their scripts are in `Fed-THOR_results_package/`.

Rules the paper track keeps (do not relax them): features come only from the first 24 hours of the stay; discharge time, length of stay and outcome columns are never inputs; APACHE IVa is a benchmark only; vocabularies, scaling, calibration and
thresholds are fitted on training or validation data only; normalisation is per hospital and nothing is pooled across hospitals in the federated arms; the split manifest `outputs/ablation/split_manifest.json` is permanent and must never be regenerated;
every selection (including the headline model) uses validation data only.

Typical commands (Windows, one line each; set `EICU_DATA_ROOT` first):

```
python run_paper.py --config configs/paper_debug.yaml
python run_paper.py --config configs/paper_fedgbdt.yaml --reuse-phase1to3 --run-id fedgbdt_strict_final
python run_paper.py --config configs/paper_flicsstyle.yaml --reuse-phase1to3 --run-id flicsstyle_top3 --set paper.cohort_filter.min_hospital_patients=3758
python run_paper.py --config configs/paper_flicsstyle.yaml --reuse-phase1to3 --run-id flicsstyle_top10 --set paper.cohort_filter.min_hospital_patients=2699
python run_paper.py --config configs/paper_sepsis_early.yaml --reuse-phase1to3 --tasks sepsis --feature-sets sepsis_clean --arms fed_gbdt --seeds 42,43,44 --set paper.window.min_icu_los_minutes=360 --set paper.fed_gbdt.max_rounds=1200 --run-id fedgbdt_sepsis6h_mr1200
```

Pooled run of the same GBDT code (default off; reuses the federated winner's hyperparameters; all six commands are in `docs/paper_track_v20.md`):

```
python run_paper.py --config configs/paper_fedgbdt.yaml --reuse-phase1to3 --arms gbdt_central --run-id gbdtcentral_strict --set paper.gbdt_central.reuse_params_from=outputs/paper/fedgbdt_strict_final
```

The other final runs (`final_*`, `fedgbdt_allstays_final`, `fedgbdt_sepsis24_final`) are described in `docs/paper_track.md` and `docs/paper_track_v10.md`. Use a distinct `--run-id` for every run and run one job at a time.

Rebuild the results package after any run:

```
python scripts/paper_collect_results.py
python Fed-THOR_results_package/reproduce/paired_subsets.py
python Fed-THOR_results_package/reproduce/paired_flicsscale.py
python Fed-THOR_results_package/reproduce/paired_sepsis6h.py
python Fed-THOR_results_package/reproduce/paired_central_gbdt.py
python Fed-THOR_results_package/reproduce/build_tables.py
python Fed-THOR_results_package/reproduce/figs.py
```

Submission copies (section 8 and the draft markers removed; the sources are never modified): `python scripts/make_submission_copies.py` writes `*_submission.md` next to each of the three text files. Re-run it after any change to them.

Label and external checks (read saved predictions plus the raw tables): `scripts/paper_label_variant_check.py` (what FLICS's ICU-discharge label does to our AUROC; add `--paired-bootstrap 2000` for a paired interval of the difference) and `scripts/paper_documented_sepsis_check.py` (proxy-trained scores against documented sepsis).

Tests: `python -m pytest -q`. The suite builds mock data and, in a sandbox run, wrote mock shard manifests into the repository's `cache/` folder; before running it on a machine that holds real caches, rename `cache` and `outputs` (for example `ren cache cache_real`) and restore them afterwards.

**Everything below this line describes the earlier phase pipeline (Phases 1-13) and the ablation framework; it is kept for reference and is not updated for the paper track.**

## Three-modality suite (added 2026-10-30)

Separate track for the GRU + autoencoder + TCD cross-attention experiments; additive, writes only under `<run.output_dir>/three_modality/`. Stage 1 is the
**incident sepsis-proxy label** (new event with onset in ICU hours 24-96 from time-resolved evidence, prevalent cases excluded, explicit censoring policy,
never overwrites a run):

```
python run_three_modality.py --config configs/three_modality.yaml --stage incident_label
python run_three_modality.py --config configs/three_modality.yaml --stage incident_label --inspect-cache
```

```
python run_three_modality.py --config configs/three_modality.yaml --stage tcd_features    # ONE unified Temporal Clinical Dynamics vector per patient (first 24 ICU h)
python run_three_modality.py --config configs/three_modality.yaml --stage tcd_encoder     # MLP encoder -> ONE embedding per patient (focal multitask; needs torch)
```

Definition, parameters, censoring policies and limitations: `docs/three_modality_incident_sepsis.md` (stage 1; primary policy `alive_discharge_as_negative`, run on the real data) and
`docs/three_modality_tcd.md` (stage 2; run on the real data) and `docs/three_modality_gru.md` (stage 3a: first-24-hour GRU inputs + pooled focal GRU encoder; sandbox-verified only).

```
python run_three_modality.py --config configs/three_modality.yaml --stage tabular_inputs # stage 3b: clean static feature set (no APACHE scores / labs), train-only, typed columns
python run_three_modality.py --config configs/three_modality.yaml --stage tabular_ae     # stage 3b: autoencoder (reconstruction + focal heads), pooled -> ONE 64-d embedding (docs/three_modality_tabular.md)
python run_three_modality.py --config configs/three_modality.yaml --stage gru_inputs     # fresh 24 x 1-hour value+mask sequences, train-only scaling, fixed length
python run_three_modality.py --config configs/three_modality.yaml --stage gru_encoder    # pooled focal multitask GRU -> ONE 128-d embedding per patient (needs torch)
```

### Three-modality: cross-attention and Non-FL hospital-wise Configs 1-4 (added 2026-11-06; sandbox-verified only until the real runs below are done)

```
python run_three_modality.py --config configs/three_modality.yaml --stage cross_attention --tcd-encoder-run-id <TCDE> --gru-encoder-run-id <GRUE> --tabular-ae-run-id <TABAE>
python run_three_modality.py --config configs/three_modality.yaml --stage nonfl_config1 --cross-attention-run-id <CA_RUN> --embedding-set all   # also nonfl_config2 / 3 / 4, nonfl_report
```
Audit, design and caveats: `docs/three_modality_audit.md`, `docs/three_modality_cross_attention.md`, `docs/three_modality_nonfl.md`. Federated downstream learning (Configs 1-4, personalisation) is described below and in `docs/three_modality_fl.md`.

**Post-run fixes (2026-11-08):** stabilised LightGBM focal objective, selectable checkpoint metric, validation-only tuning (`--stage nonfl_tune`, apply with `--hparams-run-id`), saved test predictions + pooled
metrics with intervals, recall-target operating point and per-class classification reports for every config / task. See `docs/three_modality_nonfl.md` and `docs/three_modality_targets.md`.

**Federated downstream learning on the frozen embeddings (added 2026-11-10; sandbox-verified only until the real runs are done).** Stages `fl_tune`, `fl_config1..4`, `fl_personalized`, `fl_report`. The encoders and the
cross-attention are CENTRALLY pretrained, so this is *not* end-to-end federated learning. FedAvg / FedProx / FedAdam / SCAFFOLD apply to LR, MLP and GCN; trees use federated-histogram GBDT (`--classifiers gbdt_fed`) and
are never parameter-averaged. Outputs are additive under `outputs/three_modality/{federated,fl_tuning,reports}/`; Non-FL runs and the cross-attention run are only read. Full design, commands and caveats: `docs/three_modality_fl.md`.

## Current status

See `IMPLEMENTATION_TRACKER.md` for the authoritative phase-by-phase
status. As of this commit:

- **Phase 0 (Analysis)** — complete.
- **Phases 1-4 (ingestion, hospital partitioning, cohort/label
  construction, time-series modality) — fully implemented and verified
  end-to-end against the real full eICU dataset**: 200,859 patients,
  208 hospitals, `full.yaml` (no per-hospital cap), zero errors,
  ~26 minutes total (2026-08-10). This is real confirmation at true
  scale, not just synthetic-data testing in this sandbox.
- Getting there took three rounds of real bugs surfaced and fixed by
  that real-data execution — a dtype-corruption crash, an
  out-of-memory crash pooling medication tables in `full` mode
  (fixed via hospital-wise sharding), and an unbatched-LSTM OOM crash
  (fixed via adaptive batching with backoff) — see
  `IMPLEMENTATION_TRACKER.md` for the full writeups, and
  `docs/memory_efficiency.md` / `docs/timeseries_modality.md` for the
  technical detail.
- The LSTM's batch size (default 512) held constant for all 199,108
  patients in the confirmed run — zero backoff events needed, 54
  seconds total inference time — producing real `(199108, 128)`
  embeddings.
- **132/132 tests pass** in this sandbox (4 intentionally skipped — the
  ones needing `torch`, which this sandbox lacks; not silently passed).
  The LSTM forward pass those 2 tests would cover has, in practice,
  been verified for real on the user's machine instead.
- Sepsis labeling uses a **dual-cohort design** (strict primary /
  broad sensitivity) — see `docs/sepsis_label_methodology.md` for the
  full justification, plus a real-data finding: the strict cohort is
  genuinely small (0.1% at full scale, 262/199,108) given `microLab`'s
  real ~1.5% dataset-wide coverage.
- Text modality confirmed **disabled by default** — see
  `docs/text_modality_decision.md`.
- All 13 of the user's eICU tables are now registered as confirmed
  (including `treatment`, `medication_imputed`, `medication_imputed1`
  — see `schema_registry.py` and `combine_medication_sources()`). A
  registry mislabeling (`medication_imputed1` marked as small when it's
  actually ~7.3M rows) was also caught and fixed by the real full-mode run.
- Phase 3/4 process **hospital-by-hospital** rather than pooling the
  whole cohort's raw data in memory, and log progress every 20
  hospitals (an early run went silent for 9-16 minutes at a time
  without this) — see `docs/memory_efficiency.md`.
- **Phase 5 (tabular modality)** — feature engineering and per-hospital
  normalization implemented and tested (`patient` + `apachePredVar` +
  `diagnosis`, explicit leakage-safe allowlist — see
  `docs/tabular_modality.md`). Building this phase surfaced a real gap:
  `apachePredVar.diedinhospital` (the mortality outcome itself) had
  never been registered as a leakage risk in `schema_registry.py` —
  fixed. The autoencoder is written but unverified in this sandbox (no
  torch), same status the LSTM had before real-data confirmation.
- **Phases 6-19** — Phase 6 (text) confirmed off by design. Phases
  7-13 (multimodal fusion, intra-hospital patient graph, GNN,
  multi-task heads, local training, FedAvg, and local/centralized/
  federated baselines) are all implemented and real-full-scale-verified
  since this section was first written -- this "Current status" block
  predates that work; **`IMPLEMENTATION_TRACKER.md` is the accurate,
  currently-maintained source**, this file is not. Phases 14-19
  (evaluation, ablation studies beyond the framework below,
  explainability, visualization, caching, final reproducibility) remain
  not started, except for the ablation-and-classifier framework
  described next, which was added 2026-09-10 ahead of Phase 15's own
  slot in the tracker's numbering (see that section's own note on the
  Phase 13/14 boundary tension).

## Ablation & Classifier Framework (added 2026-09-10)

A research-grade ablation study layered on top of Phases 1-13, running
**before** the federated learning stage. See
`IMPLEMENTATION_TRACKER.md`'s "Ablation & Classifier Framework"
section for the full design writeup, the decisions confirmed with the
user, and current verification status (mock-data/sandbox-verified only
so far -- not yet run at real full scale).

**What it does:** a permanent, patient-level, leakage-safe 70/15/15
train/val/test split (within each hospital); a 72-configuration grid
(4 time-series encoders x 3 tabular encoders x 2 fusion mechanisms x 3
classifiers) comparing embedding quality *before* the graph/GNN stage;
a separate 6-experiment graph-embedding ablation using the existing
GNN's output; and structured, never-overwritten results with
publication-ready summary tables.

```
eICU Raw Data
      |
Phase 1-3 (ingestion, hospital partitioning, cohort/label construction) -- UNCHANGED
      |
Permanent split manifest (train/val/test, within each hospital, leakage-checked)
      |
Phase 4-5 (raw TS/TAB tensors) -- UNCHANGED
      |
   +--+------------------------------------------+
   |  For each of 4 TS x 3 TAB x 2 Fusion combos: |
   |    per-hospital flat model, TRAIN-split only  |
   |    -> frozen embeddings for all patients      |
   |    -> AdaBoost / XGBoost / LightGBM           |
   +--+------------------------------------------+
      |
Phase 8-9 (intra-hospital graph + GNN, retrained train-split-only) -> frozen embeddings -> AdaBoost / XGBoost / LightGBM
      |
Phase 11-13 (existing federated local/FedAvg/baselines) -- UNCHANGED, runs AFTER the above
```

**Run the whole ablation suite in one command:**

```bash
python run_ablation.py --config configs/subset.yaml
```

This runs Phase 1-5 fresh, builds the split manifest, then both
ablation stages, then writes summary tables under
`outputs/ablation/<run_id>/`.

**Useful variants:**

```bash
# Re-run just the ablation stages against already-completed Phase 1-5
# output (fast iteration -- requires `python run_pipeline.py --config
# configs/subset.yaml --phase 5` to have already completed):
python run_ablation.py --config configs/subset.yaml --reuse-phase1to5 \
    --steps embedding_ablation,graph_ablation

# Just the split manifest (idempotent):
python run_ablation.py --config configs/subset.yaml --steps split

# A fast smoke test bounded to the first 3 hospitals:
python run_ablation.py --config configs/subset.yaml --max-hospitals 3
```

**Config-driven, no code changes needed** to run a different grid --
edit `configs/base.yaml`'s `ablation:` stanza (`ts_models`,
`tab_models`, `fusion_modes`, `classifiers`, `graph_classifiers`), or
set `ablation.subset` to run an explicit smaller slice.

**Outputs**, under `outputs/ablation/<run_id>/`:
- `records/<experiment_id>.json` -- one file per experiment, full
  config/split/metrics/training-info/runtime detail.
- `master_results.csv` -- every experiment, one row each, appended live.
- `summary_table.csv` -- publication-ready comparison, sorted by test
  AUROC descending.

New modules: `src/data/splitting/patient_split.py`,
`src/modalities/tabular/pca_encoder.py`,
`src/fusion/cross_attention_fusion.py`,
`src/classifiers/boosting_classifiers.py`,
`src/training/flat_multimodal_model.py`, `src/ablation/*`. All new,
additive files -- nothing under Phases 1-13's existing, locked
architecture was modified except two small, unrelated pre-existing bug
fixes documented in the tracker.

## Split-Aware Pipeline (added 2026-09-11)

Extends the ablation framework above so Phase 11/12/13 (and the
ablation framework's own Phase 4/5 preprocessing) are genuinely
leakage-safe against the permanent train/val/test split -- normalization,
vocabulary, and every trainable component are fit from TRAIN patients
only; validation/test patients only ever pass through frozen weights.
See `IMPLEMENTATION_TRACKER.md`'s "Split-Aware Leakage Audit + Phase
11/12/13" section for the full design writeup, including a real
mid-flight design correction (a train-only GNN SUBGRAPH, not just
label-masking) caught by a dedicated leakage-sensitivity test.

Same entry point, extended:
```bash
python run_ablation.py --config configs/subset.yaml
```
now also builds a split-aware `outputs/split_aware/` (Phase 4/5/7/8)
and `checkpoints/phase1{1,2,3}_*_splitaware/` (Phase 11/12/13) tree
alongside the ablation results, and writes a comprehensive
`validation_report.{json,md}` -- exact patient counts per partition,
per-hospital mortality/sepsis prevalence, a zero-overlap confirmation,
confirmation that preprocessing fitting used training data only, Phase
11/12/13 evaluation metrics (AUROC/AUPRC/sensitivity/specificity/
calibration, threshold selected from validation only), and any
failures. The original, full-cohort Phase 4-13 output is never touched
by any of this -- every split-aware artifact lands in its own,
separately-named location.

## Architecture

```
eICU Raw Data (your local machine, via EICU_DATA_ROOT)
      |
Hospital-wise Data Separation (hospitalid = federated client)
      |
Local Hospital Preprocessing
      |
   +--+------------------+
   |  Time-Series   Tabular      [Text: disabled by default -- no clinical notes in eICU]
   |  (LSTM)        (Autoencoder)
   +--+------------------+
      |
Multimodal Fusion (concatenation)
      |
Intra-Hospital Patient Graph (k-NN / cosine similarity, no cross-hospital edges)
      |
Graph Neural Network (GraphSAGE / GAT / GCN)
      |
   +--+--------+
   Mortality   Sepsis Head
   Head        (multi-task loss)
      |
Local Model Update -> FedAvg -> Global Model -> back to hospitals
```

Full design detail: `docs/` and the original architecture source
(`Federated_Multimodal_Graph_Learning_Framework.pdf` in the project
files).

## Repository layout

```
configs/            base.yaml + debug/subset/full mode overrides
data/                README only -- no data is stored here
docs/                Methodology decisions requiring explicit justification
src/
  utils/             config loading, logging, batching (adaptive batch-size + OOM backoff, torch-agnostic)
  data/
    ingestion/       schema_registry, file discovery, data reports, hospital_sharding (memory-efficient large-table processing)
    validation/      schema validation
    cohort/          Phase 2/3: hospital partitioning, mortality + dual-cohort sepsis labels, SOFA-like scoring
    temporal/        Phase 4: time binning, sequence tensor construction, per-hospital normalization
    tabular/         Phase 5: feature engineering (leakage-safe allowlist), per-hospital normalization
  modalities/
    timeseries/      Phase 4: LSTM encoder (torch -- verified at real full scale, see docs/timeseries_modality.md)
    tabular/         Phase 5: autoencoder (torch -- unverified in this sandbox, see docs/tabular_modality.md)
scripts/             generate_mock_eicu.py (synthetic test fixtures)
tests/
  unit/
  integration/
outputs/             data_reports/, cohort_reports/, embeddings/, logs/, (later: figures/, metrics/, experiments/)
run_pipeline.py      CLI entry point
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or use environment.yml with conda
pip install -r requirements.txt
```

Point the pipeline at your local eICU CSVs (see `data/README.md` for
full detail, including Windows instructions):

```bash
export EICU_DATA_ROOT=C:\Users\Saira\Downloads\FedTKG-DHGNN\EICU
```

## Running Phases 1-5

```bash
python run_pipeline.py --config configs/debug.yaml --phase 5
```
## Start from certain phase --start-phase phase_number eg(--start-phase 5)

```bash
python run_pipeline.py --config configs/full.yaml --phase 5 --start-phase 5
```

## Ablation

# Stage A first (if you haven't already) -- builds Phase 1-3 on disk
python run_ablation.py --config configs/full.yaml --steps split,federated_split_aware --run-id full_run

# Then each batch, same run-id so results accumulate together:
python run_ablation.py --config configs/full_batch_lstm.yaml --steps embedding_ablation --run-id full_run --reuse-phase1to3
python run_ablation.py --config configs/full_batch_rnn.yaml --steps embedding_ablation --run-id full_run --reuse-phase1to3
python run_ablation.py --config configs/full_batch_gru.yaml --steps embedding_ablation --run-id full_run --reuse-phase1to3
python run_ablation.py --config configs/full_batch_transformer.yaml --steps embedding_ablation --run-id full_run --reuse-phase1to3

# Graph ablation doesn't need batching (fixed 6-experiment grid):
python run_ablation.py --config configs/full.yaml --steps graph_ablation --run-id full_run --reuse-phase1to3

This runs discovery/validation, hospital partitioning, mortality +
dual-cohort sepsis labeling, time-series preprocessing (binning →
sequence tensor → per-hospital normalization), and tabular feature
engineering (`patient`+`apachePredVar`+`diagnosis` → leakage-safe
feature matrix → per-hospital normalization). If `torch` is installed,
it also runs the LSTM and autoencoder forward passes and saves patient
embeddings; if not, those steps are skipped with an explicit log
message and the underlying feature tensors (complete, usable artifacts
on their own) are still saved. Outputs land in `outputs/data_reports/`,
`outputs/cohort_reports/`, and `outputs/embeddings/`.

**"Running a phase alone" isn't meaningful past Phase 2** — each phase
depends on the previous ones' outputs (partitioned cohorts, labels,
shard manifests), so `--phase N` always runs phases `1..N` in
sequence, not phase `N` in isolation. Pass `--phase 1`, `--phase 2`,
`--phase 3`, or `--phase 4` to stop earlier.

## Verification

Because the real eICU CSVs are not available in this execution
environment (they're on the user's local machine, per the project's
own data-location constraint), Phases 1-4 have been verified as far as
this sandbox allows:

1. **132/132 tests pass, 4 intentionally skipped** (`python -m unittest
   discover -s tests -v`), using `scripts/generate_mock_eicu.py` to
   generate a schema-compatible synthetic dataset — never real patient
   data. Tests include structural invariants that must hold regardless
   of random seed: no cross-hospital patient leakage, deterministic
   sampling, the strict-sepsis-cohort-is-a-subset-of-broad invariant,
   eICU's `-1` missing-value sentinel being excluded from scoring
   rather than treated as a real physiological value, per-hospital
   normalization staying isolated (one hospital's statistics never
   leak into another's), and structural padding always ending up
   exactly zero.
2. **A live CLI run** (`python run_pipeline.py --config
   configs/debug.yaml --phase 4`) against synthetic data, confirmed to:
   correctly discover all 13 confirmed tables, correctly use chunked
   reading for tables flagged `large_table=True`, correctly flag APACHE
   outcome-derived columns as leakage risks, produce an internally
   consistent cohort report, and produce a sequence tensor with no
   leftover NaNs and correctly zeroed padding.
3. **The 2 skipped tests in this sandbox are the LSTM encoder's forward
   pass mechanics** (exact shape, padding invariance) — this sandbox
   has no `torch` installed and no internet access to install it. The
   forward pass ITSELF has, in practice, been verified for real — see
   item 7 below. These 2 tests remain useful for finer-grained checks
   the full pipeline run doesn't isolate on their own; they're marked
   `unittest.skip` with an explicit reason here, not silently passed.
4. **The user ran Phases 1-4 against the real eICU dataset on their own
   machine (2026-08-09)** and hit a real crash in `build_cohort_report`
   — a dtype-corruption bug that the mock data's unrealistic
   completeness (every mock patient had 1:1 `apachePredVar` coverage;
   real eICU doesn't) had let slip through 75 passing tests. Fixed, with
   two new regression tests that were verified to actually catch the
   bug (reverted the fix, confirmed the tests fail with the exact same
   error the user hit, restored the fix). The mock generator was also
   made more realistic (~85% `apachePredVar` coverage, ~15% `microLab`
   coverage, matching real eICU's known sparsity far more closely) so
   this class of gap is less likely to recur silently. Full writeup in
   `IMPLEMENTATION_TRACKER.md`'s "Bug found and fixed via the user's
   real-data execution" section.
5. **After that fix, the user successfully ran `debug.yaml` end-to-end
   against the real dataset** — real 200,859-patient discovery/
   validation, a real 4,064-patient cohort through Phase 4, and a real
   LSTM embedding tensor (torch was available on their machine, the
   first genuine execution of that component anywhere in this
   project). Then a `full.yaml` run (all 199,108 eligible patients, no
   per-hospital cap) hit a **second** real crash — an out-of-memory
   error pooling all medication tables for the entire dataset before
   deduplicating. Fixed via hospital-wise sharding (peak memory now
   bounded by the largest single hospital, not the whole dataset) —
   see `docs/memory_efficiency.md`, verified with 10 new tests
   including an uneven-hospital-size scenario.
6. **The next `full.yaml` re-run got further — Phase 3 completed
   successfully on real data (~861s, all 199,108 patients)** — real
   confirmation the hospital-sharding fix holds at true scale. Phase 4
   then hit a **third** real crash: the LSTM forward pass had never
   been batched, so all 199,108 patients went through in one call,
   overflowing the CPU allocator. Fixed via a generic, torch-agnostic
   adaptive-batch-size utility with automatic OOM backoff
   (`src/utils/batching.py`, 14 unit tests). Also caught while
   auditing: `medication_imputed1` was mislabeled as a small table in
   the schema registry (it's actually ~7.3M rows/3.27GB, confirmed by
   this run), causing a risky non-chunked Phase 1 read — fixed. Also
   added progress logging to Phase 3/4's per-hospital loops, which had
   gone silent for 9-16 minutes at a time with zero output.
7. **The next `full.yaml` re-run (2026-08-10) completed end-to-end with
   zero errors** — all 4 phases, 199,108 patients, 208 hospitals,
   ~26 minutes total. The LSTM's batch size held constant at the 512
   default for the entire cohort (zero backoff events needed), 54
   seconds total inference time, producing real `(199108, 128)`
   embeddings. Sepsis counts matched the earlier crashed attempt
   exactly, confirming determinism held across the sharding/batching
   rework.

**Full-scale (`full.yaml`, all ~199K patients) execution against the
real eICU dataset has been confirmed successful end-to-end** (2026-08-10)
— see item 7 above. Both `debug.yaml` (subset scale) and `full.yaml`
(full scale) have now run clean against real data through Phase 4,
including the LSTM.

## Known data-availability constraints (read before Phase 2+)

This eICU export does not include `vitalPeriodic`, `vitalAperiodic`,
`nurseCharting`, `respiratoryCare`, or `intakeOutput`. This has two
direct consequences documented in full under `docs/`:

- **Time-series modality** (Phase 4) cannot use "vital signs" as the
  architecture doc's language suggests. It uses `lab` +
  `medication`/`infusionDrug` time series instead — see
  `configs/base.yaml`'s `modalities.timeseries.source_tables` and
  `docs/timeseries_modality.md` for the full pipeline and known
  limitations (including the LSTM's unverified status in this sandbox).
- **Sepsis label** (Phase 3) cannot use a continuous SOFA trajectory or
  true qSOFA. It uses a single-timepoint SOFA-like approximation — full
  justification in `docs/sepsis_label_methodology.md`.

## License / disclaimer

Research code accompanying a graduate-level project. No warranty. See
"Clinical data safety" note above.
