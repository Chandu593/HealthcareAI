# Three-modality suite, stage 2: Temporal Clinical Dynamics (TCD) features + MLP encoder

**Status (2026-10-31): implemented, sandbox-verified (synthetic data) AND run on the real eICU export** (features run `tcdf_20261031T162203Z_3bc446e5`; first encoder run `tcde_20261031T163020Z_e5fdd41e` trained on ALL stays). Since that run, landmark-only encoder training became the default and ventilator-support settings are flagged as overlap (sections below); a landmark-only encoder run is the next one to make. Real results: see "Real-data results".
No existing file of the legacy / paper / incident-label code was modified; everything is written under `<run.output_dir>/three_modality/tcd/`.

```
python run_three_modality.py --config configs/three_modality.yaml --stage tcd_features            # ONE unified feature vector per patient + availability report + cutoff audit
python run_three_modality.py --config configs/three_modality.yaml --stage incident_label          # (stage 1; the encoder needs a completed run with the SAME censoring policy)
python run_three_modality.py --config configs/three_modality.yaml --stage tcd_encoder             # MLP encoder -> ONE embedding per patient (needs torch)
python run_three_modality.py --config configs/three_modality.yaml --stage tcd_features --inspect-cache      # lists completed runs, 'matches-current=YES/no'; computes nothing
python run_three_modality.py --config configs/three_modality.yaml --stage tcd_features --force-recompute    # NEW run directory; old runs are never overwritten
python run_three_modality.py ... --stage tcd_encoder --incident-run-id <id> --feature-run-id <id>           # pin specific completed upstream runs
```
Cache locations: `outputs/three_modality/tcd/features/<tcdf_...>/` and `outputs/three_modality/tcd/encoder/<tcde_...>/`; logs in `outputs/three_modality/logs/`.

## Window and the temporal cutoff

Inputs = lab rows with `0 <= offset <= min(ICU exit, 1440 min)` (`tcd.window_minutes`, which MUST equal `incident_sepsis.cutoff_hours*60`: a mismatch is rejected before
anything runs, so the inputs end exactly where the incident-event window begins). Minute 1440 is inside, minute 1441 is outside. Pre-ICU (negative-offset) rows are not used.
ICU exit only clips the window; it is never a feature. Enforced three ways:
1. the reader only returns rows inside each patient's window and counts what it dropped (after-window, before-ICU) and the maximum offset used;
2. `compute_tcd_feature_matrix` clips again itself (defence in depth);
3. the stage runs a **poison audit** on real data (`tcd.audit.n_patients`, default 2000): it re-reads random patients WITHOUT the window (up to 96 h), multiplies every value after the
   window by 1000 and adds 12345, and requires the features to be bit-identical with vs without that corruption, identical to the reader-enforced path, and equal to the main run
   (by feature name). The audit also requires that corrupted rows actually existed (otherwise it proves nothing). A failed audit aborts the run. Results ship in `cutoff_audit.json`.

## What the data allows (13-table export)

The only time-stamped VALUE series are in `lab`. There is no `vitalPeriodic`/`nurseCharting`, so **shock index, pulse pressure and MAP cannot be computed here**: they are declared,
listed under `unavailable` with the reason in the schema and the report, and never imputed. If a vitals table exists (`tcd.vitals.table`, default `vitalPeriodic`), the same machinery uses it
(hourly-mean binned, invasive before non-invasive channels, MAP measured-else-derived); this path is tested on synthetic data only and is inactive for your export.
Medication, infusion and treatment RECORDS (antibiotics, vasopressors, ventilation orders) are not TCD inputs; TCD is built from the `lab` value series. Note that the eICU `lab` table also carries ventilator / oxygen-support SETTINGS (`fio2`, and among the data-driven picks `peep`; `tv`, `vent_rate`, `pressure_support`, `lpm_o2` sit just below the top-30 cut). They are KEPT as inputs (known by hour 24, legitimate predictors) and flagged as overlap with the label's ventilation evidence (see below).

## Variable selection (fitted on TRAINING patients only)

Per variable: number of training patients with >= 1 measurement in their window. Selected = clinically named **roles** with >= 0.5% coverage (priority labs are kept down to
a low coverage, "where supported by actual data coverage") + the top 30 other labs by coverage with >= 2%. Lab names are matched EXACTLY (case-insensitive); several names of one role
are merged into one series. **Verify the names against `lab.labname.unique()`**: `availability_report.md` lists every role with its coverage, and a role at ~0 means its name did not match.
(The five names used by the incident label were already confirmed on the real data; the others are standard eICU names that have NOT been checked against your export.)

| Role | Accepted lab names | Reference range [low, high] |
|---|---|---|
| lactate | `lactate` | [None, 2.0] |
| creatinine | `creatinine` | [None, 1.3] |
| platelets | `platelets x 1000` | [150, 450] |
| bilirubin | `total bilirubin` | [None, 1.2] |
| wbc | `wbc x 1000` | [4, 12] |
| bun | `bun` | [None, 25] |
| sodium | `sodium` | [135, 145] |
| potassium | `potassium` | [3.5, 5.0] |
| chloride | `chloride` | None |
| bicarbonate | `bicarbonate`, `hco3` | [22, 29] |
| glucose | `glucose`, `bedside glucose` | [70, 180] |
| ph | `ph` | [7.35, 7.45] |
| paco2 | `paco2` | [35, 45] |
| pao2 | `pao2` | [80, None] |
| fio2 | `fio2` | None |
| hgb | `hgb` | [10, None] |
| inr | `pt - inr` | [None, 1.5] |
| ast | `ast (sgot)` | [None, 80] |
| alt | `alt (sgpt)` | [None, 80] |
| albumin | `albumin` | [3.0, None] |

Ranges of derived / vital variables:

| Variable | Range |
|---|---|
| pf_ratio | [300, None] |
| bun_creat_ratio | [None, 20] |
| anion_gap_calc | [None, 12] |
| hr | [60, 100] |
| rr | [12, 20] |
| spo2 | [92, None] |
| sbp | [90, None] |
| map | [65, None] |
| temp | [36.0, 38.3] |
| shock_index | [None, 0.9] |
| pulse_pressure | [25, 60] |

Reference ranges are project-defined adult limits, configurable, not diagnostic thresholds. Only variables with a range get the family-C features.

## ONE unified vector -- names and definitions

Column order is fixed: every selected lab variable (roles first, then data-driven by coverage), vitals if present, derived indicators, then patient-level, then interactions.
A feature is `NaN` when it cannot be computed (not zero, not imputed); counts that are facts (`n_meas`, `any_meas`, `hourly_coverage`) are 0 for an unmeasured variable.

**Per variable** (`<var>` = role key, sanitised lab name, or a derived indicator name):

| Feature | Family | Needs | Definition |
|---|---|---|---|
| `<var>__mean` | A | n>=1 | mean of the measurements in the window |
| `<var>__median` | A | n>=1 | median of the measurements |
| `<var>__min` | A | n>=1 | minimum measurement |
| `<var>__max` | A | n>=1 | maximum measurement |
| `<var>__std` | A | n>=2 | sample standard deviation (ddof=1) |
| `<var>__iqr` | A | n>=4 | interquartile range p75-p25 (linear interpolation) |
| `<var>__mad` | A | n>=3 | median absolute deviation: median(|x - median(x)|) |
| `<var>__p10` | A | n>=4 | 10th percentile |
| `<var>__p90` | A | n>=4 | 90th percentile |
| `<var>__cv` | A | n>=2 and |mean|>0 | coefficient of variation std/|mean| |
| `<var>__first` | A | n>=1 | earliest measurement |
| `<var>__last` | A | n>=1 | latest measurement in the window (the value known at the cutoff) |
| `<var>__early_mean` | A | >=1 in first half | mean of measurements in [0, 12h) |
| `<var>__late_mean` | A | >=1 in second half | mean of measurements in [12h, 24h] |
| `<var>__slope_per_h` | B | n>=3 and span>=4h | ordinary-least-squares slope of value on time, units per hour |
| `<var>__late_minus_early` | B | both halves observed | late_mean - early_mean |
| `<var>__rate_first_last_per_h` | B | n>=2 and span>=2h | (last - first) / hours between them |
| `<var>__median_abs_change` | B | n>=3 | median |difference| between consecutive measurements |
| `<var>__max_abs_change_6h` | B | >=1 consecutive pair 0.5-6h apart | largest |difference| between consecutive measurements taken 0.5-6 h apart |
| `<var>__frac_abnormal` | C | range defined, n>=1 | fraction of measurements outside the reference range |
| `<var>__n_abnormal` | C | range defined, n>=1 | count of measurements outside the reference range |
| `<var>__n_abn_episodes` | C | range defined, n>=1 | number of maximal runs of consecutive abnormal measurements |
| `<var>__abn_time_frac` | C | range defined, >=1 usable segment | fraction of observed time outside range (value carried forward <= 12h between measurements; longer gaps are unobserved) |
| `<var>__longest_abn_h` | C | range defined, >=1 usable segment | longest uninterrupted abnormal stretch in hours (same carry-forward rule) |
| `<var>__first_abnormal` | C | range defined, n>=1 | 1 if the earliest measurement is outside range else 0 |
| `<var>__last_abnormal` | C | range defined, n>=1 | 1 if the latest measurement is outside range else 0 |
| `<var>__recovered` | C | range defined, n>=2, first abnormal | 1 if first abnormal and last normal; 0 if first and last abnormal (persistence); NaN if first normal |
| `<var>__deteriorated` | C | range defined, n>=2, first normal | 1 if first normal and last abnormal; 0 if both normal; NaN if first abnormal |
| `<var>__n_meas` | F | always defined (0 if none) | number of measurement times in the window |
| `<var>__any_meas` | F | always defined | 1 if at least one measurement else 0 |
| `<var>__mean_interval_h` | F | n>=2 | mean hours between consecutive measurements |
| `<var>__max_gap_h` | F | n>=2 | longest gap between consecutive measurements, hours |
| `<var>__hours_since_last` | F | n>=1 | hours from the latest measurement to the window cutoff (24h) |
| `<var>__first_meas_h` | F | n>=1 | hours from ICU admission to the first measurement |
| `<var>__hourly_coverage` | F | always defined | fraction of the 24 hourly bins containing >=1 measurement |

Family E (laboratory dynamics) is these features applied to the lab variables (lactate, creatinine, platelets, bilirubin, WBC always attempted). Family D (derived indicators) is the same set applied to:

| Derived indicator | Inputs (measured at the IDENTICAL offset) | Op |
|---|---|---|
| `pf_ratio` | pao2, fio2 | ratio_fio2 |
| `bun_creat_ratio` | bun, creatinine | ratio |
| `anion_gap_calc` | sodium, chloride, bicarbonate | anion_gap |
| `shock_index` | hr, sbp | ratio |
| `pulse_pressure` | sbp, dbp | difference |
| `map` | sbp, dbp | map_from_sbp_dbp (only if MAP not measured) |

**Patient-level** (from lab DRAWS = distinct timestamps with >= 1 selected lab):

| Feature | Family | Definition |
|---|---|---|
| `pt__n_meas_total` | F | total lab measurements across all selected lab variables |
| `pt__n_vars_measured` | F | number of selected lab variables with >=1 measurement |
| `pt__n_draws` | F | number of distinct lab draw times (timestamps with >=1 selected lab) |
| `pt__mean_draw_interval_h` | F | mean hours between consecutive lab draws (>=2 draws) |
| `pt__max_draw_gap_h` | F | longest gap between consecutive lab draws, hours (>=2 draws) |
| `pt__hours_since_last_draw` | F | hours from the latest lab draw to the cutoff (>=1 draw) |
| `pt__first_draw_h` | F | hours from ICU admission to the first lab draw (>=1 draw) |
| `pt__draw_hourly_coverage` | F | fraction of the 24 hourly bins with >=1 lab draw |
| `pt__n_vars_ever_abnormal` | C | number of ranged lab variables with >=1 abnormal measurement (>=1 ranged measurement) |
| `pt__frac_vars_ever_abnormal` | C | n_vars_ever_abnormal / ranged lab variables measured |
| `pt__n_vars_recovered` | C | ranged lab variables flagged recovered |
| `pt__n_vars_deteriorated` | C | ranged lab variables flagged deteriorated |
| `pt__max_abn_per_draw` | G | max number of abnormal ranged tests within one lab draw (joint abnormality) |
| `pt__mean_abn_frac_per_draw` | G | mean over ranged draws of (abnormal ranged tests / ranged tests measured) |
| `pt__n_draws_multi_abn` | G | number of draws with >= joint_abnormal_min abnormal ranged tests |
| `pt__last_draw_n_abn` | G | abnormal ranged tests in the latest ranged draw |

**Cross-variable interactions** (family G; both variables measured at the identical offset, both in the stated abnormal direction; only created if both variables exist and have a range):

| Rule | Condition |
|---|---|
| `renal_joint` | creatinine high AND bun high |
| `acid_base_joint` | ph low AND lactate high |
| `resp_lab_joint` | pao2 low AND paco2 high |
| `coag_joint` | platelets low AND inr high |
| `shock_joint` | hr high AND sbp low |
| `resp_vitals_joint` | rr high AND spo2 low |
| `hr_map_joint` | hr high AND map low |

| Feature | Family | Definition |
|---|---|---|
| `ix__<rule>__n_aligned` | G | pairs measured at the identical offset |
| `ix__<rule>__n_joint` | G | aligned pairs with both in the stated abnormal direction |
| `ix__<rule>__frac_joint` | G | n_joint / n_aligned (n_aligned>=1) |

`feature_definitions.md` in every run lists the exact columns of THAT run (including the data-driven labs), with a `kept` flag, and every unavailable item with its reason.
Gating (`tcd.min_counts`, `min_span_hours`, `short_change_hours`, `locf_max_gap_hours`) is configuration; the defaults are in the table above.

## Preprocessing (fitted on TRAINING patients only; validation/test are only transformed)

1. drop features missing in > 95% of training patients, never observed, or constant after winsorisation (each with a recorded reason code);
2. winsorise to the training 0.5-99.5 percentiles; median-impute (training median); standardise (training mean/std of the winsorised observed values);
3. add one missing-indicator column per distinct training missingness pattern (features with identical patterns share a column).
State is persisted (`preprocessing_state.npz` + `preprocessing_meta.json`); `tcd_features_raw.npy` (NaN-bearing) and `tcd_features_transformed.npy` (encoder input) are both saved with
`patients.csv` (patient id, hospital id, split, window end) in the SAME row order; `feature_schema.json` holds name, variable, family, definition, requirement and label-overlap per feature.

## Overlap with the incident sepsis label (audited, documented)

The label's organ-dysfunction score uses platelets, bilirubin, creatinine, paO2 and FiO2 AFTER hour 24; TCD uses the same labs BEFORE it. That is forecasting signal, not leakage, but it is
flagged per feature: `direct` (a feature of one of those variables or of `pf_ratio`, or of a ventilator / oxygen-support setting: `peep`, `tv`, `vent_rate`, `pressure_support`, `lpm_o2`, which mark the
mechanical-ventilation evidence the label uses after hour 24), `partial` (a patient-level or interaction feature that aggregates over them), `none`; `overlap_basis` in `feature_schema.json` says why.
The label's antibiotic evidence and vasopressor records are not TCD inputs. **Feature runs sealed before 2026-10-31 (including `tcdf_20261031T162203Z_3bc446e5`) label `peep` as `none` in
their `feature_schema.json`** (the flag did not exist yet; the feature VALUES are unchanged and the run is still valid). Every encoder run records the authoritative `overlap_audit` (computed from the
current code over the feature run's variables) in its `manifest.json` and `config_snapshot.json`.

## Encoder

The existing `TabularMLPEncoder` (Linear-ReLU-Dropout-Linear; `hidden_dim` 128, `embedding_dim` 64, `dropout` 0.1, all configurable) maps the transformed vector to ONE embedding;
two linear heads (mortality, incident sepsis) only provide the supervised objective. Loss = `mortality_weight * focal(mortality) + sepsis_weight * focal(sepsis)` using the repo's
`masked_focal_loss` on logits (`gamma` 2.0, `alpha` 0.25, weights 1.0/1.0); a patient without a label for a task does not contribute to that task.
**Caveat:** the repo's focal `alpha` weights the POSITIVE class (0.25 down-weights it); with rare positives you may want a larger value, so it is configurable and recorded.
Adam (`lr` 0.001, weight decay 1e-05), batch 1024, up to 30 epochs, patience 5, seed 42; the best-validation weights are returned.
**Training population (`training.landmark_only`, default true since 2026-10-31):** gradient updates AND early stopping use only patients still in the ICU at the landmark (valid ICU exit >= 1440 min,
the paper track's strict landmark: 131,695 of the 199,108 patients). Short stays have clipped windows whose features (hours since the last lab, coverage, missing indicators) reveal the short stay, so an all-stays
mortality objective can teach the encoder short-stay patterns that do not exist in the landmark cohorts. Embeddings are still produced for EVERY patient; the population (counts per split, rule) is recorded in
`manifest.json` and `training_log.json`. Set `landmark_only: false` to reproduce the all-stays encoder. The flag is part of the encoder's cache identity.
Split discipline: gradient updates use TRAIN patients only; VALIDATION only drives early stopping; TEST labels are blanked before the training function can see them and the test split is never evaluated here.
Mortality = `hospitaldischargestatus` (the existing label); incident sepsis = the label of the incident run (policy-dependent; excluded patients carry NO label, never 0).
Output `tcd_embeddings.npz` with keys `patient_ids`, `embeddings` (float32, N x `embedding_dim`) -- the same format as the GRU / autoencoder embedding files, so the existing ID-based alignment
(`align_embeddings_to_cohort`) works unchanged. The existing cross-attention module projects each modality through its own linear layer, so any TCD dimension is compatible; the planned
three-modality cross-attention (NOT part of this stage) needs a third projection for TCD. Default dims: GRU 128, autoencoder 64, TCD 64.

## Real-data results (2026-10-31)

Feature run `tcdf_20261031T162203Z_3bc446e5` (445.8 s): 199,108 patients (train 139,208 / val 29,932 / test 29,968); 66,208 windows clipped at ICU exit. 50 lab variables = all 20 roles (train coverage 21.6% lactate ...
86.0% glucose; every configured name matched) + the top-30 others. 1,595 raw features -> 1,289 kept (306 dropped, all >95% missing in training) + 439 missing-indicator columns = encoder input 1,728.
Lab rows read for the selected variables 36,800,732, of which 22,272,098 were after the patient's window and 6,792,056 before ICU admission (dropped); maximum offset used 1,440 min. Poison audit on 2,000 random
patients: 105,554 rows after the window were present and corrupted; features bit-identical, reader and clipping agree, match to the main run (max abs difference 0.0): **passed**.
Unavailable (no vitals table): shock index, pulse pressure, MAP and the three vitals interactions. Overlap counts of that sealed run: direct 201, partial 25, none 1,369 (peep not yet flagged, see above).

First encoder run `tcde_20261031T163020Z_e5fdd41e` (62.4 s; ALL stays, i.e. `landmark_only: false`-equivalent): train mortality 12,590 positive / 126,618 negative, train incident sepsis (alive_discharge_as_negative run
`inc_20261030T104225Z_86c9624b`) 3,122 / 47,140. Best epoch 4 of 9 (early stopped; validation focal loss 0.04339). **VALIDATION-split** AUROC at epoch 4: mortality 0.829, incident sepsis 0.773 (peak 0.779 at epoch 3;
the sepsis AUROC then falls to 0.731 by epoch 9 while training loss keeps dropping: it overfits quickly). These are validation numbers only, mortality is over ALL stays (short-stay effect), and the sepsis negatives are
mostly short-stay alive discharges: neither is comparable to the paper-track numbers, and the test split has not been evaluated.

## Cache and identity rules

`tcd_features` identity = definition version `tcd_v1` + every parameter that can change a feature (NOT `training`, `encoder`, `audit`, chunk size) + the split-manifest hash + size/mtime of the
patient, lab (and vitals) tables. `tcd_encoder` identity = feature identity + incident-label run identity + encoder + training settings. A completed run is reused only if its `manifest.json` exists and every
file still matches its SHA-256; anything else recomputes into a NEW directory; an explicit `--run-id` that exists is an error; writes are atomic (temp file + rename, arrays streamed).

## Verification (sandbox; synthetic data)

`tests/unit/test_three_modality_tcd_features.py` (34), `tests/unit/test_three_modality_tcd_encoder.py` (22, torch-gated except the label/landmark-rule tests), `tests/integration/test_three_modality_cli.py` (7: 4 for stage 1, 3 for the TCD stages through the real CLI, chained incident_label -> tcd_features -> tcd_encoder), and the stage-1 unit file (32). 95 three-modality tests in total, all passing.
Highlights: a hand-computed 5-measurement series checked feature by feature; the vectorised code compared with an independent loop-based
reference on ~630,000 feature values (12 seeds x 4 range settings; two disagreements were traced to floating-point artefacts in the REFERENCE, not the code: whole-minute differences are exact at the
0.5 h / 2 h / 4 h / 6 h boundaries, differences of already-divided hours are not); missingness never fabricated; minute 1440 in / 1441 out; rows after ICU exit and before admission ignored; the poison
audit has teeth AND fails when the clipping is removed; selection and preprocessing identical when validation/test rows are corrupted (through the pipeline, not just the helper); encoder test labels
unreachable; determinism; best-epoch restore; focal gamma/alpha/weights take effect.
Hand-performed mutation proofs (inject a bug, see a test fail, restore byte-identical): 9 on the feature code (clipping removed; window end off by one; pre-ICU rows allowed; reader ignoring the window;
preprocessing fitted on all patients; coverage over all patients; no ICU-exit clipping; NaN imputed to 0; fit including validation) and 13 on the encoder and the landmark/overlap logic (test rows used for training; early stopping on the test split;
focal gamma ignored; sepsis weight ignored; best weights not restored; sepsis labels fabricated as 0). A first-draft problem worth remembering: a torch-free helper test would NOT have caught
"preprocessing fitted on all patients", so the pipeline-level tests were written before the mutants were run.

## Limitations

* Only lab values are time-stamped value series in this export; family D/G vital-sign items are unavailable until a vitals table exists.
* Lab-name matching is verified on the real data for all 20 roles. Several data-driven "labs" are respiratory-support settings or non-lab measurements (peep, o2_sat, urinary specific gravity ...); they are kept and, for ventilator settings, flagged.
* Reference ranges are project-defined and not sex/age adjusted.
* Sparse labs: most per-variable statistics need >= 3-4 measurements in 24 h, so many features are NaN for most patients (reported per feature and split).
* The audit covers lab rows; vitals are not audited (inactive here).
* Measured on the real run: 7.4 minutes for features (two passes over `lab`), about 1 minute for the encoder; raw matrix 199,108 x 1,595 and transformed 199,108 x 1,728 float32.

## Immediate next step (needs your machine)

Re-run `--stage tcd_encoder` (features and incident label are cache hits): with `landmark_only` now the default it creates a NEW encoder run; compare its validation AUROCs with the all-stays run above (expect mortality to drop, since short stays are the easiest). The paragraph below is the original first-run checklist, kept for reference.


Run `--stage tcd_features`, then read `availability_report.md`: (1) patient counts per split (train 139,208 / val 29,932 / test 29,968 expected from the manifest), (2) the variable table (a role at ~0 = unmatched lab name),
(3) the kept/dropped counts and the unavailable list, (4) the cutoff audit line (`passed = True`, late rows corrupted > 0). Then `--stage tcd_encoder`.
