# Three-modality suite, stage 3a: first-24-hour GRU inputs + pooled focal GRU encoder

**Status (2026-11-01): implemented and sandbox-verified on synthetic data (28 tests + 12 hand-performed mutation proofs + 1 CLI chain test). NOT yet run on the real eICU export.**
Training regime (fixed requirement): **pooled / central** training of every neural component on the training patients of all hospitals. This experiment is NOT federated learning and has no
per-hospital encoders; the hospital boundary is kept where it matters (Graph A/B are strictly hospital-local, built in a later stage).

```
python run_three_modality.py --config configs/three_modality.yaml --stage gru_inputs      # needs a tcd_features run (same config); writes outputs/three_modality/gru/inputs/
python run_three_modality.py --config configs/three_modality.yaml --stage gru_encoder     # needs gru_inputs + an incident_label run with the SAME censoring policy; torch required
python run_three_modality.py ... --stage gru_inputs --inspect-cache                       # lists completed runs, 'matches-current=YES/no'; computes nothing
python run_three_modality.py ... --stage gru_encoder --force-recompute                    # NEW run directory; completed runs are never overwritten
```

## Why not the legacy Phase-4 arrays

Audited before building (config + code): the legacy time-series grid is `time_bin_minutes: 240` x `max_sequence_length: 48` = **192 hours** (no overlay shortens it), so it contains hours 24-96, the
incident-event window; it also carries medication and infusion channels; and `sequence_length` is computed from `patient.unitdischargeoffset`, so the ICU length of stay enters the recurrent encoder
through packing. None of that is compatible with "first 24 ICU hours only", so the legacy arrays (including anything under `outputs/split_aware`) are not used. The existing `GRUTimeSeriesEncoder`
module IS reused.

## Inputs (`gru_inputs.py`)

* Window = exactly the first `tcd.window_minutes` (1440 = the incident cutoff; `gru.n_bins * gru.bin_minutes` must equal it or the config is rejected): rows with `0 <= offset <= min(ICU exit, 1440)`,
  enforced by the reader (the same one as the TCD features), by `bin_series` itself (defence in depth), and by a real-data **poison audit** (random patients re-read WITHOUT the window, every later
  value corrupted x1000+12345, arrays must be bit-identical, equal to the reader-enforced path and to the main run; the audit must have corrupted rows or it fails).
* 24 bins of 60 minutes: bin b covers (60b, 60(b+1)] minutes, offset 0 belongs to bin 0 and minute 1440 to bin 23; within a bin the LAST observed value is kept.
* Channels per variable: [scaled value, observation mask]. Unobserved cells are value 0 AND mask 0: nothing is imputed or carried forward. Variables = the lab variables the TCD run selected
  (training-patient coverage, so the selection is already train-only; 50 on the real data).
* Scaling: per variable, winsorise at the training 0.5/99.5 percentiles then standardise, fitted on TRAINING patients' observed cells only (pooled; no per-hospital statistics). Constant or never-observed
  variables are recorded as degenerate and scaled to 0.
* **Fixed sequence length (24) for every patient.** Nothing is derived from the ICU exit; a short stay just has mask 0 after the exit. (`input_report.json` states `sequence_length_derived_from_icu_exit: false`.)
* Patient order, hospital ids, splits and window ends are inherited from the verified TCD feature run, so every modality shares one ordering (the fusion code aligns by patient id anyway).
* Files (`outputs/three_modality/gru/inputs/gruin_.../`): `gru_values.npy` float32 (N,24,V), `gru_mask.npy` uint8 (N,24,V), `variables.json`, `scaling.json`, `patients.csv`, `input_report.json/.md`
  (observed-cell fraction and patients with >=1 observation per variable and split, patients with no observation at all, rows dropped), `cutoff_audit.json`, `config_snapshot.json`, `manifest.json`.

## Encoder (`gru_encoder.py`, `multitask_training.py`)

Existing `GRUTimeSeriesEncoder` (2 layers, hidden 128, dropout 0.2 by default; all configurable) over the 2V channels, with the final hidden state as the 128-d embedding, plus two linear heads (mortality,
incident sepsis proxy) that exist only to give the embedding a supervised objective.
* **Focal loss, separate per task:** `training.focal.mortality|sepsis.{alpha,gamma}` and `training.task_weights.{mortality,sepsis}` (the repo's `masked_focal_loss` on logits). A task with weight 0 ignores its
  own focal parameters entirely and its head receives no gradient (tested). **Caveat:** in the repo's implementation `alpha` weights the POSITIVE class, so the default 0.25 down-weights it; with rare positives you may
  want a larger value. It is configurable and recorded in the training log.
* Labels: mortality = `hospitaldischargestatus`; incident sepsis = the label of the incident run (policy-dependent; excluded patients carry NO label, never 0). Per-task masks: a patient without a label for a task
  contributes nothing to that task's term.
* Population: `training.landmark_only` (default true): updates and early stopping use only patients still in the ICU at the landmark (valid ICU exit >= 1440 min); every patient is still embedded.
* Split discipline: only TRAIN patients update weights; only VALIDATION selects the model (best validation total focal loss, weights restored); TEST labels are blanked before the trainer can see them.
* **Reproducibility:** the trainer takes a model FACTORY and builds the model after seeding. A first draft built the model before seeding, so two fresh processes started from different initial weights (found by a
  test I wrote, not by inspection); that is fixed and has a regression test that perturbs the global RNG state.
* Output `gru_embeddings.npz` (`patient_ids`, `embeddings` float32 N x 128), same format as the TCD embeddings so the fusion stage aligns by id; `gru_encoder.pt`, `training_log.json` (per-epoch train/validation loss
  and validation AUROC per task, population counts, hyperparameters, seed), `label_counts.json`, `manifest.json`.

## Cache and identity

`gru_inputs` identity = definition version + grid/scaling parameters + the TCD feature-run identity + window; encoder/training/audit settings do NOT change it. `gru_encoder` identity = inputs identity + incident-label
run identity + encoder + training settings (including per-task focal parameters and `landmark_only`). A completed run is reused only if its `manifest.json` exists and every file still matches its SHA-256; otherwise
a new directory is created; an explicit `--run-id` that exists is an error.

## Verification (sandbox, synthetic data)

`tests/unit/test_three_modality_gru.py` (28) + 1 chained CLI test (`gru_inputs` -> `gru_encoder`, cache hits, `--inspect-cache`, "run X first" guidance, 48-bin grid rejected). Highlights: bin edges at 0/59/60/61/1380/1381/1440/1441;
last value per hour; unobserved = 0 with mask 0 (no carry-forward); rows after the window / before admission change nothing; an early discharge keeps length 24 with unobserved bins after the exit; scaling identical when validation/test
cells (and, through the pipeline, their raw lab values) are corrupted; poison audit has teeth and FAILS when the clipping is removed; model shapes, gradient flow into the GRU and both heads, no interaction between patients in a batch,
a patient with no observation is finite, the encoder always runs over all 24 steps; per-task focal parameters and weights; per-task label masks; best-epoch restore; test labels and short-stay labels unreachable; determinism.
Mutation proofs (inject a bug, see a test fail, restore byte-identical), 12: bin edges floor vs ceil; clipping removed; first instead of last value; non-zero unobserved cells; scaling fitted on all patients; length derived from the data;
model built before seeding; mortality using the sepsis focal parameters; task weights ignored; best weights not restored; landmark ignored; test rows used for training with test labels unblanked. All caught.

## Limitations

* Only the 50 TCD-selected lab series are sequence channels (no vitals exist in this export; no medication/infusion/treatment channels, deliberately).
* Last-value-per-hour binning discards earlier values in the same hour; their information is in the TCD features.
* Real-data runtime and memory are unmeasured: the arrays are about 199,108 x 24 x 50 float32 (about 1 GB) plus the uint8 mask (0.24 GB); a CPU epoch over about 92k landmark training sequences is expected to take on the order of a minute or two.

## Immediate next step (needs your machine)

`--stage gru_inputs`, then read `input_report.md`: (1) patient counts per split, (2) observed-cell fractions per variable, (3) patients with no observation at all, (4) the cutoff audit line (`passed = True`, late rows corrupted > 0).
Then `--stage gru_encoder` and send `training_log.json` and `manifest.json`.
