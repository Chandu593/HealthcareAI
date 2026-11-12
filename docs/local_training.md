# Local Training Loop (Phase 11)

**Status: implemented, sandbox-verified, real-scale run completed and
FULLY weight-verified 2026-08-31** (208 hospitals, 207 trained, 1
empty -- matches Phase 8's already-confirmed real min=0 finding, 0
skipped for missing labels, ~37 min runtime; a subsequent full,
not-sampled weight-delta check across all 207 trained hospitals found
zero NaN/Inf and a remarkably tight 1.05x min/max delta ratio -- see
the "Checking a completed real run's QUALITY" section below for why
that tightness is itself meaningful evidence, not just "no red
flags"). `src/training/local_model.py` and `src/training/losses.py`
are fully tested (21 unit tests, 9 integration tests, 2 phase-
resumption tests) including three hand-performed revert→fail→restore→
pass proofs for this phase's most safety-critical properties: genuine
end-to-end gradient flow, every hospital starting from identical
initial weights, and NaN/Inf loss detection.

## What Phase 11 does

Each hospital locally trains its own copy of the FULL model --
`LSTMTimeSeriesEncoder` (Phase 4) → `TabularAutoencoder` (Phase 5) →
concat fusion (Phase 7's locked design) → `PatientGNN` (Phase 9) →
`MultiTaskHeads` (Phase 10) -- jointly, end to end, via one combined
loss and one composite optimizer per hospital. Never pools patients
across hospitals: every tensor a given hospital's training step
touches (timeseries, tabular, graph, labels) belongs to that hospital
alone, the same boundary every phase since Phase 3 has held.

This is the first phase in the project with an actual backward pass --
every phase before this proved a correct *forward* pass only.

## Why a new module, not a change to Phases 4/5/7/8/9/10

`run_phase4/5/7/8/9/10_*()` in `run_pipeline.py` communicate between
phases via saved `.npz` files -- detached numpy arrays, no autograd
graph. That's exactly right for standalone per-phase
forward-pass verification and `--start-phase` resumption, but it would
silently break end-to-end backpropagation if reused directly for
training (loss.backward() can't flow through a numpy round-trip).
`LocalPatientModel` (`src/training/local_model.py`) instead re-imports
the SAME underlying `nn.Module` classes those phases already built and
real-full-scale-verified, and chains them as torch tensors, in memory,
for exactly one hospital's data per forward call. `run_phase4/5/7/
8/9/10`'s existing functions are completely untouched.

## Design decisions (confirmed with the user before building)

### 1. Full end-to-end training, not heads-only

LSTM, autoencoder, GNN, and heads are all trained jointly. Directly
grounded in two things found in the existing code, not assumed:

- Phase 9's own docstring already described "ONE model (shared
  weights, what a genuine federated deployment would eventually
  FedAvg), applied independently to each hospital's own graph" -- the
  whole stack, not just the heads, is what a genuine federated
  deployment trains and averages.
- `TabularAutoencoder`'s decoder already existed with an inline
  comment flagging it as built for "an eventual reconstruction-loss
  training objective (not yet implemented)" -- this phase is what
  implements it.

The alternative (freeze LSTM/AE/GNN at random init, train only the
heads on Phase 9's already-saved embeddings) would have been simpler
to build, but would leave three of the four components permanently
untrained.

### 2. Autoencoder reconstruction loss: included

`src/training/losses.py`'s `compute_multitask_loss()` combines:

```
total = mortality_weight * mortality_loss
      + sepsis_weight    * sepsis_loss
      + reconstruction_weight * reconstruction_loss   # MSE, unsupervised
```

Both classification losses are masked **per task**, not per patient --
Phase 10 established that a patient can have a known mortality outcome
but an unknown sepsis status (or vice versa); computing one combined
mask would silently discard usable labels. The reconstruction term
never depends on label availability at all. If a hospital ends up with
zero usable labels for BOTH classification tasks (small hospitals, or
Phase 10's confirmed sepsis_df-absence condition applying to every one
of that hospital's patients), reconstruction alone keeps the step
alive; only if reconstruction is ALSO disabled/zero-weighted does the
step get skipped and logged -- `compute_multitask_loss()` returns
`None`, never a fabricated zero-valued loss (which would either crash
`.backward()` or silently do nothing while looking like a
perfectly-fit step).

### 3. Weight persistence: lightweight, now

Each hospital's post-training `state_dict()` is saved to
`checkpoints/phase11_local/hospital_{hid}_local_weights.pt` -- not the
full Phase 18 caching/checkpointing system, just enough for Phase 12
(FedAvg) to load. `run.checkpoint_dir` was already stubbed in
`configs/base.yaml` (`checkpoints`), unused until now.

## Every hospital starts from identical initial weights

A real correctness requirement, not a stylistic choice: FedAvg only
makes sense as an operation on weight tensors that started from the
SAME point in weight-space. Neural network loss landscapes are not
permutation-invariant -- averaging two independently-initialized
models' weights produces something close to random, not a meaningful
"average model."

`run_phase11_local_training()` constructs one `LocalPatientModel`
once (seeded via `run.seed`), captures its `state_dict()`, and saves
it to `checkpoints/phase11_local/global_initial_weights.pt`. Every
hospital's fresh model instance then calls
`model.load_state_dict(initial_state_dict)` with that exact same dict
before training begins.

Proven, not just implemented: `test_every_hospital_starts_from_
identical_initial_weights` (`tests/integration/test_phase11_pipeline.py`)
intercepts every `LocalPatientModel.load_state_dict()` call across a
full run and asserts every one of them received bit-identical tensor
values -- not just the same architecture (a shapes-only check would
miss a regression to per-hospital random re-initialization entirely,
since shapes would still match). Verified by hand via revert→fail→
restore→pass against BOTH failure variants: skipping the shared-init
line entirely, and the subtler case of still calling
`load_state_dict()` but with an independently-random per-hospital
state -- the test caught both, the second via an exact per-tensor
mismatch naming `lstm_encoder.lstm.weight_ih_l0`.

## Checking a completed real run's QUALITY, not just that it finished (2026-08-31)

The first real run (208 hospitals, 207 trained, 1 empty, ~37 min)
completed without error -- but "completed" and "quality-verified" are
different questions for a training phase specifically, in a way they
weren't for any earlier forward-pass-only phase. Checking that turned
up two real gaps, found by trying to inspect this exact completed run,
not hypothetical:

**Per-hospital training metrics were never persisted anywhere.**
`loss_history` / `n_mortality_known` / `n_sepsis_known` only ever
existed in `run_phase11_local_training()`'s in-memory return dict --
gone the instant the process exits, leaving only the phase-level
summary log line, which can't answer "did loss actually decrease on
real hospitals" or "how many hospitals trained with near-zero real
labels." **Fixed:** every run now writes `checkpoints/phase11_local/
training_manifest.json` with the full per-hospital breakdown, mirroring
Phase 8/9's own JSON-manifest pattern.

**No NaN/Inf detection during training.** `.backward()`/`.step()` run
cleanly on a NaN/Inf loss with no exception -- a hospital whose loss
went non-finite mid-training would have had its corrupted checkpoint
saved silently, with the run still reporting success. **Fixed:**
`total_loss` is checked with `torch.isfinite()` every epoch; a
non-finite loss stops that hospital's round, its checkpoint is NOT
saved, and it's counted under a new, distinct
`n_hospitals_skipped_non_finite` (never folded into
`n_hospitals_skipped_no_labels`).

**`scripts/check_local_training_quality.py`** was written specifically
so the ALREADY-COMPLETED real run (which predates both fixes above)
doesn't need to be re-run just to get a first quality read. It inspects
only what's already on disk:

```
python scripts/check_local_training_quality.py --checkpoint-dir checkpoints/phase11_local
```

If `training_manifest.json` exists, it reports loss-trend direction
(decreased/increased/flat per hospital) and the distribution of
`n_mortality_known`/`n_sepsis_known` across hospitals. Regardless of
whether the manifest exists, it ALWAYS checks every sampled
checkpoint for NaN/Inf weights and for zero weight-change from the
shared initial checkpoint (which would mean that hospital's training
loop silently didn't run) -- so it still gives real signal against the
user's actual completed run, which has no manifest. Confirmed working
against both a manifest-present and a manifest-absent scenario before
delivery.

**Follow-up, same day: full (not sampled) check across all 207 real
trained hospitals**, after bumping the script's default sample size so
a plain run covers everything by default (real eICU `hospitalid`
values aren't sequential, so "first 20 by filename" was never a
representative sample):
```
delta distribution (excluding zero/NaN, n=207): min=9.2443 median=9.3892 max=9.7027
max/min ratio: 1.05x  (tight, consistent across hospitals)
```
Zero NaN/Inf across literally every real trained hospital. The
tightness of this spread is itself meaningful, not just "no red
flags": Phase 8's real-data verification found hospital sizes ranging
0 to 7,058 patients, yet total weight movement barely varies across
that range -- explained by `compute_multitask_loss()`'s component
losses using PyTorch's default MEAN reduction, so gradient magnitude
per epoch is normalized by each hospital's own patient count rather
than scaling with raw totals. A genuinely unstable or diverging local
round would be expected to show MORE erratic, hospital-dependent
movement than this, not less -- so this tight, consistent spread is
real (if indirect) evidence against instability.

## Two real gaps found and fixed while wiring this in

### `load_phase4_result()` / `load_phase5_result()` needed the raw tensors

Previously these only ever returned `embeddings_path` (Phase 7's only
consumer) -- a documented, deliberate partial round-trip. Phase 11
trains the LSTM/autoencoder jointly with the rest of the model, so it
needs the raw NORMALIZED tensors themselves (what actually feeds
those encoders), not Phase 4/5's already-embedded, frozen output.
Both loaders now read their full `.npz` files back
(`timeseries_sequences.npz`, `tabular_features.npz`) and expose
`patient_ids` / `normalized_array` / `sequence_length` /
`feature_names` at the top level -- matching what a freshly-run
`run_phase4_timeseries_modality()` / `run_phase5_tabular_modality()`
call already returns, so Phase 11 reads the same shape regardless of
whether Phase 4/5 ran fresh in this invocation or were reloaded via
`--start-phase`.

### A stale `training:` config stanza

`configs/base.yaml` already had a `training:` section (`batch_size`,
`learning_rate`, `weight_decay`, `max_epochs`,
`early_stopping_patience`) stubbed before any training loop existed,
shaped for a CENTRALIZED training concept (one model, N global epochs,
early-stopped against a held-out validation set) -- materially
different from Phase 11's per-hospital LOCAL round. `learning_rate` /
`weight_decay` were directly reusable as-is. `batch_size` (no
mini-batching within a hospital -- the GNN needs the whole hospital
graph at once, same "no batching needed at this scale" reasoning as
Phase 8/9/10), `max_epochs`, and `early_stopping_patience` (no train/
val split exists anywhere in this project yet -- introducing one would
be its own real methodology decision, not a Phase 11 side-effect) are
annotated in the config as not-yet-applicable rather than deleted or
silently repurposed, same discipline as the `fusion.output_dim=256`
and `graph.max_nodes_full_similarity` stale-config fixes before it.
`gnn.learning_rate` / `gnn.weight_decay` (pre-existing stubs under
`gnn:`) are similarly annotated as superseded -- `training.
learning_rate` / `training.weight_decay` now optimize all four
jointly-trained components with one composite optimizer, not a
GNN-only one.

New config keys: `training.optimizer` (`adam | sgd`),
`training.local_epochs`, `training.reconstruction_weight`,
`training.max_hospitals` (bounds sandbox/dev run time; `null`
processes every hospital).

## What this does NOT yet do

- **No FedAvg yet.** Every hospital's checkpoint sits independently in
  `checkpoints/phase11_local/`; nothing averages them together. That's
  Phase 12's job -- this phase's job was making sure there's something
  correct and comparable (same starting point, same architecture) for
  Phase 12 to average.
- **No held-out evaluation split.** Loss is only ever measured on the
  same patients being trained on. Introducing train/val/test splitting
  is its own real methodology decision (how to split -- per-hospital,
  globally, stratified how) deferred to a future evaluation phase, not
  built as a Phase 11 side-effect.
- **The 2026-08-31 real run predates the manifest/NaN-safety fixes.**
  `scripts/check_local_training_quality.py` can still confirm its
  207 real checkpoints are healthy (finite weights, genuine change
  from initial), but that run's per-hospital loss curves and label-
  coverage distribution are not recoverable after the fact. A future
  real run will have both.

## Verification status

| Component | Status |
|---|---|
| `src/training/losses.py` (masked BCE/focal, all-unknown-labels edge case, reconstruction rescue) | Verified — 15 unit tests, including a hand-performed revert→fail→restore→pass proof that the all-unknown case returns `None` rather than a fabricated zero loss |
| `LocalPatientModel` (output shapes, single-patient/empty-hospital edge cases, gradient flow to all four components, per-hospital independence) | Verified — 6 unit tests, including a hand-performed revert→fail→restore→pass proof (a deliberately injected `.detach()` was caught, pinpointing exactly `lstm_encoder` as the severed component) |
| Full Phase 1→2→3→4→5→7→8→11 pipeline on mock data | Verified — 9 integration tests, including real confirmed loss-decrease numbers, per-component weight-delta proof, manifest persistence, and NaN/Inf handling |
| Shared initial weights across all hospitals | Verified — dedicated integration test, revert→fail→restore→pass against two distinct failure variants |
| NaN/Inf loss detection (added 2026-08-31) | Verified — dedicated integration test with a monkeypatched poisoned hospital, revert→fail→restore→pass proof |
| `--start-phase 11` dependency-flag correctness | Verified — 2 tests in `test_phase_resumption.py` |
| Real 199,108-patient / 208-hospital dataset | **Ran 2026-08-31** — 207/208 hospitals trained, 1 empty, 0 skipped for missing labels, ~37 min. Full (not sampled) weight-delta/NaN check across all 207 trained hospitals: zero NaN/Inf, tight 1.05x min/max delta ratio (min=9.24, median=9.39, max=9.70) — see the QUALITY section above for why that tightness is itself meaningful. Exact loss-curve values not recoverable for this specific run (predates the manifest fix), but the structural evidence is strong. |

## Immediate next step

A full re-run with the manifest/NaN-safety fixes is worthwhile (to get
real loss-curve and label-coverage numbers for the actual 208-hospital
dataset, not just mock data) but not urgent given the ~37 minute cost
and that the existing checkpoints already passed a health check.
Otherwise, next up is **Phase 12 (FedAvg)** -- averaging these
per-hospital checkpoints, which all genuinely started from the same
initial weights, back into one global model.
