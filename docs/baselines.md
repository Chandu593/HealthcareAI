# Baselines (Phase 13)

**Status: implemented, sandbox-verified, AND real-full-scale-verified**
(2026-09-05): local baseline 207/207 hospitals with decreasing loss
over 15 epochs each; centralized baseline 15 epochs, 207 hospitals,
0 non-finite, -44.7% net loss change (textbook decelerating
convergence, confirmed via the full epoch-by-epoch trajectory -- see
"A correction" below); federated baseline correctly referenced Phase
12's actual real result. `run_phase13_baselines()` and its two new
training paths are fully tested (9 integration tests, 2 phase-
resumption tests) including two hand-performed revert→fail→restore→
pass proofs.

## A correction, recorded rather than glossed over

Based on only 3 log-sampled points (epochs 5, 10, 15), the assistant
initially told the user the centralized baseline's improvement looked
like it was ACCELERATING and likely hadn't converged. Once
`scripts/check_local_training_quality.py` was extended to show the
full 15-epoch trajectory and the user ran it, the real picture was the
opposite: 32.4% of the entire run's improvement happened in just
epochs 1-5, which the 5/10/15 sampling had entirely missed --making
the later, already-decelerating portion look like acceleration purely
by comparison to what was visible. The full data shows textbook
decelerating convergence (first-half change -0.2726, second-half
change -0.0558) -- confirmed correct only once actually checked with
complete data, not inferred from a partial log.

## A correction, recorded rather than glossed over

Immediately after Phase 12 closed out, the assistant told the user
"evaluation" was the natural next phase -- said without checking the
tracker first. The tracker's own numbering has **Baselines (13) before
Evaluation (14)**, which makes sense in hindsight: you need something
to compare the federated model against before evaluation metrics mean
anything. Corrected before any design work or code was written, not
discovered partway through.

## What Phase 13 does

Three points of comparison: local, centralized, federated.

**Federated needs no new work.** It's Phase 12's already-completed
real result, referenced by reading `checkpoints/phase12_fedavg/
fedavg_manifest.json` directly -- the same "reuse already-completed
real work" precedent as Phase 12's own round-1 checkpoint reuse.

**Local and centralized are what this phase actually trains.**

## The "reset frequency" framing

All three baselines share the same architecture, same total
per-hospital epoch budget, and same loss function. The one variable
under test is how often (if ever) hospitals get synced to a shared
model:

| Baseline | Sync frequency |
|---|---|
| Local | Never -- each hospital trains completely independently (infinite effective resets) |
| Federated | Every `training.local_epochs` epochs -- `federated.n_rounds - 1` resets (2, with the current config) |
| Centralized | Never resets -- ONE continuously-updated model, zero resets, fully continuous |

## Two real decisions, confirmed with the user before building

### 1. Centralized baseline: sequential/continuous, not true pooling

A genuinely standard "centralized" baseline in federated-learning
research pools raw data across every client into one training set --
which here would mean building graphs that span multiple hospitals,
directly breaking the "never pool data across hospitals" rule this
project has enforced as absolute since Phase 3. This was presented as
a real fork, not assumed either way:

- **Pool anyway**, as a deliberate, clearly-labeled, comparison-only
  exception -- standard for this exact research question (measuring
  the "cost of privacy"), but real new engineering (a cross-hospital
  graph-construction path that doesn't exist anywhere in this
  project).
- **Keep hospital-scoped graphs**, and just remove the federated
  reset/averaging entirely -- cheaper (reuses almost all existing
  infrastructure), keeps Phase 8's locked intra-hospital-only design
  and the project's core privacy boundary completely untouched, but a
  weaker, non-standard notion of "centralized" (isolates the effect of
  rounds/resets specifically, not raw data pooling).

**Chosen: keep hospital-scoped graphs.**
`_run_centralized_sequential_training()` constructs ONE
`LocalPatientModel`, and that SAME instance is used for the entire
run -- never reconstructed, never reset. Each epoch, every hospital's
own intra-hospital graph gets exactly one gradient step, in a
reshuffled visit order (seeded, for reproducibility -- a fixed order
would bias the model toward whichever hospital always comes last). ONE
optimizer instance persists for the whole run too, so Adam's momentum/
variance state carries over continuously, exactly as it would for any
ordinary SGD/Adam loop fed a sequence of different mini-batches (here,
each hospital's own graph is one such "batch").

Each hospital's raw tensors/graph are re-loaded fresh on every visit
(via the newly-extracted `_load_hospital_batch()`, shared with
`_run_local_training_round()`) rather than pre-loaded once and held in
memory for the whole run -- Phase 4's LSTM already needed real-data
batching to avoid OOM holding the full 199,108-patient cohort at once;
pre-loading per-hospital copies for all ~207 hospitals simultaneously
would roughly double that memory footprint for no real benefit, since
slicing/graph-loading itself is cheap compared to the actual forward/
backward pass.

A non-finite (NaN/Inf) loss on any single hospital-visit skips just
that one optimizer step (model not updated from it) and training
continues -- there's no equivalent "exclude from this round" concept
in a single continuous run the way Phase 11/12 exclude a bad hospital
from that round's checkpoint; skipping the one bad step and moving on
is the direct analogue.

### 2. Local baseline: matched total epoch budget

Federated hospitals see 15 epochs cumulatively (3 rounds × 5 local
epochs, with 2 resets to a shared model in between). A local-only
baseline capped at Phase 11's original single-round 5 epochs would
look artificially worse than federated purely from getting a third of
the total training -- not because federation itself helped.

`total_epochs = training.local_epochs * federated.n_rounds` is
**derived automatically** (15, with the current config) rather than a
new, independently-settable value -- avoids the two numbers silently
drifting out of sync if `federated.n_rounds` is ever changed later.
`_run_local_training_round()` gained an optional
`local_epochs_override` parameter (defaults to `None`, preserving
existing Phase 11/12 behavior exactly) so the local-only baseline can
request this matched budget without touching config state or
duplicating the training loop.

## Engineering: `_load_hospital_batch()` extracted

The per-hospital tensor-slicing + label-alignment logic that was
previously inline in `_run_local_training_round()` is now a shared
helper, used by that function AND the new centralized-sequential loop.
Verified behavior-preserving by re-running Phase 11's and Phase 12's
existing 18 integration tests completely unmodified after the
extraction -- all 18 passed before any Phase 13-specific code was
written, mirroring exactly how the Phase 11→12 extraction was verified.

## What this does NOT yet do

- **No metric computation, no comparison.** This phase trains the
  local and centralized baselines; it does not compute AUROC/AUPRC/
  accuracy or compare any of the three baselines against each other.
  That's explicitly Phase 14 (Evaluation)'s job, per the tracker's own
  separation of these two phases -- and also needs a train/val/test
  split, which doesn't exist anywhere in this project yet.
- **Not yet re-verified against the real 199,108-patient / 208-hospital
  dataset.** Sandbox-verified only so far.

## Verification status

| Component | Status |
|---|---|
| `_load_hospital_batch()` extraction | Verified — Phase 11's and Phase 12's existing 18 integration tests re-run completely unmodified, all passing |
| Local baseline fair epoch budget (15, not 5) | Verified — dedicated integration test, revert→fail→restore→pass proof (caught the log itself reporting the wrong total, 3 instead of 6, under the test's own smaller config) |
| Centralized baseline never resets the model | Verified — `test_centralized_model_constructed_exactly_once`, revert→fail→restore→pass proof against a per-hospital-reconstruction regression |
| Centralized baseline reshuffles hospital order each epoch | Verified — dedicated integration test |
| Centralized baseline NaN/Inf step-skip without crashing | Verified — dedicated integration test with a monkeypatched poisoned step |
| Federated baseline reference (present and gracefully-absent cases) | Verified — 2 integration tests |
| `baselines_manifest.json` persistence | Verified — dedicated integration test |
| `--start-phase 13` dependency-flag correctness | Verified — 2 tests in `test_phase_resumption.py` |

## Immediate next step

Real run complete on both new baselines, both confirmed healthy.
`scripts/check_local_training_quality.py` now supports both manifest
shapes (extended 2026-09-05, after being flagged as a known gap at
delivery) -- it auto-detects `checkpoints/phase13_baselines/local_only/`
(per-hospital) vs. `checkpoints/phase13_baselines/centralized_sequential/`
(single continuously-trained model) and reports accordingly, including
a full epoch-by-epoch trajectory and an explicit accelerating-vs-
decelerating check for the centralized case. Next up: Phase 14
(Evaluation, global + per-hospital) -- confirmed against the tracker's
own naming this time, not assumed.
