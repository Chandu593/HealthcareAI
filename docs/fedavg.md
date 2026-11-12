# Federated Averaging (Phase 12)

**Status: implemented, sandbox-verified, AND real-full-scale-verified**
(2026-09-02): 3 rounds, round 1 reused (0.13s), rounds 2-3 fresh
(~32 min and ~31 min respectively), 207/208 hospitals trained per
fresh round, 1 empty (the same structural hospital every round), 0
non-finite losses across 414 real hospital-training-instances (207 x 2
fresh rounds). `src/training/fedavg.py` and Phase 12's orchestration in
`run_pipeline.py` are fully tested (11 unit tests, 9 integration
tests, 2 phase-resumption tests) including four hand-performed
revert→fail→restore→pass proofs for this phase's most safety-critical
properties.

## What Phase 12 does

Multi-round FedAvg: each round, every hospital trains locally starting
from the CURRENT global model, then their updated weights are combined
via weighted averaging into a new global model, which becomes the next
round's starting point.

```
round 1: [global model] --local train--> [207 hospital checkpoints] --FedAvg--> [global model round 1]
round 2: [global model round 1] --local train--> [checkpoints] --FedAvg--> [global model round 2]
round 3: [global model round 2] --local train--> [checkpoints] --FedAvg--> [global model round 3]  <- final
```

This run: **3 total rounds**, with **round 1 reusing the user's
existing real Phase 11 checkpoints** instead of re-training from
scratch (both confirmed with the user, 2026-09-01 -- see "Two real
decisions" below).

## Aggregation: weighted by patient count, not an unweighted mean

`src/training/fedavg.py`'s `fedavg_aggregate(state_dicts, weights)`
implements the standard formulation from McMahan et al. (2017,
"Communication-Efficient Learning of Deep Networks from Decentralized
Data"): each hospital's contribution is weighted by how many patients
it trained on. A hospital with 7,058 patients (Phase 8's real max)
should influence the global model more than one with 20 -- matching
how a hypothetical centralized model trained on the pooled data would
naturally weight more evidence. An unweighted mean would be a
genuinely different algorithm, not an implementation detail.

Raises `ValueError` clearly on every way this could silently go wrong:
empty input, a hospital missing a weight, non-positive total weight,
or mismatched keys/shapes across hospitals' state_dicts (every
hospital is expected to share one `LocalModelConfig`, so a mismatch
here means a real bug upstream, not a routine case to paper over).

## Two real decisions, confirmed with the user before building

Both had a genuine time cost (~37 min/round on the user's real
hardware, per Phase 11's own confirmed runtime) -- not silently
defaulted.

### 1. Three total rounds

Multi-round, not a one-shot average of existing work. This was already
committed to in Phase 11's own config comment, written before Phase 12
existed: *"`training.local_epochs`: epochs per hospital PER LOCAL
ROUND (this phase only runs one round; Phase 12/FedAvg will call this
repeatedly)."* `federated.n_rounds` is config-driven.

### 2. Round 1 reuses the existing real Phase 11 checkpoints

Rather than re-training round 1 from scratch, Phase 12 loads whatever
per-hospital checkpoints are already on disk under
`checkpoints/phase11_local/` and aggregates them directly as round 1's
result -- saving ~37 minutes. This was a reasonable choice specifically
because those checkpoints were already thoroughly quality-checked
(2026-08-31): zero NaN/Inf across all 207 real trained hospitals, and
a remarkably tight 1.05x weight-delta spread (see
`docs/local_training.md`'s own quality-check section for why that
tightness itself was meaningful evidence, not just "no red flags").

`federated.reuse_existing_local_checkpoints_for_round_1` is
config-driven. If reuse is requested but nothing usable is found (a
fresh setup, or a genuinely empty `checkpoints/phase11_local/`), round
1 falls back to fresh training automatically, with a clear warning --
never a crash.

## Engineering: `_run_local_training_round()` extracted from Phase 11

Rather than reimplementing per-hospital local training for Phase 12's
rounds 2+, `run_phase11_local_training()`'s hospital loop (loss
masking, NaN/Inf safety, checkpointing, `training_manifest.json`) was
pulled into a shared helper, parameterized by `model_config` /
`initial_state_dict` / `checkpoint_dir` / `round_tag`.
`run_phase11_local_training()` itself becomes a thin wrapper that
builds the shared `model_config`, creates one fresh seeded set of
initial weights, and calls the helper once -- **behavior completely
unchanged**, proven by re-running Phase 11's existing 9 integration
tests, unmodified, after the extraction; all 9 passed.

## Round 1 reuse mechanics

`_try_reuse_existing_local_checkpoints()`:

1. Loads every `hospital_*_local_weights.pt` already on disk under
   `checkpoints/phase11_local/`.
2. Re-validates EACH checkpoint for NaN/Inf independently -- a prior
   run predating the 2026-08-31 NaN-safety fix (like the user's own
   real Phase 11 run) could in principle have saved a corrupted one.
   Excluded and logged, never silently aggregated in.
3. Derives each hospital's `n_patients` aggregation weight directly
   from Phase 8's own graph files (`len(patient_ids)` per hospital) --
   NOT from Phase 11's `training_manifest.json`, which may not exist
   (the user's real run predates that fix too).

A consequence worth calling out: Phase 12 does NOT need
`phase11_result` as an in-memory value at all. Round 1 reuse works via
`--start-phase 12` entirely on its own, with Phase 11 having run in a
completely separate, earlier invocation -- exactly the user's real
situation (Phase 11 ran, the process exited, and Phase 12 is a fresh
invocation reading whatever Phase 11 left on disk).

## A docs claim, checked and found false, fixed before the user hit it (2026-09-02)

This doc originally claimed `scripts/check_local_training_quality.py`
"can be pointed at `checkpoints/phase12_fedavg/round_2/` and `round_3/`
the same way it was used for Phase 11's own checkpoints." That claim
was checked against the actual code, not assumed to hold, and found
false: `_run_local_training_round()` (the helper Phase 11 and Phase 12
share) never saved its own `global_initial_weights.pt` -- only
`run_phase11_local_training()` did, at its own top-level checkpoint
dir, before calling the helper. Phase 12's `round_2/`/`round_3/`
subdirectories never got that file, so the diagnostic script (which
looks for exactly that filename) would have errored out immediately
against them.

**Fixed:** `_run_local_training_round()` now saves its own copy of
whatever `initial_state_dict` it was given, inside its own
`checkpoint_dir` -- making every checkpoint directory it ever writes
to, whether Phase 11's own top-level dir or any Phase 12 round,
self-contained and independently inspectable the same way. Verified
with a dedicated integration test plus a hand-performed revert→fail→
restore→pass proof, AND by actually running the diagnostic script
against a fresh full sandbox Phase 12 pipeline (not just unit-level
mocks) -- confirmed working against both `round_2/` and `round_3/`.

One thing this fix reveals that's actually better news than it first
sounds: unlike Phase 11's original real run (which predated the
`training_manifest.json` fix and so has no recoverable loss-curve
data), Phase 12's fresh rounds (2 and 3) were trained with that fix
already in place. Once pointed correctly, `check_local_training_
quality.py` shows REAL loss-trend and label-coverage statistics for
them, not just the weight-delta/NaN fallback Phase 11's own checkpoints
were limited to.

## Scope: plain FedAvg only

Stated explicitly rather than silently assumed: this phase implements
weighted parameter averaging only. No differential privacy, no secure
aggregation. Several of this project's own reference papers explore
those directions, but they represent a substantial separate research
question, not something to fold into a "confirm before building"
question for this phase. If wanted later, that's its own phase.

## What this does NOT yet do

- **No held-out evaluation.** Same caveat as Phase 11 -- no train/val/
  test split exists anywhere in this project yet. Round-over-round
  improvement is not directly measured; introducing that is its own
  future decision.
- **No adaptive/weighted round scheduling, no early stopping.** Fixed,
  config-driven `n_rounds`, decided up front. A future phase could add
  convergence-based stopping once a validation signal exists.
- **Not yet re-verified against the real 199,108-patient / 208-hospital
  dataset.** Sandbox-verified only so far.

## Verification status

| Component | Status |
|---|---|
| `fedavg_aggregate()` (weighted-average correctness, empty/missing-weight/mismatch error cases) | Verified — 10 unit tests, including a hand-performed revert→fail→restore→pass proof against an unweighted-mean regression |
| `_run_local_training_round()` extraction | Verified — Phase 11's existing 9 integration tests re-run completely unmodified, all passing, proving the extraction was behavior-preserving |
| Multi-round orchestration (round 2+ starting from the previous round's aggregate) | Verified — dedicated integration test intercepting every `load_state_dict()` call, revert→fail→restore→pass proof against a reseed-every-round regression |
| Round 1 checkpoint reuse, NaN exclusion, and fallback-to-fresh | Verified — 3 integration tests, including a hand-performed revert→fail→restore→pass proof that a corrupted reused checkpoint gets excluded, not silently aggregated in |
| Every fresh round's checkpoint dir is self-contained (`global_initial_weights.pt` present) | Verified — dedicated integration test, revert→fail→restore→pass proof, AND confirmed with the actual diagnostic script against a real sandbox run |
| `fedavg_manifest.json` persistence | Verified — dedicated integration test |
| End-to-end correctness (pipeline's aggregate vs. independently recomputed) | Verified — `test_final_aggregate_matches_direct_fedavg_computation` |
| `--start-phase 12` dependency-flag correctness | Verified — 2 tests in `test_phase_resumption.py` |
| Real 199,108-patient / 208-hospital dataset | **Ran 2026-09-02** — 3 rounds; round 1 reused (0.13s); rounds 2-3 fresh, ~32 min and ~31 min, 207 hospitals each, 1 empty, 0 non-finite. |

## Immediate next step

Real run complete. Recommended follow-up: run
`scripts/check_local_training_quality.py --checkpoint-dir
checkpoints/phase12_fedavg/round_2` and again with `round_3` --
these will show real loss-trend and label-coverage data (not just the
weight-delta fallback), since both rounds were trained with the
manifest fix already in place. The final aggregated global model is at
whatever `final_global_model_path` the run reported
(`checkpoints/phase12_fedavg/global_model_round_3.pt`).
