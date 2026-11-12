# Multi-Task Prediction Heads (Phase 10)

**Status: implemented, sandbox-verified, and real-full-scale-verified**
(2026-08-30, `sepsis_cohort=broad`). `src/heads/multitask_heads.py` is
fully tested (19 unit tests) including direct regression tests for the
output-shape squeeze, head independence, and the label-alignment
logic's storage-order and unknown-label handling -- three separate
hand-performed revert→fail→restore→pass proofs at the unit level, plus
a fourth at the phase-resumption level (see below). The full
Phase 1→2→3→4→5→7→8→9→10 pipeline is verified end-to-end against
synthetic data (6 integration tests) AND against the real 199,108-
patient / 208-hospital dataset. `sepsis_cohort=strict` remains
sandbox-verified only -- `full.yaml` doesn't override the `broad`
default.

## What Phase 10 does

Consumes Phase 9's GNN-refined per-patient embeddings (`(N,
gnn.hidden_dim)`, default 128) and produces mortality + sepsis logits
via two independent MLP heads -- `configs/base.yaml`'s `multitask:`
stanza is now actually implemented, not just a placeholder holding
`sepsis_cohort`. Architecture + forward-pass verification only, no
training loop yet (weights are randomly initialized) -- same pattern
as Phase 4's LSTM, Phase 5's autoencoder, and Phase 9's GNN.

Also ID-aligns Phase 3's `mortality`/`sepsis_df` labels against Phase
9's own patient-ID order and logs coverage/class-balance stats --
proving what Phase 11's eventual training loop will need is correctly
wired, before that loop exists. No artifact is persisted to disk yet
(unlike Phase 4/5/7/8/9's embeddings): a forward pass through
randomly-initialized heads has nothing a future phase would reuse, and
Phase 11's training loop will re-instantiate this exact model with
real trainable weights rather than reload today's random logits. This
will be revisited if a downstream phase turns out to need it.

## No hospital looping (unlike Phase 9) -- and why that's correct here

Phase 9's GNN genuinely needs each hospital's own `edge_index` for
message passing -- a real graph-structural dependency, which is why it
runs one forward pass per hospital's own graph. A prediction head is a
plain per-patient feed-forward function with zero cross-patient
interaction, exactly like Phase 4's LSTM and Phase 5's autoencoder,
both of which already run as one batched matrix operation over the
full cohort with no hospital looping at all. The heads' weights are
shared and frozen (no gradient, no aggregation) regardless of how the
forward pass is chunked, so running it per-hospital vs. in one global
call are mathematically identical here -- there's no privacy boundary
to protect by looping, only unneeded complexity. ~200K patients ×
128-dim input through two small MLPs is well under 1GB and fine in a
single call; no adaptive batching (`src/utils/batching.py`) is needed
at this scale, unlike Phase 4's LSTM, which genuinely OOM'd unbatched
at real full scale.

## Head architecture: independent small MLPs (decided 2026-08-30, confirmed with user)

`MultiTaskHeads` builds two fully separate heads -- `Linear(in_dim,
head_hidden_dim) → ReLU → Dropout → Linear(head_hidden_dim, 1)` for
EACH of mortality and sepsis -- with no parameter sharing beyond the
GNN embedding both heads take as input. Alternatives discussed before
building:

- **Plain linear probes** (`Linear(in_dim, 1)` per task, no hidden
  layer): simplest, fewest hyperparameters, but lower-capacity than
  the user wanted.
- **A shared bottleneck layer** (one `Linear → ReLU` before the two
  heads split off): classic hard-parameter-sharing multi-task design,
  but rejected in favor of full independence -- the GNN embedding
  itself is already the shared representation both tasks pull from;
  forcing it through an *additional* shared compression risked losing
  task-specific signal neither task asked to give up.
- **Independent small MLPs (chosen):** each task gets its own
  nonlinear transformation with no cross-task interference beyond the
  input embedding itself.

`multitask.head_hidden_dim` (default 64) and `multitask.head_dropout`
(default 0.1) are both config-driven, not hardcoded, per this
project's standing rule.

Output is **raw logits**, shape `(N,)` **not** `(N,1)` -- squeezed
explicitly via `.squeeze(-1)`, never bare `.squeeze()` (which would
also collapse the batch dimension on a single-patient input down to a
0-d scalar). `multitask.loss: weighted_bce | focal` both operate in
logit space, and `(N,)` vs. `(N,1)` is a common, easy-to-miss shape bug
once paired with an `(N,)`-shaped label vector during training --
directly regression-tested (see Verification status below).

## Label alignment: mortality vs. sepsis are NOT treated symmetrically for missing patients

`align_labels_to_patient_ids()` joins Phase 3's `mortality` and
`sepsis_df` onto Phase 9's own patient-ID order -- never by row
position, the same ID-based-alignment discipline every embeddings
artifact has followed since Phase 7's `align_embeddings_to_cohort()`.
But the two labels behave differently when a patient can't be found:

- **Missing from `cohort_patients` (mortality's source): hard
  `ValueError`.** `cohort_patients` is the authoritative cohort list
  every downstream phase's `patient_ids` derive from -- a gap here
  means real ID corruption somewhere upstream, not a sparse-data edge
  case, and should stop the run.
- **Missing from `sepsis_df`: `NaN` (unknown), not an error.** Found
  via this phase's own integration test against mock data, root-caused
  before deciding how to handle it: `compute_infection_proxy()`'s
  output only covers patients with ≥1 medication/microLab row, and
  `compute_sofa_like_scores()`'s only covers patients with an
  `apachePredVar` row. `compute_sepsis_labels()`'s own inline comment
  already documents the *one-sided* version of this gap as "a real,
  common situation in eICU (~468/4064 patients in one real run)" -- a
  patient absent from *both* sides (confirmed with mock data: one
  patient with zero rows across all three tables) is the same
  phenomenon taken one step further, not a different problem. Treated
  exactly like `mortality_label()`'s own precedent for an unrecognized
  `hospitaldischargestatus`: unknown, not silently coded as
  sepsis-negative. `sepsis` is therefore returned as a float array
  (`1.0` / `0.0` / `NaN`), not bool.

**Real-data prevalence: confirmed 2026-08-30.** 4,777/199,108 (2.40%)
of the real full cohort hits the both-sides-missing condition -- lower
than the ~11.5% one-sided rate `compute_sepsis_labels()`'s comment
documented from an earlier real run, consistent with "both sides
missing" being rarer than "one side missing." These patients get
`sepsis=NaN`, not silently coded as sepsis-negative. The real
`[LABELS]` log line:
```
[LABELS] mortality: known=199108 positive=18004 negative=181104 unknown=0 |
  sepsis_broad: known=194331 positive=44412 negative=149919 unknown=4777
```
`sepsis_broad_positive=44412` here exactly matches the count Phase 3's
own real full-scale run recorded independently (2026-08-10) -- direct,
real-scale confirmation that the ID-based join preserves every
sepsis-positive label with zero loss or duplication despite Phase 9's
embeddings being in hospital-grouped order, not Phase 3's own cohort
order.

`multitask.sepsis_cohort` (`broad | strict`) is validated to be
exactly one of those two strings inside `align_labels_to_patient_ids()`
-- raises `ValueError` on anything else, including an unset/`None`
config value. Never silently defaults or guesses, per the locked
requirement from Phase 3's dual-cohort design.

## What this does NOT yet do

- **No training loop.** This phase verifies architecture and
  forward-pass mechanics only, plus that the labels needed for
  training are correctly ID-aligned. Actual gradient-based training
  against `mortality`/`sepsis_broad`/`sepsis_strict` is Phase 11's job.
- **No persisted output.** Unlike Phase 4/5/7/8/9, nothing is written
  to `outputs/embeddings/` for this phase yet -- see "What Phase 10
  does" above for why. `--start-phase 10` still works (loads Phase 3's
  and Phase 9's already-persisted results), but there's currently
  nothing to `--start-phase 11` from.
- **No FedAvg yet.** Same caveat as every phase before it -- the heads
  are one shared `MultiTaskHeads` instance applied identically
  regardless of hospital, but there's no training loop yet for weights
  to actually be averaged across hospitals after local updates.
- **`sepsis_cohort=strict` not yet run against real data.** `full.yaml`
  doesn't override the `broad` default, mirroring exactly how Phase
  9's `gat`/`gcn` remained sandbox-only for the same reason.

## Verification status

| Component | Status |
|---|---|
| `MultiTaskHeads` (construction validation, output shape/dtype, single-patient and empty-input edge cases, head independence) | Verified — 9 unit tests, including a hand-performed revert→fail→restore→pass proof for the `(N,)` vs `(N,1)` shape bug and a second for head independence |
| `align_labels_to_patient_ids()` (ID-based join, storage-order mismatch, sepsis_cohort validation, mortality-unknown vs. sepsis_df-absence handling) | Verified — 10 unit tests, including a hand-performed revert→fail→restore→pass proof for the sepsis_df-absence-is-not-an-error fix |
| Full Phase 1→2→3→4→5→7→8→9→10 pipeline on mock data | Verified — 6 integration tests (`test_phase10_pipeline.py`), including a direct proof that the real sparse-data/sepsis_df-coverage condition doesn't crash the run |
| `--start-phase 10` dependency-flag correctness | Verified — 2 tests in `test_phase_resumption.py`, including a hand-performed revert→fail→restore→pass proof reproducing the exact `AttributeError` this dependency-flag bug class has produced before (see Phase 9's own `--start-phase` history) |
| Full pipeline on the real 199,108-patient / 208-hospital dataset (`sepsis_cohort=broad`) | **Verified 2026-08-30** — `mortality_logits`/`sepsis_logits` both `(199108,)`, 2.48s, `sepsis_broad_positive=44412` exactly matching Phase 3's independently-recorded real count |

## Immediate next step

Phase 10 is closed out for the default configuration. Remaining,
non-blocking: run `--config configs/full.yaml` with `multitask.
sepsis_cohort: strict` overridden, the same way Phase 9's `gat`/`gcn`
still await a real run. Otherwise, next up is Phase 11 (local training
loop), which will consume these heads' architecture with real
gradient-based training against the `mortality`/`sepsis_broad`/
`sepsis_strict` labels this phase already proved are correctly wired.
