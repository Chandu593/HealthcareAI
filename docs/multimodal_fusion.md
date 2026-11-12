# Multimodal Fusion (Phase 7)

**Status: implemented, sandbox-verified.** `src/fusion/multimodal_fusion.py`
is fully tested (11 unit tests) including a direct regression test for
the core correctness risk (see "Why ID-based alignment, not row order"
below) and a hand-performed revert→fail→restore→pass proof. The full
Phase 1→2→3→4→5→7 pipeline is verified end-to-end against synthetic
data (4 integration tests). **Not yet run against the real dataset** --
this is the immediate next step once delivered.

Phase 6 (text) is intentionally skipped throughout -- locked decision,
no clinical notes in this eICU export. See `docs/text_modality_decision.md`.

## What Phase 7 does

Concatenates Phase 4's LSTM time-series embeddings `(N, 128)` with
Phase 5's autoencoder tabular embeddings `(N, 64)` into one unified
per-patient representation `(N, 192)`, saved to
`outputs/embeddings/fused_embeddings.npz` (`patient_ids`, `embeddings`,
plus `lstm_dim`/`autoencoder_dim` for traceability).

## Fusion type: pure concatenation, no projection layer (Option A, locked 2026-08-25)

`configs/base.yaml`'s `fusion.type: concatenation` is implemented
literally -- no learned projection layer reduces or expands the
192-dim concatenated vector to some other target size. This was a
genuine design choice, not an oversight: `fusion.output_dim` already
existed in config (previously `256`, a stale placeholder from early
planning before Phase 6/text was permanently locked out -- corrected
to `192` here, matching the real, now-confirmed `128 + 64` sum). Two
options existed:

- **Option A (chosen): pure concatenation.** No new trained parameters.
  Consistent with Phase 4/5, which also verify architecture and
  forward-pass mechanics only -- neither has a training loop yet, so
  there's no training signal yet to justify committing to a specific
  projected width.
- **Option B (not chosen): concatenate then project** via a learned
  linear layer to a fixed `output_dim`. Deferred until there's an
  actual reason (e.g. a specific downstream input-width requirement,
  or training becoming available) to justify an untrained layer's
  dimensionality.

`fusion.output_dim` in config is now **descriptive, not enforced** --
`run_phase7_multimodal_fusion()` logs a warning (not an error) if it
ever drifts from the real `lstm_dim + autoencoder_dim` sum, rather than
silently ignoring the mismatch or crashing over a non-binding
config value.

## Why ID-based alignment, not row order

Phase 4 and Phase 5 currently both derive their patient population from
the same `phase3_result['cohort_patients']` list, in the same order,
and neither drops/filters patients afterward -- so in practice, their
saved embeddings' `patient_ids` arrays already match, row for row. This
module does **not** rely on that holding by convention alone.

`align_embeddings_to_cohort()` re-indexes each modality's embeddings by
an explicit patient-ID lookup against a target order (the cohort's own
order), rather than assuming the two source arrays are already
aligned. If it assumed row-order alignment instead (a plain
`np.concatenate([lstm_embeddings, ae_embeddings], axis=1)`), a future
change to either modality's save order -- or a version skew between
cached Phase 4/5 outputs from different runs -- would silently fuse
*different patients'* data together. The fused matrix would have the
right shape and no crash, making this exactly the class of bug that's
easy to ship and hard to detect after the fact.

`tests/unit/test_multimodal_fusion.py::TestFuseEmbeddings::test_misaligned_storage_order_still_fuses_correctly`
targets this directly: it stores the two modalities' embeddings in
*deliberately different* patient orders and asserts each patient's
fused row still equals *their own* LSTM value concatenated with *their
own* autoencoder value. Reverting `align_embeddings_to_cohort()` to a
naive positional pass-through (proven by hand -- see
IMPLEMENTATION_TRACKER.md) makes 7 of 11 tests fail immediately,
confirming this suite would actually catch a regression here, not just
happen to pass against currently-aligned data.

If a patient is missing from either modality's embeddings entirely,
`align_embeddings_to_cohort()` raises rather than silently zero-filling
or dropping that patient -- Phase 4 and Phase 5 are expected to cover
the exact same cohort, so a genuine mismatch means something upstream
diverged and deserves investigation, not silent masking.

## What this does NOT yet do

- **No learned fusion/projection.** See "Fusion type" above -- Option A
  was chosen deliberately, not a placeholder for "not implemented yet."
  Revisit if/when training exists and a specific downstream width is
  actually needed.
- **No handling for a THIRD modality.** The code is deliberately
  two-modality-specific (`fuse_embeddings()`'s signature takes exactly
  an LSTM and an autoencoder source) rather than a generic N-way
  fusion, since text (the only other planned modality) is permanently
  out of scope for this eICU export.
- **Not yet re-verified against the real 199,108-patient / 208-hospital
  dataset.** Sandbox-verified only so far, same caveat as every new
  phase before its first real run.

## Verification status

| Component | Status |
|---|---|
| `multimodal_fusion.py` (ID-based alignment, concatenation, missing-patient handling) | Verified — 11 unit tests (`test_multimodal_fusion.py`), including a direct misaligned-storage-order regression test and a hand-performed revert→fail→restore→pass proof (7 of 11 tests fail against a naive positional-pass-through revert) |
| Full Phase 1→2→3→4→5→7 pipeline on mock data | Verified — 4 integration tests (`test_phase7_pipeline.py`), including a direct row-level correctness check (fused row == that patient's own LSTM row + their own autoencoder row, looked up independently by ID from each source file) and a check that a missing embeddings path raises clearly rather than fusing a partial result |

## Immediate next step

Run `python run_pipeline.py --config configs/full.yaml --phase 7` on
the real dataset and confirm the `[PHASE7] Fused embeddings ...`
log line reports shape `(199108, 192)`.

**CONFIRMED on the real 199,108-patient / 208-hospital dataset
(2026-08-26):**
```
[PHASE7_MULTIMODAL_FUSION] START
[PHASE7] Fused embeddings ((199108, 192)) written to outputs\embeddings\fused_embeddings.npz
[PHASE7_MULTIMODAL_FUSION] COMPLETE (11.25s)
```
Exact expected shape, clean run, no `fusion.output_dim` mismatch
warning (confirming the corrected config value now genuinely matches).
Phase 7 is DONE in the fullest sense: implemented, sandbox-verified,
and real-full-scale-verified.
