# Memory Efficiency (Phase 3/4 rework, 2026-08-09)

## The crash this fixes

Running Phase 3 against the real eICU dataset in `full` mode (all
199,108 eligible patients, no per-hospital cap) crashed with:

```
numpy._core._exceptions._ArrayMemoryError: Unable to allocate
123. MiB for an array with shape (16124697,)
```

inside `combine_medication_sources()`'s `drop_duplicates()` call.

**Root cause:** in `debug`/`subset` mode, filtering each large table to
"the sampled patients" is meaningful (a few thousand out of 200K+). In
`full` mode it barely filters anything — nearly every patient is
eligible — so the original approach of loading each large table once,
filtered to the whole cohort, and holding it as one pooled in-memory
DataFrame doesn't scale. `medication` (7.3M rows) +
`medication_imputed` (1.67M) + `medication_imputed1` (7.3M) ≈ 16.3M
rows, matching the crash's array shape almost exactly — the entire
combined medication history for the whole dataset was being held in
memory at once before deduplication.

## The fix: hospital-wise sharding

`src/data/ingestion/hospital_sharding.py` reads each large source table
**once**, in chunks (never held fully in memory), projecting only the
columns actually needed downstream (`SHARD_COLUMNS` — e.g. `lab`'s 10
columns become 4), and routes each chunk's rows into small per-hospital
CSV shard files on disk. Phase 3 and Phase 4 then loop hospital-by-
hospital, loading only one hospital's already-small shard at a time.

```
Large source CSV (39M rows, 9.7GB)
        |
   ONE chunked pass, column-projected
        |
   route each chunk's rows by patientunitstayid -> hospitalid
        |
   +----+----+----+----+
   |    |    |    |    |
 hosp1 hosp2 hosp3 ... hosp208   (small CSV shard files, cached)
   |    |    |    |    |
   +----+----+----+----+
        |
Phase 3 / Phase 4 loop: process ONE hospital's shard at a time
```

**Peak memory is now bounded by the largest single hospital's data, not
the whole dataset.** On the real hospital-size distribution the user's
own run showed, the largest hospital (`hospitalid=73`) has 7,059
patients vs. 199,108 total — a ~28x reduction before column projection
is even counted.

This also happens to be a closer match to the federated architecture's
own "Hospital-wise Data Separation" step than pooling ever was.

## What changed structurally

- `run_phase3_cohort_construction()` no longer loads `lab` /
  `medication` (+ imputed variants) / `infusionDrug` / `microLab` as
  pooled DataFrames for the whole cohort. It shards them once, then
  loops hospitals, calling the *same, unchanged* `compute_sofa_like_scores()`
  and `compute_infection_proxy()` on each hospital's small slice, and
  concatenates the (much smaller) per-patient **label** results — not
  raw event data — into the final cohort report.
- `run_phase4_timeseries_modality()` similarly loops hospitals for
  binning. `time_binning.py`, `sequence_builder.py`, and
  `normalization.py` needed **zero changes** — binned output is already
  small (at most patients × max_bins × features rows, not
  patients × raw_events), so accumulating small per-hospital binned
  frames and concatenating them before building the sequence tensor was
  sufficient. Only *how* the binning functions get their per-hospital
  input changed, not the functions themselves.
- `apachePredVar` (171,177 rows / ~75MB dataset-wide) is small enough
  to stay pooled — no sharding needed, just column projection (13 of
  its 51 columns).

## Caching (Section 26)

Shards are cached under `cache/hospital_shards/<table>/`, keyed by
source file mtime+size, the column projection, and the exact cohort
patient set (`hospital_sharding._cache_key()`). Re-running the pipeline
with the same data and cohort reuses existing shards instead of
re-reading a 9.7GB CSV again. The cache key changes — forcing a
re-shard — if the source file, the columns needed, or the sampled
patient set changes. Phase 4 reuses Phase 3's shards from the same run
(cache hit) rather than re-sharding; verified end-to-end in
`tests/integration/test_full_mode_scale.py`.

## Why CSV shards, not Parquet

Parquet would likely shrink shard files and speed up re-reads further.
It's not implemented here because neither this development sandbox nor
(as of this writing) the user's machine has confirmed `pyarrow`
installed, and an untested Parquet writer path risks a subtle
schema-inference bug silently corrupting data on a 39M-row real file —
worse than the original crash, which at least failed loudly. CSV
sharding is fully tested here. A genuine follow-up opportunity once
Parquet's real-data behavior can actually be verified, not a silent gap.

## Verification status

| Component | Status |
|---|---|
| `hospital_sharding.py` (routing, column projection, caching, cache invalidation, empty-hospital handling) | Verified — 8 unit tests |
| Full Phase 1→4 pipeline with an uneven hospital-size distribution (one large hospital, several small, forced multi-chunk reads via a tiny chunksize) | Verified — 2 integration tests (`test_full_mode_scale.py`) |
| Cache reuse across Phase 3 → Phase 4 in the same run | Verified — explicit mtime-unchanged assertion |
| **Actual memory usage on the real 199,108-patient dataset** | **Verified 2026-08-10** — the user's `full.yaml` run completed Phase 3 in 557s across all 199,108 patients / 208 hospitals with no crash. Peak memory bounded as designed. |

## What this does NOT yet fix

- `infection_proxy.compute_infection_proxy()`'s strict-corroboration
  step still does a per-patient antibiotic × culture pairwise merge
  (`abx.merge(micro_lab, on="patientunitstayid")`). This is now bounded
  by a single hospital's patient count instead of the whole dataset, so
  the worst case is far smaller, but a hospital with unusually many
  antibiotic orders and culture entries for the same patients could
  still see a locally large pairwise expansion. Not observed as a
  problem in any run so far; flagged as a known remaining scaling edge
  case rather than assumed fixed.
- Phase 1's data-report generation is unchanged (already chunked, and
  it didn't crash in either real run — just slow, ~4-8 minutes for the
  full dataset). Not touched in this rework since it wasn't broken.

## Update 2026-08-09: real full-mode run — second crash, found and fixed

The user ran `full.yaml` (all 199,108 patients) against the real
dataset. **Phase 3's sharding fix worked** — it completed in ~861s with
no crash, a real confirmation the architecture above holds at true
scale, not just in this sandbox's tests. **Phase 4 then crashed**:

```
RuntimeError: [enforce fail at alloc_cpu.cpp:117] data.
DefaultCPUAllocator: not enough memory: you tried to allocate
5780424704 bytes.
```

**Root cause:** Phase 4's LSTM forward pass had never been batched —
all 199,108 patients were pushed through the LSTM in a single call. An
LSTM's internal working memory (gate/hidden-state buffers) scales with
*total timesteps across the batch* (here, ~2.8M — every patient's
sequence length summed), not just the input tensor's own size (which
was a modest ~420MB on its own). A back-of-envelope estimate
(total_timesteps × hidden_dim × num_layers × 4_gates × 4 bytes) lands
at ~11.6GB — the right order of magnitude for the ~5.78GB the allocator
actually failed on.

**Fixed via `src/utils/batching.py`:** a generic, torch-agnostic
adaptive-batch-size runner with automatic backoff. On a retryable
error (OOM), it halves the batch size and retries the *same* range —
no progress lost, no range skipped or double-processed. Phase 4 now
processes the LSTM in batches of `modalities.timeseries.lstm_inference_batch_size`
(default 512 — roughly 400x smaller than the batch that failed) with
this safety net if even that turns out to be too large on a given
machine. Fully unit-tested (14 tests) without needing torch, since the
batching/retry logic itself has zero torch dependency — only the
`process_batch` callback Phase 4 supplies touches torch.

**Also fixed while auditing:** `medication_imputed1` was mislabeled
`large_table=False` in the schema registry (an artifact of an early,
ambiguous sample of the data — its real size, confirmed by this same
run, is ~7.3M rows/3.27GB, not the ~1.67M rows originally guessed).
This made Phase 1 do a risky full (non-chunked) read of a 3.27GB file.
It never affected Phase 3/4's correctness since `hospital_sharding.py`
always chunks regardless of this flag — but it's fixed now for Phase
1's sake too.

**Also added:** progress logging inside Phase 3's and Phase 4's
per-hospital loops. The real run went silent for ~9 minutes (Phase 3)
and ~16 minutes (Phase 4) with zero log output during those loops —
indistinguishable from a hang. Both now log every 20 hospitals (or on
completion) with elapsed time and an ETA estimate.

**Verification status of this update:**

| Component | Status |
|---|---|
| `run_batched_with_backoff` / `is_memory_error` (batching, retry, backoff logic) | Verified — 14 unit tests, including a simulated OOM-then-succeed scenario and a "never succeeds, raises cleanly" scenario |
| Phase 3 progress logging | Verified — live run shows correct `N/M hospitals` output |
| Phase 4 progress logging | Verified — live run shows correct `N/M hospitals` output |
| `medication_imputed1` registry fix | Verified — `TABLE_REGISTRY['medication_imputed1'].large_table` now `True`, confirmed by the user's Phase 1 log now showing `chunked read` instead of the old risky `full read` |
| **The actual batched LSTM forward pass on real data** | **Verified 2026-08-10** — the user's `full.yaml` re-run completed Phase 4 end-to-end. `batch_size=512` held constant for all 199,108 patients (zero backoff events needed), 54s total inference time, producing real `(199108, 128)` embeddings. This specific forward pass still cannot be executed in this development sandbox (no torch/internet here), but it no longer needs to be — it's been run for real, on the real dataset, at true full scale. |

## Update 2026-08-10: real full-mode run succeeded end-to-end

The user re-ran `full.yaml --phase 4` after the fixes above. **Complete
success, zero errors, all 4 phases:**

```
Phase 1: 447s   Phase 2: 4s   Phase 3: 557s   Phase 4: 564s
Total: ~26 minutes for 199,108 patients / 208 hospitals
```

Sepsis counts (`sepsis_strict_positive=262`, `sepsis_broad_positive=44412`)
matched the prior crashed attempt exactly, confirming determinism held
across the sharding/batching rework. Progress logging worked as
designed throughout (`N/207 hospitals` updates with accurate ETAs in
both Phase 3 and Phase 4's binning loop; `N/199,108` updates during
LSTM inference). The LSTM batch size never needed to shrink below the
512 default — confirming that default is both safe (large margin below
the ~5.78GB failure point) and fast (54s for the full cohort).

This closes out the memory-efficiency work for Phases 1-4. No further
action needed here unless a future phase (5+) introduces a new
dataset-scale operation.
