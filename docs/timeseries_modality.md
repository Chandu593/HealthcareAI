# Time-Series Modality (Phase 4)

**Status: implemented and verified at real full scale.** Preprocessing
(`src/data/temporal/`) is fully tested against synthetic data (29 unit
tests + 2 integration tests). The LSTM encoder
(`src/modalities/timeseries/`) cannot be executed in this development
sandbox (no torch, no internet to install it), but **has been executed
for real, on the user's own machine, against the real full 199,108-
patient / 208-hospital eICU dataset (2026-08-10)** — producing real
`(199108, 128)` embeddings with the batched-inference fix holding
`batch_size=512` constant throughout (zero backoff events, 54s total).
See "Verification status" below.

## Why lab + medication + infusionDrug, not vital signs

The architecture doc's Phase 4 section lists "vital signs" as a
candidate source. This eICU export has no `vitalPeriodic`,
`vitalAperiodic`, or `nurseCharting` (see `schema_registry.py`), so
there is no repeated-measurement vitals table to build a time series
from. `lab`, `medication`, and `infusionDrug` are the only genuinely
timestamped, repeated-observation tables available, so they're the
substitute — a modification from the architecture doc's literal
wording, not a silent omission (Section 4 of the master prompt requires
this be documented, not assumed away).

## Pipeline

```
lab, medication, infusionDrug
        |
time_binning.py    -- fixed grid, bin_minutes wide (default 240 = 4h),
        |              offset 0 = ICU admission; negative offsets
        |              (pre-admission) and offsets beyond max_bins are
        |              dropped, not folded into edge bins
        |
sequence_builder.py -- (patients, max_bins, features) array + a
        |               real-vs-padding sequence_length per patient
        |               (from unitdischargeoffset, NOT from whether
        |               data happened to be observed -- see that
        |               module's docstring for why conflating the two
        |               would leak discharge timing through padding)
        |
normalization.py    -- StandardScaler fit PER HOSPITAL on that
        |               hospital's own real bins only (never global
        |               statistics -- matches the federated privacy
        |               boundary and the "Local Hospital Preprocessing"
        |               architecture step); missing values imputed to
        |               the hospital's own per-feature mean; padding
        |               bins zeroed after standardization
        |
lstm_encoder.py      -- torch LSTM, pack_padded_sequence over the real
                        sequence_length, last-layer final hidden state
                        as the patient embedding
```

## Feature set

- **Lab** (`configs/base.yaml: modalities.timeseries.lab_features`):
  mean value per bin for labnames case-insensitive-substring-matching a
  configured list (default: platelet, bilirubin, creatinine,
  fibrinogen, PT - INR, pH — the labnames actually seen in this
  project's sample data). If none of those substrings match anything in
  a given cohort's real data, `select_lab_features()` automatically
  falls back to the top-6 most frequent labnames observed and logs a
  warning — a genuinely empty feature block is a worse failure mode
  than a slightly different (but real) feature set.
- **Medication**: `medication_n_active` (count of orders overlapping
  the bin) and `medication_antibiotic_active` (0/1), computed via
  interval overlap so a multi-bin-spanning order is marked active in
  every bin it actually covers, not just its start bin.
- **InfusionDrug**: `infusion_n_active`, `infusion_vasopressor_active`,
  `infusion_vasopressor_rate_mean` — point-binned (one row = one
  timestamped rate recording), not interval-exploded like medication,
  because this table has no stop offset — it's periodic recordings, not
  ordered intervals.

## Architecture options (added 2026-09-06, by user request)

`modalities.timeseries.encoder` selects which architecture Phase 4
uses -- this config key already existed as an unused stub (nothing in
the code ever read it) before this change; it now actually does
something.

| `encoder` value | Output file | Notes |
|---|---|---|
| `lstm` (default) | `timeseries_embeddings.npz` | **Unchanged code path.** `LSTMTimeSeriesEncoder` itself was never modified by this addition. |
| `rnn` | `timeseries_embeddings_rnn.npz` | Vanilla `nn.RNN`, same pack_padded_sequence handling as LSTM. |
| `gru` | `timeseries_embeddings_gru.npz` | `nn.GRU`, same pattern. |
| `transformer` | `timeseries_embeddings_transformer.npz` | `nn.TransformerEncoder` with a fixed sinusoidal positional encoding; padded timesteps excluded via `src_key_padding_mask` (the Transformer-native equivalent of `pack_padded_sequence`, which has no direct analogue for attention). New config: `transformer_num_heads` (default 4), `transformer_dim_feedforward` (default 256). |

**The canonical `timeseries_embeddings.npz` is never touched by any
value other than `lstm`** -- proven, not just intended: a dedicated
integration test (`test_canonical_lstm_file_is_byte_for_byte_untouched_
by_alternate_architecture_runs`) confirms the file's bytes AND mtime
are unchanged after running `rnn`/`gru`/`transformer`, and a
hand-performed revert→fail→restore→pass proof confirmed that test
actually catches a regression (temporarily made every encoder write to
the canonical filename; the test failed with the corrupted file
containing RNN output; restored, reconfirmed passing).

All three new encoders share LSTM's exact `forward(x, lengths) ->
(batch, hidden_dim)` contract and, by default, its exact output width
(`hidden_dim`, the same shared config key) -- this is what keeps
Phase 7's fusion (and everything after it) completely oblivious to
which architecture produced the embeddings it's fusing. Confirmed by
running the full Phase 1→7 pipeline with each new encoder selected:
all fuse correctly with Phase 5's embeddings at the expected combined
width.

**These are additive, standalone outputs for exploration/comparison.**
`run_phase11_local_training()`/Phase 12/Phase 13's actual TRAINING
code never reads `embeddings_path` at all -- it constructs its own
fresh `LSTMTimeSeriesEncoder` + `TabularAutoencoder` directly from the
raw normalized tensors regardless of what `encoder` was set to for a
Phase 4/5 standalone run. Selecting `rnn`/`gru`/`transformer` here does
**not** change what Phase 11/12/13 train with. See
`docs/local_training.md` if extending the actual training pipeline to
support architecture selection is wanted later -- that's a
meaningfully bigger change (touching `LocalPatientModel`, already
real-full-scale-verified) than what was asked for here.

## Known limitations (tracked, not hidden)

- **Medication active-interval binning uses an explode-based approach**
  (one row per (drug, overlapping bin) pair) that is fine at
  debug/subset scale but should be rewritten to avoid materializing
  that many rows before running at full-eICU scale.
- **Lab feature coverage in the debug/mock run was low (~5.6% of
  tensor cells observed)** — expected given the small mock cohort's
  event density (5-15 lab rows spread across up to 48 bins x 11
  features per patient), not a bug; real eICU cohorts will likely be
  denser but this should be checked against real data, not assumed.
- **Missing-value imputation is a simple per-hospital mean fill**, with
  no explicit missingness indicator passed to the LSTM. A patient
  scoring "average" on a feature because it was never measured is
  indistinguishable, post-imputation, from one who was actually
  measured and happened to be average. Adding an explicit
  observed-mask channel alongside the value channel is a natural
  follow-up if this turns out to matter for model performance.

## Verification status

| Component | Status |
|---|---|
| `time_binning.py` | Verified — 13 unit tests |
| `sequence_builder.py` | Verified — 8 unit tests |
| `normalization.py` | Verified — 4 unit tests, including the per-hospital-isolation invariant |
| Full Phase 1→4 pipeline on mock data | Verified — 2 integration tests, live CLI run inspected by hand |
| `lstm_encoder.py` (forward pass mechanics — shape, padding invariance) | **Not verified in this sandbox specifically** — torch not installed here, no internet to install it. Tests are written (`tests/unit/test_lstm_encoder.py`) and `unittest.skip`-marked with an explicit reason here. Superseded in practice by the real full-scale run below — the forward pass has genuinely run, correctly, on real data. |
| **LSTM inference at real full-scale (199,108 patients)** | **Verified 2026-08-10.** First attempt (2026-08-09) crashed with a CPU allocator OOM because inference was never batched. Fixed via `src/utils/batching.py` (14 unit tests, torch-agnostic). Re-run 2026-08-10 completed cleanly end-to-end: `batch_size=512` held constant for all 199,108 patients (zero backoff events), 54s total, producing real `(199108, 128)` embeddings. See `docs/memory_efficiency.md`'s "Update 2026-08-10" section. |
| RNN / GRU / Transformer encoders (added 2026-09-06) | Verified — 21 unit tests (shape, single-patient, NaN/Inf, gradient flow, and for Transformer specifically, a hand-performed revert→fail→restore→pass proof that the padding mask genuinely excludes padded timesteps) + 9 integration tests confirming correct output filenames, the canonical-file-untouched guarantee (with its own revert→fail→restore→pass proof), and matching `hidden_dim` across all four encoders. Full existing suite (345 tests before these, 354 after) re-run and passing, confirming zero regressions anywhere else in the pipeline. **Not yet run against real full-scale data.** |

## To get `lstm_encoder.py`'s own unit tests running too

The forward pass is now verified in practice (above), but
`tests/unit/test_lstm_encoder.py` is still worth running on the user's
machine for the finer-grained checks (exact output shape, padding
invariance) that the full pipeline run doesn't isolate on their own:

```bash
python -m unittest tests.unit.test_lstm_encoder -v
```
