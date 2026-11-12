# Methodology Report: Per-Hospital Data Partitioning, Modality Construction, and Sequential Patient Trajectory Modeling

**Scope of this report:** how the 13 raw eICU tables are partitioned by
hospital, classified into time-series vs. tabular sources, transformed
into a fixed-shape representation per patient, and processed through
the LSTM to produce one embedding per patient's temporal trajectory.
Grounded entirely in the implemented, tested, and (for Phases 1-4)
real-full-scale-verified codebase — no aspirational description of
unbuilt components without saying so explicitly.

**Verification basis:** all mechanics below are described as actually
implemented in this repository and confirmed against the real eICU
dataset (200,859 patients, 208 hospitals) on 2026-08-10 — the numbers
cited throughout (bin counts, feature counts, timing) are from that
real run, not estimates.

---

## 1. Executive summary

The pipeline takes a raw, multi-table, multi-hospital EHR export and
produces one fixed-length embedding vector per ICU stay, suitable as
input to a graph neural network in a later phase. It does this in six
stages:

```
13 raw eICU tables
        |
[1] Hospital-wise cohort partitioning        (Phase 2)
        |
[2] Table classification into modalities      (design, Section 2)
        |
[3] Memory-efficient per-hospital sharding     (Phase 3/4 infrastructure)
        |
[4] Temporal feature extraction (binning)      (Phase 4a)
        |
[5] Fixed-shape sequence tensor assembly       (Phase 4b)
        |
[6] Per-hospital local normalization           (Phase 4c)
        |
[7] Sequential processing through the LSTM     (Phase 4d)
        |
One embedding vector per patient
```

Stages 1-6 are table/array manipulation — no learning happens there.
Stage 7 is where the LSTM actually reads a patient's trajectory
timestep by timestep. That stage is the one your question is really
about, and it's covered in the most depth (Section 8).

**One thing to flag before the detail:** the LSTM currently has
**randomly initialized weights** — there is no training loop yet (that
is Phase 11 in the tracker). Everything described below about the LSTM
"learning" a trajectory describes what the *architecture* is designed
to capture once trained, and confirms the *mechanics* execute correctly
at real scale. It does not mean the embeddings are yet predictive of
mortality or sepsis. This distinction is restated in Section 9 because
it's easy to blur in a methods write-up if it isn't stated plainly.

---

## 2. Classifying the 13 tables into modalities

Not every table becomes a model input. Some construct labels, some are
excluded as leakage risks, one is site metadata. The classification
below is enforced in code (`schema_registry.py`,
`configs/base.yaml`), not just a documentation convention.

| Table | Role | Where it's used |
|---|---|---|
| `lab` | **Time-series source** | Phase 4 — repeated, timestamped measurements |
| `medication` | **Time-series source** + label construction | Phase 4 (active-drug/antibiotic signal) + Phase 3 (sepsis infection proxy) |
| `medication_imputed` | Time-series source (unioned with `medication`) | Phase 4 + Phase 3 |
| `medication_imputed1` | Time-series source (unioned with `medication`) | Phase 4 + Phase 3 |
| `infusionDrug` | **Time-series source** | Phase 4 — vasopressor exposure over time |
| `patient` | **Tabular source** + cohort base + mortality label | Phase 2 (partitioning), Phase 3 (label), Phase 5 (implemented) |
| `apachePredVar` | **Tabular source** + organ-dysfunction scoring | Phase 3 (SOFA-like scoring), Phase 5 (implemented) |
| `diagnosis` | **Tabular source** | Phase 5 (implemented) |
| `hospital` | Site metadata / federated client registry | Phase 2 |
| `microLab` | Label construction only (infection corroboration) | Phase 3 |
| `admissionDrug` | Not yet consumed | reserved |
| `treatment` | Not yet consumed | reserved (candidate corroborating signal for organ support) |
| `apachePatientResult` | **Excluded from all model inputs** — leakage risk | Baseline comparison only; flagged automatically at ingestion |

Why this split, in a sentence per category:

- **Time-series tables** (`lab`, `medication`+variants, `infusionDrug`)
  are the only tables with repeated, timestamped observations per
  patient — everything else is a single value or a handful of values
  captured once. This is a direct consequence of what's actually in
  this eICU export: there is no `vitalPeriodic` table, so labs,
  medications, and infusions are the entire temporal signal available.
- **Tabular tables** (`patient`, `apachePredVar`, `diagnosis`)
  characterize the patient largely at a single point in time
  (admission demographics, day-1 severity scores, diagnosis codes).
  Feature engineering and per-hospital normalization are implemented
  and tested (Phase 5, added 2026-08-11) — see
  `docs/tabular_modality.md` for the full leakage-prevention rationale
  and feature allowlist. The autoencoder itself is written but
  unverified in this sandbox (no torch here), same status the LSTM had
  before the user's real-data runs confirmed it.
- **`apachePatientResult`** contains the *outcome itself* in disguised
  form (predicted/actual mortality, APACHE score derived from the same
  physiology used to define severity) — using it as an input feature
  would let the model see a proxy for the label. It's ingested only for
  benchmarking APACHE's own predictions against this project's model
  later, never as a feature. `data_report.py` flags this automatically
  at Phase 1 (`[LEAKAGE_RISK]` in the logs) rather than relying on
  someone remembering not to touch it.

---

## 3. Stage 1 — Hospital-wise cohort partitioning

Before any feature engineering happens, every patient is assigned to
exactly one hospital bucket, and that assignment is never crossed
later. This is the foundational federated-learning boundary: a
hospital's data must never mix with another hospital's data at any
processing stage, because in a real federated deployment each hospital
would be a separate physical site with data that never leaves its own
infrastructure.

Mechanically (`src/data/cohort/hospital_partition.py`):

1. Read `patient` + `hospital`, intersect on `hospitalid` to get the
   set of hospitals that actually have data.
2. For each hospital, compute the mortality label
   (`patient.hospitaldischargestatus`) for every patient at that
   hospital; a patient with an unrecognized status value is excluded
   from that hospital's *eligible* set (not silently kept).
3. If a patient cap is configured (`cohort.patient_subset`), sample
   `min(cap, eligible_count)` patients per hospital with a fixed seed
   — never more than a hospital actually has, and never silently
   raising an error if a hospital is small (a real bug in an earlier
   prototype version of this logic).

On the confirmed real run: **208 hospitals**, **199,108 total eligible
patients** in `full` mode (no cap), ranging from `hospitalid=361` with
0 eligible patients (all 13 of its patients had unrecognized discharge
statuses — correctly excluded) up to `hospitalid=73` with **7,058**
eligible patients, the largest single hospital in the dataset.

From this point forward, every downstream stage operates **one
hospital's patient set at a time** — this is what Section 4 below
exists to make computationally tractable.

---

## 4. Stage 2 — Memory-efficient per-hospital sharding

This stage doesn't do any feature engineering; it exists purely so
that Stage 3 can be run per hospital without repeatedly re-reading a
multi-gigabyte source file, and without ever holding more than one
hospital's raw data in memory at once (`src/data/ingestion/hospital_sharding.py`).

For each large time-series source table:

1. Read the source CSV **once**, in chunks (never fully in memory).
2. Project down to only the columns actually needed downstream — e.g.
   `lab`'s 10 columns become 4 (`patientunitstayid`,
   `labresultoffset`, `labname`, `labresult`).
3. For each chunk, look up each row's `hospitalid` via a
   patient→hospital map (built once from the small `patient` table),
   and append the row to that hospital's own small shard file on disk.
4. Cache the resulting shard manifest, keyed on the source file's
   identity + the column projection + the exact patient cohort, so a
   re-run with the same data and cohort reuses the shards instead of
   re-reading the source file.

On the confirmed real run, this pass took ~3.5 minutes for `lab`
(38.7M matching rows across 206 hospitals) and correspondingly less for
the smaller tables. After this one-time cost, every hospital's data is
sitting in its own small file, ready for Stage 3 to read in isolation.

**Why this matters beyond performance:** before this stage existed,
Phase 3 pooled all three medication tables for the *entire* 199,108-
patient cohort into one in-memory frame before deduplicating, which
overflowed available memory. Sharding first, then processing hospital-
by-hospital, bounds peak memory to the largest single hospital's data
(~7,058 patients) rather than the whole dataset (~199,108) — full
technical writeup in `docs/memory_efficiency.md`.

---

## 5. Stage 3 — Temporal feature extraction (time binning)

This is where raw, irregularly-timed clinical events become a
structured signal. For each hospital's shard of `lab`, `medication`
(+ imputed variants, unioned), and `infusionDrug`
(`src/data/temporal/time_binning.py`):

**The time grid.** Every patient's timeline is divided into fixed
4-hour bins (`temporal.time_bin_minutes: 240`) starting at bin 0 =
ICU admission (offset 0), up to a maximum of 48 bins
(`temporal.max_sequence_length: 48`) — an 8-day observation window.
Events before admission (negative offset) or beyond the 8-day window
are dropped, not folded into an edge bin.

**Per-source feature extraction, three different aggregation rules
because the three source tables have genuinely different structure:**

| Source | What it represents | How it's aggregated per bin |
|---|---|---|
| `lab` | Point-in-time lab draws | **Mean** of all results falling in that bin, per tracked lab (platelets, bilirubin, creatinine, fibrinogen, PT-INR, pH — configurable, with an automatic fallback to the cohort's most frequent labnames if none of the configured ones match) |
| `medication` | Orders with a start AND stop offset | **Interval overlap**: a drug ordered in bin 2 and stopped in bin 5 is marked active in bins 2, 3, 4, *and* 5 — not just its start bin. Produces `medication_n_active` (count) and `medication_antibiotic_active` (0/1) |
| `infusionDrug` | Periodic rate recordings, no stop offset in this table | **Point-in-time**: each recording is assigned to exactly the one bin it falls in (not exploded across an interval, since there's no interval to explode). Produces `infusion_n_active`, `infusion_vasopressor_active`, and `infusion_vasopressor_rate_mean` |

This produces **11 features total** per bin per patient (confirmed
from the real run's log): 6 lab features + 2 medication features + 3
infusion features. The output at this stage is still "long format" —
one row per (patient, bin, feature) that actually had data, which is
far smaller than the raw event tables it was built from (this is what
keeps this stage memory-cheap even at full scale).

---

## 6. Stage 4 — Assembling the fixed-shape sequence tensor

Stage 3's long-format output gets pivoted into one dense array per
hospital, then concatenated across hospitals into a single tensor
(`src/data/temporal/sequence_builder.py`):

```
array shape: (n_patients, 48, 11)   # (patient, time bin, feature)
```

Two things are tracked alongside the raw values, and the distinction
between them matters for correctness, not just bookkeeping:

- **`observed_mask`** — true where a specific (patient, bin, feature)
  cell has a real recorded value vs. NaN because nothing was measured
  in that bin. This is a *missing observation within a real bin*.
- **`sequence_length`** — computed independently, from
  `patient.unitdischargeoffset` (how long the patient's ICU stay
  actually was), capped at 48. This is *structural padding* — bins
  that don't exist because the patient was already discharged.

These are kept as two separate concepts on purpose. If a padding bin
(patient already discharged) were treated the same as a real bin with
no lab draw in it, the model could learn to infer *when a patient was
discharged* from the padding pattern itself — an indirect leak of
information that shouldn't be available at prediction time. On the
real run: **mean real sequence length was 14.2 of the 48 possible
bins** (most stays are shorter than the full 8-day window), and
**13.1% of (patient, bin, feature) cells within real bins had an
actual recorded value** — the rest are genuine missingness, handled
explicitly in the next stage rather than silently left as gaps.

---

## 7. Stage 5 — Per-hospital local normalization

Every feature is standardized (zero mean, unit variance) before it
reaches the LSTM — but critically, the mean and variance used are
computed **separately for each hospital**, from that hospital's own
patients only (`src/data/temporal/normalization.py`).

This is not an arbitrary choice. Two things converge on it:

1. **Statistical validity.** A hospital's typical lab value
   distribution is itself a property of that hospital's case mix,
   equipment, and reporting conventions. Standardizing with global
   (cross-hospital) statistics would implicitly assume every hospital's
   population is drawn from the same distribution, which the whole
   federated-learning premise of this project explicitly rejects.
2. **The federated privacy boundary.** Computing a global mean/variance
   would require either centralizing raw data across hospitals or
   transmitting aggregate statistics across the client boundary before
   any federated training round even begins — both violate the "raw
   data never leaves the hospital" architecture this project is built
   around (Section 19 of the master specification).

Missing values (a real bin with no observation) are imputed to that
hospital's own per-feature mean — equivalent to 0.0 after
standardization — rather than a global default, for the same reason.
Structural padding bins are zeroed after standardization regardless of
source, since they carry no signal to preserve.

---

## 8. Stage 6 — How one patient's trajectory actually flows through the LSTM

This is the part your question is specifically about, so it's worth
walking through carefully, because there are two different senses of
"sequential" at play, and conflating them leads to an inaccurate
mental model.

### 8.1 What a single patient's input looks like

After Stages 3-5, one patient is represented by exactly two objects:

- A `(48, 11)` matrix: 48 time bins × 11 features, standardized,
  imputed, with structural padding zeroed.
- A single integer: their real `sequence_length` (e.g. 14, meaning
  bins 0-13 are real and bins 14-47 are padding).

### 8.2 The recurrence: what "sequential" means for ONE patient

The LSTM (`src/modalities/timeseries/lstm_encoder.py`) processes that
patient's 48-bin matrix one timestep at a time, in order, maintaining
two running vectors:

- a **hidden state** `h_t` (the network's evolving summary of
  everything seen so far — a distilled representation of the patient's
  trajectory up to and including timestep `t`)
- a **cell state** `c_t` (a separate memory pathway LSTMs use
  specifically to retain information over long gaps between
  informative timesteps, which is exactly the situation here — most
  bins have no observation at all, and the informative ones can be
  many empty bins apart)

At each timestep, the LSTM's gates (input, forget, output — the
mechanism that gives LSTMs their name) decide how much of the new
timestep's features to incorporate into `h_t`/`c_t`, how much of the
previous state to retain or discard, and what to expose as this
timestep's output. This is genuinely sequential and genuinely
stateful: timestep 13's hidden state is computed as a function of
timestep 12's hidden state and timestep 13's input — you cannot skip
ahead. This is the mechanism that, once trained, is meant to let the
model recognize a patient's *trajectory* (e.g., "creatinine climbing
across bins 3, 6, 9" or "vasopressor started at bin 5 and never
stopped") rather than seeing each bin in isolation.

Two implementation details make this correct rather than approximate:

- **Padding is excluded from the recurrence, not zero-padded through
  it.** `pack_padded_sequence` tells PyTorch's LSTM exactly which of
  the 48 bins per patient are real (via `sequence_length`), so the
  recurrence stops advancing at each patient's actual last real
  timestep — it never processes patient A's 34 padding bins as if they
  were real (zero-valued) observations, which would otherwise corrupt
  the final hidden state with meaningless computation.
- **The embedding is read from the right place.** The output taken as
  the patient's embedding is `h_n[-1]` — the *last layer's* hidden
  state at the patient's *last real timestep*, not bin 47 regardless
  of when their stay actually ended.

### 8.3 What "for all other patients" actually means computationally

Here is the nuance worth being precise about for a methods write-up:
conceptually, yes, this same process — read timestep 0, update state,
read timestep 1, update state, ... — happens independently for every
patient, and no patient's trajectory ever interacts with another
patient's trajectory inside the recurrence. That part of your framing
is exactly right.

But computationally, patients are **not** processed one at a time in a
literal loop. They're processed in **batches**: e.g. 512 patients'
`(48, 11)` matrices are stacked into one `(512, 48, 11)` tensor, and
the LSTM advances *all 512 patients' hidden states by one timestep
simultaneously*, as a single batched matrix operation, before moving
to timestep 2 for all 512 at once, and so on. This is a hardware/
throughput optimization, not a change to the underlying computation —
patient #1's hidden state at timestep 5 is still computed purely from
patient #1's own history, exactly as if it had been run alone. Batching
just means many patients' identical, independent recurrences are
executed in parallel rather than the CPU/GPU sitting idle between
patients.

Why this matters practically: the very first version of this pipeline
tried to batch **all 199,108 patients at once** (batch size =
the entire cohort), which is what actually crashed with an out-of-
memory error on the real machine — the LSTM's internal working memory
during a forward pass scales with total timesteps across the whole
batch, and 199,108 patients' worth of timesteps at once was too much.
The fix, confirmed on the real full-scale run, is to batch 512 patients
at a time (`modalities.timeseries.lstm_inference_batch_size`), write
each batch's 512 output embeddings into the right rows of a
pre-allocated `(199108, 128)` array, and move to the next 512 — with
an automatic batch-size-halving fallback if even 512 turns out to be
too large on a given machine (it wasn't, on the confirmed run: batch
size held at 512 for the entire cohort, 54 seconds total).

### 8.4 The output

After all batches complete, every patient — regardless of which
hospital they came from — has exactly one `128`-dimensional embedding
vector (`hidden_dim: 128`), summarizing their entire observed ICU
trajectory into a fixed-length representation. This is what gets
passed to the graph construction step in the next phase (not yet
built).

---

## 9. What this pipeline does NOT yet do (read before citing this as "trained")

Stated plainly, because it's the single most important caveat for
correctly describing this work:

- **The LSTM's weights are randomly initialized.** There is no loss
  function, no backward pass, no optimizer anywhere in the current
  codebase. The mechanics described in Section 8 execute completely
  correctly — the recurrence, the packing, the batching, the embedding
  extraction are all real and verified at full scale — but the network
  has not been trained on mortality, sepsis, or any other objective.
  The embeddings it currently produces are a structurally valid,
  fixed-length summary of each patient's raw feature trajectory, not
  yet a *learned* representation.
- **There is currently one shared LSTM instance across the entire
  cohort**, not one model per hospital. Hospital isolation today is
  about the *data* (Sections 3-7: no patient or statistic crosses a
  hospital boundary), not yet about *separate model weights per
  hospital*. True per-hospital local training with federated
  aggregation (FedAvg) is Phase 11-12 in `IMPLEMENTATION_TRACKER.md`
  and has not been built yet — at that point, each hospital will train
  its own local copy of this same architecture on its own embeddings,
  and only model *parameters* (never data or embeddings) will be
  shared with the central server for aggregation.
- **The tabular pathway (Section 2) is now implemented (2026-08-11),
  same untrained-weights caveat as the LSTM.** `patient`,
  `apachePredVar`, and `diagnosis` feature engineering and per-hospital
  normalization are built and tested; the autoencoder architecture is
  defined and its forward-pass mechanics are structurally verified, but
  — exactly like the LSTM before real-data confirmation — it has not
  been executed in this development sandbox (no torch/internet here)
  and has random, untrained weights. See `docs/tabular_modality.md`.

None of this diminishes what Sections 3-8 describe — the entire
data-engineering and forward-pass pipeline is real, tested, and
confirmed correct at true full scale (199,108 patients, 208 hospitals,
zero errors). It's simply not yet a trained predictive model, and a
methods section should say so explicitly rather than let a reader
infer otherwise from the presence of a working forward pass.

---

## 10. Reference: implementation status and source files

| Stage | Source file | Status |
|---|---|---|
| Hospital partitioning | `src/data/cohort/hospital_partition.py` | Implemented, verified at real full scale |
| Table classification / leakage flagging | `src/data/ingestion/schema_registry.py` | Implemented, verified |
| Per-hospital sharding | `src/data/ingestion/hospital_sharding.py` | Implemented, verified at real full scale |
| Time binning | `src/data/temporal/time_binning.py` | Implemented, verified at real full scale |
| Sequence tensor assembly | `src/data/temporal/sequence_builder.py` | Implemented, verified at real full scale |
| Per-hospital normalization | `src/data/temporal/normalization.py` | Implemented, verified at real full scale |
| LSTM encoder + batched inference | `src/modalities/timeseries/lstm_encoder.py`, `src/utils/batching.py` | Implemented, verified at real full scale (forward pass only — untrained) |
| Tabular feature engineering + normalization | `src/data/tabular/feature_engineering.py`, `src/data/tabular/normalization.py` | Implemented, verified — 26 unit tests + 2 integration tests |
| Tabular encoder (autoencoder) | `src/modalities/tabular/autoencoder.py` | Implemented (forward pass only — untrained, unverified in this sandbox, see `docs/tabular_modality.md`) |
| Training loop / loss / optimizer | *(not yet created)* | Not implemented — Phase 10-11 |
| Federated aggregation (FedAvg) | *(not yet created)* | Not implemented — Phase 12 |

Full tabular-modality methodology (feature allowlist, leakage
prevention, categorical vocabulary limitations): `docs/tabular_modality.md`.

Full phase-by-phase status: `IMPLEMENTATION_TRACKER.md`. Full
memory-architecture rationale: `docs/memory_efficiency.md`. Full
label-construction methodology (mortality + dual-cohort sepsis):
`docs/sepsis_label_methodology.md`.
