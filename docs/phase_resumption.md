# Phase Resumption (`--start-phase`)

**Status: implemented, sandbox-verified.** New `--start-phase N` CLI
argument lets `run_pipeline.py` skip already-completed phases and
reload their outputs from `run.output_dir` instead of recomputing
them. Verified with 7 end-to-end tests that invoke the actual CLI via
subprocess (`test_phase_resumption.py`), not just the underlying
loader functions in isolation. **Not yet run against the real
dataset** -- immediate next step once delivered.

## Why this exists

Every phase from 1 through 9 used to always run in sequence, every
invocation -- so checking whether a Phase 8 config change (say, a
different `k`) actually helped meant re-running Phases 1-7 first,
including the genuinely expensive ones (Phase 1's full chunked data
report, Phase 3's SOFA/sepsis labeling, Phase 4/5's forward passes
over the whole cohort). At real full scale this is minutes to tens of
minutes of pure overhead for a one-line config change three phases
later.

## Usage

```powershell
# Normal full run (unchanged) -- current default behavior
python run_pipeline.py --config configs/full.yaml --phase 9

# Only run Phases 8-9, reusing everything through Phase 7
python run_pipeline.py --config configs/full.yaml --phase 9 --start-phase 8

# Only run Phase 9, reusing everything through Phase 8
python run_pipeline.py --config configs/full.yaml --phase 9 --start-phase 9
```

`--start-phase` requires the phases before it to have already been run
**for this exact `run.output_dir`** -- it reads whatever that config
points at, so pointing two different configs at the same
`run.output_dir` (or changing `run.output_dir` between runs) will
either resume the wrong thing or fail to find what it expects. Valid
values: `1, 2, 3, 4, 5, 7, 8, 9` (`6` is permanently skipped, same as
`--phase`). `--start-phase` greater than `--phase` is rejected with a
clear error before anything runs.

## What actually gets persisted

Phases 4, 5, 7, and 8 already saved everything a later phase needs
(`embeddings_path`, `fused_embeddings_path`, `manifest_path`) --
`load_phaseN_result()` for each of these just checks the expected file
exists at the path `run.output_dir` implies and returns it. No new
persistence needed for these.

Phases 1-3 didn't previously persist enough to be resumed from, so
this feature added:
- **Phase 1**: already wrote `_phase1_summary.json` on every run --
  `load_phase1_summary()` is a pure reader, no save-side change needed.
- **Phase 2**: now additionally writes `partition_result_full.json`
  (the existing human-readable `hospital_partition.json` summary
  didn't include `patient_ids`, which is the one thing actually needed
  downstream -- a separate file was added rather than changing that
  summary's schema, so the existing test asserting its shape is
  untouched).
- **Phase 3**: now additionally writes `cohort_patients.csv`,
  `mortality_labels.csv`, `sepsis_labels.csv` under
  `outputs/cohort_reports/`.

## Dtype safety through the CSV round-trip (checked, not assumed)

This project has a documented real bug from exactly this class of
issue (bool columns silently becoming `object` dtype after a merge --
see `IMPLEMENTATION_TRACKER.md`'s "Bug found and fixed" section). CSV
does not reliably preserve dtypes, so:

- `sepsis_df`'s boolean columns are explicitly re-cast to `bool` after
  reload, not left as whatever pandas infers from the string
  "`True`"/"`False`" text.
- `mortality` (a `pd.Series` that originally shares `cohort_patients`'
  DataFrame index, not `patientunitstayid`) is saved as an explicit
  two-column `{patientunitstayid, mortality}` CSV and reconstructed via
  an ID-based `.map()`, never a positional index round-trip -- the same
  ID-based-alignment discipline this project has followed since Phase
  7's fusion logic.

Verified directly (not just asserted in a docstring): a round-trip
script ran Phase 1-3 normally, then called each loader and diffed the
reconstructed objects against the originals -- confirmed identical
`patient_ids`, `mortality` values (including `NaN` handling), and
`sepsis_df` dtypes, before any of this was wired into `main()`.

## The dependency graph is computed explicitly, not assumed sequential

The real risk with a feature like this is over-fetching: naively
"skip phases before N, but still touch everything phase N *would* have
needed if it had run from scratch" ends up reloading things nothing
downstream actually uses. This was a genuine bug caught while building
this feature, not a hypothetical -- an early version of `--start-phase
8` still tried to reconstruct `phase3_result` (and therefore
`phase1_summary`), even though neither Phase 7 (when *loaded* rather
than run) nor Phase 8 nor Phase 9 ever touch either one.

Fixed by computing explicit `need_*` boolean flags in `main()` from
each phase's *actual* parameter list (not phase number proximity):

```
Phase 2 needs: phase1_summary
Phase 3 needs: phase1_summary, partition_result, patient_df
Phase 4 needs: phase3_result (incl. shard_manifests), partition_result
Phase 5 needs: phase1_summary, phase3_result, partition_result
Phase 7 needs: phase3_result (cohort_patients only), phase4_result, phase5_result
Phase 8 needs: phase7_result, partition_result
Phase 9 needs: phase7_result, phase8_result
```

A result is only loaded (or computed) if the phase that produces it
will itself run, OR some *downstream phase that will actually run*
directly depends on it -- a phase being skipped-and-loaded contributes
nothing to any other phase's requirements, since `load_phaseN_result()`
never re-derives phase N's own upstream inputs.

**`shard_manifests` gets its own, finer-grained flag**
(`need_shard_manifests = will_run(4)`), separate from the rest of
`phase3_result`, because it's confirmed (by inspection of every
reference to `phase3_result["shard_manifests"]`) to be read by Phase 4
alone -- Phase 5 and Phase 7 never touch it. This matters for more
than speed: `shard_table_by_hospital()`'s cache-freshness check needs
to `stat()` the *original* raw source table (to detect if it changed
since caching), so reconstructing `shard_manifests` requires that raw
file to still exist on disk even on a guaranteed cache hit. Skipping
this reconstruction when Phase 4 won't run is what lets `--start-phase
7` (or `8`, `9`) work with the raw eICU data no longer present at all
(proven directly -- see below), not merely work faster.

## Verified

`tests/integration/test_phase_resumption.py` invokes the real CLI via
`subprocess`, not the underlying functions directly, because the claim
being tested ("the command line tool can skip phases") can only be
faithfully proven that way:

- **`test_full_run_then_resume_from_phase7_after_deleting_raw_data`** --
  the core proof. Runs the full pipeline once, deletes the mock data
  root *entirely*, then runs `--start-phase 7 --phase 9` and confirms
  it still succeeds with matching output shape/patient count. If
  anything from Phases 1-6 were secretly still touching raw data, this
  would fail with a file-not-found error, not just run slower --
  succeeding with the data gone is direct evidence of genuine
  resumption, not simply a fast skip.
- **`test_start_phase_8_needs_no_phase1or3_artifacts_but_does_need_phase2`** --
  direct regression test for the over-fetching bug described above.
  Deletes Phase 1 and Phase 3's saved artifacts specifically (proving
  those aren't needed for `--start-phase 8`) while deliberately
  *keeping* Phase 2's `partition_result_full.json` (since Phase 8
  genuinely, directly depends on it for per-hospital patient lists).
- **`test_resume_from_phase4_reuses_phase3_labels_correctly`** -- a
  different resumption point (not just Phase 7+), proving this isn't
  special-cased.
- Four more covering the CLI's validation and error-clarity: rejecting
  `--start-phase > --phase`, rejecting the intentionally-skipped
  Phase 6 as a resumption point, and two tests confirming that
  resuming past a phase that was never run fails with a message
  naming the specific missing phase (not a generic/silent failure).
- Two more covering the cohort-identity guard above: a direct
  regression test confirming a `cohort.patient_subset` change between
  runs sharing the same `run.output_dir` gets blocked (not silently
  mixed), plus a sanity check that an unchanged config still resumes
  normally (the guard doesn't false-positive). Revert→fail→restore→pass
  proven by hand: disabled the identity check, confirmed exactly the
  mismatch-detection test failed (silently succeeding instead of
  blocking) and nothing else, restored the check (byte-diffed to
  confirm), confirmed all 9 resumption tests passed again.

Full suite: 224 passed, 2 skipped, zero regressions.

## Cohort-identity validation (added after a real risk was found)

`debug.yaml`, `subset.yaml`, and `full.yaml` all share the same
default `run.output_dir` -- none of them override it. Before this
guard existed, running Phase 1-7 with `debug.yaml` (a tiny
`cohort.patient_subset: 20` for quick testing) and later running
`--start-phase 8 --phase 9` with `full.yaml` would have **silently**
resumed Phase 8/9 against the debug run's tiny-cohort graphs, while
the user believed they were working with the real full cohort -- same
file shapes, no crash, no warning, just quietly wrong results.

`validate_run_identity()` closes this: whenever `--start-phase > 2`,
it compares the fields that actually determine *which patients and
hospitals are in the cohort* (`data.root`, `cohort.patient_subset`,
`federated.mode`, `federated.hospitals`, `run.seed`) against whatever
values `run_identity.json` recorded the last time Phase 2 genuinely
ran for this `run.output_dir`. Any mismatch raises a clear error
naming exactly which field(s) differ, rather than proceeding.

**Deliberately does NOT check fusion/graph/gnn hyperparameters** (e.g.
`graph.k`, `gnn.type`, `gnn.hidden_dim`) -- changing those between the
run that produced Phase 7's output and a `--start-phase 8` run is
*the whole point* of this feature, not a mismatch to guard against.
Only identity of the underlying patient population is checked, not
every setting in the config.

**Remaining, narrower limitation, by design:** this only catches a
mismatch relative to what Phase 2 itself last established. If you run
Phase 1-8 once, then change `graph.k` and run `--start-phase 8` again
(correctly regenerating Phase 8 with the new `k`), then later run
`--start-phase 9` -- Phase 9 correctly uses the *latest* Phase 8 output
you just regenerated. But if you change `graph.k` and *forget* to
re-run Phase 8 before jumping straight to `--start-phase 9`, Phase 9
will run against Phase 8's older graphs without complaint -- there's
no mechanism (yet) that tracks whether Phase 8's own *config inputs*
changed since it last ran, only whether the *cohort* did. Still on you
to resume from the right phase given what you actually changed.

## What this does NOT yet do

- **Not yet run against the real 199,108-patient / 208-hospital
  dataset.** Sandbox-verified only so far, same caveat as every new
  feature before its first real run.
