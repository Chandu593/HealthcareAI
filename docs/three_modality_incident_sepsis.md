# Three-modality suite, stage 1: the incident sepsis-proxy label

**Status (2026-10-30): implemented, sandbox-verified (36 tests, mutation-checked) AND run on the real eICU export (199,108 manifest patients) with two policies; numbers below.**
This label is completely separate from the Fed-THOR sepsis label. Nothing here reads, writes or imports the old label logic
(`sepsis_label.py`, which combines antibiotics from -24 h..+48 h with first-24-h-worst SOFA-like values and therefore overlaps the
observation window). Only the antibiotic/vasopressor keyword lists and the pure SOFA band functions for coagulation, liver and renal
are reused, so the bands stay identical to the existing ones.

```
python run_three_modality.py --config configs/three_modality.yaml --stage incident_label
python run_three_modality.py --config configs/three_modality.yaml --stage incident_label --inspect-cache      # nothing is computed
python run_three_modality.py --config configs/three_modality.yaml --stage incident_label --force-recompute    # NEW run directory
python run_three_modality.py --config configs/three_modality.yaml --stage incident_label \
    --set three_modality.incident_sepsis.followup.censoring_policy=alive_discharge_as_negative
```

Output: `<run.output_dir>/three_modality/incident_sepsis/<run_id>/` (never `outputs/paper`, `outputs/ablation`, or the legacy label CSVs):
`incident_sepsis_labels.csv` (one row per split-manifest patient, with `status`, `label`, `event_hour`, ...), `hospital_summary.csv`,
`cohort_report.json` / `.md`, `label_definition.md` (the definition with this run's parameters filled in), `config_snapshot.json`, and
`manifest.json` (written last; its presence marks the run complete; holds the identity hash and a SHA-256 of every file).

## Definition (default 24-96 h)

Times are minutes since ICU unit admission, binned to 1-hour bins, `hour = ceil(minutes / 60)`. Minute 1440 is hour 24 (the last
observation-side minute); minute 1441 is hour 25 (the first event-side minute). Defaults: C = 24 (cutoff), H = 96 (horizon end),
W = 24 (co-occurrence window), P = 24 (pre-ICU look-back for the prevalent test). All are configurable under
`three_modality.incident_sepsis` (whole hours only).

1. **Observation window** = first C hours: the input window for all three modalities (GRU, autoencoder, TCD). The label never uses it.
2. **Eligibility.** A manifest patient is at risk if the ICU stay lasts beyond hour C (`unitdischargeoffset > C*60`) with a valid exit
   time. Hospitals whose medication coverage (share of their at-risk stays with >= 1 row in any medication table) is below 0.8 are
   excluded: the antibiotic evidence cannot fire there. Set `label_validity.min_medication_coverage: 0` to disable.
3. **Evidence cap.** Only records with offset <= min(ICU exit, H*60) are used. Evidence after the patient leaves the ICU never counts.
4. **Infection evidence** = an antibiotic START (drugname keyword match, `antibiotic_reference.py`) in `medication`, `medication_imputed`
   or `medication_imputed1`. Sensitivity variant `infection.evidence: antibiotic_plus_culture`: a microLab culture within +/- W hours of
   the antibiotic start, both inside the window being evaluated (microLab is ~1.5 % dense in eICU, so expect very few events).
5. **Organ dysfunction at grid hour g** = time-resolved SOFA-like total >= 2. Each domain takes its worst value among evidence in the
   trailing window `max(window_start, g-W+1) .. g`:
   respiratory = paO2/FiO2 from lab rows with the identical offset (bands <400:1, <300:2, <200:3, <100:4; bands 3-4 need a
   mechanical-ventilation `treatment` record in the same trailing window, else 2); coagulation = `platelets x 1000`; liver = `total
   bilirubin`; renal = `creatinine`; cardiovascular = a vasopressor `infusionDrug` row present -> 2 (presence only). Lab names are
   matched EXACTLY (case-insensitive), so `urinary creatinine` and `direct bilirubin` never leak in. Unobserved domains contribute 0.
   **Maximum attainable score is 18, not 24**: CNS (GCS), MAP and urine output are not available time-resolved in this 13-table export.
6. **Criteria met at grid hour g** = organ dysfunction at g AND an infection-evidence event inside the same trailing window. So both
   elements occurred within W hours of each other, whichever came first; the event hour is when the second element arrived. With
   whole-hour bins "within W hours" means the two bins differ by at most W-1 (tested).
7. **Prevalent** = criteria met at some g <= C using evidence in bins -P..C. Excluded, never labelled incident. Evidence exactly at
   hour C counts as prevalent ("before or at hour 24").
8. **Incident positive** = not prevalent and criteria first met at some g in [C+1, H] using ONLY evidence in bins C+1..H (the trailing
   windows are clipped at C+1, so nothing from the observation window contributes). A patient with an antibiotic at hour 20 and organ
   dysfunction at hour 40 is therefore NOT positive (and not prevalent) -- see the limitations below.

## Early ICU discharge or death (not automatically negative)

The at-risk window of a patient ends when they leave the ICU. Each eligible, non-prevalent patient falls in exactly one class,
independent of the policy; the policy then decides what the class means for the label:

| Class | exclude_censored (sensitivity) | alive_discharge_as_negative (PRIMARY, default) | require_full_followup_all (sensitivity) |
|---|---|---|---|
| event, ICU stay lasts to hour H or beyond | positive | positive | positive |
| event, left the ICU before hour H | positive (kept) | positive (kept) | **excluded** (`excluded_positive_incomplete_followup`) |
| no event, stay lasts to hour H or beyond | negative (`incident_negative_complete_followup`) | negative | negative |
| no event, ALIVE ICU discharge before hour H | **censored, excluded** (`censored_alive_discharge`) | negative (`incident_negative_alive_discharge`) | censored, excluded |
| no event, ICU death before hour H | censored, excluded (`censored_death_no_event`, competing risk) | same | same |
| no event, unknown exit status before hour H | censored, excluded (`censored_unknown_exit_status`) | same | same |

`cohort_report.md` prints the counts under all three policies from the same run (no recomputation), so the effect of the choice is visible.
**Policy roles (decided 2026-10-30 after the real run):**
`alive_discharge_as_negative` = PRIMARY analysis and the config default; `require_full_followup_all` = complete-follow-up sensitivity analysis;
`exclude_censored` = additional sensitivity analysis. All three stay selectable and all three are reported side by side.

**Real-data counts (eligible at hour 24: 131,578; label-invalid hospitals 56 / 23,250 patients; prevalent 35,300):**

| Policy | Positive | Negative | Prevalence | Notes |
|---|---|---|---|---|
| `alive_discharge_as_negative` (primary) | 4,542 | 67,412 | 6.3% | excludes only 1,072 ICU deaths + 2 unknown exit status (competing risk); 55,288 alive early discharges are negatives |
| `require_full_followup_all` | 2,972 | 12,124 | 19.7% | conditions on staying >= 96 h, unknowable at hour 24: a diagnostic, not a deployable cohort |
| `exclude_censored` | 4,542 | 12,124 | 27.3% | 1,570 positives (35%) left the ICU before hour 96 but NO negative did: differential selection on length of stay, the prevalence is a selection artefact |

**Caveat of the primary policy:** 82% of its negatives are short-stay alive discharges, which will inflate discrimination relative to a cohort of
equally long stays (the same short-stay effect the all-stays mortality protocol carries). That is why `require_full_followup_all` is reported as the
sensitivity analysis: report both, and lead with neither alone.

## Cohort report (what the user asked to see)

`cohort_report.md/json`: manifest size; patients not at risk or with invalid exit times; **initial eligible cohort**; other exclusions
(label-invalid hospitals); **excluded for prevalent proxy**; **excluded/censored for follow-up** (by reason); **incident positives**; **incident
negatives**; per-split counts; and a reconciliation line (eligible = sum of parts) that the build asserts before writing anything.
Also: evidence coverage (patients with >= 1 qualifying record per source in the event window, tables used/missing, rows read vs kept) and
diagnostics that quantify the definition's limitations.

## Known limitations (stated, not hidden)

* **Infection evidence must be post-cutoff.** A patient already on antibiotics since hour 5 who deteriorates at hour 40 with no NEW antibiotic
  start is a negative. The report counts `negatives with antibiotic start before or at cutoff` so the size of this can be judged.
* **Absolute SOFA-like threshold, no baseline.** Chronic organ dysfunction (e.g. dialysis-dependent creatinine) can satisfy the organ-dysfunction
  half; there is no delta-from-baseline mode.
* **Unmeasured is scored 0.** eICU labs are sparse; a domain never measured after hour 24 contributes nothing. The evidence-coverage table shows
  how many patients each source reaches.
* **paO2/FiO2 pairing needs identical lab offsets**; patients with paO2 but no same-time FiO2 get no respiratory score.
* **Ventilation support** uses the substring `mechanical ventilation` in `treatmentstring` (the eICU taxonomy string is
  `pulmonary|ventilation and oxygenation|mechanical ventilation`); verify against `treatment.treatmentstring.unique()` on the real data.
* **Lab names** are the eICU standard names; verify against `lab.labname.unique()`. A domain showing ~0 patients in the evidence-coverage table
  almost always means the name did not match.
* **Medication tables are unioned, not deduplicated** (existence of an evidence bin is all that matters); cancelled orders are not excluded
  (same as the old label).
* **The split was not stratified on this label.** The permanent manifest is reused unchanged (stratified on mortality); per-split positive
  counts are in the report.
* **Antibiotic orders are a proxy for suspected infection**, not a validated Sepsis-3 onset. This is a sepsis-RELATED PROXY event; do not call it
  Sepsis-3 onset.

## Reuse, identity and never-overwrite

The run identity hash covers: definition version (`incident_sepsis_v1`, bump on any semantic change), every parameter except chunk size, the
split-manifest hash, and the size+mtime of every input table. An identical, verified run (all file hashes still match `manifest.json`) is
reused; any change, or a tampered file, recomputes into a NEW directory. `--force-recompute` always makes a new directory; an explicit
`--run-id` that already exists is an error. Files are written atomically (temp file + `os.replace`).

## Verification (sandbox, synthetic data)

`tests/unit/test_three_modality_incident_sepsis.py` (31) and `tests/integration/test_three_modality_cli.py` (4). The end-to-end fixture runs through
the real chunked CSV reader (chunk size 3) with 19 hand-labelled patients covering: positive; prevalent; evidence exactly at hours 24 / 96 / one minute
past the horizon; pre-cutoff-only infection or organ dysfunction; elements straddling hour 24; ICU exit capping evidence; decoy lab names; label-invalid
hospital; each censoring class and policy. Hand-performed revert -> fail -> restore -> pass proofs: incident window not clipped at the cutoff;
prevalent window one hour short; event window ending at hour 95; event window starting at hour 24; evidence after ICU exit not dropped; early alive
discharge silently counted as negative; substring lab-name matching; floor instead of ceil binning; infection evidence not required. Two earlier
mutants (window not clipped, hour-24/96 boundaries) first SURVIVED because the unit tests called the evaluator with the right window directly rather
than through the orchestrator; fixed by adding the boundary patients above. One mutant (horizon cap inside the reader) cannot change output by
construction -- `evaluate_window` bounds the window independently -- so it is a redundant memory guard, not a gap.

**Immediate next step (needs your machine):** run the command above on the real data, then read `cohort_report.md`: (1) the evidence-coverage table
(a ~0 row means a lab/treatment name did not match), (2) the final counts and per-split positives, (3) the policy-sensitivity table.
