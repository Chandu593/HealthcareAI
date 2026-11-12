# Implementation Tracker

Status legend: `[ ]` not started · `[~]` implemented, verified on synthetic
data only · `[x]` implemented and verified · `[BLOCKED]` waiting on a
methodology decision.

**Environment note:** the sandbox used through 2026-08-13 had no
internet access and did not have `torch` installed. Everything markable
`[~]`/`[x]` from that period which doesn't depend on torch was
genuinely executed there. A later sandbox session (2026-08-23) does
have internet access and successfully installed `torch` (CPU-only) —
LSTM and autoencoder forward passes are now genuinely executed and
verified in-sandbox against synthetic data too, not just skip-marked,
in addition to the real full-scale confirmations documented below (both
Phase 4's LSTM and Phase 5's autoencoder were already confirmed on the
user's own machine at real full scale before this sandbox ever had
torch).

**Note on phase numbering:** the master spec's Section 6 tracker template
lists "Phase 2 Cohort construction, Phase 3 Hospital partitioning," but
Sections 8–9 (the detailed phase descriptions) present them in the
opposite order ("PHASE 2 — HOSPITAL-WISE PARTITIONING" before "PHASE 3 —
COHORT AND LABEL CONSTRUCTION"), and Section 47's implementation order
agrees with Sections 8–9. This is an internal inconsistency in the
source spec. Resolved here by following Sections 8–9 / 47 (hospital
partitioning before cohort/label construction), since they contain the
actual implementation detail. Flagged for the user's awareness rather
than silently picked.

| # | Phase | Status | Verified |
|---|---|---|---|
| 0 | Analysis (architecture, tables, keys, leakage, ambiguities) | [x] | This document + chat response |
| 1 | Data ingestion & validation | [x] | Verified on synthetic data AND on the real 200,859-patient eICU dataset (user's own run, 2026-08-09). `treatment` promoted to required table 2026-08-08 per user confirmation. |
| 2 | Hospital-wise partitioning | [x] | `src/data/cohort/hospital_partition.py` + tests. Verified on synthetic data AND at real full scale (208 hospitals, 199,108 patients, 2026-08-10): no cross-hospital patient leakage, deterministic sampling, undersampling handled without crashing (fixes the bug in the user's original prototype code) |
| 3 | Cohort & label construction (mortality + sepsis) | [x] | Dual-cohort design locked 2026-08-08 (strict primary / broad sensitivity). Verified on synthetic data (28 dedicated tests) AND end-to-end on the real dataset at BOTH subset scale and real FULL scale (199,108 patients, 208 hospitals, 557s, 2026-08-10 — reproducible: sepsis counts matched the prior crashed attempt exactly). Hospital-wise sharded processing — see `docs/memory_efficiency.md`. See `docs/sepsis_label_methodology.md` for scoring-band detail and limitations. |
| 4 | Time-series modality (LSTM) | [x] | Preprocessing AND LSTM inference verified on synthetic data AND real FULL-scale data (199,108 patients, 208 hospitals, 2026-08-10). The batching fix (`src/utils/batching.py`) held `batch_size=512` constant for all 199,108 patients — zero backoff events needed, 54s total inference time. Produced real `(199108, 128)` embeddings. See `docs/memory_efficiency.md` and `docs/timeseries_modality.md`. |
| 5 | Tabular modality (Autoencoder) | [x] | Feature engineering + per-hospital normalization implemented and verified (48 unit + 6 integration tests as of 2026-08-24, up from 26+2 — added across the vocabulary-sampling-bias fix and the follow-on hospital-source fix below), including explicit leakage-exclusion checks against the assembled pipeline output. Autoencoder forward pass genuinely executed and verified on the user's own machine at real full scale (199,108 patients, 208 hospitals, 2026-08-13) — produced real `(199108, 64)` embeddings; also now genuinely executed in-sandbox (torch installed 2026-08-23), not just written/skip-marked. See `docs/tabular_modality.md`. |
| 6 | Text modality (MiniBERT) | [x] design decision | Confirmed OFF by default 2026-08-08 — no clinical notes in this eICU export; diagnosis-string pseudo-text mode not being built (user declined) |
| 7 | Multimodal fusion (concatenation) | [x] | LSTM (128) + autoencoder (64) → fused (192) via ID-based concatenation, `src/fusion/multimodal_fusion.py`. Sandbox-verified (11 unit + 4 integration tests, including a hand-performed revert→fail→restore→pass proof for the ID-alignment logic) AND verified on real full-scale data (2026-08-26): `(199108, 192)`, clean run, 11.25s. Option A (pure concatenation, no projection layer) locked 2026-08-25. See `docs/multimodal_fusion.md`. |
| 8 | Intra-hospital patient graph | [x] | One graph per hospital from Phase 7's fused embeddings, `src/graph/patient_graph.py`. Sandbox-verified (20 unit + 4 integration tests, including hand-performed revert→fail→restore→pass proofs for both the exact-vs-chunked similarity claim and the no-cross-hospital-edges guarantee) AND real-full-scale-verified (2026-08-26): 208 hospitals, 199,108 nodes, 1,462,202 edges, zero cross-hospital edges confirmed directly against real saved files, chunked-similarity path genuinely exercised (29 real hospitals >2000 patients, up to 7,058). Option B (exact similarity, chunked for memory, no ANN library) locked 2026-08-26. See `docs/patient_graph_construction.md`. |
| 9 | Graph neural network (GraphSAGE/GAT/GCN) | [x] | All 3 architectures implemented, `src/gnn/patient_gnn.py` -- consumes Phase 8's graphs + Phase 7's fused embeddings, one forward pass per hospital's own graph, no training loop yet. Sandbox-verified (11 unit + 4 integration tests, including a hand-performed revert→fail→restore→pass proof for the differentiated edge_weight design) AND real-full-scale-verified (2026-08-27, `graphsage` default): `(199108, 128)`, 208 hospitals, 22.23s, zero errors. `gat`/`gcn` remain sandbox-verified only. Edge weight used only for GCN (native, published mechanism); GraphSAGE/GAT use standard unweighted/self-attention formulations (locked 2026-08-26, confirmed with user). See `docs/gnn_message_passing.md`. |
| 10 | Multi-task prediction heads | [x] | Independent MLP heads (mortality + sepsis) on Phase 9's GNN embeddings, `src/heads/multitask_heads.py`. Architecture + forward-pass verification only, no training loop yet. Sandbox-verified (19 unit + 6 integration + 2 phase-resumption tests, four hand-performed revert→fail→restore→pass proofs) AND real-full-scale-verified (2026-08-30): `mortality_logits`/`sepsis_logits` both `(199108,)`, 2.48s. `sepsis_broad_positive=44412` exactly matches Phase 3's own originally-recorded real count. `sepsis_cohort=strict` remains sandbox-verified only. See `docs/multitask_heads.md`. |
| 11 | Local training loop | [x] | Full end-to-end joint local training (LSTM+autoencoder+GNN+heads, `src/training/local_model.py`'s `LocalPatientModel`), per hospital, never pooling across hospitals. Includes the autoencoder's reconstruction loss (`src/training/losses.py`). Every hospital starts from identical initial weights, trains locally, checkpoints to `checkpoints/phase11_local/`, ready for Phase 12/FedAvg. Sandbox-verified: real loss decrease confirmed on mock data, all four components proven to receive gradients (incl. two hand-performed revert→fail→restore→pass proofs), 28 new unit/integration tests + 2 new phase-resumption tests. **Not yet run against real full-scale data.** See `docs/local_training.md`. |
| 12 | Federated learning (FedAvg) | [x] | Multi-round FedAvg (`src/training/fedavg.py`'s `fedavg_aggregate()`, weighted by n_patients per hospital -- standard McMahan et al. 2017), reusing Phase 11's exact per-hospital mechanics via a shared `_run_local_training_round()` helper. `configs/base.yaml`'s `federated:` stanza (`n_rounds`, `reuse_existing_local_checkpoints_for_round_1`). Sandbox-verified (19 unit/integration tests + 2 phase-resumption tests, four hand-performed revert→fail→restore→pass proofs) AND real-full-scale-verified (2026-09-02): 3 rounds, round 1 reused (0.13s), rounds 2-3 fresh (~32 min each), 207 hospitals/round, 0 non-finite across 414 real training instances. See `docs/fedavg.md`.
| 13 | Baselines (local / centralized / federated) | [x] | Local-only (each hospital trains independently, 15 epochs matching federated's total budget, never averaged) and centralized-sequential (ONE continuously-updated model, no resets, cycles through every hospital's own intra-hospital graph -- keeps Phase 8's privacy boundary untouched) baselines, `run_phase13_baselines()`. Federated baseline references Phase 12's already-completed result, not re-run. Sandbox-verified (9 integration tests + 2 phase-resumption tests, two hand-performed revert→fail→restore→pass proofs) AND real-full-scale-verified (2026-09-05): local 207/207 decreasing, centralized -44.7% net over 15 epochs (textbook decelerating convergence once the FULL trajectory was checked, not just 3 sampled points), zero NaN/Inf across both. See `docs/baselines.md`. **Extended 2026-09-10 with a research-grade ablation framework that runs BEFORE this phase's federated stage -- see the dedicated section below. `run_phase13_baselines()` itself is untouched.**
| 14 | Evaluation (global + per-hospital) | [ ] | |
| 15 | Ablation studies | [ ] | |
| 16 | Explainability | [ ] | |
| 17 | Visualization | [ ] | |
| 18 | Caching & checkpointing | [ ] | Interfaces stubbed in config (`cache_dir`, `checkpoint_dir`); no logic yet |
| 19 | Final integration & reproducibility | [ ] | |

## Three-modality stage 3b: clean static inputs + pooled tabular autoencoder -- 2026-11-04 (sandbox-verified on mock data; NOT run on real data)

Build-order step 2. `src/three_modality/{tabular_inputs,tabular_ae}.py`, CLI stages `tabular_inputs` / `tabular_ae`, `docs/three_modality_tabular.md`. Inputs reuse the paper track's `build_feature_bundle` with its `sepsis_clean` exclusions (imported, not copied) plus every lab group,
APACHE scores off, all stays, labels stubbed; train-only preprocessing; columns typed continuous/binary/indicator so the reconstruction loss is MSE on continuous and BCE on binary columns; the autoencoder also gets two focal heads (separate alpha/gamma/weights per task),
so focal loss is used in the embedding generation as requested while reconstruction stays a separate, separately weighted term. Pooled training, landmark_only default, TEST labels blanked, seeded model factory (stage 3a's lesson).
Verified: 8 tests, 3 targeted mutation proofs (all caught), a real-CLI smoke chain on mock data (tcd_features -> tabular_inputs -> incident_label -> tabular_ae, cache hit, inspect). Per the user's request to speed up, only tests for new/changed code were run (no full-chain re-verification this step).
Known limit: no real-data poison audit for the static set (paper-track builder reused as is); verified at pipeline level on mock data only.

## Three-modality stage 3a: first-24-hour GRU inputs + pooled focal GRU encoder -- 2026-11-01 (sandbox-verified on synthetic data; NOT run on real data)

User approved the pooled design and fixed the requirements: all neural components train CENTRALLY on pooled training patients (not federated, no per-hospital encoders; Graph A/B stay strictly hospital-local); GRU gets fresh 24 x 1-hour
sequences with value + observation-mask channels and train-only scaling; the autoencoder uses the clean static feature set (no APACHE scores, no labs); TCD embeddings are reused; configurable focal loss with separate alpha/gamma/task weights per task
wherever possible; new sibling modules only. Build order: GRU -> autoencoder -> three-token cross-attention -> Config 1 -> graphs/GCN + Configs 2,3 -> Config 4 -> consolidated results. This entry is step 1.

**Audit finding that shaped the stage:** the legacy Phase-4 time series are NOT first-24-hour data: base config `time_bin_minutes: 240` x `max_sequence_length: 48` = 192 h (no overlay changes it), medication/infusion channels are included,
and `sequence_length` is derived from `unitdischargeoffset` (ICU length of stay leaks into the recurrent encoder). The legacy arrays (including `outputs/split_aware`) are therefore not used; the existing `GRUTimeSeriesEncoder` module is reused.
Also corrected an inconsistency of mine: I had said encoders would train per hospital like the legacy ablation, but had built the TCD encoder pooled; the user chose pooled for all.

New: `src/three_modality/{gru_inputs,gru_encoder,multitask_training}.py`, CLI stages `gru_inputs` / `gru_encoder`, `docs/three_modality_gru.md`. `multitask_training.py` is the generic pooled focal multitask trainer (separate alpha/gamma/weights per task, per-task
label masks, auxiliary-loss hook for the autoencoder's reconstruction, best-validation restore) that the autoencoder and fusion stages reuse.

**Verified (sandbox):** 124 three-modality tests pass (32 incident + 34 TCD features + 22 TCD encoder + 28 GRU + 8 CLI), 166 with the neighbouring paper tests; 12 new mutation proofs, all caught. **A real bug found by my own test:** the trainer originally received a
model built BEFORE it seeded the RNGs, so two fresh processes started from different initial weights (confirmed: three fresh processes gave three different initial weights); runs would not have reproduced. Fixed structurally (the trainer now takes a model factory and
builds the model after seeding) with a regression test that perturbs the global RNG state; the TCD encoder was not affected (it already built its model after seeding). Two other test failures were my own wrong expectations (an under-trained synthetic task: measured 0.92/0.95
val AUROC at lr 0.01 x 30 epochs against 0.67 at lr 0.003 x 10; and a baseline built with the wrong seeding order) and were fixed after measuring, not by loosening blindly. **Not yet run against the real dataset.**

## Three-modality stage 2, real-data run + two follow-up changes -- 2026-10-31

Real run (user, full dataset): features `tcdf_20261031T162203Z_3bc446e5` (445.8 s) and first encoder `tcde_20261031T163020Z_e5fdd41e` (62.4 s, ALL stays). Everything checkable passed: patient counts match the manifest (139,208 / 29,932 / 29,968); the
window numbers reconcile exactly with the paper-track landmark (199,108 - 66,208 clipped - 1,205 invalid exit = 131,695; minus 117 stays with exit exactly at minute 1440 = the incident label's 131,578); all 20 lab roles matched
(coverage 21.6%-86.0%); 1,595 raw -> 1,289 kept (+439 indicators = 1,728 inputs; the 306 dropped were all >95% missing); poison audit passed (2,000 patients, 105,554 corrupted late rows, bit-identical, max diff 0.0). Validation
AUROC at the selected epoch: mortality 0.829 (all stays), incident sepsis 0.773; best epoch 4 of 9; sepsis overfits quickly. Validation only; test never evaluated.

Two things were flagged from the real report and changed on the user's instruction: (1) **landmark-only encoder training is now the default** (`tcd.training.landmark_only: true`): gradient updates and early stopping use only patients with a valid ICU
exit >= 1440 min (the paper track's strict landmark); embeddings are still produced for every patient; part of the encoder identity, so the next `--stage tcd_encoder` creates a NEW run (the all-stays run stays untouched). Reason: 66,208 clipped
windows whose features reveal a short stay let an all-stays mortality objective learn short-stay patterns that do not exist in the landmark cohorts. (2) **Ventilator / oxygen-support settings that the `lab` table carries (`fio2`, `peep`; `tv`, `vent_rate`,
`pressure_support`, `lpm_o2` just missed the top-30) are KEPT as features and flagged as direct overlap with the label's ventilation evidence**; the earlier statement "interventions are not TCD inputs" was inaccurate for these and the docs were
corrected. The sealed feature run still labels `peep` as `none` (flag added afterwards; values unchanged); every encoder run records the authoritative `overlap_audit` in its manifest.

**Verified (sandbox):** 95 three-modality tests pass (34 features + 22 encoder + 7 CLI + 32 incident); 7 new mutation proofs (landmark mask ignored, flag ignored, off-by-one at exactly 1440, missing exit treated as landmark, early stopping on all val,
embeddings only for landmark patients, ventilation not flagged), all caught. Process note: the first version of the landmark test compared only embeddings, and a mutant (early stopping on all validation patients) slipped through because the best
epoch happened not to change; the test now compares the per-epoch validation loss, which must be bit-identical when only non-landmark validation labels change. **The landmark-only encoder has not been run on real data yet.**

## Three-modality suite, stage 2: TCD features + MLP encoder -- 2026-10-30 (sandbox-verified on synthetic data; NOT run on real data)

Requested after the stage-1 real run. `src/three_modality/{tcd_features,tcd_encoder,artifacts}.py`, `run_three_modality.py --stage tcd_features|tcd_encoder`, `docs/three_modality_tcd.md`. No existing legacy/paper/incident file was modified in behaviour (incident_sepsis.py now imports its atomic-write/run-identity helpers from the new shared `artifacts.py`; its 36 tests still pass). Also done in this step, as instructed: `alive_discharge_as_negative` is now the config default and primary incident policy, `require_full_followup_all` the complete-follow-up sensitivity, `exclude_censored` an additional sensitivity; all three remain selectable and no existing run was touched.

ONE unified vector per patient from the first 24 ICU hours: families A statistics, B trends, C abnormality burden/recovery, D derived indicators, E lab dynamics, F observation/missingness, G interactions. Decisions made (flag any you disagree with): (1) this export has only `lab` as time-stamped value series, so shock index / pulse pressure / MAP are DECLARED UNAVAILABLE (never imputed); an optional vitals path exists for a table that is not in your export and is tested on synthetic data only. (2) Variables = 20 named lab roles (kept down to 0.5% training coverage) + top-30 other labs (>= 2%), coverage fitted on TRAINING patients only; lab names matched exactly; only the 5 names used by the incident label are confirmed on real data. (3) Interventions are not TCD inputs. (4) Reference ranges are project-defined adult limits, configurable. (5) Statistics below their minimum counts are NaN, never zero. (6) Preprocessing (missingness/constant filter, winsorise, median impute, standardise, deduplicated missing indicators) is fitted on TRAINING patients only. (7) Encoder = existing TabularMLPEncoder + two linear heads, focal multitask loss (repo masked_focal_loss; gamma 2.0, alpha 0.25 by default -- note the repo's alpha weights the POSITIVE class), train rows update, validation early-stops, TEST labels blanked and never evaluated; embedding dim 64; output file in the same format as the GRU/autoencoder embeddings.

**Verified (sandbox):** 32 feature tests + 15 encoder tests + 3 new CLI tests; vectorised features vs an independent loop-based reference on ~630,000 values with zero mismatches; 9 feature-code and 6 encoder mutation proofs (revert->fail->restore, byte-identical). Process notes worth keeping: (a) both reference-vs-vectorised disagreements were floating-point artefacts in the REFERENCE (differences of already-divided hours are inexact at the 0.5/2/4/6 h boundaries; whole-minute differences are exact); (b) a helper-level test would not have caught "preprocessing fitted on all patients", so pipeline-level tests were added before running mutants -- the same lesson as stage 1; (c) a test found that the draw-key packing assumed offsets < 2048 min and would silently overflow into the next patient's id space for unclipped data -- the multiplier is now data-driven; (d) several of my own test expectations were wrong (swapped min/max, tampering that wrote already-zero bytes, a too-high sanity threshold) and were corrected, not the code, after checking the cause each time. **Not yet run against the real dataset; real runtime/memory unmeasured.**

## Three-modality suite, stage 1: incident sepsis-proxy label (ICU hours 24-96) -- 2026-10-30 (sandbox-verified on synthetic data; NOT run on real data; torch-dependent tests not run here)

New, separate track (`src/three_modality/`, `run_three_modality.py`, `configs/three_modality.yaml`, `docs/three_modality_incident_sepsis.md`); no existing file was modified and nothing is written into `outputs/paper`, `outputs/ablation` or the legacy label files (a test hashes them before/after). Requested by the user with explicit rules: first 24 h = input window for all three modalities; event window fixed at hours 24-96 (configurable); patients meeting the proxy before or at hour 24 are excluded (prevalent), never counted as incident; the old label logic (antibiotics -24..+48 h with first-24-h-worst SOFA) is not reused; patients leaving the ICU before hour 96 are not automatically negative; separate counts for eligible / prevalent / follow-up-censored / positive / negative / other exclusions; the new label is kept apart from the Fed-THOR sepsis label.

Decisions made while building (flag any you disagree with): (1) this export has no `vitalPeriodic`/`nurseCharting`/`intakeOutput`, so the time-resolved SOFA-like score has 5 domains from `lab` (platelets, total bilirubin, creatinine, paO2/FiO2 pairs at identical offsets), `infusionDrug` (vasopressor presence) and `treatment` (mechanical ventilation gating respiratory bands 3-4); maximum attainable score is 18 and CNS/MAP/urine output are unscored. (2) Infection evidence = antibiotic start only, post-cutoff only; `antibiotic_plus_culture` is a config option (microLab is ~1.5 % dense). (3) Criteria met when organ dysfunction (>= 2) and infection evidence fall in the same 24-bin trailing window, whichever came first. (4) Default follow-up policy `exclude_censored` (alive early discharge and ICU death without an event are excluded as censored/competing risk, positives kept); `alive_discharge_as_negative` and `require_full_followup_all` are options and the report prints the counts under all three from the same run. (5) Hospitals with medication coverage < 0.8 are excluded (the antibiotic half cannot fire there). (6) The permanent split manifest is only LOADED, never built; it was not stratified on this label.

**Verified (sandbox):** 31 unit + 4 CLI integration tests; end-to-end fixture of 19 hand-labelled patients through the real chunked CSV reader. Hand-performed revert->fail->restore->pass proofs for nine injected bugs (see the doc). Process note worth keeping: two mutants SURVIVED the first suite (incident window not clipped; hour-24/96 boundaries) because the unit tests called the evaluator with the correct window directly instead of through the orchestrator -- caught only by running the mutants, fixed with boundary patients in the end-to-end fixture. One further mutant is equivalent by construction (redundant horizon cap in the reader). A bug found by the tests: two auto-named runs in the same second collided on the run id (now `_vN` suffix; explicit ids that exist still raise). **Not yet run against the real dataset.** Lab/treatment-name matching in particular is unverified on real data; the report's evidence-coverage table is the check (a ~0 row = the name did not match).

## Paper-track v21: pooled-GBDT results in the text; final comparison in FLICS's format (Table 14) -- 2026-10-28 (sandbox-verified; full suite not run in the sandbox: no torch)

Requested by the user: keep the FLICS lines on Figure 2's all-stays panel; add the federated and central LR, MLP, LightGBM and GBDT models for strict AND all stays (and each setting), highlight the best figure in each case as FLICS does, and include accuracy and F1 as FLICS does. The six `gbdtcentral_*` runs, `paired_central_gbdt.json` and the Table 13 build were already present in the repository zip (made 2026-10-27/28), so this patch only brings them into the text.
New `reproduce/final_comparison.py` (Table 14a-14f from `results.json` only; bold = best federated per metric, underline = best central per metric, ties marked; FLICS's reported rows added unranked at the two FLICS scales; accuracy and F1 at the validation-chosen F1-maximising threshold, whereas FLICS tunes on its test set). 19 new unit tests; four injected-bug revert, fail, restore, pass proofs (lowest-instead-of-highest mark, pooled fed+central marking, sample sd, wrong-seed interval). 52 torch-free package tests pass; every number in the new text was checked by script against the tables and result files. All earlier tables rebuild byte-identical.
Findings recorded honestly: the federated GBDT is the best federated model on all three metrics in all six settings; our F1 (0.44-0.47 at FLICS's scales) and accuracy are below FLICS's reported values and the text draws no conclusion (prevalence, label and threshold procedure differ); FLICS's centralized XGBoost is nominally above our pooled models at both scales (intervals overlap); the pooled-GBDT control shows no resolvable federation cost in five of six settings and an unresolved -0.0102 (seed 42; seed mean -0.0056) at 3 hospitals. Details: docs/paper_track_v21.md.

## Paper-track v20: pooled run of our own GBDT (gbdt_central) in six settings -- 2026-10-25 (sandbox-verified; real runs completed 2026-10-27/28, see v21)

Requested by the user: a pooled GBDT next to the pooled LR, MLP and LightGBM, in every setting including all stays and the 3/10-hospital federations (central MLP already existed everywhere). Default-off arm `gbdt_central` (one client, reuses the federated winner's hyperparameters), `paired_central_gbdt.py`, central rows and Table 13. 15 new tests (12 unit, 3 integration); eight injected-bug proofs. Runs and commands: docs/paper_track_v20.md. Text numbers wait for the runs.

## Paper-track v19: FLICS as the only base paper; submission-copy script -- 2026-10-24

User decision: the second eICU reference study is dropped from the research entirely. Removed from the text, tables (3b, 5), figure 2, abstract/discussion, `reference_results`, config, probe script (renamed `paper_hospital_size_probe.py`), tests and docs; the reference-choice disclosure stays. New `scripts/make_submission_copies.py` (+5 tests, four injected-bug proofs). User ran the final suite on v18: 689 passed, 2 skipped, 0 failed, split manifest intact. Details: docs/paper_track_v19.md.

## Paper-track v18: paired label effect in Table 12, final wording -- 2026-10-24

User ran the five label checks with `--paired-bootstrap 2000`: label effect +0.0127 to +0.0343, every paired interval excludes zero. Table 12 (script values, checked cell by cell), results text sections 5/2/4, abstract and discussion (final FLICS-comparison and limitations wording). Remaining: final full suite, final package check, delete section 8 and the draft notes. Details: docs/paper_track_v18.md.

## Paper-track v17: paired interval for the label effect (script option) -- 2026-10-24

Optional `--paired-bootstrap N` in `scripts/paper_label_variant_check.py` (default off, one shared resample for both labels; +5 tests, three injected-bug proofs). User to run the five label checks with it and send the outputs; Table 12 and section 5 are then updated.

## Paper-track v16: 6 h GBDT rerun as headline, Table 12 label check, README and abstract/discussion draft -- 2026-10-24

User runs (2026-10-24): label check at FLICS scale (four runs), 6 h documented-sepsis check, 6 h GBDT rerun with max_rounds=1200 (208 min), full suite on v15 (679 passed, 2 skipped, 0 failed; the split manifest was intact afterwards), encoder set back to lstm. v16: rerun selected by validation AUROC (0.8603 against 0.8595; test +0.0006 [+0.0001, +0.0011]); Tables 3a (caption), 4b, 8, 9 and new Table 12; `reproduce/paired_sepsis6h.py` (+5 tests, two injected-bug proofs); README section; analyze_ablation_results.py moved to scripts/; Fed-THOR_abstract_and_discussion.md (draft). Details: docs/paper_track_v16.md.

## Paper-track v15: FLICS-scale federations, Tables 11a-11c, correction to a v14 remark -- 2026-10-24 (real-run results; analysis on saved predictions)

Real runs (user, 2026-10-22): `flicsstyle_top3` (3 hospitals, 12,911 stays) and `flicsstyle_top10` (10 hospitals, 33,179 stays) with every arm trained on the federation only, plus the label check. v15 adds `reproduce/paired_flicsscale.py` (paired comparisons, training-pool contrast, guards against mismatched prediction folders), Tables 11a-11c and results-text updates (details: docs/paper_track_v15.md). A federated LR trained on the 3 hospitals scores 0.8523, so the v14 remark that 0.8702 matched FLICS's 0.87 is withdrawn. Verified: 16 new tests; five injected-bug proofs (fail, restore, pass); existing 408 tests' outcomes unchanged (sandbox without torch). The first top-3 zip held the top-10 predictions; the guards exist because of that.

## Paper-track v14: FLICS-comparison investigation, strict-only Table 3a, same-patient Table 10 -- 2026-10-21 (analysis on saved predictions; replication not run)

Triggered by the question why neither federated model surpasses FLICS on mortality and whether the all-stays 0.9066 shares FLICS's settings (it does not: all 207 hospitals, 199,108 stays, test n = 29,968).
Findings (details in docs/paper_track_v14.md): the +0.005 over FLICS's 0.87 is 0.3 sd of sampling noise on a FLICS-style 1,000-patient test (our federated GBDT scores 0.874 +- 0.014 there); FLICS's hospital tables do not match real eICU hospitals;
APACHE scores add nothing (pooled LightGBM 0.9092 -> 0.9096, all stays); the federated gap to pooled LightGBM is not significant on the 10 or 3 largest hospitals. Method-over-method on identical patients the GBDT beats a FLICS-style federated LR by
+0.0207 [+0.0159, +0.0262] (all), +0.0218 (10 largest), +0.0161 (3 largest).
Changes: Table 3a is strict-only and labelled a cross-setting comparison (the all-stays 'ahead' verdicts misled and are removed); new Table 10; reproduce/paired_subsets.py; scripts/paper_label_variant_check.py (+3 tests, revert proof);
configs/paper_flicsstyle.yaml (optional, cache-key verified, not run); results section rewritten on this point. Not verified on real data: the label check and the FLICS-scale runs.

## Paper-track v13: federated GBDT on sepsis -- real-run results (24 h and 6 h) -- 2026-10-21 (real-run verified)

Runs (user's machine, uploaded results + predictions): `fedgbdt_sepsis24_final` (102 min) and `fedgbdt_sepsis6h_final` (147 min; the user's folder name is `fedgbdt_sepsis6h_fina` and must be renamed), 3 seeds, 151 label-valid hospitals as clients.
Federated GBDT: 24 h 0.8599 +- 0.0005 [0.8548, 0.8660], 6 h 0.8566 +- 0.0002 [0.8525, 0.8617]. Paired over the same test patients: +0.0181 [+0.0150, +0.0211] / +0.0177 [+0.0149, +0.0204] over the federated MLP;
-0.0024 [-0.0035, -0.0012] / -0.0023 [-0.0033, -0.0014] against pooled LightGBM; ~88% of the MLP-to-LightGBM gap closed; wins the validation headline rule at both windows (0.8631 vs 0.8465; 0.8595 vs 0.8422).
Findings: (1) the 6 h run hit the 600-round limit on all three seeds (all other GBDT runs early-stopped); validation gain over the last 100 rounds was +0.0003 to +0.0008, so the effect is small; disclosed, not re-run.
(2) Documented-sepsis check (24 h, seed 42): 0.7683 [0.7585, 0.7787] vs 0.7669 (federated MLP) and 0.7721 (pooled LightGBM): no measurable gain on documented sepsis despite +0.018 on the proxy label; quintile rates 5.2/4.1/9.7/18.6/40.0%.
The 6 h check has not been run. (3) `build_tables.py` now warns loudly when a federated-GBDT run folder is missing (a mistyped folder name previously would have dropped the row silently).
Changes: build_tables.py (Table 8 GBDT 24 h row, Table 9 sepsis rows, missing-run warnings), regenerated tables, final results section (sepsis text, claims, threats, Table 4c row).

## Paper-track v12: federated GBDT on the sepsis task (path verified, runs pending) -- 2026-10-20

The `fed_gbdt` arm was verified on the sepsis path (label-valid hospital filter, unknown labels) with a new integration test; no code change was needed. The test also exposed a gap in the mortality wiring guard
(training on every row of every hospital passed silently), now closed by asserting `extra.n_train == split_sizes.train` for both tasks; proven by revert -> fail -> restore -> pass. `build_tables.py` adds a federated-GBDT row to
Tables 4a/4b only once `fedgbdt_sepsis24_final` / `fedgbdt_sepsis6h_final` exist in the summary (absent runs change nothing). Runs: configs/paper_sepsis.yaml (24 h, strict landmark, bundle e49025...) and
configs/paper_sepsis_early.yaml (6 h, 60-minute bins, bundle c92ebe...), both with `--tasks sepsis --feature-sets sepsis_clean --arms fed_gbdt --seeds 42,43,44`. Not yet run on real data.

## Paper-track v11: final federated-GBDT results, FLICS as base paper, regenerated tables and figures -- 2026-10-20 (real-run verified)

Real runs (user's machine, uploaded results.json + predictions): `fedgbdt_strict_final` (97 min) and `fedgbdt_allstays_final` (445 min), 3 seeds each, same split and seeds as the final MLP/LightGBM runs.
Federated GBDT: strict 0.8746 +- 0.0005 [0.8662, 0.8818], all stays 0.9066 +- 0.0009 [0.9020, 0.9125]; paired +0.0129 [+0.0088, +0.0170] and +0.0151 [+0.0119, +0.0182] over the federated MLP,
-0.0050 [-0.0067, -0.0031] and -0.0026 [-0.0035, -0.0016] against pooled LightGBM; wins the validation-based headline rule on both protocols. Search gave no test benefit (strict default 0.8748 vs winner 0.8742).
Cost: about 11 MB per client per round (strict), 653-730 GB per model across clients (sparse upper bound) against 15-17 GB for FedAvg-MLP.
Base-paper position: FLICS (strict: level with the 3-hospital 0.87, ahead of the 10-hospital figures).
Changes: `reference_results` now FLICS only (DP-IG-FL and Kerkouche removed; test updated, revert->fail->restore->pass proven); results package (`parse.py` locates the summary itself, `build_tables.py` adds
fed_gbdt rows, the FLICS table and the cost table, `figs.py` adds fed_gbdt and labelled reference lines), regenerated tables and figures; final results section.
Verified: regenerated tables differ from the previous package only by the new rows, replaced reference tables and cost table; every pre-existing summary row is byte-identical.

## Paper-track v10: federated histogram GBDT + hospital-size cohort filter -- 2026-10-18 (sandbox-verified; real run pending)

Requested after the base-paper check showed the strict federated MLP (0.8617) is behind FLICS's 3-hospital 0.87. Additive, default-off.
No promise of outcome: the strict ceiling is pooled LightGBM (0.8796).

- `src/paper/fed_gbdt.py`, arm `fed_gbdt` (`paper.fed_gbdt.*`): horizontal federated histogram GBDT (clients send gradient/hessian histograms; server sums). Equals a pooled histogram GBDT
  (tested). Train-only edges/splits; validation only selects round/trial. No privacy guarantee; communication much larger than FedAvg-MLP (reported in results.json).
- `src/paper/cohort_filters.py`: `paper.cohort_filter.min_hospital_patients` (0 = off) restricts every arm to hospitals above a size; `scripts/paper_hospital_size_probe.py` (read-only) picks the threshold.
- `configs/paper_fedgbdt.yaml` (strict cohort; cache-key-identical to the final strict run).
- `reference_results` corrected: Kerkouche and DP-IG-FL marked not eICU-only. Docs: `docs/paper_track_v10.md`.
- Verified: 14 + 9 unit and 4 integration tests; 5 hand-performed revert->fail->restore->pass proofs; 44 existing paper tests identical on original vs patched.
- Not verified: real data; torch-dependent tests (no torch in the sandbox); real wall-clock.

## Paper track delivered (sandbox-verified; real run pending) -- 2026-10-08

Additive `src/paper/` + `run_paper.py` + `configs/paper*.yaml` + `docs/paper_track.md`. Motivation: the 2026-09 flat/sampling/graph results
(mortality AUROC ~0.72-0.75, graph < flat, sampling < no sampling, mortality << sepsis) were traced to pipeline design, not the task:
5 full-batch steps per hospital for encoders, unweighted "weighted_bce", per-hospital unaligned encoders pooled into one classifier, graph
heads trained 10 steps on tiny local sets, whole-stay sequences with discharge-derived lengths (undefined prediction time), sepsis-label
ingredients present in the inputs, inconsistent train/val thresholds, no raw-feature baseline/CIs/seeds. See docs/paper_track.md.

Delivered: first-24h window; leakage-safe wide feature builder from all 13 tables (train-only vocabularies); mortality feature sets
(full / no_apache_scores) and sepsis_controlled/uncontrolled; arms logreg / tuned LightGBM / central MLP / FedAvg MLP / FedAvg+Graph none|A|B;
calibration + one threshold rule + bootstrap CIs + paired tests + per-hospital metrics + patient-level predictions + report.
Fixes in existing code: LightGBM restored to the classifier registry (classifier choice stays config-driven; fixes 3 failing tests);
`masked_bce_loss(pos_weight=None)` option (default unchanged); `ResultsWriter` CSV rebuilt from per-record JSON with a union header
(fixes the silent column-shift bug; `rebuild_dedicated_csvs()` heals old CSVs).
Status: sandbox-verified incl. hand mutation checks; NOT yet run on the real 199,108-patient dataset.

**First real result (user's run, 2026-10-08, LightGBM/LR ceiling check, mortality, 1 seed, test n=29,968):** `full`: LightGBM AUROC 0.9096
[0.9042, 0.9148], AUPRC 0.608, F1 0.558, Brier 0.0528, ECE 0.0035, per-hospital macro AUROC 0.9009; logistic regression 0.8892 (paired
dAUROC -0.0204, p=0.002); APACHE IVa benchmark 0.8642 (n=20,503 with a prediction). `no_apache_scores`: 0.9092 / 0.8878 -- APACHE-derived
scores add ~0.0004, so raw measurements carry the signal. Pending audit: short-stay truncation signal (strict 24h landmark run, LOS-stratified
AUROC), end-of-life tokens (`no_apache_no_eol`). Sepsis and the neural/federated/graph arms have NOT yet been run on real data.
Added 2026-10-08 (third pass): CLI `--feature-sets`, `--seeds`, `--arms` now runs EXACTLY the listed arms; results.json/report/CSV written after every finished experiment;
report notes when the paired-comparison reference arm is absent; staged commands in docs/paper_track.md. Added 2026-10-08 (second pass): `lightgbm_sampling` arm, LightGBM gain-importance export, `exclude_name_regex` feature sets, `src/paper/subgroups.py`
+ `scripts/paper_subgroup_analysis.py`, configs paper_landmark24 / paper_sepsis / paper_sepsis_early / paper_sampling.

**Stage 1 real result (2026-10-10):** all-stays LightGBM 0.9092 = 0.8815 (stays >=24h) / 0.9528 (stays <24h); APACHE IVa on the same 20,503 test patients 0.8618
(ours 0.9099, +0.048 [0.042, 0.055]); per-hospital median AUROC 0.9126 over 123 hospitals. Strict 24h landmark (131,695 patients): LightGBM 0.8799 [0.8721, 0.8874],
AUPRC 0.491, F1 0.478; LR 0.8603. Decision: strict landmark is the primary protocol; all-stays secondary. The end-of-life robustness run was vacuous (identical
feature count and metrics) -> `scripts/paper_eol_audit.py`. Fourth pass: `--set`, LightGBM eval_X/eval_y (no deprecation noise), numpy warning cleanup.

**Real results 2026-10-10 (strict 24h landmark, 1 seed, full data):** mortality LightGBM 0.8799 / central MLP 0.8699 / LR 0.8603 / FedAvg MLP 0.8306 (graph none/A/B
0.8306/0.8316/0.8316); APACHE IVa on the same patients 0.8281 (ours +0.054). Sepsis 24h controlled LightGBM 0.9097 (uncontrolled 0.9930; FedAvg MLP 0.8206), 6h controlled
0.9067 (uncontrolled 0.9765; FedAvg 0.8207). Sampling: no gain (0.8769 / 0.8786 vs 0.8799). EOL strings: ~1.4% of deaths (mostly alcohol/drug withdrawal matches).
Open: FedAvg MLP 0.04-0.07 below central (val AUROC peaks round 6-8 then declines), graph neutral, "controlled" sepsis still contains static infection/organ-failure signal.
Fifth pass: FedAvg stabilisers (`lr_decay`, `max_local_steps`, `server_momentum`/`server_lr`), `fedavg_logreg` arm, `neural.use_sequence` switch, sepsis ladder feature sets
(`sepsis_deconfounded`, `sepsis_no_labs`), staged tuning protocol in docs/paper_track.md.

**First sepsis reports (2026-10-10):** controlled-model top features = paCO2 (~13%), hospital size/region (~7.5%), activetx, graftcount, BUN, albumin, paO2; pooled LightGBM 0.9097 vs
per-hospital macro 0.8676; FedAvg per-hospital macro 0.8393 > its pooled 0.8206 (cross-hospital score offsets). Sixth pass: ladder regex fixed (ABG family; the v5 regex had missed paCO2),
`sepsis_no_hospital`, per-hospital intercept calibration (runner + `scripts/paper_local_calibration.py`), `scripts/paper_sepsis_label_audit.py`.

**5A results (2026-10-11):** best federated mortality variant on validation = v4 (lr 1e-3, E=2, FedProx mu 0.01, tab-only): val 0.8374, test 0.8402 [0.8306, 0.8487], with per-hospital
calibration 0.8469; fedavg_logreg 0.8334 (0.8478 calibrated); v1/v2/v3 0.8341/0.8341/0.8362. Central: LightGBM 0.8799, MLP 0.8684 tab-only (0.8699 with the GRU branch: no multimodal gain),
LR 0.8603. APACHE IVa on the same 15,151 patients 0.8309 (the earlier 0.8281 came from not preferring the IVa row; fixed). Sepsis ladder: deconfounded 0.8978 (24h) / 0.8972 (6h);
no-labs 0.8014 / 0.8131; `sepsis_no_hospital` pending. Seventh pass: `scripts/paper_collect_results.py`, APACHE IVa preference in the subgroup script.

**Round 2 (2026-10-11):** FedProx mu 0.1 is the best federated mortality setting so far: test 0.8506 [0.8417, 0.8589] (val 0.8479; per-hospital calibrated 0.8559); mu 0.05 0.8470, 0.01 0.8402;
federated LR mu 0.1 0.8453. Graph arms neutral. Sepsis ladder (24h LightGBM): 0.9930 / 0.9097 / 0.8978 / 0.8793 / 0.8014; per-hospital calibration lifts FedAvg sepsis 0.8206 -> 0.8863.
Label audit: proxy vs documented sepsis precision 29% / recall 50%; 46/188 hospitals have zero positives; Spearman(rate, medication coverage)=0.71. Eighth pass: `paper.sepsis_hospital_filter`
(sepsis only, default on), `sepsis_clean` feature set, `src/paper/label_validity.py`, corrected `scripts/paper_sepsis_label_audit.py` (unknown labels excluded, coverage bins, threshold options).

**Round 3 + all-stays + valid-sepsis (2026-10-12):** federated mortality (strict, tab-only, 1 seed) FedProx mu 1.0: test 0.8611 [0.8521, 0.8688], val 0.8571 (central MLP 0.8684); mu 0.3: 0.8567; federated LR mu 1.0: 0.8535.
All-stays mu 0.1: federated MLP 0.8822 [0.8758, 0.8882] (0.8879 calibrated), LR 0.8773; central MLP 0.8989. Sepsis on 151/207 label-valid hospitals: clean LightGBM 0.8621 (24h) / 0.8591 (6h), per-hospital 0.8387 / 0.8372;
hospital-offset effect gone (pooled == calibrated). Coverage bimodal (49 hospitals with proxy rate 0.000). graftcount is constant 3 for ~98% of rows. Ninth pass: `scripts/paper_documented_sepsis_check.py`.

## Cross-cutting: Phase 4/5 architecture options (RNN/GRU/Transformer, MLP) — 2026-09-06

User request, not a numbered phase: extend Phase 4/5 to support
multiple encoder architectures, selectable via config, with each
architecture's output going to its own file -- while guaranteeing zero
impact on every other phase.

**Filled in, not invented: `modalities.timeseries.encoder` and
`modalities.tabular.encoder`.** Both config keys already existed as
unused stubs (`encoder: lstm` and `encoder: autoencoder  # autoencoder
| mlp` -- the tabular one had ALREADY anticipated `mlp` in its own
comment) before this change -- confirmed via a direct grep that
neither was ever read anywhere in the codebase. Same "found a stale
stub, filled it in correctly rather than inventing a parallel new
mechanism" pattern as the `training:` stanza fix before Phase 11.

**New sibling encoders, existing ones untouched:**
`src/modalities/timeseries/{rnn,gru,transformer}_encoder.py` and
`src/modalities/tabular/mlp_encoder.py` -- `LSTMTimeSeriesEncoder` and
`TabularAutoencoder` themselves were never modified. All three new
timeseries encoders share LSTM's exact `forward(x, lengths) -> (batch,
hidden_dim)` contract (RNN/GRU via the same `pack_padded_sequence`
handling; Transformer via a `src_key_padding_mask` instead, the
Transformer-native equivalent, plus a fixed sinusoidal positional
encoding and per-patient extraction at their own last real timestep --
matching the RNN-family's "final hidden state" semantic exactly, so
all four extract a patient embedding the same conceptual way).

**The core safety guarantee, proven twice (once per modality):**
`timeseries_embeddings.npz` / `tabular_embeddings.npz` are NEVER
touched by a non-default encoder -- confirmed via dedicated
integration tests checking the canonical file's bytes AND mtime are
unchanged after running every alternate architecture, each with its
own hand-performed revert→fail→restore→pass proof (temporarily made
every encoder write to the canonical filename; both tests failed with
the corrupted file containing the wrong architecture's output;
restored, reconfirmed passing).

**Confirmed by direct code inspection, not assumed: Phase 11/12/13
cannot be affected by this change at all.** Their training code never
reads `phase4_result`/`phase5_result`'s `embeddings_path` -- only
Phase 7's fusion does, and Phase 7's own fusion logic is already
completely architecture-agnostic (reads whatever `embeddings_path`
points to, sizes itself dynamically off the array's actual shape).
Phase 11/12/13 construct their own fresh `LSTMTimeSeriesEncoder`/
`TabularAutoencoder` directly from the raw normalized tensors,
regardless of what `encoder` was set to for a Phase 4/5 standalone
run.

**A real bug found and fixed during this work, not hypothetical:**
`TransformerTimeSeriesEncoder` validated `input_dim` and the
`hidden_dim % num_heads` divisibility but never validated `hidden_dim`
itself -- `hidden_dim=0` fell through to a raw `ZeroDivisionError`
deep inside the positional encoding instead of a clean `ValueError`,
caught by the encoder's own unit test suite before this was ever
delivered.

**Verified (sandbox, mock data):**
- 21 unit tests (`test_timeseries_encoder_variants.py`, shared across
  RNN/GRU/Transformer via a common contract mixin) + 9 unit tests
  (`test_mlp_encoder.py`) + 9 integration tests
  (`test_modality_architecture_options.py`) = 39 new tests.
- **Revert → fail → restore → pass proven by hand, four separate
  times**: (1) the Transformer's padding mask polarity inverted --
  confirmed the dedicated masking test failed, restored, reconfirmed
  passing; (2) the timeseries canonical-filename branch collapsed to
  always write `timeseries_embeddings.npz` regardless of encoder --
  confirmed the untouched-file test failed with the log itself showing
  RNN output overwriting the LSTM file, restored, reconfirmed passing;
  (3) the same regression, tabular side (`mlp` overwriting
  `tabular_embeddings.npz`) -- confirmed, restored, reconfirmed
  passing.
- **The complete pre-existing test suite (315 tests) re-run and
  passing throughout** -- this is the primary proof for the user's
  explicit requirement that every other phase run unaffected. Final
  count: 354 passed, 2 skipped, zero regressions.
- The real CLI workflow itself exercised end-to-end (not just unit-
  level mocks): edited `configs/base.yaml`'s `encoder:` lines exactly
  as a real user would, ran `python run_pipeline.py --config
  configs/debug.yaml --phase 4/5` via subprocess, confirmed the
  correct architecture-specific files were produced with correct
  shapes and log messages, confirmed the default (unedited) config
  still produces byte-identical canonical output through Phase 7.
- **Not yet run against the real 199,108-patient / 208-hospital
  dataset.**

## Senior-review analysis of the first real 3-hospital pilot + a genuine coverage gap found and closed — 2026-09-13

The user's real 3-hospital pilot (73, 92, 384 — 1500 patients, 350/75/75
split each) completed cleanly: zero overlap, correct per-hospital
stratification, all three phases trained with zero hospitals skipped.
Asked to review it "like a senior research scientist," which surfaced
one structural point worth recording precisely (not a bug) and one
real, concrete gap that was fixed the same day.

**Structural point, not a bug**: hospital 384 showed 0.0% sepsis-broad
prevalence across all three splits (vs. 33-41% and 23-27% for the
other two hospitals). Flagged for the user to spot-check against that
hospital's raw antibiotic/medication/microLab coverage before drawing
conclusions from it at full scale — real, extreme inter-hospital
heterogeneity in eICU's sepsis-relevant documentation is already
well-established in this project (see the sepsis_label_methodology
history), so this is plausible as a genuine hospital characteristic,
not assumed to be a processing artifact either way.

**A real, concrete coverage gap, found and fixed**: of the three
baselines Phase 13 exists specifically to compare (local / centralized
/ federated — the entire point of the "reset frequency" framing), the
validation report was only evaluating ONE of them (centralized).
Phase 11's own section reported a single LOCAL ROUND's performance,
which is a genuinely different, smaller-budget quantity than Phase
13's matched-epoch-budget "local-only" baseline -- not a substitute
for it. And Phase 12 (federated) had NO evaluation section at all.

**Root cause**: FedAvg's per-round local training produces predictions
from each hospital's POST-LOCAL-TRAINING weights (that round's
contribution to the average) -- never from the FINAL AGGREGATED
weights FedAvg actually outputs as its result. Evaluating the
aggregate requires one dedicated extra frozen-inference pass that
nothing already computed. Likewise, Phase 13's local-only baseline WAS
already being trained and checkpointed correctly (with per-hospital
predictions, same as Phase 11) -- it just was never plugged into the
report's evaluation step.

**Fixed**: new `evaluate_final_model_on_full_graphs()`
(`src/split_aware/evaluation.py`) loads a saved checkpoint (Phase 12's
final aggregated model) and runs one frozen forward pass per hospital
over that hospital's full graph, in the same per-hospital prediction
shape the rest of this module already consumes. `run_ablation.py`'s
`run_federated_split_aware()` now calls this for Phase 12's result, and
also evaluates Phase 13's `local_only` hospital_results (previously
computed, never evaluated) via the already-existing
`evaluate_hospital_results()`. The report now shows all three baselines
side by side for both tasks, both splits.

**Verified**: 2 new integration tests
(`test_evaluate_final_model.py`), including a direct check that
predictions cover every patient (train+val+test) and that a bad
checkpoint path raises clearly rather than silently returning nothing.
Re-ran the full 4-hospital mock CLI smoke test end to end (6.6s, zero
failures) and confirmed by inspection that the regenerated report now
shows LOCAL-ONLY, FEDERATED, and CENTRALIZED evaluation blocks for both
tasks -- previously only CENTRALIZED appeared. Full suite: 464 passed,
2 skipped (up from 462), same 4 pre-existing unrelated failures, zero
regressions.

**Immediate next step**: re-run
`python run_ablation.py --config configs/subset.yaml` against the real
3-hospital data (no config changes needed) and confirm the regenerated
`validation_report.md` now shows all three baselines' evaluation
metrics. Once that's clean, the split-aware subset validation this
project's own standing rule requires before scaling up is genuinely
complete -- full 208-hospital run remains explicitly deferred until
then.



Direct follow-up to the ablation framework above, per the user's
explicit next-priority request: "make the entire experimental pipeline
split-aware and leakage-safe." Twelve numbered requirements confirmed
2026-09-11; the ones with real engineering weight are covered here.
`run_phase13_baselines()` and everything numbered before it (Phases
1-13's original, full-cohort behavior) are untouched -- every change
below is additive and behind a new, optional `train_patient_ids`
parameter that defaults to `None` and reproduces the exact original
behavior when omitted (verified: the full pre-existing suite passes
unmodified throughout this whole delivery).

### 1. A leakage gap found in Phase 4/5 that the ablation framework had NOT closed

Auditing "every learned/fitted component" (requirement #2) surfaced a
real gap: Phase 4's per-hospital time-series normalization and Phase
5's per-hospital normalization + categorical vocabulary determination
were both fit from a hospital's FULL patient population (train+val+
test) -- even in the ablation framework delivered the day before,
whose FLAT MODEL encoder weights were already correctly trained
train-only. "Fit representation" (requirement #3) means the whole
representation pipeline, not just the neural encoder's own weights --
a normalization mean/std or a one-hot vocabulary computed with val/test
included is still leakage, independent of how the downstream model is
trained.

**Fixed with new, additive sibling functions** (originals completely
untouched, verified via dedicated regression tests):
`fit_transform_local_split_aware()`
(`src/data/temporal/normalization.py`) and
`fit_transform_tabular_local_split_aware()`
(`src/data/tabular/normalization.py`) fit each hospital's
StandardScaler from train-split rows only, then transform that
hospital's full population with the frozen, train-fit scaler.
`run_phase4_timeseries_modality()`/`run_phase5_tabular_modality()` both
gained an optional `train_patient_ids` parameter (default `None`,
original behavior byte-identical) that routes to these new functions
and, for Phase 5, also pre-filters `cohort_patients`/diagnosis shards
to train-only rows before calling the EXISTING (unmodified)
`determine_full_cohort_vocabularies()`/
`determine_diagnosis_vocabulary_from_shards()`.

### 2. A design correction made mid-flight, caught by a test built specifically to catch it

The first working version of split-aware Phase 11 training kept the
GNN's FULL hospital graph in every epoch's forward pass and masked
only the mortality/sepsis LABELS to NaN for non-train patients (the
standard semi-supervised/transductive GNN regime used by published
benchmarks). A dedicated revert-style test
(`test_phase11_split_aware.py::test_split_aware_training_is_NOT_
sensitive_to_a_non_train_patients_raw_values`) proved this
insufficient: corrupting a non-train patient's raw tabular values
still changed a train NEIGHBOR's trained weights, because the two
shared a graph edge -- labels never leaked, but raw FEATURE values
could, through message-passing. That's a real violation of
requirement #2's literal wording ("validation/test patients must only
pass through frozen components").

**Fixed by switching to a TRAIN-ONLY SUBGRAPH for the gradient-computing
forward/backward pass** -- new `_build_train_only_subgraph()` helper
in `run_pipeline.py` (shared by Phase 11's `_run_local_training_round()`,
Phase 13's `_run_centralized_sequential_training()`, and the ablation
framework's `graph_embedding_ablation_runner.py`, which had the exact
same design flaw and was fixed the same way). A non-train patient's row
is never even present in the tensors the model touches during
training -- no pathway, message-passing or otherwise, for it to
influence a gradient. This is also, not incidentally, the textbook
GraphSAGE INDUCTIVE-learning pattern (train on a subgraph, run
inference later on an expanded graph including new nodes), not a
bespoke restriction. AFTER training, one frozen (`eval()`, `no_grad()`)
forward pass over the FULL graph produces val/test predictions against
the now-fixed weights -- genuinely "passing through frozen components."
Both the "corruption changes weights without split-awareness" and
"corruption does NOT change weights WITH split-awareness" directions
are proven by dedicated tests for both Phase 11 and the graph ablation
runner.

### 3. Phase 11/12/13, all split-aware, all writing to separate, `_splitaware`-suffixed directories by default

- **Phase 11** (`run_phase11_local_training`): new `train_patient_ids`/
  `checkpoint_dir_name` parameters. `checkpoint_dir_name` defaults to
  `phase11_local_splitaware` (vs. the original `phase11_local`) the
  moment `train_patient_ids` is given -- safe by default, a caller
  cannot silently overwrite the original full-cohort checkpoints by
  forgetting to pass a directory name.
- **Phase 12** (`run_phase12_fedavg`): same pattern
  (`phase12_fedavg_splitaware`). Two FedAvg-specific correctness fixes:
  (a) round-1 checkpoint reuse is redirected to only ever reuse a PRIOR
  split-aware Phase 11 run, never the original -- reusing the original
  would silently reintroduce the exact leakage this effort removes;
  (b) aggregation weights use each hospital's TRAIN-only patient count
  (a new `n_train_patients` field), not its full population, since
  that's what actually went into that hospital's update.
- **Phase 13** (`run_phase13_baselines`): same pattern
  (`phase13_baselines_splitaware`) for both new baselines (local-only,
  centralized-sequential -- the latter via the same train-only-subgraph
  fix per hospital-visit). The federated reference is also redirected
  to the split-aware FedAvg manifest -- comparing a split-aware
  local/centralized baseline against the ORIGINAL full-cohort federated
  result would not be the apples-to-apples "reset frequency"
  comparison this baseline suite exists to isolate (dedicated test
  confirms it does NOT fall back to the original manifest when only
  that one exists).

### 4. Research-grade evaluation reporting (requirement #9)

`src/classifiers/boosting_classifiers.py` gained `sensitivity`
(explicit alias for recall), `specificity`, `brier_score`
(calibration), and a `threshold` parameter on
`compute_binary_classification_metrics()`. New
`select_threshold_from_validation()` picks the F1-maximizing threshold
from VALIDATION data only, applied to test -- closes a real gap
flagged in the 2026-09-10 pilot review (a fixed 0.5 cutoff produced
`precision=recall=0` for an otherwise-reasonable model at 8% mortality
prevalence). A dedicated adversarial test proves test labels are never
consulted when selecting the threshold. This benefits the 72-config
ablation automatically (same shared function) and a new
`src/split_aware/evaluation.py` applies the identical val-then-test
protocol to Phase 11/13's own MLP-head/centralized predictions
(aggregating each hospital's per-patient logits, which
`_run_local_training_round()`/`_run_centralized_sequential_training()`
now also return when split-aware).

### 5. Single-command orchestration, extended -- `run_ablation.py`

Rewritten to run split generation -> split-aware Phase 4/5 -> the
72-config + graph ablations -> split-aware Phase 7/8 (reused
UNCHANGED, fed split-aware tensors, writing to a redirected
`{output_dir}/split_aware/` so the original full-cohort Phase 7/8
output is untouched) -> split-aware Phase 11/12/13 -> a comprehensive
validation report, all in one command:
```
python run_ablation.py --config configs/subset.yaml
```
`--steps split,embedding_ablation,graph_ablation,federated_split_aware`
(comma list, any subset) preserves start/stop-from-any-step;
`--reuse-phase1to3` skips re-running the expensive Phase 1/3 (Phase 4/5
are always redone fresh, split-aware -- they're cheap, seconds not
minutes); `--max-hospitals N` bounds every step for a smoke test.

### 6. Validation report (`src/split_aware/reporting.py`) -- requirement #12, exactly

Every run writes `validation_report.{json,md}` under the ablation run
directory: exact train/val/test patient counts (global and
per-hospital), per-hospital mortality/sepsis prevalence by split, an
explicit zero-overlap confirmation (computed directly from the
manifest, not assumed -- a hand-corrupted-manifest test confirms a real
overlap gets caught, not silently passed), a plain-language
confirmation of what was fit from train-only, Phase 11/12/13 results
with the new evaluation metrics, the ablation summary, and a `failures`
list (each pipeline stage is wrapped so one stage's exception doesn't
prevent the others from running or the report from being written).

### Verification status

**33 new tests**, all against real (not mocked) mechanics:
`test_normalization_split_aware.py` (9), `test_tabular_normalization_
split_aware.py` (5), `test_phase4_5_split_aware.py` (4, real Phase
1-2-3-4-5 mock pipeline), `test_phase11_split_aware.py` (5, including
the central train-only-subgraph leakage proof), `test_phase12_split_
aware.py` (4), `test_phase13_split_aware.py` (4), a new leakage test
added to `test_graph_embedding_ablation_runner.py` (1), classifier
threshold/sensitivity/specificity/Brier tests (7), `test_split_aware_
evaluation.py` (7), `test_split_aware_reporting.py` (5) -- some
double-counted across categories above; **462 total passing** project-
wide, zero regressions (confirmed by re-running the complete
pre-existing suite unmodified after every change in this delivery, not
just at the end).

**CLI-smoke-tested for real**, not just pytest: a 4-hospital,
160-patient mock run through ALL FOUR steps in one command completed in
5.2s with zero failures, produced a correct 112/24/24 (70/15/15) split
with zero overlap, and a validation report showing real per-hospital
prevalence and real (if tiny-scale-noisy, as expected) AUROC/AUPRC/
sensitivity/specificity/Brier numbers for Phase 11 and Phase 13's
centralized baseline -- this is what caught the `run.output_dir`
resolution and the `_run_local_training_round`/`_run_centralized_
sequential_training` prediction-shape details a pure-pytest run
wouldn't have exercised.

### What this does NOT yet do / immediate next step

- **Not yet run against real data.** Sandbox/mock-data-verified only,
  per this project's standard practice for a newly-delivered phase.
  **Immediate next step, per the user's explicit instruction**:
  ```
  python run_ablation.py --config configs/subset.yaml
  ```
  on `configs/subset.yaml` updated with at least 3 real eligible
  hospital IDs (the previous pilot found hospital 147 ineligible --
  only 2 of the 3 configured hospitals were actually used). Report back
  the exact contents of the generated `validation_report.md`.
- Full 208-hospital run remains explicitly deferred until this
  real-data subset validation passes, per the user's instruction.
- FedProto: still intentionally not implemented (unchanged from the
  ablation framework's decision #5) -- `federated.algorithm` correctly
  raises `NotImplementedError` rather than silently running FedAvg.
- The 72-config grid was NOT reduced to save runtime (requirement
  #10) -- Transformer stays in `ablation.ts_models`; `ablation.subset`
  remains the config-only mechanism for running the grid in batches
  while preserving the same split/seed/config/output structure.



New, additive work layered on top of the already-closed 13-phase
pipeline, requested as a research-grade ablation study. **A genuine
scope tension, flagged rather than silently resolved**: the request
asked for this to be "Phase 13 only," but also required a genuine
held-out train/val/test split with retraining (not evaluating the
existing full-cohort-trained Phase 11-13 models) -- which is precisely
what `docs/baselines.md` had already scoped as Phase 14's job ("No
metric computation, no comparison... needs a train/val/test split,
which doesn't exist anywhere in this project"). Resolved with the user
(2026-09-10): tracked as its own entry, not overloading the closed
Phase 13 table row above, and NOT the general-purpose Phase 14 (global
+ per-hospital evaluation) either -- a narrower, ablation-specific slice
of that work. `run_phase13_baselines()` and everything before it in
`run_pipeline.py` are completely untouched.

**Execution order, confirmed with the user before building (decision
#8)**: split -> TS/TAB embedding generation -> fusion -> 72-config
frozen-embedding classifier ablation -> graph construction/GNN ->
graph-embedding classifier experiments -> THEN federated learning
(Phase 11/12/13's existing, unmodified mechanics). All of the new work
below runs BEFORE Phase 11 ever starts.

### 1. Permanent split manifest -- `src/data/splitting/patient_split.py`

Patient-level 70/15/15 split WITHIN each hospital (never a
hospital-level split), stratified on mortality by default, with a
documented, logged fallback to unstratified for any hospital too small
or too imbalanced to stratify. Hard invariant enforced and tested: every
patient lands in exactly one partition, checked directly (not just
assumed from the splitting logic), raising rather than warning on any
violation. ONE manifest, generated once and reused by every downstream
experiment (`build_and_save_split_manifest()` refuses to silently
regenerate an existing manifest unless `overwrite=True`).

**A real bug found and fixed while building this**: the two-step
train/val/test split (carve off test, then split the remainder) passed
a redundant second positional array into `sklearn.train_test_split()`
to co-split alongside patient IDs -- harmless when stratifying, but
that second array was `None` in exactly the unstratified-fallback case,
and sklearn tried to index into it, crashing. Invisible in the first
round of tests because the mock fixtures used there happened not to
trigger the fallback path. Found via a dedicated regression test built
specifically to force the fallback (a hospital with 10 patients, only 1
positive), fixed by deriving the remainder's own stratify labels via a
position lookup instead of co-splitting a second array.

### 2. Per-hospital PCA tabular "encoder" -- `src/modalities/tabular/pca_encoder.py`

Fit PER HOSPITAL, on that hospital's TRAIN-split rows only, never
pooled across hospitals -- same privacy boundary as per-hospital
normalization (Phase 5). Zero-pads when a hospital's train set is too
small to support the full requested embedding width. Not an
`nn.Module` -- has no gradient, unlike Autoencoder/MLP -- so a
PCA-based combo's flat model receives the PCA embedding as a frozen
input rather than containing a tabular-encoder submodule at all.
Direct regression test proves hospital-to-hospital isolation (corrupting
one hospital's data leaves every other hospital's PCA embedding
byte-identical).

### 3. Cross-Attention fusion -- `src/fusion/cross_attention_fusion.py`

New, strictly-additive sibling to Phase 7's locked concatenation
fusion (`multimodal_fusion.py`, completely untouched). Treats each
patient's two modality embeddings as a 2-token sequence and applies
standard multi-head self/cross-attention over them (residual +
LayerNorm), defaulting its output width to `ts_dim + tab_dim` so it's
a genuine drop-in alternative -- nothing downstream needs
reconfiguring based on which fusion mode is selected. Has real
trainable parameters (confirmed receiving real gradients in a
dedicated test), unlike concatenation.

### 4. Classifier wrapper -- `src/classifiers/boosting_classifiers.py`

AdaBoost/XGBoost/LightGBM behind one common `fit`/`predict_proba`
interface, plus standard metric computation (AUROC, AUPRC, accuracy,
precision, recall, F1). Confirmed with the user: **classifier training
is completely decoupled from encoder training** -- these are
non-differentiable and cannot sit inside the same backward pass the
MLP heads use, so they are fit separately, afterward, on frozen
embeddings, never federated. NaN labels are masked out (never coerced
to a class), and a split with only one class present reports
`auroc=None`/`auprc=None` (mathematically undefined) rather than a
fabricated value or a crash.

### 5. Flat multimodal model -- `src/training/flat_multimodal_model.py`

A NEW, graph-free joint model -- TS encoder (any of
lstm/rnn/gru/transformer) + TAB encoder (autoencoder/mlp, or a frozen
PCA passthrough) + fusion (concatenation/cross_attention) +
MultiTaskHeads -- deliberately without Phase 9's GNN, since the
72-config ablation is about TS/TAB/fusion quality BEFORE the graph
stage even runs. Completely separate from `src/training/local_model.py`'s
`LocalPatientModel` (Phase 11, hardcoded to LSTM+Autoencoder+GNN+heads,
confirmed byte-unmodified by a dedicated test) -- that module and
everything downstream of it (Phase 11/12/13) needed no change at all.

### 6. Config-driven grid + never-overwriting results -- `src/ablation/config_grid.py`, `src/ablation/results_writer.py`

The 72-config grid (4 TS x 3 TAB x 2 Fusion x 3 Classifiers) and the
6-experiment graph-embedding grid (1 fixed representation source x 3
classifiers x 2 tasks -- confirmed with the user: NOT the 72-combo grid
again, and NOT multiplied by GNN type or federated algorithm, per
decision #6) are both derived entirely from `configs/base.yaml`'s new
`ablation:` stanza -- changing `ablation.ts_models`/`tab_models`/
`fusion_modes`/`classifiers` immediately changes the matrix, no code
change required. `ablation.subset` lets a person run an explicit,
smaller slice (e.g. a pilot) the same way. Every experiment gets a
unique, descriptive ID (e.g.
`ts-lstm_tab-mlp_fusion-beforeca_clf-xgboost_task-mortality`,
`graph-embedding_clf-xgboost_task-mortality`) distinguishing Before-CA/
After-CA/Graph by filename. Results are written as one JSON per
experiment plus one row appended to a master CSV; re-writing the same
experiment id within a run directory raises `FileExistsError` rather
than silently overwriting, unless `overwrite=True` is passed
explicitly. `save_summary_csv()` builds the publication-ready
comparison table, sorted by test AUROC descending.

### 7. Orchestrators

`src/ablation/embedding_ablation_runner.py` (Section 1, the 72-config
grid) and `src/ablation/graph_embedding_ablation_runner.py` (Section
2): for every hospital, trains a fresh, independently-initialized flat
model (or, for the graph runner, the existing PatientGNN + MultiTaskHeads
on top of a freshly-trained default-architecture flat model) using
ONLY that hospital's TRAIN-split patients, never pooled across
hospitals and never FedAvg'd/FedProto'd (decision #2) -- then extracts
FROZEN embeddings for that hospital's full train+val+test population
using the now-fixed trained weights. The graph runner builds each
hospital's intra-hospital graph (Phase 8's `build_hospital_graph()`,
completely unmodified) over the FULL population (train+val+test, since
graph structure is unsupervised -- cosine similarity of features, no
label used anywhere in it), then trains the GNN+heads with the loss
masked to train-split rows only (val/test patients still flow through
message passing normally, exactly the standard semi-supervised/
transductive GNN training regime -- Cora/Citeseer-style -- so held-out
FEATURES inform message passing but held-out LABELS never influence a
weight). Both reuse `src/heads/multitask_heads.py`'s
`align_labels_to_patient_ids()` and `src/training/losses.py`'s masked
`compute_multitask_loss()` completely unchanged.

### 8. Single-command orchestrator -- `run_ablation.py`

```
python run_ablation.py --config configs/subset.yaml
```
runs the whole suite (fresh Phase 1-5 -> split -> both ablation
stages -> summary tables) in one command, matching the existing
pipeline's "one command, can start/stop from any phase" property.
`--steps split,embedding_ablation,graph_ablation` (comma list) selects
a subset of steps; `--reuse-phase1to5` loads already-completed Phase
1-5 output from disk (via the EXISTING `load_phase1_summary`/
`load_partition_result`/`load_phase3_result`/`load_phase4_result`/
`load_phase5_result` loaders, unmodified) instead of re-running them,
for fast iteration; `--max-hospitals N` bounds every step to a quick
smoke test without touching the config file.

**A real bug found and fixed via the CLI smoke test itself (not just
unit tests)**: `ablation.output_dir`/`ablation.split.manifest_path`
were resolved relative to the process's current working directory
instead of `run.output_dir`, unlike every other phase in this project
(Phase 8 writes to `{run.output_dir}/graphs`, etc.) -- invisible in
the pytest suite (which always runs from the repo root, so the
mismatch never showed), but would have silently written results
outside a real run's own output tree the moment `run.output_dir` was
pointed anywhere else, e.g. a full-scale run's configured location.
Fixed with a dedicated path-resolution helper; re-run and confirmed
fixed by inspecting the actual files on disk, not just the log output.

### 9. Two more real, pre-existing bugs found and fixed along the way (unrelated to the ablation work itself, found via its own baseline test run)

1. **`load_phase4_result()`/`load_phase5_result()` resumption bug**:
   hardcoded the canonical (default-encoder) embeddings filename
   regardless of `modalities.timeseries.encoder`/`tabular.encoder`.
   Once those keys were filled in with a real, non-default value
   (2026-09-06), a `--start-phase 7+` run would silently look for the
   wrong file and report "missing Phase 4 embeddings" even though the
   real file existed under its encoder-specific name. Found by running
   the pre-existing test suite as a clean baseline before starting this
   session's own work; fixed to mirror `run_phase4_timeseries_modality()`/
   `run_phase5_tabular_modality()`'s own filename logic.
2. **Duplicate top-level `federated:` key in `configs/base.yaml`**:
   the file declared two separate `federated:` mappings; YAML silently
   keeps only the second, meaning `client_key`/`mode`/`hospitals`/
   `algorithm`/`rounds`/`local_epochs`/`client_fraction` were dropped
   every time `base.yaml` loaded alone. Harmless by accident only
   (every real mode config re-specifies `mode`/`hospitals`, and nothing
   read the other three) -- found while giving `federated.algorithm`
   its first real meaning (see below). Consolidated into one block.

`federated.algorithm` (fedavg | fedproto) now has real meaning for the
first time: `run_phase12_fedavg()` raises `NotImplementedError` with a
clear message if `fedproto` is selected, rather than silently running
FedAvg anyway. **FedProto itself is intentionally not implemented**
(confirmed with the user, decision #5) -- the config key and dispatch
point exist specifically so it can be added later as a new branch, not
a pipeline rewrite.

### Verification status

- **415 tests passing** (up from the pre-session baseline), covering
  every new module: `test_patient_split.py` (9, incl. a hand-verified
  leakage-invariant proof and the unstratified-fallback bug's
  regression test), `test_pca_encoder.py` (8, incl. a direct
  hospital-isolation proof), `test_cross_attention_fusion.py` (6, incl.
  a real-gradient proof), `test_boosting_classifiers.py` (10, incl. the
  NaN-masking proof), `test_flat_multimodal_model.py` (8 tests / 24
  subtests covering every TS x TAB x Fusion branch, incl. a
  gradient-flow proof and a proof that `LocalPatientModel` needed no
  change), `test_config_grid.py` (11), `test_results_writer.py` (8,
  incl. the no-silent-overwrite proof), plus two full, REAL (not
  mocked) end-to-end integration tests exercising the actual training ->
  embedding-extraction -> classifier-fit -> results-writing pipeline
  against generated mock data (`test_embedding_ablation_runner.py`,
  `test_graph_embedding_ablation_runner.py`).
- **CLI smoke-tested for real** (not just pytest): `python
  run_ablation.py --config configs/smoke_test.yaml` run against a
  freshly-generated 200-patient/5-hospital mock dataset end-to-end in
  under 2 seconds, producing real, inspected-on-disk JSON records, a
  master CSV, and a sorted summary table -- this is what caught the
  output-path bug in item 8 above; a pure-pytest run (always invoked
  from the repo root) would never have surfaced it.
- **4 pre-existing test failures, NOT caused by this session's work**:
  `test_modality_architecture_options.py`'s "default still writes
  canonical filename" tests assume `modalities.timeseries.encoder`/
  `tabular.encoder` are at their documented defaults (`lstm`/
  `autoencoder`) unless a test explicitly overrides them -- but
  `configs/base.yaml` currently has these checked in as `transformer`/
  `mlp` (2026-09-06). Confirmed present in a clean baseline run before
  any of this session's changes; not touched, since reverting
  base.yaml's encoder choice isn't this session's call to make. Flagged
  for the user's awareness.
- **NOT YET run against the real 199,108-patient / 208-hospital
  dataset** -- sandbox/mock-data-verified only so far, per this
  project's own standard practice for a newly-delivered phase. See
  "Immediate next step" below.

### What this delivery does NOT yet do

- **Phase 11/12/13 are not yet retrained on the split's train
  partition.** The original request's Section 6 ("retrain the
  trainable stages... using the new training split... do not simply
  evaluate the existing Phase 11-13 models") applies to the FEDERATED
  stage too, not just the two new ablation stages -- but reworking
  Phase 11's `_run_local_training_round()`, Phase 12's FedAvg rounds,
  and Phase 13's centralized-sequential baseline to all be genuinely
  split-aware (mirroring the same train-only-loss-mask design already
  proven in the graph-embedding ablation runner above) is a real,
  separate, similarly-sized piece of work, not a small addition to
  this delivery. Flagged explicitly rather than silently left
  half-done and called complete. See "Immediate next step."
- FedProto (see above -- intentionally deferred, config/dispatch point
  ready).
- Real full-scale execution (see Verification status above).
- Excel-format (`.xlsx`) summary output -- `master_results.csv`/
  `summary_table.csv` open natively in Excel, so this covers the
  "Excel-style tables" requirement without a new heavy dependency; a
  true `.xlsx` writer can be added cheaply later if genuinely needed.

### Immediate next step

1. **Pilot on `configs/subset.yaml` against the real dataset** (not
   mock data) -- see the run command in the chat response. This is the
   generic "measure real timing on a subset before committing to full
   scale" step this project has used before every prior expensive
   phase.
2. Make Phase 11/12/13 split-aware (train-only local training, masked
   loss for the federated/centralized stages, same pattern already
   proven in `graph_embedding_ablation_runner.py`) so the federated
   result itself is evaluable against a genuine held-out set -- the one
   piece of the original request not yet done, per the note above.



New phase. First correction worth recording: the assistant initially
told the user "evaluation" was the natural next step after FedAvg --
said without checking the tracker first. The tracker's own numbering
has Baselines (13) before Evaluation (14), which makes sense in
hindsight (you need something to compare the federated model against
before evaluation metrics mean anything). Corrected before any work
started, not after.

Two real decisions discussed and confirmed with the user before
building:

1. **Centralized baseline: sequential/continuous, NOT true
   cross-hospital pooling.** A genuinely standard "centralized"
   baseline in FL research pools raw data across clients -- which
   would mean building cross-hospital graphs, directly breaking the
   "never pool data across hospitals" rule enforced as absolute since
   Phase 3. Presented as a real fork (pool anyway, as a deliberate
   labeled exception, vs. keep hospital-scoped graphs and just remove
   the federated reset/averaging) rather than assumed either way. User
   chose the latter: `_run_centralized_sequential_training()` runs ONE
   continuously-updated model -- never reset, never split into
   per-hospital copies -- that cycles through every hospital's own
   intra-hospital graph. Isolates specifically the effect of federated
   ROUNDS/RESETS/AVERAGING, holding the privacy-preserving
   architecture itself completely constant.
2. **Local baseline: matched total epoch budget, not Phase 11's
   original 5-epoch checkpoints reused as-is.** Federated hospitals
   see 15 epochs cumulatively (3 rounds x 5 local epochs, with 2
   resets in between). A local baseline capped at 5 would look
   artificially worse than federated purely from less total training,
   not from federation itself helping. `total_epochs =
   training.local_epochs * federated.n_rounds` is DERIVED
   automatically (15 here) rather than a new, independently-settable
   config value that could silently drift out of sync if
   `federated.n_rounds` changes later.

**The "reset frequency" framing that ties all three baselines
together** (documented in `run_phase13_baselines()`'s own docstring):
local = infinite effective resets (never synced at all); federated =
`n_rounds - 1` resets (2, here); centralized = zero resets, fully
continuous. All three otherwise share the same architecture, same
total per-hospital epoch budget, same loss function -- isolating
sync frequency as the one variable under test.

**Engineering: `_load_hospital_batch()` extracted** from what was
previously inline in `_run_local_training_round()` -- now shared by
that function AND the new centralized-sequential loop, which needs
the identical per-hospital batch but in a training loop that never
resets the model between hospitals. Verified behavior-preserving by
re-running Phase 11's and Phase 12's existing 18 integration tests
completely unmodified after the extraction; all 18 passed before any
Phase 13-specific code was written.

`_run_local_training_round()` also gained an optional
`local_epochs_override` parameter (defaults to `None`, preserving
existing behavior for Phase 11/12) so Phase 13's local-only baseline
can request the matched 15-epoch budget without touching config state
or duplicating the function.

**Federated baseline needs no new training**: `run_phase13_baselines()`
reads `checkpoints/phase12_fedavg/fedavg_manifest.json` directly (same
"reuse already-completed real work" precedent as Phase 12's own
round-1 checkpoint reuse) and references Phase 12's final model path.
If that manifest doesn't exist, Phase 13 still completes normally
(local + centralized are independent of it) with the federated
reference simply absent -- never a crash.

**Verified (sandbox, mock data):**
- 9 integration tests (`test_phase13_pipeline.py`) + 2 phase-
  resumption tests (`test_phase_resumption.py`) = 11 new tests, plus
  Phase 11's and Phase 12's existing 18 integration tests re-run
  unmodified to prove the `_load_hospital_batch()` extraction was
  behavior-preserving.
- **Revert → fail → restore → pass proven by hand, twice**: (1)
  injected a per-hospital model reconstruction into the centralized
  loop (defeating this baseline's entire purpose) -- confirmed
  `test_centralized_model_constructed_exactly_once` failed with the
  exact expected construction-count mismatch, restored, reconfirmed
  passing; (2) reverted the fair epoch-budget derivation back to
  plain `training.local_epochs` (dropping `* federated.n_rounds`) --
  confirmed both epoch-budget tests failed, with the log itself
  showing the wrong total (3 instead of 6), restored, reconfirmed
  passing.
- A genuinely useful side-effect caught during testing, not
  hypothetical: at this test suite's learning_rate=0.01, the LOCAL
  baseline naturally produced a real NaN for one mock hospital mid-run
  (confirmed in the actual log output) -- the existing NaN-safety
  mechanism (from Phase 11's own 2026-08-31 fix) handled it correctly
  without any special-casing needed for Phase 13, additional real-world
  confirmation that fix generalizes beyond the exact scenario it was
  built for.
- Full suite: 315 passed, 2 skipped, zero regressions.

**Real full-scale run, 2026-09-05:** local baseline 207/208 trained
(1 empty, matching every prior phase's known pattern), 207/207 with
decreasing loss, weight-delta 1.26x spread (min=20.20, median=21.45,
max=25.45 -- larger than Phase 11's single-round deltas of 8.8-9.6, as
expected given 3x the epochs). Centralized baseline: 15 epochs, 207
hospitals, 0 non-finite, net change -44.7% (0.7338 -> 0.4055),
weight-delta 122.36 (larger still than local's per-hospital deltas,
as expected -- this ONE model absorbs all 207 hospitals x 15 epochs
of signal, not just one hospital's own).

**A correction, recorded rather than glossed over:** based on only 3
log-sampled points (epochs 5/10/15), the assistant told the user
improvement looked like it was ACCELERATING and the model likely
hadn't converged. Once the diagnostic script was extended to show the
full 15-epoch trajectory (see the diagnostic-script entry below) and
the user ran it, the real picture was the opposite: 32.4% of the
ENTIRE run's improvement happened in just epochs 1-5, which the
5/10/15 sampling had entirely missed -- making the later, already-
decelerating portion look like acceleration purely by comparison. The
full data shows textbook decelerating convergence (first-half change
-0.2726, second-half change -0.0558), confirmed correct once actually
checked rather than inferred from a partial log.

**A real, useful diagnostic-script gap closed the same day:** `scripts/
check_local_training_quality.py` had no support for the centralized-
sequential manifest shape at all (flagged as a known limitation in
`docs/baselines.md` at delivery, not discovered later) -- pointed at
`centralized_sequential/`, it would have found no `hospital_*` files
and reported nothing useful. Extended to auto-detect manifest shape
(`"hospitals"` dict vs. `"epoch_history"` list) and report accordingly
-- full epoch-by-epoch loss, net change, an explicit accelerating-vs-
decelerating check, and a single-final-model weight-delta/NaN check
in place of the per-hospital one. Verified against real sandbox data
reproducing the user's exact pattern before delivery; the existing
per-hospital path was regression-tested against prior real Phase
11/12 output to confirm unchanged behavior.

Phase 13 is DONE: implemented, sandbox-verified, and real-full-scale-
verified for all three baselines (local, centralized trained fresh;
federated correctly referencing Phase 12's actual result).

## Phase 12 real run + a docs claim checked and found false, fixed before the user hit it — 2026-09-02

User's real run: Phase 11 (2397.25s, ~40 min) then Phase 12, 3 rounds.
Round 1 reused the existing 207 checkpoints (0.13s, as designed).
Round 2: 1920.57s (~32 min), 207 trained, 1 empty, 0 no-labels, 0
non-finite. Round 3: 1885.22s (~31 min), identical skip counts. Total
Phase 12: 3811.73s (~64 min). Every consistency signal matches
expectations: same single empty hospital across every round (a
structural property of that hospital, not a training artifact), zero
NaN/Inf across 414 real hospital-training-instances (207 x 2 fresh
rounds), round 2/3 runtimes closely matching each other and Phase 11's
own.

**Before calling this done, `docs/fedavg.md`'s own "Immediate next
step" claim was checked rather than trusted** ("scripts/check_local_
training_quality.py can be pointed at checkpoints/phase12_fedavg/
round_2/ and round_3/ the same way it was used for Phase 11's own
checkpoints") -- and found FALSE. `_run_local_training_round()` (the
helper Phase 11 and Phase 12 share) never saved its own `global_
initial_weights.pt` -- only `run_phase11_local_training()` did, at its
own top-level checkpoint dir, before calling the helper. Phase 12's
`round_2/`/`round_3/` subdirectories therefore never got that file,
and the diagnostic script (which looks for exactly that filename)
would have errored out immediately against them. **Fixed:**
`_run_local_training_round()` now saves its own copy of whatever
`initial_state_dict` it was given, inside its own `checkpoint_dir`,
making every checkpoint directory it ever writes to -- Phase 11's own,
or any Phase 12 round -- self-contained and independently inspectable
the same way. The now-redundant explicit save in
`run_phase11_local_training()` was removed (the helper does it now).

**Verified:** 1 new integration test
(`test_each_fresh_round_checkpoint_dir_is_self_contained`), a
hand-performed revert→fail→restore→pass proof (removed the self-save,
confirmed the test failed exactly as it would have for the user,
restored, reconfirmed passing), Phase 11's existing 9 integration
tests re-confirmed passing unmodified (the fix didn't change Phase
11's own observable behavior, just added a file), and the ACTUAL
diagnostic script run against a fresh full sandbox Phase 12 pipeline
(not just unit-level mocks) -- confirmed working against both
`round_2/` and `round_3/`, this time showing real loss-trend and
label-coverage data (unlike Phase 11's original real run, Phase 12's
fresh rounds use the manifest-generating code from the start, so the
user's own `round_2`/`round_3` directories will have this data too,
not just weight-delta/NaN checks). Full suite: 304 passed, 2 skipped,
zero regressions.

**Round 3's 3/207 increasing-loss hospitals, investigated and closed
out (2026-09-02):** identified by name (201, 364, 256) via a further
diagnostic-script extension, and checked against two hypotheses with
the user's actual real data rather than assumed:
- Small raw hospital size -- RULED OUT (19th/31st/64th percentiles;
  one is above the population median).
- Sparse effective labels (many raw patients but few usable labels) --
  RULED OUT (100%/100%/99% coverage, essentially at the 100%/100%
  population median).

What the actual loss curves show instead: all three share an identical
shape (overshoot on epoch 1, an over-correction on epoch 2 larger than
the overshoot, then mild oscillation), net effect 0.83-0.9% relative
increase over 5 epochs -- not divergence, and consistent with ordinary
optimization noise at the tail of 207 independent local rounds (a
fixed learning rate sized for round 2's rougher starting point
occasionally overshooting near round 2's already-refined optimum for a
specific hospital's local data), not a property tied to any hospital
characteristic tested. Zero NaN/Inf, 204/207 clean, doesn't affect
FedAvg's aggregation validity. Closed -- no further action.

## Phase 12 (FedAvg) delivered — 2026-09-01

New phase, immediate next step after Phase 11 closed out. Two real
decisions discussed and confirmed with the user before building, both
with a genuine time cost (~37 min/round on the user's real hardware,
per Phase 11's own confirmed runtime) rather than being silently
defaulted:

1. **3 total rounds.** Multi-round, not a one-shot average -- this
   matches what Phase 11's own config comment already committed to
   ("this phase only runs one round; Phase 12/FedAvg will call this
   repeatedly," `configs/base.yaml`'s `training.local_epochs` comment,
   written 2026-08-30 before Phase 12 existed).
2. **Round 1 reuses the user's existing real Phase 11 checkpoints**
   (`checkpoints/phase11_local/`) rather than re-training from
   scratch -- saves ~37 min, and those checkpoints were already
   confirmed healthy (2026-08-31: zero NaN/Inf across all 207 real
   hospitals, 1.05x weight-delta spread).

**Engineering: `_run_local_training_round()` extracted from Phase 11.**
`run_phase11_local_training()`'s per-hospital loop (loss masking, NaN/
Inf safety, checkpointing, `training_manifest.json`) was pulled into a
shared helper, parameterized by `model_config`/`initial_state_dict`/
`checkpoint_dir`/`round_tag`, so Phase 12's rounds 2+ reuse the EXACT
SAME, already real-full-scale-verified mechanics rather than
reimplementing them. `run_phase11_local_training()` itself becomes a
thin wrapper calling this helper once -- BEHAVIOR UNCHANGED, proven by
re-running Phase 11's existing 9 integration tests completely
unmodified after the extraction; all 9 passed, confirming the refactor
didn't alter anything observable.

**`src/training/fedavg.py`'s `fedavg_aggregate()`:** weighted average
by n_patients (standard FedAvg, not an unweighted mean -- a
7,058-patient hospital, Phase 8's real max, should influence the
global model more than a 20-patient one). Raises clearly on empty
input, missing weights, non-positive total weight, or any
key/shape mismatch across hospitals' state_dicts (would indicate a
genuine architecture mismatch, never expected in normal operation
since every hospital shares one `LocalModelConfig`).

**Scope, stated rather than silently assumed:** plain FedAvg
(weighted parameter averaging) only. No differential privacy, no
secure aggregation -- several of this project's own reference papers
explore those, but they're a separate research direction, not folded
into this phase.

**Round 1 reuse mechanics:** `_try_reuse_existing_local_checkpoints()`
loads whatever's already on disk under `checkpoints/phase11_local/`,
re-validates EACH checkpoint for NaN/Inf independently (a prior run
predating the 2026-08-31 NaN-safety fix, like the user's own real
Phase 11 run, could in principle have saved a corrupted one -- excluded
and logged, never silently aggregated in), and derives each hospital's
n_patients weight directly from Phase 8's own graph files rather than
from Phase 11's `training_manifest.json` (which may not exist -- the
user's real run predates that fix too). This also means Phase 12 does
NOT need `phase11_result` as an in-memory value at all: round 1 reuse
works via `--start-phase 12` on its own, with Phase 11 having run in a
completely separate, prior invocation -- exactly the user's real
situation. If reuse is requested but nothing usable is found, round 1
falls back to a fresh round automatically, with a clear warning, never
a crash.

**Verified (sandbox, mock data):**
- 10 unit tests (`test_fedavg.py`) + 8 integration tests
  (`test_phase12_pipeline.py`) + 2 phase-resumption tests
  (`test_phase_resumption.py`) = 20 new tests, plus Phase 11's
  existing 9 integration tests re-run unmodified to prove the
  extraction was behavior-preserving.
- **Revert → fail → restore → pass proven by hand, three separate
  times**: (1) switched `fedavg_aggregate()` to an unweighted mean --
  confirmed the hand-calculated weighted-average test failed with
  exactly the expected wrong values, restored, reconfirmed passing;
  (2) made every round independently reseed from scratch instead of
  carrying the previous round's aggregate forward -- confirmed
  `test_round2_starts_from_round1_aggregate_not_a_fresh_random_init`
  failed, restored, reconfirmed passing; (3) removed the NaN/Inf check
  from the checkpoint-reuse path -- confirmed a deliberately-corrupted
  reused checkpoint was silently aggregated in (count stayed at 4
  instead of dropping to 3), restored, reconfirmed passing.
- A direct end-to-end correctness proof, not just "a file exists":
  `test_final_aggregate_matches_direct_fedavg_computation`
  independently recomputes the weighted average from a round's own
  saved per-hospital checkpoints and confirms it matches the
  pipeline's own aggregated result exactly.
- Full suite: 303 passed, 2 skipped, zero regressions.
- **Not yet run against the real 199,108-patient / 208-hospital
  dataset.**

## Phase 11 real-run quality check exposed two gaps, both fixed — 2026-08-31

User's real run: 208 hospitals, 207 trained, 1 skipped (empty --
matches Phase 8's already-confirmed real min=0 finding), 0 skipped for
missing labels (expected -- `training.reconstruction_weight=0.5` by
default means the reconstruction term alone keeps a hospital's step
alive even with zero classification labels; only a hospital that is
ALSO fully empty would ever hit zero here). Runtime 2229.23s (~37 min)
for 207 hospitals x 5 local epochs each through four chained neural
network components -- the first phase in this project with a runtime
of this order, expected given it's the first with a real backward
pass, not a red flag by itself.

Asked directly whether this "completed successfully" also meant
"quality-verified," and checking that turned up two real, concrete
gaps -- not hypothetical, found specifically while trying to inspect
THIS completed real run and discovering it couldn't be done:

1. **Per-hospital training metrics were never persisted anywhere.**
   `loss_history` / `n_mortality_known` / `n_sepsis_known` only ever
   existed in `run_phase11_local_training()`'s in-memory return dict.
   `main()` doesn't capture it (nothing downstream needs it yet, same
   gap class as Phase 9's before Phase 10 needed it), and even if it
   did, that dict is gone the instant the process exits -- so a
   completed real run's per-hospital loss curves and label coverage
   were UNRECOVERABLE after the fact, leaving only the phase-level
   summary log line (trained/skipped counts), which cannot answer "did
   loss actually decrease on real hospitals" or "how many hospitals
   trained with near-zero real labels."
   **Fixed:** every run now writes `checkpoints/phase11_local/
   training_manifest.json` with the full per-hospital breakdown plus a
   top-level summary, mirroring the JSON-manifest pattern already
   established for Phase 8/9 (`graph_manifest.json`, `gnn_manifest.json`).
2. **No NaN/Inf detection during training.** `.backward()`/`.step()`
   run cleanly on a NaN/Inf loss with no exception -- so without a
   check, a hospital whose loss went non-finite mid-training (extreme
   values, a degenerate small-hospital graph) would have its corrupted
   checkpoint saved silently, while the run still reports success.
   **Fixed:** `total_loss` is checked with `torch.isfinite()` every
   epoch; a non-finite loss stops that hospital's round immediately,
   its checkpoint is NOT saved, and it's counted under a NEW, distinct
   `n_hospitals_skipped_non_finite` (never folded into
   `n_hospitals_skipped_no_labels` -- a different failure mode
   deserves a different, clearly-labeled count).

**A diagnostic script delivered specifically so the user's ALREADY-
COMPLETED real run doesn't need to be re-run just to get a first
quality read:** `scripts/check_local_training_quality.py` inspects
only what's already on disk (checkpoints + manifest, if present) --
reports loss-trend direction and label-coverage distribution from the
manifest when available, and ALWAYS checks every sampled checkpoint
for NaN/Inf weights and zero-delta-from-initial (a checkpoint
identical to the shared starting point would mean that hospital's
training loop silently didn't run), regardless of whether the run
predates the manifest fix. Confirmed working against both a
manifest-present and a manifest-absent (simulating the user's actual
already-completed run, which predates this fix) scenario.

**Verified (sandbox, mock data):**
- 2 new integration tests: `test_manifest_json_persisted_with_expected_
  fields`, `test_non_finite_loss_skips_that_hospital_without_saving_a_
  corrupted_checkpoint` (the latter poisons one hospital's loss via a
  monkeypatched `compute_multitask_loss` returning `total * nan`, then
  confirms: not in `hospital_results`, no checkpoint file written for
  it, counted specifically under `n_hospitals_skipped_non_finite`
  (not `n_hospitals_skipped_no_labels`), and that OTHER hospitals
  still trained normally).
- **Revert → fail → restore → pass proven by hand:** removed the
  `torch.isfinite()` check entirely -- confirmed the poisoned
  hospital's NaN loss silently trained through with
  `n_hospitals_skipped_non_finite=0` (`AssertionError: 0 != 1`,
  exactly the vulnerability this fix closes), restored, reconfirmed
  passing.
- Full suite: 283 passed, 2 skipped, zero regressions.
- The diagnostic script itself run against a fresh real (if small)
  sandbox checkpoint set, confirmed to correctly report healthy
  weight deltas and loss-decreased-for-all-hospitals, AND confirmed to
  degrade gracefully (falls back to weight-delta/NaN checks only,
  clearly labeled) when `training_manifest.json` is absent -- the
  exact situation the user's own already-completed real run is in.

**What the user's already-completed real run can still verify right
now, at zero additional cost (no re-run needed):** run
`python scripts/check_local_training_quality.py --checkpoint-dir
checkpoints/phase11_local` against the existing output -- this gives a
genuine weight-delta/NaN health check for all 207 real hospital
checkpoints (loss-curve and label-coverage numbers specifically are
not recoverable for this particular run, since it predates the
manifest fix). A full re-run to also get FUTURE runs' loss curves and
label-coverage distribution is worthwhile but not urgent -- left to
the user's judgment given the ~37 minute cost.

**Update, same day: full (not sampled) weight-delta check across ALL
207 real trained hospitals**, using the improved script (default
sample size raised so a plain run now covers everything rather than
an arbitrary alphabetically-first slice -- real eICU `hospitalid`
values aren't sequential, so "first 20 by filename" was never a
representative sample):
```
delta distribution (excluding zero/NaN, n=207): min=9.2443 median=9.3892 max=9.7027
max/min ratio: 1.05x  (tight, consistent across hospitals)
```
Zero NaN/Inf, zero zero-delta, across literally every real trained
hospital, not a sample of them. The tightness of this spread (1.05x
across ALL 207) is itself informative, not just "no red flags": Phase
8's real-data verification already confirmed hospital sizes range from
0 to 7,058 patients, yet total weight movement barely varies across
that range. This is explained by `compute_multitask_loss()`'s
component losses using PyTorch's default MEAN reduction (never sum) --
gradient magnitude per epoch is normalized by that hospital's own
patient count, not raw totals, so a 7,058-patient hospital and a
20-patient hospital produce comparably-SCALED gradients despite the
huge size difference. A genuinely unstable or diverging local round
would be expected to show much MORE erratic, hospital-dependent
movement than this, not less -- so this tight, consistent spread is
real (if indirect) evidence against instability, on top of already
being healthy on every individual metric checked.

**Verdict:** real-data verification for Phase 11 is now considered
adequately complete for practical purposes. The one remaining gap --
exact loss-curve VALUES for the real 208-hospital run specifically
(vs. the strong structural/behavioral evidence above) -- is a minor,
optional loose end, not a blocker; a future re-run will have it
automatically via `training_manifest.json` regardless of whether one
is forced now.


## Phase 11 (Local Training Loop) delivered — 2026-08-30

New phase, immediate next step after Phase 10 closed out. Three real
design decisions discussed and confirmed with the user BEFORE building
(per this project's standing rule):

1. **Training scope: full end-to-end, not heads-only.** LSTM +
   autoencoder + GNN + heads are all trained JOINTLY per hospital, not
   just the heads. Grounded in two things found in the existing code,
   not assumed: Phase 9's own docstring already describes "ONE model
   (shared weights, what a genuine federated deployment would
   eventually FedAvg)"; `TabularAutoencoder`'s decoder already existed
   with an inline comment flagging it as built for "an eventual
   reconstruction-loss training objective (not yet implemented)."
   `src/training/local_model.py`'s `LocalPatientModel` chains the
   EXACT SAME nn.Module classes every prior phase already
   real-full-scale-verified (`LSTMTimeSeriesEncoder`,
   `TabularAutoencoder`, `PatientGNN`, `MultiTaskHeads`) into one
   differentiable forward pass -- a NEW, ADDITIVE module.
   `run_phase4/5/7/8/9/10`'s existing runner functions are completely
   untouched, still valid for their own forward-pass-verification and
   `--start-phase` purposes; they intentionally communicate via saved
   `.npz` files (detached numpy, no autograd graph), which is fine for
   standalone per-phase verification but would silently break
   end-to-end backprop if reused directly for training.
2. **Autoencoder reconstruction loss: included.** `src/training/
   losses.py`'s `compute_multitask_loss()` combines mortality + sepsis
   (masked per-task, reusing Phase 10's `align_labels_to_patient_ids()`
   as-is) with a `training.reconstruction_weight`-scaled MSE
   reconstruction term. The unsupervised reconstruction term never
   depends on label availability -- if BOTH classification tasks have
   zero usable labels for a hospital, reconstruction alone still keeps
   that hospital's training step alive; only if reconstruction is ALSO
   disabled/zero-weighted does the step get skipped and logged
   (returns `None`, never a fabricated zero-valued loss that would
   crash `.backward()` or silently do nothing while looking
   successful).
3. **Weight persistence: yes, lightweight, now.** Each hospital's
   post-training `state_dict()` is saved to
   `checkpoints/phase11_local/hospital_{hid}_local_weights.pt` --
   NOT the full Phase 18 caching/checkpointing system, just enough for
   Phase 12 (FedAvg) to load. `run.checkpoint_dir` was already stubbed
   in `configs/base.yaml` (`checkpoints`), unused until now.

**A real correctness requirement identified while building, not
hypothetical:** FedAvg only makes sense as an operation on weight
tensors that started from the SAME point in weight-space (neural
network loss landscapes are not permutation-invariant -- averaging two
independently-initialized models' weights produces something close to
random, not a meaningful "average model"). Every hospital's local
round therefore starts from IDENTICAL initial weights -- one
`LocalPatientModel` is constructed once (seeded), its `state_dict()`
captured and saved to `checkpoints/phase11_local/
global_initial_weights.pt`, and every hospital's fresh model instance
calls `.load_state_dict()` with that exact same dict before training.
Proven, not just implemented: a hand-performed revert→fail→restore→pass
against BOTH failure variants (skipping the shared init entirely, and
the subtler case of still calling `load_state_dict()` but with an
independently-random per-hospital state) -- see Verification below.

**A second real gap found and fixed while wiring this in:**
`load_phase4_result()`/`load_phase5_result()` previously only ever
returned `embeddings_path` (Phase 7's only consumer), a documented,
deliberate partial round-trip. Phase 11 trains the LSTM/autoencoder
jointly with the rest of the model, so it needs the raw NORMALIZED
tensors themselves, not Phase 4/5's already-embedded, frozen output.
Both loaders now read their full `.npz` files back
(`timeseries_sequences.npz`, `tabular_features.npz`) and expose
`patient_ids`/`normalized_array`/`sequence_length`/`feature_names` at
the top level, matching what a freshly-run
`run_phase4_timeseries_modality()`/`run_phase5_tabular_modality()`
call already returns -- so Phase 11 reads the same shape regardless of
whether Phase 4/5 ran fresh or were reloaded via `--start-phase`.

**A third, smaller finding: a stale `training:` config stanza.**
`configs/base.yaml` already had a `training:` section (`batch_size`,
`learning_rate`, `weight_decay`, `max_epochs`,
`early_stopping_patience`) stubbed before any training loop existed,
for a CENTRALIZED training concept (one model, N global epochs,
early-stopped against a held-out validation set) -- a materially
different shape than Phase 11's per-hospital LOCAL round. `learning_
rate`/`weight_decay` were directly reusable as-is; `batch_size` (no
mini-batching within a hospital -- the GNN needs the whole hospital
graph at once, same "no batching needed at this scale" reasoning as
Phase 8/9/10), `max_epochs`, and `early_stopping_patience` (no train/
val split exists anywhere in this project yet) are annotated in place
as not-yet-applicable rather than deleted or silently repurposed --
same "found stale config, fixed and explained why" discipline as the
`fusion.output_dim=256` and `graph.max_nodes_full_similarity` fixes
before it. `gnn.learning_rate`/`gnn.weight_decay` (pre-existing stubs
under `gnn:`) are similarly annotated as superseded by `training.
learning_rate`/`training.weight_decay`, which now optimize all four
jointly-trained components with one composite optimizer, not a
GNN-only one. New keys added: `training.optimizer` (adam | sgd),
`training.local_epochs`, `training.reconstruction_weight`, `training.
max_hospitals`.

**Verified (sandbox, mock data):**
- 15 unit tests (`test_losses.py`) + 6 unit tests (`test_local_model.py`)
  + 7 integration tests (`test_phase11_pipeline.py`) + 2 phase-
  resumption tests (`test_phase_resumption.py`) = 30 new tests.
- **Real trainability, confirmed by hand, not assumed:** on the exact
  mock dataset used by the integration tests, all 4 hospitals' total
  loss strictly decreased over 5 local epochs (e.g. 1.4878 -> 1.4009,
  1.4753 -> 1.3488) -- `test_loss_decreases_over_local_epochs_for_
  every_trained_hospital` encodes these exact confirmed numbers as its
  regression target.
- **All four components proven to receive real gradients, confirmed
  by hand:** per-component weight-delta norms after training
  (lstm_encoder=4.16, autoencoder=1.66, gnn=2.28, heads=0.90 for one
  hospital vs. the shared initial checkpoint) -- not just "loss went
  down" (which could be true from the heads alone).
- **Genuine per-hospital divergence, confirmed by hand:** two
  different hospitals' post-training checkpoints differ from each
  other (total delta 5.26 in the same manual run) -- hospitals did not
  all converge to the same point.
- **Revert → fail → restore → pass proven by hand, three separate
  times**: (1) severed the autograd graph between the encoders and the
  GNN/heads with a `.detach()` (simulating a stray call copied from
  the forward-pass-only phase code) -- confirmed
  `test_gradients_flow_to_every_component` failed specifically
  pinpointing `lstm_encoder` (the one component with no alternate
  gradient path), restored, reconfirmed passing; (2) reverted
  `compute_multitask_loss()`'s all-unknown-labels case to return a
  fabricated `torch.tensor(0.0)` instead of `None` -- confirmed the
  dedicated test failed, restored, reconfirmed passing; (3) reverted
  the shared-initial-weights line in `run_phase11_local_training()`,
  in BOTH the "skipped entirely" and the subtler "called but with an
  independently-random state" variants -- confirmed
  `test_every_hospital_starts_from_identical_initial_weights` caught
  both (the second via an exact per-tensor mismatch message naming
  `lstm_encoder.lstm.weight_ih_l0`), restored, reconfirmed passing.
- Full suite: 281 passed, 2 skipped, zero regressions.
- **Not yet run against the real 199,108-patient / 208-hospital
  dataset.**


## Phase 10 (Multi-Task Prediction Heads) delivered — 2026-08-30

New phase, immediate next step after Phase 9 closed out. `src/heads/
multitask_heads.py` implements `configs/base.yaml`'s `multitask:`
stanza. Architecture + forward-pass verification only, no training
loop yet, same pattern as every prior modeling phase. Runs ONE global
forward pass over Phase 9's full per-patient GNN embeddings -- no
per-hospital looping, unlike Phase 9 -- because a prediction head has
no cross-patient interaction at all (see the module docstring for the
full reasoning; same category as Phase 4/5, which also don't loop by
hospital).

**Design decision, discussed and confirmed with the user before
building:** independent small MLP heads per task (Linear → ReLU →
Dropout → Linear for EACH of mortality and sepsis), no shared
bottleneck beyond the GNN trunk itself. Alternatives considered: plain
linear probes (lower capacity than wanted) and a shared bottleneck
layer before the two heads split off (rejected in favor of full
independence). `multitask.head_hidden_dim` (default 64) and
`multitask.head_dropout` (default 0.1) added to `configs/base.yaml`,
config-driven per this project's standing rule.

**Prerequisite infrastructure fix, found while wiring Phase 10 in, not
hypothetical:** Phase 9 had no `load_phase9_result()` and its return
dict was silently discarded in `main()` -- nothing needed to reload
Phase 9's output until Phase 10 existed to consume it. Worse, Phase
9's `gnn_type` was never persisted to disk anywhere, only ever
available in-memory. Fixed by adding a `gnn_manifest.json` alongside
`gnn_embeddings.npz` (mirroring Phase 8's `graph_manifest.json`
exactly) and a `load_phase9_result()` loader. `need_phase9_result`
(`will_run(9) or will_run(10)`) and `need_phase3_result` (extended to
include `will_run(10)`) added to `main()`'s dependency-flag block,
following the same direct-consumer-mapping discipline the Phase-9
over-fetching bug (see the `--start-phase` section below) already
established -- Phase 10 needs phase9_result and phase3_result
directly, NOT phase7_result/phase8_result (those are Phase 9's own
inputs).

**Real (if narrow) gap found via this phase's own integration test
against mock data, root-caused before deciding how to handle it, not
assumed:** with the test's seed, one patient has zero rows across
`apachePredVar`, `medication`, AND `microLab` -- entirely absent from
`sofa_df` and `infection_df`, and therefore from `sepsis_df`, despite
being a fully valid `cohort_patients` member with a real GNN embedding.
Root cause: `compute_infection_proxy()`'s output only covers patients
with ≥1 medication/microLab row, and `compute_sofa_like_scores()`'s
only covers patients with an `apachePredVar` row -- `compute_sepsis_
labels()`'s own inline comment already documents the *one-sided*
version of this ("a real, common situation in eICU, ~468/4064 patients
in one real run"); a patient missing from *both* sides is the same
phenomenon, not a different problem. Treated exactly like `mortality_
label()`'s own precedent for an unrecognized `hospitaldischargestatus`:
sepsis status is `NaN` (unknown) for that patient, not an error and
not silently coded as sepsis-negative. `align_labels_to_patient_ids()`
still hard-raises if a patient is missing from `cohort_patients`
itself, which stays the authoritative, expected-100%-coverage source.
**Real-data prevalence of the both-sides-missing case is NOT YET
CONFIRMED** -- only seen in mock data so far; flagged here rather than
assumed rare, pending the user's real run.

**Verified:**
- 19 new unit tests (`test_multitask_heads.py`) + 6 new integration
  tests (`test_phase10_pipeline.py`) + 2 new phase-resumption tests
  (`test_phase_resumption.py`, covering Phase 10's dependency flags
  specifically).
- **Revert → fail → restore → pass proven by hand, four separate
  times**: (1) removed the `squeeze(-1)` call, confirmed exactly the
  three shape-related tests failed with `(N, 1)` instead of `(N,)`
  shapes; (2) made both heads share one module instance, confirmed
  exactly the two independence tests failed; (3) reverted `align_
  labels_to_patient_ids()` to hard-raise on sepsis_df-absence, confirmed
  exactly the unit + integration tests for that behavior failed; (4)
  reverted `need_phase9_result` to `will_run(9)` only (omitting `will_
  run(10)`), confirmed the new phase-resumption test failed with the
  exact `AttributeError: 'NoneType' object has no attribute 'get'`
  this dependency-flag bug class has produced before. All four
  restored and the full suite reconfirmed passing after each.
- Full suite: 251 passed, 2 skipped, zero regressions.
- **Not yet run against the real 199,108-patient / 208-hospital
  dataset** -- sandbox-verified only so far, including the sepsis_df-
  coverage finding above, which specifically needs real-data
  confirmation of actual prevalence, not just the mock-data existence
  proof.

**CONFIRMED on the real 199,108-patient / 208-hospital dataset
(2026-08-30, `sepsis_cohort=broad`):**
```
[LABELS] mortality: known=199108 positive=18004 negative=181104 unknown=0 |
  sepsis_broad: known=194331 positive=44412 negative=149919 unknown=4777
[PHASE10] heads forward pass over 199108 patients --
  mortality_logits (199108,), sepsis_logits (199108,)
[PHASE10_MULTITASK_HEADS] COMPLETE (2.48s)
```
Exact expected shapes, full cohort covered, 2.48s. Two things this run
specifically confirms:

1. **The sepsis_df-absence question is now answered, not just flagged.**
   4,777/199,108 (2.40%) of the real cohort has zero rows across
   `apachePredVar`, `medication`, AND `microLab` -- lower than the
   ~11.5% one-sided rate `compute_sepsis_labels()`'s own comment
   already documented (468/4,064 in an earlier real run), consistent
   with "both sides missing" being rarer than "one side missing."
   These 4,777 patients get `sepsis=NaN` (unknown), not silently
   coded as sepsis-negative -- exactly the intended behavior.
2. **`sepsis_broad_positive=44412` exactly matches** the count
   originally recorded from Phase 3's own real full-scale run
   (2026-08-10, "Sepsis counts... `sepsis_broad_positive=44412`" --
   see that section below). This is a genuine, independent, real-scale
   confirmation that `align_labels_to_patient_ids()` preserves every
   sepsis-positive label with zero loss or duplication when re-joining
   Phase 3's labels onto Phase 9's differently-ordered (hospital-
   grouped, not cohort-order) GNN embedding patient list -- not just
   proven in a mock-data unit test, proven at the exact real number.

Mortality shows `unknown=0` at full real scale (every patient's
`hospitaldischargestatus` parsed cleanly) -- worth noting since this
means the mortality-unknown code path, while still correct and tested,
hasn't actually been exercised by real data the way the sepsis-unknown
path now has.

Phase 10 is DONE: implemented, sandbox-verified (including four
separate revert→fail→restore→pass proofs), and real-full-scale-
verified for the default `sepsis_cohort=broad`. `sepsis_cohort=strict`
remains sandbox-verified only -- `full.yaml` doesn't override the
`broad` default, mirroring exactly how Phase 9's `gat`/`gcn` stayed
sandbox-only when `full.yaml` didn't override `gnn.type` either.



Found while reviewing the phase-resumption feature for drawbacks, not
during a real run: `debug.yaml`, `subset.yaml`, and `full.yaml` all
share the same default `run.output_dir` -- none override it. Without a
guard, running Phase 1-7 with `debug.yaml`'s tiny
`cohort.patient_subset: 20` and later `--start-phase 8 --phase 9` with
`full.yaml` would have silently resumed Phase 8/9 against the debug
run's tiny-cohort graphs, believing it was the real full cohort -- same
shapes, no crash, no warning, quietly wrong.

`validate_run_identity()` closes this: whenever `--start-phase > 2`, it
compares `data.root`/`cohort.patient_subset`/`federated.mode`/
`federated.hospitals`/`run.seed` (the fields that determine WHICH
patients/hospitals are in the cohort) against `run_identity.json`
(written by Phase 2 whenever it actually runs) and raises a clear
error naming exactly what differs on any mismatch. Deliberately does
NOT check `graph.k`/`gnn.type`/etc -- changing hyperparameters between
a full run and a `--start-phase 8+` run is the whole point of the
feature, not something to guard against.

**Verified:**
- 2 new tests in `test_phase_resumption.py`: a direct regression test
  (changes `cohort.patient_subset` between two runs sharing the same
  `run.output_dir`, confirms it's blocked with a clear error naming
  the field) and a sanity check that an unchanged config still resumes
  normally.
- **Revert → fail → restore → pass proven by hand**: disabled the
  identity check, reran the resumption suite, confirmed exactly the
  mismatch-detection test failed (silently succeeded instead of
  blocking) and nothing else, restored the check (byte-diffed to
  confirm), confirmed all 9 resumption tests passed again.
- Full suite: 224 passed, 2 skipped, zero regressions.
- **Not yet run against the real dataset** -- sandbox-verified only,
  same as the resumption feature itself.

## Phase resumption (`--start-phase`) delivered — 2026-08-27

Not a pipeline phase -- cross-cutting infrastructure, requested by the
user after re-running the full Phase 1-9 sequence repeatedly became
real overhead now that later phases (8: 15s, 9: 22s at real full
scale) are fast while earlier ones (Phase 1's chunked data report,
Phase 3's SOFA/sepsis labeling, Phase 4/5's forward passes) remain the
expensive part. New `--start-phase N` CLI argument skips already-
completed phases, reloading their outputs from `run.output_dir`
instead of recomputing. See `docs/phase_resumption.md` for the full
writeup.

**Persistence added:** Phase 2 now also writes
`partition_result_full.json` (the existing `hospital_partition.json`
summary didn't include `patient_ids`, the one thing actually needed
downstream -- added as a separate file rather than changing that
summary's schema). Phase 3 now also writes `cohort_patients.csv`,
`mortality_labels.csv`, `sepsis_labels.csv`. Phases 4/5/7/8 needed no
new persistence -- their existing saved artifacts (`embeddings_path`,
`fused_embeddings_path`, `manifest_path`) were already everything a
later phase needs.

**Dtype safety checked, not assumed**, given this project's own
documented history with exactly this bug class (bool columns silently
becoming `object` dtype after a merge): `sepsis_df`'s bool columns are
explicitly re-cast after the CSV round-trip; `mortality` (a Series
originally sharing `cohort_patients`' DataFrame index, not
`patientunitstayid`) is saved as an explicit ID-keyed two-column CSV
and reconstructed via `.map()`, never a positional-index round-trip.
Verified directly with a round-trip script (run Phase 1-3, reload via
each loader, diff against the originals) before any of this was wired
into `main()`.

**Real bug found and fixed while building this, not hypothetical:** an
early version of `--start-phase 8` still transitively tried to
reconstruct `phase3_result` (and therefore `phase1_summary`), because
every phase's load-or-run block executed unconditionally whenever
`args.phase >= N`, regardless of whether anything downstream actually
needed that phase's result. Fixed by computing explicit `need_*`
boolean flags in `main()` from each phase's real parameter list (not
phase-number proximity) -- a result is only loaded if its own phase
will run, or a downstream phase that will actually run directly
depends on it. A second, related issue: `shard_manifests` (part of
`phase3_result`) is read by Phase 4 alone, never Phase 5 or 7 -- but
reconstructing it requires the *original raw table* to still exist on
disk (the sharding cache's freshness check needs to `stat()` it, even
on a guaranteed cache hit), so a naive "always rebuild all of
phase3_result" would have silently required raw data access even for
`--start-phase 7`. Fixed with a separate, finer-grained
`need_shard_manifests = will_run(4)` flag.

**Verified:**
- 7 new end-to-end tests (`test_phase_resumption.py`), invoking the
  real CLI via `subprocess` rather than testing loader functions in
  isolation, because the actual claim ("the command line tool can skip
  phases") can only be faithfully proven that way.
- **The core proof**: full pipeline run once, mock data root deleted
  *entirely*, then `--start-phase 7 --phase 9` run again and confirmed
  to still succeed with matching output. Succeeding with the data gone
  is direct evidence of genuine resumption, not just a fast skip.
- **Direct regression test for the over-fetching bug above**: deletes
  Phase 1 and Phase 3's artifacts specifically (proving they aren't
  needed for `--start-phase 8`) while deliberately keeping Phase 2's
  `partition_result_full.json` (Phase 8 genuinely, directly depends on
  it) -- confirms the fix is precisely scoped, not overcorrected into
  "skip everything."
- A second resumption point (`--start-phase 4`) tested too, proving
  this isn't special-cased to Phase 7+.
- Four tests covering CLI validation and error clarity (rejecting
  `--start-phase > --phase`, rejecting Phase 6 as a resumption point,
  and two tests confirming a specific missing-phase name appears in
  the error rather than a generic failure).
- Full suite: 222 passed, 2 skipped, zero regressions.
- **Not yet run against the real 199,108-patient / 208-hospital
  dataset** -- sandbox-verified only so far.

## Phase 9 (Graph Neural Network) delivered — 2026-08-26

New phase, immediate next step after Phase 8 closed out (confirmed on
real full-scale data the same day, with a direct edge-level
verification, not just aggregate counts). `src/gnn/patient_gnn.py`
implements `configs/base.yaml`'s `gnn:` stub -- previously a
placeholder, now real. Architecture + forward-pass verification only,
no training loop yet, same pattern as every prior modeling phase.

**Design decisions, both discussed and confirmed with the user before
building (not picked silently):**

1. **All three architectures implemented** (GraphSAGE, GAT, GCN), not
   just the configured default -- matches the ablation-comparison
   capability a research write-up benefits from, and consistent with
   how `fusion.type`/`graph.method` were also both/all fully
   implemented in prior phases.
2. **Edge weight (Phase 8's cosine similarity scores) used only for
   GCN**, where PyG's `GCNConv` has a native, published mechanism for
   it (Kipf & Welling 2017's weighted-graph extension). NOT used for
   GraphSAGE (`SAGEConv` has no edge-weight parameter at all in its
   standard formulation) or GAT (which computes its own attention from
   node features; re-injecting the same cosine similarity that built
   the graph's edges in the first place would be largely circular
   rather than adding genuine signal). Confirmed directly against the
   installed PyG API (2.8.0.post1) before committing to this design,
   not assumed from memory:
   ```
   SAGEConv.forward(x, edge_index, size=None)
   GATConv.forward(x, edge_index, edge_attr=None, ...)
   GCNConv.forward(x, edge_index, edge_weight=None)
   ```
   This was an explicit response to being asked for "excellent and
   actual, not fake or dummy" use of edge weights -- the honest answer
   differs by architecture, so the implementation does too, rather than
   uniformly wiring in a "weighted" mechanism that would look more
   sophisticated without being genuinely justified everywhere it
   touched.

**Real-data-informed edge case, tested before it was needed:** Phase
8's real run found at least one real hospital with zero patients
(`min=0`). Before writing any surrounding code, all three raw PyG
layer types were directly tested against empty and single-node graphs
to confirm graceful degradation -- confirmed clean for all three,
covered by dedicated tests.

**Verified:**
- 11 new unit tests (`test_patient_gnn.py`) + 4 new integration tests
  (`test_phase9_pipeline.py`, covering all three architectures
  end-to-end).
- **Revert → fail → restore → pass proven by hand**: temporarily made
  the forward pass never pass `edge_weight` through to any layer
  regardless of architecture, reran the suite, confirmed exactly the
  GCN-sensitivity test failed (and nothing else), restored the fix
  (byte-diffed to confirm), confirmed all tests passed again.
- Full suite: 215 passed, 2 skipped, zero regressions.

**CONFIRMED on the real 199,108-patient / 208-hospital dataset
(2026-08-27, `graphsage` default):**
```
[GNN] progress: 208/208 hospitals
[PHASE9] GNN (graphsage) embeddings ((199108, 128)) written to outputs\embeddings\gnn_embeddings.npz
[PHASE9_GNN_FORWARD_PASS] COMPLETE (22.23s)
```
Exact expected shape, all 208 hospitals, zero errors. Phase 9 is DONE:
implemented, sandbox-verified, and real-full-scale-verified for the
default architecture. `gat`/`gcn` remain sandbox-verified only --
`full.yaml` doesn't override `gnn.type`, so only `graphsage` has been
run against real data so far.

## Phase 8 (Intra-Hospital Patient Graph Construction) delivered — 2026-08-26

New phase, immediate next step after Phase 7 closed out (confirmed on
real full-scale data the same day: `(199108, 192)` fused embeddings).
`src/graph/patient_graph.py` builds one similarity graph PER HOSPITAL
from Phase 7's fused embeddings -- `configs/base.yaml`'s pre-existing
`graph:` stub (`method`, `k`, `metric`, `cosine_threshold`,
`max_nodes_full_similarity`) is now actually implemented, not just a
placeholder.

**Design decision (Option B, locked with user confirmation):** exact
similarity computation everywhere, chunked (not approximate) above
`max_nodes_full_similarity` to bound peak memory -- mirrors
`run_batched_with_backoff`'s existing pattern in this project. Rejected
adding an ANN library (e.g. faiss) given realistic hospital sizes (a
few thousand patients at most) make exact computation trivial, and
exact results are more defensible for a research write-up than an
approximation needing its own justification. Corrected the config's
stale comment (previously said "force approximate/ANN construction"
above the threshold -- no longer accurate).

Confirmed `torch-geometric` (already declared in `requirements.txt`
for "Phase 4-10: modeling") installs and works cleanly in this sandbox
before committing to it as Phase 9's intended consumer of this
phase's output format.

**Non-negotiable, inherited from every prior phase's own privacy
boundary:** each hospital's graph is built from ONLY that hospital's
own patients -- no cross-hospital edges, ever, mirroring per-hospital
normalization (Phase 5), per-hospital vocabulary aggregation (Phase
5's 2026-08-24 fix), and hospital-sharded large tables (never pooled
in memory).

**Two independent correctness risks, both directly regression-tested:**

1. **Exact-vs-chunked similarity.** Chunking must never change the
   result, only bound intermediate memory. A subtly-wrong-but-plausible
   chunking bug (comparing each chunk only against itself instead of
   the full set) was injected by hand and confirmed to break exactly
   the two tests targeting this claim, nothing else.
2. **No cross-hospital edges.** Pooling the full cohort into every
   hospital's graph (instead of that hospital's own patients only) was
   injected by hand at the pipeline-wiring level and confirmed to
   break both the direct boundary-crossing check AND the total-node-count
   check (4x too many nodes in the test scenario, since every hospital's
   "graph" absorbed all 4 hospitals' patients) -- proving the test
   suite catches this class of regression, not just a correctly-wired
   pipeline that happens to pass.

Also caught a real gap in my own first-draft test while performing the
node-identity revert→fail→restore→pass check: the original
`test_node_index_correctly_maps_to_patient_id` used a patient-ID list
that happened to already be in sorted order, so injecting a "silently
re-sorts patient_ids" bug had NO effect on that test -- it kept passing
against broken code. Fixed by using a deliberately unsorted patient-ID
list instead, then re-confirmed the injected bug actually gets caught.

**Verified:**
- 20 new unit tests (`test_patient_graph.py`) + 4 new integration
  tests (`test_phase8_pipeline.py`).
- Full suite: 200 passed, 2 skipped, zero regressions.

**CONFIRMED on the real 199,108-patient / 208-hospital dataset
(2026-08-26):**
```
[PHASE8] 208 hospital graphs written to outputs\graphs
(total nodes=199,108, total edges=1,462,202, method=knn)
[PHASE8_PATIENT_GRAPH_CONSTRUCTION] COMPLETE (15.11s)
```
Followed by a dedicated read-only verification script
(`verify_phase8_graphs.py`) confirming, directly against the real
saved output rather than inferred from aggregate counts: zero
cross-hospital edges across all 1,462,202 real directed edge entries;
the chunked-similarity path genuinely exercised (29 real hospitals
exceeded `max_nodes_full_similarity=2000`, largest 7,058 patients,
completing in the same 15.11s total with zero errors); real
per-hospital size distribution min=0 (at least one real hospital has
zero patients in the final cohort -- handled gracefully by the
empty-hospital case) / median=548 / max=7,058 -- genuine, extreme
non-IID heterogeneity across federated clients, relevant to later
FedAvg work. Phase 8 is now more rigorously confirmed on real data
(aggregate log line AND direct file-level edge check) than any prior
phase.

## Phase 7 (Multimodal Fusion) delivered — 2026-08-25

New phase, not a bug fix -- immediate next step after Phase 5's
vocabulary-bias investigation closed out (both fixes confirmed on real
full-scale data the same day). `src/fusion/multimodal_fusion.py`
concatenates Phase 4's LSTM `(N, 128)` + Phase 5's autoencoder
`(N, 64)` embeddings into one unified `(N, 192)` per-patient
representation.

**Design decision (Option A, locked with user confirmation):** pure
concatenation, no learned projection layer -- consistent with Phase
4/5, neither of which has a training loop yet either. Found and
corrected a stale config value along the way: `fusion.output_dim` was
`256` (a leftover placeholder from early planning, before Phase 6/text
was permanently locked out), not matching the real, now-confirmed
`128 + 64 = 192` sum. Corrected to `192`, now descriptive rather than
enforced -- `run_phase7_multimodal_fusion()` warns, doesn't error, if
it ever drifts from the real sum.

**Core correctness risk addressed:** Phase 4 and Phase 5 currently
derive their patient population from the same
`phase3_result['cohort_patients']` list and never drop/filter
patients, so their saved embeddings' `patient_ids` already match by
construction -- but `multimodal_fusion.py` does not rely on that
holding by convention. `align_embeddings_to_cohort()` re-indexes each
modality's embeddings by explicit patient-ID lookup against the
cohort's own order, rather than assuming row-order alignment. A naive
positional concatenation would silently fuse *different patients'*
data together if the two modalities' storage orders ever diverged --
same shape, no crash, corrupted every downstream sample. See
`docs/multimodal_fusion.md`'s "Why ID-based alignment, not row order"
for the full argument.

Also fixed two small pre-existing gaps found while wiring this in:
`main()`'s CLI orchestration wasn't capturing Phase 4/5's return
values at all (`run_phase4_timeseries_modality(...)` called for its
side effects only) -- needed for Phase 7 to receive them, now captured
as `phase4_result`/`phase5_result`. `--phase` validation only allowed
1-5; extended to also accept 7 (6 stays permanently excluded, per the
locked text-modality decision).

**Verified:**
- 11 new unit tests (`test_multimodal_fusion.py`), including
  `test_misaligned_storage_order_still_fuses_correctly` -- the direct
  regression target for the alignment risk above (stores the two
  modalities in deliberately different patient orders, asserts each
  fused row is still that patient's own data).
- **Revert → fail → restore → pass proven by hand**: temporarily
  replaced `align_embeddings_to_cohort()` with a naive positional
  pass-through (`return embeddings`, ignoring patient_ids/target_order
  entirely), reran the suite, confirmed 7 of 11 tests failed
  immediately (including the misalignment test), restored the fix
  (byte-diffed to confirm), confirmed all 11 passed again.
- 4 new integration tests (`test_phase7_pipeline.py`), running the
  actual Phase 1→2→3→4→5→7 pipeline against mock data -- including a
  direct row-level correctness check (a real patient's fused row
  equals their own LSTM row concatenated with their own autoencoder
  row, each looked up independently by ID from its own source file,
  not assumed from position) and a check that a missing embeddings
  path raises clearly rather than silently fusing a partial result.
- Full suite: 176 passed, 2 skipped, zero regressions.
- **Not yet run against the real 199,108-patient / 208-hospital
  dataset** -- sandbox-verified only so far, same as every new phase
  before its first real run. Immediate next step: `python
  run_pipeline.py --config configs/full.yaml --phase 7`, confirm the
  `[PHASE7] Fused embeddings ...` log line reports shape `(199108, 192)`.

**CONFIRMED on the real 199,108-patient / 208-hospital dataset
(2026-08-26):** `(199108, 192)`, clean run (11.25s), no
`fusion.output_dim` mismatch warning. Phase 7 fully closed out.

## Decisions resolved 2026-08-08

1. **Sepsis cohort strictness** — RESOLVED: build both. `strict`
   (antibiotic + microLab culture corroboration within
   `corroboration_window_minutes`) is the primary analysis; `broad`
   (antibiotic alone) is the sensitivity analysis. Both are always
   computed and reported side by side, including their overlap — see
   `docs/sepsis_label_methodology.md`.
2. **Text modality** — RESOLVED: stays disabled by default. Diagnosis-
   string pseudo-text mode was not built.
3. **`treatment.csv`** — RESOLVED: confirmed present by the user, now a
   required table in `schema_registry.py`, included in the mock data
   generator and Phase 1 validation.
4. **`medication_imputed` / `medication_imputed1`** — RESOLVED
   (second confirmation, same day): both confirmed present, listed
   explicitly among the user's 13 tables. Both promoted to required.
   Their exact relationship to `medication` (distinct imputation
   passes vs. independent order sets) is still genuinely unknown, so
   `infection_proxy.combine_medication_sources()` unions all three and
   deduplicates exact repeated rows as a documented heuristic rather
   than guessing which single table is authoritative — see
   `docs/sepsis_label_methodology.md` for the caveat this doesn't fully
   resolve.

## No open data-availability items remain

All 13 tables the user has are now registered as `availability:
"confirmed"` in `schema_registry.py` (verified by
`test_no_unconfirmed_tables_remain_in_registry`).

## Leakage-registry gap found while building Phase 5 (2026-08-11)

Building Phase 5's tabular feature allowlist required reviewing
`apachePredVar`'s full column set closely, which surfaced a real gap:
`apachePredVar.diedinhospital` — the mortality outcome itself — was
never registered in `schema_registry.py` at all, meaning nothing had
been flagging it as a leakage risk (unlike `apachePatientResult`'s
outcome columns, which were flagged from Phase 1 onward). It existed in
the real table the whole time; Phase 5 is simply the first phase that
looked closely enough at every one of that table's columns to notice.
Fixed: added to `apachePredVar`'s `optional_columns` and
`leakage_risk_columns`. Phase 5's own feature allowlist never included
it regardless (allowlist design, not blocklist — see
`docs/tabular_modality.md`), so this was never actually consumed as a
feature; the fix closes the gap in the automated flagging, not a
correction to already-shipped feature data.

## Windows-only test-teardown bug: open log file blocks tempdir cleanup — found + RESOLVED 2026-08-24

Found while the user verified the vocabulary-sampling-bias fix
above on their real (Windows) machine — every integration test failed
in `tearDown()` with:
```
PermissionError: [WinError 32] The process cannot access the file
because it is being used by another process: '...\outputs\logs\pipeline.log'
```
12 of the ~14 integration tests failed this way; the vocabulary-bias
fix itself was confirmed working correctly in the same run's logs
(`teachtype=2, region=4` — no longer collapsed) before this unrelated
issue was found.

**Root cause:** `logging_setup.setup_logging()` attaches a
`logging.FileHandler` to the shared `"eicu_fl"` logger and correctly
closes the *previous* handler -- but only when called again at the
START of the NEXT phase. Nothing ever closed the LAST phase's handler
at the end of a run. Each integration test's `tearDown()` calls
`tempfile.TemporaryDirectory.cleanup()`, which tries to delete the
whole temp tree including that still-open `pipeline.log`. Windows
refuses to delete a file an open handle still points at; Linux (this
sandbox) permits it, which is exactly why this was invisible here and
only surfaced on the user's real machine -- a genuine, pre-existing
resource leak that happened to be harmless on one platform and fatal on
another.

**Fix (`src/utils/logging_setup.py`):** added `close_logging()`, which
closes and detaches every handler from the `"eicu_fl"` logger. Called
in `tearDown()`, before `self.tmpdir.cleanup()`, in all 6 integration
test files (`test_full_mode_scale.py`, `test_phase1_pipeline.py`,
`test_phase2_3_pipeline.py`, `test_phase4_pipeline.py`,
`test_phase5_pipeline.py`, `test_phase5_vocabulary_bias_regression.py`).

**Verified:**
- New `tests/unit/test_logging_setup.py` (8 tests) -- including a
  direct check that `FileHandler.stream` is actually `None` after
  `close_logging()` (the precise state that prevents the Windows
  `PermissionError`), and an end-to-end check that a tempdir containing
  an active log file can be deleted immediately after `close_logging()`.
- **Revert → fail → restore → pass proven by hand**: temporarily made
  `close_logging()` a no-op (exactly the pre-fix state), reran the new
  tests, confirmed 3 failed with `FileHandler.stream still set...`
  (the precise pre-fix symptom), restored the fix (byte-diffed to
  confirm), confirmed all 8 passed again.
- Full suite: 155 passed, 2 skipped (up from 147+2 -- 8 new tests),
  zero regressions in this (Linux) sandbox.
- **CONFIRMED on the user's real Windows machine (2026-08-24):** full
  suite run there -- `155 passed, 2 skipped in 29.01s`, zero failures,
  including all 6 previously-failing integration test files and the
  new `test_logging_setup.py` suite. The Windows `PermissionError` this
  section exists to fix is gone. This is the first real-Windows-machine
  confirmation of ANY test in this project (previously only real-scale
  *pipeline runs*, not the test suite itself, had been run there).

## Vocabulary-sampling-bias bug in Phase 5, found via real full-scale data (2026-08-13) — RESOLVED 2026-08-23

Phase 5's categorical vocabulary determination (for `patient` /
`apachePredVar` fields and the top-level `diagnosis` category)
originally sampled only `list(partition_result.partitions.keys())[:10]`
— since `hospital_partition.py` builds `partitions` by iterating
`sorted(available_hospitals)`, this always picked the 10 **smallest**
hospital IDs, not a representative sample of the 208-hospital cohort.

**Real symptom (user's full-scale run, 2026-08-13 —
`outputs/logs/pipeline.log`):**
```
[TABULAR] vocabularies determined from a 10-hospital sample: gender=3,
ethnicity=6, hospitaladmitsource=10, unitadmitsource=9, unitstaytype=3,
apacheadmissiondx=10, admitsource=7, teachtype=1, region=1, diagnosis=10
```
`teachtype` and `region` both collapsed to a single category
cohort-wide — clearly wrong for a 208-hospital dataset that genuinely
spans multiple US Census regions and both teaching and non-teaching
hospitals. `hospitaladmitsource`, `apacheadmissiondx`, and `diagnosis`
all separately hit the `top_k=10` ceiling, meaning real category
diversity was being truncated there too, independent of the sampling
bug.

Despite this bug, Phase 5 completed successfully end-to-end at real
full scale that same run, producing real `(199108, 64)` autoencoder
embeddings — the bug corrupted vocabulary *quality*, not pipeline
correctness/crash-safety, which is why it wasn't caught by the
crash-oriented real-data feedback loop that found the earlier
dtype/OOM/batching bugs.

**Fix (`src/data/tabular/feature_engineering.py`, `run_pipeline.py`):**
- `patient`/`apachePredVar` vocabularies now come from the **full
  cohort** (`determine_full_cohort_vocabularies()`) — safe re: the
  project's never-pool-across-hospitals rule specifically because both
  tables are already loaded once for the whole cohort upstream, not
  hospital-sharded (see that function's docstring for the full
  argument).
- `diagnosis` vocabulary (genuinely large, hospital-sharded) now
  aggregates running value-counts **hospital-by-hospital across every
  shard** (`accumulate_value_counts()` +
  `determine_diagnosis_vocabulary_from_shards()`), never pooling raw
  diagnosis rows across hospitals at once — preserving the sharding
  architecture for the one Phase 5 source table that actually needs it.
- Phase 4's `select_lab_features()` sampling (same bug pattern, smaller
  consequence — biased labname-substring detection rather than
  vocabulary collapse) switched from sorted-first-10 to a **seeded
  random** 10-hospital sample (`random.Random(cfg.get("run.seed",
  42))`) — kept bounded/sampled, not full-cohort, since `lab` is tens
  of millions of rows.
- Added `modalities.tabular.top_k_categorical_overrides` config (per-
  column top_k). Seeded with `apacheadmissiondx: 30`, `diagnosis: 20`,
  `hospitaladmitsource: 15` — a **starting point** informed by which
  columns hit the old ceiling in the real log above, NOT confirmed true
  cardinalities from eICU's data dictionary. Revisit with real figures
  if available.
- `configs/base.yaml`'s now-unused `vocabulary_sample_hospitals` key
  removed.
- `scripts/generate_mock_eicu.py`: `apachePredVar.teachtype`/`region`
  used to be hardcoded constants (`0`/`3`) for every mock patient
  regardless of hospital — meaning no mock-data test could ever have
  exercised real hospital-to-hospital categorical diversity for these
  fields. Now derived per-patient from their own hospital's
  `teachingstatus`/`region` (already generated per-hospital in
  `hospital.csv`), matching real eICU's actual field semantics.

**Verified:**
- Full baseline suite (147 tests as of 2026-08-23) passes with zero
  regressions from any of the above changes.
- 13 new unit tests
  (`tests/unit/test_tabular_feature_engineering.py`:
  `TestAccumulateValueCounts`, `TestDetermineCategoricalVocabularyFromValueCounts`,
  `TestDetermineFullCohortVocabularies`, `TestDetermineDiagnosisVocabularyFromShards`),
  including a direct reproduction of the exact teachtype/region
  collapse-to-1 symptom.
- 4 new integration tests
  (`tests/integration/test_phase5_vocabulary_bias_regression.py`),
  running the actual Phase 1→2→3→5 pipeline against 15-hospital mock
  data, including two "premise" tests that verify the fixture itself
  still reproduces the bug's shape (protects against silent fixture
  drift making the regression tests pass for the wrong reason).
- **Revert → fail → restore → pass proven twice**, by hand, both at the
  unit level and the full-pipeline integration level: temporarily
  reintroduced the exact sorted-first-10 sampling bug into
  `feature_engineering.py`, re-ran both suites and confirmed the
  relevant tests failed with the exact real-world symptom (region
  vocabulary `2 not greater than or equal to 4`, diagnosis vocabulary
  collapsing to `['diag_cardiovascular', 'diag_other']`), then restored
  the fix (byte-for-byte diffed against the pre-revert version to
  confirm) and confirmed all tests passed again.

**CONFIRMED on the user's real Windows machine, test-suite level
(2026-08-24):** full suite passed there (`155 passed, 2 skipped`),
including the vocabulary log line itself showing the fix working
end-to-end on real infrastructure:
```
[TABULAR] vocabularies determined from the full 75-patient cohort
(patient/apachePredVar) and all 15 diagnosis-bearing hospitals:
gender=2, ethnicity=6, ..., teachtype=2, region=4, diagnosis=4
```
`teachtype`/`region` no longer collapsed to 1, as designed. This was
still against generator-produced mock data (75-150 patients, 15
hospitals), not the real dataset, though.

**Not yet re-verified against the real 199,108-patient / 208-hospital
dataset** — this fix still needs to be run through Phase 5 at real full
scale to give the final confirmation this project actually cares about:
that `teachtype`/`region` show real diversity (not collapsed to 1)
against the real cohort, not just mock data. Run `python
run_pipeline.py --phase 5` (with `EICU_DATA_ROOT` set) and check the
`[TABULAR] vocabularies determined from the full ... cohort` log line.

## Second, independent bug behind the same symptom: apachePredVar.teachtype/region are genuinely constant — found + RESOLVED 2026-08-24

The real full-scale run (above) confirmed the vocabulary-sampling-bias
fix genuinely worked -- `hospitaladmitsource=15`, `apacheadmissiondx=30`,
`diagnosis=17` all showed real diversity beyond the old `top_k=10`
cap. But `teachtype=1, region=1` **persisted unchanged**, even though
the code was now provably using the full 199,108-patient cohort (not a
10-hospital sample) -- confirmed via the log's own
`(patient/apachePredVar)` wording and cohort-size figure.

**Root cause, confirmed by direct inspection of the raw CSV** (at my
request, bypassing this pipeline entirely):
```python
apv["teachtype"].value_counts(dropna=False)   # 0    171177  (ALL rows)
apv["region"].value_counts(dropna=False)      # 3    171177  (ALL rows)
hosp["region"].value_counts(dropna=False)     # Midwest 70, South 56, West 43, NaN 26, Northeast 13
hosp["teachingstatus"].value_counts(dropna=False)  # f 189, t 19
```
`apachePredVar.teachtype`/`region` are genuinely constant across the
entire real table -- not a sampling artifact, not a code bug. No
vocabulary-determination strategy could have surfaced diversity that
isn't in the source data. The real diversity was in `hospital.csv` all
along -- exactly the "site metadata... useful for characterizing
non-IID FL clients" `schema_registry.py` already described that table
as.

**Fix (`src/data/tabular/feature_engineering.py`, `run_pipeline.py`):**
- Removed `teachtype`/`region` from `APACHEPREDVAR_CATEGORICAL_FEATURES`
  entirely (a constant column is worse than useless as a one-hot
  feature -- it's dead weight, always encoding to the same category).
- Added `attach_hospital_features()`: merges `hospital.csv`'s
  `region`/`teachingstatus` onto `cohort_patients` via `hospitalid`, as
  new `hospital_region`/`hospital_teachingstatus` columns, now in
  `PATIENT_CATEGORICAL_FEATURES`. Includes a defensive
  duplicate-hospitalid guard (an `assert len(merged) == len(cohort_patients)`)
  -- a left-merge against a hospital-ID-duplicated reference table
  would otherwise silently multiply patient rows, exactly the class of
  silent corruption this project treats as unacceptable.
- Reverted an earlier, WRONG assumption baked into
  `scripts/generate_mock_eicu.py` on 2026-08-23: that mock
  `apachePredVar.teachtype`/`region` should vary per-hospital (mirroring
  `hospital.csv`). That assumption is now known to be false for the
  real data and has been reverted back to constants (`teachtype=0,
  region=3`, matching the real observed values), so mock data no longer
  misrepresents this table's actual behavior.

**Verified:**
- Full suite: 161 passed, 2 skipped (up from 155+2 -- 6 new unit tests
  for `attach_hospital_features()`, 2 redesigned to target the new
  `hospital_region`/`hospital_teachingstatus` columns instead of the
  now-removed `apachePredVar.teachtype`/`region`), zero regressions.
- **Revert → fail → restore → pass proven by hand again**, both unit
  and integration level: temporarily removed `hospital_region`/
  `hospital_teachingstatus` from `PATIENT_CATEGORICAL_FEATURES` and
  removed the `attach_hospital_features()` call from Phase 5 (the
  pre-fix state), reran the targeted tests, confirmed the vocabulary
  log line reverted to the old shape (no `hospital_region`/
  `hospital_teachingstatus` keys at all) and the region-diversity
  assertion failed (`0 not greater than or equal to 4`), restored both
  files (byte-diffed to confirm), confirmed all tests passed again.
- **Not yet re-verified against the real 199,108-patient / 208-hospital
  dataset** -- sandbox-verified only so far. Needs one more real
  `python run_pipeline.py --phase 5` run to confirm
  `hospital_region`/`hospital_teachingstatus` show real diversity (not
  vocab=1) against the actual cohort.

**CONFIRMED on the real 199,108-patient / 208-hospital dataset
(2026-08-25):**
```
[TABULAR] vocabularies determined from the full 199,108-patient cohort
(patient/hospital/apachePredVar) and all 206 diagnosis-bearing
hospitals: gender=4, ethnicity=6, hospitaladmitsource=15,
unitadmitsource=10, unitstaytype=4, apacheadmissiondx=30,
hospital_region=4, hospital_teachingstatus=2, admitsource=8, diagnosis=17
```
`hospital_region=4` and `hospital_teachingstatus=2` — exactly matching
the real diversity found in the raw `hospital.csv` inspection above (4
named regions, 2 teaching statuses). Both the vocabulary-sampling-bias
fix AND this hospital-source fix are now confirmed working end-to-end,
at every level: sandbox, the real Windows test suite, and real
full-scale data. This closes out the vocabulary-bias investigation.

**Phase 5 completion confirmed clean, same real run (2026-08-25):**
```
[PHASE5] Tabular feature matrix written to outputs\embeddings\tabular_features.npz
[AUTOENCODER_INFERENCE] progress: 81,920/199,108 (batch_size=4096)
[AUTOENCODER_INFERENCE] progress: 163,840/199,108 (batch_size=4096)
[AUTOENCODER_INFERENCE] progress: 199,108/199,108 (batch_size=4096)
[PHASE5] Autoencoder embeddings ((199108, 64)) written to outputs\embeddings\tabular_embeddings.npz
[PHASE5_TABULAR_MODALITY] COMPLETE (36.77s)
```
Real `(199108, 64)` embeddings produced, batched inference (OOM-backoff
mechanism from `src/utils/batching.py`) working correctly, zero errors.
Phase 5 is now DONE in the fullest sense: implemented, sandbox-verified,
Windows-test-suite-verified, and real-full-scale-verified, including
both bug fixes above. Next: Phase 7 (Multimodal Fusion) -- see below.

Note: an unrelated, transient `pandas.errors.ParserError: ... out of
memory` occurred on the FIRST attempt at this run, while chunk-reading
`lab.csv` during Phase 1's data-quality report (`data_report.py`) --
confirmed unrelated to any of the above (that code path was untouched
by this investigation, and is architecturally clean: it only ever
holds one chunk in memory, never accumulates raw rows). A plain retry
succeeded, consistent with transient system memory pressure rather
than a code or data bug. Worth keeping an eye out for if it recurs.

## Memory-efficiency rework (2026-08-09) -- second real-data crash, fixed

Running `full.yaml` (all 199,108 eligible patients, no per-hospital
cap) against real data crashed with an out-of-memory error inside
`combine_medication_sources()`'s `drop_duplicates()` -- pooling
`medication`+`medication_imputed`+`medication_imputed1` for the ENTIRE
dataset (~16.3M rows) before deduplicating doesn't scale once the
per-cohort filter stops meaningfully reducing row count (which is
exactly what happens in `full` mode).

**Fixed via `src/data/ingestion/hospital_sharding.py`:** large tables
are now read once, in chunks, column-projected, and routed to small
per-hospital CSV shards on disk (cached). Phase 3 and Phase 4 both loop
hospital-by-hospital rather than pooling. Peak memory is now bounded by
the largest single hospital's data (~7,059 patients for `hospitalid=73`
per the user's own hospital list) rather than the whole dataset.
Full design rationale and what changed in `docs/memory_efficiency.md`
(memory usage at the actual 200K-patient full scale is now confirmed —
see the "Update 2026-08-10" section there).

**Verified:** 10 new tests (8 unit + 2 integration), including an
uneven-hospital-size scenario (one large hospital, several small,
forced multi-chunk reads) that exercises the same code paths a
full-mode run would, plus an explicit cache-reuse-across-phases check.
**Also verified at real scale (2026-08-10):** the user's `full.yaml`
run completed Phase 3 in 557s across all 199,108 patients / 208
hospitals with no crash — peak memory bounded as designed.

## Third real-data crash, found and fixed (2026-08-09, full-mode) — RESOLVED 2026-08-10

Running `full.yaml` (all 199,108 patients) succeeded through Phase 3
(~861s, no crash — real confirmation the hospital-sharding fix holds at
true scale), then Phase 4 crashed:

```
RuntimeError: [enforce fail at alloc_cpu.cpp:117] data.
DefaultCPUAllocator: not enough memory: you tried to allocate
5780424704 bytes.
```

**Root cause:** the LSTM forward pass had never been batched — all
199,108 patients went through in one call. Fixed with a generic,
torch-agnostic adaptive-batch-size runner with OOM backoff
(`src/utils/batching.py`, 14 unit tests). Also fixed while auditing:
`medication_imputed1` was mislabeled as small in the schema registry
(confirmed by this run to actually be ~7.3M rows/3.27GB, same size as
raw `medication`) causing a risky non-chunked Phase 1 read; and added
progress logging to Phase 3/4's per-hospital loops, which had gone
silent for 9-16 minutes at a time with zero log output. Full writeup:
`docs/memory_efficiency.md`'s "Update 2026-08-09" section.

**Verified:** the batching/backoff logic (14 tests, fully torch-free).
**Also verified at real full scale (2026-08-10):** re-ran `full.yaml`
end-to-end — all 4 phases completed with zero errors. LSTM inference
processed all 199,108 patients at `batch_size=512` **constant
throughout, zero backoff events needed** — 54 seconds total. Produced
real `(199108, 128)` embeddings. Total pipeline time end-to-end:
~26 minutes (Phase 1: 447s, Phase 2: 4s, Phase 3: 557s, Phase 4: 564s).
Sepsis counts (`sepsis_strict_positive=262`, `sepsis_broad_positive=44412`)
matched the prior (crashed) attempt exactly — confirms determinism held.

## Bug found and fixed via the user's real-data execution (2026-08-09)

Running Phase 1-4 against the user's actual eICU data (200,859
patients, 208 hospitals) crashed in `build_cohort_report` with:
```
KeyError: "None of [Index([-1, -1, -1, -2, ...], dtype='object')] are in the [index]"
```

**Root cause:** ~468 of 4,064 sampled patients had no `apachePredVar`
row at all (real, expected -- only 171,177/200,859 patients dataset-wide
have one). Merging tables together introduced `NaN` into what should
have been boolean columns (`organ_dysfunction`, `sepsis_broad`,
`sepsis_strict`); pandas silently upcasts a bool column to `object`
dtype to hold that NaN, and `.fillna(False)` does not restore the dtype.
`~column` (meant as logical NOT) then called Python's bitwise invert on
the object-dtype bools instead (bool is an int subclass: `~True == -2`,
`~False == -1`), which is where the crash surfaced. Without the crash,
this same corruption would have silently produced wrong
`cohort_overlap` counts rather than an error -- worse than a loud
failure.

**Why 77 passing tests didn't catch it:** the mock generator gave every
patient a 1:1 `apachePredVar` row and near-universal `microLab`
coverage (~67% of mock patients vs. ~1.5% in real eICU), so the
merge-introduces-NaN-into-a-bool-column path never actually triggered
in any test.

**Fixed in:**
- `src/data/cohort/sepsis_label.py`, `infection_proxy.py`,
  `cohort_report.py` -- every `.fillna(False)` on what's meant to be a
  boolean column now has an explicit `.astype(bool)` immediately after
  it, at every merge boundary where NaN could have been introduced.
- `scripts/generate_mock_eicu.py` -- now gives only ~85% of mock
  patients an `apachePredVar` row (matching real eICU's ratio) and only
  ~15% of patients a `microLab` entry (still ~10x denser than real
  eICU's ~1.5%, as a deliberate testability compromise, but no longer
  ~45x denser) and allows 0 medication orders for some patients. This
  is what should have caught the bug originally.
- Two new regression tests
  (`test_partial_coverage_regression_no_dtype_corruption` and
  `test_patients_missing_from_sepsis_df_do_not_crash_or_corrupt_overlap`)
  explicitly construct the partial-coverage scenario and assert both
  correct dtype AND correct output values. **Verified these tests
  actually catch the bug**: reverted the fix, confirmed both tests fail
  with the exact same `KeyError` and `dtype('O')` the user hit, then
  restored the fix and confirmed they pass.

**A genuine research finding, not a bug:** the user's real run showed
`sepsis_strict_positive=2` out of 4,064 sampled patients (0.05%) vs.
`sepsis_broad_positive=1331`. This is expected given `microLab`'s
extreme sparsity (2,923 patients with any culture order out of 200,859
total, ~1.5%) combined with random per-hospital patient sampling not
being stratified toward microLab coverage. Worth deciding before
drawing conclusions from the strict cohort at this subset scale:
increase `patient_subset`, run in `full` mode, or treat the strict
cohort's current near-emptiness as itself a documented limitation in
the methodology write-up (as `docs/sepsis_label_methodology.md`
already anticipates re: microLab's coverage).

## What Phase 3 currently does NOT do (by design, not oversight)

- Does not use `treatment.csv` in scoring yet, despite it being a
  confirmed table — it's discovered/validated in Phase 1 but not yet
  consumed as a corroborating signal for organ-support interventions
  (e.g. cross-checking vasopressor/ventilation treatment strings
  against `infusionDrug`/`apachePredVar`). Candidate Phase 3
  refinement, not required for the current dual-cohort design to
  function.
- `n_domains_observed` / `max_possible_score` are computed per patient
  in `sofa_like_scoring.py` but not yet surfaced in the cohort report —
  worth adding before treating `sofa_like_total` comparisons across
  patients as apples-to-apples in a write-up.
