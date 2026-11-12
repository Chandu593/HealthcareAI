# Three-modality audit (2026-11-06) -- existing behaviour, deviations, corrections

Scope: the three-modality package (`src/three_modality/`), `run_three_modality.py`, the reusable graph / GCN / sampling / classifier / evaluation code, the configs and the real run
artifacts uploaded with the repo (`outputs/three_modality/...`). Nothing under the paper track, `src/fusion/cross_attention_fusion.py` (two-token), v21, s1-s5 outputs, label runs,
encoder runs or the split manifest was modified.

## 1. Verified OK (evidence from the real files, not assumptions)

| Check | Result |
|---|---|
| Same patients / order / split / hospital in the TCD, GRU and tabular input runs | `patients.csv` is byte-identical in the three runs (same hash recorded in each manifest); the incident-label run covers all 199,108 patients with identical `split` and `hospitalid` |
| Permanent split | train / val / test = 139,208 / 29,932 / 29,968 over 207 hospitals; loaded from the manifest, never rebuilt |
| Strict landmark | reproduced exactly: `icu_exit_hours >= 24` gives 131,695 (92,116 / 19,777 / 19,802) |
| Incident-sepsis eligibility | the incident cohort (71,954; train 50,262 / val 10,851 / test 10,841) lies entirely inside the landmark; no NaN label among included patients; all 4,542 positive event hours are in (24, 96] |
| First-24-hour inputs | TCD cutoff audit passed (features unchanged when late rows were corrupted); GRU cutoff audit `passed=True` (2,000 patients, 105,554 late rows corrupted); the tabular diagnosis / admission-drug groups are window-bounded by `iter_windowed` (code review) and the 181 kept tabular column names contain no outcome / discharge / APACHE-score name |
| Encoders pooled and labelled as such | no three-modality code or doc calls any stage federated |
| Graph builders | `build_hospital_graph` (Graph A = kNN AND cosine threshold; Graph B = MiniBatchKMeans candidates + the same rule) is hospital-local by construction and never reads labels |
| Legacy sampling | `src/classifiers/imbalance_sampling.py` is hospital-wise and train-only |

## 2. Deviations found and what was done

| # | Finding | Consequence | Action |
|---|---|---|---|
| D1 | `src/three_modality/cross_attention3.py` and `downstream_config.py` were already in the repo: untested, not wired into the CLI/config, no docs | unverified code that could be used silently | audited line by line, tested (32 tests), wired as stage `cross_attention`, documented |
| D2 | Draft cross-attention had a **dead parameter**: `missing_in` could never affect the output (the unavailable token is masked as a key and its output slot is overridden), found by the gradient-flow test | violates "gradient flows through all parameters"; misleading | removed; the unavailable token is now only its modality type embedding (still masked), output slot `missing_out` stays learned, availability flags still reach the output layer |
| D3 | Draft did not verify that each encoder's embedding file matches the SHA-256 in its own manifest | a swapped / modified embedding file would pass | verification added; recorded in the manifest (`embedding_file_sha256_verified`) |
| D4 | Draft loaded the full TCD/GRU preprocessing state just to compute availability | needless coupling and memory | reads only `tcd_features_raw.npy`, `feature_schema.json`, `gru_mask.npy` (memory-mapped) |
| D5 | `patients.csv:window_end_minutes >= 1440` gives **132,900** patients, not the 131,695 landmark (1,205 stays with an invalid ICU-exit time are not clipped) | deriving the landmark from `window_end_minutes` would silently include them | the downstream code takes the landmark only from the cross-attention run's `landmark` array (built by `landmark_mask`); regression test `test_patient_with_invalid_exit_time_is_not_landmark` |
| D6 | The legacy flat / graph ablation classifiers (`fit_and_evaluate_classifier`, `graph_embedding_ablation_runner`, `embedding_ablation_runner`) concatenate embeddings of ALL hospitals and fit ONE classifier on the pooled training rows | **pooled classification** -- forbidden in the new design | left untouched (legacy, out of scope) and NOT reused; the new layer (`hw_classifiers.py`) fits one model per hospital |
| D7 | Legacy graph evaluation attaches validation/test nodes to each other (the evaluation graph spans train+val+test) | a test patient's prediction can depend on other test patients' features | new design: the graph is built from TRAINING patients only; val and test patients are attached separately, only to training patients, with directed edges train -> query |
| D8 | `task_labels.npz` of a cross-attention run contains the test labels (needed for evaluation) | test labels sit next to training inputs | `HospitalTaskData.labels()` refuses the test split; test labels are reachable only through `test_labels_for_evaluation()`, called after thresholds are fixed; proven by tests (flipping test labels changes only test metrics, for Configs 1-4) |
| D9 | No real-data poison audit exists for the tabular static feature set | cannot claim the tabular set was audited on the real data | **open** -- code review and the forbidden-name check only; offered as a follow-up |
| D10 | Real hospitals are tiny for hospital-wise learning: sepsis median 10 train / 3 val / 2 test positives per hospital; only 97 of 151 hospitals have a positive in every split | per-hospital metrics are very noisy; many hospitals cannot be trained | explicit skip rules (`downstream.min_samples`) with reasons; per-hospital low-power flag; macro AND sample-weighted averages; skips reported |

## 3. Not changed on purpose
The encoders stay centrally pooled (their runs are the frozen inputs); the two-token cross-attention, paper track, label runs, encoder runs and the split manifest are untouched.
