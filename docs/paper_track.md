# Paper track (`run_paper.py`, `src/paper/`)

Added 2026-10-08. **Additive**: Phase 1-13 and the 2026-09 ablation code paths keep their behaviour, so every
earlier result stays reproducible. The paper track exists because the earlier flat/graph results were limited by
how the pipeline was built, not by the task (diagnosis below).

## Why the earlier results were weak (verified in code)

| Finding | Where | Consequence |
|---|---|---|
| Encoders were trained for `epochs=5` **full-batch** steps per hospital (= 5 Adam steps) | `embedding_ablation_runner._train_one_hospital_flat_model` | embeddings ~ random projections; architecture choice barely mattered |
| `weighted_bce` is plain BCE (no class weight) | `losses.masked_bce_loss` | 9%-prevalence mortality: the few steps mostly learned the prior |
| One independently seeded encoder **per hospital**, embeddings pooled for one classifier | embedding ablation | embedding dimension *j* means different things in different hospitals |
| Graph GNN: 10 full-batch steps/hospital, local only; flat classifier pooled over all hospitals | graph runners | "graph < flat" was confounded with data scarcity, not evidence about graphs |
| Sequence = whole stay (<=48x4h bins), length derived from `unitdischargeoffset` | `sequence_builder` | no defined prediction time; LOS leaks into the input |
| TS inputs contain `medication_antibiotic_active`, `infusion_vasopressor_active`, platelet/bilirubin/creatinine = the sepsis label's own ingredients | `time_binning`, `sofa_like_scoring`, `infection_proxy` | sepsis AUROC ~0.87-0.89 is partly label reconstruction -- why sepsis >> mortality |
| Train metrics at 0.5, val/test at a val-tuned threshold | classifier wrapper | train and test numbers not comparable |
| No raw-feature baseline, no CIs, single seed, no per-hospital metrics | -- | not reviewer-proof |

## What the paper track does

1. **Defined observation window.** Features use only events with `-lookback <= offset <= min(24h, unitdischargeoffset)`.
   `unitdischargeoffset` clips the window and is never a feature. `window.min_icu_los_minutes: 1440` gives the strict
   landmark cohort as a sensitivity run.
2. **Wide, leakage-safe features from the full export** (`src/paper/features.py`, streaming, chunked): demographics,
   hospital metadata, `apachePredVar` day-1 inputs (whitelist; `diedinhospital`, `dischargelocation`, SAPS3 today/yesterday
   never read), APACHE-derived scores (optional group), top-40 labs x {mean,min,max,first,last,std,count,delta}, treatment /
   diagnosis / admission-drug / medication / infusion token flags, antibiotic & vasopressor flags, microLab counts, plus an
   hourly (2h-binned) lab tensor with an observed-mask. Vocabularies are chosen from **train patients only**.
3. **Feature sets as an honest ablation.** Mortality: `full` and `no_apache_scores`. Sepsis: `sepsis_controlled` removes every
   label ingredient (antibiotic, vasopressor, SOFA labs, APACHE physiology/scores, microLab, treatment/medication/infusion
   tokens, infection diagnoses); `sepsis_uncontrolled` is available to *measure* the circularity gap.
4. **Arms** on the same patients and the same permanent split manifest: `logreg_central`, tuned `lightgbm_central`
   (val-selected random search), optional `xgboost_central`, `mlp_central` (non-federated reference), optional `mlp_local_only`,
   `fedavg_mlp` (ONE shared encoder trained by FedAvg -> aligned embedding space), and `fedavg_graph_{none,A,B}`
   (per-hospital GraphSAGE heads on the frozen shared embeddings, FedAvg-trained; `none` = same net without edges, so message
   passing is isolated; residual to the base logit).
5. **Training that can actually learn:** mini-batches, real `pos_weight` (sqrt of neg/pos by default), early stopping on
   validation, best-state restore, per-hospital train-only standardization (fallback to federated-aggregated sufficient
   statistics for tiny hospitals), FedAvg with batch 128 / 5 local epochs / lr 3e-3 (the log prints the n-weighted
   sequential steps per round; underfitting FedAvg is the same failure mode as the old encoders).
6. **Evaluation:** one threshold rule (max-F1 on val) applied to train/val/test; Platt calibration fit on val; AUROC/AUPRC
   (+ prevalence), F1, Brier, ECE; 95% bootstrap CIs; paired bootstrap against `lightgbm_central`; per-hospital macro/median
   AUROC; seeds; APACHE IVa predicted mortality as a **benchmark only**; patient-level predictions saved to
   `predictions/*.npz` (patient_id, hospital_id, split, y, score, calibrated p).

## Run it

```
# 1) smoke (mock or a few hospitals)
python run_paper.py --config configs/paper_debug.yaml
# 2) real data, first pass (one seed)
python run_paper.py --config configs/paper.yaml --reuse-phase1to3
# restrict while iterating
python run_paper.py --config configs/paper.yaml --reuse-phase1to3 --tasks mortality --feature-sets no_apache_scores --arms lightgbm_central,logreg_central
# final table: add --seeds 42,43,44 and include mlp_local_only in --arms (see the staged plan below)
```
Outputs: `outputs/paper/<run_id>/{results.json, results_table.csv, paper_report.md, predictions/}`; the feature matrix is cached
under `outputs/paper/feature_cache/` (keyed on cohort/split/config). Config files must sit next to `base.yaml` (existing convention).
Order to run: `lightgbm_central` first -- it tells you the information ceiling of the features before any neural/federated work.

## Interpreting results (what a reviewer will ask)

* The honest headline for mortality is `fedavg_*` vs `lightgbm_central`/`logreg_central` on identical patients, with the paired
  bootstrap deltas. A federated model within ~0.01 AUROC of the centralized reference is the usual FL claim.
* If `fedavg_graph_A/B` do not beat `fedavg_graph_none`, say so: the report prints `best_round` (0 = the graph never helped on
  validation) and the train-edge label agreement vs chance (if neighbours are no more label-similar than random, a graph cannot help).
* Report `no_apache_scores` next to `full`: the first uses raw measurements only.
* Sepsis is a **proxy label**. Report the controlled number as the real one; the uncontrolled-vs-controlled gap is itself a finding.
* Base-paper numbers (config `reference_results`) come from cohorts/windows/feature sets that differ from ours -- cite them with that caveat.

## Verification status

Sandbox (mock data): 33 new tests (features 6, preprocessing/evaluation 9, trainers/graph 6, results-writer CSV 3, loss 2, end-to-end 7)
plus 3 previously failing LightGBM tests now passing; full suite 595 passed / 2 failed / 2 skipped. The 2 failures are pre-existing and
unrelated (`test_modality_architecture_options` assumes the default time-series encoder is `lstm`, but `configs/base.yaml` currently sets
`modalities.timeseries.encoder: transformer`). Guards were mutation-checked by hand (revert -> fail -> restore -> pass): window upper bound,
discharge clipping, train-only vocabulary, outcome-column whitelist; FedAvg n-weighting, cross-hospital graph pooling, train-only statistics
(per-hospital and global-fallback); old CSV header behaviour. Scale check: 12K patients / 40 hospitals in 41 s, ~0.9 GB peak. FedAvg with the
default settings reaches the centralized MLP on mock data (0.726 vs 0.727 test AUROC). **Not yet run on the real 199,108-patient dataset** -- first real run is the next step.

## Staged full-run plan (each stage has a decision gate)

Run **mortality and sepsis as separate invocations** (separate models, separate feature sets). The feature bundle is cached in
`outputs/paper/feature_cache/`, keyed on window/cohort/feature settings -- every strict-24h run (mortality or sepsis) shares one bundle, all-stays
another, the 6h family a third. Flags: `--tasks`, `--feature-sets`, `--arms` (runs EXACTLY the listed arms), `--seeds`, and
`--set key.path=value` (repeatable) for any config value. Results are written after every finished experiment.

**Stage 1 finding (2026-10-10, real data).** All-stays first-24h LightGBM AUROC 0.9092 splits into 0.8815 for stays that last >=24h and 0.9528 for stays
that end earlier (window truncated), so the all-stays headline contains a short-stay effect of ~0.03 AUROC. On the strict 24h landmark cohort
(131,695 patients; train/val/test 92,116/19,777/19,802; prevalence 9.1%) LightGBM reaches 0.8799 [0.8721, 0.8874] (AUPRC 0.491) and logistic regression
0.8603. **Primary cohort for the paper = strict 24h landmark**; all-stays is reported as the secondary protocol (comparable with prior work that uses
every stay). The `no_apache_no_eol` run was identical to `no_apache_scores` (696 = 696 features) because none of the end-of-life patterns is in the top-60
treatment/diagnosis vocabulary -- that run proves nothing; use `scripts/paper_eol_audit.py` instead.

| Stage | Command (single line; repo root; `EICU_DATA_ROOT` set) | Gate |
|---|---|---|
| 0 (done) all-stays central | `python run_paper.py --config configs/paper.yaml --reuse-phase1to3 --tasks mortality --arms lightgbm_central,logreg_central` | 0.9096 / 0.8892 |
| 1 (done) strict 24h central | `python run_paper.py --config configs/paper_landmark24.yaml --reuse-phase1to3 --run-id landmark24` | 0.8799 / 0.8603 |
| 1B strict-cohort subgroups | `python scripts/paper_subgroup_analysis.py --run-dir outputs/paper/landmark24 --experiment mortality__no_apache_scores --arm lightgbm_central --data-root %EICU_DATA_ROOT%` | APACHE-matched comparison on the strict cohort |
| 1C end-of-life audit | `python scripts/paper_eol_audit.py --data-root %EICU_DATA_ROOT% --min-los 1440` | few patients flagged, or weak link to death |
| 2 federated mortality (strict) | `python run_paper.py --config configs/paper_landmark24.yaml --reuse-phase1to3 --tasks mortality --feature-sets no_apache_scores --arms lightgbm_central,mlp_central,fedavg_mlp,fedavg_graph --seeds 42 --run-id mort_fed_s42` | fedavg within ~0.02 of centralized |
| 3A sepsis 24h strict, central | `python run_paper.py --config configs/paper_sepsis.yaml --reuse-phase1to3 --tasks sepsis --arms logreg_central,lightgbm_central --set paper.window.min_icu_los_minutes=1440 --run-id sepsis24_central` | controlled vs uncontrolled gap |
| 3B sepsis 24h strict, federated | `python run_paper.py --config configs/paper.yaml --reuse-phase1to3 --tasks sepsis --feature-sets sepsis_controlled --arms lightgbm_central,mlp_central,fedavg_mlp,fedavg_graph --set paper.window.min_icu_los_minutes=1440 --seeds 42 --run-id sepsis24_fed_s42` | |
| 3C sepsis 6h early warning (stays >=6h), central | `python run_paper.py --config configs/paper_sepsis_early.yaml --reuse-phase1to3 --tasks sepsis --arms logreg_central,lightgbm_central --set paper.window.min_icu_los_minutes=360 --run-id sepsis6h_central` | |
| 3D sepsis 6h, federated | `python run_paper.py --config configs/paper_sepsis_early.yaml --reuse-phase1to3 --tasks sepsis --feature-sets sepsis_controlled --arms lightgbm_central,mlp_central,fedavg_mlp,fedavg_graph --set paper.window.min_icu_los_minutes=360 --seeds 42 --run-id sepsis6h_fed_s42` | |
| 4 sampling negative result (strict) | `python run_paper.py --config configs/paper_sampling.yaml --reuse-phase1to3 --tasks mortality --set paper.window.min_icu_los_minutes=1440 --run-id sampling_check` | expect <= no sampling |
| 5 final (3 seeds + local-only) | stage 2 / 3B / 3D with `--arms logreg_central,lightgbm_central,mlp_central,mlp_local_only,fedavg_mlp,fedavg_graph --seeds 42,43,44`, run-ids `final_*` | mean +/- sd, paired CIs |

Paired comparisons are made against `lightgbm_central`, so keep it in `--arms` whenever you want deltas. Select any further modelling choices
(wider vocabularies, more LightGBM trials, FedProx) on **validation** metrics only.

## Sepsis in this track

The proxy label is (antibiotic within -24h..+48h) AND (SOFA-like >= 2 from first-24h worst values). `sepsis_controlled` removes every
group that feeds or duplicates it, so it answers "how well can sepsis be predicted from information the label does NOT use?".
The 24h-window result is therefore lower than the old ~0.87-0.89, by construction. `paper_sepsis_early.yaml` (6h window) makes
the task forward-looking because most of the label's evidence arrives after the window. Strict sepsis (microLab-corroborated) has only
~262 positives -- too few to model; use broad.


## Real results so far (2026-10-10; 1 seed; strict 24h landmark unless noted)

| Task / protocol | LR | LightGBM | MLP (central) | FedAvg MLP | Graph none / A / B |
|---|---|---|---|---|---|
| Mortality, strict 24h (test n=19,802) | 0.8603 | **0.8799** | 0.8699 | 0.8306 | 0.8306 / 0.8316 / 0.8316 |
| Mortality, all stays (n=29,968) | 0.8892 | 0.9092 | -- | -- | -- |
| Sepsis 24h controlled (n=19,610) | 0.8749 | 0.9097 | 0.8949 | 0.8206 | 0.8206 / 0.8210 / 0.8208 |
| Sepsis 24h uncontrolled | 0.9759 | 0.9930 | -- | -- | -- |
| Sepsis 6h controlled, stays >=6h (n=27,191) | 0.8752 | 0.9067 | 0.8936 | 0.8207 | 0.8207 / 0.8208 / 0.8207 |
| Sepsis 6h uncontrolled | 0.9517 | 0.9765 | -- | -- | -- |

* APACHE IVa on the same strict-cohort patients: 0.8281 vs ours 0.8817 (+0.054 [0.046, 0.062]).
* End-of-life audit (strict cohort): 2.48% of patients carry a matching string, but most matches are *alcohol/drug withdrawal* (the pattern "withdraw" is too broad);
  true palliative-care strings cover ~726 patients holding ~1.4% of deaths -> cannot explain the AUROC.
* Sampling (strict): undersample 0.8769, oversample 0.8786 vs 0.8799 -> no gain (negative result confirmed on strong features).
* **Open problems:** (1) FedAvg MLP is 0.04-0.07 AUROC below the centralized MLP (gate was 0.02) and its validation AUROC peaks at round ~6-8 then declines;
  (2) the graph arms add nothing measurable; (3) the "controlled" sepsis score (0.91) is not lower than the old 0.89 and barely changes between a 6h and a 24h
  window, which points to static admission information (infection-related admitting diagnoses, organ-failure diagnoses, oxygenation labs) that the controlled set
  still contains -> the `sepsis_deconfounded` / `sepsis_no_labs` ladder.

## Stage 5, revised (do NOT start multi-seed federated runs until 5A passes)

**5A. Federated tuning, validation only** (mortality, strict cohort). Screen tab-only for speed (`use_sequence=false`); each variant is one `--arms fedavg_mlp` run:

```
python run_paper.py --config configs/paper_landmark24.yaml --reuse-phase1to3 --tasks mortality --feature-sets no_apache_scores --arms mlp_central --set paper.neural.use_sequence=false --seeds 42 --run-id mlp_central_noseq
python run_paper.py --config configs/paper_landmark24.yaml --reuse-phase1to3 --tasks mortality --feature-sets no_apache_scores --arms fedavg_logreg --seeds 42 --run-id fedavg_logreg_s42
python run_paper.py --config configs/paper_landmark24.yaml --reuse-phase1to3 --tasks mortality --feature-sets no_apache_scores --arms fedavg_mlp --set paper.neural.use_sequence=false --set paper.federated.lr=0.001 --set paper.federated.local_epochs=2 --set paper.federated.rounds=60 --set paper.federated.patience=12 --seeds 42 --run-id fed_v1
python run_paper.py --config configs/paper_landmark24.yaml --reuse-phase1to3 --tasks mortality --feature-sets no_apache_scores --arms fedavg_mlp --set paper.neural.use_sequence=false --set paper.federated.max_local_steps=20 --set paper.federated.lr_decay=0.95 --set paper.federated.rounds=60 --set paper.federated.patience=12 --seeds 42 --run-id fed_v2
python run_paper.py --config configs/paper_landmark24.yaml --reuse-phase1to3 --tasks mortality --feature-sets no_apache_scores --arms fedavg_mlp --set paper.neural.use_sequence=false --set paper.federated.lr=0.001 --set paper.federated.local_epochs=2 --set paper.federated.server_momentum=0.9 --set paper.federated.rounds=60 --set paper.federated.patience=12 --seeds 42 --run-id fed_v3
python run_paper.py --config configs/paper_landmark24.yaml --reuse-phase1to3 --tasks mortality --feature-sets no_apache_scores --arms fedavg_mlp --set paper.neural.use_sequence=false --set paper.federated.lr=0.001 --set paper.federated.local_epochs=2 --set paper.federated.fedprox_mu=0.01 --set paper.federated.rounds=60 --set paper.federated.patience=12 --seeds 42 --run-id fed_v4
```
Pick the variant with the highest **validation** AUROC (`results_table.csv`, rows `split=val`); do not choose on test. Accept if within ~0.02 of the central MLP's validation AUROC.
`fedavg_logreg` is a convex baseline that should be stable; it is also what the base papers' federated linear models correspond to.

**5B. Sepsis ladder** (central LightGBM, strict 24h and 6h): `--feature-sets sepsis_deconfounded,sepsis_no_hospital,sepsis_no_labs` with `--arms lightgbm_central`. Report the whole ladder
uncontrolled -> controlled -> deconfounded -> no-hospital -> no-labs and rely on the **per-hospital macro AUROC** for sepsis (pooled AUROC is inflated by between-hospital label-rate differences).

**5C. Final multi-seed** with the chosen federated flags (repeat them as `--set` options) and `--arms logreg_central,lightgbm_central,mlp_central,mlp_local_only,fedavg_logreg,fedavg_mlp --seeds 42,43,44`,
run-ids `final_mortality`, `final_sepsis24`, `final_sepsis6h`. Add `fedavg_graph` only if you decide to keep the graph component (it did not help in the first run).

**5D. Reporting:** per-seed mean +/- sd, bootstrap CIs, paired deltas vs LightGBM, per-hospital median/IQR, subgroup tables
(`scripts/paper_subgroup_analysis.py` on the final predictions), APACHE-matched comparison, the end-of-life audit sentence, strict **and** all-stays protocols.


## What the first sepsis reports showed (sepsis24_central, sepsis24_fed_s42) and the new diagnostics

Top LightGBM features of the **controlled** sepsis model (gain share): paCO2 (mean/max/last/min, ~13.4% together), hospital size (`hosp_log_n_patients`, 5.2%) and region (~2.3%),
`apv_activetx`, `apv_graftcount` (CABG), BUN, albumin, paO2. Reading: (1) the arterial-blood-gas family stands in for ventilation / the respiratory SOFA input;
(2) hospital identity carries signal because the proxy label's positive rate varies by hospital (antibiotic-charting coverage); (3) `graftcount` suggests the proxy flags
routine post-cardiac-surgery antibiotics + sedation/ventilation as "sepsis". Evidence in the numbers: pooled LightGBM AUROC 0.9097 but per-hospital macro AUROC 0.8676;
for FedAvg the per-hospital macro AUROC (0.8393) is *higher* than its pooled AUROC (0.8206), i.e. the federated model ranks within hospitals nearly as well as the
centralized MLP (0.8508) but its scores are not comparable across hospitals. v6 adds:

* `scripts/paper_local_calibration.py` -- re-scores saved predictions with per-hospital intercept calibration (each hospital fits ONE scalar on its own validation rows;
  shrinkage prior; no retraining) and prints pooled vs per-hospital AUROC. Every new run also reports "AUROC with local calibration" and writes `val_local`/`test_local` rows.
* `scripts/paper_sepsis_label_audit.py` -- CABG share, overlap with documented sepsis diagnoses, per-hospital proxy rate vs medication-record coverage.
* Ladder fixed: `sepsis_deconfounded` now removes the whole blood-gas family (paCO2, pH, HCO3, base excess, paO2, FiO2, O2 sat), infection admitting diagnoses and organ-failure
  diagnosis tokens; new `sepsis_no_hospital` also removes hospital-level covariates; `sepsis_no_labs` removes labs too. Strictly nested (tested).


## 5A results (2026-10-11) and the decisions taken

Mortality, strict 24h cohort, 1 seed, test n=19,802, tab-only unless noted (validation AUROC = peak over rounds):

| Federated variant | val AUROC | test AUROC [95% CI] | per-hosp macro | AUROC with local calibration |
|---|---|---|---|---|
| (earlier) FedAvg MLP lr 3e-3, E=5, tab+seq | ~0.822 | 0.8306 | | |
| fedavg_logreg | 0.8281 | 0.8334 [0.8236, 0.8422] | 0.8372 | 0.8478 |
| v1 lr 1e-3, E=2 | 0.8298 | 0.8341 | 0.8324 | 0.8420 |
| v2 step cap 20 + lr decay 0.95 | 0.8285 | 0.8341 | 0.8331 | 0.8384 |
| v3 v1 + server momentum 0.9 | 0.8355 | 0.8362 | 0.8357 | 0.8457 |
| **v4 v1 + FedProx mu=0.01** | **0.8374** | **0.8402 [0.8306, 0.8487]** | 0.8373 | 0.8469 |

Central references: LightGBM 0.8799, MLP tab+seq 0.8699, MLP tab-only 0.8684 (the GRU branch adds ~0.0015, inside the CI -> no measurable multimodal gain over the
engineered lab statistics), LR 0.8603. Federation costs ~0.03 AUROC for BOTH the linear and the MLP model, which points at the federated setup
(non-IID clients, unequal local steps, tiny hospitals) rather than model capacity. Selected on validation: **v4** (provisional winning flags):
`--set paper.neural.use_sequence=false --set paper.federated.lr=0.001 --set paper.federated.local_epochs=2 --set paper.federated.fedprox_mu=0.01 --set paper.federated.rounds=60 --set paper.federated.patience=12`.
Round 2 (time-boxed) tests stronger FedProx / momentum+FedProx / step cap without decay, plus a 1-minute-per-run grid on federated logistic regression to see which lever closes the convex gap.

Sepsis ladder so far (LightGBM, pooled AUROC): uncontrolled 0.9930 / 0.9765 (24h / 6h) -> controlled 0.9097 / 0.9067 -> deconfounded 0.8978 / 0.8972 -> no-labs-no-hospital 0.8014 / 0.8131.
`sepsis_no_hospital` (labs kept, hospital covariates removed) is the missing rung that separates the hospital-identity effect from the lab effect.

Tooling added in v7: `scripts/paper_collect_results.py` (one summary file for all runs), APACHE benchmark in `paper_subgroup_analysis.py` now prefers the IVa row (matches the runner's 0.8309).


## Round 2 results and the sepsis label audit (2026-10-11)

**FedProx is the lever.** Mortality, strict 24h, 1 seed (test n=19,802): FedProx mu 0.01 -> 0.8402, 0.05 -> 0.8470, **0.1 -> 0.8506 [0.8417, 0.8589]** (val 0.8479; with
per-hospital calibration 0.8559). Momentum on top of FedProx adds nothing (0.8472); a local-step cap does not help (logreg cap 10: 0.8329); federated logistic regression with mu=0.1:
0.8453 (from 0.8334). Gap to the central MLP is now 0.018 on test (val gap 0.015), i.e. inside the 0.02 gate. mu has not saturated -> round 3 tries 0.3 and 1.0.
Graph arms (old settings): within-hospital AUROC +0.0014/+0.0016 over no-graph -> no meaningful benefit.

**Sepsis ladder (LightGBM, 24h; pooled / per-hospital macro / pooled with per-hospital calibration):** uncontrolled 0.9930/0.9900/-; controlled 0.9097/0.8676/0.9109;
deconfounded 0.8978/0.8449/0.8989; no-hospital 0.8793/0.8394/0.8947; no-labs 0.8014/0.7846/0.8570 (6h: 0.9765, 0.9067, 0.8972, 0.8758, 0.8131). Per-hospital calibration restores
most of what hospital covariates provided -> it is a between-hospital base-rate effect. FedAvg sepsis (old settings) 0.8206 pooled -> 0.8863 with per-hospital calibration
(central MLP 0.9010). Top features without labs: `apv_activetx` (15-19% of gain) -> excluded in the new `sepsis_clean` set.

**Label audit (strict cohort, 131,695 stays):** proxy prevalence 28.4% vs documented sepsis diagnosis 16.5%; the proxy catches 50.1% of documented cases and only 29.1% of proxy positives
have a documented diagnosis. Across 188 hospitals the proxy rate has median 0.28 but 46 hospitals have **zero** positives and 49 have <50% medication-record coverage; Spearman(rate, coverage)=0.71.
The antibiotic half of the label cannot fire without medication rows, so the pooled sepsis AUROC partly measures data availability. Fix in v8: `paper.sepsis_hospital_filter`
(default ON for the sepsis task only; coverage >= 0.8; `--set paper.sepsis_hospital_filter.enabled=false` reproduces the old numbers), the `sepsis_clean` feature set, and a corrected audit script.
(Correction: the first audit's "CABG" line was an artefact of treating unknown labels as negative and of assuming what `graftcount` encodes; the corrected script prints the raw value counts instead.)


## Round 3 / all-stays / valid-sepsis results (2026-10-12)

**Federated mortality, strict 24h (test n=19,802):** FedProx mu 0.01 -> 0.8402, 0.05 -> 0.8470, 0.1 -> 0.8506, 0.3 -> 0.8567, **1.0 -> 0.8611 [0.8521, 0.8688]** (val 0.8571, still creeping at round 60;
per-hospital calibrated 0.8630; central MLP 0.8684 -> gap 0.007). Federated LR: mu 0.3 -> 0.8510, 1.0 -> 0.8535 (central LR 0.8603). A large FedProx weight pins each local update to the global
model, so training approaches centralized optimisation at the price of more rounds -- report rounds / communication alongside accuracy.

**All-stays first-24h (test n=29,968), mu=0.1 only:** central MLP 0.8989, LightGBM 0.9096, LR 0.8892; federated MLP 0.8822 [0.8758, 0.8882] (0.8879 with per-hospital calibration), federated LR 0.8773.
mu=1.0 / 100 rounds on all stays is the next run (the strict-cohort gain from 0.1 -> 1.0 was +0.0105).

**Sepsis on label-valid hospitals (coverage >= 0.8: 151/207 hospitals, 108,425 of 131,695 strict-cohort stays; proxy rate 34.5%):** controlled LightGBM 0.8869 pooled / 0.8697 per-hospital / 0.8861 with
per-hospital calibration (the hospital-offset effect is gone); **clean LightGBM 0.8621 [0.8568, 0.8676], AUPRC 0.7604, per-hospital 0.8387**; clean LR 0.8460. 6h early warning: clean 0.8591 (per-hospital 0.8372).
Audit: coverage is bimodal (49 hospitals < 0.5 coverage have a proxy rate of exactly 0.000; 133 hospitals >= 0.95 have 0.345), so the threshold (0.5-0.95) is immaterial. `apachePredVar.graftcount` is the
constant 3 for 168,810 of ~171,000 rows, so it is effectively uninformative (the earlier "CABG" reading was wrong).
New: `scripts/paper_documented_sepsis_check.py` -- AUROC of the proxy-trained score for clinician-documented sepsis diagnoses (never a training target).
