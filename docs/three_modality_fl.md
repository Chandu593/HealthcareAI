# Federated downstream learning on the frozen three-modality embeddings (FL Configs 1-4 and personalisation)

Status: implemented and sandbox-verified (synthetic upstream runs). **Not yet run on the real 199,108-patient / 207-hospital data.**

## What this is — and is not
The GRU / TCD / tabular encoders and the three-token cross-attention were trained **centrally** on pooled training patients, with supervised heads, and are frozen. Everything here is
**federated downstream learning on those frozen representations**. It is not end-to-end federated learning, the embeddings are in-sample for the encoders' training labels, and no claim is made
about the encoders' privacy. What is federated: the downstream LR / MLP / GCN / GBDT. What leaves a client: model parameters (or FedGBDT histograms), two class-count scalars, and per-class histograms
of validation logits. What never leaves a client: embeddings, labels, graphs, SMOTE nodes, scalers, thresholds.

## Clients, split and isolation
* Client = hospital × task. The permanent 70/15/15 split and the cohorts are loaded from the frozen cross-attention run (`hw_data.load_cohort`); nothing is rebuilt.
* `scope = all` (default): every hospital with >= 1 training patient contributes its training data to the federation. Hospitals that satisfy the Non-FL minimum-data rule (`min_samples`: train n >= 20,
  >= 3 positives and negatives, validation with both classes) form the **paired set** — the hospitals and test patients the Non-FL regime also evaluates. Other clients are evaluated and reported
  separately as `fl_only`. `scope = nonfl_matched` trains only the Non-FL hospitals (control that isolates federation from the larger client population).
* **Test labels are architecturally unreachable.** `FLClient` holds train/validation labels only (test class counts are not even stored). Only `fl_eval.py` reads test labels, after every round,
  hyper-parameter and threshold is fixed. Enforced by: no test-label attribute; a source scan of every training/tuning module; and the poison test (flip every test label ⇒ parameters, selected round,
  validation history, thresholds and predictions are bit-identical, only test metrics change).
* Preprocessing is hospital-local: each client's StandardScaler is fitted on its own training rows. Graphs (Graph A kNN+cosine threshold, Graph B MiniBatchKMeans) and training-only SMOTE are built per
  client by the same `hw_graph` functions as the Non-FL regime; validation/test patients attach inductively with directed train→query edges only, never to each other.
* Focal α: `global` (default; from the federated class counts, two scalars per client) or `local` (per-client, as in Non-FL).

## Algorithms
Parameter-averaging (LR, MLP, GCN only), p_i = n_i / Σ n (real training patients), K full-batch local steps per round, one common local optimiser (plain SGD) so the algorithms differ only in their
server / correction logic:

| algorithm | local | server |
|---|---|---|
| FedAvg | SGD | w ← w + Σ p_i (w_i − w) |
| FedProx | SGD + (μ/2)‖w − w_t‖² (refused when lr·μ ≥ 2: explicit-SGD instability) | as FedAvg |
| FedAdam | SGD | server Adam on D = Σ p_i (w_i − w): m, v moments, w ← w + η m /(√v + τ) (Reddi et al. 2021, no bias correction) |
| SCAFFOLD | SGD with gradient g − c_i + c; option II c_i⁺ = c_i − c + (w_t − w_i)/(Kη); stateful clients | as FedAvg; c ← c + Σ p_i (c_i⁺ − c_i) (sample-weighted objective) |

FedAvg with a local Adam optimiser is available (`federated.local.optimizer: adam`; FedAvg/FedProx only) as a control. **Trees are never parameter-averaged:** `gbdt_fed` is federated histogram
boosting (`src/paper/fed_gbdt.py`, imported unchanged): clients send per-bin gradient/Hessian histograms of their training rows, the server grows each tree on the aggregate. It uses the logistic loss with an
optional positive-class weight; **Non-FL LightGBM (focal) and Non-FL GBDT (standard loss) have no exact federated counterpart**, so FedGBDT is reported next to both.

## Federated validation (round selection, early stopping, global threshold)
After every round each client scores its own validation patients and sends a per-class histogram of the logits (default 3,000 bins over [−15, 15]) and its validation focal-loss sum. The server adds them.
The histogram AUROC treats same-bin positive/negative pairs as ties: |AUROC_hist − AUROC_exact| ≤ 0.5 × (tied-pair fraction), recorded every round as `val_auroc_tie_bound`; the exact pooled validation
AUROC at the selected round is stored as `val_auroc_exact_diag` (diagnostic, never used for selection). For FedGBDT the same histogram metric temporarily replaces `fed_gbdt._auroc` (restored in a
`finally`). Thresholds: each client's F1-maximising threshold on its own validation patients (identical to Non-FL); a client whose validation split lacks a class uses the global threshold from the
summed validation histogram (`threshold_source = global_federated_validation_histogram`) and has no recall-target point.

## Configs
1. `fl_config1` — fused (and `--embedding-set all`: gru, tcd, tabular) embeddings; LR / MLP × {FedAvg, FedProx, FedAdam, SCAFFOLD}; FedGBDT.
2. `fl_config2` — federated GCN on hospital-local Graph A / B.
3. `fl_config3` — Config 2 + hospital-local, training-only graph SMOTE (synthetic nodes live only inside the owning client's training graph; edges only from real training nodes).
4. `fl_config4` — stage 1: federated GCN (`--c4-gcn-algorithm`); stage 2: every client embeds its own patients with the final global GCN and LR / MLP / FedGBDT are trained federatedly on those
   hospital-local embeddings. As in Non-FL, train-node embeddings are in-sample for the GCN.
5. `fl_personalized` — FedPer and FedRep (shared body aggregated with FedAvg, hospital-specific heads that never leave the client; FedRep alternates head-only then body-only steps), and FedAvg followed by
   local fine-tuning of the head (epoch 0 = the global model is a candidate, so fine-tuning cannot end worse on the client's validation loss). Models: MLP, GCN (LR has no body/head split). Clients that fail the
   minimum-data rule use the sample-weighted mean head (`mean_head_fallback`) or the global model (`global_fallback`).

## Artifacts (`outputs/three_modality/federated/<run_id>/`, additive; Non-FL and cross-attention runs are only read)
`per_hospital.{json,csv}` (counts, threshold + source, metrics, confusion matrices, recall-target point, skip reasons, head source, runtime), `summary.json` (hospital-macro and sample-weighted aggregates for the
paired set and for all evaluated clients), `pooled_summary.json` (pooled metrics with patient-bootstrap CIs, hospital-bootstrap macro CIs), `classification_reports.{json,md}` +
`per_hospital_classification_reports.csv` (class-wise), `predictions.npz` (same layout as Non-FL), `training_logs/<variant>.json` (round history, selected round, stop reason, histogram bound, communication,
runtime), `checkpoints/<variant>/final.pt|final.pkl`, `manifest.json` (identity, hashes, split identity, regime statement), `run_state.json`.
Tuning: `outputs/three_modality/fl_tuning/<run_id>/{best_hparams.json,tuning_table.csv,manifest.json}`. Report: `outputs/three_modality/reports/fl_report_*/{report.md,comparison.csv,comparison.json}`.

## Checkpoint / resume
Round state (global parameters, server moments, SCAFFOLD controls, personalised heads, best snapshot, history) is written atomically every `federated.checkpoint.every_rounds` rounds and local randomness is seeded
per (round, client), so `--resume-existing-run --run-id <id>` reproduces the uninterrupted run (identity checked; completed runs are never resumed or overwritten). FedGBDT is a single call: a finished model is
reloaded, an interrupted boosting run restarts from round 0.

## Hospitals with an empty split (fixed in s10)
A hospital can have no patients in a split after the 70/15/15 split. Zero test patients: the hospital still trains (it is a client), is reported `skipped: no test patients`, and has no predictions.
Zero validation patients: it still trains and contributes to the federation, no validation histogram entry is sent, and it is evaluated with the global federated-histogram threshold
(`threshold_source = global_federated_validation_histogram`, no recall-target point). Before s10, zero test patients crashed Config 4's embedding step and zero validation patients silently dropped
the hospital under a misleading `graph construction failed` reason (also for LR / MLP). Now only the GCN family can drop a client (a genuine graph-construction failure, reported with its reason);
an LR / MLP construction error is raised.

## Known limitations (state them in any write-up)
* Centrally pretrained, supervised encoders: not end-to-end FL; embeddings in-sample for the encoders' labels.
* FedGBDT ≠ focal LightGBM / standard GBDT (loss differs). FedGBDT's boosting loop is not resumable mid-run.
* Pooled AUROC of a federated global model has one score scale across hospitals, whereas independent Non-FL hospital models do not: lead with hospital-macro.
* Histogram validation is approximate (bound recorded). FedProx with plain SGD needs lr·μ < 2.
* One tuned hyper-parameter set per family, shared by tasks and hospitals (disclosed); tuned on validation only (`fl_tune`).
* The sepsis label is a proxy; base-paper figures are not directly comparable.
