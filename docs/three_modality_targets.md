# What AUROC / F1 / recall / accuracy are attainable on this cohort (honest feasibility note)

Requested targets: AUROC >= 0.88, F1 >= 0.65, recall >= 0.65, accuracy > 0.90 -- together, hospital-wise, on the strict 24 h mortality cohort (test prevalence 9.3%; an
all-negative prediction already scores 0.907 accuracy). They are not mutually consistent at this prevalence, and they sit above what the same data supports even without privacy
or hospital-wise constraints. Nothing in the code chases them: thresholds, checkpoints and hyper-parameters are chosen on training/validation data only, and test labels are read once.

**Equal-variance binormal approximation at 9.3% prevalence (best attainable F1 over thresholds):**

| AUROC | best F1 | recall at that point | precision | accuracy |
|---|---|---|---|---|
| 0.83 | 0.42 | 0.48 | 0.38 | 0.877 |
| 0.85 | 0.46 | 0.50 | 0.41 | 0.888 |
| 0.88 | 0.51 | 0.54 | 0.48 | 0.903 |
| 0.90 | 0.55 | 0.57 | 0.53 | 0.913 |
| 0.94 | 0.65 | 0.65 | 0.65 | 0.936 |

So F1 >= 0.65 together with recall >= 0.65 needs AUROC of about 0.94. Empirical anchors on the same cohort (paper track, strict landmark): central LightGBM on the full raw feature
set reaches AUROC 0.8796 with AUPRC 0.4894 (an F1 near 0.5); federated MLP 0.8617 with AUPRC 0.4444; federated GBDT 0.8746. The frozen fused embedding's pooled validation AUROC is 0.843-0.848
(mortality) and 0.787-0.793 (sepsis) with a linear head (`xattn_real_001/training_log.json`), i.e. the representation itself caps near 0.85 before any hospital-wise effect.

**FLICS' F1 (0.58-0.65) and accuracy (0.87-0.91) are not reachable by matching its model:** it has ~14% positives, a pooled 1,000-patient test set, its 3 / 10 largest hospitals and thresholds
tuned on the test set to maximise F1. Reproducing that protocol would mean tuning on test, which this project forbids.

**What legitimately moves the numbers** (in order of expected size): federated aggregation (paper track: +0.12 pooled AUROC over local-only); a richer shared representation (the encoders
currently cap at ~0.85); fewer degenerate per-hospital fits (LightGBM stabilisation, selection that cannot pick a near-initial checkpoint, validation-tuned hyper-parameters); and, for the
recall target, the secondary validation-selected operating point reported next to the F1-optimal one (it trades precision for recall, it cannot create ranking power).
