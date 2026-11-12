# Fed-THOR: results section (draft)

Primary protocol: **strict 24-hour landmark** (ICU stays lasting at least 24 h). Secondary protocol: **all stays, first-24-h window** (comparable with prior work, but it contains a short-stay effect).
All AUROCs are test-set values on a fixed per-hospital 70/15/15 split. "mean ± sd" is over 3 training seeds (same split); 95% CIs are bootstrap intervals over test patients (seed 42). Accuracy and F1 (Table 14) are test values at the decision threshold that maximises F1 on the validation set.
Reference numbers are taken from the base paper as reported. They carry no test-set CIs of ours (FLICS gives bootstrap CIs on a 1,000-patient test set), and cohorts, windows and protocols differ. Base paper: FLICS (2026, eICU only; 3 and 10 hospitals). Studies that use other databases are not compared.

## 1. Protocol in brief
- **Data:** eICU export with 13 tables, 207 hospitals. For the MLP, the 204 hospitals (205 on all stays) with at least two training patients act as federated clients; the federated GBDT uses every hospital with at least one training row (206 strict, 207 all stays).
- **Window and leakage control:** features use only events inside the first 24 h (clipped at ICU discharge); discharge time and length of stay are never inputs; vocabularies, scaling, calibration and thresholds are fitted on training/validation data only. APACHE-derived scores are excluded from the primary feature set; APACHE IVa predicted mortality is a benchmark only.
- **Federated training:** FedAvg (sample-size-weighted), all clients every round, FedProx proximal term μ = 1.0, learning rate 0.001, 2 local epochs, at most 100 rounds with early stopping (patience 15) on validation AUROC. No secure aggregation and no differential privacy.
- **Model selection:** every choice (FedProx weight, feature sets, cohort filter) was made on validation metrics.
- **Headline federated model:** the federated arm with the highest validation AUROC. That is the federated histogram GBDT on both protocols (3-seed mean validation AUROC 0.8698 against 0.8567 for the federated MLP on the strict cohort; 0.8984 against 0.8836 on all stays; on sepsis it is again the GBDT, 0.8631 against 0.8465 at 24 h and 0.8603 against 0.8422 at 6 h for the 1,200-round rerun, 0.8595 for the first 6 h run). The federated MLP stays in every table as the neural baseline, and all federated arms are reported.
- **Federated GBDT:** horizontal histogram boosting. Each client sends gradient and hessian histograms per tree level; the server sums them, so the trees equal those of a pooled histogram GBDT on the same bin edges (tested). Bin edges, base score and splits use training rows only; validation rows only choose the kept round and the winning trial. Up to four validation-selected trials on the first seed (strict winner: depth 8, learning rate 0.05, square-root class weighting; all-stays winner: the default depth 6, learning rate 0.08), re-used for the other two seeds. On sepsis the 151 label-valid hospitals are the clients. No differential privacy or secure aggregation.
- **Pooled-GBDT control (Table 13):** the same GBDT code trained on one client that holds every training row (arm `gbdt_central`), with the hyperparameters the federated search chose and no new search. It exists to measure what federation costs this model: the only difference from the federated GBDT is that the federated one merges bin edges from client quantile summaries. It was run in all six settings (strict, all stays, sepsis 24 h and 6 h, the 3 and the 10 largest hospitals), three seeds each.
- **FLICS-scale federations (Table 11):** for a like-for-like comparison with FLICS's setting, every model on the strict cohort was re-trained on only the 3 or the 10 largest hospitals by strict-cohort size (3 hospitals with at least 3,758 stays, 12,911 stays; 10 hospitals with at least 2,699 stays, 33,179 stays), with the same split manifest, features (no APACHE scores), federated settings and three seeds (`configs/paper_flicsstyle.yaml`; the GBDT trials are searched again on each federation, on validation only). Results: Tables 11a-11c, built by `reproduce/paired_flicsscale.py`.

## 2. Findings the numbers support
1. **Federated mortality performance.** The federated GBDT, the headline federated model, reaches **0.8746 ± 0.0005** [0.8662, 0.8818] on the strict cohort and **0.9066 ± 0.0009** [0.9020, 0.9125] on all stays. The federated MLP reaches **0.8617 ± 0.0006** [0.8521, 0.8688] and **0.8915 ± 0.0006** [0.8841, 0.8967]. On the same test patients the GBDT is higher by +0.0129 [+0.0088, +0.0170] (strict) and +0.0151 [+0.0119, +0.0182] (all stays); paired bootstrap over test patients, 2,000 resamples, mean of the three seed-wise differences.
2. **Against the base paper, FLICS (Table 3).** Under the strict protocol the federated GBDT (trained on all 206 hospitals) is **level** with FLICS's 3-hospital figure (0.87; our interval [0.8662, 0.8818] contains it, +0.005) and ahead of both 10-hospital figures (+0.045 against LR, +0.035 against LSTM); the federated MLP is behind the 3-hospital figure (-0.008). These compare reported numbers across different settings (Section 5), and FLICS's own reference interval is about ±0.035 because it scores 1,000 patients, so no modelling change of realistic size could turn 'level' into a resolvable 'ahead'. Scored the way FLICS scored its models, on stratified 1,000-patient test sets at about 14% mortality, our federated GBDT gets 0.874 ± 0.014 (95% range 0.845-0.901): the 0.005 margin over 0.87 is about 0.3 standard deviations of that noise, and 37% of such draws fall below 0.87. We therefore make no superiority claim against FLICS's numbers. The comparison that does resolve is method against method on identical patients (Table 10): the federated GBDT improves on a federated logistic regression (FLICS's model class) by +0.0207 [+0.0159, +0.0262] on all strict test patients, by +0.0218 [+0.0135, +0.0316] on those of the 10 largest hospitals and by +0.0161 [+0.0028, +0.0301] on those of the 3 largest (models trained on all hospitals). Table 11 repeats the comparison with every model trained on a federation of FLICS's own size (item 3). FLICS itself names federated gradient boosting as an open problem and as future work; the federated GBDT is that step. Our all-stays figures are not compared with FLICS.
3. **Like-for-like at FLICS scale (Table 11).** Re-trained on federations of FLICS's size, each of the four models compared scores lower on the same test patients than the same model trained on all hospitals (Table 11c), and the ordering of methods holds where the data can resolve it. On the 10 largest hospitals (test n = 5,002, 483 events) the federated GBDT reaches 0.8662 ± 0.0017 [0.8524, 0.8835] and improves on the federated logistic regression by +0.0147 [+0.0053, +0.0239] and on the federated MLP by +0.0140 [+0.0045, +0.0233], positive in all three seeds. On the 3 largest (test n = 1,936, 177 events) it reaches 0.8635 ± 0.0024 [0.8342, 0.8836] with differences of +0.0050 [-0.0116, +0.0228] and +0.0092 [-0.0085, +0.0268]: positive in all three seeds, but not resolved by 177 events. The federated logistic regression, FLICS's model class, scores 0.8523 ± 0.0022 on the 3 hospitals and 0.8524 ± 0.0008 on the 10 (FLICS reports 0.87 and 0.83; our seed-42 interval contains 0.87 at 3 hospitals and lies above 0.83 at 10). Trained on all hospitals, the same model scores 0.8702 on the 3 largest hospitals' test patients (Table 10) against 0.8553 when trained on those three, so the 0.870 in Table 10 reflects training on 204 hospitals, not FLICS's setting. Against the FLICS references the GBDT is level at 3 hospitals (-0.006) and above FLICS's 0.84 at 10 hospitals by our interval (+0.026; Table 11a); FLICS's interval is up to ±0.035, so we still make no superiority claim. Federating costs nothing resolvable for the MLP or the LR against their central versions (federated minus central MLP +0.0033 [-0.0075, +0.0149] at 3 and -0.0015 [-0.0096, +0.0063] at 10, the latter slightly below in all three seeds) and a small amount for the GBDT against pooled LightGBM (-0.0140 [-0.0242, -0.0028] at 3 hospitals, -0.0038 [-0.0091, +0.0015] at 10, negative in all three seeds); the pooled-GBDT control (Table 13, item 5) splits that gap: at 3 hospitals federated minus pooled GBDT is -0.0102 [-0.0187, -0.0017] and pooled GBDT minus pooled LightGBM is -0.0037 [-0.0135, +0.0058] (seed 42, paired); at 10 hospitals -0.0002 [-0.0054, +0.0052] and -0.0036 [-0.0092, +0.0016]. At 10 hospitals the gap to LightGBM is therefore not a federation effect; at 3 hospitals most of it is the federated-versus-pooled difference, which across three seeds is about the size of the training-seed spread (item 5). Against local-only MLPs the federated MLP gains +0.0272 [+0.0013, +0.0520] at 3 and +0.0829 [+0.0632, +0.1022] at 10, less than across 204 clients because the largest hospitals hold the most data each. Training on all hospitals instead of only the federation raises the same patients' AUROC by +0.0260 [+0.0100, +0.0419] (GBDT), +0.0203 (MLP), +0.0149 (LR) and +0.0152 (pooled LightGBM) at 3 hospitals, and by +0.0110 [+0.0033, +0.0187], +0.0082, +0.0038 (interval includes 0) and +0.0082 at 10; these intervals reflect test-patient sampling only. Scored against FLICS's own label (Table 12) the same predictions reach 0.8890 [0.8628, 0.9134] for the 3-hospital LR and 0.8718 [0.8495, 0.8895] for the 10-hospital LR; the gain from the label change is +0.0337 [+0.0069, +0.0623] and +0.0190 [+0.0030, +0.0362], so the label alone can account for a difference of the size between our 3-hospital LR (0.855) and FLICS's reported 0.87; this is not a superiority claim.
4. **Federation versus isolation.** Local-only MLPs reach 0.742 ± 0.006 pooled AUROC on the strict cohort, so federation adds +0.120 on that measure. Pooled AUROC penalises local-only models for scores that are not comparable across hospitals, so we also give two within-hospital views: per-hospital macro AUROC 0.7907 (local-only) versus 0.8573 (federated), +0.067; and with per-hospital intercept calibration 0.7675 versus 0.8640, about +0.10. We state the gain as **+0.07 to +0.12 AUROC depending on the measure**.
5. **Federation versus pooling.** For the MLP the federation gap is about 0.007 (strict) and 0.008 (all stays): central MLP 0.8691 and 0.8994. Central LightGBM is higher (0.8796 and 0.9092) but needs pooled raw data. The federated GBDT is below pooled LightGBM by 0.0050 [0.0031, 0.0067] (strict) and 0.0026 [0.0016, 0.0035] (all stays), paired, so federating trees costs little accuracy; it closes about 72% (strict) and 85% (all stays) of the gap between the federated MLP and pooled LightGBM. Per-hospital macro AUROC tells the same story: 0.8651 (GBDT), 0.8573 (MLP), 0.8708 (pooled LightGBM) on strict; 0.8985, 0.8881 and 0.9011 on all stays. **The pooled-GBDT control (Table 13) shows that the remaining gap to LightGBM is not the price of federation.** The same GBDT code trained on one client with every training row, with the federated winner's hyperparameters, reaches 0.8746 ± 0.0008 (strict) and 0.9063 ± 0.0003 (all stays); federated minus pooled is +0.0004 [-0.0016, +0.0024] and +0.0006 [-0.0009, +0.0019] (seed 42, paired), and the three seed-wise differences average -0.0000 and +0.0003. The same holds on sepsis (24 h: -0.0002 [-0.0018, +0.0012]; 6 h: +0.0002 [-0.0009, +0.0012]) and on the 10 largest hospitals (-0.0002 [-0.0054, +0.0052]; the federated model is below the pooled one in each of the three seeds, by 0.0000 to 0.0024). The pooled GBDT is itself below pooled LightGBM (seed 42, paired: -0.0060 [-0.0085, -0.0038] strict, -0.0024 [-0.0038, -0.0009] all stays, -0.0016 [-0.0032, -0.0001] and -0.0019 [-0.0031, -0.0006] on sepsis), so the gap to LightGBM belongs to our implementation and search, not to federation; we have not isolated which part (the four-trial search against LightGBM's wider one is a candidate, untested). The exception is the 3-hospital federation: -0.0102 [-0.0187, -0.0017] at seed 42, but the seed-wise differences are -0.0102, +0.0020 and -0.0087 (mean -0.0056), the pooled GBDT's three seeds range from 0.8622 to 0.8747, and the test set has 177 events, so that gap is about the size of the seed spread and we neither claim nor rule out a federation cost there. The price is communication (Table 9): roughly 40-50 times the total uplink of FedAvg for the MLP on strict (653-730 GB against 15-17 GB per model; 25-55 times on all stays), and 97 minutes (strict) and 7.4 hours (all stays) of wall-clock for the search plus three seeds on one machine.
6. **What made federation work.** The FedProx weight (Figure 1, Table 6): μ = 0 gives 0.8341, 0.01 gives 0.8402, 0.1 gives 0.8506, 1.0 gives 0.8611, 3 gives 0.8620, 10 gives 0.8551. Validation was flat between 1 and 3, so μ = 1.0 was chosen. Cost: the best validation round was 49-56 of 64-71 rounds run, with 204 clients per round.
7. **Versus APACHE IVa on the same 15,151 test patients:** federated MLP +0.035 [0.027, 0.044]; central LightGBM +0.051 [0.043, 0.059].
8. **Heterogeneity (Table 7, federated MLP and central LightGBM).** Best in patients under 45 (0.932 for the MLP), weakest at 80+ (0.795). Per-hospital AUROC over the 102 hospitals with enough events (strict, seed 42): federated GBDT median 0.879 (IQR 0.830-0.914, range 0.63-0.99), federated MLP 0.867 (0.826-0.898, range 0.46-0.98), central LightGBM 0.881 (0.839-0.917). The GBDT raises the worst hospital from 0.46 to 0.63.
9. **Accuracy and F1 in FLICS's format (Table 14).** Table 14 reports accuracy, AUROC and F1 for the federated and central models of each setting, with the best federated result per metric in bold and the best central result underlined, as in FLICS's Tables IV and V. The federated GBDT is the best federated model on all three metrics in all six settings (strict: accuracy 0.8888 ± 0.0052, AUROC 0.8746, F1 0.4722 ± 0.0011; all stays: 0.9146 ± 0.0090, 0.9066, 0.5433 ± 0.0031). Pooled LightGBM is the best central model on AUROC in all six; the pooled GBDT takes the accuracy mark in the strict and the 3-hospital settings and the F1 mark at 10 hospitals. The marks follow point estimates: accuracy differences between the leading models are mostly inside the seed sd (strict: 0.8888 ± 0.0052 against 0.8858 ± 0.0046), so we read the AUROC differences, which paired intervals resolve (items 1 and 5), and treat accuracy and F1 rankings as descriptive. **Our F1 and accuracy are below FLICS's reported values, and Table 14 does not support a comparison.** FLICS reports accuracy 0.88-0.91 and F1 0.59-0.65; at its scales our federated GBDT scores accuracy 0.8833 ± 0.0080 and F1 0.4386 ± 0.0041 (3 hospitals) and 0.8864 ± 0.0043 and 0.4704 ± 0.0062 (10 hospitals), with an AUROC level with FLICS's. Both metrics depend on prevalence (9.1-9.7% in-hospital death in our test sets against 12.5-17.0% ICU-discharge death in FLICS: at the same sensitivity and specificity a lower prevalence gives lower precision and so a lower F1), on the label (Table 12) and on the threshold. FLICS tunes its threshold on its test set to maximise F1; ours is fixed on validation, and a threshold tuned on the test set cannot score a lower F1 there than a validation-chosen one. On the strict cohort a classifier that predicts nobody positive scores accuracy 0.908, above every accuracy in Table 14a, because an F1-maximising threshold trades accuracy for recall (0.909 and 0.903 at the 3- and 10-hospital test sets). We draw no conclusion about accuracy or F1 against FLICS. FLICS's centralized XGBoost (AUC 0.90 at 3 hospitals, 0.88 at 10) is nominally above our pooled LightGBM (0.8740 [0.8504, 0.8948] and 0.8704 [0.8557, 0.8873]) and pooled GBDT (0.8692 and 0.8670) at the same scales; the intervals overlap widely and the labels differ, so we make no claim in either direction.
10. **Neutral or negative results to report.**
   - Graph head: no benefit. Differences are at most 0.0001 AUROC across seeds; the best validation round was 1-2; neighbour label agreement 0.90 versus 0.834 expected by chance (the embedding already encodes the label signal).
   - GRU lab-sequence branch: +0.0015 AUROC centrally (one seed, inside the CI).
   - Class-balancing by resampling: no gain (LightGBM 0.8769 and 0.8786 versus 0.8799).
   - FedProx weights above 3 hurt.
   - Validation search for the federated GBDT: no test benefit. On the strict cohort (seed 42) the default parameters scored 0.8748 against 0.8742 for the searched winner, and the four trials differed by at most 0.003 in validation AUROC; on all stays the winner was the default configuration.

## 3. Sepsis (secondary task, proxy label)
- **The label is a proxy** (antibiotic and SOFA-like score of at least 2). It agrees with documented sepsis diagnoses at only 29% precision and 50% recall.
- **Label validity filter.** The antibiotic half cannot fire in hospitals without medication records: 49 of 188 hospitals have a proxy rate of exactly 0. The task is therefore restricted to the 151 hospitals with at least 80% medication-record coverage (coverage is bimodal; thresholds from 0.5 to 0.95 keep 133-139 hospitals).
- **Ladder (Figure 3).** Allowing the label's own ingredients gives AUROC 0.993; removing them gives 0.910; progressively removing blood gases, infection and organ-failure diagnoses, hospital covariates and labs gives 0.898, 0.879 and 0.801. On label-valid hospitals the clean feature set reaches 0.862.
- **Results (Tables 4a, 4b).** Federated MLP 0.8418 ± 0.0004 (24 h) and 0.8389 ± 0.0001 (6 h); central MLP 0.8511 and 0.8505; LightGBM 0.8623 and 0.8589. Per-hospital macro AUROC: 0.8315 (federated), 0.8298 (central MLP), 0.8387 (LightGBM) at 24 h. The 6 h window performs almost as well as 24 h. The federated GBDT, the headline model on sepsis too, reaches **0.8599 ± 0.0005** (24 h) and **0.8574 ± 0.0000** (6 h, the 1,200-round rerun; the first 6 h run reached 0.8566 ± 0.0002). On the same test patients that is +0.0181 [+0.0150, +0.0211] and +0.0184 [+0.0153, +0.0214] above the federated MLP, and 0.0024 [0.0012, 0.0035] and 0.0017 [0.0004, 0.0030] below pooled LightGBM (paired bootstrap, 2,000 resamples); it closes about 88% (24 h) and 92% (6 h) of the gap between the federated MLP and pooled LightGBM. The same GBDT code trained pooled scores 0.8600 ± 0.0006 and 0.8571 ± 0.0002, so federating it costs -0.0002 [-0.0018, +0.0012] and +0.0002 [-0.0009, +0.0012] AUROC (Table 13): nothing resolvable. Per-hospital macro AUROC: 0.8367 and 0.8344 (federated MLP 0.8313 and 0.8277; pooled LightGBM 0.8385 and 0.8366). Cost (Table 9): about 7 MB per client per round at 24 h and 9 MB at 6 h, 411-642 GB and 1,082-1,390 GB per model across the 151 clients, and 102 and 208 minutes of wall-clock for the search plus three seeds.
- **External check (Table 8).** Proxy-trained scores rank clinician-documented sepsis with AUROC 0.77-0.78 (never a target); documented-sepsis rate rises from about 4% in the lowest risk quintile to 34-40% in the highest. The federated GBDT (24 h, seed 42) scores 0.7683 [0.7585, 0.7787], in line with the federated MLP (0.7669) and pooled LightGBM (0.7721): its +0.018 gain on the proxy label shows no measurable gain on documented sepsis (the intervals overlap; the comparison is not paired). Its documented-sepsis rates by risk quintile are 5.2%, 4.1%, 9.7%, 18.6% and 40.0%, not monotone at the bottom. At 6 h (seed 42, the 1,200-round run) it scores 0.7796 [0.7717, 0.7873] against 0.7769 for the federated MLP and 0.7798 for pooled LightGBM, so again no measurable gain on documented sepsis; its quintile rates are 4.5%, 1.9%, 8.5%, 17.7% and 35.8%, again not monotone at the bottom.
- **Related published work (Table 4c).** Fuzail et al. (2026) report AUC 0.96-1.00 for tree and neighbour models and federated accuracy up to 0.985 for sepsis on the PhysioNet/CinC 2019 data. We do not compare against it. The data are not eICU, the target is the PhysioNet sepsis label rather than a proxy built from eICU tables, and the evaluation is not leakage-safe: the confusion-matrix counts indicate that SMOTE was applied before the 80/20 split, so the test set is balanced (164,818 positives and 164,090 negatives) although the whole dataset holds only 16,592 real positives; under a random split about 98% of those test positives would be synthetic. The paper does not state a patient-level split, its federation is five simulated clients cut from one dataset, and its federated logistic regression labels almost every case as sepsis (accuracy 0.5011).
- **Claim boundary.** There is no valid published sepsis figure to compare against. Claim internal validity, early-warning performance and the documented-sepsis check, not diagnostic accuracy.

## 4. Claims we do not make
- Any superiority over FLICS's reported numbers: its settings (3 or 10 hospitals, ICU-discharge mortality, whole-stay summaries, a 1,000-patient test set) differ from ours and its reference interval is about ±0.035.
- Superiority over the 3-hospital FLICS figure (0.87) under the strict protocol: the federated GBDT is level with it (CI [0.8662, 0.8818] contains 0.87) and the federated MLP is behind it; trained on the 3-hospital federation itself (Table 11a) the federated GBDT, MLP and LR are all level with it.
- That the federated GBDT beats the federated LR or MLP in a 3-hospital federation: Table 11b gives positive but unresolved differences (177 test events); the advantage is resolved with 10 hospitals and with all hospitals.
- That our federated logistic regression reproduces FLICS's 3-hospital 0.87: trained on those three hospitals it scores 0.8523 (its interval contains 0.87).
- That our figures are better than FLICS's: the labels differ, and the label change alone raises our own federated LR's AUROC by +0.019 to +0.034 at FLICS scale, with paired intervals that exclude zero (Table 12).
- Any conclusion from accuracy or F1 against FLICS (Table 14): prevalence, label and threshold procedure differ, and our F1 at FLICS's scales (0.44-0.47) is below its reported 0.59-0.65.
- That our pooled models match or exceed FLICS's centralized XGBoost: they are nominally below its AUC at both FLICS scales (0.874 and 0.869 against 0.90 at 3 hospitals; 0.870 and 0.867 against 0.88 at 10), with widely overlapping intervals.
- That federation costs the GBDT nothing at every scale: Table 13 shows no resolvable cost in five settings, but -0.0102 [-0.0187, -0.0017] at seed 42 (mean over three seeds -0.0056) for the 3 largest hospitals, about the size of the seed spread; we neither claim nor rule out a cost there.
- Any comparison with papers that use other databases or mixed databases.
- Any comparison with the sepsis results of Fuzail et al. (different data, label and evaluation; Section 3).
- Any privacy guarantee (no differential privacy or secure aggregation), or communication efficiency for the federated GBDT (Table 9).
- Any benefit from the graph component.
- That the sepsis proxy is sepsis, or that the federated GBDT's gain on the proxy label means better detection of documented sepsis (Table 8 shows no measurable gain).
- Generalisation beyond one database and one fixed split: the seed sd reflects training randomness only.

## 5. Threats to validity
- **Selection on a visible test set.** Selection used validation, but test numbers for each FedProx weight were visible; Table 6 shows a plateau (0.857-0.862 for μ = 0.3-3), not a tuned peak.
- **Bootstrap resolution.** Paired bootstrap p-values bottom out at 0.002 (500 resamples); quote the confidence intervals instead.
- **Reference comparability.** FLICS tunes its decision threshold on its test set, uses a 1,000-patient test set, and reports hospital sizes (for example 8,000 stays) larger than any single eICU hospital contributes, so its partitions may be simulated. Our cohort has 204 hospital clients, a 24-hour window and a pooled test set of 19,802 (strict) or 29,968 (all stays) stays. None of these comparisons is like-for-like, and the reference values have no test-set CIs.
- **Round limit on sepsis 6 h.** The first 6 h federated GBDT run reached the 600-round limit in all three seeds (best rounds 600, 592 and 596). We re-ran it with a 1,200-round limit (`fedgbdt_sepsis6h_mr1200`; the same four-trial search chose the same configuration), and early stopping then ended every seed before the limit (best rounds 849, 945 and 732). Validation AUROC rose in all three seeds (3-seed mean 0.8603 against 0.8595), so by our validation rule the rerun is the reported 6 h GBDT, and the first run stays as a row in Table 4b. The test difference is +0.0006 [+0.0001, +0.0011], positive in all three seeds (`reproduce/paired_sepsis6h.py`). The rerun was prompted by seeing the cap in the first run's logs, after test results were visible; it cost about 208 minutes against 147, one seed taking about three times as long as the other two (probably machine load). Every other GBDT run (strict, all stays, sepsis 24 h) early-stopped before the limit.
- **FLICS settings and reference reliability.** FLICS predicts mortality at ICU discharge from whole-stay summaries (including the last observed value) of vital signs and labs, on 3 or 10 hospitals, and scores 1,000 patients; we predict in-hospital mortality from first-24 h features of 206 hospitals and score 19,802. Its hospital tables do not match real eICU hospitals: they list six hospitals of at least 5,000 stays and one of 8,000, whereas in this cohort three hospitals exceed 5,000 stays and the largest has 7,058; its ten hospitals total 53,500 stays against 47,211 for the ten largest real ones; and its mortality rates (12.5-17.0% at ICU discharge) exceed the 9.0% in-hospital mortality of the whole cohort (only 6 of the 43 real hospitals with at least 1,500 stays reach 12.5% even for in-hospital mortality, and an ICU-discharge rate is never higher than the in-hospital rate for the same patients). The paper does not say how its hospitals were formed, so we cannot tell whether they are simulated; its settings cannot be reproduced and its numbers are context, not a target.
- **Label check on FLICS's target (Table 12).** `scripts/paper_label_variant_check.py` scores the same predictions against in-hospital death (our target) and ICU-discharge death (FLICS's label). The label change raises the AUROC in all five settings we checked, and every paired interval excludes zero: +0.0127 [+0.0055, +0.0203] for the federated GBDT on all strict test patients (n = 19,802); +0.0190 [+0.0030, +0.0362] and +0.0337 [+0.0069, +0.0623] for the federated logistic regression on the 10 and the 3 largest hospitals; and +0.0214 [+0.0062, +0.0365] and +0.0343 [+0.0089, +0.0617] for the GBDT. The effect is larger where more of the hospital deaths occur after ICU discharge (44%, 48% and 56% in the three test sets); that is three settings, not a test. For the 3-hospital federation the point estimate of the label effect for the logistic regression (+0.034) is more than twice the gap between our seed-42 estimate (0.855) and FLICS's reported 0.87 (0.015), although the lower end of its interval (+0.007) is below that gap. These are single-seed estimates, the models were trained on in-hospital death, and no model was trained on ICU-discharge death; nothing here is applied as a correction. Our ICU-discharge mortality (4.0-5.2%) is also far below FLICS's 12.5-17.0%.
- **FLICS-scale federations are small.** The 3-hospital test set has 1,936 patients and 177 events, so its intervals are wide (about ±0.025 around 0.86) and most paired differences there are unresolved. The seed sd (0.0022-0.0156 at 3 hospitals) reflects training randomness only and the bootstrap intervals reflect test sampling only; the training-pool contrast (Table 11c) compares different training runs on identical patients. Hospital-level inputs (region, teaching status, bed-size category, log hospital size) identify the site when only 3 or 10 hospitals are present; every arm sees the same inputs, so the comparison between arms is on equal terms, but absolute levels are not comparable with a model without them.
- **Reference choice.** FLICS was designated the base paper after the comparison with published figures was known, so the comparison was not pre-specified; for that reason we make no claim of superiority over it.
- **Federated GBDT cost and boundary.** It sends gradient and hessian histograms at every tree level: about 11 MB per client per round on strict and 653-730 GB per model across clients (sparse upper bound), against 1.2 MB and 15-17 GB for the MLP (Table 9), and histograms disclose more than a weight average. The validation AUROC used for early stopping is pooled at the server, as for the federated MLP. No privacy guarantee is claimed.
- **Accuracy and F1 (Table 14).** Both are computed at the threshold that maximises F1 on the validation set, so neither is tuned on the test set; FLICS tunes its threshold on its test set, which favours its F1, and its prevalence, label and cohort differ from ours. The best-model marks follow point estimates, and accuracy differences are often inside the seed sd. Only AUROC carries an interval (seed-42 bootstrap); accuracy and F1 are mean ± sd over seeds, and central logistic regression has one seed. Local-only models are not in Table 14.
- **Pooled-GBDT control (Table 13).** The pooled GBDT reuses the hyperparameters that the federated search chose, so any advantage from tuning would favour the federated arm, not the pooled one; the two differ only in how bin edges are formed. The intervals in Table 13 pair seed 42 only (test-patient sampling); the seed counts use all three seeds. Training randomness is largest at 3 hospitals (pooled GBDT seeds 0.8622 to 0.8747), so the 3-hospital difference is not settled by its interval alone.
- **Short-stay effect.** All-stays AUROC is inflated by short stays (LightGBM 0.953 in stays under 24 h versus 0.882 in longer ones); lead with the strict protocol.

## 6. Tables
**Table 1. Mortality, strict 24h landmark cohort (131,695 stays; 204 hospital clients for the MLP, 206 for the GBDT; test n = 19,802).**

| Model | AUROC (mean ± sd, 3 seeds) | 95% CI (seed 42) | AUPRC | Brier | ECE | per-hosp. macro AUROC | AUROC, per-hosp. calibration |
|---|---|---|---|---|---|---|---|
| Local-only MLP (no federation) | 0.7421 ± 0.0055 | [0.7207, 0.7479] | 0.2817 | 0.0756 | 0.0111 | 0.7907 | 0.7675 |
| Federated logistic regression | 0.8535 ± 0.0001 | [0.8445, 0.8616] | 0.4282 | 0.0663 | 0.0094 | 0.8523 | 0.8607 |
| Federated MLP (FedAvg + FedProx) | 0.8617 ± 0.0006 | [0.8521, 0.8688] | 0.4444 | 0.0654 | 0.0089 | 0.8573 | 0.8640 |
| Federated MLP + graph head, no edges | 0.8618 ± 0.0005 | [0.8522, 0.8690] | 0.4450 | 0.0650 | 0.0056 | 0.8573 | 0.8659 |
| Federated MLP + graph A (kNN + threshold) | 0.8617 ± 0.0007 | [0.8518, 0.8688] | 0.4441 | 0.0653 | 0.0088 | 0.8572 | 0.8643 |
| Federated MLP + graph B (cluster-restricted) | 0.8618 ± 0.0006 | [0.8521, 0.8688] | 0.4443 | 0.0653 | 0.0100 | 0.8575 | 0.8644 |
| Federated GBDT (histogram boosting) | 0.8746 ± 0.0005 | [0.8662, 0.8818] | 0.4781 | 0.0627 | 0.0050 | 0.8651 | 0.8750 |
| Central logistic regression | 0.8603 ± 0.0000 | [0.8521, 0.8679] | 0.4507 | 0.0648 | 0.0045 | 0.8524 | 0.8612 |
| Central MLP | 0.8691 ± 0.0005 | [0.8597, 0.8760] | 0.4623 | 0.0638 | 0.0036 | 0.8621 | 0.8712 |
| Central LightGBM | 0.8796 ± 0.0002 | [0.8721, 0.8874] | 0.4894 | 0.0619 | 0.0059 | 0.8708 | 0.8790 |
| Central GBDT (same code, pooled) | 0.8746 ± 0.0008 | [0.8661, 0.8816] | 0.4790 | 0.0627 | 0.0056 | 0.8657 | 0.8748 |
| APACHE IVa predicted mortality (benchmark, n=15,151) | 0.8309 | [0.8206, 0.8415] | 0.3767 | – | – | – | – |

**Table 2. Mortality, all stays with the first-24h window (199,108 stays; test n = 29,968). Includes the short-stay effect.**

| Model | AUROC (mean ± sd, 3 seeds) | 95% CI (seed 42) | AUPRC | Brier | ECE | per-hosp. macro AUROC | AUROC, per-hosp. calibration |
|---|---|---|---|---|---|---|---|
| Federated logistic regression | 0.8813 ± 0.0001 | [0.8751, 0.8870] | 0.5201 | 0.0593 | 0.0107 | 0.8799 | 0.8887 |
| Federated MLP (FedAvg + FedProx) | 0.8915 ± 0.0006 | [0.8841, 0.8967] | 0.5531 | 0.0572 | 0.0086 | 0.8881 | 0.8951 |
| Federated GBDT (histogram boosting) | 0.9066 ± 0.0009 | [0.9020, 0.9125] | 0.5926 | 0.0541 | 0.0055 | 0.8985 | 0.9071 |
| Central logistic regression | 0.8878 ± 0.0000 | [0.8819, 0.8931] | 0.5466 | 0.0574 | 0.0071 | 0.8799 | 0.8897 |
| Central MLP | 0.8994 ± 0.0010 | [0.8930, 0.9038] | 0.5733 | 0.0554 | 0.0050 | 0.8932 | 0.9017 |
| Central LightGBM | 0.9092 ± 0.0002 | [0.9039, 0.9141] | 0.6037 | 0.0532 | 0.0041 | 0.9011 | 0.9094 |
| Central GBDT (same code, pooled) | 0.9063 ± 0.0003 | [0.9013, 0.9122] | 0.5936 | 0.0540 | 0.0060 | 0.8983 | 0.9068 |
| APACHE IVa predicted mortality (benchmark, n=20,503) | 0.8642 | [0.8558, 0.8728] | 0.4518 | – | – | – | – |

**Table 3. Base paper: FLICS (2026, eICU only). 'ahead' = our 95% CI lies above the reference; 'level' = it contains it; 'behind' = below. This is a comparison of reported numbers across different settings, not a replication: FLICS uses 3 or 10 hospitals, ICU-discharge mortality, whole-stay summaries and a 1,000-patient test set (reference CI up to ±0.035); ours is the strict 24 h cohort of 206 hospitals with hospital mortality (test n = 19,802). Our all-stays figures are not compared with FLICS (different setting and a short-stay effect). Tables 10 and 11 give same-patient comparisons (Table 11: models trained on federations of FLICS's size); Table 12 measures what FLICS's label alone does to our AUROC.**

| Reference (as reported) | Setting | AUROC | ours, strict: federated GBDT 0.8746 | ours, strict: federated MLP 0.8617 |
|---|---|---|---|---|
| FLICS, FedAvg LR / LSTM | 3 hospitals, 5,000 stays, 1,000-patient test set (CI 0.83-0.90) | 0.87 | level (+0.005) | behind (-0.008) |
| FLICS, FedAvg LR | 10 hospitals, 53,500 stays (CI 0.79-0.87) | 0.83 | ahead (+0.045) | ahead (+0.032) |
| FLICS, FedAvg LSTM | 10 hospitals, 53,500 stays (CI 0.79-0.88) | 0.84 | ahead (+0.035) | ahead (+0.022) |

**Table 10. Paired method comparisons on identical strict-cohort test patients (seed 42; 1,000 paired bootstrap resamples; from reproduce/paired_subsets.py). The models were trained on all hospitals and are only evaluated on the 10 and 3 largest hospitals by cohort size: this is not a replication of FLICS's federations (Table 11 trains on federations of FLICS's size).**

| Test patients | n (events) | AUROC: federated GBDT | AUROC: federated LR (FLICS's model class) | GBDT - federated LR [95% CI] | GBDT - federated MLP [95% CI] | GBDT - pooled LightGBM [95% CI] |
|---|---|---|---|---|---|---|
| all hospitals | 19,802 (1817) | 0.8742 | 0.8535 | +0.0207 [+0.0159, +0.0262] | +0.0131 [+0.0083, +0.0176] | -0.0057 [-0.0079, -0.0033] |
| 10 largest hospitals | 5,002 (483) | 0.8785 | 0.8566 | +0.0218 [+0.0135, +0.0316] | +0.0168 [+0.0082, +0.0259] | -0.0010 [-0.0055, +0.0034] |
| 3 largest hospitals | 1,936 (177) | 0.8863 | 0.8702 | +0.0161 [+0.0028, +0.0301] | +0.0149 [+0.0004, +0.0306] | -0.0032 [-0.0109, +0.0045] |

**Table 11a. FLICS-scale federations: every model re-trained on only the 3 or the 10 largest hospitals by strict-cohort size (configs/paper_flicsstyle.yaml; 3 largest: 3 hospitals, 12,911 stays, train/val/test = 9,032/1,943/1,936 and 10 largest: 10 hospitals, 33,179 stays, train/val/test = 23,187/4,990/5,002). Strict cohort, hospital mortality, first-24 h features without APACHE scores. 'ahead' = our seed-42 interval lies above the reference, 'level' = it contains it, 'behind' = below (as in Table 3); FLICS's own interval is up to ±0.035, so these labels are context, not claims. Central logistic regression has one seed. Hospital-level inputs (region, teaching status, bed-size category, log hospital size) identify the site in such small federations; they are identical for every arm.**

| Federation | Model | AUROC (mean ± sd, 3 seeds) | 95% CI (seed 42) | AUPRC | per-hosp. macro AUROC | vs FLICS, as reported |
|---|---|---|---|---|---|---|
| 3 largest hospitals | Federated GBDT | 0.8635 ± 0.0024 | [0.8342, 0.8836] | 0.4115 | 0.8590 | level (-0.006) vs 0.87 (LR / LSTM) |
| 3 largest hospitals | Federated MLP | 0.8490 ± 0.0062 | [0.8181, 0.8787] | 0.4075 | 0.8461 | level (-0.021) vs 0.87 (LR / LSTM) |
| 3 largest hospitals | Federated LR (FLICS's model class) | 0.8523 ± 0.0022 | [0.8274, 0.8812] | 0.3856 | 0.8490 | level (-0.018) vs 0.87 (LR / LSTM) |
| 3 largest hospitals | Pooled LightGBM | 0.8740 ± 0.0007 | [0.8504, 0.8948] | 0.4280 | 0.8698 | – |
| 3 largest hospitals | Central MLP | 0.8459 ± 0.0032 | [0.8188, 0.8764] | 0.3947 | 0.8432 | – |
| 3 largest hospitals | Central LR | 0.8477 (1 seed) | [0.8180, 0.8751] | 0.3880 | 0.8449 | – |
| 3 largest hospitals | Local-only MLP | 0.8078 ± 0.0156 | [0.7897, 0.8505] | 0.3367 | 0.8260 | – |
| 10 largest hospitals | Federated GBDT | 0.8662 ± 0.0017 | [0.8524, 0.8835] | 0.4559 | 0.8628 | ahead (+0.036) vs 0.83 (LR); ahead (+0.026) vs 0.84 (LSTM) |
| 10 largest hospitals | Federated MLP | 0.8544 ± 0.0015 | [0.8360, 0.8693] | 0.4272 | 0.8523 | ahead (+0.024) vs 0.83 (LR); level (+0.014) vs 0.84 (LSTM) |
| 10 largest hospitals | Federated LR (FLICS's model class) | 0.8524 ± 0.0008 | [0.8342, 0.8689] | 0.4160 | 0.8495 | ahead (+0.022) vs 0.83 (LR); level (+0.012) vs 0.84 (LSTM) |
| 10 largest hospitals | Pooled LightGBM | 0.8704 ± 0.0009 | [0.8557, 0.8873] | 0.4667 | 0.8666 | – |
| 10 largest hospitals | Central MLP | 0.8572 ± 0.0017 | [0.8379, 0.8707] | 0.4217 | 0.8547 | – |
| 10 largest hospitals | Central LR | 0.8506 (1 seed) | [0.8317, 0.8681] | 0.4156 | 0.8474 | – |
| 10 largest hospitals | Local-only MLP | 0.7750 ± 0.0077 | [0.7476, 0.7930] | 0.3274 | 0.8039 | – |

**Table 11b. Paired AUROC differences on identical test patients inside each FLICS-scale federation (seed 42; 2,000 paired bootstrap resamples over test patients; test n = 1,936 (177 events) and 5,002 (483 events)). 'seeds with A > B' counts the three training seeds whose point difference on the same patients is positive ('–' = one of the two arms has a single seed).**

| Comparison (A - B) | 3 largest: A - B [95% CI] | 3 largest: seeds with A > B | 10 largest: A - B [95% CI] | 10 largest: seeds with A > B |
|---|---|---|---|---|
| Federated GBDT - Federated LR (FLICS's model class) | +0.0050 [-0.0116, +0.0228] | 3/3 | +0.0147 [+0.0053, +0.0239] | 3/3 |
| Federated GBDT - Federated MLP | +0.0092 [-0.0085, +0.0268] | 3/3 | +0.0140 [+0.0045, +0.0233] | 3/3 |
| Federated GBDT - Pooled LightGBM | -0.0140 [-0.0242, -0.0028] | 0/3 | -0.0038 [-0.0091, +0.0015] | 0/3 |
| Federated MLP - Federated LR (FLICS's model class) | -0.0042 [-0.0162, +0.0077] | 1/3 | +0.0006 [-0.0067, +0.0075] | 3/3 |
| Federated MLP - Central MLP | +0.0033 [-0.0075, +0.0149] | 2/3 | -0.0015 [-0.0096, +0.0063] | 0/3 |
| Federated LR (FLICS's model class) - Central LR | +0.0076 [-0.0076, +0.0224] | – | +0.0022 [-0.0056, +0.0096] | – |
| Federated MLP - Local-only MLP | +0.0272 [+0.0013, +0.0520] | 3/3 | +0.0829 [+0.0632, +0.1022] | 3/3 |
| Pooled LightGBM - Central LR | +0.0266 [+0.0086, +0.0461] | – | +0.0206 [+0.0106, +0.0310] | – |

**Table 11c. Does training on more hospitals help? The same test patients scored by a model trained only on the federation (this run) and by one trained on all hospitals (the final strict runs), seed 42, patients matched by id. Intervals reflect test-patient sampling only, not training randomness.**

| Model | 3 largest: AUROC, trained on the federation → on all hospitals | 3 largest: all - federation [95% CI] | 10 largest: AUROC, trained on the federation → on all hospitals | 10 largest: all - federation [95% CI] |
|---|---|---|---|---|
| Federated GBDT | 0.8603 → 0.8863 | +0.0260 [+0.0100, +0.0419] | 0.8675 → 0.8785 | +0.0110 [+0.0033, +0.0187] |
| Federated MLP | 0.8511 → 0.8714 | +0.0203 [+0.0044, +0.0369] | 0.8534 → 0.8617 | +0.0082 [+0.0013, +0.0156] |
| Federated LR (FLICS's model class) | 0.8553 → 0.8702 | +0.0149 [+0.0026, +0.0283] | 0.8528 → 0.8566 | +0.0038 [-0.0018, +0.0097] |
| Pooled LightGBM | 0.8743 → 0.8895 | +0.0152 [+0.0011, +0.0301] | 0.8712 → 0.8795 | +0.0082 [+0.0018, +0.0145] |

**Table 4a. Sepsis proxy, 24h window, clean features, label-valid hospitals (151 hospitals; test n = 16,238).**

| Model | AUROC (mean ± sd, 3 seeds) | 95% CI (seed 42) | AUPRC | Brier | ECE | per-hosp. macro AUROC | AUROC, per-hosp. calibration |
|---|---|---|---|---|---|---|---|
| Federated logistic regression | 0.8267 ± 0.0003 | [0.8211, 0.8333] | 0.6979 | 0.1589 | 0.0181 | 0.8154 | 0.8384 |
| Federated MLP (FedAvg + FedProx) | 0.8418 ± 0.0004 | [0.8362, 0.8477] | 0.7219 | 0.1528 | 0.0234 | 0.8313 | 0.8510 |
| Federated GBDT (histogram boosting) | 0.8599 ± 0.0005 | [0.8548, 0.8660] | 0.7571 | 0.1433 | 0.0082 | 0.8367 | 0.8613 |
| Central logistic regression | 0.8460 ± 0.0000 | [0.8404, 0.8517] | 0.7280 | 0.1502 | 0.0124 | 0.8238 | 0.8497 |
| Central MLP | 0.8511 ± 0.0003 | [0.8458, 0.8569] | 0.7392 | 0.1476 | 0.0101 | 0.8310 | 0.8550 |
| Central LightGBM | 0.8623 ± 0.0001 | [0.8568, 0.8676] | 0.7608 | 0.1421 | 0.0095 | 0.8385 | 0.8634 |
| Central GBDT (same code, pooled) | 0.8600 ± 0.0006 | [0.8555, 0.8662] | 0.7576 | 0.1432 | 0.0085 | 0.8369 | 0.8612 |

**Table 4b. Sepsis proxy, 6h early-warning window, clean features, label-valid hospitals (test n = 22,597). The federated GBDT row is the rerun with a 1,200-round limit (higher validation AUROC in all three seeds, early-stopped in every seed); the first run hit its 600-round limit in every seed.**

| Model | AUROC (mean ± sd, 3 seeds) | 95% CI (seed 42) | AUPRC | Brier | ECE | per-hosp. macro AUROC | AUROC, per-hosp. calibration |
|---|---|---|---|---|---|---|---|
| Federated logistic regression | 0.8204 ± 0.0001 | [0.8152, 0.8266] | 0.6422 | 0.1520 | 0.0213 | 0.8088 | 0.8318 |
| Federated MLP (FedAvg + FedProx) | 0.8389 ± 0.0001 | [0.8342, 0.8442] | 0.6756 | 0.1444 | 0.0119 | 0.8277 | 0.8484 |
| Federated GBDT (histogram boosting) | 0.8574 ± 0.0000 | [0.8531, 0.8622] | 0.7090 | 0.1364 | 0.0081 | 0.8344 | 0.8595 |
| Federated GBDT, first run (600-round limit) | 0.8566 ± 0.0002 | [0.8525, 0.8617] | 0.7068 | 0.1368 | 0.0092 | 0.8333 | 0.8590 |
| Central logistic regression | 0.8397 ± 0.0000 | [0.8353, 0.8448] | 0.6740 | 0.1442 | 0.0122 | 0.8169 | 0.8437 |
| Central MLP | 0.8505 ± 0.0007 | [0.8462, 0.8559] | 0.7010 | 0.1390 | 0.0066 | 0.8317 | 0.8546 |
| Central LightGBM | 0.8589 ± 0.0002 | [0.8547, 0.8644] | 0.7144 | 0.1356 | 0.0097 | 0.8366 | 0.8608 |
| Central GBDT (same code, pooled) | 0.8571 ± 0.0002 | [0.8529, 0.8620] | 0.7095 | 0.1365 | 0.0074 | 0.8340 | 0.8593 |

**Table 4c. Sepsis results of Fuzail et al. (2026) next to ours, for orientation only. The two columns are not comparable (see Section 3).**

| Model | Fuzail et al.: AUC (central) | Fuzail et al.: test accuracy (central) | Fuzail et al.: federated accuracy | Ours, 24 h: AUROC | Ours, 24 h: accuracy |
|---|---|---|---|---|---|
| Random forest | 1.00 | 0.9926 | 0.9850 | not run | not run |
| LightGBM | 0.99 | 0.9489 | 0.9624 | 0.8623 (central) | 0.7689 |
| XGBoost | 0.99 | 0.9579 | 0.9605 | not run | not run |
| KNN | 0.96 | 0.8671 | 0.9172 | not run | not run |
| Logistic regression | 0.73 | 0.6740 | 0.5011 | 0.8460 (central), 0.8267 (federated) | 0.7599 (central), 0.7338 (federated) |
| MLP | not reported | not reported | not reported | 0.8511 (central), 0.8418 (federated) | 0.7521 (central), 0.7440 (federated) |
| Federated GBDT (histogram boosting) | not applicable | – | – | 0.8599 (federated) | 0.7608 (federated) |

Fuzail et al. report no AUC for any federated model. Our accuracies use the validation-selected threshold on a test set with 34.3% proxy-positive prevalence; theirs use a SMOTE-balanced test set with 50% prevalence.

**Table 9. Communication of the two federated models. GBDT figures are sparse upper bounds (a client sends at most its active rows times the selected features per level, 12 bytes per cell) for the winning configuration of each seed; one 'round' is one tree (the 6 h sepsis row is the 1,200-round rerun, which early-stopped in every seed; its first run hit the 600-round limit; the 6 h wall-clock includes one seed that took about three times as long as the other two, probably machine load). The 'rounds kept' column shows the best round per seed. MLP uplink = one float32 copy of the weights per client per round. No compression, differential privacy or secure aggregation in either.**

| Protocol | Model | rounds (trees) kept | uplink per client per round (MB) | uplink, all clients, one model (GB) | wall-clock, 4-trial search + 3 seeds (min) |
|---|---|---|---|---|---|
| Strict | Federated GBDT | 248-281 | 11.4 | 653-730 | 97 |
| Strict | Federated MLP (295,937 parameters) | 64-71 rounds run | 1.2 | 15-17 | – |
| All stays | Federated GBDT | 296-553 | 8.6 | 580-1033 | 445 |
| All stays | Federated MLP | 75-100 rounds run | 1.2 | 18-24 | – |
| Sepsis 24 h | Federated GBDT (151 label-valid hospitals) | 342-541 | 7.4 | 411-642 | 102 |
| Sepsis 6 h | Federated GBDT (151 label-valid hospitals) | 732-945 | 9.4 | 1082-1390 | 208 |

**Table 6. Federated MLP versus the FedProx proximal weight (strict cohort, seed 42, lr 0.001, 2 local epochs). Selection on validation (μ = 1.0 chosen: μ = 3 was not better by 0.002).**

| FedProx μ | validation AUROC | test AUROC | per-hosp. macro AUROC | AUROC, per-hosp. calibration |
|---|---|---|---|---|
| 0 | 0.8298 | 0.8341 | 0.8324 | 0.8420 |
| 0.01 | 0.8374 | 0.8402 | 0.8373 | 0.8469 |
| 0.05 | 0.8447 | 0.8470 | 0.8440 | 0.8529 |
| 0.1 | 0.8479 | 0.8506 | 0.8470 | 0.8559 |
| 0.3 | 0.8527 | 0.8567 | 0.8526 | 0.8607 |
| 1.0 | 0.8571 | 0.8611 | 0.8573 | 0.8630 |
| 3.0 | 0.8572 | 0.8620 | 0.8595 | 0.8656 |
| 10.0 | 0.8484 | 0.8551 | 0.8541 | 0.8606 |

**Table 7. Subgroup AUROC on the test split. Per-hospital AUROC over 102 hospitals with at least 5 events and 5 non-events: federated MLP median 0.867 (IQR 0.826–0.898), central LightGBM median 0.881 (IQR 0.839–0.917).**

| Subgroup (strict cohort, seed 42) | n | events | Federated MLP AUROC | Central LightGBM AUROC |
|---|---|---|---|---|
| overall | 19,802 | 1,817 | 0.8611 | 0.8799 |
| age < 45 | 2,658 | 117 | 0.9318 | 0.9341 |
| age 45–64 | 6,822 | 512 | 0.8786 | 0.9007 |
| age 65–79 | 6,681 | 673 | 0.8414 | 0.8603 |
| age 80+ | 3,632 | 515 | 0.7954 | 0.8198 |
| Med-Surg ICU | 10,608 | 991 | 0.8574 | 0.8763 |
| MICU | 1,867 | 223 | 0.8499 | 0.8763 |
| CCU-CTICU | 1,661 | 152 | 0.8709 | 0.8839 |
| Neuro ICU | 1,486 | 111 | 0.8685 | 0.8816 |
| Cardiac ICU | 1,374 | 133 | 0.8657 | 0.8811 |
| SICU | 1,336 | 119 | 0.8341 | 0.8651 |
| same 15,151 patients with APACHE IVa | 15,151 | – | 0.8660 vs APACHE 0.8309 (Δ +0.0351 [+0.0273, +0.0437]) | 0.8817 vs APACHE 0.8309 (Δ +0.0508 [+0.0433, +0.0587]) |

**Table 8. External check: scores trained on the antibiotic-plus-organ-dysfunction proxy also rank clinician-documented sepsis diagnoses (never a training target; infection diagnosis tokens are excluded from the features). The federated GBDT rows are a single seed (42); the 6 h row is the 1,200-round run (fedgbdt_sepsis6h_mr1200), whose five quintile rates, low to high, are 4.5%, 1.9%, 8.5%, 17.7% and 35.8%: the lowest quintile is above the second. Intervals are not paired.**

| Model (clean features) | window | AUROC vs proxy (target) | AUROC vs documented sepsis [95% CI] | AUPRC (prevalence) | documented rate, lowest → highest risk quintile |
|---|---|---|---|---|---|
| Central LightGBM | 24h | 0.8621 | 0.7721 [0.7626, 0.7827] | 0.409 (0.155) | 5.3% → 40.4% |
| Federated MLP | 24h | 0.8418 | 0.7669 [0.7562, 0.7768] | 0.386 (0.155) | 3.9% → 38.0% |
| Federated GBDT (seed 42) | 24h | 0.8603 | 0.7683 [0.7585, 0.7787] | 0.399 (0.155) | 5.2% → 40.0% |
| Central LightGBM | 6h | 0.8591 | 0.7798 [0.7720, 0.7878] | 0.361 (0.137) | 4.4% → 36.0% |
| Federated MLP | 6h | 0.8390 | 0.7769 [0.7690, 0.7852] | 0.345 (0.137) | 3.3% → 34.4% |
| Federated GBDT (seed 42) | 6h | 0.8574 | 0.7796 [0.7717, 0.7873] | 0.359 (0.137) | 4.5% → 35.8% |

**Table 12. What FLICS's label does to our AUROC. The same seed-42 predictions are scored against in-hospital death (the training target) and against ICU-discharge death (FLICS's label); no model was trained on ICU-discharge death. Values are copied from the printed output of scripts/paper_label_variant_check.py with --paired-bootstrap 2000. The two AUROC columns carry marginal bootstrap intervals; the difference column is a paired bootstrap over the test patients (one resample feeds both labels), so its interval is for the difference itself. One seed (42).**

| Test patients (seed 42) | Model | test n | in-hospital mortality | ICU-discharge mortality | hospital deaths after ICU discharge | AUROC vs in-hospital death (our target) | AUROC vs ICU-discharge death (FLICS's label) | difference [95% paired CI] |
|---|---|---|---|---|---|---|---|---|
| 3 largest hospitals | Federated LR | 1,936 | 9.1% | 4.0% | 55.9% | 0.8553 [0.8256, 0.8832] | 0.8890 [0.8628, 0.9134] | +0.0337 [+0.0069, +0.0623] |
| 3 largest hospitals | Federated GBDT | 1,936 | 9.1% | 4.0% | 55.9% | 0.8603 [0.8344, 0.8824] | 0.8946 [0.8678, 0.9198] | +0.0343 [+0.0089, +0.0617] |
| 10 largest hospitals | Federated LR | 5,002 | 9.7% | 5.1% | 47.8% | 0.8528 [0.8353, 0.8701] | 0.8718 [0.8495, 0.8895] | +0.0190 [+0.0030, +0.0362] |
| 10 largest hospitals | Federated GBDT | 5,002 | 9.7% | 5.1% | 47.8% | 0.8675 [0.8521, 0.8827] | 0.8889 [0.8710, 0.9055] | +0.0214 [+0.0062, +0.0365] |
| All hospitals (strict cohort) | Federated GBDT | 19,802 | 9.2% | 5.2% | 44.1% | 0.8742 [0.8673, 0.8816] | 0.8870 [0.8780, 0.8966] | +0.0127 [+0.0055, +0.0203] |

**Table 13. Federation cost for the GBDT. The federated GBDT against the same code trained pooled on one client holding every training row, with the federated winner's hyperparameters, on identical test patients (seed 42; 2,000 paired bootstrap resamples; means over three seeds). The only difference between the two GBDTs is that the federated one merges bin edges from client summaries. The last column is pooled GBDT minus pooled LightGBM: the part of the gap to LightGBM that comes from implementation and search, not from federation.**

| Setting | test n (events) | Federated GBDT AUROC (3 seeds) | Pooled GBDT, same code (3 seeds) | federated - pooled [95% CI] | seeds with federated > pooled | pooled GBDT - pooled LightGBM [95% CI] |
|---|---|---|---|---|---|---|
| Mortality, strict 24 h landmark | 19,802 (1817) | 0.8746 ± 0.0005 | 0.8746 ± 0.0008 | +0.0004 [-0.0016, +0.0024] | 2/3 | -0.0060 [-0.0085, -0.0038] |
| Mortality, all stays | 29,968 (2710) | 0.9066 ± 0.0009 | 0.9063 ± 0.0003 | +0.0006 [-0.0009, +0.0019] | 2/3 | -0.0024 [-0.0038, -0.0009] |
| Sepsis proxy, 24 h | 16,238 (5563) | 0.8599 ± 0.0005 | 0.8600 ± 0.0006 | -0.0002 [-0.0018, +0.0012] | 1/3 | -0.0016 [-0.0032, -0.0001] |
| Sepsis proxy, 6 h | 22,597 (6614) | 0.8574 ± 0.0000 | 0.8571 ± 0.0002 | +0.0002 [-0.0009, +0.0012] | 3/3 | -0.0019 [-0.0031, -0.0006] |
| Mortality, 3 largest hospitals | 1,936 (177) | 0.8635 ± 0.0024 | 0.8692 ± 0.0052 | -0.0102 [-0.0187, -0.0017] | 1/3 | -0.0037 [-0.0135, +0.0058] |
| Mortality, 10 largest hospitals | 5,002 (483) | 0.8662 ± 0.0017 | 0.8670 ± 0.0024 | -0.0002 [-0.0054, +0.0052] | 0/3 | -0.0036 [-0.0092, +0.0016] |

**Table 14a. Mortality, strict 24 h landmark cohort: accuracy, AUROC and F1 in the format of FLICS's Tables IV and V (test n = 19,802; mortality or proxy prevalence 9.2%, so a classifier that predicts nobody positive scores accuracy 0.908). Mean ± sd over the training seeds (same split); the AUROC interval is the seed-42 bootstrap. Bold = highest mean among the federated models, underline = highest mean among the central (non-private) models, per metric, ties marked on every tied row; the marks follow the point estimates and do not mean the difference is resolved (it is often smaller than the seed sd). Accuracy and F1 are test values at the threshold that maximises F1 on the validation set (never tuned on the test set; FLICS tunes its threshold on its test set). Both depend on prevalence, so they are not comparable across cohorts. Central LR has one seed.**

| Model | Accuracy (mean ± sd, 3 seeds) | AUROC (mean ± sd, 3 seeds) [95% CI, seed 42] | F1 (mean ± sd, 3 seeds) |
|---|---|---|---|
| Federated LR | 0.8810 ± 0.0029 | 0.8535 ± 0.0001 [0.8445, 0.8616] | 0.4447 ± 0.0005 |
| Federated MLP | 0.8858 ± 0.0046 | 0.8617 ± 0.0006 [0.8521, 0.8688] | 0.4494 ± 0.0047 |
| Federated GBDT | **0.8888 ± 0.0052** | **0.8746 ± 0.0005 [0.8662, 0.8818]** | **0.4722 ± 0.0011** |
| Central LR | 0.8864 (1 seed) | 0.8603 (1 seed) [0.8521, 0.8679] | 0.4480 (1 seed) |
| Central MLP | 0.8874 ± 0.0028 | 0.8691 ± 0.0005 [0.8597, 0.8760] | 0.4628 ± 0.0005 |
| Central LightGBM | 0.8884 ± 0.0018 | <u>0.8796 ± 0.0002 [0.8721, 0.8874]</u> | <u>0.4773 ± 0.0005</u> |
| Central GBDT (same code, pooled) | <u>0.8935 ± 0.0037</u> | 0.8746 ± 0.0008 [0.8661, 0.8816] | 0.4708 ± 0.0022 |

**Table 14b. Mortality, all stays (first-24 h window; includes the short-stay effect): accuracy, AUROC and F1 in the format of FLICS's Tables IV and V (test n = 29,968; mortality or proxy prevalence 9.0%, so a classifier that predicts nobody positive scores accuracy 0.910). Mean ± sd over the training seeds (same split); the AUROC interval is the seed-42 bootstrap. Bold = highest mean among the federated models, underline = highest mean among the central (non-private) models, per metric, ties marked on every tied row; the marks follow the point estimates and do not mean the difference is resolved (it is often smaller than the seed sd). Accuracy and F1 are test values at the threshold that maximises F1 on the validation set (never tuned on the test set; FLICS tunes its threshold on its test set). Both depend on prevalence, so they are not comparable across cohorts. Central LR has one seed.**

| Model | Accuracy (mean ± sd, 3 seeds) | AUROC (mean ± sd, 3 seeds) [95% CI, seed 42] | F1 (mean ± sd, 3 seeds) |
|---|---|---|---|
| Federated LR | 0.9061 ± 0.0019 | 0.8813 ± 0.0001 [0.8751, 0.8870] | 0.4998 ± 0.0025 |
| Federated MLP | 0.9089 ± 0.0036 | 0.8915 ± 0.0006 [0.8841, 0.8967] | 0.5151 ± 0.0014 |
| Federated GBDT | **0.9146 ± 0.0090** | **0.9066 ± 0.0009 [0.9020, 0.9125]** | **0.5433 ± 0.0031** |
| Central LR | 0.9015 (1 seed) | 0.8878 (1 seed) [0.8819, 0.8931] | 0.5074 (1 seed) |
| Central MLP | 0.9158 ± 0.0031 | 0.8994 ± 0.0010 [0.8930, 0.9038] | 0.5277 ± 0.0074 |
| Central LightGBM | <u>0.9207 ± 0.0021</u> | <u>0.9092 ± 0.0002 [0.9039, 0.9141]</u> | <u>0.5515 ± 0.0039</u> |
| Central GBDT (same code, pooled) | 0.9183 ± 0.0004 | 0.9063 ± 0.0003 [0.9013, 0.9122] | 0.5481 ± 0.0011 |

**Table 14c. Mortality, 3 largest hospitals (FLICS-scale federation): accuracy, AUROC and F1 in the format of FLICS's Tables IV and V (test n = 1,936; mortality or proxy prevalence 9.1%, so a classifier that predicts nobody positive scores accuracy 0.909). Mean ± sd over the training seeds (same split); the AUROC interval is the seed-42 bootstrap. Bold = highest mean among the federated models, underline = highest mean among the central (non-private) models, per metric, ties marked on every tied row; the marks follow the point estimates and do not mean the difference is resolved (it is often smaller than the seed sd). Accuracy and F1 are test values at the threshold that maximises F1 on the validation set (never tuned on the test set; FLICS tunes its threshold on its test set). Both depend on prevalence, so they are not comparable across cohorts. Central LR has one seed. The four FLICS rows are its reported numbers (3 hospitals, 5,000 stays; 10 hospitals, 53,500 stays; mortality 12.5-17%; ICU-discharge label; local models omitted) and are not ranked or marked: cohort, label and threshold procedure differ from ours.**

| Model | Accuracy (mean ± sd, 3 seeds) | AUROC (mean ± sd, 3 seeds) [95% CI, seed 42] | F1 (mean ± sd, 3 seeds) |
|---|---|---|---|
| Federated LR | 0.8580 ± 0.0124 | 0.8523 ± 0.0022 [0.8274, 0.8812] | 0.4263 ± 0.0059 |
| Federated MLP | 0.8752 ± 0.0081 | 0.8490 ± 0.0062 [0.8181, 0.8787] | 0.4379 ± 0.0063 |
| Federated GBDT | **0.8833 ± 0.0080** | **0.8635 ± 0.0024 [0.8342, 0.8836]** | **0.4386 ± 0.0041** |
| Central LR | 0.8492 (1 seed) | 0.8477 (1 seed) [0.8180, 0.8751] | 0.4065 (1 seed) |
| Central MLP | 0.8743 ± 0.0066 | 0.8459 ± 0.0032 [0.8188, 0.8764] | 0.4308 ± 0.0043 |
| Central LightGBM | 0.8743 ± 0.0014 | <u>0.8740 ± 0.0007 [0.8504, 0.8948]</u> | <u>0.4608 ± 0.0044</u> |
| Central GBDT (same code, pooled) | <u>0.8910 ± 0.0033</u> | 0.8692 ± 0.0052 [0.8471, 0.8934] | 0.4557 ± 0.0165 |
| FLICS, FedAvg LR (as reported) | 0.90 (0.88–0.92) | 0.87 (0.83–0.90) | 0.60 |
| FLICS, FedAvg LSTM (as reported) | 0.89 (0.87–0.91) | 0.87 (0.83–0.90) | 0.61 |
| FLICS, Centralized LR (as reported) | 0.88 (0.86–0.90) | 0.86 (0.83–0.90) | 0.60 |
| FLICS, Centralized XGBoost (as reported) | 0.91 (0.89–0.93) | 0.90 (0.87–0.93) | 0.65 |

**Table 14d. Mortality, 10 largest hospitals (FLICS-scale federation): accuracy, AUROC and F1 in the format of FLICS's Tables IV and V (test n = 5,002; mortality or proxy prevalence 9.7%, so a classifier that predicts nobody positive scores accuracy 0.903). Mean ± sd over the training seeds (same split); the AUROC interval is the seed-42 bootstrap. Bold = highest mean among the federated models, underline = highest mean among the central (non-private) models, per metric, ties marked on every tied row; the marks follow the point estimates and do not mean the difference is resolved (it is often smaller than the seed sd). Accuracy and F1 are test values at the threshold that maximises F1 on the validation set (never tuned on the test set; FLICS tunes its threshold on its test set). Both depend on prevalence, so they are not comparable across cohorts. Central LR has one seed. The four FLICS rows are its reported numbers (3 hospitals, 5,000 stays; 10 hospitals, 53,500 stays; mortality 12.5-17%; ICU-discharge label; local models omitted) and are not ranked or marked: cohort, label and threshold procedure differ from ours.**

| Model | Accuracy (mean ± sd, 3 seeds) | AUROC (mean ± sd, 3 seeds) [95% CI, seed 42] | F1 (mean ± sd, 3 seeds) |
|---|---|---|---|
| Federated LR | 0.8737 ± 0.0045 | 0.8524 ± 0.0008 [0.8342, 0.8689] | 0.4507 ± 0.0012 |
| Federated MLP | 0.8752 ± 0.0071 | 0.8544 ± 0.0015 [0.8360, 0.8693] | 0.4447 ± 0.0037 |
| Federated GBDT | **0.8864 ± 0.0043** | **0.8662 ± 0.0017 [0.8524, 0.8835]** | **0.4704 ± 0.0062** |
| Central LR | 0.8713 (1 seed) | 0.8506 (1 seed) [0.8317, 0.8681] | 0.4458 (1 seed) |
| Central MLP | 0.8697 ± 0.0021 | 0.8572 ± 0.0017 [0.8379, 0.8707] | 0.4504 ± 0.0038 |
| Central LightGBM | <u>0.8891 ± 0.0023</u> | <u>0.8704 ± 0.0009 [0.8557, 0.8873]</u> | 0.4697 ± 0.0042 |
| Central GBDT (same code, pooled) | 0.8879 ± 0.0023 | 0.8670 ± 0.0024 [0.8515, 0.8834] | <u>0.4740 ± 0.0015</u> |
| FLICS, FedAvg LR (as reported) | 0.89 (0.87–0.91) | 0.83 (0.79–0.87) | 0.59 |
| FLICS, FedAvg LSTM (as reported) | 0.89 (0.87–0.91) | 0.84 (0.79–0.88) | 0.61 |
| FLICS, Centralized LR (as reported) | 0.89 (0.87–0.91) | 0.84 (0.79–0.88) | 0.61 |
| FLICS, Centralized XGBoost (as reported) | 0.90 (0.88–0.91) | 0.88 (0.85–0.91) | 0.63 |

**Table 14e. Sepsis proxy, 24 h window, clean features, label-valid hospitals: accuracy, AUROC and F1 in the format of FLICS's Tables IV and V (test n = 16,238; mortality or proxy prevalence 34.3%, so a classifier that predicts nobody positive scores accuracy 0.657). Mean ± sd over the training seeds (same split); the AUROC interval is the seed-42 bootstrap. Bold = highest mean among the federated models, underline = highest mean among the central (non-private) models, per metric, ties marked on every tied row; the marks follow the point estimates and do not mean the difference is resolved (it is often smaller than the seed sd). Accuracy and F1 are test values at the threshold that maximises F1 on the validation set (never tuned on the test set; FLICS tunes its threshold on its test set). Both depend on prevalence, so they are not comparable across cohorts. Central LR has one seed.**

| Model | Accuracy (mean ± sd, 3 seeds) | AUROC (mean ± sd, 3 seeds) [95% CI, seed 42] | F1 (mean ± sd, 3 seeds) |
|---|---|---|---|
| Federated LR | 0.7338 ± 0.0057 | 0.8267 ± 0.0003 [0.8211, 0.8333] | 0.6720 ± 0.0007 |
| Federated MLP | 0.7440 ± 0.0033 | 0.8418 ± 0.0004 [0.8362, 0.8477] | 0.6838 ± 0.0004 |
| Federated GBDT | **0.7608 ± 0.0066** | **0.8599 ± 0.0005 [0.8548, 0.8660]** | **0.7007 ± 0.0010** |
| Central LR | 0.7599 (1 seed) | 0.8460 (1 seed) [0.8404, 0.8517] | 0.6929 (1 seed) |
| Central MLP | 0.7521 ± 0.0040 | 0.8511 ± 0.0003 [0.8458, 0.8569] | 0.6913 ± 0.0004 |
| Central LightGBM | <u>0.7689 ± 0.0024</u> | <u>0.8623 ± 0.0001 [0.8568, 0.8676]</u> | <u>0.7031 ± 0.0009</u> |
| Central GBDT (same code, pooled) | 0.7665 ± 0.0046 | 0.8600 ± 0.0006 [0.8555, 0.8662] | 0.7006 ± 0.0004 |

**Table 14f. Sepsis proxy, 6 h early-warning window, clean features, label-valid hospitals: accuracy, AUROC and F1 in the format of FLICS's Tables IV and V (test n = 22,597; mortality or proxy prevalence 29.3%, so a classifier that predicts nobody positive scores accuracy 0.707). Mean ± sd over the training seeds (same split); the AUROC interval is the seed-42 bootstrap. Bold = highest mean among the federated models, underline = highest mean among the central (non-private) models, per metric, ties marked on every tied row; the marks follow the point estimates and do not mean the difference is resolved (it is often smaller than the seed sd). Accuracy and F1 are test values at the threshold that maximises F1 on the validation set (never tuned on the test set; FLICS tunes its threshold on its test set). Both depend on prevalence, so they are not comparable across cohorts. Central LR has one seed.**

| Model | Accuracy (mean ± sd, 3 seeds) | AUROC (mean ± sd, 3 seeds) [95% CI, seed 42] | F1 (mean ± sd, 3 seeds) |
|---|---|---|---|
| Federated LR | 0.7350 ± 0.0061 | 0.8204 ± 0.0001 [0.8152, 0.8266] | 0.6255 ± 0.0009 |
| Federated MLP | 0.7646 ± 0.0076 | 0.8389 ± 0.0001 [0.8342, 0.8442] | 0.6423 ± 0.0003 |
| Federated GBDT | **0.7714 ± 0.0046** | **0.8574 ± 0.0000 [0.8531, 0.8622]** | **0.6628 ± 0.0014** |
| Central LR | 0.7526 (1 seed) | 0.8397 (1 seed) [0.8353, 0.8448] | 0.6466 (1 seed) |
| Central MLP | 0.7689 ± 0.0019 | 0.8505 ± 0.0007 [0.8462, 0.8559] | 0.6558 ± 0.0017 |
| Central LightGBM | <u>0.7801 ± 0.0089</u> | <u>0.8589 ± 0.0002 [0.8547, 0.8644]</u> | <u>0.6625 ± 0.0007</u> |
| Central GBDT (same code, pooled) | 0.7739 ± 0.0057 | 0.8571 ± 0.0002 [0.8529, 0.8620] | 0.6618 ± 0.0006 |

## 7. Figures
- `figures/fig1_fedprox_sweep.png`: AUROC versus FedProx weight.
- `figures/fig2_mortality_main.png`: main comparison, now including the pooled GBDT, with the FLICS reference lines on both panels, labelled with their source; on the all-stays panel they are context only (no comparison is claimed).
- `figures/fig3_sepsis_ladder.png`: sepsis feature ladder, pooled versus per-hospital AUROC.
- `figures/fig4_per_hospital.png`: per-hospital AUROC distribution (strict, seed 42), including the federated GBDT.

## 8. Revision notes (delete before submission)
- FLICS is the base reference; it was chosen after the comparison was known (disclosed in Section 5).
- DP-IG-FL removed from every table and sentence (it uses MIMIC-III as well). Kerkouche et al. removed on the same principle (Premier Healthcare Database, not eICU). Both can stay in related work.
- Headline federated model is the federated GBDT, selected by validation AUROC on both protocols (Section 1); the federated MLP is the neural baseline. Final 3-seed numbers on strict (`fedgbdt_strict_final`) and all stays (`fedgbdt_allstays_final`); paired tests against the federated MLP and pooled LightGBM use the saved predictions.
- Verdicts on FLICS: GBDT level on the strict 3-hospital figure, ahead elsewhere; no claim of superiority on strict.
- Table 9 (communication and wall-clock) is new. Federation-versus-isolation restated (+0.07 to +0.12); Fuzail et al. (sepsis) described as considered-but-not-comparable (Table 4c).
- Tables and figures were regenerated by `Fed-THOR_results_package/reproduce/` from `outputs/paper/summary_all_runs.md` (re-run `scripts/paper_collect_results.py` first).
- Sepsis: federated GBDT added (Tables 4a, 4b, 8 for 24 h, 9 and 4c); headline on sepsis by the same validation rule. The first 6 h run hit the 600-round limit (superseded in v16, see the v16 note below).
- Table 3 is now strict-only (all-stays columns removed: FLICS's settings are not comparable, and all stays carry a short-stay effect) and says it compares reported numbers across settings. New Table 10: paired same-patient comparisons (federated GBDT vs a FLICS-style federated LR, the federated MLP and pooled LightGBM on all, the 10 largest and the 3 largest hospitals' test patients), from `reproduce/paired_subsets.py`. New §2 and §5 text on evaluation noise and on FLICS's setting differences.
- v15: FLICS-scale federations added (Tables 11a-11c from `reproduce/paired_flicsscale.py`; runs `flicsstyle_top3` and `flicsstyle_top10`), new §2 item 3 (later items renumbered), a label-check paragraph and a small-federation paragraph in §5, two new entries in §4. Table 10's 'FLICS-style' LR is relabelled 'FLICS's model class': those models were trained on all hospitals. The v14 remark that the 0.8702 LR 'supports reading FLICS's numbers as what a federated LR gives' is withdrawn (trained on the three hospitals the LR scores 0.8523).
- v16: the 6 h federated GBDT is now the 1,200-round rerun `fedgbdt_sepsis6h_mr1200`, selected by validation AUROC; the first run stays in Table 4b. Tables 3a (caption), 4b, 8 and 9 and §1, §3 and §5 updated; the paired 6 h numbers come from `reproduce/paired_sepsis6h.py`. New Table 12 and a rewritten §5 label paragraph: what FLICS's label does to our AUROC (values copied from the printed output of `scripts/paper_label_variant_check.py`). The 6 h documented-sepsis check was run on the rerun. `analyze_ablation_results.py` moved to `scripts/`; the README has a paper-track section. The abstract and discussion are drafted in `Fed-THOR_abstract_and_discussion.md`.
- v18: Table 12 now carries the paired interval of the label effect (`scripts/paper_label_variant_check.py --paired-bootstrap 2000`, five runs); the strict row's difference is the script's +0.0127 (v16 showed +0.0128 from subtracting rounded values). §5, §2 item 3 and §4 reworded accordingly. The abstract and discussion carry the final wording of the FLICS comparison and the limitations.
- v19: the second reference study was removed from every paper-facing file (text, tables, figure 2, abstract and discussion) and from the repository (config, probe script renamed `scripts/paper_hospital_size_probe.py`, reference rows, tests, docs). Former Table 3b and Table 5 are gone and Table 3a is now Table 3; table numbers still have gaps and are renumbered at submission.
- v21: pooled GBDT (`gbdt_central`) run in all six settings; Table 13 (federation cost for the GBDT, `reproduce/paired_central_gbdt.py`) and central-GBDT rows in Tables 1, 2, 4a and 4b; new Table 14a-14f (accuracy, AUROC and F1 for every federated and central model per setting; bold = best federated, underline = best central, as in FLICS's Tables IV and V; FLICS's own rows added at the two FLICS scales, unranked), built by `reproduce/final_comparison.py` from `results.json` only. Accuracy and F1 use the validation-chosen F1-maximising threshold (FLICS tunes on its test set). New §1 bullet, §2 item 9 (items renumbered), §2 items 3 and 5 rewritten around the control, §3 sentence, three §4 entries and two §5 paragraphs. Figure 2 now has a Central GBDT bar and keeps the FLICS lines on the all-stays panel (context only, decided by the author). Our F1 at FLICS's scales (0.44-0.47) is below FLICS's reported 0.59-0.65; the text says why and draws no conclusion.
