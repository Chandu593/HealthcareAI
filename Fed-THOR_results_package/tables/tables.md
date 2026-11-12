**Mortality, strict 24h landmark cohort (131,695 stays; 204 hospital clients for the MLP, 206 for the GBDT; test n = 19,802).**

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

**Mortality, all stays with the first-24h window (199,108 stays; test n = 29,968). Includes the short-stay effect.**

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

**Base paper: FLICS (2026, eICU only). 'ahead' = our 95% CI lies above the reference; 'level' = it contains it; 'behind' = below. This is a comparison of reported numbers across different settings, not a replication: FLICS uses 3 or 10 hospitals, ICU-discharge mortality, whole-stay summaries and a 1,000-patient test set (reference CI up to ±0.035); ours is the strict 24 h cohort of 206 hospitals with hospital mortality (test n = 19,802). Our all-stays figures are not compared with FLICS (different setting and a short-stay effect). Tables 10 and 11 give same-patient comparisons (Table 11: models trained on federations of FLICS's size); Table 12 measures what FLICS's label alone does to our AUROC.**

| Reference (as reported) | Setting | AUROC | ours, strict: federated GBDT 0.8746 | ours, strict: federated MLP 0.8617 |
|---|---|---|---|---|
| FLICS, FedAvg LR / LSTM | 3 hospitals, 5,000 stays, 1,000-patient test set (CI 0.83-0.90) | 0.87 | level (+0.005) | behind (-0.008) |
| FLICS, FedAvg LR | 10 hospitals, 53,500 stays (CI 0.79-0.87) | 0.83 | ahead (+0.045) | ahead (+0.032) |
| FLICS, FedAvg LSTM | 10 hospitals, 53,500 stays (CI 0.79-0.88) | 0.84 | ahead (+0.035) | ahead (+0.022) |

**Paired method comparisons on identical strict-cohort test patients (seed 42; 1,000 paired bootstrap resamples; from reproduce/paired_subsets.py). The models were trained on all hospitals and are only evaluated on the 10 and 3 largest hospitals by cohort size: this is not a replication of FLICS's federations (Table 11 trains on federations of FLICS's size).**

| Test patients | n (events) | AUROC: federated GBDT | AUROC: federated LR (FLICS's model class) | GBDT - federated LR [95% CI] | GBDT - federated MLP [95% CI] | GBDT - pooled LightGBM [95% CI] |
|---|---|---|---|---|---|---|
| all hospitals | 19,802 (1817) | 0.8742 | 0.8535 | +0.0207 [+0.0159, +0.0262] | +0.0131 [+0.0083, +0.0176] | -0.0057 [-0.0079, -0.0033] |
| 10 largest hospitals | 5,002 (483) | 0.8785 | 0.8566 | +0.0218 [+0.0135, +0.0316] | +0.0168 [+0.0082, +0.0259] | -0.0010 [-0.0055, +0.0034] |
| 3 largest hospitals | 1,936 (177) | 0.8863 | 0.8702 | +0.0161 [+0.0028, +0.0301] | +0.0149 [+0.0004, +0.0306] | -0.0032 [-0.0109, +0.0045] |

**FLICS-scale federations: every model re-trained on only the 3 or the 10 largest hospitals by strict-cohort size (configs/paper_flicsstyle.yaml; 3 largest: 3 hospitals, 12,911 stays, train/val/test = 9,032/1,943/1,936 and 10 largest: 10 hospitals, 33,179 stays, train/val/test = 23,187/4,990/5,002). Strict cohort, hospital mortality, first-24 h features without APACHE scores. 'ahead' = our seed-42 interval lies above the reference, 'level' = it contains it, 'behind' = below (as in Table 3); FLICS's own interval is up to ±0.035, so these labels are context, not claims. Central logistic regression has one seed. Hospital-level inputs (region, teaching status, bed-size category, log hospital size) identify the site in such small federations; they are identical for every arm.**

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

**Paired AUROC differences on identical test patients inside each FLICS-scale federation (seed 42; 2,000 paired bootstrap resamples over test patients; test n = 1,936 (177 events) and 5,002 (483 events)). 'seeds with A > B' counts the three training seeds whose point difference on the same patients is positive ('–' = one of the two arms has a single seed).**

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

**Does training on more hospitals help? The same test patients scored by a model trained only on the federation (this run) and by one trained on all hospitals (the final strict runs), seed 42, patients matched by id. Intervals reflect test-patient sampling only, not training randomness.**

| Model | 3 largest: AUROC, trained on the federation → on all hospitals | 3 largest: all - federation [95% CI] | 10 largest: AUROC, trained on the federation → on all hospitals | 10 largest: all - federation [95% CI] |
|---|---|---|---|---|
| Federated GBDT | 0.8603 → 0.8863 | +0.0260 [+0.0100, +0.0419] | 0.8675 → 0.8785 | +0.0110 [+0.0033, +0.0187] |
| Federated MLP | 0.8511 → 0.8714 | +0.0203 [+0.0044, +0.0369] | 0.8534 → 0.8617 | +0.0082 [+0.0013, +0.0156] |
| Federated LR (FLICS's model class) | 0.8553 → 0.8702 | +0.0149 [+0.0026, +0.0283] | 0.8528 → 0.8566 | +0.0038 [-0.0018, +0.0097] |
| Pooled LightGBM | 0.8743 → 0.8895 | +0.0152 [+0.0011, +0.0301] | 0.8712 → 0.8795 | +0.0082 [+0.0018, +0.0145] |

**Sepsis proxy, 24h window, clean features, label-valid hospitals (151 hospitals; test n = 16,238).**

| Model | AUROC (mean ± sd, 3 seeds) | 95% CI (seed 42) | AUPRC | Brier | ECE | per-hosp. macro AUROC | AUROC, per-hosp. calibration |
|---|---|---|---|---|---|---|---|
| Federated logistic regression | 0.8267 ± 0.0003 | [0.8211, 0.8333] | 0.6979 | 0.1589 | 0.0181 | 0.8154 | 0.8384 |
| Federated MLP (FedAvg + FedProx) | 0.8418 ± 0.0004 | [0.8362, 0.8477] | 0.7219 | 0.1528 | 0.0234 | 0.8313 | 0.8510 |
| Federated GBDT (histogram boosting) | 0.8599 ± 0.0005 | [0.8548, 0.8660] | 0.7571 | 0.1433 | 0.0082 | 0.8367 | 0.8613 |
| Central logistic regression | 0.8460 ± 0.0000 | [0.8404, 0.8517] | 0.7280 | 0.1502 | 0.0124 | 0.8238 | 0.8497 |
| Central MLP | 0.8511 ± 0.0003 | [0.8458, 0.8569] | 0.7392 | 0.1476 | 0.0101 | 0.8310 | 0.8550 |
| Central LightGBM | 0.8623 ± 0.0001 | [0.8568, 0.8676] | 0.7608 | 0.1421 | 0.0095 | 0.8385 | 0.8634 |
| Central GBDT (same code, pooled) | 0.8600 ± 0.0006 | [0.8555, 0.8662] | 0.7576 | 0.1432 | 0.0085 | 0.8369 | 0.8612 |

**Sepsis proxy, 6h early-warning window, clean features, label-valid hospitals (test n = 22,597). The federated GBDT row is the rerun with a 1,200-round limit (higher validation AUROC in all three seeds, early-stopped in every seed); the first run hit its 600-round limit in every seed.**

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

**Communication of the two federated models. GBDT figures are sparse upper bounds (a client sends at most its active rows times the selected features per level, 12 bytes per cell) for the winning configuration of each seed; one 'round' is one tree (the 6 h sepsis row is the 1,200-round rerun, which early-stopped in every seed; its first run hit the 600-round limit; the 6 h wall-clock includes one seed that took about three times as long as the other two, probably machine load). The 'rounds kept' column shows the best round per seed. MLP uplink = one float32 copy of the weights per client per round. No compression, differential privacy or secure aggregation in either.**

| Protocol | Model | rounds (trees) kept | uplink per client per round (MB) | uplink, all clients, one model (GB) | wall-clock, 4-trial search + 3 seeds (min) |
|---|---|---|---|---|---|
| Strict | Federated GBDT | 248-281 | 11.4 | 653-730 | 97 |
| Strict | Federated MLP (295,937 parameters) | 64-71 rounds run | 1.2 | 15-17 | – |
| All stays | Federated GBDT | 296-553 | 8.6 | 580-1033 | 445 |
| All stays | Federated MLP | 75-100 rounds run | 1.2 | 18-24 | – |
| Sepsis 24 h | Federated GBDT (151 label-valid hospitals) | 342-541 | 7.4 | 411-642 | 102 |
| Sepsis 6 h | Federated GBDT (151 label-valid hospitals) | 732-945 | 9.4 | 1082-1390 | 208 |

**Federated MLP versus the FedProx proximal weight (strict cohort, seed 42, lr 0.001, 2 local epochs). Selection on validation (μ = 1.0 chosen: μ = 3 was not better by 0.002).**

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

**Subgroup AUROC on the test split. Per-hospital AUROC over 102 hospitals with at least 5 events and 5 non-events: federated MLP median 0.867 (IQR 0.826–0.898), central LightGBM median 0.881 (IQR 0.839–0.917).**

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

**External check: scores trained on the antibiotic-plus-organ-dysfunction proxy also rank clinician-documented sepsis diagnoses (never a training target; infection diagnosis tokens are excluded from the features). The federated GBDT rows are a single seed (42); the 6 h row is the 1,200-round run (fedgbdt_sepsis6h_mr1200), whose five quintile rates, low to high, are 4.5%, 1.9%, 8.5%, 17.7% and 35.8%: the lowest quintile is above the second. Intervals are not paired.**

| Model (clean features) | window | AUROC vs proxy (target) | AUROC vs documented sepsis [95% CI] | AUPRC (prevalence) | documented rate, lowest → highest risk quintile |
|---|---|---|---|---|---|
| Central LightGBM | 24h | 0.8621 | 0.7721 [0.7626, 0.7827] | 0.409 (0.155) | 5.3% → 40.4% |
| Federated MLP | 24h | 0.8418 | 0.7669 [0.7562, 0.7768] | 0.386 (0.155) | 3.9% → 38.0% |
| Federated GBDT (seed 42) | 24h | 0.8603 | 0.7683 [0.7585, 0.7787] | 0.399 (0.155) | 5.2% → 40.0% |
| Central LightGBM | 6h | 0.8591 | 0.7798 [0.7720, 0.7878] | 0.361 (0.137) | 4.4% → 36.0% |
| Federated MLP | 6h | 0.8390 | 0.7769 [0.7690, 0.7852] | 0.345 (0.137) | 3.3% → 34.4% |
| Federated GBDT (seed 42) | 6h | 0.8574 | 0.7796 [0.7717, 0.7873] | 0.359 (0.137) | 4.5% → 35.8% |

**What FLICS's label does to our AUROC. The same seed-42 predictions are scored against in-hospital death (the training target) and against ICU-discharge death (FLICS's label); no model was trained on ICU-discharge death. Values are copied from the printed output of scripts/paper_label_variant_check.py with --paired-bootstrap 2000. The two AUROC columns carry marginal bootstrap intervals; the difference column is a paired bootstrap over the test patients (one resample feeds both labels), so its interval is for the difference itself. One seed (42).**

| Test patients (seed 42) | Model | test n | in-hospital mortality | ICU-discharge mortality | hospital deaths after ICU discharge | AUROC vs in-hospital death (our target) | AUROC vs ICU-discharge death (FLICS's label) | difference [95% paired CI] |
|---|---|---|---|---|---|---|---|---|
| 3 largest hospitals | Federated LR | 1,936 | 9.1% | 4.0% | 55.9% | 0.8553 [0.8256, 0.8832] | 0.8890 [0.8628, 0.9134] | +0.0337 [+0.0069, +0.0623] |
| 3 largest hospitals | Federated GBDT | 1,936 | 9.1% | 4.0% | 55.9% | 0.8603 [0.8344, 0.8824] | 0.8946 [0.8678, 0.9198] | +0.0343 [+0.0089, +0.0617] |
| 10 largest hospitals | Federated LR | 5,002 | 9.7% | 5.1% | 47.8% | 0.8528 [0.8353, 0.8701] | 0.8718 [0.8495, 0.8895] | +0.0190 [+0.0030, +0.0362] |
| 10 largest hospitals | Federated GBDT | 5,002 | 9.7% | 5.1% | 47.8% | 0.8675 [0.8521, 0.8827] | 0.8889 [0.8710, 0.9055] | +0.0214 [+0.0062, +0.0365] |
| All hospitals (strict cohort) | Federated GBDT | 19,802 | 9.2% | 5.2% | 44.1% | 0.8742 [0.8673, 0.8816] | 0.8870 [0.8780, 0.8966] | +0.0127 [+0.0055, +0.0203] |

**Federation cost for the GBDT. The federated GBDT against the same code trained pooled on one client holding every training row, with the federated winner's hyperparameters, on identical test patients (seed 42; 2,000 paired bootstrap resamples; means over three seeds). The only difference between the two GBDTs is that the federated one merges bin edges from client summaries. The last column is pooled GBDT minus pooled LightGBM: the part of the gap to LightGBM that comes from implementation and search, not from federation.**

| Setting | test n (events) | Federated GBDT AUROC (3 seeds) | Pooled GBDT, same code (3 seeds) | federated - pooled [95% CI] | seeds with federated > pooled | pooled GBDT - pooled LightGBM [95% CI] |
|---|---|---|---|---|---|---|
| Mortality, strict 24 h landmark | 19,802 (1817) | 0.8746 ± 0.0005 | 0.8746 ± 0.0008 | +0.0004 [-0.0016, +0.0024] | 2/3 | -0.0060 [-0.0085, -0.0038] |
| Mortality, all stays | 29,968 (2710) | 0.9066 ± 0.0009 | 0.9063 ± 0.0003 | +0.0006 [-0.0009, +0.0019] | 2/3 | -0.0024 [-0.0038, -0.0009] |
| Sepsis proxy, 24 h | 16,238 (5563) | 0.8599 ± 0.0005 | 0.8600 ± 0.0006 | -0.0002 [-0.0018, +0.0012] | 1/3 | -0.0016 [-0.0032, -0.0001] |
| Sepsis proxy, 6 h | 22,597 (6614) | 0.8574 ± 0.0000 | 0.8571 ± 0.0002 | +0.0002 [-0.0009, +0.0012] | 3/3 | -0.0019 [-0.0031, -0.0006] |
| Mortality, 3 largest hospitals | 1,936 (177) | 0.8635 ± 0.0024 | 0.8692 ± 0.0052 | -0.0102 [-0.0187, -0.0017] | 1/3 | -0.0037 [-0.0135, +0.0058] |
| Mortality, 10 largest hospitals | 5,002 (483) | 0.8662 ± 0.0017 | 0.8670 ± 0.0024 | -0.0002 [-0.0054, +0.0052] | 0/3 | -0.0036 [-0.0092, +0.0016] |

**Mortality, strict 24 h landmark cohort: accuracy, AUROC and F1 in the format of FLICS's Tables IV and V (test n = 19,802; mortality or proxy prevalence 9.2%, so a classifier that predicts nobody positive scores accuracy 0.908). Mean ± sd over the training seeds (same split); the AUROC interval is the seed-42 bootstrap. Bold = highest mean among the federated models, underline = highest mean among the central (non-private) models, per metric, ties marked on every tied row; the marks follow the point estimates and do not mean the difference is resolved (it is often smaller than the seed sd). Accuracy and F1 are test values at the threshold that maximises F1 on the validation set (never tuned on the test set; FLICS tunes its threshold on its test set). Both depend on prevalence, so they are not comparable across cohorts. Central LR has one seed.**

| Model | Accuracy (mean ± sd, 3 seeds) | AUROC (mean ± sd, 3 seeds) [95% CI, seed 42] | F1 (mean ± sd, 3 seeds) |
|---|---|---|---|
| Federated LR | 0.8810 ± 0.0029 | 0.8535 ± 0.0001 [0.8445, 0.8616] | 0.4447 ± 0.0005 |
| Federated MLP | 0.8858 ± 0.0046 | 0.8617 ± 0.0006 [0.8521, 0.8688] | 0.4494 ± 0.0047 |
| Federated GBDT | **0.8888 ± 0.0052** | **0.8746 ± 0.0005 [0.8662, 0.8818]** | **0.4722 ± 0.0011** |
| Central LR | 0.8864 (1 seed) | 0.8603 (1 seed) [0.8521, 0.8679] | 0.4480 (1 seed) |
| Central MLP | 0.8874 ± 0.0028 | 0.8691 ± 0.0005 [0.8597, 0.8760] | 0.4628 ± 0.0005 |
| Central LightGBM | 0.8884 ± 0.0018 | <u>0.8796 ± 0.0002 [0.8721, 0.8874]</u> | <u>0.4773 ± 0.0005</u> |
| Central GBDT (same code, pooled) | <u>0.8935 ± 0.0037</u> | 0.8746 ± 0.0008 [0.8661, 0.8816] | 0.4708 ± 0.0022 |

**Mortality, all stays (first-24 h window; includes the short-stay effect): accuracy, AUROC and F1 in the format of FLICS's Tables IV and V (test n = 29,968; mortality or proxy prevalence 9.0%, so a classifier that predicts nobody positive scores accuracy 0.910). Mean ± sd over the training seeds (same split); the AUROC interval is the seed-42 bootstrap. Bold = highest mean among the federated models, underline = highest mean among the central (non-private) models, per metric, ties marked on every tied row; the marks follow the point estimates and do not mean the difference is resolved (it is often smaller than the seed sd). Accuracy and F1 are test values at the threshold that maximises F1 on the validation set (never tuned on the test set; FLICS tunes its threshold on its test set). Both depend on prevalence, so they are not comparable across cohorts. Central LR has one seed.**

| Model | Accuracy (mean ± sd, 3 seeds) | AUROC (mean ± sd, 3 seeds) [95% CI, seed 42] | F1 (mean ± sd, 3 seeds) |
|---|---|---|---|
| Federated LR | 0.9061 ± 0.0019 | 0.8813 ± 0.0001 [0.8751, 0.8870] | 0.4998 ± 0.0025 |
| Federated MLP | 0.9089 ± 0.0036 | 0.8915 ± 0.0006 [0.8841, 0.8967] | 0.5151 ± 0.0014 |
| Federated GBDT | **0.9146 ± 0.0090** | **0.9066 ± 0.0009 [0.9020, 0.9125]** | **0.5433 ± 0.0031** |
| Central LR | 0.9015 (1 seed) | 0.8878 (1 seed) [0.8819, 0.8931] | 0.5074 (1 seed) |
| Central MLP | 0.9158 ± 0.0031 | 0.8994 ± 0.0010 [0.8930, 0.9038] | 0.5277 ± 0.0074 |
| Central LightGBM | <u>0.9207 ± 0.0021</u> | <u>0.9092 ± 0.0002 [0.9039, 0.9141]</u> | <u>0.5515 ± 0.0039</u> |
| Central GBDT (same code, pooled) | 0.9183 ± 0.0004 | 0.9063 ± 0.0003 [0.9013, 0.9122] | 0.5481 ± 0.0011 |

**Mortality, 3 largest hospitals (FLICS-scale federation): accuracy, AUROC and F1 in the format of FLICS's Tables IV and V (test n = 1,936; mortality or proxy prevalence 9.1%, so a classifier that predicts nobody positive scores accuracy 0.909). Mean ± sd over the training seeds (same split); the AUROC interval is the seed-42 bootstrap. Bold = highest mean among the federated models, underline = highest mean among the central (non-private) models, per metric, ties marked on every tied row; the marks follow the point estimates and do not mean the difference is resolved (it is often smaller than the seed sd). Accuracy and F1 are test values at the threshold that maximises F1 on the validation set (never tuned on the test set; FLICS tunes its threshold on its test set). Both depend on prevalence, so they are not comparable across cohorts. Central LR has one seed. The four FLICS rows are its reported numbers (3 hospitals, 5,000 stays; 10 hospitals, 53,500 stays; mortality 12.5-17%; ICU-discharge label; local models omitted) and are not ranked or marked: cohort, label and threshold procedure differ from ours.**

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

**Mortality, 10 largest hospitals (FLICS-scale federation): accuracy, AUROC and F1 in the format of FLICS's Tables IV and V (test n = 5,002; mortality or proxy prevalence 9.7%, so a classifier that predicts nobody positive scores accuracy 0.903). Mean ± sd over the training seeds (same split); the AUROC interval is the seed-42 bootstrap. Bold = highest mean among the federated models, underline = highest mean among the central (non-private) models, per metric, ties marked on every tied row; the marks follow the point estimates and do not mean the difference is resolved (it is often smaller than the seed sd). Accuracy and F1 are test values at the threshold that maximises F1 on the validation set (never tuned on the test set; FLICS tunes its threshold on its test set). Both depend on prevalence, so they are not comparable across cohorts. Central LR has one seed. The four FLICS rows are its reported numbers (3 hospitals, 5,000 stays; 10 hospitals, 53,500 stays; mortality 12.5-17%; ICU-discharge label; local models omitted) and are not ranked or marked: cohort, label and threshold procedure differ from ours.**

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

**Sepsis proxy, 24 h window, clean features, label-valid hospitals: accuracy, AUROC and F1 in the format of FLICS's Tables IV and V (test n = 16,238; mortality or proxy prevalence 34.3%, so a classifier that predicts nobody positive scores accuracy 0.657). Mean ± sd over the training seeds (same split); the AUROC interval is the seed-42 bootstrap. Bold = highest mean among the federated models, underline = highest mean among the central (non-private) models, per metric, ties marked on every tied row; the marks follow the point estimates and do not mean the difference is resolved (it is often smaller than the seed sd). Accuracy and F1 are test values at the threshold that maximises F1 on the validation set (never tuned on the test set; FLICS tunes its threshold on its test set). Both depend on prevalence, so they are not comparable across cohorts. Central LR has one seed.**

| Model | Accuracy (mean ± sd, 3 seeds) | AUROC (mean ± sd, 3 seeds) [95% CI, seed 42] | F1 (mean ± sd, 3 seeds) |
|---|---|---|---|
| Federated LR | 0.7338 ± 0.0057 | 0.8267 ± 0.0003 [0.8211, 0.8333] | 0.6720 ± 0.0007 |
| Federated MLP | 0.7440 ± 0.0033 | 0.8418 ± 0.0004 [0.8362, 0.8477] | 0.6838 ± 0.0004 |
| Federated GBDT | **0.7608 ± 0.0066** | **0.8599 ± 0.0005 [0.8548, 0.8660]** | **0.7007 ± 0.0010** |
| Central LR | 0.7599 (1 seed) | 0.8460 (1 seed) [0.8404, 0.8517] | 0.6929 (1 seed) |
| Central MLP | 0.7521 ± 0.0040 | 0.8511 ± 0.0003 [0.8458, 0.8569] | 0.6913 ± 0.0004 |
| Central LightGBM | <u>0.7689 ± 0.0024</u> | <u>0.8623 ± 0.0001 [0.8568, 0.8676]</u> | <u>0.7031 ± 0.0009</u> |
| Central GBDT (same code, pooled) | 0.7665 ± 0.0046 | 0.8600 ± 0.0006 [0.8555, 0.8662] | 0.7006 ± 0.0004 |

**Sepsis proxy, 6 h early-warning window, clean features, label-valid hospitals: accuracy, AUROC and F1 in the format of FLICS's Tables IV and V (test n = 22,597; mortality or proxy prevalence 29.3%, so a classifier that predicts nobody positive scores accuracy 0.707). Mean ± sd over the training seeds (same split); the AUROC interval is the seed-42 bootstrap. Bold = highest mean among the federated models, underline = highest mean among the central (non-private) models, per metric, ties marked on every tied row; the marks follow the point estimates and do not mean the difference is resolved (it is often smaller than the seed sd). Accuracy and F1 are test values at the threshold that maximises F1 on the validation set (never tuned on the test set; FLICS tunes its threshold on its test set). Both depend on prevalence, so they are not comparable across cohorts. Central LR has one seed.**

| Model | Accuracy (mean ± sd, 3 seeds) | AUROC (mean ± sd, 3 seeds) [95% CI, seed 42] | F1 (mean ± sd, 3 seeds) |
|---|---|---|---|
| Federated LR | 0.7350 ± 0.0061 | 0.8204 ± 0.0001 [0.8152, 0.8266] | 0.6255 ± 0.0009 |
| Federated MLP | 0.7646 ± 0.0076 | 0.8389 ± 0.0001 [0.8342, 0.8442] | 0.6423 ± 0.0003 |
| Federated GBDT | **0.7714 ± 0.0046** | **0.8574 ± 0.0000 [0.8531, 0.8622]** | **0.6628 ± 0.0014** |
| Central LR | 0.7526 (1 seed) | 0.8397 (1 seed) [0.8353, 0.8448] | 0.6466 (1 seed) |
| Central MLP | 0.7689 ± 0.0019 | 0.8505 ± 0.0007 [0.8462, 0.8559] | 0.6558 ± 0.0017 |
| Central LightGBM | <u>0.7801 ± 0.0089</u> | <u>0.8589 ± 0.0002 [0.8547, 0.8644]</u> | <u>0.6625 ± 0.0007</u> |
| Central GBDT (same code, pooled) | 0.7739 ± 0.0057 | 0.8571 ± 0.0002 [0.8529, 0.8620] | 0.6618 ± 0.0006 |
