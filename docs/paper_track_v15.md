# Paper track v15: FLICS-scale federations (like-for-like check) and a correction to v14

v14 could only compare with FLICS indirectly: its models were trained on all hospitals and merely evaluated on FLICS-sized subsets. v15 adds the runs and tables that train on federations of FLICS's own size.
Nothing here changes a model, a config default or any earlier number. New: `Fed-THOR_results_package/reproduce/paired_flicsscale.py`, `tests/unit/test_paper_flicsscale.py`, `tables/paired_flicsscale.json`, Tables 11a-11c.

## 1. What was run (by you, 2026-10-22)

`configs/paper_flicsstyle.yaml` with `paper.cohort_filter.min_hospital_patients` = 3758 (run `flicsstyle_top3`) and 2699 (run `flicsstyle_top10`), reusing the cached strict feature bundle and the permanent split manifest.
Every arm (federated GBDT, MLP and LR; pooled LightGBM; central MLP and LR; local-only MLP) is trained and tested only on the kept hospitals' patients.

| Federation | Hospitals | Stays | Train / val / test | Test events |
|---|---|---|---|---|
| 3 largest (size >= 3,758) | 3 | 12,911 | 9,032 / 1,943 / 1,936 | 177 |
| 10 largest (size >= 2,699) | 10 | 33,179 | 23,187 / 4,990 / 5,002 | 483 |

## 2. What it shows (Tables 11a-11c; seed 42 for paired tests, three seeds for the counts)

| | 3 largest | 10 largest |
|---|---|---|
| Federated GBDT AUROC (3 seeds) | 0.8635 ± 0.0024 | 0.8662 ± 0.0017 |
| Federated LR (FLICS's model class) | 0.8523 ± 0.0022 | 0.8524 ± 0.0008 |
| Federated MLP | 0.8490 ± 0.0062 | 0.8544 ± 0.0015 |
| Pooled LightGBM | 0.8740 ± 0.0007 | 0.8704 ± 0.0009 |
| GBDT - federated LR | +0.0050 [-0.0116, +0.0228] | +0.0147 [+0.0053, +0.0239] |
| GBDT - federated MLP | +0.0092 [-0.0085, +0.0268] | +0.0140 [+0.0045, +0.0233] |
| GBDT - pooled LightGBM | -0.0140 [-0.0242, -0.0028] | -0.0038 [-0.0091, +0.0015] |
| Trained on all hospitals minus on the federation, GBDT | +0.0260 [+0.0100, +0.0419] | +0.0110 [+0.0033, +0.0187] |

* The GBDT beats the LR and the MLP in every seed at both sizes; the difference is resolved at 10 hospitals and not at 3 (177 events).
* A federated LR trained on the three hospitals scores 0.8523, not the 0.8702 of Table 10 (that model was trained on all hospitals). FLICS's 0.87 sits inside our seed-42 interval but about 0.018 above our point estimate.
* Against the FLICS references the GBDT is level at 3 hospitals and above 0.84 at 10 by our interval; FLICS's own interval is up to ±0.035, so no superiority claim is made.
* Federating costs nothing resolvable for the MLP or LR; for the GBDT against pooled LightGBM it costs 0.014 at 3 hospitals and 0.004 at 10. The reason for the larger 3-hospital gap was not isolated.
* Training on more hospitals helps every model compared (LR at 10 hospitals: +0.0038, interval includes 0).

## 3. Correction to v14

`docs/paper_track_v14.md` section 2 said that the federated LR's 0.8702 on the 3 largest hospitals was "essentially FLICS's 0.87 for the same method" and supported reading FLICS's numbers as what a federated LR gives. That 0.8702 came from a model
trained on all 204 hospitals. Trained on FLICS's three hospitals the LR scores 0.8523. The remark is withdrawn (a note was added at the top of the v14 document); Table 10 now says "FLICS's model class" and points to Table 11.

## 4. The label check you ran (`scripts/paper_label_variant_check.py`, federated GBDT, seed 42, strict test n = 19,802)

9.2% die in hospital, 5.2% by ICU discharge; 44.1% of hospital deaths occur after ICU discharge. The same predictions score 0.8742 [0.8673, 0.8816] against in-hospital death and 0.8870 [0.8780, 0.8966] against ICU-discharge death
(FLICS's label): about +0.013 from the label change alone, for one model and one seed. It is reported in section 5 of the results text and is not applied as a correction.

## 5. Safety checks in `paired_flicsscale.py`

During hand-over, the first top-3 zip held the top-10 run's predictions (all 19 files byte-identical to the top-10 ones). Nothing in the files themselves reveals that, so the script now refuses to run unless:
1. each run's predictions have the split sizes recorded in that run's own `results.json`;
2. every file it pairs holds the same patient ids, splits and labels, in the same order;
3. the all-hospital model's predictions are matched to the federation's test patients by patient id (never by row position), with duplicate ids, absent patients and label disagreement all rejected.
Each guard has a test, and each was verified by injecting the bug, watching its test fail, restoring and watching it pass (see `tests/unit/test_paper_flicsscale.py`).

## 6. Reproduce

```
python scripts/paper_collect_results.py
python Fed-THOR_results_package/reproduce/paired_flicsscale.py
python Fed-THOR_results_package/reproduce/build_tables.py
```
Needs `outputs/paper/flicsstyle_top3`, `flicsstyle_top10` (results.json and predictions/), `fedgbdt_strict_final` and `final_mortality_strict`. If `tables/paired_flicsscale.json` is missing, `build_tables.py` warns and leaves Table 11 out; everything else is unchanged.

## 7. Limits

The 3-hospital test set has 177 events (intervals about ±0.025), the bootstrap reflects test sampling only, the seed sd reflects training randomness only, and the training-pool contrast compares different training runs on identical patients.
Hospital-level inputs (region, teaching status, bed-size category, log hospital size) identify the site in such small federations; every arm sees them. One fixed split, no privacy guarantee, and reference figures without usable intervals, as before.
