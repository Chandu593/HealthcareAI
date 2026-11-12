# Paper track v14: why we are "level" with FLICS, and what can honestly be done about it

> **Correction (v15).** Section 2 below says the federated LR's 0.8702 on the 3 largest hospitals is "essentially FLICS's 0.87 for the same method" and supports reading FLICS's numbers as what a federated LR gives. That model was trained on all hospitals. Trained on FLICS's three hospitals alone it scores 0.8523 (`docs/paper_track_v15.md`, Table 11), so the remark is withdrawn.

Investigation of the FLICS comparison (strict cohort, seed 42, predictions saved in `outputs/paper/`). Nothing here changes a model or a number; it adds a same-patient comparison table, two read-only
diagnostics and an optional replication config.

## 1. The finding in one paragraph

On the strict cohort the federated GBDT scores 0.8746 against FLICS's 3-hospital 0.87: +0.005, "level" by our rule (our interval [0.8662, 0.8818] contains 0.87). That is not a modelling shortfall. FLICS scores 1,000 patients,
so its own interval is about ±0.035 (0.83-0.90). Scored the same way (stratified 1,000-patient test sets at about 14% mortality, 5,000 random draws) our federated GBDT gets **0.874 ± 0.014** (95% range 0.845-0.901): the 0.005
margin is 0.3 standard deviations of pure sampling noise and 37% of draws fall below 0.87. A real improvement of a few thousandths cannot be resolved against such a reference; chasing it would be fitting noise.

## 2. Candidate causes, with evidence

| cause | evidence | can code fix it? |
|---|---|---|
| Evaluation noise in the reference | FLICS 1,000-patient test set, CI 0.83-0.90; ours sd 0.014 per draw | no (statistics) |
| Different, non-reproducible setting | FLICS: 3 or 10 hospitals, ICU-discharge mortality, whole-stay summaries (mean/SD/min/max/last), vitals. Ours: 206 hospitals, in-hospital mortality, first-24 h features, no vitals tables in the export | partly: scale-matched replication (section 4) |
| FLICS's hospital tables do not match real eICU | six hospitals of at least 5,000 stays and one of 8,000 listed; real cohort has three at 5,000+ and a maximum of 7,058; ten hospitals total 53,500 vs 47,211 for the ten largest real ones; mortality 12.5-17.0% at ICU discharge vs 9.0% in-hospital for the whole cohort (6 of 43 real hospitals with 1,500+ stays reach 12.5%) | no; its numbers are context, not a target |
| Information ceiling of our inputs | pooled LightGBM, strict: 0.8796; adding the APACHE scores (`full`) lifts pooled LightGBM on all stays only from 0.9092 to 0.9096. The excluded features are not hiding headroom | only with new information (vitals, longer window) |
| Federated vs pooled gap | federated GBDT 0.0057 [0.0033, 0.0079] below pooled LightGBM overall; **not** significant on the 10 largest (-0.0010 [-0.0055, +0.0034]) or 3 largest (-0.0032 [-0.0109, +0.0045]) hospitals; concentrated in 300-2,500-stay hospitals (-0.008), which hold 61% of test events | slightly: leaf-wise growth, 255 bins, larger search; realistic gain is a few thousandths at best |

Method-over-method, on identical patients (Table 10 of the results section): federated GBDT minus a FLICS-style federated LR is +0.0207 [+0.0159, +0.0262] overall, +0.0218 [+0.0135, +0.0316] on the 10 largest
hospitals and +0.0161 [+0.0028, +0.0301] on the 3 largest (models trained on all hospitals, evaluated on those hospitals' test patients). Our federated LR scores 0.8702 on the 3 largest hospitals, essentially FLICS's
0.87 for the same method, which supports reading FLICS's numbers as what a federated LR gives, and the GBDT as the improvement. FLICS lists federated gradient boosting as an open problem and as future work.

## 3. Two read-only diagnostics (nothing trains)

```
python Fed-THOR_results_package/reproduce/paired_subsets.py
python scripts/paper_label_variant_check.py --run-dir outputs/paper/fedgbdt_strict_final --experiment mortality__no_apache_scores --arm fed_gbdt --data-root "%EICU_DATA_ROOT%"
```

The first regenerates `tables/paired_subsets.json` (the noise study and Table 10) from saved predictions. The second scores the saved in-hospital-mortality predictions against the ICU-discharge death label (FLICS's label)
on the same test patients and reports how many hospital deaths occur after ICU discharge. Run `build_tables.py` afterwards.

## 4. Optional: FLICS-scale replication on real hospitals (`configs/paper_flicsstyle.yaml`, not run)

Trains and evaluates the arms side by side on the 3 and 10 largest real hospitals of the strict cohort, so methods are compared with paired tests, not against a number with a +-0.035 interval:

```
python run_paper.py --config configs/paper_flicsstyle.yaml --reuse-phase1to3 --run-id flicsstyle_top3 --set paper.cohort_filter.min_hospital_patients=3758
python run_paper.py --config configs/paper_flicsstyle.yaml --reuse-phase1to3 --run-id flicsstyle_top10 --set paper.cohort_filter.min_hospital_patients=2699
```

3758 and 2699 are the sizes of the 3rd and 10th largest hospital (12,911 and 33,179 strict-cohort stays; FLICS lists 5,000 and 53,500). Arms: federated LR (FLICS's model), federated MLP, federated GBDT, local-only MLP, pooled LR, MLP and LightGBM
(FLICS's "centralized XGBoost" analogue). The window matches the final strict run, so the cached features are reused (checked). Expect minutes to well under an hour per run (small cohorts); not measured.
Not reproduced: ICU-discharge mortality, whole-stay summaries, vitals, differential privacy. No outcome is promised.

## 5. What this does not do

It does not make the strict federated GBDT "ahead" of 0.87 by any resolvable margin, and it should not be tuned to cross that threshold. It does not change the model, the split or any reported number.
