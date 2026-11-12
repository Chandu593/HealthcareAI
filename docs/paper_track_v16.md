# Paper track v16: 6 h GBDT rerun, label check at FLICS scale, README, abstract and discussion draft

> **Update (v18).** Section 2 below says the paired interval of the label effect was not computed. That is superseded: v17 added `--paired-bootstrap` and v18 reports all five paired intervals, which exclude zero (Table 12). The strict row's difference is +0.0127, not the +0.0128 in the table below (v16 subtracted rounded values).

Applies on top of v15. No model, config default or earlier number changes except the 6 h federated GBDT row (below).

## 1. Sepsis 6 h: the 1,200-round rerun is now the reported 6 h GBDT

The first 6 h run (`fedgbdt_sepsis6h_final`) hit the 600-round limit in all three seeds. You re-ran it with `--set paper.fed_gbdt.max_rounds=1200` (`fedgbdt_sepsis6h_mr1200`, same four-trial search, 208 minutes against 147).

| | first run (600 cap) | rerun (1,200 cap) |
|---|---|---|
| test AUROC, 3 seeds | 0.8566 ± 0.0002 | 0.8574 ± 0.0000 |
| validation AUROC, 3-seed mean | 0.8595 | 0.8603 |
| best round per seed | 600, 592, 596 (cap reached) | 849, 945, 732 (early-stopped) |
| chosen configuration | identical | identical |

Rerun minus first run on identical test patients (n = 22,597, seed 42): +0.0006 [+0.0001, +0.0011], positive in all three seeds. By the validation rule the rerun becomes the 6 h GBDT; the first run stays as a row in Table 4b and the rerun is disclosed in section 5
(it was prompted by seeing the cap in the logs with test results visible). Against the same patients: +0.0184 [+0.0153, +0.0214] over the federated MLP (was +0.0177) and -0.0017 [-0.0030, -0.0004] against pooled LightGBM (was -0.0023).
`reproduce/paired_sepsis6h.py` computes these (same guards as v15: it refuses files that do not hold the same patients, splits and labels).

Documented-sepsis check on the rerun (seed 42): AUROC 0.7796 [0.7717, 0.7873], AUPRC 0.3589 (prevalence 0.137); documented rate by risk quintile 4.5%, 1.9%, 8.5%, 17.7%, 35.8% (the lowest quintile is above the second). Table 8 carries this row; Table 9's 6 h row now shows 732-945 rounds, 1,082-1,390 GB per model and 208 minutes.

## 2. Table 12: what FLICS's label does to our AUROC

Run by you (`scripts/paper_label_variant_check.py --run-dir outputs/paper/<run> --experiment mortality__no_apache_scores --arm <arm> --data-root "%EICU_DATA_ROOT%"`), values copied into `build_tables.py`:

| Test patients | Model | in-hospital label | ICU-discharge label | difference | deaths after ICU discharge |
|---|---|---|---|---|---|
| 3 largest | Federated LR | 0.8553 | 0.8890 | +0.0337 | 55.9% |
| 3 largest | Federated GBDT | 0.8603 | 0.8946 | +0.0343 | 55.9% |
| 10 largest | Federated LR | 0.8528 | 0.8718 | +0.0190 | 47.8% |
| 10 largest | Federated GBDT | 0.8675 | 0.8889 | +0.0214 | 47.8% |
| all strict | Federated GBDT | 0.8742 | 0.8870 | +0.0128 | 44.1% |

Single seed, marginal intervals only; the paired interval of the difference was not computed (it needs the raw tables; a small addition to the script, not done here). The models were trained on in-hospital death. Nothing is applied as a correction.

## 3. Housekeeping

* `analyze_ablation_results.py` now lives in `scripts/` (you moved it; this patch ships the file with its usage line updated, line endings kept).
* `README.md` has a "Paper track (current)" section (commands, rules, rebuild steps) and says the sections below it describe the earlier phase pipeline.
* `Fed-THOR_abstract_and_discussion.md` (draft; every number points to the results text).
* `--resume-existing-run` is untouched, as decided. `configs/base.yaml` was set back to `encoder: lstm` by you (not in this patch).

## 4. Rebuild

```
python scripts/paper_collect_results.py
python Fed-THOR_results_package/reproduce/paired_flicsscale.py
python Fed-THOR_results_package/reproduce/paired_sepsis6h.py
python Fed-THOR_results_package/reproduce/build_tables.py
```
Needs `outputs/paper/fedgbdt_sepsis6h_mr1200` (results.json and predictions/) next to the first run. If it is absent, `build_tables.py` warns that the 6 h GBDT row and cost row are missing.

## 5. Not done

No privacy analysis, no paired interval for the label effect, no second split. The 6 h rerun changes one headline row by +0.0008; the 3 and 10 hospital intervals are still wide.
