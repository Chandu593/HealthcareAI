# Paper track v20: a pooled run of our own GBDT (`gbdt_central`), in all six settings

Applies on top of v19. Additive and default-off: nothing changes unless the arm is asked for, and no existing number or table changes until the runs exist.

## 1. Why

The paper has pooled LR, a central MLP and pooled LightGBM, but no pooled run of our own histogram GBDT. By construction the federated GBDT grows the same trees as a pooled histogram GBDT (same bin edges, any split of the rows across clients; unit-tested),
so a pooled run should reproduce it. Running it shows that empirically and separates two things that are mixed in the gap to pooled LightGBM (0.002 to 0.005 AUROC, 0.014 at three hospitals): federation (expected to cost about nothing for this model)
and our simpler implementation and search (the rest). Whatever the runs give is reported.

## 2. What was added

* **Arm `gbdt_central`** (`src/paper/runner.py`): the same `search_and_train` code on ONE client that holds every training row (`{0: train rows}`). With `paper.gbdt_central.reuse_params_from=<finished federated run>` it reuses that run's winning hyperparameters (first-seed `fed_gbdt` record),
  so the two differ only in not being federated (the federated model merges bin edges from client summaries; the pooled one summarises once). With no source it runs its own validation search. It refuses to run if it does not train on exactly one client holding exactly the training rows.
* `reuse_params_from_run()` (`src/paper/fed_gbdt.py`) raises if the run, the experiment, the arm or any parameter is missing; it never falls back to a search silently.
* **`reproduce/paired_central_gbdt.py`**: for each setting, federated GBDT, pooled GBDT and pooled LightGBM on identical test patients (seed 42, 2,000 paired bootstrap resamples; three seeds for means and the stability count). It refuses files that hold different patients, splits or labels, and runs whose hyperparameters differ;
  a setting whose pooled run does not exist yet is skipped and listed.
* **`build_tables.py`**: a "Central GBDT (same code, pooled)" row in the strict, all-stays, sepsis 24 h and sepsis 6 h tables once the runs are collected (a WARNING names any that are missing), and **Table 13** (federation cost for the GBDT, six settings) once `paired_central_gbdt.json` exists.
* Central MLP already exists in every setting (strict, all stays, sepsis 24 h and 6 h, 3 and 10 hospitals); nothing was added for it.

## 3. The six runs (you, one at a time; same config and overrides as the federated run, plus the arm and the parameter source)

```
python run_paper.py --config configs/paper_fedgbdt.yaml --reuse-phase1to3 --arms gbdt_central --run-id gbdtcentral_strict --set paper.gbdt_central.reuse_params_from=outputs/paper/fedgbdt_strict_final
python run_paper.py --config configs/paper_fedgbdt.yaml --reuse-phase1to3 --arms gbdt_central --run-id gbdtcentral_allstays --set paper.window.min_icu_los_minutes=0 --set paper.gbdt_central.reuse_params_from=outputs/paper/fedgbdt_allstays_final
python run_paper.py --config configs/paper_sepsis.yaml --reuse-phase1to3 --tasks sepsis --feature-sets sepsis_clean --arms gbdt_central --seeds 42,43,44 --set paper.window.min_icu_los_minutes=1440 --set paper.gbdt_central.reuse_params_from=outputs/paper/fedgbdt_sepsis24_final --run-id gbdtcentral_sepsis24
python run_paper.py --config configs/paper_sepsis_early.yaml --reuse-phase1to3 --tasks sepsis --feature-sets sepsis_clean --arms gbdt_central --seeds 42,43,44 --set paper.window.min_icu_los_minutes=360 --set paper.fed_gbdt.max_rounds=1200 --set paper.gbdt_central.reuse_params_from=outputs/paper/fedgbdt_sepsis6h_mr1200 --run-id gbdtcentral_sepsis6h
python run_paper.py --config configs/paper_flicsstyle.yaml --reuse-phase1to3 --arms gbdt_central --run-id gbdtcentral_top3 --set paper.cohort_filter.min_hospital_patients=3758 --set paper.gbdt_central.reuse_params_from=outputs/paper/flicsstyle_top3
python run_paper.py --config configs/paper_flicsstyle.yaml --reuse-phase1to3 --arms gbdt_central --run-id gbdtcentral_top10 --set paper.cohort_filter.min_hospital_patients=2699 --set paper.gbdt_central.reuse_params_from=outputs/paper/flicsstyle_top10
```

Checks for every run: the first log lines must say `reusing cached feature bundle` with the same key as the matching federated run (strict, sepsis 24 h and the two FLICS-scale runs e49025..., sepsis 6 h c92ebe..., all stays da5552...); if one says it is building a bundle, stop it (Ctrl+C) and check the config and `--set` values against the command you used for the federated run.
Use the all-stays command that matches how you launched `fedgbdt_allstays_final` (config and window override); the run folder names above are what the table code looks for.

Then:
```
python scripts/paper_collect_results.py
python Fed-THOR_results_package/reproduce/paired_central_gbdt.py
python Fed-THOR_results_package/reproduce/build_tables.py
```
`paired_central_gbdt.py` takes a few minutes. Send me the printed output and the six `results.json` files (the predictions folders stay on your machine unless asked for) and I will update the text.

## 4. Tests

`tests/unit/test_paper_gbdt_central.py` (5) and `tests/unit/test_paper_central_gbdt_paired.py` (7), plus 3 integration tests in `tests/integration/test_paper_fed_gbdt_and_size_filter.py` (off by default; one client, no search, the federated winner's parameters, the very same patients and splits; a missing parameter source is an error).
Eight injected bugs (silent fallback, hospital clients instead of one, ignoring the federated winner, on by default, different hyperparameters accepted, patient check removed, a missing run treated as present, no pooled run not an error) each made a test fail; the restored code passed.
The integration tests were run here with a stand-in for torch (the arm uses no torch); they run for real in your full suite.
