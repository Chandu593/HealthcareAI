# Paper track v21: pooled-GBDT results in the text, and the final comparison in FLICS's format (Table 14)

Applies on top of v20. Text, tables and one figure only: no model, config default, pipeline or earlier number changes. Everything below was computed from files that already exist (`results.json`, `paired_central_gbdt.json`); nothing was retrained.

## 1. What changed

* **The six pooled-GBDT runs (v20) are now in the text.** Central-GBDT rows in Tables 1, 2, 4a and 4b, Table 13, a new setup bullet, rewritten items 3 and 5 of section 2, a sentence in section 3, three entries in section 4, two paragraphs in section 5, and matching edits in the abstract, discussion, limitations and conclusion.
* **Table 14a-14f (new): the final comparison in the format of FLICS's Tables IV and V.** Accuracy, AUROC and F1 for the federated models (LR, MLP, GBDT) and the central models (LR, MLP, LightGBM, GBDT), in six settings: strict, all stays, the 3 and the 10 largest hospitals, sepsis 24 h and 6 h.
  * FLICS marks the best federated result per metric in **bold** and its non-private oracle (centralized XGBoost) with an underline. Here: bold = highest mean among the federated models, underline = highest mean among the four central models, per metric, ties marked on every tied row (ties are decided on the printed four decimals).
  * Only our own rows are ranked. In the two FLICS-scale tables (14c, 14d) FLICS's own reported rows (FedAvg LR, FedAvg LSTM, centralized LR, centralized XGBoost; local models omitted) are added underneath, unmarked.
  * There is no federated LightGBM (the federated tree model is the histogram GBDT), so LightGBM appears as a central model only.
  * Cells: accuracy and F1 as mean ± sd over three seeds, AUROC as mean ± sd with the seed-42 bootstrap interval (as in the other tables). Central LR has one seed and says "(1 seed)".
* **`reproduce/final_comparison.py` (new):** reads `outputs/paper/<run>/results.json` only (per-seed test accuracy, F1 and AUROC were already stored there). `build_tables.py` calls it and converts the marks to `\textbf` / `\underline` in `tables.tex`. All earlier tables are byte-identical to the v20 build (checked).
* **Figure 2:** a Central GBDT bar was added; the FLICS reference lines stay on both panels (author's decision), and the footnote says that on the all-stays panel they are context only.
* **`make_submission_copies.py`** was re-run, so the three `*_submission.md` files match the sources.

## 2. How accuracy and F1 are computed (and how this differs from FLICS)

Both are TEST values at the decision threshold that maximises F1 on the VALIDATION set (the project's existing rule, stored in every `results.json` record). FLICS tunes its threshold on its test set. The two procedures are therefore not the same, and ours cannot score a higher F1 than a test-tuned threshold would on the same predictions. A test-tuned variant was not computed (it would break the validation-only rule); it could be added as a clearly labelled sensitivity analysis if wanted.

## 3. What the numbers show (all from the generated tables)

* The federated GBDT is the best federated model on accuracy, AUROC and F1 in all six settings. Pooled LightGBM is the best central model on AUROC in all six; the pooled GBDT takes the accuracy mark in the strict and 3-hospital settings and the F1 mark at 10 hospitals. The marks follow point estimates; accuracy differences between leading models are mostly inside the seed sd.
* **Our F1 and accuracy are below FLICS's reported values.** Strict: federated GBDT accuracy 0.8888, F1 0.4722. At FLICS's scales: F1 0.4386 (3 hospitals) and 0.4704 (10) against FLICS's 0.59-0.65, accuracy 0.8833 and 0.8864 against 0.88-0.91, with AUROC level with FLICS's. The text states the reasons that can be checked (prevalence 9.1-9.7% against 12.5-17.0%, label, threshold procedure; on the strict cohort a classifier that predicts nobody positive has accuracy 0.908, above every accuracy in Table 14a) and draws no conclusion. The size of each reason was not measured.
* FLICS's centralized XGBoost (AUC 0.90 at 3 hospitals, 0.88 at 10) is nominally above our pooled LightGBM (0.8740, 0.8704) and pooled GBDT (0.8692, 0.8670) at the same scales, with widely overlapping intervals and a different label. Earlier tables compared only against FLICS's federated figures; this row is new in the paper and is reported in both directions of caution (section 4, section 2 item 9).
* Pooled-GBDT control (Table 13): federated minus pooled GBDT is between -0.0002 and +0.0006 in five of six settings (intervals include zero) and -0.0102 [-0.0187, -0.0017] at seed 42 for the 3 largest hospitals (seed-wise -0.0102, +0.0020, -0.0087; mean -0.0056; pooled seeds 0.8622 to 0.8747; 177 events). The text neither claims nor rules out a federation cost at 3 hospitals. The pooled GBDT reuses the federated winner's hyperparameters, which if anything favours the federated arm.
* Table 13's last column (pooled GBDT minus pooled LightGBM) is seed 42, paired. Earlier text quotes three-seed means for the federated GBDT against LightGBM (for example -0.0050 strict). The text says which it uses.

## 4. Verification

* 19 new unit tests (`tests/unit/test_paper_final_comparison.py`): mark placement per metric, central models never take the federated mark (and the reverse), ties and the rounding rule, population sd (as `results.json`'s own aggregate), one-seed formatting, interval taken from seed 42, FLICS rows only at the two FLICS scales and never marked, transcription of FLICS's Tables IV and V, missing optional arm (warning) versus missing required arm (error), arms scored on different test sets (refused), a missing base run (block skipped with a warning), and the LaTeX conversion.
* **Revert, fail, restore, pass, four times:** marking the lowest mean instead of the highest (6 tests failed); pooling federated and central models when choosing the bold mark (4 failed); sample sd instead of the population sd (1 failed); taking the interval from the last seed instead of seed 42 (1 failed). Each time the file was restored (byte-compared) and all 19 passed.
* The 52 tests of the reproduce/package area that can run without torch or LightGBM pass (`test_paper_flicsscale`, `test_make_submission_copies`, `test_paper_sepsis6h`, `test_paper_central_gbdt_paired`, the new file). The full suite was NOT run in the sandbox (no torch); run it on your machine (expected: v20's count plus 19 new tests).
* Every number quoted in the new text was checked by script against the generated table cells and the result files (accuracy and F1 cells, interval ends, Table 13 values, per-seed differences, prevalence and all-negative accuracy, which model holds each mark).
* Rebuilding `build_tables.py` from the unmodified v20 files reproduces the shipped `tables.md`, `tables.tex` and `tables.json` (modulo line endings), so the only differences after v21 are the new blocks.

## 5. Apply and rebuild

Unzip over the repo root, then `python Fed-THOR_results_package/reproduce/build_tables.py` and `python Fed-THOR_results_package/reproduce/figs.py` (both are idempotent; the shipped `tables/` and `figures/` are already rebuilt). `python scripts/make_submission_copies.py` after any further text edit. Quick check: `python -m pytest -q tests/unit/test_paper_final_comparison.py`.

## 6. Open points (not decided here)

* Table numbers still have gaps (1, 2, 3, 10, 11a-c, 4a-c, 9, 6, 7, 8, 12, 13, 14a-f); renumber at submission.
* Table 14 has no interval for accuracy or F1 (FLICS gives one for accuracy). It would need a bootstrap over the saved predictions: a new script, not added.
* A test-tuned-threshold F1 (FLICS's procedure) as a labelled sensitivity analysis: not computed.
* Table 12's values are still transcribed from printed output; Table 4c is hand-maintained.
* The `outputs/paper/*/paper_report.md` files of runs made before v19 still contain the dropped reference rows (v19 left run folders untouched); regenerate or exclude them before sharing outputs.
