# Paper track v18: paired label effect in Table 12, and the final wording of the FLICS comparison and the limitations

Applies on top of v17. Text and table literals only; no new code logic and no new tests.

## 1. Paired intervals for the label effect (you ran the five commands with `--paired-bootstrap 2000`, seed 42)

| Test patients | Model | in-hospital label | ICU-discharge label | difference [95% paired CI] |
|---|---|---|---|---|
| 3 largest hospitals | Federated LR | 0.8553 | 0.8890 | +0.0337 [+0.0069, +0.0623] |
| 3 largest hospitals | Federated GBDT | 0.8603 | 0.8946 | +0.0343 [+0.0089, +0.0617] |
| 10 largest hospitals | Federated LR | 0.8528 | 0.8718 | +0.0190 [+0.0030, +0.0362] |
| 10 largest hospitals | Federated GBDT | 0.8675 | 0.8889 | +0.0214 [+0.0062, +0.0365] |
| All strict | Federated GBDT | 0.8742 | 0.8870 | +0.0127 [+0.0055, +0.0203] |

Every interval excludes zero, so the label effect is measured; the lower ends are small (0.003 to 0.009), so its size is uncertain. For the 3-hospital LR the point estimate (+0.034) is more than twice the gap between our seed-42 estimate (0.855) and FLICS's reported 0.87 (0.015),
but the lower end of its interval (+0.007) is below that gap; the text says so. One seed, models trained on in-hospital death. Not applied as a correction.

Table 12 is built from these values in `build_tables.py` (copied from the printed output; checked cell by cell against it). The strict row's difference is the script's +0.0127 (v16 showed +0.0128 from subtracting rounded values; a note was added to the v16 doc).

## 2. Text

* Results text: section 5 label paragraph rewritten around the measured intervals; section 2 item 3 and the section 4 claim bullet adjusted; Table 12 replaced; v18 note in section 8.
* `Fed-THOR_abstract_and_discussion.md`: final wording of the FLICS comparison (no "ahead" or "above"; "within the uncertainty of FLICS's 3-hospital figure", "numerically higher than its 10-hospital figures, by 0.026 to 0.036, smaller than FLICS's own interval"; the label sentence says the label can account for differences of this size, not that it does)
  and of the limitations (one paragraph under three headings: design and data; outcome definitions and references; selection, privacy and cost). One sentence added to the abstract (label effect 0.013 to 0.034, paired intervals exclude zero); the conclusion now names training on the ICU-discharge label as the open question.

## 3. Rebuild

```
python Fed-THOR_results_package/reproduce/build_tables.py
```
Nothing else needs re-running (the two paired scripts are unchanged since v16).

## 4. Before submission

Run the full suite once (rename `cache` and `outputs` first and restore them), remove section 8 from both copies of the results text, and remove the draft note and the bracketed table pointers from the abstract file. Check the data use agreement before releasing any row-level prediction file.
