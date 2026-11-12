# Paper track v17: paired interval for the label effect (script option only)

Applies on top of v16. One additive, default-off option; the printed output and every number elsewhere are unchanged unless the option is used.

`scripts/paper_label_variant_check.py --paired-bootstrap N` adds one line: the AUROC difference between the two death labels (ICU-discharge label minus in-hospital label) with a paired bootstrap interval over the test patients.
One resample of the test patients feeds both labels, so the interval is for the difference itself. (Subtracting the two marginal intervals would ignore that the two AUROCs share the same patients and the same score.)
Resamples in which either label has a single class are skipped; if none are usable, the result is NaN and no error is raised.

```
python scripts/paper_label_variant_check.py --run-dir outputs/paper/flicsstyle_top3 --experiment mortality__no_apache_scores --arm fedavg_logreg --data-root "%EICU_DATA_ROOT%" --paired-bootstrap 2000
```

Tests (`tests/unit/test_paper_label_variant_check.py`, 5 new): off by default and asking for it changes nothing else; the point is exactly the difference of the two AUROCs and the interval is deterministic; identical labels give exactly zero (this needs one shared resample);
a label with no deaths returns NaN; the CLI prints the new line only when asked and the four original lines are identical either way. Three bugs were injected (independent resamples, flipped sign, option leaking into the default result); each made its test fail and the restored script passed.

Table 12 and the section 5 text are not changed by this patch: they say the paired interval was not computed, which stays true until the five runs are made and the numbers are added.
