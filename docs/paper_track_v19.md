# Paper track v19: one base paper (FLICS only), submission-copy script

Applies on top of v18. The second eICU reference study is removed from the research and from the repository; FLICS is the only base paper.

## 1. What was removed

* **Paper-facing:** the results text (both copies: the reference paragraph, the section 2 item on that study, two claims-not-made bullets, a comparability sentence, the reference-choice bullet, Table 3b, Table 5, the figure caption and the revision notes), the abstract and discussion, `tables.md/.tex/.json`, figure 2 (its two dotted reference lines and the footnote; the other three figures regenerate byte-identically), and the constants and tables in `build_tables.py` and `figs.py`.
* **Repository:** the run config for that study (deleted), the reference rows in `src/paper/config.py` (FLICS only now), comments in `src/paper/cohort_filters.py` and `configs/paper_flicsstyle.yaml`, the docs and tracker entries, and the tests that named it.
* **Renamed:** the cohort probe script is now `scripts/paper_hospital_size_probe.py` (the old file must be deleted by hand; same behaviour: it prints the hospital-size table and the threshold that keeps the N largest hospitals; the reference line is gone and `--target-hospitals` defaults to 10).
* **Table numbers:** Table 3a is now Table 3. Tables 3b and 5 no longer exist, so the numbering has gaps (1, 2, 3, 10, 11a-c, 4a-c, 9, 6, 7, 8, 12). Renumber in order of appearance at submission.
* **Reference choice:** the disclosure is kept and is now the only one: FLICS was designated the base paper after the comparison with published figures was known, so the comparison was not pre-specified and no superiority is claimed (section 5 and the limitations paragraph).

## 2. Not touched (your data)

The local run folder of that study under `outputs/paper` and its rows in `outputs/paper/summary_all_runs.md/.csv`. The paper no longer reads them. To drop them from the summary, move the folder out of `outputs/paper` and run `python scripts/paper_collect_results.py` again (commands in the hand-over message).

## 3. `scripts/make_submission_copies.py`

Writes `Fed-THOR_results_section_submission.md`, `Fed-THOR_results_package/results_section_submission.md` and `Fed-THOR_abstract_and_discussion_submission.md`: section 8 cut, the draft-status note and the bracketed pointers (`[Table 12]`, `[Section 5]`, `[item 4]`) removed, " (draft)" dropped from the titles. Sources are never modified; the run stops if a draft marker is still present.
Tests: 5 (`tests/unit/test_make_submission_copies.py`); four injected bugs (pointers kept, section 8 kept, source overwritten, leftover check off) each made a test fail and the restored script passed.

## 4. Rebuild

```
python Fed-THOR_results_package/reproduce/build_tables.py
python Fed-THOR_results_package/reproduce/figs.py
```
(figs.py only changes figure 2.) Then run the full suite once more, rename `cache` and `outputs` first and restore them afterwards.
