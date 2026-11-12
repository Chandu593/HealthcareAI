# Non-FL hospital-wise Configs 1-4 (stages `nonfl_config1..4`, `nonfl_report`)

**Regime.** One independent model per hospital and task, from that hospital's own train / val / test patients (permanent split). No parameter sharing, no aggregation, no pooled
patient-level classification, no shared scaler or graph. The encoders and the three-token cross-attention are centrally pretrained, so this is hospital-wise **downstream** learning, not
end-to-end learning; no federated claim is made.

**Tasks / eligibility.** Mortality: strict 24 h landmark with a known hospital-discharge status. Incident sepsis: incident cohort (hours 24-96 proxy, `alive_discharge_as_negative`) inside the
landmark. Labels, masks, models and metrics are kept separate per task.

**Skip rules** (`three_modality.downstream.min_samples`, defaults train >= 20, train positives >= 3, train negatives >= 3, val positives >= 1, val negatives >= 1). A hospital below them is
reported as skipped with every failing criterion; an unexpected error is recorded as `failed` and counted, never swallowed. A hospital with one class in its TEST set is trained; AUROC / AUPRC
(and any other undefined metric) are `None` and excluded -- with a count -- from the averages. Hospitals with < 5 test patients in a class are flagged low-power.

| Config | Method |
|---|---|
| 1 | embedding set (`fused`, or `gru` / `tabular` / `tcd` for the individual-modality comparison) + LR, MLP (focal, torch), LightGBM (custom focal objective, analytic gradient/Hessian validated against finite differences, Hessian clipped to >= 1e-6), GradientBoosting (sklearn, **standard deviance loss, no focal / class weighting**) |
| 2 | hospital-local Graph A (kNN + cosine threshold) / Graph B (MiniBatchKMeans candidates, same rule inside clusters) + 2-layer GCN, focal loss, no sampling |
| 3 | Config 2 + hospital-local graph-compatible SMOTE |
| 4 | the Config-2 GCN's hospital-local node embeddings (Graph A / B) + the four classifiers (one GCN per hospital/task/graph feeds all four classifiers) |

**Graph design.** Train graph from TRAINING patients only (repo `build_hospital_graph`, identical edges). Validation and test patients are attached separately and only to training patients
(directed train -> query edges, top-k, cosine >= threshold, same cluster for Graph B), so a training node's representation is identical with or without query nodes and no query patient depends
on another validation/test patient. **Graph SMOTE:** synthetic positives interpolate between a training positive and one of its k nearest training positives of the same hospital; wired by
directed edges FROM real training nodes only; no synthetic -> real or synthetic-synthetic edges, no adjacency interpolation; hospitals with fewer than 2 training positives (or already balanced)
are reported. With SMOTE the focal alpha is neutral (0.5) so classes are not re-weighted twice. Config 4's classifiers are fitted on train-node embeddings of a GCN trained on the same train
labels (standard frozen-encoder stack; validation and test embeddings come from inductive attachment).

**Selection.** Scaling, graph, SMOTE, checkpoint / boosting stage and the F1-maximising threshold all use only the hospital's training (and, for selection, validation) rows; test labels are read
once, after everything is fixed.

**Outputs** (`outputs/three_modality/non_federated/<run_id>/`): `per_hospital.json` (counts, class distribution, threshold, all metrics, confusion matrix, per-class report, info, skip reason),
`per_hospital.csv`, `summary.json` (macro and sample-weighted averages, trained / skipped / failed counts, skip-reason counts), `manifest.json`. `nonfl_report` writes
`outputs/three_modality/reports/<id>/` (`summary_by_variant.csv`, `skipped_hospitals.csv`, `comparisons.json`, `report.md`): Graph A vs B, sampling vs none, individual vs fused, with paired
hospital differences and bootstrap intervals. Results are **not directly comparable** with pooled / federated / base-paper numbers.

```
python run_three_modality.py --config configs/three_modality.yaml --stage nonfl_config1 --cross-attention-run-id <CA_RUN> --embedding-set all
python run_three_modality.py --config configs/three_modality.yaml --stage nonfl_config2 --cross-attention-run-id <CA_RUN>
python run_three_modality.py --config configs/three_modality.yaml --stage nonfl_config3 --cross-attention-run-id <CA_RUN>
python run_three_modality.py --config configs/three_modality.yaml --stage nonfl_config4 --cross-attention-run-id <CA_RUN>
python run_three_modality.py --config configs/three_modality.yaml --stage nonfl_report  --cross-attention-run-id <CA_RUN>
```
Add `--max-hospitals 5 --tasks mortality` for a quick smoke run (marked partial, excluded from the report). FL (Configs 1-4) is not implemented until these results are validated on the real data.

## Post-run fixes (nonfl-v2)
Triggered by the first real runs (`xattn_real_001`, `nonfl_config{1..4}_complete`):
* **LightGBM focal objective stabilised** -- the true focal Hessian is ~0/negative for confidently-wrong samples; the old 1e-6 floor produced raw scores up to hundreds (overflow warnings,
  thresholds > 0.95 in 21% of hospitals, best iteration <= 2 in 29%). Now floored at `hessian_floor` x class weight with `lambda_l2`, `max_delta_step`, `min_sum_hessian_in_leaf`; overflow-free
  sigmoid everywhere. On a high-dimensional low-prevalence synthetic fixture the old settings reach |raw| ~ 260 and stop at <= 2 trees in 17% of runs; the new ones stay < 4 with none.
* **Checkpoint selection** can use `selection_metric` = `loss` | `auroc` | `auprc` and a `min_epochs` floor (LR, MLP, GCN, LightGBM, GBDT), so a near-initial model cannot be chosen on a tiny, noisy
  validation set. Defaults keep the previous behaviour; the tuning stage decides.
* **`nonfl_tune`** -- validation-only hyper-parameter tuning (see `src/three_modality/hw_tune.py`): recipes are fitted on 70% of each tuning hospital's TRAINING patients, checkpointed on its
  validation patients and scored on the held-out 30% of train, so the ranking is not biased toward recipes that select on validation AUROC. Test data is never loaded (proven by tests).
  One set per family shared by all hospitals; recipe 0 is always the current defaults. Apply with `--hparams-run-id`.
* **Saved patient-level test predictions** (`predictions.npz`) -> pooled AUROC / AUPRC / F1 / recall / accuracy with patient-bootstrap intervals for all trained hospitals and for the 3 / 10
  largest ("FLICS-style"), in the report.
* **Secondary operating point**: highest validation threshold reaching `operating_points.recall_target` (0.65), reported next to the F1-optimal one.
* **Per-class classification reports** for every config / task / variant: `classification_reports.{md,json}` per run and consolidated in the report, `per_hospital_classification_reports.csv`.
  Pooled (counts summed over hospital models) and hospital-macro versions, undefined values excluded and counted.

```
python run_three_modality.py --config configs/three_modality.yaml --stage nonfl_tune --cross-attention-run-id xattn_real_001 --run-id tune_real_001
python run_three_modality.py --config configs/three_modality.yaml --stage nonfl_config1 --cross-attention-run-id xattn_real_001 --embedding-set all --hparams-run-id tune_real_001 --run-id nonfl_config1_v2
```
See `docs/three_modality_targets.md` for what the requested AUROC / F1 / recall / accuracy targets would require.

### Tuning robustness (added after the first real `nonfl_tune` attempt)
The first real tuning run died on LightGBM's fatal `num_features() > 0` check: a recipe with `min_child_samples=20` on a tiny hospital's ~14-row fitting slice left no splittable feature.
Fixed in three ways: (1) the LightGBM leaf minimum is capped at a quarter of the training rows and the feature pre-filter is off, so an unsplittable tree returns the prior instead of
crashing (`min_data_in_leaf_effective` is recorded); (2) a hospital that errors inside a tuning recipe is counted (`n_failed`, `first_error` in `tuning_table.csv`) instead of aborting, and a
recipe that errors on more than 10% of its hospital-tasks is disqualified; (3) every finished recipe is cached in `tuning/_partial_<identity>.json`, so re-running the identical command
resumes after a crash or Ctrl-C and the file is deleted on success.
