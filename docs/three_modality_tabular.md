# Three-modality suite, stage 3b: clean static inputs + pooled tabular autoencoder

**Status (2026-11-04): implemented and sandbox-verified on mock eICU data (8 tests + 3 targeted mutation proofs + a real-CLI smoke chain). NOT yet run on the real export.**
Pooled (central) training, not federated; no per-hospital encoders.

```
python run_three_modality.py --config configs/three_modality.yaml --stage tabular_inputs   # needs a tcd_features run; writes outputs/three_modality/tabular/inputs/
python run_three_modality.py --config configs/three_modality.yaml --stage tabular_ae       # needs tabular_inputs + an incident_label run (same policy); torch required
python run_three_modality.py ... --stage tabular_ae --inspect-cache | --force-recompute
```

## Inputs (`tabular_inputs.py`)
Reuses the paper track's `build_feature_bundle` (window-bounded, allowlisted, vocabularies from TRAINING patients only), not the legacy Phase-5 features. Window = `tcd.window_minutes` after ICU admission with the paper
convention's pre-ICU look-back (1440 min before admission); ALL stays are included (no minimum-stay filter) because every patient needs an embedding; labels are never used (NaN stubs).
Exclusions = the paper's `sepsis_clean` set (imported from `src/paper/config.py`, so it cannot drift) PLUS every `lab` group: `apache_physiology, apache_score, lab_sofa, medication_antibiotic, medication_vasopressor,
infusion_vasopressor, microlab, diagnosis_infection, treatment, medication, infusion, hospital, lab`, the deconfounding name patterns and `apv_activetx`. What remains is admission context: demographics (incl. admission
diagnosis one-hots), APACHE predictor inputs (comorbidity/admission type), non-infection diagnoses, admission drugs. The kept lists are resolved into the run's identity; a forbidden-name check (labs, APACHE scores,
outcome-like names) aborts the run if anything slips through.
**Typed columns** (needed for the reconstruction loss): continuous columns are winsorised (training percentiles), median-imputed and standardised, with deduplicated missing indicators, using the TCD preprocessing code
fitted on TRAINING patients only; binary columns (all training values in {0,1}) stay exactly 0/1, constant or >95%-missing ones are dropped, remaining NaN are imputed to 0 and counted in the report; indicators are binary.
`tab_is_binary.npy` records every column's type. Row order = the TCD run's patient order.
Files: `tab_inputs.npy`, `tab_is_binary.npy`, `feature_schema.json` (name, kind, group), `preprocessing_state.npz/meta.json`, `binary_columns.json`, `patients.csv`, `input_report.json/.md`, `manifest.json`.

## Autoencoder (`tabular_ae.py`)
Existing `TabularAutoencoder` (hidden 128, bottleneck 64, dropout 0.1; configurable) + two linear heads on the bottleneck. Two objectives, separate and separately weighted:
* **unsupervised reconstruction** (`training.reconstruction_weight`, default 1): MSE on continuous columns + BCE-with-logits on binary and missing-indicator columns (weights `reconstruction.continuous_weight/binary_weight`);
* **supervised focal** on the bottleneck (separate alpha/gamma/task weights per task, as for the GRU). Either can be switched off by weight (at least one must remain).
Same population and split discipline as the GRU stage: pooled; `landmark_only` (default true) for updates and early stopping, everyone embedded; TEST labels blanked; validation selects the model; seeded model factory (reproducible).
The log records `val_focal` and `val_aux` (reconstruction) per epoch. Output `tab_ae_embeddings.npz` (`patient_ids`, `embeddings` float32 N x 64), `tab_ae.pt`, `training_log.json`, `label_counts.json`, `manifest.json`.

## Verification
`tests/unit/test_three_modality_tabular.py` (8): typed preprocessing and train-only fitting (corrupting validation/test rows changes nothing in the fit or in the training rows); only the clean groups and no forbidden names, TCD patient order;
events after the window (diagnosis/admission-drug/treatment/medication rows planted at minute 5000) leave the inputs bit-identical; cache/force/tamper/identity; the typed reconstruction loss against hand-computed MSE + BCE; gradients reach
encoder, decoder and both heads; reconstruction actually improves; test labels unreachable; reconstruction-only and supervised-only settings. Mutation proofs: preprocessing fitted on all patients, exclusions not applied, binary columns
reconstructed with MSE; all caught. The shared trainer's mutation proofs (stage 3a) are not repeated.
**Limits:** no real-data audit equivalent to the TCD/GRU poison audit exists for this static set (the paper-track builder is reused as is; the pipeline-level late-rows test uses mock data); the hospital covariates are excluded (as in the paper's clean set).

## Next (needs your machine)
`--stage tabular_inputs`, read `input_report.md` (kept columns by group, dropped by reason, forbidden-name check), then `--stage tabular_ae` and send `training_log.json` + `manifest.json`.
