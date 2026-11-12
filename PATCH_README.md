# Three-modality suite, stage 3b patch (s5) -- 2026-11-04: clean static inputs + pooled tabular autoencoder

Apply by extracting over the repo root, on top of v21 and patches s1-s4.
NEW: src/three_modality/{tabular_inputs,tabular_ae}.py, tests/unit/test_three_modality_tabular.py, docs/three_modality_tabular.md
CHANGED: src/three_modality/config.py (+ tabular block and validators; nothing existing altered), run_three_modality.py (+ stages tabular_inputs, tabular_ae),
         configs/three_modality.yaml, IMPLEMENTATION_TRACKER.md, README.md.  No other existing file is touched.
Inputs reuse the paper track's feature bundle with its clean-set exclusions + every lab group (no APACHE scores, no labs). Pooled training, not federated.

Run (TCD features and incident label are cache hits):
  python run_three_modality.py --config configs/three_modality.yaml --stage tabular_inputs
  python run_three_modality.py --config configs/three_modality.yaml --stage tabular_ae
Send: outputs/three_modality/tabular/inputs/<run>/input_report.md, then outputs/three_modality/tabular/encoder/<run>/{training_log,manifest}.json
Tests: python -m pytest tests/unit/test_three_modality_tabular.py
