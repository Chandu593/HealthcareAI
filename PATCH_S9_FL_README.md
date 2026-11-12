# patch_s9_fl_configs1to4 (2026-11-10) -- federated downstream learning, Configs 1-4 + personalisation

Additive. Extract over the repository root (the same way as s6/s7/s8). It creates 9 modules, 1 doc, 5 test files and modifies two existing files:

* `run_three_modality.py` -- new stages `fl_tune`, `fl_config1..4`, `fl_personalized`, `fl_report` and their flags (the only removed lines are the usage string and the STAGES tuple, which were extended).
* `README.md` -- one paragraph + pointer to `docs/three_modality_fl.md` (if you edited README.md since the 2026-11-09 zip, re-apply that paragraph by hand instead of overwriting).

Nothing under `src/` other than the new `fl_*.py` / `federated.py` is touched; `src/paper/fed_gbdt.py` is imported unchanged; no existing run, checkpoint, embedding, label, config or output is modified.
Sandbox-verified only (synthetic upstream runs). NOT run on the real data. See `docs/three_modality_fl.md`.
