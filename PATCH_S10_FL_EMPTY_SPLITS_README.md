# patch_s10_fl_empty_splits (2026-11-14) -- fix for hospitals with an empty split (apply AFTER s9)

Found on the real data (`fl_config4`, run `flcfg4_v1`): `ValueError: Found array with 0 sample(s) (shape=(0, 128)) ... StandardScaler` after the stage-1 GCN had finished (a hospital with zero test patients).
Root cause + a second, related bug found while reproducing it:
1. `FLClient.scaled()` sent an EMPTY split through sklearn's StandardScaler (crash in Config 4's embedding step, the only path that scores the test split of every client).
2. A hospital with ZERO VALIDATION patients was silently dropped from the federation for every family (LR / MLP / GCN) with the misleading reason "hospital-local graph construction failed", because
   `make_problems` swallowed the constructor error. It is now kept (it trains; evaluated with the global federated-histogram threshold), and only the GCN family may drop a client (genuine graph failure, reported).
Also: the query graph of an empty split is the training graph (no attach call).

Files: `src/three_modality/fl_clients.py`, `src/three_modality/fl_algorithms.py` (modified), `docs/three_modality_fl.md` (modified), 2 new test files. Run identity is unchanged (`fl-v1`), so an interrupted run resumes:
repeat the exact command with `--resume-existing-run` and the same `--run-id` -- the finished stage-1 GCN checkpoint is reloaded, not retrained.
