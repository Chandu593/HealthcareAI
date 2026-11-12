# Paper track v10: federated histogram GBDT and the hospital-size cohort filter

Everything here is **additive and off by default**: no existing arm, config key, split or number changes. Nothing in this patch has been run on the
real eICU data yet (sandbox-verified on mock data and synthetic arrays only; see "What is verified").

## 1. Why these two additions, and what to expect

Strict 24 h landmark cohort, 3 seeds, fixed split (existing results):

| arm | test AUROC |
|---|---|
| federated MLP (FedAvg + FedProx) | 0.8617 |
| central MLP | 0.8691 |
| central LightGBM | 0.8796 |

Federation costs about 0.007 (federated vs central MLP). The larger gap, about 0.011 more, is the model family: tabular nets trail gradient-boosted
trees here. A federated tree ensemble attacks that gap without pooling any raw row.

**Honest expectation.** Federated histogram boosting reproduces a *pooled* histogram GBDT (Section 2), so its ceiling is pooled-GBDT performance, i.e.
roughly 0.88 on strict, not more. A from-scratch depth-wise GBDT will usually land a little below tuned LightGBM, so the realistic range is between the
federated MLP and pooled LightGBM. That could put the strict result level with or marginally above FLICS's 3-hospital 0.87. No outcome is promised.

## 2. `fed_gbdt`: federated histogram gradient boosting (`src/paper/fed_gbdt.py`)

Horizontal federated learning, hospital = client, the same family of protocol as federated XGBoost/FedTree. numpy only; no new dependency.

1. **Bin edges.** Each client sends per-feature quantiles of its own *train* rows; the server merges them, weighted by non-missing count, into global edges.
2. **Every tree level.** Each client computes (sum of gradients, sum of hessians) per (frontier node, feature, bin) on its own train rows and sends them;
   the server sums them, picks the best split per node (with a learned default direction for missing values) and broadcasts it. Leaf values come from summed (G, H).
3. **Selection.** Early stopping and the trial search use validation AUROC only, the same rule as `lightgbm_central`. Trial 0 is `paper.fed_gbdt.fixed`; the
   other `n_trials - 1` are random draws from `paper.fed_gbdt.search`. The search runs on the first seed only; later seeds re-use the winner.

**Exactness.** Summing client histograms equals the histogram of the pooled rows, so the trees equal those of a pooled histogram GBDT with the same bin edges.
This is tested (`tests/unit/test_paper_fed_gbdt.py`): re-partitioning the same train rows over 1, 5 or 60 clients gives identical predictions, and the literal
per-client message path (`paper.fed_gbdt.explicit_client_histograms=true`) matches the default aggregate path. The default skips materialising the K per-client
slabs because their sum is what the server uses.

**Leakage rules (same as the rest of the track).** Bin edges, base score, class counts and every split use train rows only. Validation rows decide *which round is
kept* and which trial wins, never the content of a tree (tested by scrambling val/test features and val labels). Test rows are only binned with train edges and scored.

**What is not claimed.**
- No privacy guarantee. Gradient histograms disclose more than a weight average, and there is no differential privacy or secure aggregation (same boundary as every other arm).
- Validation AUROC is pooled at the server, the same simulation shortcut `run_fedavg` uses. A deployment would aggregate a validation loss instead.
- **Communication is much larger than FedAvg-MLP.** Each tree level ships histograms. `results.json` reports `comm_mb_per_client_per_round_dense` and `..._sparse`
  (sparse = only cells a client touched, at most active-rows x selected-features per level) and `comm_gb_total_sparse_all_clients`. Report these next to the AUROC; they are the
  honest cost of the accuracy.
- Hyperparameter search is small (4 trials by default) and depth-wise growth is not leaf-wise; do not describe this arm as "LightGBM".

**Measured here (sandbox, synthetic arrays, not your data):** about 2.5 s per round at 92k train rows x 699 features x 204 clients, 1.2 GB peak memory, about 40 s once
for the quantile exchange and 8 s to bin all rows. At that speed one fit of about 300 rounds is on the order of 12 minutes; the default search (4 trials) plus 2 more seeds is
roughly 1.2 hours. Your machine will differ; check the first round's log line `[FED_GBDT] round 25 ...` for the real rate.

```
python run_paper.py --config configs/paper_fedgbdt.yaml --reuse-phase1to3 --run-id fedgbdt_strict --seeds 42 --set paper.fed_gbdt.n_trials=1
python run_paper.py --config configs/paper_fedgbdt.yaml --reuse-phase1to3 --run-id fedgbdt_strict_final
```

`configs/paper_fedgbdt.yaml` keeps the window/feature/sequence settings identical to the final strict run so its cached feature bundle is re-used (the cache key hashes those
stanzas; verified by comparison). Run one job at a time on the 7.7 GB machine.

## 3. Hospital-size cohort filter and probe

`paper.cohort_filter.min_hospital_patients` (default 0 = off) restricts **every arm** (federated, central, local) to hospitals with at least N cohort patients. Sizes are counted over the whole cohort and use no label,
so the filter only decides which sites exist. It is what the FLICS-scale federations use (the 3 or the 10 largest hospitals; `configs/paper_flicsstyle.yaml`). The report gives pooled AUROC and per-hospital macro AUROC.

**Choosing the threshold.** Run the probe first; it is read-only:

```
python scripts/paper_hospital_size_probe.py --config configs/paper_fedgbdt.yaml --target-hospitals 3
python run_paper.py --config configs/paper_flicsstyle.yaml --reuse-phase1to3 --run-id flicsstyle_top3 --set paper.cohort_filter.min_hospital_patients=3758
```

The federated flags in that config are the ones frozen on the strict cohort; they are not re-tuned here. If you want to tune on a filtered cohort, do it on validation and say so.

## 4. Reference table (`reference_results` in `src/paper/config.py`) -- final state after v11

FLICS (3- and 10-hospital figures; the base paper) is the only reference: a paper whose experiments use eICU alone. DP-IG-FL (MIMIC-III + eICU
combined) and Kerkouche et al. (Premier Healthcare Database) were removed. `tests/unit/test_paper_cohort_filters.py` fails if a non-eICU paper is ever added back or labelled "(eICU)".
The tables and figures under `Fed-THOR_results_package/` were regenerated from the final runs (v11).

## 5. What is verified

| item | how |
|---|---|
| histogram aggregation, partition invariance, train-only fitting, missing-value routing, early stopping, search | 14 unit tests (`test_paper_fed_gbdt.py`), numpy/sklearn only |
| size filter, profile, config wiring, reference rows | 9 unit tests (`test_paper_cohort_filters.py`) |
| arm + filter + probe through `run_paper_track` on mock eICU | 4 integration tests (`test_paper_fed_gbdt_and_size_filter.py`) |
| each safety property catches its bug | hand-performed revert -> fail -> restore -> pass (byte-diffed): bin edges from all rows; histograms averaged instead of summed; missing always left; strict `>` threshold; runner ignoring `cohort_filter` |
| no behaviour change to existing paper track | 44 existing paper tests run on the original zip and on the patched tree under identical conditions: identical result (37 pass; the same 7 that need real torch fail in both) |

**Not verified:** anything on the real 199,108-stay data; the torch-dependent tests (this sandbox has no torch, so they were compared original-vs-patched under a stand-in, not run); wall-clock on your machine.
Please run the full suite (`python -m pytest -q`) and expect the same 2 pre-existing failures (`test_modality_architecture_options`, from `base.yaml`'s `encoder: transformer`).
