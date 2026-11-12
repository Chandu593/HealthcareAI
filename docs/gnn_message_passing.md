# Graph Neural Network (Phase 9)

**Status: implemented, sandbox-verified.** `src/gnn/patient_gnn.py` is
fully tested (11 unit tests) including a direct regression test for
the differentiated edge_weight design (see below) and a hand-performed
revert→fail→restore→pass proof. The full
Phase 1→2→3→4→5→7→8→9 pipeline is verified end-to-end against
synthetic data (4 integration tests), covering all three architectures.
**Not yet run against the real dataset** -- immediate next step once
delivered.

## What Phase 9 does

Consumes Phase 8's per-hospital graphs (topology + edge weights) and
Phase 7's fused embeddings (node features, `(N, 192)`), and produces
refined per-patient embeddings via message passing --
`configs/base.yaml`'s `gnn:` stub is now actually implemented, not
just a placeholder. Architecture + forward-pass verification only, no
training loop yet (weights are randomly initialized) -- same pattern
as Phase 4's LSTM and Phase 5's autoencoder.

Output: `outputs/embeddings/gnn_embeddings.npz` (`patient_ids`,
`embeddings` at `gnn.hidden_dim`, default 128). Patient order is
grouped by hospital (the manifest's own iteration order), not
necessarily Phase 7's cohort order -- `patient_ids` is saved
explicitly precisely so downstream consumers align by ID, never by row
position, the same discipline every embeddings artifact has followed
since Phase 7.

Still one independent forward pass **per hospital's own graph**, same
model weights, never pooling patients across hospitals -- no exception
starts here either. Phase 8's saved graph files deliberately don't
duplicate node features (to avoid storing Phase 7's embeddings 208
times over) -- this phase reloads them and re-slices by hospital via
an explicit patient-ID lookup, not row position.

## All three architectures implemented (Option: build all, locked 2026-08-26)

`gnn.type: graphsage | gat | gcn` -- all three are implemented, not
just the configured default, matching how `fusion.type` and
`graph.method` were also both/all fully implemented. For a research
write-up, this directly enables an architecture ablation/comparison,
which the config already anticipated as a real choice rather than
picking one and hardcoding it.

## Edge weight usage: differentiated by architecture, not applied uniformly

Phase 8 produces `edge_weight` (cosine similarity scores) alongside
graph topology. Whether a GNN layer *should* use it is a real modeling
question, not just an implementation detail -- and the honest answer
differs by architecture, confirmed directly against the installed PyG
API (2.8.0.post1) before committing to a design, not assumed from
memory:

```
SAGEConv.forward(x, edge_index, size=None)                    -- no edge_weight/edge_attr param at all
GATConv.forward(x, edge_index, edge_attr=None, ...)            -- edge_attr feeds its own attention, not a direct weight
GCNConv.forward(x, edge_index, edge_weight=None)                -- native edge_weight, the published weighted-graph extension
```

- **GCN: uses `edge_weight`.** PyG's `GCNConv` has a native parameter
  for exactly this, directly matching Kipf & Welling (2017)'s standard
  weighted-graph extension (edge weights modulate the normalized graph
  Laplacian). Genuinely well-founded -- not bolted on for appearances.
- **GAT: does not use it.** `GATConv` computes its own attention
  coefficients from node features at every layer. Re-injecting the
  exact cosine similarity that was already used to *build* the graph's
  edges in the first place (the edge exists *because* of that
  similarity) would be largely circular rather than adding genuine new
  signal, and isn't how GAT is standardly used in the literature.
  Standard self-attention only.
- **GraphSAGE: does not use it.** `SAGEConv` has no edge-weight
  mechanism in its standard formulation (mean/pooling aggregation over
  an unweighted neighbor set, per Hamilton et al. 2017). Forcing one in
  would require a non-standard custom aggregator whose benefit is
  unproven, deviating from what "GraphSAGE" means in published work.

This was a deliberate choice, not the default/uncontested option --
discussed and confirmed with the user before building, specifically to
avoid uniformly wiring in a "weighted" mechanism that would look more
sophisticated without being genuinely justified for every architecture
it touched.

`tests/unit/test_patient_gnn.py::TestEdgeWeightUsageIsDifferentiatedByArchitecture`
targets this directly: GCN's output must change when `edge_weight`
changes; GraphSAGE's and GAT's must not. Reverting the forward pass to
never pass `edge_weight` through to any layer (proven by hand -- see
IMPLEMENTATION_TRACKER.md) breaks exactly the GCN-sensitivity test and
nothing else, confirming the suite actually enforces the differentiated
design rather than just describing it in a docstring.

## Small/degenerate graphs handled correctly (confirmed necessary by real data)

Phase 8's real-data run found at least one real hospital with zero
patients in the final cohort (`min=0` in the per-hospital node-count
distribution -- see `docs/patient_graph_construction.md`). Before
writing `PatientGNN`, all three PyG layer types (`SAGEConv`, `GATConv`,
`GCNConv`) were directly tested against empty (`0` nodes) and
single-node (`1` node, `0` edges) graphs to confirm they degrade
gracefully rather than crash -- confirmed clean for all three before
any surrounding code was written, and covered by
`test_empty_graph_does_not_crash` / `test_single_node_zero_edges_does_not_crash`.

## What this does NOT yet do

- **No training loop, no task head.** This phase verifies architecture
  and forward-pass mechanics only. Mortality/sepsis prediction is
  Phase 10's job, consuming this phase's refined embeddings.
- **No FedAvg yet.** "Same weights across hospitals, applied
  independently" is implemented (one `PatientGNN` instance, called once
  per hospital's graph) -- but there's no training loop yet for those
  weights to actually be *averaged* across hospitals after local
  updates. That's further down the roadmap.
- **Not yet re-verified against the real 199,108-patient / 208-hospital
  dataset.** Sandbox-verified only so far, same caveat as every new
  phase before its first real run.

## Verification status

| Component | Status |
|---|---|
| `patient_gnn.py` (all 3 architectures, output shapes, empty/single-node graphs, differentiated edge_weight usage) | Verified — 11 unit tests (`test_patient_gnn.py`), including a direct regression test for the edge_weight design and a hand-performed revert→fail→restore→pass proof |
| Full Phase 1→2→3→4→5→7→8→9 pipeline on mock data, all 3 architectures | Verified — 4 integration tests (`test_phase9_pipeline.py`), including a full-cohort-coverage check (no patient loss or duplication when per-hospital forward passes are reassembled) and a check that a missing Phase 8 manifest raises clearly |

## Immediate next step

Run `python run_pipeline.py --config configs/full.yaml --phase 9` on
the real dataset and confirm the `[PHASE9] GNN (graphsage) embeddings
...` log line reports shape `(199108, 128)`.

**CONFIRMED on the real 199,108-patient / 208-hospital dataset
(2026-08-27):**
```
[GNN] progress: 208/208 hospitals
[PHASE9] GNN (graphsage) embeddings ((199108, 128)) written to outputs\embeddings\gnn_embeddings.npz
[PHASE9_GNN_FORWARD_PASS] COMPLETE (22.23s)
```
Exact expected shape, all 208 hospitals processed, zero errors. Phase 9
is DONE in the fullest sense: implemented, sandbox-verified (including
the revert→fail→restore→pass proof for the differentiated edge_weight
design), and real-full-scale-verified with the default `graphsage`
architecture. `gat`/`gcn` remain sandbox-verified only -- the user
hasn't run those against real data yet, since `full.yaml` doesn't
override `gnn.type` from the `graphsage` default.
