# Intra-Hospital Patient Graph Construction (Phase 8)

**Status: implemented, sandbox-verified.** `src/graph/patient_graph.py`
is fully tested (20 unit tests) including direct regression tests for
both core correctness risks (see below) and hand-performed
revert→fail→restore→pass proofs for each. The full
Phase 1→2→3→4→5→7→8 pipeline is verified end-to-end against synthetic
data (4 integration tests), including a direct proof that no edge ever
crosses a hospital boundary. **Not yet run against the real dataset**
-- immediate next step once delivered.

## What Phase 8 does

Builds ONE similarity graph PER HOSPITAL from Phase 7's fused
`(N, 192)` patient embeddings, saved as
`outputs/graphs/hospital_<id>_graph.npz` (`patient_ids`, `edge_index`,
`edge_weight`) plus a `graph_manifest.json` indexing all of them.
`torch-geometric` (already declared in `requirements.txt` for "Phase
4-10: modeling") is the intended consumer of this output in Phase 9 --
confirmed installing and working cleanly
(`torch_geometric.data.Data(x=..., edge_index=...)`) before this phase
was built, so the output format targets it directly. Phase 8 itself
stays framework-agnostic (plain numpy), matching the same convention
Phase 4/5/7 already established (embeddings saved as `.npz`, not
framework-specific tensors, converting only at model-consumption time).

## Non-negotiable: intra-hospital only, no exceptions

Every other phase in this pipeline draws a hard line at the hospital
boundary -- per-hospital normalization, per-hospital vocabulary
aggregation (Phase 5's fix, 2026-08-24), hospital-sharded large tables
(never pooled in memory across hospitals). Graph construction doesn't
get an exception: each hospital's graph is built from *only* that
hospital's own patients' fused embeddings, mirroring what a genuine
federated deployment would look like (each hospital only ever builds
and holds its own local graph).

`run_phase8_patient_graph_construction()` slices Phase 7's fused
embeddings by hospital via `partition_result.partitions[hid].patient_ids`
(the same per-hospital iteration pattern established in Phase 4/5),
never passing the full cohort's embeddings into a single graph-building
call.
`tests/integration/test_phase8_pipeline.py::test_no_edges_ever_cross_a_hospital_boundary`
checks this directly against each hospital's own saved graph file (not
inferred from the construction code) -- for every edge in every
hospital's graph, both endpoint patients are confirmed to actually
belong to that hospital. Reverting the per-hospital slicing to pool
the full cohort into every hospital's graph (proven by hand -- see
IMPLEMENTATION_TRACKER.md) makes this test fail immediately on real
cross-hospital edges, and also breaks `total_nodes` (4x too many, in
the test scenario) -- confirming this suite would actually catch a
regression here, not just happen to pass against a correctly-wired
pipeline.

## Similarity: exact everywhere, chunked (not approximate) for memory (Option B, locked 2026-08-26)

`configs/base.yaml`'s `graph.max_nodes_full_similarity` (default 2000)
already existed with a comment saying "above this, force
approximate/ANN construction." That comment has been corrected --
**similarity is computed exactly on both sides of this threshold.**
Below it, the full N×N pairwise cosine-similarity matrix is computed
in one shot; above it, the same exact computation happens in
row-chunks (mirroring `src/utils/batching.py`'s
`run_batched_with_backoff` pattern already used elsewhere in this
project), bounding peak memory without ever approximating the result.

Two options existed, and this was discussed and confirmed with the
user before building:

- **Option B (chosen): stay exact, chunk for memory only.** No new
  dependency. Even the largest real hospitals in eICU-CRD are on the
  order of a few thousand patients -- a 5,000×5,000 float32 similarity
  matrix is ~100MB, trivially exact-computable. Exact k-NN/cosine-
  threshold results are also more defensible in a research write-up
  than an approximation, which would need its own recall/error
  justification for no practical benefit at this scale.
- **Option A (not chosen): add a real ANN library** (e.g. `faiss-cpu`)
  above the threshold, matching the original config comment literally.
  Rejected as unnecessary complexity/dependency weight given the
  realistic scale here.

`tests/unit/test_patient_graph.py::TestCosineSimilarityChunked::test_chunked_matches_unchunked_exactly`
and `TestBuildHospitalGraph::test_chunking_triggered_above_max_nodes_full_similarity_gives_same_result`
directly target this: chunking must produce bit-for-bit (well,
floating-point-exact within tolerance) identical results to computing
the full matrix at once, across several different chunk sizes,
including sizes that don't evenly divide the patient count. Reverting
the chunked-computation loop to a subtly wrong (but plausible-looking)
version -- comparing each chunk only against itself instead of the
full set, a classic chunking bug -- was proven by hand to make exactly
these two tests fail, and only these two.

## Graph construction mechanics

- **Two methods**, both config-driven (`graph.method`): `knn` (each
  patient connects to their `k` most similar patients within the same
  hospital) and `cosine_threshold` (every pair above
  `graph.cosine_threshold` gets an edge). Both symmetrized -- if A is
  among B's k-nearest (or A-B's cosine similarity clears the
  threshold), an undirected edge is added, listed in both directions
  per PyG convention, so message passing flows both ways regardless of
  which direction the raw relationship pointed.
- **Self-loops excluded** at graph-construction time -- left to the
  GNN layer (Phase 9) to add if a given architecture needs them (e.g.
  PyG's `GCNConv` typically adds them internally), rather than baking
  that model-specific detail into the graph artifact itself.
- **Small hospitals degrade gracefully, never crash.** `k` is
  automatically capped at `n_patients - 1` (a 3-patient hospital with
  `k=5` configured just gets `k=2`); a single-patient hospital produces
  a valid graph with 0 edges; an empty hospital produces a valid,
  empty graph. A `cosine_threshold` hospital with no qualifying pairs
  also produces a valid, empty edge set.
- **Node-identity alignment**, the same class of correctness risk
  Phase 7 addressed for fusion: `edge_index` refers to row position
  *within that hospital's own patient list*, and the saved
  `patient_ids` array (same order) is the only correct way to map a
  node index back to a real patient.
  `TestBuildHospitalGraph::test_node_index_correctly_maps_to_patient_id`
  uses a deliberately *unsorted* patient-ID list specifically so that a
  bug which silently re-sorted `patient_ids` before returning it (a
  real gap this test's first draft actually had, caught during its own
  hand-performed revert→fail→restore→pass check -- the original
  version happened to use already-sorted test IDs, so the bug injection
  had no effect) would be caught, not accidentally invisible due to
  coincidentally-already-sorted test data.

## What this does NOT yet do

- **No GNN forward pass yet.** Phase 8 produces the graph structure
  only; Phase 9 (not started) consumes it with `torch-geometric`,
  per `configs/base.yaml`'s `gnn:` section (`graphsage`/`gat`/`gcn`,
  already stubbed).
- **Only `metric: cosine` is implemented.** `run_phase8_patient_graph_construction()`
  raises `NotImplementedError` for anything else in config, rather
  than silently ignoring the setting.
- **Not yet re-verified against the real 199,108-patient / 208-hospital
  dataset.** Sandbox-verified only so far, same caveat as every new
  phase before its first real run.

## Verification status

| Component | Status |
|---|---|
| `patient_graph.py` (exact-vs-chunked similarity, k-NN edges, cosine-threshold edges, node-identity alignment, small/empty-hospital edge cases) | Verified — 20 unit tests (`test_patient_graph.py`), including two hand-performed revert→fail→restore→pass proofs (chunking correctness; node-identity alignment) |
| Full Phase 1→2→3→4→5→7→8 pipeline on mock data, no-cross-hospital-edges guarantee specifically | Verified — 4 integration tests (`test_phase8_pipeline.py`), including a hand-performed revert→fail→restore→pass proof that pooling the full cohort into every hospital's graph gets caught immediately (wrong edge count AND wrong total node count) |

## Immediate next step

Run `python run_pipeline.py --config configs/full.yaml --phase 8` on
the real dataset and confirm the `[PHASE8] ... hospital graphs written`
log line reports 208 hospitals and a total node count of 199,108.

**CONFIRMED on the real 199,108-patient / 208-hospital dataset
(2026-08-26):**
```
[PHASE8] 208 hospital graphs written to outputs\graphs
(total nodes=199,108, total edges=1,462,202, method=knn)
[PHASE8_PATIENT_GRAPH_CONSTRUCTION] COMPLETE (15.11s)
```
Followed by a dedicated read-only verification script
(`verify_phase8_graphs.py`) checking two things the log alone doesn't
directly prove:
- **Zero cross-hospital edges**, checked directly against all
  1,462,202 real directed edge entries across all 208 saved graph
  files (not inferred from the aggregate node count matching).
- **The chunked-similarity path (`max_nodes_full_similarity=2000`) was
  genuinely exercised**, not just sandbox-tested: 29 real hospitals
  exceeded 2000 patients (largest: 7,058), and the whole 208-hospital
  run -- including chunking that 7,058x7,058 similarity matrix --
  completed in 15.11s with zero errors.
- Real per-hospital size distribution: min=0 (at least one real
  hospital has zero patients in the final cohort, likely filtered out
  upstream in Phase 3 -- handled gracefully by the empty-hospital case,
  exactly as `test_empty_hospital_produces_valid_empty_graph` verifies),
  median=548, max=7,058 -- genuine, extreme non-IID heterogeneity
  across federated clients, relevant to later FedAvg work.

Phase 8 is now DONE in the fullest sense: implemented, sandbox-verified
(including both revert→fail→restore→pass proofs), and real-full-scale-
verified at BOTH the aggregate level (log line) AND the direct,
file-level edge-by-edge check -- more rigorously confirmed on real data
than any phase before it in this project.
