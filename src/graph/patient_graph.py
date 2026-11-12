"""
Phase 8 -- Intra-hospital patient graph construction.

Builds one similarity graph PER HOSPITAL from Phase 7's fused patient
embeddings -- never across hospitals, preserving the same federated
privacy boundary as every other phase (per-hospital normalization,
per-hospital vocabulary aggregation, hospital-sharded large tables).
In a genuine federated deployment, each hospital's graph would only
ever be built and consumed locally on that hospital's own client.

DESIGN DECISION (locked 2026-08-26, Option B, confirmed with user):
similarity is computed EXACTLY everywhere, never approximated.
`graph.max_nodes_full_similarity` (config) switches between computing
the full N x N pairwise similarity matrix in one shot (small/typical
hospitals) and computing it in row-chunks (large hospitals) to bound
peak memory -- NOT between exact and approximate nearest-neighbor
search. This mirrors src/utils/batching.py's run_batched_with_backoff
pattern already used elsewhere in this project. Chosen over adding an
ANN library (e.g. faiss) because: (1) even the largest real hospitals
in eICU-CRD are on the order of a few thousand patients, well within
exact-computation reach -- a 5,000 x 5,000 float32 similarity matrix
is ~100MB, trivial; (2) exact results are more defensible for a
research write-up than introducing an approximation whose recall/error
would itself need justifying; (3) no new dependency. See
docs/patient_graph_construction.md.

OUTPUT FORMAT: framework-agnostic (plain numpy), matching this
project's established convention (Phase 4/5/7 also save embeddings as
plain numpy .npz, not framework-specific tensors) of only converting
to torch/PyG tensors at model-consumption time (Phase 9).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def cosine_similarity_chunked(embeddings: np.ndarray, chunk_size: int | None = None) -> np.ndarray:
    """Exact pairwise cosine similarity, computed in row-chunks if
    `chunk_size` is given and smaller than `embeddings.shape[0]` --
    never materializes more than one chunk-by-N slice of intermediate
    computation at once. Still returns the full N x N matrix either
    way (chunking bounds intermediate memory during computation, not
    the output size) -- fine at the realistic per-hospital scale this
    project operates at (see module docstring); would need a different
    approach (e.g. returning only top-k per row without ever forming a
    dense N x N array) if hospital sizes ever grew by orders of
    magnitude.

    A zero-vector embedding row (norm == 0) gets similarity 0 with
    every other row rather than a NaN/divide-by-zero.
    """
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norm = np.where(norm == 0, 1.0, norm)
    normalized = embeddings / safe_norm
    normalized[np.squeeze(norm == 0)] = 0.0  # zero-norm rows stay all-zero -> similarity 0 with everything

    n = embeddings.shape[0]
    if chunk_size is None or chunk_size >= n:
        return normalized @ normalized.T

    out = np.empty((n, n), dtype=embeddings.dtype)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        out[start:end] = normalized[start:end] @ normalized.T
    return out


def build_knn_edges(similarity: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (edge_index, edge_weight) for a k-NN graph over a
    precomputed similarity matrix, SYMMETRIZED -- if A is among B's k
    nearest neighbors OR B is among A's, an undirected edge A-B is
    added, listed in both directions (PyG convention). Standard
    practice for patient-similarity GNN graphs, so information flows
    both ways during message passing regardless of which direction the
    raw k-NN relationship pointed.

    Self-similarity (the diagonal) is always excluded from neighbor
    selection. `k` is capped at n-1 -- a hospital with fewer than
    k+1 patients never crashes, it just gets a smaller effective k
    (down to 0 for a single-patient hospital, producing an empty,
    valid, node-only graph).
    """
    n = similarity.shape[0]
    k = min(k, max(n - 1, 0))
    if k <= 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=similarity.dtype)

    sim = similarity.copy()
    np.fill_diagonal(sim, -np.inf)  # exclude self from neighbor selection

    # Exact top-k per row via argpartition (avoids a full O(n log n)
    # sort when only the top k are needed -- still exact, not
    # approximate; see module docstring).
    top_k_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]

    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in top_k_idx[i]:
            edges.add((int(min(i, j)), int(max(i, j))))

    if not edges:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=similarity.dtype)

    src = np.array([e[0] for e in edges], dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64)
    weights = similarity[src, dst]

    edge_index = np.stack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src]),
    ])
    edge_weight = np.concatenate([weights, weights]).astype(similarity.dtype)
    return edge_index, edge_weight


def build_cosine_threshold_edges(similarity: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (edge_index, edge_weight) for every pair with cosine
    similarity >= threshold, excluding self-pairs. Naturally symmetric
    (similarity(i, j) == similarity(j, i)); both directions are listed
    explicitly per PyG convention. A hospital with no qualifying pairs
    gets a valid, empty edge set, not an error.
    """
    sim = similarity.copy()
    np.fill_diagonal(sim, -np.inf)
    src, dst = np.where(sim >= threshold)
    weights = sim[src, dst]
    edge_index = np.stack([src.astype(np.int64), dst.astype(np.int64)])
    return edge_index, weights.astype(similarity.dtype)


def build_knn_and_threshold_edges(similarity: np.ndarray, k: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (edge_index, edge_weight) for the INTERSECTION of a k-NN
    graph and a cosine-similarity-threshold graph (confirmed with the
    user 2026-09-20 as the definition of "Graph A" for the sampling/
    graph-comparison work): an edge is kept only if it is among the
    top-k neighbor relationships (see `build_knn_edges`) AND its cosine
    similarity is >= `threshold`. Deliberately reuses `build_knn_edges`
    rather than re-deriving top-k candidates from scratch, so this
    method inherits its exact, already-tested candidate selection
    (argpartition-based, symmetrized, self-excluded, k capped at n-1)
    unchanged -- the only new behavior is the threshold filter applied
    on top.

    A hospital where no top-k neighbor relationship clears the
    threshold gets a valid, empty edge set (some nodes may end up with
    fewer than k edges, or none at all) -- never an error, matching
    every other edge-builder in this module's "small/sparse hospital
    never crashes" discipline. This is intersection (AND), not union
    (OR): a top-k neighbor below the threshold is dropped rather than
    kept, and a below-threshold pair is never added just because one
    side ran out of closer neighbors.
    """
    edge_index, edge_weight = build_knn_edges(similarity, k)
    if edge_index.shape[1] == 0:
        return edge_index, edge_weight
    keep = edge_weight >= threshold
    return edge_index[:, keep], edge_weight[keep]


def build_minibatch_kmeans_candidate_edges(
    embeddings: np.ndarray, k: int, threshold: float, n_clusters: int, seed: int = 42,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """"Graph B" candidate generation (confirmed with the user
    2026-09-23): per-hospital MiniBatch K-Means RESTRICTS which pairs
    of patients are even considered -- same cluster only -- then the
    exact SAME edge rule Graph A uses (`build_knn_and_threshold_edges`:
    top-k AND cosine >= threshold) is applied independently within each
    cluster, using EXACT cosine similarity among that cluster's members
    only. Clustering is a candidate-restriction mechanism ONLY: a
    cluster is never fully connected, and this never introduces
    approximate similarity -- every edge weight is still an exact
    cosine value on the raw embeddings, identical in definition to
    Graph A's (see the comparison-fairness requirements in the module
    docstring's companion doc, docs/patient_graph_construction.md).

    A patient whose cluster has fewer than k qualifying same-cluster
    neighbors ends up with fewer than k edges (or zero, if alone in
    its cluster) rather than being backfilled from outside the cluster
    -- same "small/sparse graph never crashes, just gets fewer edges"
    precedent as every other edge-builder in this module (Graph A
    included). `n_clusters` is capped at `max(1, n // 2)` so a hospital
    smaller than the configured cluster count never produces
    degenerate single-patient clusters purely from an oversized
    request -- the cap actually used is visible via the returned
    `cluster_labels`' own distinct-value count, never silently applied
    without a way to notice.

    K-Means itself runs on L2-NORMALIZED embeddings (so its euclidean
    distance ranks candidates consistently with cosine direction,
    standard practice for cosine-oriented clustering) -- but every edge
    WEIGHT is computed from the RAW, un-normalized `embeddings` array
    (mathematically identical to a value computed on the normalized
    copy for cosine specifically, but kept explicit and separate from
    the clustering step so the edge-weight computation path can never
    accidentally diverge from Graph A's).

    Returns `(edge_index, edge_weight, cluster_labels)`: `edge_index`/
    `edge_weight` in the SAME `(2, E)`/`(E,)` shape as
    `build_knn_edges()` et al., indices into the ORIGINAL (un-clustered)
    `embeddings` array; `cluster_labels` is the `(n,)` cluster
    assignment, for diagnostics/visualization -- not required by the
    caller.
    """
    n = embeddings.shape[0]
    if n < 2:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float64), np.zeros((n,), dtype=np.int64)

    effective_n_clusters = max(1, min(int(n_clusters), n // 2))
    if effective_n_clusters == 1:
        cluster_labels = np.zeros(n, dtype=np.int64)
    else:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.where(norms == 0, 1.0, norms)
        kmeans = MiniBatchKMeans(n_clusters=effective_n_clusters, random_state=seed, n_init="auto")
        cluster_labels = kmeans.fit_predict(normalized).astype(np.int64)

    src_parts: list = []
    dst_parts: list = []
    weight_parts: list = []
    for cluster_id in np.unique(cluster_labels):
        member_idx = np.where(cluster_labels == cluster_id)[0]
        if len(member_idx) < 2:
            continue  # a lone member has no same-cluster candidates -- zero edges from this cluster, by design
        cluster_similarity = cosine_similarity_chunked(embeddings[member_idx])
        local_edge_index, local_edge_weight = build_knn_and_threshold_edges(cluster_similarity, k, threshold)
        if local_edge_index.shape[1] == 0:
            continue
        global_edge_index = member_idx[local_edge_index]
        src_parts.append(global_edge_index[0])
        dst_parts.append(global_edge_index[1])
        weight_parts.append(local_edge_weight)

    if not src_parts:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float64), cluster_labels
    edge_index = np.stack([np.concatenate(src_parts), np.concatenate(dst_parts)]).astype(np.int64)
    edge_weight = np.concatenate(weight_parts)
    return edge_index, edge_weight, cluster_labels


def build_hospital_graph(
    patient_ids: list[int],
    embeddings: np.ndarray,
    method: str = "knn",
    k: int = 5,
    cosine_threshold: float = 0.7,
    max_nodes_full_similarity: int = 2000,
    n_clusters: int = 10,
    cluster_seed: int = 42,
) -> dict:
    """Builds one hospital's patient similarity graph from that
    hospital's own fused embeddings ONLY -- the caller is responsible
    for ensuring `embeddings`/`patient_ids` never include another
    hospital's patients (see run_pipeline.py's Phase 8, which slices
    Phase 7's fused embeddings by hospital via partition_result, never
    pooling across hospitals).

    `n_clusters`/`cluster_seed`, added 2026-09-23 for
    `method="minibatch_kmeans_and_threshold"` ("Graph B" -- see
    `build_minibatch_kmeans_candidate_edges()`): unused by every other
    method, so their defaults never affect `knn`/`cosine_threshold`/
    `knn_and_threshold` callers, including every existing caller that
    doesn't pass them.

    Returns a dict: `patient_ids` (node index i -> real patient ID,
    same order as `embeddings`'s rows -- the node-identity mapping
    every downstream consumer must use, never a bare row index alone),
    `edge_index` (2, E) int64, `edge_weight` (E,), `n_nodes`, `n_edges`
    -- plus `cluster_labels` (n,) when `method="minibatch_kmeans_and_
    threshold"` (absent for every other method, since they never
    cluster).
    """
    if len(patient_ids) != embeddings.shape[0]:
        raise ValueError(
            f"patient_ids length ({len(patient_ids)}) doesn't match embeddings row count "
            f"({embeddings.shape[0]}) -- would silently misalign node index to patient identity."
        )
    if embeddings.shape[0] > 0 and np.isnan(embeddings).any():
        raise ValueError("embeddings contain NaN -- cannot compute similarity; investigate upstream.")

    n = embeddings.shape[0]

    if method == "minibatch_kmeans_and_threshold":
        # Deliberately skips the full N x N similarity matrix this
        # function otherwise computes -- Graph B's entire point is
        # bounding which pairs are ever compared via clustering first;
        # computing the full matrix here first would defeat that.
        edge_index, edge_weight, cluster_labels = build_minibatch_kmeans_candidate_edges(
            embeddings, k=k, threshold=cosine_threshold, n_clusters=n_clusters, seed=cluster_seed,
        )
        return {
            "patient_ids": np.array(patient_ids), "edge_index": edge_index, "edge_weight": edge_weight,
            "n_nodes": n, "n_edges": edge_index.shape[1], "cluster_labels": cluster_labels,
        }

    chunk_size = max_nodes_full_similarity if n > max_nodes_full_similarity else None
    similarity = cosine_similarity_chunked(embeddings, chunk_size=chunk_size) if n > 0 else np.empty((0, 0))

    if method == "knn":
        edge_index, edge_weight = build_knn_edges(similarity, k)
    elif method == "cosine_threshold":
        edge_index, edge_weight = build_cosine_threshold_edges(similarity, cosine_threshold)
    elif method == "knn_and_threshold":
        edge_index, edge_weight = build_knn_and_threshold_edges(similarity, k, cosine_threshold)
    else:
        raise ValueError(
            f"graph.method={method!r} not recognized -- must be 'knn', 'cosine_threshold', 'knn_and_threshold', "
            f"or 'minibatch_kmeans_and_threshold'."
        )

    return {
        "patient_ids": np.array(patient_ids),
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "n_nodes": n,
        "n_edges": edge_index.shape[1],
    }
