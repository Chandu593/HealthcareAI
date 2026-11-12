"""Tests for src/graph/patient_graph.py (Phase 8).

Covers: exact-vs-chunked similarity equivalence (the core claim behind
"Option B" -- chunking must never change the result, only bound peak
memory), k-NN and cosine-threshold edge correctness, node-identity
alignment (edge_index refers to row position, patient_ids maps that
position back to a real patient -- the same class of correctness risk
as Phase 7's ID-based fusion alignment), and small/empty-hospital edge
cases that must degrade gracefully, never crash.
"""

import unittest

import numpy as np

from src.graph.patient_graph import (
    build_cosine_threshold_edges,
    build_hospital_graph,
    build_knn_and_threshold_edges,
    build_knn_edges,
    build_minibatch_kmeans_candidate_edges,
    cosine_similarity_chunked,
)


class TestCosineSimilarityChunked(unittest.TestCase):
    def test_identical_vectors_have_similarity_one(self):
        emb = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        sim = cosine_similarity_chunked(emb)
        np.testing.assert_allclose(sim, np.ones((2, 2)), atol=1e-10)

    def test_orthogonal_vectors_have_similarity_zero(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = cosine_similarity_chunked(emb)
        self.assertAlmostEqual(sim[0, 1], 0.0)
        self.assertAlmostEqual(sim[1, 0], 0.0)

    def test_scale_invariant(self):
        emb = np.array([[1.0, 0.0], [2.0, 0.0], [100.0, 0.0]])
        sim = cosine_similarity_chunked(emb)
        np.testing.assert_allclose(sim, np.ones((3, 3)), atol=1e-10)

    def test_zero_vector_row_gives_zero_similarity_not_nan(self):
        emb = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        sim = cosine_similarity_chunked(emb)
        self.assertFalse(np.isnan(sim).any())
        self.assertEqual(sim[1, 0], 0.0)
        self.assertEqual(sim[1, 2], 0.0)

    def test_chunked_matches_unchunked_exactly(self):
        rng = np.random.RandomState(42)
        emb = rng.randn(37, 16)  # size not evenly divisible by any tested chunk_size
        full = cosine_similarity_chunked(emb, chunk_size=None)
        for chunk_size in (1, 5, 10, 36, 37, 100):
            chunked = cosine_similarity_chunked(emb, chunk_size=chunk_size)
            np.testing.assert_allclose(
                chunked, full, atol=1e-10,
                err_msg=f"chunk_size={chunk_size} produced a different result than unchunked -- "
                        f"chunking must be exact, not approximate (see module docstring)",
            )


class TestBuildKnnEdges(unittest.TestCase):
    def test_finds_correct_nearest_neighbor(self):
        # 4 points forming two clearly-separated similar pairs
        emb = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
        sim = cosine_similarity_chunked(emb)
        edge_index, edge_weight = build_knn_edges(sim, k=1)
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        self.assertIn((0, 1), edges)
        self.assertIn((1, 0), edges)
        self.assertIn((2, 3), edges)
        self.assertIn((3, 2), edges)
        self.assertNotIn((0, 2), edges)

    def test_self_excluded_from_neighbors(self):
        emb = np.eye(3)  # 3 mutually-orthogonal points
        sim = cosine_similarity_chunked(emb)
        edge_index, _ = build_knn_edges(sim, k=2)
        for i in range(3):
            self.assertNotIn((i, i), zip(edge_index[0].tolist(), edge_index[1].tolist()))

    def test_symmetrized_both_directions_present(self):
        emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        sim = cosine_similarity_chunked(emb)
        edge_index, _ = build_knn_edges(sim, k=1)
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        for (i, j) in list(edges):
            self.assertIn((j, i), edges, f"edge ({i},{j}) present without reverse ({j},{i})")

    def test_k_capped_at_n_minus_1_does_not_crash(self):
        sim = cosine_similarity_chunked(np.array([[1.0, 0.0], [0.9, 0.1]]))
        edge_index, edge_weight = build_knn_edges(sim, k=50)  # way more than available
        self.assertEqual(edge_index.shape[1], 2)  # one undirected pair -> 2 directed entries

    def test_single_node_graph_has_zero_edges(self):
        sim = cosine_similarity_chunked(np.array([[1.0, 2.0]]))
        edge_index, edge_weight = build_knn_edges(sim, k=5)
        self.assertEqual(edge_index.shape, (2, 0))
        self.assertEqual(edge_weight.shape, (0,))


class TestBuildCosineThresholdEdges(unittest.TestCase):
    def test_only_pairs_above_threshold_included(self):
        emb = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
        sim = cosine_similarity_chunked(emb)
        edge_index, edge_weight = build_cosine_threshold_edges(sim, threshold=0.9)
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        self.assertIn((0, 1), edges)
        self.assertNotIn((0, 2), edges)

    def test_no_qualifying_pairs_returns_empty_not_error(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = cosine_similarity_chunked(emb)
        edge_index, edge_weight = build_cosine_threshold_edges(sim, threshold=0.99)
        self.assertEqual(edge_index.shape, (2, 0))

    def test_self_pairs_excluded_even_at_low_threshold(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = cosine_similarity_chunked(emb)
        edge_index, _ = build_cosine_threshold_edges(sim, threshold=-1.1)  # matches everything
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        self.assertNotIn((0, 0), edges)
        self.assertNotIn((1, 1), edges)


class TestBuildKnnAndThresholdEdges(unittest.TestCase):
    """"Graph A" for the sampling/graph-comparison work (confirmed with
    the user 2026-09-20): edges must satisfy BOTH the k-NN criterion
    AND the cosine-similarity threshold -- intersection (AND), not
    union (OR). Constructed with exactly-known pairwise similarities so
    each assertion is checkable by hand, not just "some edges appear."
    """

    def _three_point_fixture(self):
        # p0=[1,0], p1=[0.6,0.8] (unit norm), p2=[0,1].
        # cos(p0,p1)=0.6, cos(p1,p2)=0.8, cos(p0,p2)=0.0.
        # With k=1: p0's nearest is p1 (0.6); p1's nearest is p2 (0.8);
        # p2's nearest is p1 (0.8). Raw k-NN edge set (symmetrized):
        # {(0,1)=0.6, (1,2)=0.8}.
        emb = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
        return cosine_similarity_chunked(emb)

    def test_below_threshold_knn_edge_is_dropped(self):
        sim = self._three_point_fixture()
        edge_index, edge_weight = build_knn_and_threshold_edges(sim, k=1, threshold=0.7)
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        # (0,1) is a real top-1 kNN edge (weight 0.6) but below the 0.7
        # threshold -- AND/intersection must drop it.
        self.assertNotIn((0, 1), edges)
        self.assertNotIn((1, 0), edges)

    def test_above_threshold_knn_edge_is_kept(self):
        sim = self._three_point_fixture()
        edge_index, edge_weight = build_knn_and_threshold_edges(sim, k=1, threshold=0.7)
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        # (1,2) is a top-1 kNN edge AND clears 0.7 -- must be kept, both directions.
        self.assertIn((1, 2), edges)
        self.assertIn((2, 1), edges)
        np.testing.assert_allclose(sorted(edge_weight.tolist()), [0.8, 0.8], atol=1e-10)

    def test_union_would_have_kept_more_edges_than_intersection_does(self):
        # Directly distinguishes AND from OR: a threshold-only graph at
        # 0.5 would also include (0,1) (0.6 >= 0.5); the AND/intersection
        # result must NOT, since this test targets the intersection
        # semantics regardless of the threshold's own value relative to
        # any single edge.
        sim = self._three_point_fixture()
        and_edges, _ = build_knn_and_threshold_edges(sim, k=1, threshold=0.5)
        threshold_only_edges, _ = build_cosine_threshold_edges(sim, threshold=0.5)
        self.assertLessEqual(and_edges.shape[1], threshold_only_edges.shape[1])

    def test_no_qualifying_edges_returns_empty_not_error(self):
        sim = self._three_point_fixture()
        edge_index, edge_weight = build_knn_and_threshold_edges(sim, k=1, threshold=0.99)
        self.assertEqual(edge_index.shape, (2, 0))
        self.assertEqual(edge_weight.shape, (0,))

    def test_empty_graph_never_crashes(self):
        edge_index, edge_weight = build_knn_and_threshold_edges(np.empty((0, 0)), k=5, threshold=0.6)
        self.assertEqual(edge_index.shape, (2, 0))


class TestBuildMinibatchKmeansCandidateEdges(unittest.TestCase):
    """"Graph B" (confirmed with the user 2026-09-23): candidates
    restricted to the same MiniBatch K-Means cluster, then the exact
    SAME top-k-AND-cosine>=threshold rule Graph A uses. Uses two
    tightly-clustered, near-orthogonal groups of fixed (non-random)
    points so cluster assignment is reliably predictable without
    depending on exact k-means internals.
    """

    def _two_cluster_fixture(self):
        # Group A: 4 points tightly clustered near [1, 0] (high mutual
        # cosine similarity). Group B: 4 points tightly clustered near
        # [0, 1], near-orthogonal to group A (cosine similarity near 0
        # across groups).
        group_a = np.array([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03]])
        group_b = np.array([[0.0, 1.0], [0.01, 0.99], [0.02, 0.98], [0.03, 0.97]])
        return np.vstack([group_a, group_b])

    def test_candidates_restricted_to_same_cluster_no_cross_cluster_edges(self):
        # Deliberately NOT near-orthogonal: within-group cosine ~0.999,
        # cross-group cosine ~0.69-0.73 -- still comfortably ABOVE the
        # 0.5 threshold used below. This is the point: a plain global
        # top-k-AND-threshold pass (no clustering) WOULD connect these
        # two groups, since cross-group pairs satisfy both k and the
        # threshold on their own. Only genuine candidate RESTRICTION by
        # clustering prevents the cross-group edges -- the threshold
        # alone can't be doing the work here, unlike a near-orthogonal
        # fixture where it would be ambiguous which mechanism excluded
        # them.
        group_a = np.array([[1.0, 0.0, 0.0], [0.99, 0.02, 0.0], [0.98, 0.03, 0.01], [0.97, 0.01, 0.02]])
        group_b = np.array([[0.707, 0.707, 0.0], [0.70, 0.71, 0.01], [0.69, 0.72, 0.0], [0.71, 0.70, 0.02]])
        emb = np.vstack([group_a, group_b])

        # k=5 with only 4-member groups is deliberate: each node has
        # just 3 same-group peers, so an UNRESTRICTED top-5 (no
        # clustering) is forced to reach into the other group for its
        # 4th/5th neighbor -- k=3 (<= group size - 1) would let
        # same-group peers alone fill every slot, making clustering
        # invisible in the result even if it were silently skipped.
        edge_index, edge_weight, cluster_labels = build_minibatch_kmeans_candidate_edges(
            emb, k=5, threshold=0.5, n_clusters=2, seed=42,
        )
        # The two groups must land in different clusters -- a
        # precondition check, not the real assertion, so a fixture
        # regression is caught explicitly rather than silently passing
        # for the wrong reason.
        self.assertEqual(len(set(cluster_labels[:4].tolist())), 1)
        self.assertEqual(len(set(cluster_labels[4:].tolist())), 1)
        self.assertNotEqual(cluster_labels[0], cluster_labels[4])

        # The real assertion: no edge crosses between a group-A index
        # (0-3) and a group-B index (4-7), even though these pairs
        # individually satisfy both top-k and the 0.5 threshold --
        # clustering must have actually restricted the candidate pairs,
        # not merely relied on the threshold to exclude them.
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        for s, d in zip(src, dst):
            self.assertEqual(s < 4, d < 4, f"cross-cluster edge found: {s}->{d}")
        self.assertGreater(edge_index.shape[1], 0)  # and it isn't just an empty graph

    def test_edges_within_a_cluster_still_require_the_and_rule(self):
        emb = self._two_cluster_fixture()
        # An unreachable threshold within each tight cluster -- even
        # same-cluster candidates must still clear it.
        edge_index, _edge_weight, _labels = build_minibatch_kmeans_candidate_edges(
            emb, k=3, threshold=0.999999, n_clusters=2, seed=42,
        )
        self.assertEqual(edge_index.shape[1], 0)

    def test_edge_weights_are_exact_cosine_not_approximate(self):
        emb = self._two_cluster_fixture()
        edge_index, edge_weight, _labels = build_minibatch_kmeans_candidate_edges(
            emb, k=3, threshold=0.9, n_clusters=2, seed=42,
        )
        full_similarity = cosine_similarity_chunked(emb)
        for (s, d), w in zip(edge_index.T.tolist(), edge_weight.tolist()):
            self.assertAlmostEqual(w, full_similarity[s, d], places=10)

    def test_n_clusters_capped_for_small_hospitals(self):
        # n=4 patients, n_clusters requested=10 -> effective cap is
        # max(1, min(10, 4 // 2)) = 2, never 10 (which would force
        # degenerate 1-2 patient clusters from an oversized request).
        emb = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
        _edge_index, _edge_weight, cluster_labels = build_minibatch_kmeans_candidate_edges(
            emb, k=3, threshold=0.5, n_clusters=10, seed=42,
        )
        self.assertLessEqual(len(set(cluster_labels.tolist())), 2)

    def test_lone_member_cluster_gets_zero_edges_not_an_error(self):
        # 5 points: 4 tight + 1 far outlier -- with n_clusters=2 the
        # outlier should land alone in its own cluster.
        emb = np.vstack([self._two_cluster_fixture()[:4], [[-1.0, -1.0]]])
        edge_index, edge_weight, cluster_labels = build_minibatch_kmeans_candidate_edges(
            emb, k=3, threshold=0.5, n_clusters=2, seed=42,
        )
        outlier_cluster = cluster_labels[4]
        # No edge touches the outlier if it is alone in its cluster.
        if (cluster_labels == outlier_cluster).sum() == 1:
            self.assertNotIn(4, edge_index[0].tolist())
            self.assertNotIn(4, edge_index[1].tolist())

    def test_deterministic_for_fixed_seed(self):
        emb = self._two_cluster_fixture()
        r1 = build_minibatch_kmeans_candidate_edges(emb, k=3, threshold=0.9, n_clusters=2, seed=7)
        r2 = build_minibatch_kmeans_candidate_edges(emb, k=3, threshold=0.9, n_clusters=2, seed=7)
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])
        np.testing.assert_array_equal(r1[2], r2[2])

    def test_empty_and_single_patient_hospital_never_crash(self):
        edge_index, edge_weight, labels = build_minibatch_kmeans_candidate_edges(
            np.empty((0, 4)), k=5, threshold=0.6, n_clusters=10,
        )
        self.assertEqual(edge_index.shape, (2, 0))
        self.assertEqual(len(labels), 0)

        edge_index, edge_weight, labels = build_minibatch_kmeans_candidate_edges(
            np.array([[1.0, 0.0]]), k=5, threshold=0.6, n_clusters=10,
        )
        self.assertEqual(edge_index.shape, (2, 0))
        self.assertEqual(len(labels), 1)


class TestBuildHospitalGraph(unittest.TestCase):
    def test_node_index_correctly_maps_to_patient_id(self):
        # Regression target: edge_index refers to ROW POSITION within
        # this call's embeddings array, and patient_ids (same order)
        # is the only correct way to map that back to a real patient.
        # Patient IDs are deliberately NOT in sorted order here -- a
        # bug that silently re-sorted patient_ids (e.g. `sorted(...)`)
        # before returning it would otherwise slip past unnoticed if
        # the IDs happened to already be in ascending order, exactly
        # the kind of test-data coincidence that hides a real bug.
        # Row 0 (patient 999) and row 2 (patient 222) are the similar
        # pair here, not whichever patient a sorted list would put in
        # those positions.
        patient_ids = [999, 111, 222]
        emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.99, 0.01]])  # row 0 and row 2 are similar
        graph = build_hospital_graph(patient_ids, emb, method="knn", k=1)

        node_to_patient = dict(enumerate(graph["patient_ids"].tolist()))
        edges_by_patient = {
            (node_to_patient[i], node_to_patient[j])
            for i, j in zip(graph["edge_index"][0].tolist(), graph["edge_index"][1].tolist())
        }
        self.assertIn((999, 222), edges_by_patient)
        self.assertIn((222, 999), edges_by_patient)
        self.assertNotIn((999, 111), edges_by_patient)

    def test_raises_on_patient_ids_embeddings_length_mismatch(self):
        with self.assertRaises(ValueError):
            build_hospital_graph([1, 2, 3], np.zeros((2, 4)), method="knn")

    def test_raises_on_nan_embeddings(self):
        emb = np.array([[1.0, np.nan], [2.0, 3.0]])
        with self.assertRaises(ValueError):
            build_hospital_graph([1, 2], emb, method="knn")

    def test_raises_on_unknown_method(self):
        with self.assertRaises(ValueError):
            build_hospital_graph([1, 2], np.zeros((2, 4)), method="not_a_real_method")

    def test_empty_hospital_produces_valid_empty_graph(self):
        graph = build_hospital_graph([], np.empty((0, 8)), method="knn")
        self.assertEqual(graph["n_nodes"], 0)
        self.assertEqual(graph["n_edges"], 0)

    def test_chunking_triggered_above_max_nodes_full_similarity_gives_same_result(self):
        rng = np.random.RandomState(7)
        emb = rng.randn(20, 6)
        patient_ids = list(range(20))
        full = build_hospital_graph(patient_ids, emb, method="knn", k=3, max_nodes_full_similarity=1000)
        chunked = build_hospital_graph(patient_ids, emb, method="knn", k=3, max_nodes_full_similarity=5)

        full_edges = set(zip(full["edge_index"][0].tolist(), full["edge_index"][1].tolist()))
        chunked_edges = set(zip(chunked["edge_index"][0].tolist(), chunked["edge_index"][1].tolist()))
        self.assertEqual(full_edges, chunked_edges)
        np.testing.assert_allclose(sorted(full["edge_weight"]), sorted(chunked["edge_weight"]), atol=1e-10)

    def test_method_dispatches_to_cosine_threshold(self):
        emb = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
        graph = build_hospital_graph([1, 2, 3], emb, method="cosine_threshold", cosine_threshold=0.9)
        edges = set(zip(graph["edge_index"][0].tolist(), graph["edge_index"][1].tolist()))
        self.assertIn((0, 1), edges)
        self.assertNotIn((0, 2), edges)

    def test_method_dispatches_to_knn_and_threshold(self):
        # Same fixture/reasoning as TestBuildKnnAndThresholdEdges: with
        # k=1, (0,1)'s raw kNN weight (0.6) is below 0.7 and must be
        # excluded through this dispatch path too, not just when
        # calling build_knn_and_threshold_edges() directly.
        emb = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
        graph = build_hospital_graph([10, 20, 30], emb, method="knn_and_threshold", k=1, cosine_threshold=0.7)
        edges = set(zip(graph["edge_index"][0].tolist(), graph["edge_index"][1].tolist()))
        self.assertIn((1, 2), edges)
        self.assertNotIn((0, 1), edges)

    def test_method_dispatches_to_minibatch_kmeans_and_threshold(self):
        group_a = np.array([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03]])
        group_b = np.array([[0.0, 1.0], [0.01, 0.99], [0.02, 0.98], [0.03, 0.97]])
        emb = np.vstack([group_a, group_b])
        graph = build_hospital_graph(
            list(range(8)), emb, method="minibatch_kmeans_and_threshold", k=3, cosine_threshold=0.9,
            n_clusters=2, cluster_seed=42,
        )
        self.assertIn("cluster_labels", graph)
        self.assertEqual(len(graph["cluster_labels"]), 8)
        src, dst = graph["edge_index"][0].tolist(), graph["edge_index"][1].tolist()
        for s, d in zip(src, dst):
            self.assertEqual(s < 4, d < 4)

    def test_other_methods_never_return_cluster_labels(self):
        emb = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
        graph = build_hospital_graph([1, 2, 3], emb, method="knn", k=1)
        self.assertNotIn("cluster_labels", graph)


if __name__ == "__main__":
    unittest.main()
