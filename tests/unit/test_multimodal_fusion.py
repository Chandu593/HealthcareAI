"""Tests for src/fusion/multimodal_fusion.py (Phase 7).

The core correctness risk in fusion isn't the concatenation itself --
it's silently misaligning two modalities' embeddings if they don't
happen to be stored in the same patient order. A row-order-based (as
opposed to ID-based) concatenation would produce a fused matrix that
LOOKS fine (right shape, no crash) while every row after the first
divergence is actually a mix of two DIFFERENT patients' data -- a
silent, hard-to-detect corruption that would poison every downstream
sample. TestFuseEmbeddings.test_misaligned_storage_order_still_fuses_
correctly is the test that directly targets this.
"""

import unittest

import numpy as np

from src.fusion.multimodal_fusion import align_embeddings_to_cohort, fuse_embeddings


class TestAlignEmbeddingsToCohort(unittest.TestCase):
    def test_identity_when_order_already_matches(self):
        embeddings = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        result = align_embeddings_to_cohort([10, 20, 30], embeddings, [10, 20, 30])
        np.testing.assert_array_equal(result, embeddings)

    def test_reorders_to_match_target_order(self):
        embeddings = np.array([[1.0], [2.0], [3.0]])  # rows for patients 10, 20, 30 respectively
        result = align_embeddings_to_cohort([10, 20, 30], embeddings, [30, 10, 20])
        np.testing.assert_array_equal(result, np.array([[3.0], [1.0], [2.0]]))

    def test_subset_of_target_order_selected_correctly(self):
        embeddings = np.array([[1.0], [2.0], [3.0], [4.0]])
        result = align_embeddings_to_cohort([10, 20, 30, 40], embeddings, [40, 10])
        np.testing.assert_array_equal(result, np.array([[4.0], [1.0]]))

    def test_raises_on_missing_target_patient(self):
        embeddings = np.array([[1.0], [2.0]])
        with self.assertRaises(ValueError) as ctx:
            align_embeddings_to_cohort([10, 20], embeddings, [10, 20, 999])
        self.assertIn("999", str(ctx.exception))

    def test_raises_on_duplicate_patient_ids(self):
        embeddings = np.array([[1.0], [2.0]])
        with self.assertRaises(ValueError):
            align_embeddings_to_cohort([10, 10], embeddings, [10])


class TestFuseEmbeddings(unittest.TestCase):
    def test_basic_concatenation_lstm_columns_then_autoencoder_columns(self):
        lstm_ids, lstm_emb = [1, 2], np.array([[1.0, 1.0], [2.0, 2.0]])
        ae_ids, ae_emb = [1, 2], np.array([[10.0], [20.0]])

        patient_ids, fused = fuse_embeddings(lstm_ids, lstm_emb, ae_ids, ae_emb)

        self.assertEqual(patient_ids, [1, 2])
        self.assertEqual(fused.shape, (2, 3))
        np.testing.assert_array_equal(fused, np.array([[1.0, 1.0, 10.0], [2.0, 2.0, 20.0]]))

    def test_misaligned_storage_order_still_fuses_correctly(self):
        # The critical regression target: LSTM and autoencoder embeddings
        # are stored in DIFFERENT patient orders (as could genuinely
        # happen if either modality's underlying save order ever
        # changed independently). A naive positional concatenation
        # (np.concatenate([lstm_emb, ae_emb], axis=1) with no
        # ID-based reindexing) would silently mix different patients'
        # data together here. Each patient's fused row must equal
        # THEIR OWN lstm value concatenated with THEIR OWN ae value,
        # regardless of storage order in either source array.
        lstm_ids = [3, 1, 2]          # patient 3's row is lstm_emb[0], etc.
        lstm_emb = np.array([[30.0], [10.0], [20.0]])
        ae_ids = [1, 2, 3]              # patient 1's row is ae_emb[0], etc.
        ae_emb = np.array([[100.0], [200.0], [300.0]])

        patient_ids, fused = fuse_embeddings(lstm_ids, lstm_emb, ae_ids, ae_emb, target_order=[1, 2, 3])

        self.assertEqual(patient_ids, [1, 2, 3])
        expected = np.array([
            [10.0, 100.0],   # patient 1: lstm=10 (from lstm_emb[1]), ae=100 (from ae_emb[0])
            [20.0, 200.0],   # patient 2: lstm=20 (from lstm_emb[2]), ae=200 (from ae_emb[1])
            [30.0, 300.0],   # patient 3: lstm=30 (from lstm_emb[0]), ae=300 (from ae_emb[2])
        ])
        np.testing.assert_array_equal(fused, expected)

    def test_uses_explicit_target_order_over_lstm_order(self):
        lstm_ids, lstm_emb = [1, 2], np.array([[1.0], [2.0]])
        ae_ids, ae_emb = [1, 2], np.array([[10.0], [20.0]])

        patient_ids, fused = fuse_embeddings(lstm_ids, lstm_emb, ae_ids, ae_emb, target_order=[2, 1])

        self.assertEqual(patient_ids, [2, 1])
        np.testing.assert_array_equal(fused, np.array([[2.0, 20.0], [1.0, 10.0]]))

    def test_defaults_to_lstm_order_when_target_order_omitted(self):
        lstm_ids, lstm_emb = [5, 6], np.array([[1.0], [2.0]])
        ae_ids, ae_emb = [6, 5], np.array([[60.0], [50.0]])  # different storage order

        patient_ids, fused = fuse_embeddings(lstm_ids, lstm_emb, ae_ids, ae_emb)

        self.assertEqual(patient_ids, [5, 6])
        np.testing.assert_array_equal(fused, np.array([[1.0, 50.0], [2.0, 60.0]]))

    def test_raises_when_autoencoder_missing_a_target_patient(self):
        lstm_ids, lstm_emb = [1, 2], np.array([[1.0], [2.0]])
        ae_ids, ae_emb = [1], np.array([[10.0]])  # missing patient 2
        with self.assertRaises(ValueError):
            fuse_embeddings(lstm_ids, lstm_emb, ae_ids, ae_emb, target_order=[1, 2])

    def test_output_dim_is_sum_of_both_modalities(self):
        lstm_ids, lstm_emb = [1], np.zeros((1, 128))
        ae_ids, ae_emb = [1], np.zeros((1, 64))
        _, fused = fuse_embeddings(lstm_ids, lstm_emb, ae_ids, ae_emb)
        self.assertEqual(fused.shape, (1, 192))


if __name__ == "__main__":
    unittest.main()
