import unittest

import numpy as np
from sklearn.decomposition import PCA

from src.modalities.tabular.pca_encoder import (
    PCAEncoderConfig,
    PerHospitalPCAEncoder,
    fit_transform_per_hospital_pca,
)


class TestPerHospitalPCAEncoder(unittest.TestCase):
    def test_fit_transform_shape(self):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 20).astype(np.float32)
        enc = PerHospitalPCAEncoder(PCAEncoderConfig(input_dim=20, embedding_dim=8))
        enc.fit(X)
        out = enc.transform(X)
        self.assertEqual(out.shape, (50, 8))

    def test_transform_before_fit_raises(self):
        enc = PerHospitalPCAEncoder(PCAEncoderConfig(input_dim=10, embedding_dim=4))
        with self.assertRaises(RuntimeError):
            enc.transform(np.zeros((5, 10), dtype=np.float32))

    def test_fit_matches_direct_sklearn_pca_on_the_same_train_slice(self):
        """The embedding must come from a PCA fit on exactly the given
        rows -- not some other subset, not the full array."""
        rng = np.random.RandomState(1)
        X = rng.randn(30, 12).astype(np.float32)
        train_idx = np.arange(0, 20)  # first 20 rows only

        enc = PerHospitalPCAEncoder(PCAEncoderConfig(input_dim=12, embedding_dim=5))
        enc.fit(X[train_idx])
        got = enc.transform(X)

        reference = PCA(n_components=5, random_state=0)
        reference.fit(X[train_idx])
        expected = reference.transform(X)
        np.testing.assert_allclose(got, expected.astype(np.float32), rtol=1e-4, atol=1e-4)

    def test_pads_with_zeros_when_hospital_too_small_for_full_embedding_dim(self):
        rng = np.random.RandomState(2)
        X = rng.randn(3, 20).astype(np.float32)  # only 3 rows -> max 3 components, requested 8
        enc = PerHospitalPCAEncoder(PCAEncoderConfig(input_dim=20, embedding_dim=8))
        enc.fit(X)
        out = enc.transform(X)
        self.assertEqual(out.shape, (3, 8))
        # Columns beyond what PCA could actually produce must be exactly zero.
        np.testing.assert_array_equal(out[:, enc._n_components_fit:], 0.0)

    def test_fit_requires_at_least_two_rows(self):
        enc = PerHospitalPCAEncoder(PCAEncoderConfig(input_dim=10, embedding_dim=4))
        with self.assertRaises(ValueError):
            enc.fit(np.zeros((1, 10), dtype=np.float32))


class TestFitTransformPerHospitalPCA(unittest.TestCase):
    def test_hospitals_are_fully_isolated_from_each_other(self):
        """Revert->fail->restore->pass-style proof: changing hospital B's
        data must NEVER change hospital A's PCA embedding -- this is
        the direct regression target for the 'never pool across
        hospitals' privacy guarantee this module exists to uphold."""
        rng = np.random.RandomState(3)
        X_a = rng.randn(20, 10).astype(np.float32)
        X_b_original = rng.randn(15, 10).astype(np.float32)
        train_idx = {"A": np.arange(15), "B": np.arange(10)}

        data_by_hid = {"A": X_a, "B": X_b_original}
        embeddings_before, _ = fit_transform_per_hospital_pca(data_by_hid, train_idx, embedding_dim=4)

        # Now corrupt hospital B's data completely (simulating a bug
        # that pools/leaks data) and re-run.
        X_b_corrupted = rng.randn(15, 10).astype(np.float32) * 1000
        data_by_hid_corrupted = {"A": X_a, "B": X_b_corrupted}
        embeddings_after, _ = fit_transform_per_hospital_pca(data_by_hid_corrupted, train_idx, embedding_dim=4)

        np.testing.assert_array_equal(embeddings_before["A"], embeddings_after["A"])
        # Sanity: hospital B's own embedding DID change (proves the
        # test setup is actually exercising something, not a no-op).
        self.assertFalse(np.allclose(embeddings_before["B"], embeddings_after["B"]))

    def test_too_few_train_rows_skips_gracefully_with_zero_embedding(self):
        data_by_hid = {"tiny": np.random.randn(5, 8).astype(np.float32)}
        train_idx = {"tiny": np.array([0])}  # only 1 train row
        embeddings, skipped = fit_transform_per_hospital_pca(data_by_hid, train_idx, embedding_dim=4)
        self.assertIn("tiny", skipped)
        self.assertEqual(skipped["tiny"], 1)
        np.testing.assert_array_equal(embeddings["tiny"], np.zeros((5, 4), dtype=np.float32))

    def test_val_test_rows_never_influence_the_fitted_basis(self):
        """Fitting must use ONLY the given train row indices -- val/test
        rows present in the same per-hospital array must have zero
        influence on the PCA components, even though they ARE present
        in the array passed in (they still need to be transformed)."""
        rng = np.random.RandomState(4)
        X_train_rows = rng.randn(20, 10).astype(np.float32)
        # Extreme outlier rows that would drastically change a PCA
        # basis if they leaked into fitting.
        X_val_test_rows = rng.randn(5, 10).astype(np.float32) * 500
        X_full = np.concatenate([X_train_rows, X_val_test_rows], axis=0)
        train_idx = {"H": np.arange(20)}

        embeddings, _ = fit_transform_per_hospital_pca({"H": X_full}, train_idx, embedding_dim=4)

        reference = PCA(n_components=4, random_state=0)
        reference.fit(X_train_rows)  # fit on train rows only, exactly what should have happened internally
        expected_train_embeddings = reference.transform(X_train_rows)
        np.testing.assert_allclose(embeddings["H"][:20], expected_train_embeddings.astype(np.float32), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
