"""Tests for `_hospital_model_seed()`, added 2026-09-20 to fix a real
non-determinism finding: `_train_one_hospital_flat_model()`'s model
construction was never explicitly seeded, so regenerating the same
combo's embeddings in a separate process produced different random
initial weights (caught via the sampling comparison's reproduction
check unexpectedly failing -- see
`_train_combo_and_extract_embeddings()`'s docstring in
embedding_ablation_runner.py for the full story).
"""

import unittest

from src.ablation.embedding_ablation_runner import _hospital_model_seed


class TestHospitalModelSeed(unittest.TestCase):
    def test_deterministic_for_identical_inputs(self):
        s1 = _hospital_model_seed(42, "ts-lstm_tab-mlp_fusion-beforeca", 73)
        s2 = _hospital_model_seed(42, "ts-lstm_tab-mlp_fusion-beforeca", 73)
        self.assertEqual(s1, s2)

    def test_different_hospitals_usually_get_different_seeds(self):
        seeds = {_hospital_model_seed(42, "ts-lstm_tab-mlp_fusion-beforeca", hid) for hid in range(1, 50)}
        # Not a mathematical guarantee of no collisions, but 49 hospital
        # ids should not collapse to a single seed -- if they did, every
        # hospital in a combo would get an IDENTICAL random init, which
        # is exactly the bug this function exists to avoid reintroducing.
        self.assertGreater(len(seeds), 40)

    def test_different_combos_usually_get_different_seeds_for_the_same_hospital(self):
        seeds = {
            _hospital_model_seed(42, combo_id, 73)
            for combo_id in [
                "ts-lstm_tab-mlp_fusion-beforeca", "ts-rnn_tab-mlp_fusion-beforeca",
                "ts-gru_tab-autoencoder_fusion-afterca", "ts-transformer_tab-autoencoder_fusion-beforeca",
            ]
        }
        self.assertEqual(len(seeds), 4)

    def test_different_base_seeds_change_the_result(self):
        s1 = _hospital_model_seed(42, "ts-lstm_tab-mlp_fusion-beforeca", 73)
        s2 = _hospital_model_seed(1, "ts-lstm_tab-mlp_fusion-beforeca", 73)
        self.assertNotEqual(s1, s2)

    def test_result_is_a_valid_torch_manual_seed_argument(self):
        # torch.manual_seed() requires a value within its accepted
        # 64-bit range; the modulus in _hospital_model_seed() must keep
        # every output non-negative and in range regardless of hid or
        # base_seed magnitude.
        for hid in (0, 1, 208, 999999):
            for base_seed in (0, 1, 42, 2**31):
                seed = _hospital_model_seed(base_seed, "ts-transformer_tab-pca_fusion-afterca", hid)
                self.assertGreaterEqual(seed, 0)
                self.assertLess(seed, 2**31 - 1)

    def test_not_using_pythons_randomized_string_hash(self):
        """Regression guard against reintroducing the exact bug this
        function was written to avoid: PYTHONHASHSEED-randomized
        hash(str) would make this non-deterministic across separate
        interpreter processes even with identical inputs. This process
        can only check internal consistency (see
        test_deterministic_for_identical_inputs for the real
        cross-call guarantee); this test instead pins a known
        crc32-based expected value so a future edit that swaps in
        hash() would change this constant and fail loudly."""
        # zlib.crc32(b"ts-lstm_tab-mlp_fusion-beforeca") is a fixed,
        # well-known constant -- independent of PYTHONHASHSEED.
        import zlib

        expected_combo_component = zlib.crc32(b"ts-lstm_tab-mlp_fusion-beforeca")
        base_seed, hid = 42, 73
        expected = (base_seed * 1_000_003 + expected_combo_component + hid) % (2**31 - 1)
        self.assertEqual(_hospital_model_seed(base_seed, "ts-lstm_tab-mlp_fusion-beforeca", hid), expected)


if __name__ == "__main__":
    unittest.main()
