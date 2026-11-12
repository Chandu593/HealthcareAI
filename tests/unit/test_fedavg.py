"""Tests for src/training/fedavg.py."""

import unittest

import torch

from src.training.fedavg import fedavg_aggregate


class TestFedavgAggregate(unittest.TestCase):
    def test_weighted_average_matches_hand_calculation(self):
        state_dicts = {
            "1": {"w": torch.tensor([1.0, 2.0])},
            "2": {"w": torch.tensor([3.0, 4.0])},
        }
        weights = {"1": 1, "2": 3}  # hospital 2 has 3x hospital 1's patients
        result = fedavg_aggregate(state_dicts, weights)
        # (1*[1,2] + 3*[3,4]) / 4 = ([1,2] + [9,12]) / 4 = [10,14] / 4 = [2.5, 3.5]
        torch.testing.assert_close(result["w"], torch.tensor([2.5, 3.5]))

    def test_equal_weights_equals_unweighted_mean(self):
        state_dicts = {
            "1": {"w": torch.tensor([2.0, 4.0])},
            "2": {"w": torch.tensor([6.0, 8.0])},
        }
        weights = {"1": 50, "2": 50}
        result = fedavg_aggregate(state_dicts, weights)
        torch.testing.assert_close(result["w"], torch.tensor([4.0, 6.0]))

    def test_single_hospital_returns_that_hospitals_weights_unchanged(self):
        state_dicts = {"1": {"w": torch.tensor([7.0, -3.0])}}
        weights = {"1": 42}
        result = fedavg_aggregate(state_dicts, weights)
        torch.testing.assert_close(result["w"], torch.tensor([7.0, -3.0]))

    def test_preserves_multiple_keys_independently(self):
        state_dicts = {
            "1": {"a": torch.tensor([1.0]), "b": torch.tensor([10.0])},
            "2": {"a": torch.tensor([3.0]), "b": torch.tensor([20.0])},
        }
        weights = {"1": 1, "2": 1}
        result = fedavg_aggregate(state_dicts, weights)
        torch.testing.assert_close(result["a"], torch.tensor([2.0]))
        torch.testing.assert_close(result["b"], torch.tensor([15.0]))

    def test_preserves_original_dtype(self):
        state_dicts = {
            "1": {"w": torch.tensor([1.0, 2.0], dtype=torch.float32)},
            "2": {"w": torch.tensor([3.0, 4.0], dtype=torch.float32)},
        }
        weights = {"1": 1, "2": 1}
        result = fedavg_aggregate(state_dicts, weights)
        self.assertEqual(result["w"].dtype, torch.float32)

    def test_raises_on_empty_state_dicts(self):
        with self.assertRaises(ValueError):
            fedavg_aggregate({}, {})

    def test_raises_on_missing_weight(self):
        state_dicts = {"1": {"w": torch.tensor([1.0])}, "2": {"w": torch.tensor([2.0])}}
        weights = {"1": 10}  # "2" missing
        with self.assertRaises(ValueError) as ctx:
            fedavg_aggregate(state_dicts, weights)
        self.assertIn("2", str(ctx.exception))

    def test_raises_on_zero_total_weight(self):
        state_dicts = {"1": {"w": torch.tensor([1.0])}}
        weights = {"1": 0}
        with self.assertRaises(ValueError):
            fedavg_aggregate(state_dicts, weights)

    def test_raises_on_key_mismatch(self):
        state_dicts = {
            "1": {"a": torch.tensor([1.0])},
            "2": {"b": torch.tensor([2.0])},
        }
        weights = {"1": 1, "2": 1}
        with self.assertRaises(ValueError) as ctx:
            fedavg_aggregate(state_dicts, weights)
        self.assertIn("2", str(ctx.exception))

    def test_raises_on_shape_mismatch(self):
        state_dicts = {
            "1": {"w": torch.tensor([1.0, 2.0])},
            "2": {"w": torch.tensor([1.0, 2.0, 3.0])},
        }
        weights = {"1": 1, "2": 1}
        with self.assertRaises(ValueError) as ctx:
            fedavg_aggregate(state_dicts, weights)
        self.assertIn("shape", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
