"""Tests for src/training/losses.py."""

import math
import unittest

import torch

from src.training.losses import compute_multitask_loss, masked_bce_loss, masked_focal_loss


class TestMaskedBceLoss(unittest.TestCase):
    def test_matches_manual_bce_on_known_subset_only(self):
        logits = torch.tensor([2.0, -2.0, 0.5, 10.0])
        labels = torch.tensor([1.0, 0.0, float("nan"), 1.0])  # index 2 unknown
        loss, n_used = masked_bce_loss(logits, labels)

        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([2.0, -2.0, 10.0]), torch.tensor([1.0, 0.0, 1.0]),
        )
        self.assertEqual(n_used, 3)
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)

    def test_all_unknown_returns_none(self):
        logits = torch.tensor([2.0, -2.0])
        labels = torch.tensor([float("nan"), float("nan")])
        loss, n_used = masked_bce_loss(logits, labels)
        self.assertIsNone(loss)
        self.assertEqual(n_used, 0)

    def test_no_unknown_uses_every_patient(self):
        logits = torch.tensor([1.0, -1.0, 0.3])
        labels = torch.tensor([1.0, 0.0, 1.0])
        _, n_used = masked_bce_loss(logits, labels)
        self.assertEqual(n_used, 3)

    def test_loss_is_differentiable(self):
        logits = torch.tensor([1.5, -0.5], requires_grad=True)
        labels = torch.tensor([1.0, 0.0])
        loss, _ = masked_bce_loss(logits, labels)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())


class TestMaskedFocalLoss(unittest.TestCase):
    def test_all_unknown_returns_none(self):
        logits = torch.tensor([2.0, -2.0])
        labels = torch.tensor([float("nan"), float("nan")])
        loss, n_used = masked_focal_loss(logits, labels)
        self.assertIsNone(loss)
        self.assertEqual(n_used, 0)

    def test_confident_correct_prediction_gives_lower_loss_than_wrong(self):
        """Core focal-loss property: down-weights easy (confidently
        correct) examples relative to hard (wrong) ones."""
        labels = torch.tensor([1.0])
        confident_correct = torch.tensor([8.0])  # sigmoid(8) ~= 1.0, matches label=1
        confident_wrong = torch.tensor([-8.0])   # sigmoid(-8) ~= 0.0, contradicts label=1
        loss_correct, _ = masked_focal_loss(confident_correct, labels)
        loss_wrong, _ = masked_focal_loss(confident_wrong, labels)
        self.assertLess(loss_correct.item(), loss_wrong.item())

    def test_loss_is_differentiable(self):
        logits = torch.tensor([1.5, -0.5], requires_grad=True)
        labels = torch.tensor([1.0, 0.0])
        loss, _ = masked_focal_loss(logits, labels)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())


class TestComputeMultitaskLoss(unittest.TestCase):
    def setUp(self):
        self.mortality_logits = torch.tensor([1.0, -1.0, 0.5])
        self.sepsis_logits = torch.tensor([0.2, -0.3, 1.0])
        self.mortality_labels = torch.tensor([1.0, 0.0, 1.0])
        self.sepsis_labels = torch.tensor([0.0, 1.0, 1.0])

    def test_raises_on_invalid_loss_type(self):
        with self.assertRaises(ValueError):
            compute_multitask_loss(
                self.mortality_logits, self.sepsis_logits, self.mortality_labels, self.sepsis_labels,
                1.0, 1.0, "cross_entropy",
            )

    def test_both_tasks_known_combines_with_configured_weights(self):
        total, components, counts = compute_multitask_loss(
            self.mortality_logits, self.sepsis_logits, self.mortality_labels, self.sepsis_labels,
            mortality_weight=2.0, sepsis_weight=0.5, loss_type="weighted_bce",
        )
        expected_mortality, _ = masked_bce_loss(self.mortality_logits, self.mortality_labels)
        expected_sepsis, _ = masked_bce_loss(self.sepsis_logits, self.sepsis_labels)
        expected_total = 2.0 * expected_mortality + 0.5 * expected_sepsis

        self.assertAlmostEqual(total.item(), expected_total.item(), places=5)
        self.assertIn("mortality_loss", components)
        self.assertIn("sepsis_loss", components)
        self.assertNotIn("reconstruction_loss", components)
        self.assertEqual(counts["n_mortality"], 3)
        self.assertEqual(counts["n_sepsis"], 3)

    def test_one_task_all_unknown_only_other_task_contributes(self):
        sepsis_labels_unknown = torch.full((3,), float("nan"))
        total, components, counts = compute_multitask_loss(
            self.mortality_logits, self.sepsis_logits, self.mortality_labels, sepsis_labels_unknown,
            mortality_weight=1.0, sepsis_weight=1.0, loss_type="weighted_bce",
        )
        expected_mortality, _ = masked_bce_loss(self.mortality_logits, self.mortality_labels)

        self.assertAlmostEqual(total.item(), expected_mortality.item(), places=5)
        self.assertIn("mortality_loss", components)
        self.assertNotIn("sepsis_loss", components)
        self.assertEqual(counts["n_sepsis"], 0)

    def test_both_tasks_all_unknown_no_reconstruction_returns_none(self):
        """Direct regression target: this is the signal the training
        loop must use to SKIP a hospital's step entirely -- calling
        .backward() on a fabricated zero-valued tensor here instead of
        None would either crash (no grad_fn) or, worse, silently do
        nothing while looking like a successful, perfectly-fit step."""
        mortality_unknown = torch.full((3,), float("nan"))
        sepsis_unknown = torch.full((3,), float("nan"))
        total, components, counts = compute_multitask_loss(
            self.mortality_logits, self.sepsis_logits, mortality_unknown, sepsis_unknown,
            mortality_weight=1.0, sepsis_weight=1.0, loss_type="weighted_bce",
        )
        self.assertIsNone(total)
        self.assertEqual(components, {})
        self.assertEqual(counts["n_mortality"], 0)
        self.assertEqual(counts["n_sepsis"], 0)

    def test_reconstruction_alone_rescues_the_all_unknown_case(self):
        """If both classification tasks are unusable for a hospital
        but reconstruction is enabled, the step should NOT be skipped
        -- the unsupervised reconstruction term never depends on label
        availability at all."""
        mortality_unknown = torch.full((3,), float("nan"))
        sepsis_unknown = torch.full((3,), float("nan"))
        reconstruction = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        tabular_target = torch.tensor([[1.1, 2.1], [2.9, 4.2], [5.0, 5.8]])

        total, components, _ = compute_multitask_loss(
            self.mortality_logits, self.sepsis_logits, mortality_unknown, sepsis_unknown,
            mortality_weight=1.0, sepsis_weight=1.0, loss_type="weighted_bce",
            reconstruction=reconstruction, tabular_target=tabular_target, reconstruction_weight=0.5,
        )
        self.assertIsNotNone(total)
        self.assertIn("reconstruction_loss", components)
        self.assertNotIn("mortality_loss", components)
        self.assertNotIn("sepsis_loss", components)

    def test_reconstruction_weight_zero_excludes_the_term_even_if_tensors_given(self):
        reconstruction = torch.tensor([[1.0, 2.0]])
        tabular_target = torch.tensor([[1.5, 2.5]])
        _, components, _ = compute_multitask_loss(
            self.mortality_logits, self.sepsis_logits, self.mortality_labels, self.sepsis_labels,
            mortality_weight=1.0, sepsis_weight=1.0, loss_type="weighted_bce",
            reconstruction=reconstruction, tabular_target=tabular_target, reconstruction_weight=0.0,
        )
        self.assertNotIn("reconstruction_loss", components)

    def test_total_loss_is_differentiable_end_to_end(self):
        mortality_logits = self.mortality_logits.clone().requires_grad_()
        sepsis_logits = self.sepsis_logits.clone().requires_grad_()
        total, _, _ = compute_multitask_loss(
            mortality_logits, sepsis_logits, self.mortality_labels, self.sepsis_labels,
            mortality_weight=1.0, sepsis_weight=1.0, loss_type="weighted_bce",
        )
        total.backward()
        self.assertIsNotNone(mortality_logits.grad)
        self.assertIsNotNone(sepsis_logits.grad)

    def test_focal_loss_type_also_works(self):
        total, components, _ = compute_multitask_loss(
            self.mortality_logits, self.sepsis_logits, self.mortality_labels, self.sepsis_labels,
            mortality_weight=1.0, sepsis_weight=1.0, loss_type="focal",
        )
        self.assertIsNotNone(total)
        self.assertFalse(math.isnan(total.item()))


if __name__ == "__main__":
    unittest.main()
