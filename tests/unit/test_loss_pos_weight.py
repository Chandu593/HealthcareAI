import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.training.losses import masked_bce_loss  # noqa: E402


class TestPosWeight(unittest.TestCase):
    def setUp(self):
        self.l = torch.tensor([0.2, -1.0, 0.5, 1.5]); self.y = torch.tensor([1.0, 0.0, float("nan"), 1.0])

    def test_default_is_unchanged_historical_behaviour(self):
        a, n = masked_bce_loss(self.l, self.y); b, _ = masked_bce_loss(self.l, self.y, pos_weight=None)
        self.assertEqual(float(a), float(b)); self.assertEqual(n, 3)

    def test_pos_weight_matches_manual_weighted_bce_and_ignores_nan(self):
        w = 4.0; c, n = masked_bce_loss(self.l, self.y, pos_weight=w)
        z = self.l[[0, 1, 3]]; t = torch.tensor([1.0, 0.0, 1.0])
        manual = -(w * t * torch.nn.functional.logsigmoid(z) + (1 - t) * torch.nn.functional.logsigmoid(-z)).mean()
        self.assertAlmostEqual(float(c), float(manual), places=6); self.assertEqual(n, 3)


if __name__ == "__main__":
    unittest.main()
