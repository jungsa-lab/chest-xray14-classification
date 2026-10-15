import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from training.focal_loss import FocalLoss
from evaluation.metrics import calculate_metrics


class TrainingTests(unittest.TestCase):
    def test_gamma_zero_equals_weighted_bce(self):
        torch.manual_seed(42)
        logits = torch.randn(4, 14, requires_grad=True)
        targets = torch.randint(0, 2, (4, 14)).float()
        pw = torch.linspace(1, 10, 14)
        got = FocalLoss(0, pw)(logits, targets)
        expected = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        self.assertTrue(torch.allclose(got, expected, atol=1e-7))

    def test_metrics_perfect_predictions(self):
        true = np.tile([[0] * 14, [1] * 14], (8, 1))
        probs = np.where(true == 1, 1., 0.)
        metrics = calculate_metrics(true, probs)
        self.assertAlmostEqual(metrics['mean_auroc'], 1.)
        self.assertAlmostEqual(metrics['mean_auprc'], 1.)
        self.assertAlmostEqual(metrics['ece'], 0.)


if __name__ == '__main__':
    unittest.main()
