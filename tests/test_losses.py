import unittest

import torch
import torch.nn.functional as F

from src.training.losses import FocalLoss

class TestFocalLoss(unittest.TestCase):

    def test_gamma_zero_matches_bce_with_pos_weight(self):
        logits = torch.randn(3, 14)
        targets = torch.randint(0, 2, (3, 14)).float()
        pos_weight = torch.linspace(1, 3, 14)

        actual = FocalLoss(
            gamma=0,
            pos_weight=pos_weight,
        )(logits,targets)


        expected = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight
        )

        torch.testing.assert_close(actual, expected)


    def test_gamma_two_reduces_easy_example_more(self):
        targets = torch.ones(1, 14)
        easy_logits = torch.full((1, 14), 5.0)
        hard_logits = torch.full((1, 14), -1.0)

        bce = FocalLoss(gamma=0)
        focal = FocalLoss(gamma=2)

        easy_ratio = (
            focal(easy_logits, targets)
            / bce(easy_logits, targets)
        )

        hard_ratio = (
            focal(hard_logits, targets)
            / bce(hard_logits, targets)
        )

        self.assertLess(
            easy_ratio.item(),
            hard_ratio.item(),
        )


    def test_backward_computes_finite_gradients(self):
        logits = torch.randn(
            3, 14, requires_grad=True
        )

        targets = torch.randint(
            0, 2, (3, 14)
        ).float()

        loss = FocalLoss(gamma=2)(logits, targets)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(
            torch.isfinite(logits.grad).all().item()
        )


    def test_mismatched_shapes_raise_error(self):
        with self.assertRaises(ValueError):
            FocalLoss()(
                torch.randn(2, 14),
                torch.zeros(2, 13),
            )

if __name__ == "__main__":
    unittest.main()