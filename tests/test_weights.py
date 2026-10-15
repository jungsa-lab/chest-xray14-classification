import unittest

import pandas as pd
import torch

from src.constants import DISEASES
from src.training.weights import compute_pos_weight

class TestPosWeight(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({
            disease: [1, 0, 0, 0]
            for disease in DISEASES
        })
        
    def test_negative_positive_ratio(self):
        weights = compute_pos_weight(self.df)
        
        expected = torch.full((14,), 3.0)
        
        torch.testing.assert_close(weights, expected)
        
    def test_disease_specific_weights(self):
        self.df.loc[1, DISEASES[0]] = 1
        
        weights = compute_pos_weight(self.df)
        
        self.assertEqual(weights[0].item(), 1.0)
        self.assertEqual(weights[1].item(), 3.0)
        
    def test_no_positive_raises_error(self):
        self.df[DISEASES[0]] = 0
        
        with self.assertRaises(ValueError):
            compute_pos_weight(self.df)

    def test_invalid_label_raises_error(self):
        self.df.loc[0, DISEASES[0]] = 2
        
        with self.assertRaises(ValueError):
            compute_pos_weight(self.df)

if __name__ == "__main__":
    unittest.main()