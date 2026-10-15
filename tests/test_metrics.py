import unittest

import numpy as np

from src.constants import DISEASES
from src.evaluation.metrics import evaluate_multilabel

class TestMultilabelMetrics(unittest.TestCase):
    
    def setUp(self):
        
        # 모든 질환에 양성 2개, 음성 2개 생성
        labels = np.array([0, 1, 0 ,1])
        
        self.targets = np.tile(
            labels[:, None],
            (1, len(DISEASES),)
        )
        
    
    def test_perfect_predictions(self):
        probabilities = self.targets.astype(float)
        
        result = evaluate_multilabel(
            self.targets,
            probabilities,
        )
        
        self.assertAlmostEqual(result.mean_auroc, 1.0)
        self.assertAlmostEqual(result.mean_auprc, 1.0)
        self.assertAlmostEqual(result.mean_ece, 0.0)
        
        self.assertEqual(len(result.per_disease), 14)
        
    def test_constant_predictions(self):
        probabilities = np.full(
            self.targets.shape,
            0.8,
        )
    
        result = evaluate_multilabel(
            self.targets,
            probabilities,
        )
        
        self.assertAlmostEqual(result.mean_auroc, 0.5)
        self.assertAlmostEqual(result.mean_auprc, 0.5)
        self.assertAlmostEqual(result.mean_ece, 0.3)
        
        
    def test_invalid_probabilities(self):
        probabilities = np.full(
            self.targets.shape,
            1.5,
        )
        
        with self.assertRaises(ValueError):
            evaluate_multilabel(
                self.targets,
                probabilities,
            )
            
    
    def test_missing_positive_class(self):
        targets = self.targets.copy()
        targets[:, 0] = 0
        
        with self.assertRaises(ValueError):
            evaluate_multilabel(
                targets,
                self.targets.astype(float),
            )
        
    def test_invalid_shape(self):
        probabilities = np.zeros((4, 13))
        
        with self.assertRaises(ValueError):
            evaluate_multilabel(
                self.targets,
                probabilities,
            )
            

if __name__ == "__mean__":
    unittest.main()