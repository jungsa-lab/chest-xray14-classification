import unittest

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.training.engine import (
    train_one_epoch,
    validate_one_epoch,
)

from src.training.losses import FocalLoss


class DummyDataset(Dataset):
    """학습 루프 테스트를 위한 간단한 데이터셋."""
    
    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return {
            "image": self.images[index],
            "label": self.labels[index],
        }
        
class TestTrainEngine(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        
        self.device = torch.device("cpu")
        self.model = nn.Linear(4, 14)
        
        self.criterion = FocalLoss(gamma=2.0)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=0.01,
            weight_decay=0.01,
        )
        
        images = torch.randn(8, 4)
        labels = torch.randint(0, 2, (8, 14)).float()
        
        # samples = [
        #     {"image": image, "label": label}
        #     for image, label in zip(images, labels)
        # ]
        
        dataset = DummyDataset(images, labels)
        
        self.loader = DataLoader(
        #   samples,
            dataset,
            batch_size=4,
        )
    
    def test_training_updates_weights(self):
        before = [
            p.detach().clone()
            for p in self.model.parameters()
        ]
        
        loss = train_one_epoch(
            self.model,
            self.loader,
            self.criterion,
            self.optimizer,
            self.device,
        )
        
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(self.model.training)
        
        changed = any(
            not torch.equal(old, new)
            for old, new in zip(
                before, self.model.parameters()
            )
        )
        
        self.assertTrue(changed)
        
    def test_validation_does_not_update_weights(self):
        before = [
            p.detach().clone()
            for p in self.model.parameters()
        ]
        
        loss = validate_one_epoch(
            self.model,
            self.loader,
            self.criterion,
            self.device,
        )
        
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(self.model.training)
        
        for old, new in zip(
            before, self.model.parameters()
        ):
            torch.testing.assert_close(old, new)
            

if __name__ == "__main__":
    unittest.main()