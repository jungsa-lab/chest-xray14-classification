
import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()

        if gamma < 0:
            raise ValueError("gamma must be non-negative")

        self.gamma = float(gamma)

        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else
            torch.as_tensor(
                pos_weight, dtype=torch.float32
            )
        )

    def forward(self, logits, targets):
        targets = targets.float()

        # DirectML에서 log_sigmoid 호출을 피하기 위한 방식
        probs = torch.sigmoid(logits)

        eps = 1e-7
        probs = probs.clamp(min=eps, max=1-eps)

        # 양성/음성 각각의 로그 손실
        positive_loss = -torch.log(probs)
        negative_loss = -torch.log(1.0 - probs)

        if self.pos_weight is not None:
            positive_loss = positive_loss * self.pos_weight

        bce = (
            targets * positive_loss
            + (1.0 - targets) * negative_loss
        )

        # 정답에 해당하는 예측 확률
        pt = (
            targets * probs
            + (1.0 - targets) * (1.0 - probs)
        )

        focal_factor = (1.0 - pt).pow(self.gamma)

        return (focal_factor * bce).mean()


def calculate_pos_weight(dataset):
    from dataset import DISEASES

    labels = torch.tensor(
        dataset.df[DISEASES].to_numpy(dtype="float32")
    )

    positive = labels.sum(dim=0)

    if (positive == 0).any():
        raise ValueError(
            "Positive sample count cannot be zero"
        )

    negative = len(dataset) - positive

    return negative / positive
