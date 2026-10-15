import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """다중 라벨 분류용 Focal Loss."""

    def __init__(       # FocalLoss 생성자(초기화) 메서드
            self,
            gamma: float = 2.0,
            pos_weight: torch.Tensor | None = None,     # Tensor 객체 or None
    ) -> None:
        super().__init__()      # nn.Module 생성자(초기화) 메서드 호출

        if gamma < 0:
            raise ValueError("gamma는 0 이상이어야 합니다.")

        self.gamma = gamma

        if pos_weight is not None:
            if pos_weight.ndim != 1:
                raise ValueError("pos_weight는 1차원 텐서여야 합니다.")

            pos_weight = pos_weight.detach().clone().float()

        self.register_buffer("pos_weight", pos_weight)  # pos_weight: 질환별 양성 사례의 손실에 적용하는 가중치

    def forward(        # loss = loss_fn(logits, targets) << loss는 FocalLoss 객체
            self,
            logits: torch.Tensor,   # 모델이 예측한 원시 점수
            targets: torch.Tensor,  # 실제 정답(0 또는 1)
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError("logits와 targets의 크기가 같아야 합니다.")

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        if self.gamma == 0:
            return bce.mean()

        probs = torch.sigmoid(logits)   # Sigmoid 적용해서 0~1 사이의 확률로 변환

        p_t = (
            probs * targets
            + (1.0 - probs) * (1.0 - targets)
        )   # 정답이 1이면 예측한 양성 확률을, 정답이 0이면 예측한 음성 확률을 가져옴.

        focal_factor = (1.0 - p_t).pow(self.gamma)

        return (focal_factor * bce).mean()  # BCE 손실에 Focal Factor를 곱한 후 평균을 반환

