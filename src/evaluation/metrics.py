from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from src.constants import DISEASES

@dataclass(frozen=True)     # 여러 결과를 하나의 객체에 묶어서 반환할 때 사용하는 문법
class MultilabelMetrics:
    mean_auroc: float
    mean_auprc: float
    mean_ece: float
    per_disease: dict[str, dict[str, float]]
    
def _binary_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
) -> float:
    """질환의 양성 확률에 대한 Binary ECE를 계산"""

    bins = np.minimum(
        (y_prob * n_bins).astype(int),
        n_bins - 1,
    )
    
    ece = 0.0
    
    for i in range(n_bins):
        mask = bins == i
        
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(y_true[mask].mean())
                - float(y_prob[mask].mean())
            )
    
    return ece

def evaluate_multilabel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> MultilabelMetrics:
    """14개 질환의 AUROC, AP 및 ECE를 계산"""
    
    y_true = np.asarray(y_true)     # 입력 데이터를 NumPy 배열 형태로 변환
    y_prob = np.asarray(y_prob)
    
    if y_true.ndim != 2 or y_true.shape != y_prob.shape:
        raise ValueError(
            "정답과 예측 확률은 동일한 2차원 배열이어야 합니다."
        )
    
    if y_true.shape[0] == 0 or y_true.shape[1] != len(DISEASES):
        raise ValueError(
            "배열 크기는 (이미지 수, 14)여야 합니다."
        )
        
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins는 2 이상의 정수여야 합니다.")
    
    if not np.isfinite(y_true).all() or not np.isfinite(y_prob).all():
        raise ValueError(
            "정답 및 예측값은 모두 유한해야 합니다."
        )
        
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("정답은 0 또는 1이어야 합니다.")
    
    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError(
            "예측값은 Sigmoid 적용 후 0~1 확률이어야 합니다."
        )
        
    per_disease: dict[str, dict[str, float]] = {}
    
    for i, disease in enumerate(DISEASES):
        target = y_true[:, i]
        prob = y_prob[:, i]
        
        if np.unique(target).size != 2:     # 해당 질환에 양성과 음성이 모두 존재하는지
            raise ValueError(
                f"{disease}에서 양성과 음성이 모두 필요합니다."
            )
        
        per_disease[disease] = {
            "AUROC": float(roc_auc_score(target, prob)),    # 모델이 양성 사례를 음성 사례보다 높은 점수로 평가하는 능력을 측정
            "AUPRC": float(average_precision_score(target, prob)),      # 정밀도와 재현율의 관계를 바탕으로 AP 계산
            "ECE": _binary_ece(target, prob, n_bins),
        }
        
    def mean(metric: str) -> float:
        return float(
            np.mean([
                row[metric]
                for row in per_disease.values()
            ])
        )
    
    return MultilabelMetrics(
        mean_auroc=mean("AUROC"),
        mean_auprc=mean("AUPRC"),
        mean_ece=mean("ECE"),
        per_disease=per_disease,
    )
