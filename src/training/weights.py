import pandas as pd
import torch

from src.constants import DISEASES

def compute_pos_weight(
    train_df: pd.DataFrame,
) -> torch.Tensor:
    """학습 데이터에서 질환별 양성 가중치를 계산"""
    
    if train_df.empty:
        raise ValueError("학습 데이터가 비어 있습니다.")
    
    missing = set(DISEASES) - set(train_df.columns)
    
    if missing:
        raise ValueError(
            f"질환 열이 없습니다.: {sorted(missing)}"
        )
    
    labels = train_df[DISEASES]
    
    if (
        labels.isna().any().any()
        or not labels.isin([0, 1]).all().all()
    ):
        raise ValueError(
            "질환 라벨은 0 또는 1이어야 합니다."
        )
    
    positives = labels.sum(axis=0)      # 질환별 양성 이미지 수
    negatives = len(labels) - positives # 질환별 음성 이미지 수
    
    invalid = [
        disease for disease in DISEASES
        if positives[disease] == 0
        or negatives[disease] == 0
    ]
    
    if invalid:
        raise ValueError(
            f"양성 또는 음성이 없는 질환: {invalid}"
        )
        
    weights = negatives / positives # 질환별 양성 가중치 계산
    
    return torch.tensor(
        weights.to_numpy(dtype="float32")   # 14개의 숫자를 가진 Tensor
    )