import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    densenet121,
)

from src.constants import DISEASES

def build_densenet121(
        pretrained: bool = True,    # 기본값 True: 사전 학습된 가중치를 사용할지 여부
) -> nn.Module:                     # 함수가 반환할 값의 타입
    """14개 질환 분류용 DenseNet-121 모델 생성"""

    weights = (                     # 가중치
        DenseNet121_Weights.IMAGENET1K_V1   # IMAGENET1K_V1은 1,000개 이미지 클래스를 분류하도록 학습된 DenseNet-121의 가중치 버전
        if pretrained   # if pretrained: weights = DenseNet121_Weights.IMAGENET1K_V1 else: weights = None
        else None
    )

    model = densenet121(weights=weights)

    in_features = model.classifier.in_features  # classifier에 입력되는 특징의 개수: 1024

    model.classifier = nn.Linear(       # nn.Linear(10, 3): 입력 특징 개수 10개, 출력 특징 개수 3개
        in_features,        
        len(DISEASES),
    )                                   # 입력 1024개, 출력 14개

    return model


# 설정 파일의 model 이름 → 생성 함수. 새 모델은 여기에 한 줄 추가
MODEL_BUILDERS = {
    "densenet121": build_densenet121,
}


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """설정 파일의 모델 이름으로 14개 질환 분류 모델 생성"""
    try:
        builder = MODEL_BUILDERS[name]
    except KeyError:
        raise ValueError(
            f"지원하지 않는 모델: {name} (지원: {sorted(MODEL_BUILDERS)})"
        ) from None

    return builder(pretrained=pretrained)
