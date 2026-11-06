"""테스트용: 아주 작은 모델을 등록한 뒤 src.training.run을 실행한다.

run_queue는 작업마다 별도 프로세스를 띄우므로, 테스트에서는 이 모듈을 실행 명령으로 넘긴다.
    python -m tests._tiny_run --folds 0 --set model=tiny ...
"""
import sys

from torch import nn

from src.constants import DISEASES
from src.training import model as model_module


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=8, stride=8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, len(DISEASES)),
        )

    def forward(self, x):
        return self.body(x)


def register() -> None:
    """'tiny', 'tiny2' 두 이름으로 등록 (1단계·2단계 모델 구분용)"""
    model_module.MODEL_BUILDERS["tiny"] = lambda pretrained=True: TinyNet()
    model_module.MODEL_BUILDERS["tiny2"] = lambda pretrained=True: TinyNet()


if __name__ == "__main__":
    register()
    from src.training.run import main
    raise SystemExit(main(sys.argv[1:]))
