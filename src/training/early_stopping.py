import math

class EarlyStopping:
    """검증 AUROC가 일정 Epoch 동안 개선되지 않으면 학습 종료를 알림"""

    def __init__(
            self,
            patience: int = 5,              # 성능 개선 없이 허용할 Epoch 수
            min_delta: float = 0.0,         # 성능 개선으로 인정할 최소 증가량
    ) -> None:
        if (
            isinstance(patience, bool)
            or not isinstance(patience, int)
            or patience < 1
        ):
            raise ValueError(
                "patience는 1 이상의 정수여야 합니다."
            )

        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError(
                "min_detal는 0 이상의 유한한 값이어야 합니다."
            )

        self.patience = patience
        self.min_delta = min_delta

        self.best_score: float | None = None    # 지금까지 기록한 최고 검증 AUROC
        self.bad_epochs = 0                     # 연속으로 개선되지 않은 Epoch 수
        self.improved = False                   # 이번 Epoch에서 최고 성능을 갱신했는지
        self.should_stop = False

    def step(self, val_auroc: float) -> bool:
        """검증 AUROC를 전달하고 학습 중단 여부를 반환"""

        if (
            not math.isfinite(val_auroc)
            or not 0.0 <= val_auroc <= 1.0
        ):
            raise ValueError(
                "val_auroc는 0~1 사이의 유한한 값이어야 합니다."
            )

        self.improved = False

        if self.should_stop:
            return True

        if (
            self.best_score is None
            or val_auroc > self.best_score + self.min_delta
        ):
            self.best_score = val_auroc
            self.bad_epochs = 0
            self.improved = True
        else:
            self.bad_epochs += 1

            if self.bad_epochs >= self.patience:
                self.should_stop = True

        return self.should_stop

    