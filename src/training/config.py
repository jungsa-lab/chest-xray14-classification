"""학습 설정: YAML 파일을 읽고 명령줄 값으로 덮어쓴다.

결과에 영향을 주는 설정값으로 짧은 해시(fingerprint)를 만들어 실험 ID에 붙인다.
설정이 하나라도 다르면 저장 폴더가 달라지므로 다른 실험의 결과를 실수로 재사용하지 않는다.
"""
import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from src.paths import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"

SCHEDULERS = ("plateau", "cosine")

# 결과(모델 가중치·지표)에 영향을 주지 않는 실행 환경 항목 → 실험 ID 계산에서 제외
RUNTIME_FIELDS = frozenset({
    "name", "device", "num_workers",
    "metadata_path", "raw_dir", "output_dir",
})


@dataclass(frozen=True)
class TrainConfig:
    name: str = ""
    model: str = "densenet121"
    pretrained: bool = True
    use_clahe: bool = True
    gamma: float = 2.0
    use_pos_weight: bool = True
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "plateau"
    plateau_factor: float = 0.1
    plateau_patience: int = 1
    early_stop_patience: int = 4
    min_delta: float = 1e-4
    amp: bool = True
    seed: int = 42
    device: str = "auto"
    num_workers: int = 4
    metadata_path: str = "data/processed/fold_metadata.csv"
    raw_dir: str = "data/raw"
    output_dir: str = "outputs/training"

    def __post_init__(self) -> None:
        for f in fields(self):
            object.__setattr__(
                self, f.name, _coerce(f.name, getattr(self, f.name), f.default)
            )

        positive = ("batch_size", "epochs", "lr", "early_stop_patience")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name}는 0보다 커야 합니다.")

        non_negative = ("gamma", "weight_decay", "min_delta", "plateau_patience", "num_workers", "seed")
        for name in non_negative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name}는 0 이상이어야 합니다.")

        # name은 저장 폴더 이름이 되므로 경로 구분자를 허용하지 않음
        if any(sep in self.name for sep in ("/", "\\")) or self.name in (".", ".."):
            raise ValueError(f"name에는 경로 구분자를 쓸 수 없습니다: {self.name!r}")

        if not 0 < self.plateau_factor < 1:
            raise ValueError("plateau_factor는 0과 1 사이여야 합니다.")

        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler는 {SCHEDULERS} 중 하나여야 합니다.")

    # ---- 실험 식별 ----

    def fingerprint(self) -> str:
        """결과에 영향을 주는 설정값의 해시 (8자리)"""
        payload = {
            k: v for k, v in sorted(asdict(self).items())
            if k not in RUNTIME_FIELDS
        }
        text = json.dumps(payload, sort_keys=True)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]

    @property
    def display_name(self) -> str:
        if self.name:
            return self.name
        clahe = "clahe" if self.use_clahe else "raw"
        return f"{self.model}_g{self.gamma:g}_{clahe}"

    @property
    def run_id(self) -> str:
        """저장 폴더 이름. 예: densenet121_g2_clahe-1a2b3c4d"""
        return f"{self.display_name}-{self.fingerprint()}"

    # ---- 기타 ----

    def resolve(self, path: str) -> Path:
        """상대 경로를 프로젝트 루트 기준 절대 경로로 변환"""
        p = Path(path).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(name: str, value: Any, default: Any) -> Any:
    """YAML·명령줄 값을 기본값과 같은 타입으로 변환

    PyYAML은 `1e-4`처럼 소수점 없는 지수 표기를 문자열로 읽으므로 float 변환이 필요하다.
    """
    kind = type(default)

    if kind is bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"{name}는 true/false여야 합니다: {value!r}")

    if kind is int:
        if isinstance(value, bool):
            raise ValueError(f"{name}는 정수여야 합니다: {value!r}")
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, int):
            return value
        raise ValueError(f"{name}는 정수여야 합니다: {value!r}")

    if kind is float:
        if isinstance(value, bool):
            raise ValueError(f"{name}는 숫자여야 합니다: {value!r}")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name}는 숫자여야 합니다: {value!r}") from None
        if not math.isfinite(number):
            raise ValueError(f"{name}는 유한한 숫자여야 합니다: {value!r}")
        return number

    if kind is str:
        if isinstance(value, str):
            return value
        raise ValueError(f"{name}는 문자열이어야 합니다: {value!r}")

    return value


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """['gamma=1', 'amp=false'] → {'gamma': 1, 'amp': False}"""
    overrides: dict[str, Any] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set 형식은 key=value 입니다: {item!r}")
        overrides[key.strip()] = yaml.safe_load(raw) if raw.strip() else ""
    return overrides


def load_config(
    path: str | Path | None = DEFAULT_CONFIG_PATH,
    overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    """YAML 설정을 읽고 overrides로 덮어쓴 TrainConfig 반환 (path=None이면 기본값)"""
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"설정 파일 최상위는 key: value 형식이어야 합니다: {path}")

    data.update(overrides or {})

    allowed = {f.name for f in fields(TrainConfig)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"알 수 없는 설정 항목: {sorted(unknown)}")

    return TrainConfig(**data)
