"""모델 학습 실행 (5-Fold 교차검증)

사용 예:
    # 설정과 저장 위치만 확인 (학습하지 않음)
    python -m src.training.run --dry-run

    # 코드 동작 확인: 소량 표본, 1 Epoch (결과는 _smoke 폴더, 요약에서 제외)
    python -m src.training.run --folds 0 --limit 256 --set epochs=1 --set pretrained=false

    # 실험 1: DenseNet-121, gamma 0/1/2 × 5-Fold (끝나면 gamma 비교표·최적 gamma 자동 생성)
    python -m src.training.run --gammas 0 1 2

    # 실험 2: EfficientNet-B0, 선택한 gamma로 5-Fold
    python -m src.training.run --set model=efficientnet_b0 --set gamma=2

    # 시간 제한 (Kaggle 12시간 세션 등): 11시간 안에 끝낼 수 없는 Epoch는 시작하지 않고 멈춤
    python -m src.training.run --gammas 0 1 2 --time-limit-hours 11

중단된 Fold는 다시 실행하면 마지막 Epoch 다음부터 이어서 학습한다.
완료된 Fold(result.json 존재)는 건너뛴다. 처음부터 다시 하려면 --force.

종료 코드: 0 = 모두 완료, 75 = 시간 제한으로 멈춤(다시 실행하면 이어서 진행)
"""
import argparse
import json
import time

import pandas as pd

from src.training.config import DEFAULT_CONFIG_PATH, load_config, parse_overrides
from src.training.summarize import write_summary
from src.training.trainer import (
    N_FOLDS, TimeLimitReached, fold_dir, select_device, train_fold,
)

EXIT_TIME_LIMIT = 75    # sysexits.h의 EX_TEMPFAIL: "잠시 후 다시 시도하면 되는 실패"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ChestX-ray14 모델 학습 (5-Fold 교차검증)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML 설정 파일")
    p.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                   help="설정값 덮어쓰기 (여러 번 사용 가능)")
    p.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    p.add_argument("--gammas", type=float, nargs="+", default=None,
                   help="여러 gamma를 차례로 학습 (생략하면 설정 파일의 gamma)")
    p.add_argument("--limit", type=int, default=None,
                   help="동작 확인용 학습 이미지 수 (결과는 _smoke 폴더, 요약 제외)")
    p.add_argument("--force", action="store_true", help="완료된 Fold도 처음부터 다시 학습")
    p.add_argument("--dry-run", action="store_true", help="설정과 저장 위치만 출력")
    p.add_argument("--time-limit-hours", type=float, default=None,
                   help="이 시간 안에 끝낼 수 없는 Epoch는 시작하지 않고 멈춤 (종료 코드 75)")
    p.add_argument("--no-summary", action="store_true",
                   help="끝난 뒤 요약표를 만들지 않음 (여러 프로세스를 동시에 돌릴 때)")
    args = p.parse_args(argv)

    if args.time_limit_hours is not None and args.time_limit_hours <= 0:
        p.error("--time-limit-hours는 0보다 커야 합니다.")

    if args.limit is not None and args.limit < 1:
        p.error("--limit은 1 이상이어야 합니다.")
    bad = [f for f in args.folds if f not in range(N_FOLDS)]
    if bad:
        p.error(f"--folds는 0~{N_FOLDS - 1} 중에서 고르세요: {bad}")
    return args


def main(argv=None) -> int:
    started = time.time()
    args = parse_args(argv)
    deadline = (
        started + args.time_limit_hours * 3600
        if args.time_limit_hours is not None else None
    )
    overrides = parse_overrides(args.overrides)
    if args.gammas is not None and "gamma" in overrides:
        raise SystemExit("--gammas와 --set gamma=...는 함께 쓸 수 없습니다.")

    base = load_config(args.config, overrides)
    gammas = args.gammas if args.gammas is not None else [base.gamma]
    configs = [load_config(args.config, {**overrides, "gamma": g}) for g in gammas]
    smoke = args.limit is not None

    print(f"장치 : {select_device(base.device)}")
    print(f"Fold : {args.folds}{' (smoke)' if smoke else ''}")
    for cfg in configs:
        print(f"실험 : {cfg.run_id}")
        for fold in args.folds:
            print(f"  fold {fold} → {fold_dir(cfg, fold, smoke=smoke)}")

    if args.dry_run:
        print(json.dumps(base.to_dict(), ensure_ascii=False, indent=2))
        return 0

    metadata = pd.read_csv(base.resolve(base.metadata_path), low_memory=False)
    code = 0
    try:
        for cfg in configs:
            for fold in args.folds:
                train_fold(cfg, fold, metadata=metadata, limit=args.limit,
                           force=args.force, deadline=deadline)
    except TimeLimitReached:
        print(f"시간 제한({args.time_limit_hours}시간)으로 멈춤. 같은 명령을 다시 실행하면 이어서 진행합니다.")
        code = EXIT_TIME_LIMIT

    if not smoke and not args.no_summary:
        write_summary(base.resolve(base.output_dir))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
