"""전체 실험을 GPU마다 하나씩 자동 배정해 실행 (Kaggle T4 ×2 등)

    python -m src.training.run_queue --gpus 0 1 --time-limit-hours 11 \
        --set raw_dir=/data --set output_dir=/out --set num_workers=2 --report-dir /reports

1단계  DenseNet-121 (설정 파일의 model) gamma 0/1/2 × 5-Fold
2단계  1단계가 모두 끝나면 selected_gamma.json 기준 최적 gamma로 EfficientNet-B0 × 5-Fold

- 작업 하나 = (설정, Fold) 하나. 비어 있는 GPU에 다음 작업을 바로 넣는다.
- 이미 끝난 Fold(result.json 존재)는 건너뛴다.
- 시간 제한이 되면 새 작업을 시작하지 않고, 진행 중인 Fold는 last.pt를 남기고 멈춘다.
  같은 명령을 다시 실행하면 남은 작업부터 이어서 진행한다.
- 작업별 로그: <output_dir>/_logs/<실험 ID>_fold<k>.log

종료 코드: 0 = 전체 완료, 75 = 시간 제한으로 멈춤(다시 실행), 1 = 실패한 작업 있음
"""
import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from src.paths import PROJECT_ROOT
from src.training.config import (
    DEFAULT_CONFIG_PATH, TrainConfig, load_config, parse_overrides,
)
from src.training.model import build_model
from src.training.run import EXIT_TIME_LIMIT
from src.training.summarize import (
    DEFAULT_REPORT_DIR, collect_results, comparison_group, select_gamma, summarize,
    write_summary,
)
from src.training.trainer import N_FOLDS, RESULT_FILE, fold_dir

DEFAULT_COMMAND = (sys.executable, "-m", "src.training.run")


@dataclass(frozen=True)
class Job:
    sets: tuple[str, ...]       # 이 작업에 넘길 --set 값들
    fold: int
    cfg: TrainConfig

    @property
    def label(self) -> str:
        return f"{self.cfg.run_id} | fold {self.fold}"

    @property
    def done(self) -> bool:
        return (fold_dir(self.cfg, self.fold) / RESULT_FILE).is_file()


def make_jobs(config: str, sets: list[str]) -> list[Job]:
    cfg = load_config(config, parse_overrides(sets))
    return [Job(tuple(sets), fold, cfg) for fold in range(N_FOLDS)]


def run_jobs(
    jobs: list[Job],
    gpus: list[str],
    config: str,
    log_dir: Path,
    deadline: float | None,
    min_start_seconds: float,
    poll_seconds: float,
    command: tuple[str, ...] = DEFAULT_COMMAND,
) -> dict[str, list[Job]]:
    """남은 작업을 GPU에 배정해 실행하고 상태별 작업 목록을 반환"""
    queue = [job for job in jobs if not job.done]
    running: dict[str, tuple[Job, subprocess.Popen, object]] = {}
    status: dict[str, list[Job]] = {"done": [], "paused": [], "failed": [], "not_started": []}
    stop_launching = False
    log_dir.mkdir(parents=True, exist_ok=True)

    while running or (queue and not stop_launching):
        # ---- 비어 있는 GPU에 작업 배정 ----
        for gpu in gpus:
            if gpu in running or not queue or stop_launching:
                continue
            remaining = deadline - time.time() if deadline is not None else None
            if remaining is not None and remaining < min_start_seconds:
                print(f"[queue] 남은 시간 {remaining / 60:.0f}분 → 새 작업을 시작하지 않음", flush=True)
                stop_launching = True
                break

            job = queue.pop(0)
            cmd = [*command, "--config", config, "--folds", str(job.fold), "--no-summary"]
            for item in job.sets:
                cmd += ["--set", item]
            if remaining is not None:
                cmd += ["--time-limit-hours", f"{remaining / 3600:.4f}"]

            log_path = log_dir / f"{job.cfg.run_id}_fold{job.fold}.log"
            log_file = open(log_path, "a", encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env,
                                    stdout=log_file, stderr=subprocess.STDOUT)
            running[gpu] = (job, proc, log_file)
            print(f"[queue] GPU {gpu} 시작: {job.label} (로그 {log_path.name})", flush=True)

        # ---- 끝난 작업 확인 ----
        for gpu, (job, proc, log_file) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log_file.close()
            del running[gpu]
            if code == 0 and job.done:
                status["done"].append(job)
                print(f"[queue] GPU {gpu} 완료: {job.label}", flush=True)
            elif code == EXIT_TIME_LIMIT:
                status["paused"].append(job)
                stop_launching = True
                print(f"[queue] GPU {gpu} 시간 제한으로 멈춤: {job.label}", flush=True)
            else:
                status["failed"].append(job)
                print(f"[queue] GPU {gpu} 실패(종료 코드 {code}): {job.label} → 로그 확인", flush=True)

        if running:
            time.sleep(poll_seconds)

    status["not_started"] = queue
    return status


def chosen_gamma(output_dir: Path, stage1_cfg: TrainConfig) -> float | None:
    """1단계 실험 그룹의 selected_gamma (5-Fold를 모두 마친 gamma가 2개 이상일 때)"""
    results = collect_results(output_dir)
    if not results:
        return None
    _, summary, _ = summarize(results)
    choice = select_gamma(summary).get(comparison_group(stage1_cfg.to_dict()))
    return None if choice is None else choice["gamma"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="전체 실험 자동 실행 (GPU별 작업 대기열)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="모든 작업에 공통으로 적용할 설정 (gamma는 지정 불가)")
    p.add_argument("--gpus", nargs="+", default=["0"], help="사용할 GPU 번호 (예: 0 1)")
    p.add_argument("--gammas", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    p.add_argument("--second-model", default="efficientnet_b0",
                   help="2단계 모델 ('none'이면 2단계 생략)")
    p.add_argument("--time-limit-hours", type=float, default=None)
    p.add_argument("--min-start-minutes", type=float, default=30.0,
                   help="남은 시간이 이보다 적으면 새 Fold를 시작하지 않음")
    p.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--dry-run", action="store_true", help="작업 계획만 출력")
    args = p.parse_args(argv)

    if any(item.split("=", 1)[0].strip() == "gamma" for item in args.overrides):
        p.error("gamma는 --gammas로 지정하세요.")
    return args


def main(argv=None, command: tuple[str, ...] = DEFAULT_COMMAND) -> int:
    started = time.time()
    args = parse_args(argv)
    deadline = (
        started + args.time_limit_hours * 3600
        if args.time_limit_hours is not None else None
    )
    common = list(args.overrides)

    base = load_config(args.config, parse_overrides(common))
    output_dir = base.resolve(base.output_dir)
    report_dir = Path(args.report_dir)
    log_dir = output_dir / "_logs"

    stage1 = [job for g in args.gammas for job in make_jobs(args.config, [*common, f"gamma={g:g}"])]
    print(f"[queue] 1단계: {base.model} gamma {[f'{g:g}' for g in args.gammas]} × {N_FOLDS}-Fold "
          f"→ 남은 작업 {sum(not j.done for j in stage1)}/{len(stage1)}")
    if args.dry_run:
        for job in stage1:
            print(f"  {'완료' if job.done else '대기'}  {job.label}")
        return 0

    # 사전학습 가중치를 미리 한 번 받아 둠 (여러 프로세스가 동시에 받다가 충돌하지 않게)
    if base.pretrained:
        models = {base.model} | ({args.second_model} if args.second_model != "none" else set())
        for name in sorted(models):
            build_model(name, pretrained=True)

    run = dict(gpus=args.gpus, config=args.config, log_dir=log_dir, deadline=deadline,
               min_start_seconds=args.min_start_minutes * 60, poll_seconds=args.poll_seconds,
               command=command)

    status = run_jobs(stage1, **run)
    write_summary(output_dir, report_dir)
    if status["failed"]:
        return 1
    if not all(job.done for job in stage1):
        print("[queue] 1단계 미완료 → 같은 명령을 다시 실행하면 이어서 진행합니다.")
        return EXIT_TIME_LIMIT

    if args.second_model == "none":
        print("[queue] 전체 완료")
        return 0

    gamma = args.gammas[0] if len(args.gammas) == 1 else chosen_gamma(output_dir, stage1[0].cfg)
    if gamma is None:
        print("[queue] 최적 gamma를 정할 수 없습니다 (selected_gamma.json 확인).")
        return 1

    stage2 = make_jobs(args.config, [*common, f"model={args.second_model}", f"gamma={gamma:g}"])
    print(f"[queue] 2단계: {args.second_model} gamma {gamma:g} × {N_FOLDS}-Fold "
          f"→ 남은 작업 {sum(not j.done for j in stage2)}/{len(stage2)}")
    status = run_jobs(stage2, **run)
    write_summary(output_dir, report_dir)
    if status["failed"]:
        return 1
    if not all(job.done for job in stage2):
        print("[queue] 2단계 미완료 → 같은 명령을 다시 실행하면 이어서 진행합니다.")
        return EXIT_TIME_LIMIT

    print("[queue] 전체 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
