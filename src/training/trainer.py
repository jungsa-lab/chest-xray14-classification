"""Fold 하나를 학습한다: 학습 → 검증 → 체크포인트 → Early Stopping.

저장 위치: <output_dir>/<run_id>/fold_<k>/
    best.pt              검증 Mean AUROC가 가장 높았던 시점의 모델 가중치
    val_predictions.csv  그 시점의 검증 정답과 logit (임계값·Calibration 분석용)
    history.csv          Epoch별 학습률, Loss, 검증 지표
    result.json          Fold 최종 결과 (있으면 완료된 Fold로 보고 건너뜀)
    last.pt              학습 중에만 존재. 중단되면 다음 실행에서 이어서 학습

Test 데이터는 이 단계에서 전혀 사용하지 않는다.
"""
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.constants import DISEASES
from src.data.chest_xray_dataset import ChestXrayDataset, select_fold
from src.evaluation.metrics import MultilabelMetrics, evaluate_multilabel
from src.training.config import TrainConfig
from src.training.losses import FocalLoss
from src.training.model import build_model
from src.training.weights import compute_pos_weight

N_FOLDS = 5
SMOKE_DIR = "_smoke"            # --limit 실행 결과는 여기에 따로 저장하고 요약에서 제외

BEST_FILE = "best.pt"
LAST_FILE = "last.pt"
RESULT_FILE = "result.json"
HISTORY_FILE = "history.csv"
PREDICTION_FILE = "val_predictions.csv"

Logger = Callable[[str], None]


class TimeLimitReached(Exception):
    """시간 제한 안에 다음 Epoch를 끝낼 수 없어 멈춤. last.pt가 남아 있어 다시 실행하면 이어서 학습한다."""


# ---------------------------------------------------------------------------
# 실행 환경
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    """DataLoader worker마다 numpy·random 난수도 고정"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device(name: str = "auto") -> torch.device:
    """auto면 CUDA → MPS → CPU 순서로 선택"""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fold_dir(cfg: TrainConfig, fold: int, smoke: bool = False) -> Path:
    root = cfg.resolve(cfg.output_dir)
    if smoke:
        root = root / SMOKE_DIR
    return root / cfg.run_id / f"fold_{fold}"


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

@dataclass
class BestTracker:
    """검증 점수의 최고 기록을 추적하고, 개선이 멈추면 종료 시점을 알린다."""

    patience: int
    min_delta: float = 0.0
    best_score: float = -math.inf
    best_epoch: int = 0
    stale_epochs: int = 0       # 마지막 개선 이후 지난 Epoch 수

    def update(self, score: float, epoch: int) -> bool:
        """이번 점수가 최고 기록을 갱신했으면 True"""
        if not math.isfinite(score):
            raise ValueError(f"검증 점수가 유한하지 않습니다: {score}")

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.stale_epochs = 0
            return True

        self.stale_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.stale_epochs >= self.patience


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """동작 확인용 소량 표본

    지표 계산에는 질환마다 양성·음성이 모두 있어야 하므로 질환별로 양성 2장, 음성 1장을 먼저 넣고
    나머지는 무작위로 채운다. (그래서 결과가 n보다 클 수 있다.)
    """
    shuffled = df.sample(frac=1.0, random_state=seed)
    keep: list = []
    for disease in DISEASES:
        keep.extend(shuffled.index[shuffled[disease].eq(1)][:2])
        keep.extend(shuffled.index[shuffled[disease].eq(0)][:1])
    keep = list(dict.fromkeys(keep))

    rest = shuffled.index.difference(keep, sort=False)
    keep.extend(rest[: max(0, n - len(keep))])
    return df.loc[keep]


def make_loader(
    df: pd.DataFrame,
    cfg: TrainConfig,
    device: torch.device,
    train: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = ChestXrayDataset(df, cfg.resolve(cfg.raw_dir), use_clahe=cfg.use_clahe)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        generator=generator,
        num_workers=cfg.num_workers,
        worker_init_fn=_seed_worker,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )


# ---------------------------------------------------------------------------
# 1 Epoch 학습·검증
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> tuple[float, int]:
    """1 Epoch 학습. (평균 Loss, 건너뛴 step 수) 반환

    AMP에서 float16 overflow로 Loss나 기울기가 inf/nan이면 GradScaler가 그 step을 건너뛰고
    scale을 줄인다. 이런 step의 개수를 세고, Loss가 inf/nan인 step은 평균 Loss에서 제외한다.
    AMP를 쓰지 않는데 Loss가 유한하지 않으면 학습이 망가진 것이므로 즉시 중단한다.
    """
    model.train()
    total_loss, total_samples, skipped = 0.0, 0, 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True, memory_format=memory_format)
        targets = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
        loss = criterion(logits.float(), targets)   # Loss는 float32로 계산

        loss_value = loss.item()
        finite = math.isfinite(loss_value)
        if not finite and not amp:
            raise FloatingPointError("학습 Loss가 유한하지 않습니다.")

        # AMP가 꺼져 있으면 scaler는 그대로 backward·step만 수행
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Loss가 정상이어도 기울기에 inf가 있으면 scaler가 step을 건너뛰고 scale을 줄인다
        if not finite or (amp and scaler.get_scale() < scale_before):
            skipped += 1
        if not finite:
            continue

        total_loss += loss_value * images.size(0)
        total_samples += images.size(0)

    if total_samples == 0:
        raise FloatingPointError("유효한 학습 step이 하나도 없습니다.")

    return total_loss / total_samples, skipped


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> tuple[float, np.ndarray, np.ndarray]:
    """검증 데이터 전체의 (평균 Loss, 정답, logit) 반환"""
    model.eval()
    total_loss, total_samples = 0.0, 0
    labels, logits_all = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True, memory_format=memory_format)
        targets = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
        logits = logits.float()

        total_loss += criterion(logits, targets).item() * images.size(0)
        total_samples += images.size(0)
        labels.append(batch["label"].numpy())
        logits_all.append(logits.cpu().numpy())

    if total_samples == 0:
        raise ValueError("검증 데이터가 비어 있습니다.")

    return (
        total_loss / total_samples,
        np.concatenate(labels).astype(np.float32),
        np.concatenate(logits_all).astype(np.float32),
    )


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return torch.sigmoid(torch.from_numpy(logits)).numpy()


# ---------------------------------------------------------------------------
# 저장
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """임시 파일에 쓴 뒤 교체 → 저장 도중 중단돼도 기존 파일이 깨지지 않음"""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    tmp.replace(path)


def save_predictions(df: pd.DataFrame, y_true: np.ndarray, logits: np.ndarray, path: Path) -> None:
    """검증 정답과 logit 저장 (확률은 sigmoid(logit))"""
    expected = df[DISEASES].to_numpy(dtype=np.float32)
    if not np.array_equal(expected, y_true):
        raise RuntimeError("예측 순서가 메타데이터 순서와 다릅니다.")

    out = df[["Image Index", "Patient ID"]].reset_index(drop=True)
    labels = pd.DataFrame(y_true.astype(int), columns=[f"label_{d}" for d in DISEASES])
    scores = pd.DataFrame(logits, columns=[f"logit_{d}" for d in DISEASES])
    table = pd.concat([out, labels, scores], axis=1)
    _atomic_write(path, lambda p: table.to_csv(p, index=False))


def _rng_state(train_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": train_generator.get_state(),
    }


def _restore_rng_state(state: dict, train_generator: torch.Generator) -> None:
    # RNG 상태는 반드시 CPU ByteTensor여야 한다
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    train_generator.set_state(state["loader"].cpu())


# ---------------------------------------------------------------------------
# Fold 학습
# ---------------------------------------------------------------------------

def train_fold(
    cfg: TrainConfig,
    fold: int,
    metadata: pd.DataFrame | None = None,
    limit: int | None = None,
    force: bool = False,
    log: Logger = print,
    deadline: float | None = None,
    first_epoch_estimate: float = 1800.0,
) -> dict:
    """Fold 하나를 학습하고 최고 검증 성능 시점의 결과를 반환

    - result.json이 있으면 학습하지 않고 저장된 결과를 반환 (force=True면 처음부터 다시)
    - last.pt가 있으면 그 Epoch 다음부터 이어서 학습
    - limit을 주면 소량 표본으로 동작만 확인하고, 결과는 _smoke 폴더에 저장
    - deadline(time.time() 기준 시각)을 주면, 다음 Epoch가 그 전에 끝나지 않을 것 같을 때
      TimeLimitReached를 발생시키고 멈춘다. Epoch 시간은 직전 Epoch로 추정하고,
      아직 기록이 없으면 first_epoch_estimate(초)로 본다.
    """
    if fold not in range(N_FOLDS):
        raise ValueError(f"fold는 0~{N_FOLDS - 1} 중 하나여야 합니다: {fold}")

    smoke = limit is not None
    out = fold_dir(cfg, fold, smoke=smoke)
    tag = f"[{cfg.run_id} | fold {fold}{' | smoke' if smoke else ''}]"

    result_path, last_path = out / RESULT_FILE, out / LAST_FILE
    if force:
        for name in (RESULT_FILE, LAST_FILE, BEST_FILE, HISTORY_FILE, PREDICTION_FILE):
            (out / name).unlink(missing_ok=True)
    elif result_path.is_file():
        log(f"{tag} 완료된 결과가 있어 건너뜀 → {result_path}")
        return json.loads(result_path.read_text(encoding="utf-8"))

    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seed_everything(cfg.seed)
    device = select_device(cfg.device)
    cuda = device.type == "cuda"
    amp = cfg.amp and cuda

    # 속도 옵션은 CUDA에서만 적용 (MPS·CPU에서는 효과가 없거나 오히려 느림)
    channels_last = cfg.channels_last and cuda
    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    if cuda:
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    # ---- 데이터 ----
    if metadata is None:
        metadata = pd.read_csv(cfg.resolve(cfg.metadata_path), low_memory=False)
    train_df, val_df = select_fold(metadata, fold)
    if smoke:
        train_df = subsample(train_df, limit, cfg.seed)
        val_df = subsample(val_df, max(limit // 4, 1), cfg.seed)

    train_generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = make_loader(train_df, cfg, device, train=True, generator=train_generator)
    val_loader = make_loader(val_df, cfg, device, train=False)

    # ---- 모델·손실·최적화 ----
    model = build_model(cfg.model, pretrained=cfg.pretrained).to(device, memory_format=memory_format)

    # 양성 가중치는 이번 Fold의 학습 데이터로만 계산 (검증 라벨 누수 방지)
    pos_weight = compute_pos_weight(train_df) if cfg.use_pos_weight else None
    criterion = FocalLoss(gamma=cfg.gamma, pos_weight=pos_weight).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Cosine Annealing: epochs 동안 lr을 코사인 곡선으로 0 근처까지 감소 (Epoch 단위 step)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    tracker = BestTracker(patience=cfg.early_stop_patience, min_delta=cfg.min_delta)

    history: list[dict] = []
    best_metrics: dict | None = None
    start_epoch = 1
    elapsed_before = 0.0        # 중단 전까지 학습에 쓴 시간 (이어서 학습할 때)

    def elapsed() -> float:
        return elapsed_before + time.time() - started

    # ---- 중단된 학습 이어서 하기 ----
    if last_path.is_file():
        # RNG 상태(numpy 배열 등)가 들어 있어 weights_only=False. 직접 만든 파일만 읽는다.
        # CPU로 읽어야 RNG 상태를 복원할 수 있다. 모델·optimizer 상태는 load_state_dict가 장치로 옮긴다.
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        if ckpt["fingerprint"] != cfg.fingerprint():
            raise RuntimeError(f"{last_path}는 다른 설정으로 만든 체크포인트입니다.")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        tracker = BestTracker(**ckpt["tracker"])
        history = ckpt["history"]
        best_metrics = ckpt["best_metrics"]
        start_epoch = ckpt["epoch"] + 1
        elapsed_before = ckpt["elapsed"]
        _restore_rng_state(ckpt["rng"], train_generator)
        log(f"{tag} epoch {ckpt['epoch']}까지 학습된 체크포인트에서 이어서 학습")

    log(
        f"{tag} device={device} amp={amp} channels_last={channels_last} "
        f"train={len(train_df):,} val={len(val_df):,} epochs≤{cfg.epochs}"
    )

    # ---- 학습 루프 ----
    for epoch in range(start_epoch, cfg.epochs + 1):
        if tracker.should_stop:
            break

        if deadline is not None:
            estimate = history[-1]["seconds"] if history else first_epoch_estimate
            if time.time() + estimate > deadline:
                log(
                    f"{tag} 시간 제한: epoch {epoch} (예상 {estimate:.0f}s)을 끝낼 시간이 없어 멈춤. "
                    f"다시 실행하면 이어서 학습"
                )
                raise TimeLimitReached(f"{tag} epoch {epoch} 전에 시간 제한")

        epoch_started = time.time()
        lr = optimizer.param_groups[0]["lr"]

        train_loss, skipped = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp, memory_format
        )
        val_loss, y_true, logits = evaluate(
            model, val_loader, criterion, device, amp, memory_format
        )
        metrics: MultilabelMetrics = evaluate_multilabel(y_true, sigmoid(logits))

        scheduler.step()

        improved = tracker.update(metrics.mean_auroc, epoch)
        if improved:
            best_metrics = asdict(metrics)
            _atomic_write(out / BEST_FILE, lambda p: torch.save({
                "model_name": cfg.model,
                "state_dict": model.state_dict(),
                "diseases": list(DISEASES),
                "fold": fold,
                "epoch": epoch,
                "val_mean_auroc": metrics.mean_auroc,
                "run_id": cfg.run_id,
                "config": cfg.to_dict(),
            }, p))
            save_predictions(val_df, y_true, logits, out / PREDICTION_FILE)

        history.append({
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_auroc": metrics.mean_auroc,
            "val_mean_auprc": metrics.mean_auprc,
            "val_mean_ece": metrics.mean_ece,
            "improved": improved,
            "skipped_steps": skipped,
            "seconds": round(time.time() - epoch_started, 1),
        })
        _atomic_write(out / HISTORY_FILE, lambda p: pd.DataFrame(history).to_csv(p, index=False))

        _atomic_write(last_path, lambda p: torch.save({
            "fingerprint": cfg.fingerprint(),
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "tracker": asdict(tracker),
            "history": history,
            "best_metrics": best_metrics,
            "elapsed": elapsed(),
            "rng": _rng_state(train_generator),
        }, p))

        log(
            f"  epoch {epoch:02d} | lr {lr:.1e} | train {train_loss:.4f} | val {val_loss:.4f} "
            f"| AUROC {metrics.mean_auroc:.4f} | AUPRC {metrics.mean_auprc:.4f} "
            f"| ECE {metrics.mean_ece:.4f} | {history[-1]['seconds']:.0f}s"
            f"{' *' if improved else ''}"
            f"{f' | skipped {skipped}' if skipped else ''}"
        )

        if tracker.should_stop:
            log(f"  Early stopping: {cfg.early_stop_patience} epoch 동안 개선 없음")

    if best_metrics is None:
        raise RuntimeError(f"{tag} 학습된 Epoch가 없습니다.")

    # ---- 결과 저장 ----
    result = {
        "run_id": cfg.run_id,
        "display_name": cfg.display_name,
        "fingerprint": cfg.fingerprint(),
        "fold": fold,
        "smoke": smoke,
        "best_epoch": tracker.best_epoch,
        "epochs_run": len(history),
        "stopped_early": tracker.should_stop,
        "mean_auroc": best_metrics["mean_auroc"],
        "mean_auprc": best_metrics["mean_auprc"],
        "mean_ece": best_metrics["mean_ece"],
        "per_disease": best_metrics["per_disease"],
        "n_train": len(train_df),
        "n_val": len(val_df),
        "device": str(device),
        "amp": amp,
        "seconds": round(elapsed(), 1),     # 중단 전 학습 시간 포함
        "config": cfg.to_dict(),
    }
    _atomic_write(result_path, lambda p: p.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    ))
    last_path.unlink(missing_ok=True)   # 완료됐으니 이어서 학습용 체크포인트는 삭제

    log(
        f"{tag} best epoch {tracker.best_epoch} | AUROC {result['mean_auroc']:.4f} "
        f"| AUPRC {result['mean_auprc']:.4f} | ECE {result['mean_ece']:.4f}"
    )
    return result
