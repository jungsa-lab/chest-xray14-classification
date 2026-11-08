"""환자 단위 그룹화 폴드(patient-grouped fold)에서 Focal Loss 및 코사인 학습률(Cosine LR)을 적용한 DenseNet121 학습 스크립트.

실행 방법 (프로젝트 루트 경로에서 실행):
    python -m scripts.run_training --fold 0 --gamma 0 --epochs 10 --batch-size 8

참고 사항:
    - 감마(gamma) 비교 실험 시 동일한 환자 분할(patient split), 시드(seed), 옵티마이저(optimizer), 스케줄러(schedule) 사용
    - 각 gamma 및 fold 조합별로 독립적인 체크포인트(checkpoint) 디렉터리 생성
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.constants import DISEASES
from src.data.chest_xray_dataset import ChestXrayDataset, select_fold
from src.evaluation.metrics import evaluate_multilabel
from src.paths import PROCESSED_DIR, RAW_DIR, PROJECT_ROOT
from src.training.early_stopping import EarlyStopping
from src.training.engine import train_one_epoch
from src.training.losses import FocalLoss
from src.training.model import build_densenet121
from src.training.weights import compute_pos_weight


HISTORY_COLUMNS = [
    "epoch", "learning_rate", "train_loss", "val_loss",
    "mean_auroc", "mean_auprc", "mean_ece", "seconds",
]


def validate_with_metrics(model, loader, criterion, device):
    """단일 순회(one pass)로 검증 손실(validation loss) 및 14개 질환 평가 지표 계산."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_probs = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="검증", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)

            if not torch.isfinite(loss).item():
                raise ValueError("검증 Loss가 유한하지 않습니다.")

            n = images.size(0)
            total_loss += loss.item() * n
            total_samples += n
            all_targets.append(targets.cpu().numpy())
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    if total_samples == 0:
        raise ValueError("검증 데이터가 비어 있습니다.")

    metrics = evaluate_multilabel(
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_probs, axis=0),
    )
    return total_loss / total_samples, metrics


def atomic_save(state, path: Path):
    """파일 쓰기 중단 시 손상된 최종 체크포인트(checkpoint)가 남는 것을 방지."""
    temp_path = path.with_name(path.name + ".tmp")
    torch.save(state, temp_path)
    temp_path.replace(path)


def atomic_write_json(value, path: Path):
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--gamma", type=float, default=2.0, choices=[0.0, 1.0, 2.0])
    parser.add_argument("--epochs", type=int, default=10, help="총 Epoch 수 (resume 시에도 동일)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if (args.epochs < 1 or args.batch_size < 1 or args.lr <= 0
            or args.patience < 1 or args.num_workers < 0 or args.seed < 0
            or not (0 <= args.eta_min < args.lr)):
        parser.error("epochs/batch-size/lr/patience는 양수, num-workers/seed는 0 이상, 0 <= eta-min < lr 이어야 합니다.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}", flush=True)

    df = pd.read_csv(
        PROCESSED_DIR / "fold_metadata.csv",
        usecols=["Image Index", "Image Path", "Patient ID", "Split", "Fold", *DISEASES],
    )
    train_df, val_df = select_fold(df, args.fold)
    print(
        f"Fold {args.fold} / gamma={args.gamma:g}: "
        f"학습 {len(train_df):,}장 / 검증 {len(val_df):,}장",
        flush=True,
    )

    train_loader = DataLoader(
        ChestXrayDataset(train_df, RAW_DIR, use_clahe=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        ChestXrayDataset(val_df, RAW_DIR, use_clahe=True),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # 출력 경로 분리: 기존 outputs/training/fold0 결과물은 그대로 유지
    gamma_label = f"gamma{int(args.gamma)}"
    output_dir = PROJECT_ROOT / "outputs" / "training" / gamma_label / f"fold{args.fold}"
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.csv"
    best_metrics_path = output_dir / "best_metrics.json"

    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"재개할 체크포인트가 없습니다: {latest_path}")
    elif any(p.exists() for p in (latest_path, best_path, history_path, best_metrics_path)):
        raise FileExistsError(
            f"기존 학습 결과가 존재합니다: {output_dir}\n"
            "기존 결과를 유지하려면 --resume을 사용하세요. "
            "새 실험이라면 결과 폴더를 별도로 백업하세요."
        )

    model = build_densenet121(pretrained=not args.resume).to(device)
    criterion = FocalLoss(
        gamma=args.gamma,
        pos_weight=compute_pos_weight(train_df).to(device),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # 각 epoch의 옵티마이저 업데이트 완료 후 scheduler.step()을 1회 호출
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min,
    )
    early_stopping = EarlyStopping(patience=args.patience)

    config = {
        "fold": args.fold, "gamma": args.gamma, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr,
        "eta_min": args.eta_min, "patience": args.patience,
        "num_workers": args.num_workers, "seed": args.seed,
        "model": "densenet121_imagenet1k_v1", "use_clahe": True,
        "scheduler": "CosineAnnealingLR", "optimizer": "AdamW",
    }
    start_epoch = 1

    if args.resume:
        state = torch.load(latest_path, map_location="cpu", weights_only=True)
        if state.get("config") != config:
            raise ValueError(
                "저장된 체크포인트와 현재 학습 설정이 다릅니다. "
                "--epochs, --gamma 등을 포함한 모든 설정을 동일하게 사용해야 합니다."
            )
        model.load_state_dict(state["model_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        early_stopping.best_score = state["early_stopping"]["best_score"]
        early_stopping.bad_epochs = state["early_stopping"]["bad_epochs"]
        early_stopping.should_stop = state["early_stopping"]["should_stop"]
        if "cpu_rng_state" in state:
            torch.set_rng_state(state["cpu_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_states" in state:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng_states"]])
        start_epoch = state["epoch"] + 1
        if early_stopping.should_stop:
            print("이미 EarlyStopping이 발동했습니다. 재개하지 않습니다.")
            return
        print(f"이전 학습 {state['epoch']} Epoch까지 복구했습니다.", flush=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        with history_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HISTORY_COLUMNS)

    if start_epoch > args.epochs:
        print(f"이미 {args.epochs} Epoch까지 완료했습니다.")
        return

    print(
        f"학습 설정: AdamW lr={args.lr:g} | "
        f"CosineAnnealingLR T_max={args.epochs}, eta_min={args.eta_min:g} | "
        f"EarlyStopping patience={args.patience}",
        flush=True,
    )
    print(f"결과 폴더: {output_dir}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model,
            tqdm(train_loader, desc=f"학습 {epoch}/{args.epochs}", leave=False),
            criterion, optimizer, device,
        )
        val_loss, metrics = validate_with_metrics(model, val_loader, criterion, device)
        seconds = time.perf_counter() - start

        should_stop = early_stopping.step(metrics.mean_auroc)
        print(
            f"Epoch {epoch}/{args.epochs} | LR {current_lr:.7f} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"AUROC {metrics.mean_auroc:.4f} | "
            f"AUPRC {metrics.mean_auprc:.4f} | "
            f"ECE {metrics.mean_ece:.4f} | {seconds:.1f}초",
            flush=True,
        )

        if early_stopping.improved:
            atomic_save({
                "epoch": epoch,
                "config": config,
                "mean_auroc": metrics.mean_auroc,
                "model_state_dict": model.state_dict(),
            }, best_path)
            atomic_write_json({
                "epoch": epoch,
                "gamma": args.gamma,
                "fold": args.fold,
                "mean_auroc": metrics.mean_auroc,
                "mean_auprc": metrics.mean_auprc,
                "mean_ece": metrics.mean_ece,
                "per_disease": metrics.per_disease,
            }, best_metrics_path)
            print("  최고 성능 모델 저장", flush=True)

        # 다음 epoch에서 학습이 올바르게 재개되도록 step() 호출 후의 LR 스케줄 상태 저장
        scheduler.step()
        atomic_save({
            "epoch": epoch,
            "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "early_stopping": {
                "best_score": early_stopping.best_score,
                "bad_epochs": early_stopping.bad_epochs,
                "should_stop": early_stopping.should_stop,
            },
        }, latest_path)
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, current_lr, train_loss, val_loss,
                metrics.mean_auroc, metrics.mean_auprc,
                metrics.mean_ece, seconds,
            ])

        if should_stop:
            print("검증 AUROC 개선이 없어 EarlyStopping으로 종료합니다.")
            break

    print(f"학습 종료. 결과 폴더: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
