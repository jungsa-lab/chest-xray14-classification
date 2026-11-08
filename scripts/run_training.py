"""환자 기준 그룹화된 단일 ChestX-ray14 폴드에 대해 DenseNet121 학습"""
import argparse
import csv
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


def validate_with_metrics(model, loader, criterion, device):
    """단일 패스(one pass)로 검증 손실 및 다중 레이블 성능 지표 계산"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_probs = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="검증", leave=False):
            images = batch["image"].to(device)
            targets = batch["label"].to(device)
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


def atomic_save(state, path):
    temp_path = path.with_name(path.name + ".tmp")
    torch.save(state, temp_path)
    temp_path.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.num_workers < 0:
        parser.error("epochs, batch-size, lr은 양수이고 num-workers는 0 이상이어야 합니다.")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}", flush=True)

    df = pd.read_csv(
        PROCESSED_DIR / "fold_metadata.csv",
        usecols=["Image Index", "Image Path", "Patient ID", "Split", "Fold", *DISEASES],
    )
    train_df, val_df = select_fold(df, args.fold)
    print(f"Fold {args.fold}: 학습 {len(train_df):,}장 / 검증 {len(val_df):,}장", flush=True)

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

    model = build_densenet121(pretrained=not args.resume).to(device)
    criterion = FocalLoss(
        gamma=2.0,
        pos_weight=compute_pos_weight(train_df).to(device),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patience)

    output_dir = PROJECT_ROOT / "outputs" / "training" / f"fold{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.csv"
    start_epoch = 1

    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"재개할 체크포인트가 없습니다: {latest_path}")
        state = torch.load(latest_path, map_location=device, weights_only=True)
        if (state["fold"] != args.fold
            or state["batch_size"] != args.batch_size
            or state["lr"] != args.lr
            or state["patience"] != args.patience):
            raise ValueError("이전 학습과 fold/batch-size/lr/patience 설정이 다릅니다.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        early_stopping.best_score = state["early_stopping"]["best_score"]
        early_stopping.bad_epochs = state["early_stopping"]["bad_epochs"]
        early_stopping.should_stop = state["early_stopping"]["should_stop"]
        start_epoch = state["epoch"] + 1
        if early_stopping.should_stop:
            print("이미 EarlyStopping이 발동한 학습입니다. 재개하지 않습니다.")
            return
        print(f"이전 학습 {state['epoch']} Epoch까지 복구했습니다.", flush=True)
    else:
        if latest_path.exists() or best_path.exists():
            raise FileExistsError(
                f"이전 학습 결과가 존재합니다: {output_dir}\n"
                "덮어쓰지 않으려면 --resume을 사용하거나 결과를 먼저 백업하세요."
            )
        with history_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "mean_auroc", "mean_auprc", "mean_ece", "seconds"])

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            tqdm(train_loader, desc=f"학습 {epoch}/{args.epochs}", leave=False),
            criterion, optimizer, device,
        )
        val_loss, metrics = validate_with_metrics(model, val_loader, criterion, device)
        seconds = time.perf_counter() - start

        score = metrics.mean_auroc
        should_stop = early_stopping.step(score)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"AUROC {score:.4f} | AUPRC {metrics.mean_auprc:.4f} | "
            f"ECE {metrics.mean_ece:.4f} | {seconds:.1f}초",
            flush=True,
        )

        if early_stopping.improved:
            atomic_save({
                "epoch": epoch, "fold": args.fold,
                "mean_auroc": score,
                "model_state_dict": model.state_dict(),
            }, best_path)
            print(f"  최고 성능 모델 저장: {best_path}", flush=True)

        atomic_save({
            "epoch": epoch, "fold": args.fold,
            "batch_size": args.batch_size, "lr": args.lr, "patience": args.patience,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "early_stopping": {
                "best_score": early_stopping.best_score,
                "bad_epochs": early_stopping.bad_epochs,
                "should_stop": early_stopping.should_stop,
            },
        }, latest_path)
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss, metrics.mean_auroc,
                metrics.mean_auprc, metrics.mean_ece, seconds,
            ])

        if should_stop:
            print("검증 AUROC 개선이 없어 EarlyStopping으로 종료합니다.")
            break

    print(f"학습 종료. 결과 폴더: {output_dir}")


if __name__ == "__main__":
    main()
