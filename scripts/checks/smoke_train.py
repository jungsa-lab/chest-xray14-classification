import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.constants import DISEASES
from src.paths import PROCESSED_DIR, RAW_DIR
from src.data.chest_xray_dataset import ChestXrayDataset, select_fold
from src.training.model import build_densenet121
from src.training.losses import FocalLoss
from src.training.weights import compute_pos_weight
from src.training.engine import train_one_epoch, validate_one_epoch

def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"사용 장치: {device}", flush=True)

    # 1. 전처리된 메타데이터 불러오기
    metadata = pd.read_csv(
        PROCESSED_DIR / "fold_metadata.csv",
        usecols=[
            "Image Index", "Image Path", "Patient ID",
            "Split", "Fold", *DISEASES,
        ],
    )

    train_df, val_df = select_fold(metadata, fold=0)

    # 전체 학습 데이터 기준으로 클래스 가중치 계산
    pos_weight = compute_pos_weight(train_df).to(device)

    # 2. 소규모 학습 테스트용 데이터 추출
    train_df = train_df.sample(n=128, random_state=42)
    val_df = val_df.sample(n=64, random_state=42)

    train_loader = DataLoader(
        ChestXrayDataset(train_df, RAW_DIR, use_clahe=True),
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        ChestXrayDataset(val_df, RAW_DIR, use_clahe=True),
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 3. 모델 학습 및 학습 구성요소 생성
    model = build_densenet121(pretrained=True).to(device)

    criterion = FocalLoss(
        gamma=2.0,
        pos_weight=pos_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    # 4. 실제 학습 및 검증
    print("학습 시작", flush=True)
    start = time.perf_counter()

    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    train_seconds = time.perf_counter() - start

    print("검증 시작", flush=True)

    val_loss = validate_one_epoch(
        model, val_loader, criterion, device
    )

    total_seconds = time.perf_counter() - start

    print("\n=== 소규모 학습 결과 ===")
    print(f"Train Loss: {train_loss:.6f}")  # 소규모라 특정 loss가 높거나 낮은걸로 성능을 판단하기에 부적합함
    print(f"Val Loss: {val_loss:.6f}")
    print(f"학습 시간: {train_seconds:.2f}초")
    print(f"총 소요 시간: {total_seconds:.2f}초")
    print("소규모 학습 테스트 끗")

if __name__ == "__main__":
    main()