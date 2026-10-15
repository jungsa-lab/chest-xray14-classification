"""지정 Fold에서 학습/검증 각 한 배치만 읽어 입력 연결을 확인한다."""
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.constants import DISEASES
from src.data.chest_xray_dataset import ChestXrayDataset, select_fold
from src.paths import PROCESSED_DIR, RAW_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    args = parser.parse_args()
    metadata = pd.read_csv(
        PROCESSED_DIR / "fold_metadata.csv",
        usecols=["Image Index", "Image Path", "Patient ID", "Split", "Fold", *DISEASES],
    )
    train_df, val_df = select_fold(metadata, args.fold)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"검사 대상 Fold: {args.fold}")
    print(f"질환 정답 순서: {DISEASES}")
    for title, subset, shuffle in [("학습", train_df, True), ("검증", val_df, False)]:
        dataset = ChestXrayDataset(subset, RAW_DIR, use_clahe=True)
        loader = DataLoader(dataset, batch_size=8, shuffle=shuffle, num_workers=0,
                            generator=torch.Generator().manual_seed(42))
        batch = next(iter(loader))
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        n = min(8, len(dataset))
        if images.shape != (n, 3, 224, 224) or labels.shape != (n, len(DISEASES)):
            raise ValueError("배치 크기가 예상과 다릅니다.")
        if images.dtype != torch.float32 or labels.dtype != torch.float32:
            raise ValueError("이미지/정답의 자료형이 float32가 아닙니다.")
        if not torch.isfinite(images).all().item() or not ((labels == 0) | (labels == 1)).all().item():
            raise ValueError("이미지/정답 배치에 잘못된 값이 있습니다.")
        positives = [disease for disease, value in zip(DISEASES, labels[0].cpu().tolist()) if value == 1]
        print(f"\n[{title} 배치]")
        print(f"Dataset 이미지 수: {len(dataset):,}")
        print(f"이미지 배치: {tuple(images.shape)}")
        print(f"정답 배치: {tuple(labels.shape)}")
        print(f"첫 이미지: {batch['image_name'][0]}")
        print(f"첫 이미지의 양성 라벨: {positives}")
        print(f"이미지 장치: {images.device}")
        print(f"정답 장치: {labels.device}")
    print("\nDataset / DataLoader 검사: 정상")


if __name__ == "__main__":
    main()
