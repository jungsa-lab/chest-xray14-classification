from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FOLD_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "fold_metadata.csv"

DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ChestXrayDataset(Dataset):
    """fold_metadata.csv를 읽어 5-Fold 학습/검증 또는 Test 샘플을 반환한다."""

    def __init__(
        self,
        mode: str = "train",
        fold: int = 0,
        use_clahe: bool = False,
    ) -> None:
        if mode not in {"train", "val", "test"}:
            raise ValueError("mode는 'train', 'val', 'test' 중 하나여야 합니다.")
        if fold not in range(5):
            raise ValueError("fold는 0~4 중 하나여야 합니다.")
        if not FOLD_METADATA_PATH.is_file():
            raise FileNotFoundError(
                f"Fold 메타데이터를 찾을 수 없습니다: {FOLD_METADATA_PATH}"
            )

        self.df = pd.read_csv(FOLD_METADATA_PATH, low_memory=False)
        required = ["Image Index", "Image Path", "Split", "Fold", *DISEASES]
        missing = [column for column in required if column not in self.df.columns]
        if missing:
            raise ValueError(f"fold_metadata.csv에 필요한 열이 없습니다: {missing}")

        self.df["Split"] = self.df["Split"].astype(str).str.strip().str.lower()
        if "Quality Pass" in self.df.columns:
            quality_pass = self.df["Quality Pass"].astype(str).str.lower().isin(
                {"true", "1"}
            )
        else:
            quality_pass = self.df["Fold"].ge(0) | self.df["Split"].eq("test")

        if mode == "test":
            # Test는 최종 평가를 위해 품질 필터로 임의 제외하지 않는다.
            mask = self.df["Split"].eq("test")
        elif mode == "val":
            mask = (
                self.df["Split"].eq("train")
                & self.df["Fold"].eq(fold)
                & quality_pass
            )
        else:
            mask = (
                self.df["Split"].eq("train")
                & self.df["Fold"].ge(0)
                & self.df["Fold"].ne(fold)
                & quality_pass
            )

        self.df = self.df.loc[mask].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"mode={mode}, fold={fold}에 해당하는 이미지가 없습니다.")

        self.mode = mode
        self.fold = fold
        self.use_clahe = use_clahe
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        steps = []
        if mode == "train":
            steps.append(transforms.RandomRotation(degrees=5))
        steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = RAW_DIR / str(row["Image Path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.uint8)

        if self.use_clahe:
            pixels = self.clahe.apply(pixels)

        # 사전학습 ImageNet 모델 입력에 맞춰 grayscale을 3채널로 복제
        image = Image.fromarray(pixels).convert("RGB")
        image_tensor = self.transform(image)
        labels = torch.tensor(row[DISEASES].to_numpy(dtype=np.float32))
        return image_tensor, labels


if __name__ == "__main__":
    selected_fold = 0
    train_dataset = ChestXrayDataset(mode="train", fold=selected_fold)
    val_dataset = ChestXrayDataset(mode="val", fold=selected_fold)

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0
    )

    images, labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))
    print(f"Fold: {selected_fold}")
    print(f"Train images: {len(train_dataset):,}; val images: {len(val_dataset):,}")
    print(f"Train batch image tensor: {tuple(images.shape)} ({images.dtype})")
    print(f"Train batch label tensor: {tuple(labels.shape)} ({labels.dtype})")
    print(f"Val batch image tensor: {tuple(val_images.shape)}")
    print(f"Val batch label tensor: {tuple(val_labels.shape)}")
