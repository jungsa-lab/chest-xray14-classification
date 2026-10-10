
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "processed" / "preprocessed_metadata.csv"

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion",
    "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening",
    "Hernia"
]


class ChestXrayDataset(Dataset):
    def __init__(self, split="train", use_clahe=False):
        self.df = pd.read_csv(CSV_PATH, low_memory=False)

        required = [
            "Absolute Image Path", "Split", *DISEASES
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV 열 누락: {missing}")

        self.df = self.df[
            self.df["Split"] == split
        ].reset_index(drop=True)

        if self.df.empty:
            raise ValueError(f"{split} 데이터가 없습니다.")

        self.split = split
        self.use_clahe = use_clahe

        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        steps = [
            transforms.Resize((224, 224))
        ]

        if split == "train":
            steps.append(
                transforms.RandomRotation(5)
            )

        steps.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row["Absolute Image Path"])

        with Image.open(path) as image:
            pixels = np.array(
                image.convert("L"), dtype=np.uint8
            )

        if self.use_clahe:
            pixels = self.clahe.apply(pixels)

        image = Image.fromarray(pixels).convert("RGB")
        image = self.transform(image)

        labels = row[DISEASES].to_numpy(dtype=np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels


if __name__ == "__main__":
    dataset = ChestXrayDataset(
        split="train",
        use_clahe=False
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    images, labels = next(iter(loader))

    print("전체 학습 이미지:", len(dataset))
    print("이미지 텐서:", images.shape)
    print("라벨 텐서:", labels.shape)
    print("이미지 자료형:", images.dtype)
