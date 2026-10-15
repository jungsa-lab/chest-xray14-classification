from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.image_transforms import ChestXrayTransform
from src.constants import DISEASES


class ChestXrayDataset(Dataset):
    """메타데이터를 기준으로 이미지와 14개 질환 정답을 반환."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        raw_dir: str | Path,
        use_clahe: bool = True,
    ):
        self.metadata = metadata.copy().reset_index(drop=True)
        self.raw_dir = Path(raw_dir)
        self.transform = ChestXrayTransform(use_clahe=use_clahe)

        required_columns = {"Image Index", "Image Path", *DISEASES}
        missing = required_columns - set(self.metadata.columns)

        if missing:
            raise ValueError(f"필수 열이 없습니다: {sorted(missing)}")

        if self.metadata.empty:
            raise ValueError("Dataset에 사용할 이미지가 없습니다.")

        if self.metadata["Image Index"].duplicated().any():
            raise ValueError("Dataset에 중복된 이미지명이 있습니다.")

        if self.metadata["Image Path"].isna().any():
            raise ValueError("이미지 경로가 비어 있습니다.")

        label_values = self.metadata[DISEASES].to_numpy(
            dtype=np.float32
        ).copy()

        if not np.isin(label_values, [0.0, 1.0]).all():
            raise ValueError("질환 정답에 0과 1 이외의 값이 있습니다.")

        # 정답은 미리 준비하고, 이미지는 요청받을 때 한 장씩 읽음
        self.labels = torch.from_numpy(label_values)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        image_path = self.raw_dir / row["Image Path"]

        image = self.transform(image_path)
        label = self.labels[index]

        return {
            "image": image,
            "label": label,
            "image_name": row["Image Index"],
        }


def select_fold(
    metadata: pd.DataFrame,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """지정한 Fold의 학습·검증 데이터를 선택."""
    if fold not in range(5):
        raise ValueError("Fold는 0~4로 지정하세요.")

    eligible = metadata["Split"].eq("train") & metadata["Fold"].ge(0)

    train_df = metadata.loc[
        eligible & metadata["Fold"].ne(fold)
    ].copy()

    val_df = metadata.loc[
        eligible & metadata["Fold"].eq(fold)
    ].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("학습 또는 검증 데이터가 비어 있습니다.")

    train_patients = set(train_df["Patient ID"])
    val_patients = set(val_df["Patient ID"])
    test_patients = set(
        metadata.loc[metadata["Split"].eq("test"), "Patient ID"]
    )

    if train_patients & val_patients:
        raise ValueError("학습과 검증에 동일 환자가 포함돼 있습니다.")

    if (train_patients | val_patients) & test_patients:
        raise ValueError("학습·검증과 Test에 동일 환자가 포함돼 있습니다.")

    return train_df, val_df
