from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ChestXrayTransform:
    """PNG 이미지를 모델 입력용 정규화 텐서로 변환."""

    def __init__(self, use_clahe: bool = True):
        self.use_clahe = use_clahe

        self.mean = torch.tensor(
            IMAGENET_MEAN, dtype=torch.float32
        ).view(3, 1, 1)

        self.std = torch.tensor(
            IMAGENET_STD, dtype=torch.float32
        ).view(3, 1, 1)

    def __call__(self, image_path: str | Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            image.load()
            pixels = self.gray_from_pil(image, source=image_path)
        return self.tensor_from_gray(pixels)

    @staticmethod
    def gray_from_pil(image: Image.Image, source: str | Path = "") -> np.ndarray:
        """PIL 이미지 → 224×224 흑백 uint8 (CLAHE 전). 시각화·API 입력에서도 같은 규칙을 쓴다."""
        if image.mode not in {"L", "LA", "RGB"}:
            raise ValueError(
                f"지원하지 않는 이미지 모드: "
                f"{image.mode} / {source}"
            )

        if image.mode == "LA":
            alpha_range = image.getchannel("A").getextrema()

            if alpha_range != (255, 255):
                raise ValueError(
                    f"투명 픽셀이 있는 이미지: {source}"
                )

        gray = image.convert("L")

        # 전체 영상을 유지하며 크기를 맞춤
        # 이미 224×224이면 재조정하지 않음
        if gray.size != IMAGE_SIZE:
            gray = gray.resize(
                IMAGE_SIZE,
                resample=Image.Resampling.BILINEAR,
            )

        return np.array(gray, dtype=np.uint8)

    def tensor_from_gray(self, pixels: np.ndarray) -> torch.Tensor:
        """224×224 흑백 uint8 → (CLAHE) → 3채널 ImageNet 정규화 텐서"""
        if self.use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8),
            )
            pixels = clahe.apply(pixels)

        # (224, 224), uint8 → float32, 0~1
        tensor = torch.from_numpy(pixels).to(torch.float32) / 255.0

        # (224, 224) → (1, 224, 224) → (3, 224, 224)
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)

        # 채널별 ImageNet 정규화
        tensor = (tensor - self.mean) / self.std

        return tensor.contiguous()
