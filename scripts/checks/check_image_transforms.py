"""Train 3장으로 CLAHE 사용/미사용 변환을 점검한다. 데이터는 저장하지 않는다."""
import pandas as pd
import torch

from src.data.image_transforms import ChestXrayTransform, IMAGENET_MEAN, IMAGENET_STD
from src.paths import PROCESSED_DIR, RAW_DIR


def main():
    df = pd.read_csv(PROCESSED_DIR / "fold_metadata.csv",
                     usecols=["Image Index", "Image Path", "Split", "Fold"])
    samples = df.loc[df["Split"].eq("train") & df["Fold"].ge(0)].head(3)
    if len(samples) != 3:
        raise ValueError("점검할 Train 이미지가 3장 미만입니다.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    for use_clahe in [False, True]:
        transform = ChestXrayTransform(use_clahe=use_clahe)
        images = []
        print(f"\n[CLAHE 사용: {use_clahe}]")
        for name, relative_path in samples[["Image Index", "Image Path"]].itertuples(index=False, name=None):
            tensor = transform(RAW_DIR / relative_path)
            if tensor.shape != (3, 224, 224) or tensor.dtype != torch.float32:
                raise ValueError(f"잘못된 텐서 모양 또는 자료형: {name}")
            restored = tensor * std + mean
            if (not torch.isfinite(tensor).all().item()
                    or restored.min().item() < -1e-6
                    or restored.max().item() > 1 + 1e-6
                    or not torch.allclose(restored[0], restored[1], atol=1e-6)
                    or not torch.allclose(restored[0], restored[2], atol=1e-6)):
                raise ValueError(f"정규화 또는 회색조 채널 검사 실패: {name}")
            print(f"{name} | shape={tuple(tensor.shape)} | dtype={tensor.dtype} | "
                  f"min={tensor.min().item():.4f} | max={tensor.max().item():.4f}")
            images.append(tensor)
        batch = torch.stack(images).to(device)
        print(f"배치 모양: {tuple(batch.shape)}")
        print(f"배치 장치: {batch.device}")
    print("\n이미지 변환 검사: 정상")


if __name__ == "__main__":
    main()
