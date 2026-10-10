
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_PATH = PROCESSED_DIR / "preprocessed_metadata.csv"

SEED = 42
SAMPLE_COUNT = 4
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"전처리 메타데이터가 없습니다: {METADATA_PATH}"
        )

    df = pd.read_csv(METADATA_PATH)

    required_columns = [
        "Image Index",
        "Absolute Image Path",
        "Patient ID",
        "Split",
        "Nodule",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"누락된 CSV 열: {missing}")

    # Train 이미지 중 Nodule 양성인 사례 선택
    candidates = df.loc[
        df["Split"].eq("train")
        & df["Nodule"].eq(1)
    ].copy()

    # 환자 중복 제거
    candidates = (
        candidates.sort_values("Image Index")
        .drop_duplicates("Patient ID")
    )

    if len(candidates) < SAMPLE_COUNT:
        raise ValueError("Nodule 비교 이미지가 부족합니다.")

    samples = candidates.sample(
        n=SAMPLE_COUNT,
        random_state=SEED,
    ).reset_index(drop=True)

    clahe = cv2.createCLAHE(
        clipLimit=CLIP_LIMIT,
        tileGridSize=TILE_GRID_SIZE,
    )

    fig, axes = plt.subplots(
        SAMPLE_COUNT,
        2,
        figsize=(9, 4 * SAMPLE_COUNT),
        squeeze=False,
    )

    for i, row in samples.iterrows():
        image_path = Path(row["Absolute Image Path"])

        if not image_path.is_file():
            raise FileNotFoundError(
                f"이미지를 찾을 수 없습니다: {image_path}"
            )

        with Image.open(image_path) as image:
            original = np.array(image.convert("L"))

        enhanced = clahe.apply(original)

        for j, pixels in enumerate([original, enhanced]):
            axes[i, j].imshow(
                pixels,
                cmap="gray",
                vmin=0,
                vmax=255,
            )
            axes[i, j].axis("off")

        axes[i, 0].set_title(
            f"Original\n{row['Image Index']}"
        )
        axes[i, 1].set_title(
            f"CLAHE\nclip={CLIP_LIMIT}, grid={TILE_GRID_SIZE}"
        )

    fig.suptitle("Nodule: Original vs CLAHE")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_image = PROCESSED_DIR / "clahe_comparison.png"
    output_csv = PROCESSED_DIR / "clahe_samples.csv"

    fig.savefig(output_image, dpi=160)
    plt.close(fig)

    samples["CLAHE Clip Limit"] = CLIP_LIMIT
    samples["CLAHE Grid X"] = TILE_GRID_SIZE[0]
    samples["CLAHE Grid Y"] = TILE_GRID_SIZE[1]

    samples.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"비교 이미지: {len(samples)}장")
    print(f"비교 그림 저장: {output_image}")
    print(f"비교 대상 저장: {output_csv}")


if __name__ == "__main__":
    main()
