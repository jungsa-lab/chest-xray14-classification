from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SEED = 42
SAMPLE_COUNT = 4
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)


def main():
    df = pd.read_csv(
        PROCESSED_DIR / "fold_metadata.csv",
        usecols=[
            "Image Index",
            "Image Path",
            "Patient ID",
            "Split",
            "Fold",
            "Nodule",
        ],
    )

    candidates = df.loc[
        df["Split"].eq("train")
        & df["Fold"].ge(0)
        & df["Nodule"].eq(1)
    ].copy()

    # 같은 환자의 영상만 여러 장 뽑히지 않도록 처리
    candidates = (
        candidates.sort_values("Image Index")
        .drop_duplicates("Patient ID")
    )

    if len(candidates) < SAMPLE_COUNT:
        raise ValueError("비교할 이미지가 부족합니다.")

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

    for row_index, row in samples.iterrows():
        image_path = RAW_DIR / row["Image Path"]

        with Image.open(image_path) as image:
            original = np.array(
                image.convert("L"),
                dtype=np.uint8,
            )

        enhanced = clahe.apply(original)

        for column, pixels in enumerate(
            [original, enhanced]
        ):
            axes[row_index, column].imshow(
                pixels,
                cmap="gray",
                vmin=0,
                vmax=255,
                interpolation="nearest",
            )
            axes[row_index, column].axis("off")

        axes[row_index, 0].set_title(
            f"Original\n{row['Image Index']}"
        )
        axes[row_index, 1].set_title(
            f"CLAHE\nclip={CLIP_LIMIT}, grid={TILE_GRID_SIZE}"
        )

    fig.suptitle(
        "CLAHE Comparison — Train Images with Nodule Label",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    image_output = PROCESSED_DIR / "clahe_comparison.png"
    sample_output = PROCESSED_DIR / "clahe_samples.csv"

    fig.savefig(image_output, dpi=160)
    plt.close(fig)

    # 어떤 이미지와 설정으로 비교했는지 기록
    samples["CLAHE Clip Limit"] = CLIP_LIMIT
    samples["CLAHE Grid X"] = TILE_GRID_SIZE[0]
    samples["CLAHE Grid Y"] = TILE_GRID_SIZE[1]

    samples.to_csv(
        sample_output,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"비교 이미지: {len(samples)}장")
    print(f"CLAHE 설정: clip={CLIP_LIMIT}, grid={TILE_GRID_SIZE}")
    print(f"비교 그림 저장: {image_output}")
    print(f"비교 대상·설정 저장: {sample_output}")


if __name__ == "__main__":
    main()