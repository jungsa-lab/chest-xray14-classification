from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FOLD_METADATA_PATH = PROCESSED_DIR / "fold_metadata.csv"

BBOX_CSV_CANDIDATES = [
    RAW_DIR / "BBox_List_2017_Official_NIH.csv",
    RAW_DIR / "BBox_List_2017.csv",
]

SAMPLE_COUNT = 4
SEED = 42
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)
BBOX_REFERENCE_SIZE = 1024


def find_bbox_csv() -> Path:
    for path in BBOX_CSV_CANDIDATES:
        if path.is_file():
            return path
    expected = " 또는 ".join(str(path) for path in BBOX_CSV_CANDIDATES)
    raise FileNotFoundError(
        f"BBox 주석 CSV를 찾을 수 없습니다. 다음 위치 중 하나에 넣어주세요: {expected}"
    )


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not FOLD_METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"분할 메타데이터를 찾을 수 없습니다: {FOLD_METADATA_PATH}"
        )

    bbox_path = find_bbox_csv()
    bbox = pd.read_csv(bbox_path)
    bbox.columns = bbox.columns.str.strip()

    bbox_columns = ["Image Index", "Finding Label", "Bbox [x", "y", "w", "h]"]
    missing = [column for column in bbox_columns if column not in bbox.columns]
    if missing:
        raise ValueError(
            f"BBox CSV 열이 예상한 형식과 다릅니다. 누락: {missing}; "
            f"실제 열: {bbox.columns.tolist()}"
        )

    bbox = bbox.loc[
        bbox["Finding Label"].astype(str).str.strip().eq("Nodule")
    ].copy()

    metadata = pd.read_csv(
        FOLD_METADATA_PATH,
        usecols=["Image Index", "Image Path", "Patient ID", "Split", "Fold", "Nodule"],
        low_memory=False,
    )
    metadata["Split"] = metadata["Split"].astype(str).str.strip().str.lower()

    metadata = metadata.loc[
        metadata["Split"].eq("train")
        & metadata["Fold"].ge(0)
        & metadata["Nodule"].eq(1)
    ].copy()

    merged = bbox.merge(metadata, on="Image Index", how="inner")
    merged = (
        merged.sort_values("Image Index")
        .drop_duplicates("Patient ID")
        .reset_index(drop=True)
    )

    if len(merged) < SAMPLE_COUNT:
        raise ValueError(
            f"Train의 Nodule BBox와 연결된 이미지가 {len(merged)}장뿐입니다. "
            f"필요한 수: {SAMPLE_COUNT}. BBox CSV와 이미지 파일 구성을 확인하세요."
        )

    samples = merged.sample(n=SAMPLE_COUNT, random_state=SEED).reset_index(drop=True)
    clahe = cv2.createCLAHE(
        clipLimit=CLIP_LIMIT,
        tileGridSize=TILE_GRID_SIZE,
    )

    fig, axes = plt.subplots(
        SAMPLE_COUNT,
        3,
        figsize=(13, 4 * SAMPLE_COUNT),
        squeeze=False,
    )

    for row_index, row in samples.iterrows():
        image_path = RAW_DIR / str(row["Image Path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

        with Image.open(image_path) as image:
            original = np.asarray(image.convert("L"), dtype=np.uint8)
        enhanced = clahe.apply(original)

        height, width = original.shape
        x = float(row["Bbox [x"]) * width / BBOX_REFERENCE_SIZE
        y = float(row["y"]) * height / BBOX_REFERENCE_SIZE
        box_width = float(row["w"]) * width / BBOX_REFERENCE_SIZE
        box_height = float(row["h]"]) * height / BBOX_REFERENCE_SIZE

        x1 = max(0, int(np.floor(x)))
        y1 = max(0, int(np.floor(y)))
        x2 = min(width, int(np.ceil(x + box_width)))
        y2 = min(height, int(np.ceil(y + box_height)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"유효하지 않은 BBox 좌표입니다: {row['Image Index']} "
                f"({x1}, {y1}, {x2}, {y2})"
            )

        for column, pixels, title in [
            (0, original, "Original + BBox"),
            (1, enhanced, "CLAHE + BBox"),
        ]:
            ax = axes[row_index, column]
            ax.imshow(pixels, cmap="gray", vmin=0, vmax=255)
            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.5,
                )
            )
            ax.set_title(title)
            ax.axis("off")

        axes[row_index, 2].imshow(
            enhanced[y1:y2, x1:x2], cmap="gray", vmin=0, vmax=255
        )
        axes[row_index, 2].set_title(
            f"CLAHE BBox Crop\n{row['Image Index']}"
        )
        axes[row_index, 2].axis("off")

    fig.suptitle("Nodule BBox: Original vs CLAHE", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    image_output = PROCESSED_DIR / "bbox_clahe_comparison.png"
    samples_output = PROCESSED_DIR / "bbox_clahe_samples.csv"
    fig.savefig(image_output, dpi=160)
    plt.close(fig)
    samples.to_csv(samples_output, index=False, encoding="utf-8-sig")

    print(f"사용한 BBox CSV: {bbox_path}")
    print(f"비교 이미지: {len(samples)}장")
    print(f"CLAHE 설정: clip={CLIP_LIMIT}, grid={TILE_GRID_SIZE}")
    print(f"비교 그림 저장: {image_output}")
    print(f"비교 대상 저장: {samples_output}")


if __name__ == "__main__":
    main()
