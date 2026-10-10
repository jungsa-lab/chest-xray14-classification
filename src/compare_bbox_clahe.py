
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

BBOX_CSV = RAW / "BBox_List_2017_Official_NIH.csv"
META_CSV = PROCESSED / "preprocessed_metadata.csv"

SAMPLE_COUNT = 4
SEED = 42
CLIP_LIMIT = 2.0
GRID_SIZE = (8, 8)

# NIH BBox 좌표의 기준 이미지 크기
BBOX_REFERENCE_SIZE = 1024


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1. BBox CSV 읽기
    bbox = pd.read_csv(BBOX_CSV)
    bbox.columns = bbox.columns.str.strip()

    print("BBox CSV 열:", bbox.columns.tolist())

    required = [
        "Image Index", "Finding Label",
        "Bbox [x", "y", "w", "h]"
    ]
    missing = [c for c in required if c not in bbox.columns]

    if missing:
        raise ValueError(f"BBox CSV 열 이름 확인 필요: {missing}")

    # 2. Nodule BBox만 선택
    bbox = bbox[
        bbox["Finding Label"].astype(str).str.strip().eq("Nodule")
    ].copy()

    # 3. 전처리된 메타데이터 읽기
    metadata = pd.read_csv(
        META_CSV,
        usecols=[
            "Image Index",
            "Absolute Image Path",
            "Patient ID",
            "Split",
        ],
        low_memory=False,
    )

    # Train 데이터만 사용
    metadata = metadata[
        metadata["Split"].eq("train")
    ]

    # 4. BBox와 이미지 경로 연결
    merged = bbox.merge(
        metadata,
        on="Image Index",
        how="inner",
    )

    # 서로 다른 환자 선택
    merged = (
        merged.sort_values("Image Index")
        .drop_duplicates("Patient ID")
    )

    merged = merged[
        merged["Absolute Image Path"].notna()
    ].copy()

    if len(merged) < SAMPLE_COUNT:
        raise ValueError(
            f"비교 가능한 Nodule BBox가 부족합니다: {len(merged)}개"
        )

    samples = merged.sample(
        n=SAMPLE_COUNT,
        random_state=SEED,
    ).reset_index(drop=True)

    clahe = cv2.createCLAHE(
        clipLimit=CLIP_LIMIT,
        tileGridSize=GRID_SIZE,
    )

    fig, axes = plt.subplots(
        SAMPLE_COUNT, 3,
        figsize=(13, SAMPLE_COUNT * 4),
        squeeze=False,
    )

    for i, row in samples.iterrows():
        path = Path(row["Absolute Image Path"])

        with Image.open(path) as img:
            original = np.asarray(
                img.convert("L"),
                dtype=np.uint8,
            )

        enhanced = clahe.apply(original)

        height, width = original.shape

        # 1024 원본 좌표를 현재 이미지 크기로 변환
        scale_x = width / BBOX_REFERENCE_SIZE
        scale_y = height / BBOX_REFERENCE_SIZE

        x = float(row["Bbox [x"]) * scale_x
        y = float(row["y"]) * scale_y
        w = float(row["w"]) * scale_x
        h = float(row["h]"]) * scale_y

        # 화면 밖 영역 방지
        x1 = max(0, int(np.floor(x)))
        y1 = max(0, int(np.floor(y)))
        x2 = min(width, int(np.ceil(x + w)))
        y2 = min(height, int(np.ceil(y + h)))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("잘못된 BBox 좌표입니다.")

        # 원본 및 CLAHE에 동일한 사각형 그리기
        for j, pixels in enumerate([original, enhanced]):
            ax = axes[i, j]

            ax.imshow(
                pixels,
                cmap="gray",
                vmin=0,
                vmax=255,
            )

            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )

            ax.add_patch(rect)
            ax.set_title(
                "Original + BBox" if j == 0
                else "CLAHE + BBox"
            )
            ax.axis("off")

        # 세 번째 열: 결절 부위 확대 비교
        crop = enhanced[y1:y2, x1:x2]

        axes[i, 2].imshow(
            crop,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        axes[i, 2].set_title(
            f"CLAHE BBox Crop\n{row['Image Index']}"
        )
        axes[i, 2].axis("off")

    fig.suptitle("Nodule BBox: Original vs CLAHE")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output = PROCESSED / "bbox_clahe_comparison.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)

    samples.to_csv(
        PROCESSED / "bbox_clahe_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("BBox 비교 완료!")
    print(f"결과 저장: {output}")


if __name__ == "__main__":
    main()
