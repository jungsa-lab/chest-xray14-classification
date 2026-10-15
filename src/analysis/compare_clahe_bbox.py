from src.paths import RAW_DIR, PROCESSED_DIR
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image


from src.paths import PROJECT_ROOT
REPORT_DIR = PROJECT_ROOT / "reports" / "preprocessing"

SEED = 42
SAMPLE_COUNT = 4
BBOX_REFERENCE_SIZE = 1024
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)


def main():
    tables_dir = REPORT_DIR / "tables"
    figures_dir = REPORT_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 원본 헤더 대신 의미가 명확한 열 이름 지정
    boxes = pd.read_csv(
        RAW_DIR / "BBox_List_2017_Official_NIH.csv",
        header=0,
        usecols=range(6),
        names=[
            "Image Index", "Finding Label",
            "x", "y", "w", "h",
        ],
    )

    boxes["Image Index"] = boxes["Image Index"].str.strip()
    boxes["Finding Label"] = boxes["Finding Label"].str.strip()

    metadata = pd.read_csv(
        PROCESSED_DIR / "fold_metadata.csv",
        usecols=[
            "Image Index", "Image Path",
            "Patient ID", "Split", "Fold",
        ],
    )

    eligible = metadata.loc[
        metadata["Split"].eq("train")
        & metadata["Fold"].ge(0)
    ]

    # BBox의 질환명으로 Nodule 영역을 선택
    candidates = boxes.loc[
        boxes["Finding Label"].eq("Nodule")
    ].merge(
        eligible,
        on="Image Index",
        how="inner",
        validate="many_to_one",
    )

    # 환자별로 하나의 영역을 선택한 뒤 재현 가능한 무작위 추출
    candidates = (
        candidates.sort_values(["Image Index", "x", "y"])
        .drop_duplicates("Patient ID")
    )

    if candidates.empty:
        raise ValueError("사용 가능한 Train Nodule BBox가 없습니다.")

    samples = candidates.sample(
        n=min(SAMPLE_COUNT, len(candidates)),
        random_state=SEED,
    ).reset_index(drop=True)

    clahe = cv2.createCLAHE(
        clipLimit=CLIP_LIMIT,
        tileGridSize=TILE_GRID_SIZE,
    )

    fig, axes = plt.subplots(
        len(samples), 4,
        figsize=(16, 4 * len(samples)),
        squeeze=False,
    )

    records = []

    for i, row in samples.iterrows():
        with Image.open(RAW_DIR / row["Image Path"]) as image:
            image.load()

            if image.size != (224, 224):
                raise ValueError(
                    f"예상과 다른 이미지 크기: {row['Image Index']}"
                )

            if image.mode == "LA":
                if image.getchannel("A").getextrema() != (255, 255):
                    raise ValueError(
                        f"투명 픽셀 포함: {row['Image Index']}"
                    )

            original = np.array(image.convert("L"), dtype=np.uint8)

        # 전체 이미지에 먼저 CLAHE 적용
        enhanced = clahe.apply(original)
        height, width = original.shape

        x, y, w, h = [
            float(row[column]) for column in ["x", "y", "w", "h"]
        ]

        if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
            raise ValueError(f"잘못된 BBox: {row['Image Index']}")

        # 1024 기준 좌표를 현재 이미지 크기로 환산
        left = x * width / BBOX_REFERENCE_SIZE
        top = y * height / BBOX_REFERENCE_SIZE
        right = (x + w) * width / BBOX_REFERENCE_SIZE
        bottom = (y + h) * height / BBOX_REFERENCE_SIZE

        # 영상 경계를 벗어나는 부분은 잘라 표시
        left = float(np.clip(left, 0, width))
        top = float(np.clip(top, 0, height))
        right = float(np.clip(right, 0, width))
        bottom = float(np.clip(bottom, 0, height))

        if right <= left or bottom <= top:
            raise ValueError(
                f"영상 내부에 유효한 BBox가 없습니다: {row['Image Index']}"
            )

        # 주변 구조도 볼 수 있도록 BBox 바깥에 여백 추가
        margin = max(6, int(np.ceil(max(right - left, bottom - top) * 0.5)))

        x0 = max(0, int(np.floor(left)) - margin)
        y0 = max(0, int(np.floor(top)) - margin)
        x1 = min(width, int(np.ceil(right)) + margin)
        y1 = min(height, int(np.ceil(bottom)) + margin)

        views = [
            original,
            enhanced,
            original[y0:y1, x0:x1],
            enhanced[y0:y1, x0:x1],
        ]

        titles = [
            f"Original\n{row['Image Index']}",
            "CLAHE",
            "Original ROI",
            "CLAHE ROI",
        ]

        for j, (pixels, title) in enumerate(zip(views, titles)):
            ax = axes[i, j]
            ax.imshow(
                pixels,
                cmap="gray",
                vmin=0,
                vmax=255,
                interpolation="nearest",
            )
            ax.set_title(title)
            ax.axis("off")

            # imshow의 픽셀 경계에 맞춰 좌표를 표시
            offset_x = x0 if j >= 2 else 0
            offset_y = y0 if j >= 2 else 0

            ax.add_patch(Rectangle(
                (
                    left - offset_x - 0.5,
                    top - offset_y - 0.5,
                ),
                right - left,
                bottom - top,
                fill=False,
                edgecolor="yellow",
                linewidth=1,
            ))

        record = row.to_dict()
        record.update({
            "BBox Reference Size": BBOX_REFERENCE_SIZE,
            "Scaled Left": left,
            "Scaled Top": top,
            "Scaled Right": right,
            "Scaled Bottom": bottom,
            "Crop x0": x0,
            "Crop y0": y0,
            "Crop x1": x1,
            "Crop y1": y1,
            "CLAHE Clip Limit": CLIP_LIMIT,
            "CLAHE Grid X": TILE_GRID_SIZE[0],
            "CLAHE Grid Y": TILE_GRID_SIZE[1],
        })
        records.append(record)

    fig.suptitle(
        "Train Nodule BBox: Original vs CLAHE\n"
        "Coordinates assume full-image resizing from 1024 to 224",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    image_output = figures_dir / "clahe_bbox_comparison.png"
    csv_output = tables_dir / "clahe_bbox_samples.csv"

    fig.savefig(image_output, dpi=160)
    plt.close(fig)

    pd.DataFrame(records).to_csv(
        csv_output, index=False, encoding="utf-8-sig"
    )

    print(f"비교 가능한 고유 Train 환자: {len(candidates):,}명")
    print(f"선택한 환자·BBox: {len(samples)}개")
    print(f"비교 그림 저장: {image_output}")
    print(f"좌표·설정 저장: {csv_output}")


if __name__ == "__main__":
    main()
