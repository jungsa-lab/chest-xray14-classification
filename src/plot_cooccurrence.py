from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_training_data import DISEASES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_PATH = PROCESSED_DIR / "fold_metadata.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "preprocessing"
REPORT_TABLES_DIR = REPORT_DIR / "tables"
REPORT_FIGURES_DIR = REPORT_DIR / "figures"


def main():
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        METADATA_PATH,
        usecols=["Split", "Fold", *DISEASES],
    )

    # 품질 검사를 통과하고 Fold가 배정된 Train만 사용
    train_df = df.loc[
        df["Split"].eq("train") & df["Fold"].ge(0)
    ].copy()

    if train_df.empty:
        raise ValueError("분석할 Train 이미지가 없습니다.")

    # 큰 개수를 안전하게 누적할 수 있도록 int64 사용
    labels = train_df[DISEASES].to_numpy(dtype=np.int64)

    # (14, 이미지 수) @ (이미지 수, 14) -> (14, 14)
    counts = labels.T @ labels

    cooccurrence = pd.DataFrame(
        counts,
        index=DISEASES,
        columns=DISEASES,
    )
    cooccurrence.index.name = "Disease"

    # csv_path = PROCESSED_DIR / "train_cooccurrence.csv"
    # image_path = PROCESSED_DIR / "train_cooccurrence.png"
    
    csv_path = REPORT_TABLES_DIR / "train_cooccurrence.csv"
    image_path = REPORT_FIGURES_DIR / "train_cooccurrence.png"

    cooccurrence.to_csv(
        csv_path,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(14, 12))

    # 빈도 차이가 크므로 색상에는 log(1 + 개수)를 사용
    # 각 칸에 적는 숫자는 실제 이미지 개수
    color_values = np.log1p(counts)
    heatmap = ax.imshow(color_values, cmap="Blues")

    positions = np.arange(len(DISEASES))

    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(
        DISEASES, rotation=60, ha="right"
    )
    ax.set_yticklabels(DISEASES)

    ax.set_title(
        f"Train Label Co-occurrence (n={len(train_df):,})\n"
        "Diagonal: positive count for each disease"
    )

    threshold = color_values.max() * 0.6

    for row in positions:
        for column in positions:
            text_color = (
                "white"
                if color_values[row, column] > threshold
                else "black"
            )

            ax.text(
                column,
                row,
                f"{counts[row, column]:,}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    colorbar = fig.colorbar(
        heatmap, ax=ax, fraction=0.046, pad=0.04
    )
    colorbar.set_label("log(1 + image count)")

    fig.tight_layout()
    fig.savefig(image_path, dpi=180)
    plt.close(fig)

    print(f"분석 대상 Train 이미지: {len(train_df):,}장")
    print(f"동시 출현 행렬 크기: {counts.shape}")
    print(f"행렬 CSV 저장: {csv_path}")
    print(f"히트맵 저장: {image_path}")


if __name__ == "__main__":
    main()