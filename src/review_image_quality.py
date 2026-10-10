from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

METADATA_PATH = PROCESSED_DIR / "split_metadata.csv"
PREVIEW_PATH = PROCESSED_DIR / "train_quality_candidates.png"
CANDIDATES_PATH = PROCESSED_DIR / "train_quality_candidates.csv"

def main():
    df = pd.read_csv(METADATA_PATH)

    # 필터링 기준 검토에는 Train만 사용
    train_df = df.loc[df["Split"] == "train"].copy()

    if len(train_df) < 5:
        raise ValueError("후보를 확인할 Train 이미지가 부족합니다.")

    columns = ["Mean Brightness", "Brightness Std"]

    if train_df[columns].isna().any().any():
        raise ValueError("밝기 검사 결과에 결측값이 있습니다.")

    groups = [
        (
            "Darkest",
            train_df.nsmallest(5, "Mean Brightness"),
        ),
        (
            "Brightest",
            train_df.nlargest(5, "Mean Brightness"),
        ),
        (
            "Lowest std",
            train_df.nsmallest(5, "Brightness Std"),
        ),
    ]

    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
    candidate_tables = []

    for row_index, (group_name, subset) in enumerate(groups):
        candidate_tables.append(
            subset.assign(**{"Review Group": group_name})
        )

        for column_index, (_, row) in enumerate(subset.iterrows()):
            image_path = RAW_DIR / row["Image Path"]

            with Image.open(image_path) as image:
                gray = image.convert("L")

                axes[row_index, column_index].imshow(
                    gray,
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )

            axes[row_index, column_index].set_title(
                f"{group_name}\n"
                f"{row['Image Index']}\n"
                f"mean={row['Mean Brightness']:.1f}, "
                f"std={row['Brightness Std']:.1f}",
                fontsize=9,
            )
            axes[row_index, column_index].axis("off")

    fig.suptitle("Train image quality candidates", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PREVIEW_PATH, dpi=150)
    plt.close(fig)

    candidates = pd.concat(candidate_tables, ignore_index=True)

    candidates[
        [
            "Review Group",
            "Image Index",
            "Image Path",
            "Mean Brightness",
            "Brightness Std",
        ]
    ].to_csv(
        CANDIDATES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("[Train 밝기 분포]")
    print(
        train_df[columns].describe(
            percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
        ).to_string()
    )

    print(f"\n후보 이미지 저장: {PREVIEW_PATH}")
    print(f"후보 목록 저장: {CANDIDATES_PATH}")


if __name__ == "__main__":
    main()