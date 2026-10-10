from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

QUALITY_REPORT_PATH = PROCESSED_DIR / "image_quality_audit.csv"
ALPHA_REPORT_PATH = PROCESSED_DIR / "image_alpha_audit.csv"


def main():
    quality_df = pd.read_csv(QUALITY_REPORT_PATH)
    quality_df["Error"] = quality_df["Error"].fillna("")

    targets = quality_df.loc[quality_df["Mode"] == "LA"]
    records = []

    for row_index, row in tqdm(
        targets.iterrows(),
        total=len(targets),
        desc="LA 이미지 검사 보완",
    ):
        image_name = row["Image Index"]
        relative_path = row["Image Path"]

        with Image.open(RAW_DIR / relative_path) as image:
            image.load()

            if image.mode != "LA":
                raise ValueError(
                    f"이전 검사와 이미지 모드가 다릅니다: {image_name}"
                )

            alpha_min, alpha_max = image.getchannel("A").getextrema()
            fully_opaque = (alpha_min, alpha_max) == (255, 255)

            if fully_opaque:
                pixels = np.asarray(
                    image.convert("L"), dtype=np.float32
                )

                quality_df.at[row_index, "Mean Brightness"] = float(
                    pixels.mean()
                )
                quality_df.at[row_index, "Brightness Std"] = float(
                    pixels.std()
                )
                quality_df.at[row_index, "Error"] = ""

            else:
                quality_df.at[row_index, "Mean Brightness"] = np.nan
                quality_df.at[row_index, "Brightness Std"] = np.nan
                quality_df.at[row_index, "Error"] = (
                    "투명 또는 반투명 픽셀 포함"
                )

        records.append({
            "Image Index": image_name,
            "Image Path": relative_path,
            "Alpha Min": alpha_min,
            "Alpha Max": alpha_max,
            "Fully Opaque": fully_opaque,
        })

    alpha_df = pd.DataFrame(
        records,
        columns=[
            "Image Index",
            "Image Path",
            "Alpha Min",
            "Alpha Max",
            "Fully Opaque",
        ],
    )

    alpha_df.to_csv(
        ALPHA_REPORT_PATH, index=False, encoding="utf-8-sig"
    )
    quality_df.to_csv(
        QUALITY_REPORT_PATH, index=False, encoding="utf-8-sig"
    )

    print("\n[LA 이미지 투명도 검사]")
    print(f"검사 대상: {len(alpha_df):,}")

    if not alpha_df.empty:
        opaque_count = int(alpha_df["Fully Opaque"].sum())

        print(f"전체 픽셀이 완전 불투명: {opaque_count:,}")
        print(
            "투명 또는 반투명 픽셀 포함: "
            f"{len(alpha_df) - opaque_count:,}"
        )

    print("\n[보완된 이미지 검사 결과]")
    print(
        "밝기 측정 완료: "
        f"{int(quality_df['Mean Brightness'].notna().sum()):,}"
    )
    print(
        "오류 또는 추가 확인 기록: "
        f"{int(quality_df['Error'].ne('').sum()):,}"
    )
    print(f"검사 CSV 갱신: {QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main()