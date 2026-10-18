from src.paths import PROCESSED_DIR, PREPROCESSING_REPORT_DIR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import DISEASES


METADATA_PATH = PROCESSED_DIR / "fold_metadata.csv"
REPORT_TABLES_DIR = PREPROCESSING_REPORT_DIR / "tables"
REPORT_FIGURES_DIR = PREPROCESSING_REPORT_DIR / "figures"

TOP_PAIR_COUNT = 15


def conditional_cooccurrence(labels: np.ndarray) -> np.ndarray:
    """행 질환이 있는 이미지 중 열 질환도 있는 비율(%)

    값[i, j] = P(질환 j | 질환 i) × 100
    개수 행렬은 흔한 질환이 색을 지배하므로,
    각 행을 해당 질환의 양성 수로 나눠 질환 간 관계를 비교한다.
    양성이 없는 질환의 행은 NaN이다.
    """
    counts = labels.T.astype(np.int64) @ labels.astype(np.int64)
    positives = np.diag(counts).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        rates = counts / positives[:, None] * 100

    rates[positives == 0] = np.nan
    return rates


def top_pairs(labels: np.ndarray, names: list[str], n: int) -> pd.DataFrame:
    """함께 표시된 이미지가 많은 질환 쌍과 양방향 조건부 비율"""
    counts = labels.T.astype(np.int64) @ labels.astype(np.int64)
    rates = conditional_cooccurrence(labels)

    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rows.append({
                "Disease A": names[i],
                "Disease B": names[j],
                "Images": int(counts[i, j]),
                "P(B|A) (%)": rates[i, j],
                "P(A|B) (%)": rates[j, i],
            })

    return (
        pd.DataFrame(rows)
        .sort_values("Images", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def single_label_rates(labels: np.ndarray, names: list[str]) -> pd.DataFrame:
    """각 질환 양성 이미지 중 다른 질환 없이 단독으로 표시된 비율"""
    label_count = labels.sum(axis=1)
    rows = []

    for i, name in enumerate(names):
        positive = labels[:, i] == 1
        total = int(positive.sum())
        alone = int((positive & (label_count == 1)).sum())
        rows.append({
            "Disease": name,
            "Positive Images": total,
            "Single-label Images": alone,
            "Single-label Rate (%)": alone / total * 100 if total else np.nan,
        })

    return pd.DataFrame(rows).sort_values("Single-label Rate (%)").reset_index(drop=True)


def main():
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        METADATA_PATH,
        usecols=["Split", "Fold", *DISEASES],
    )

    # 기존 동시 출현 분석과 같은 대상: 품질 검사를 통과한 Train
    train_df = df.loc[
        df["Split"].eq("train") & df["Fold"].ge(0)
    ]

    if train_df.empty:
        raise ValueError("분석할 Train 이미지가 없습니다.")

    labels = train_df[DISEASES].to_numpy(dtype=np.int64)
    rates = conditional_cooccurrence(labels)

    rate_table = pd.DataFrame(rates, index=DISEASES, columns=DISEASES)
    rate_table.index.name = "Given Disease"

    # 상위 쌍과 단독 표시 비율은 화면에만 출력하고 보고서 표로 정리한다
    pairs = top_pairs(labels, DISEASES, TOP_PAIR_COUNT)
    single = single_label_rates(labels, DISEASES)

    rate_path = REPORT_TABLES_DIR / "train_conditional_cooccurrence.csv"
    image_path = REPORT_FIGURES_DIR / "train_conditional_cooccurrence.png"

    rate_table.to_csv(rate_path, encoding="utf-8-sig")

    # 대각선(자기 자신, 항상 100%)은 색 범위를 왜곡하므로 비워 둔다
    display = rates.copy()
    np.fill_diagonal(display, np.nan)

    fig, ax = plt.subplots(figsize=(14, 12))
    heatmap = ax.imshow(display, cmap="Reds", vmin=0)

    positions = np.arange(len(DISEASES))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(DISEASES, rotation=60, ha="right")
    ax.set_yticklabels(DISEASES)
    ax.set_xlabel("Also present (column)")
    ax.set_ylabel("Given (row)")
    ax.set_title(
        f"Train Conditional Co-occurrence (n={len(train_df):,})\n"
        "Cell = % of row-disease images that also have the column disease"
    )

    threshold = np.nanmax(display) * 0.6

    for row in positions:
        for column in positions:
            if row == column:
                continue
            value = display[row, column]
            ax.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > threshold else "black",
            )

    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("P(column | row) (%)")

    fig.tight_layout()
    fig.savefig(image_path, dpi=180)
    plt.close(fig)

    print(f"분석 대상 Train 이미지: {len(train_df):,}장")
    print("\n[동시 표시 상위 질환 쌍]")
    print(pairs.round(1).to_string(index=False))
    print("\n[단독 표시 비율]")
    print(single.round(1).to_string(index=False))
    print(f"\n조건부 비율 CSV 저장: {rate_path}")
    print(f"히트맵 저장: {image_path}")


if __name__ == "__main__":
    main()
