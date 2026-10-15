from src.paths import PROCESSED_DIR

import pandas as pd

from src.constants import DISEASES


from src.paths import PROJECT_ROOT
METADATA_PATH = PROCESSED_DIR / "fold_metadata.csv"

REPORT_TABLES_DIR = (
    PROJECT_ROOT / "reports" / "preprocessing" / "tables"
)

def main():
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        METADATA_PATH,
        usecols=["Split", "Fold", *DISEASES],
    )

    # Fold가 배정된, 품질 검사를 통과한 Train만 사용
    train_df = df.loc[
        df["Split"].eq("train") & df["Fold"].ge(0)
    ].copy()

    if train_df.empty:
        raise ValueError("분석할 Train 이미지가 없습니다.")

    positive_counts = train_df[DISEASES].sum()

    summary = pd.DataFrame({
        "Disease": DISEASES,
        "Positive Images": positive_counts.to_numpy(),
        "Negative Images": (
            len(train_df) - positive_counts
        ).to_numpy(),
        "Positive Rate (%)": (
            positive_counts / len(train_df) * 100
        ).to_numpy(),
    })

    summary = summary.sort_values(
        "Positive Images", ascending=False
    ).reset_index(drop=True)

    # 각 Fold가 검증 세트로 사용될 때의 질환별 양성 비율
    fold_rates = (
        train_df.groupby("Fold")[DISEASES].mean() * 100
    ).T

    fold_rates.columns = [
        f"Fold {int(fold)} (%)"
        for fold in fold_rates.columns
    ]
    fold_rates.index.name = "Disease"

    # summary_path = PROCESSED_DIR / "train_label_summary.csv"
    # fold_path = PROCESSED_DIR / "fold_label_rates.csv"
    summary_path = REPORT_TABLES_DIR / "train_label_summary.csv"
    fold_path = REPORT_TABLES_DIR / "fold_label_rates.csv"

    summary.to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )
    fold_rates.to_csv(
        fold_path, encoding="utf-8-sig"
    )

    print(f"[Train 질환별 분포 — 총 {len(train_df):,}장]")
    print(summary.round(2).to_string(index=False))

    print("\n[검증 Fold별 질환 양성 비율 (%)]")
    print(fold_rates.round(2).to_string())

    print(f"\n질환별 요약 저장: {summary_path}")
    print(f"Fold별 비율 저장: {fold_path}")


if __name__ == "__main__":
    main()
