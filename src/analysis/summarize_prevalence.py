from src.paths import PROCESSED_DIR, PREPROCESSING_REPORT_DIR

import pandas as pd

from src.constants import DISEASES


METADATA_PATH = PROCESSED_DIR / "fold_metadata.csv"
REPORT_TABLES_DIR = PREPROCESSING_REPORT_DIR / "tables"

LABELS = [*DISEASES, "No Finding"]


def prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    """질환별 이미지 기준·환자 기준 유병률

    - 이미지 기준: 해당 라벨이 있는 이미지 비율
    - 환자 기준: 한 번이라도 해당 라벨이 기록된 환자 비율
    """
    if df.empty:
        raise ValueError("유병률을 계산할 데이터가 없습니다.")

    image_counts = df[LABELS].sum()
    patient_counts = df.groupby("Patient ID")[LABELS].max().sum()
    patient_total = df["Patient ID"].nunique()

    return pd.DataFrame({
        "Disease": LABELS,
        "Positive Images": image_counts.to_numpy(),
        "Image Prevalence (%)": (
            image_counts / len(df) * 100
        ).to_numpy(),
        "Positive Patients": patient_counts.to_numpy(),
        "Patient Prevalence (%)": (
            patient_counts / patient_total * 100
        ).to_numpy(),
    })


def main():
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        METADATA_PATH,
        usecols=["Patient ID", "Split", "Fold", *LABELS],
    )

    # 데이터셋 전체의 분포를 기술하는 표
    # 모델 선택·임계값 결정에는 사용하지 않는다
    table = prevalence_table(df)

    # 세트별 이미지 기준 유병률 비교
    # Train은 품질 검사를 통과하고 Fold가 배정된 이미지만 사용
    eligible_train = df["Split"].eq("train") & df["Fold"].ge(0)
    test = df["Split"].eq("test")

    table["Train Prevalence (%)"] = (
        df.loc[eligible_train, LABELS].mean() * 100
    ).to_numpy()
    table["Test Prevalence (%)"] = (
        df.loc[test, LABELS].mean() * 100
    ).to_numpy()
    table["Train-Test Gap (%p)"] = (
        table["Train Prevalence (%)"] - table["Test Prevalence (%)"]
    )

    # No Finding은 마지막 행에 유지하고 질환만 많은 순으로 정렬
    diseases = table.loc[table["Disease"].ne("No Finding")]
    no_finding = table.loc[table["Disease"].eq("No Finding")]
    table = pd.concat([
        diseases.sort_values("Positive Images", ascending=False),
        no_finding,
    ]).reset_index(drop=True)

    output_path = REPORT_TABLES_DIR / "dataset_prevalence.csv"
    table.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(
        f"[전체 데이터 유병률 - 이미지 {len(df):,}장, "
        f"환자 {df['Patient ID'].nunique():,}명]"
    )
    print(table.round(2).to_string(index=False))
    print(
        f"\nTrain {int(eligible_train.sum()):,}장 / "
        f"Test {int(test.sum()):,}장"
    )
    print(
        "Train-Test 최대 차이: "
        f"{table['Train-Test Gap (%p)'].abs().max():.2f}%p"
    )
    print(f"\n유병률 표 저장: {output_path}")


if __name__ == "__main__":
    main()
