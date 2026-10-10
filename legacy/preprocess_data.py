from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# 0. 기본 설정
# ============================================================

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

EXPECTED_IMAGE_SIZE = (224, 224)

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 1. NIH ChestX-ray14 질환 목록
# ============================================================

DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


# ============================================================
# 2. Git 프로젝트 루트 자동 탐색
# ============================================================

def find_project_root() -> Path:
    """
    현재 preprocess_data.py 위치에서 상위 디렉터리를 탐색하여
    Git 프로젝트의 루트 디렉터리를 찾는다.

    우선 .git 디렉터리를 확인한다.

    .git 디렉터리를 찾을 수 없는 환경에서는
    README.md와 requirements.txt가 동시에 존재하는 위치를
    프로젝트 루트로 판단한다.
    """

    current = Path(__file__).resolve().parent

    for directory in [current, *current.parents]:

        if (directory / ".git").exists():
            return directory.resolve()

        if (
            (directory / "README.md").exists()
            and (directory / "requirements.txt").exists()
        ):
            return directory.resolve()

    raise RuntimeError(
        "프로젝트 루트 디렉터리를 찾을 수 없습니다.\n"
        "preprocess_data.py가 Git 프로젝트의 src 폴더 내부에 "
        "있는지 확인하세요."
    )


# ============================================================
# 3. 프로젝트 절대경로 설정
# ============================================================

PROJECT_ROOT = find_project_root()

DATA_DIR = (
    PROJECT_ROOT
    / "data"
).resolve()

RAW_DIR = (
    DATA_DIR
    / "raw"
).resolve()

IMAGE_DIR = (
    RAW_DIR
    / "images"
).resolve()

PROCESSED_DIR = (
    DATA_DIR
    / "processed"
).resolve()


# ------------------------------------------------------------
# 입력 데이터
# ------------------------------------------------------------

METADATA_PATH = (
    RAW_DIR
    / "Data_Entry_2017.csv"
).resolve()

BBOX_PATH = (
    RAW_DIR
    / "BBox_List_2017_Official_NIH.csv"
).resolve()


# ------------------------------------------------------------
# 출력 데이터
# ------------------------------------------------------------

PREPROCESSED_METADATA_PATH = (
    PROCESSED_DIR
    / "preprocessed_metadata.csv"
).resolve()

LABEL_DISTRIBUTION_PATH = (
    PROCESSED_DIR
    / "split_label_distribution.csv"
).resolve()

IMAGE_VALIDATION_PATH = (
    PROCESSED_DIR
    / "image_validation.csv"
).resolve()

PREPROCESSING_REPORT_PATH = (
    PROCESSED_DIR
    / "preprocessing_report.txt"
).resolve()

MISSING_SUMMARY_PATH = (
    PROCESSED_DIR
    / "missing_value_summary.csv"
).resolve()


# ============================================================
# 4. 프로젝트 경로 출력
# ============================================================

def print_project_paths():

    print("=" * 70)
    print("NIH ChestX-ray14 Data Preprocessing")
    print("=" * 70)

    print(f"Project Root : {PROJECT_ROOT}")
    print(f"Data         : {DATA_DIR}")
    print(f"Raw          : {RAW_DIR}")
    print(f"Images       : {IMAGE_DIR}")
    print(f"Processed    : {PROCESSED_DIR}")

    print()

    print(f"Metadata     : {METADATA_PATH}")
    print(f"BBox         : {BBOX_PATH}")


# ============================================================
# 5. 필수 파일 및 디렉터리 검사
# ============================================================

def validate_project_structure():

    print("\n[1/10] 프로젝트 구조 확인")

    required_directories = [
        DATA_DIR,
        RAW_DIR,
        IMAGE_DIR,
    ]

    for directory in required_directories:

        if not directory.exists():

            raise FileNotFoundError(
                f"필수 디렉터리를 찾을 수 없습니다:\n"
                f"{directory}"
            )

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            "Data_Entry_2017.csv를 찾을 수 없습니다:\n"
            f"{METADATA_PATH}"
        )

    if not BBOX_PATH.exists():

        print(
            "\n[WARNING] "
            "BBox_List_2017_Official_NIH.csv를 찾을 수 없습니다."
        )

        print(
            "Bounding Box 데이터는 현재 multi-label 분류 "
            "전처리의 필수 입력은 아니므로 계속 진행합니다."
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("프로젝트 구조 확인 완료")


# ============================================================
# 6. Metadata 로딩
# ============================================================

def load_metadata():

    print("\n[2/10] Metadata 로딩")

    df = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"전체 메타데이터 행: "
        f"{len(df):,}"
    )

    print(
        f"고유 환자 수: "
        f"{df['Patient ID'].nunique():,}"
    )

    print(
        f"컬럼 수: "
        f"{len(df.columns):,}"
    )

    return df


# ============================================================
# 7. Metadata 기본 정제
# ============================================================

def clean_metadata(df):

    print("\n[3/10] Metadata 기본 정제")

    # --------------------------------------------------------
    # Unnamed 컬럼 제거
    # --------------------------------------------------------

    unnamed_columns = [
        column
        for column in df.columns
        if column.startswith("Unnamed:")
    ]

    if unnamed_columns:

        df = df.drop(
            columns=unnamed_columns
        )

        print(
            "제거된 불필요 컬럼:",
            unnamed_columns
        )

    # --------------------------------------------------------
    # 컬럼명 정리
    # --------------------------------------------------------

    rename_map = {

        "OriginalImage[Width":
            "Original Width",

        "Height]":
            "Original Height",

        "OriginalImagePixelSpacing[x":
            "Pixel Spacing X",

        "y]":
            "Pixel Spacing Y",
    }

    df = df.rename(
        columns=rename_map
    )

    print("Metadata 기본 정제 완료")

    return df


# ============================================================
# 8. 결측치 및 중복 검사
# ============================================================

def check_missing_and_duplicates(df):

    print("\n[4/10] 결측치 및 중복 검사")

    # 빈 문자열/공백도 결측으로 취급하되 원본 df 자체는 변경하지 않는다.
    normalized = df.replace(r"^\s*$", np.nan, regex=True)
    missing_count = normalized.isna().sum()
    missing_rate = (missing_count / len(df) * 100) if len(df) else 0

    missing_summary = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": [int(missing_count[c]) for c in df.columns],
        "Missing Rate (%)": [float(missing_rate[c]) for c in df.columns],
    })

    print("\n컬럼별 결측치 요약:")
    print(missing_summary.to_string(index=False))

    duplicate_count = int(df["Image Index"].duplicated().sum())
    print(f"\nImage Index 중복: {duplicate_count:,}")

    return duplicate_count, missing_summary


# ============================================================
# 9. Patient Age 정제
# ============================================================

def preprocess_age(df):

    print("\n[5/10] Patient Age 정제")

    raw_age = df["Patient Age"]
    df["Patient Age Raw"] = raw_age.astype("string")
    df["Age Missing"] = raw_age.isna() | raw_age.astype("string").str.strip().eq("")

    df["Patient Age Years"] = pd.to_numeric(
        df["Patient Age Raw"].str.extract(r"(\d+)")[0],
        errors="coerce"
    )

    df["Age Valid"] = (
        (~df["Age Missing"])
        & df["Patient Age Years"].between(0, 120, inclusive="both")
    )

    df["Age Group"] = pd.cut(
        df["Patient Age Years"],
        bins=[-1, 19, 39, 59, 79, 120],
        labels=["0-19", "20-39", "40-59", "60-79", "80+"],
        include_lowest=True
    ).astype("string")
    df.loc[~df["Age Valid"], "Age Group"] = pd.NA

    missing_age_count = int(df["Age Missing"].sum())
    invalid_age_count = int((~df["Age Valid"] & ~df["Age Missing"]).sum())
    valid_age_count = int(df["Age Valid"].sum())

    print(f"연령 결측: {missing_age_count:,}")
    print(f"연령 이상값: {invalid_age_count:,}")
    print(f"유효 연령: {valid_age_count:,}")

    invalid_age_values = df.loc[
        ~df["Age Valid"] & ~df["Age Missing"],
        ["Image Index", "Patient ID", "Patient Age Raw", "Patient Age Years"]
    ]
    if not invalid_age_values.empty:
        print("\n비정상 연령 샘플:")
        print(invalid_age_values.head(20).to_string(index=False))

    return df, missing_age_count, invalid_age_count, valid_age_count


# ============================================================
# 10. Multi-label Encoding
# ============================================================

def encode_multilabel(df):

    print(
        "\n[6/10] 14개 질환 Multi-label Encoding"
    )

    label_sets = (
        df["Finding Labels"]
        .fillna("")
        .str.split("|")
        .apply(set)
    )

    # --------------------------------------------------------
    # 14개 질환 → 각각 0/1
    # --------------------------------------------------------

    for disease in DISEASES:

        df[disease] = (
            label_sets.apply(
                lambda labels, disease=disease:
                int(disease in labels)
            )
        )

    # --------------------------------------------------------
    # No Finding 별도 보존
    # --------------------------------------------------------

    df["No Finding"] = (
        label_sets.apply(
            lambda labels:
            int("No Finding" in labels)
        )
    )

    # --------------------------------------------------------
    # 하나의 영상에 존재하는 질환 개수
    # --------------------------------------------------------

    df["Positive Label Count"] = (
        df[DISEASES]
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # 두 개 이상의 질환을 가진 영상
    # --------------------------------------------------------

    df["Is Multi Label"] = (
        df["Positive Label Count"] > 1
    ).astype(int)

    multi_label_count = (
        df["Is Multi Label"]
        .sum()
    )

    no_finding_count = (
        df["No Finding"]
        .sum()
    )

    print(
        f"다중질환 영상: "
        f"{multi_label_count:,}"
    )

    print(
        f"No Finding 영상: "
        f"{no_finding_count:,}"
    )

    # 원본에 정의되지 않은 라벨과 No Finding/질환 동시 표기를 검사한다.
    allowed_labels = set(DISEASES) | {"No Finding", ""}
    unknown_label_values = sorted(set().union(*label_sets.tolist()) - allowed_labels)
    df["Label Conflict"] = ((df["No Finding"] == 1) & (df["Positive Label Count"] > 0)).astype(int)
    label_conflict_count = int(df["Label Conflict"].sum())

    print(f"Unknown Labels: {len(unknown_label_values):,}")
    if unknown_label_values:
        print("Unknown Label Values:", unknown_label_values)
    print(f"No Finding / Disease Conflicts: {label_conflict_count:,}")

    return (
        df,
        multi_label_count,
        no_finding_count,
        unknown_label_values,
        label_conflict_count
    )


# ============================================================
# 11. Uncertain Label 처리
# ============================================================

def handle_uncertainty(df):

    print(
        "\n[7/10] Uncertain Label 처리"
    )

    # --------------------------------------------------------
    # NIH ChestX-ray14의 제공 Metadata에는
    # CheXpert와 같은 explicit uncertain label 상태가 없다.
    #
    # 따라서 존재하지 않는 uncertain label을
    # 임의 생성하지 않는다.
    #
    # 이 컬럼은 데이터셋의 한계를 명시하기 위한 것이다.
    # --------------------------------------------------------

    df["Uncertainty Available"] = 0

    print(
        "Explicit uncertain label 없음 "
        "→ 임의 생성하지 않음"
    )

    return df


# ============================================================
# 12. 이미지 파일 탐색 및 Metadata 매칭
# ============================================================

def match_images(df):

    print(
        "\n[8/10] 이미지 파일 매칭"
    )

    # --------------------------------------------------------
    # 하위 디렉터리가 존재하더라도 모두 검색
    # --------------------------------------------------------

    image_files = list(
        IMAGE_DIR.rglob("*.png")
    )

    image_names = [
        image_path.name
        for image_path in image_files
    ]

    unique_image_names = set(
        image_names
    )

    print(
        f"발견된 PNG 파일: "
        f"{len(image_files):,}"
    )

    print(
        f"고유 이미지 파일명: "
        f"{len(unique_image_names):,}"
    )

    # --------------------------------------------------------
    # 동일 파일명이 여러 위치에 존재하는지 검사
    # --------------------------------------------------------

    duplicate_file_count = (
        len(image_names)
        - len(unique_image_names)
    )

    print(
        f"중복 이미지 파일명: "
        f"{duplicate_file_count:,}"
    )

    # --------------------------------------------------------
    # Metadata와 실제 이미지 매칭
    # --------------------------------------------------------

    df["Image Exists"] = (
        df["Image Index"]
        .isin(unique_image_names)
        .astype(int)
    )

    missing_images = (
        df.loc[
            df["Image Exists"] == 0,
            "Image Index"
        ]
    )

    print(
        f"Metadata에는 있으나 파일이 없는 이미지: "
        f"{len(missing_images):,}"
    )

    # --------------------------------------------------------
    # 이미지 절대경로 생성
    # --------------------------------------------------------

    image_path_map = {

        image_path.name:
            str(
                image_path.resolve()
            )

        for image_path in image_files
    }

    df["Absolute Image Path"] = (
        df["Image Index"]
        .map(image_path_map)
    )

    return (
        df,
        image_files,
        missing_images,
        duplicate_file_count
    )


# ============================================================
# 13. Patient ID 기준 Train / Validation / Test 분할
# ============================================================

def split_by_patient(df):

    print(
        "\n[9/10] Patient ID 기준 데이터 분할"
    )

    patient_ids = np.array(
        sorted(
            df["Patient ID"]
            .dropna()
            .unique()
        )
    )

    rng = (
        np.random
        .default_rng(SEED)
    )

    rng.shuffle(
        patient_ids
    )

    patient_count = (
        len(patient_ids)
    )

    train_end = int(
        patient_count
        * TRAIN_RATIO
    )

    val_end = (
        train_end
        + int(
            patient_count
            * VAL_RATIO
        )
    )

    train_patients = set(
        patient_ids[
            :train_end
        ]
    )

    val_patients = set(
        patient_ids[
            train_end:val_end
        ]
    )

    test_patients = set(
        patient_ids[
            val_end:
        ]
    )

    def assign_split(patient_id):

        if patient_id in train_patients:
            return "train"

        if patient_id in val_patients:
            return "val"

        return "test"

    df["Split"] = (
        df["Patient ID"]
        .apply(assign_split)
    )

    # --------------------------------------------------------
    # Patient Leakage 검사
    # --------------------------------------------------------

    patient_split_count = (
        df.groupby(
            "Patient ID"
        )["Split"]
        .nunique()
    )

    leakage_count = (
        patient_split_count
        .gt(1)
        .sum()
    )

    print(
        f"Patient Leakage: "
        f"{leakage_count}"
    )

    if leakage_count != 0:

        raise RuntimeError(
            "동일 Patient ID가 여러 데이터 Split에 "
            "포함되어 있습니다."
        )

    # --------------------------------------------------------
    # Split 결과 출력
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test",
    ]:

        split_df = (
            df[
                df["Split"] == split
            ]
        )

        print(
            f"{split.upper():5} | "
            f"Images: {len(split_df):,} | "
            f"Patients: "
            f"{split_df['Patient ID'].nunique():,}"
        )

    return (
        df,
        leakage_count
    )


# ============================================================
# 14. 이미지 무결성 검사
# ============================================================

def validate_images(image_files):

    print(
        "\n[10/10] 이미지 무결성 검사"
    )

    results = []

    for image_path in tqdm(
        image_files,
        desc="Validating images"
    ):

        valid = True

        width = None
        height = None

        mode = None

        error = ""

        try:

            # ------------------------------------------------
            # PNG 파일 자체의 손상 여부 검사
            # ------------------------------------------------

            with Image.open(
                image_path
            ) as image:

                image.verify()

            # ------------------------------------------------
            # verify() 이후 다시 열어서 속성 검사
            # ------------------------------------------------

            with Image.open(
                image_path
            ) as image:

                width, height = (
                    image.size
                )

                mode = (
                    image.mode
                )

                # --------------------------------------------
                # Kaggle 224x224 resized 버전 규격 확인
                # --------------------------------------------

                if (
                    width
                    != EXPECTED_IMAGE_SIZE[0]
                    or
                    height
                    != EXPECTED_IMAGE_SIZE[1]
                ):

                    valid = False

                    error = (
                        f"Unexpected image size: "
                        f"{width}x{height}"
                    )

        except Exception as exception:

            valid = False

            error = str(
                exception
            )

        results.append(
            {
                "Image Index":
                    image_path.name,

                "Absolute Path":
                    str(
                        image_path.resolve()
                    ),

                "Valid Image":
                    int(valid),

                "Width":
                    width,

                "Height":
                    height,

                "Mode":
                    mode,

                "Error":
                    error,
            }
        )

    validation_df = (
        pd.DataFrame(
            results
        )
    )

    invalid_images = (
        validation_df[
            validation_df[
                "Valid Image"
            ] == 0
        ]
    )

    print(
        f"\n손상/규격 이상 이미지: "
        f"{len(invalid_images):,}"
    )

    return (
        validation_df,
        invalid_images
    )


# ============================================================
# 14-1. 이미지 검증 결과를 Metadata에 병합
# ============================================================

def merge_image_validation(df, image_validation_df):
    cols = ["Image Index", "Valid Image", "Width", "Height", "Mode", "Error"]
    if image_validation_df.empty:
        for col in cols[1:]:
            df[col] = pd.NA
        return df

    validation = image_validation_df[cols].copy()
    return df.merge(validation, on="Image Index", how="left")


# ============================================================
# 15. Split별 질환 분포 생성
# ============================================================

def create_label_distribution(df):

    print(
        "\n질환별 Split 분포 계산"
    )

    rows = []

    for split in [
        "train",
        "val",
        "test",
    ]:

        split_df = (
            df[
                df["Split"] == split
            ]
        )

        total = (
            len(split_df)
        )

        for disease in DISEASES:

            positive = int(
                split_df[
                    disease
                ].sum()
            )

            prevalence = (
                positive / total
                if total > 0
                else 0.0
            )

            rows.append(
                {
                    "Split":
                        split,

                    "Disease":
                        disease,

                    "Positive Images":
                        positive,

                    "Total Images":
                        total,

                    "Prevalence":
                        prevalence,
                }
            )

    distribution_df = (
        pd.DataFrame(
            rows
        )
    )

    return distribution_df


# ============================================================
# 16. 전처리 결과 저장
# ============================================================

def save_results(
    df,
    distribution_df,
    image_validation_df,
    missing_summary
):

    print(
        "\n전처리 결과 저장"
    )

    df.to_csv(
        PREPROCESSED_METADATA_PATH,
        index=False
    )

    distribution_df.to_csv(
        LABEL_DISTRIBUTION_PATH,
        index=False
    )

    image_validation_df.to_csv(
        IMAGE_VALIDATION_PATH,
        index=False
    )

    missing_summary.to_csv(
        MISSING_SUMMARY_PATH,
        index=False
    )

    print(
        f"\nMetadata:\n"
        f"{PREPROCESSED_METADATA_PATH}"
    )

    print(
        f"\nLabel Distribution:\n"
        f"{LABEL_DISTRIBUTION_PATH}"
    )

    print(
        f"\nImage Validation:\n"
        f"{IMAGE_VALIDATION_PATH}"
    )

    print(
        f"\nMissing Value Summary:\n"
        f"{MISSING_SUMMARY_PATH}"
    )


# ============================================================
# 17. 전처리 Report 저장
# ============================================================

def save_report(
    df,
    duplicate_metadata_count,
    missing_summary,
    missing_age_count,
    invalid_age_count,
    valid_age_count,
    multi_label_count,
    no_finding_count,
    missing_images,
    duplicate_file_count,
    leakage_count,
    invalid_images,
    unknown_label_values,
    label_conflict_count
):

    lines = []

    lines.append(
        "NIH ChestX-ray14 Preprocessing Report"
    )

    lines.append(
        "=" * 60
    )

    lines.append(
        f"Total Images (Metadata): "
        f"{len(df):,}"
    )

    lines.append(
        f"Unique Patients: "
        f"{df['Patient ID'].nunique():,}"
    )

    lines.append("")
    lines.append("Missing Value Summary")
    lines.append("-" * 60)
    for _, row in missing_summary.iterrows():
        lines.append(
            f"{row['Column']}: {int(row['Missing Count']):,} "
            f"({row['Missing Rate (%)']:.3f}%)"
        )

    lines.append("")
    lines.append("Age Validation")
    lines.append("-" * 60)
    lines.append(f"Missing Age: {missing_age_count:,}")
    lines.append(f"Invalid Age: {invalid_age_count:,}")
    lines.append(f"Valid Age: {valid_age_count:,}")

    lines.append(
        f"Duplicate Metadata Image Index: "
        f"{duplicate_metadata_count:,}"
    )


    lines.append(
        f"Multi-label Images: "
        f"{multi_label_count:,}"
    )

    lines.append(
        f"No Finding Images: "
        f"{no_finding_count:,}"
    )

    lines.append(
        f"Missing Image Files: "
        f"{len(missing_images):,}"
    )

    lines.append(
        f"Duplicate Image Filenames: "
        f"{duplicate_file_count:,}"
    )

    lines.append(
        f"Invalid Image Files: "
        f"{len(invalid_images):,}"
    )

    lines.append(
        f"Patient Leakage: "
        f"{leakage_count:,}"
    )

    lines.append("")
    lines.append("Derived Variables")
    lines.append("-" * 60)
    lines.append("14 Disease Binary Targets: Created")
    lines.append("Positive Label Count: Created")
    lines.append("Is Multi Label: Created")
    lines.append("No Finding: Created")
    lines.append("Age Group: Created")
    lines.append("Split: Created")

    lines.append("")
    lines.append("Label Validation")
    lines.append("-" * 60)
    lines.append(f"Unknown Labels: {len(unknown_label_values):,}")
    if unknown_label_values:
        lines.append("Unknown Label Values: " + ", ".join(unknown_label_values))
    lines.append(f"No Finding / Disease Conflicts: {label_conflict_count:,}")

    lines.append("")

    lines.append(
        "Split Summary"
    )

    lines.append(
        "-" * 60
    )

    for split in [
        "train",
        "val",
        "test",
    ]:

        split_df = (
            df[
                df["Split"] == split
            ]
        )

        lines.append(
            f"{split}: "
            f"images={len(split_df):,}, "
            f"patients="
            f"{split_df['Patient ID'].nunique():,}"
        )

    lines.append("")

    lines.append(
        "Label Limitation"
    )

    lines.append(
        "-" * 60
    )

    lines.append(
        "NIH ChestX-ray14 labels were generated from "
        "radiology reports using NLP."
    )

    lines.append(
        "The supplied metadata does not contain an "
        "explicit uncertain-label state."
    )

    lines.append(
        "Uncertain labels were therefore not fabricated "
        "during preprocessing."
    )

    report = "\n".join(
        lines
    )

    PREPROCESSING_REPORT_PATH.write_text(
        report,
        encoding="utf-8"
    )

    print(
        f"\nPreprocessing Report:\n"
        f"{PREPROCESSING_REPORT_PATH}"
    )


# ============================================================
# 18. 최종 검증 결과 출력
# ============================================================

def print_final_summary(
    df,
    missing_images,
    invalid_images,
    leakage_count
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PREPROCESSING RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTotal Images: "
        f"{len(df):,}"
    )

    print(
        f"Unique Patients: "
        f"{df['Patient ID'].nunique():,}"
    )

    for split in [
        "train",
        "val",
        "test",
    ]:

        split_df = (
            df[
                df["Split"] == split
            ]
        )

        print(
            f"\n[{split.upper()}]"
        )

        print(
            f"Images   : "
            f"{len(split_df):,}"
        )

        print(
            f"Patients : "
            f"{split_df['Patient ID'].nunique():,}"
        )

    print(
        f"\nPatient Leakage : "
        f"{leakage_count}"
    )

    print(
        f"Missing Images  : "
        f"{len(missing_images):,}"
    )

    print(
        f"Invalid Images  : "
        f"{len(invalid_images):,}"
    )

    print(
        "\n전처리 완료"
    )


# ============================================================
# 19. Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 프로젝트 경로
    # --------------------------------------------------------

    print_project_paths()

    # --------------------------------------------------------
    # 폴더/파일 검사
    # --------------------------------------------------------

    validate_project_structure()

    # --------------------------------------------------------
    # Metadata 로딩
    # --------------------------------------------------------

    df = load_metadata()

    # --------------------------------------------------------
    # Metadata 정제
    # --------------------------------------------------------

    df = clean_metadata(
        df
    )

    # --------------------------------------------------------
    # 결측 / 중복
    # --------------------------------------------------------

    (
        duplicate_metadata_count,
        missing_summary
    ) = check_missing_and_duplicates(df)

    # --------------------------------------------------------
    # 연령 정제
    # --------------------------------------------------------

    (
        df,
        missing_age_count,
        invalid_age_count,
        valid_age_count
    ) = preprocess_age(df)

    # --------------------------------------------------------
    # Multi-label Encoding
    # --------------------------------------------------------

    (
        df,
        multi_label_count,
        no_finding_count,
        unknown_label_values,
        label_conflict_count
    ) = encode_multilabel(df)

    # --------------------------------------------------------
    # Uncertainty
    # --------------------------------------------------------

    df = handle_uncertainty(
        df
    )

    # --------------------------------------------------------
    # 이미지 ↔ Metadata 매칭
    # --------------------------------------------------------

    (
        df,
        image_files,
        missing_images,
        duplicate_file_count
    ) = match_images(
        df
    )

    # --------------------------------------------------------
    # Patient-level Split
    # --------------------------------------------------------

    (
        df,
        leakage_count
    ) = split_by_patient(
        df
    )

    # --------------------------------------------------------
    # 이미지 검증
    # --------------------------------------------------------

    (
        image_validation_df,
        invalid_images
    ) = validate_images(
        image_files
    )

    # 이미지 검증 결과를 최종 metadata에 병합
    df = merge_image_validation(df, image_validation_df)

    # --------------------------------------------------------
    # 질환별 분포
    # --------------------------------------------------------

    distribution_df = (
        create_label_distribution(
            df
        )
    )

    # --------------------------------------------------------
    # CSV 저장
    # --------------------------------------------------------

    save_results(
        df,
        distribution_df,
        image_validation_df,
        missing_summary
    )

    # --------------------------------------------------------
    # Report 저장
    # --------------------------------------------------------

    save_report(
        df,
        duplicate_metadata_count,
        missing_summary,
        missing_age_count,
        invalid_age_count,
        valid_age_count,
        multi_label_count,
        no_finding_count,
        missing_images,
        duplicate_file_count,
        leakage_count,
        invalid_images,
        unknown_label_values,
        label_conflict_count
    )

    # --------------------------------------------------------
    # 최종 출력
    # --------------------------------------------------------

    print_final_summary(
        df,
        missing_images,
        invalid_images,
        leakage_count
    )


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    main()
