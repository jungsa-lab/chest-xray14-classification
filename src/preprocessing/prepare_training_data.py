from src.paths import RAW_DIR
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from src.paths import PROJECT_ROOT
import argparse
METADATA_PATH = RAW_DIR / "Data_Entry_2017.csv"
IMAGE_DIR = RAW_DIR / "images"

REPORT_TABLES_DIR = (
    PROJECT_ROOT / "reports" / "preprocessing" / "tables"
)

QUALITY_REPORT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "image_quality_audit.csv"
)

SEED = 42

SPLIT_METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "split_metadata.csv"
)

from src.constants import DISEASES

MIN_MEAN_BRIGHTNESS = 5.0
MAX_MEAN_BRIGHTNESS = 250.0
MIN_BRIGHTNESS_STD = 3.0

QUALITY_METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "quality_metadata.csv"
)

EXCLUDED_TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "excluded_train_images.csv"
)

N_FOLDS = 5

FOLD_METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "fold_metadata.csv"
)

FOLD_SUMMARY_PATH = REPORT_TABLES_DIR / "fold_summary.csv"


def load_metadata() -> pd.DataFrame:
    """원본 메타데이터를 읽고 불필요한 열 제거"""
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"메타데이터 파일을 찾을 수 없습니다: {METADATA_PATH}"
        )

    df = pd.read_csv(METADATA_PATH)

    required_columns = {
        "Image Index",
        "Finding Labels",
        "Patient ID",
        "Patient Age",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"필수 열이 없습니다: {sorted(missing_columns)}")

    unnamed_columns = [
        column for column in df.columns
        if column.startswith("Unnamed:")
    ]
    df = df.drop(columns=unnamed_columns)

    return df

def preprocess_age(df: pd.DataFrame) -> pd.DataFrame:
    """Y/M/D 단위를 반영해 나이를 연 단위로 변환"""
    df = df.copy()

    # 원본 Patient Age는 보존하고, 계산용 문자열 만듦
    raw_age = df["Patient Age"].astype("string").str.strip().str.upper()

    # 예: 058Y -> value=058, unit=Y
    # 형식이 맞지 않는 값은 추출 결과가 결측값이 됨
    parts = raw_age.str.extract(
        r"^(?P<value>\d+)(?P<unit>[YMD])$"
    )
    values = pd.to_numeric(parts["value"], errors="coerce")

    # 연 단위 환산: 년은 그대로, 월은 /12, 일은 /365.25
    divisors = parts["unit"].map({
        "Y": 1.0,
        "M": 12.0,
        "D": 365.25,
    })
    age_years = values / divisors

    age_missing = raw_age.isna() | raw_age.eq("").fillna(False)

    # 프로젝트 처리 기준: 0~120년 범위만 유효하게 사용
    age_valid = age_years.between(0, 120).fillna(False)

    df["Age Missing"] = age_missing
    df["Age Valid"] = age_valid

    # 이상 값은 실제 계산용 열에서도 결측으로 바꿈
    df["Patient Age Years"] = age_years.where(age_valid)

    return df

def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """질환 라벨을 검사하고 14개의 0/1 정답 열 만듦"""
    df = df.copy()

    raw_labels =  df["Finding Labels"].astype("string").str.strip()

    # 정답이 비어 있으면 질환 없음으로 취급하지 않고 중단
    missing_mask = raw_labels.isna() | raw_labels.eq("").fillna(False)

    if missing_mask.any():
        examples = df.loc[
            missing_mask, "Image Index"
        ].head(5).tolist()

        raise ValueError(
            f"질환 라벨이 비어 있는 행: {int(missing_mask.sum())}개."
            f"예시: {examples}"
        )

    # 예: "Cardiomegaly|Effusion"
    # -> {"Cardiomegaly", "Effusion"}
    labels_sets = raw_labels.apply(
        lambda text: {label.strip() for label in text.split("|")}
    )

    allowed_labels = set(DISEASES) | {"No Finding"}
    found_labels = set().union(*labels_sets.tolist())
    unknown_labels = found_labels - allowed_labels

    if unknown_labels:
        raise ValueError(
            f"정의되지 않은 질환 라벨: {sorted(unknown_labels)}"
        )


    # No Finding과 질환이 동시에 적혀 있으면 모순
    conflict_mask = labels_sets.apply(
        lambda labels: "No Finding" in labels and len(labels) > 1
    )    

    if conflict_mask.any():
        examples = df.loc[
            conflict_mask, "Image Index"
        ].head(5).tolist()

        raise ValueError(
            f"No Finding과 질환이 함께 있는 행: "
            f"{int(conflict_mask.sum())}개. 예시: {examples}"
        )

    for disease in DISEASES:
        df[disease] = labels_sets.apply(
            lambda labels: int(disease in labels)
        ).astype("int8")

    df["No Finding"] = labels_sets.apply(
        lambda labels: int("No Finding" in labels)
    ).astype("int8")

    df["Positive Label Count"] = df[DISEASES].sum(axis=1)

    return df

def match_images(df: pd.DataFrame) -> pd.DataFrame:
    """메타데이터의 이미지 파일명을 실제 파일과 연결"""
    df = df.copy()

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"이미지 폴더를 찾을 수 없습니다: {IMAGE_DIR}"
        )

    # 파일명과 환자 ID는 이후 매칭/분할에 반드시 필요
    for column in ["Image Index", "Patient ID"]:
        values = df[column].astype("string").str.strip()
        missing_mask = values.isna() | values.eq("").fillna(False)

        if missing_mask.any():
            raise ValueError(
                f"{column}이 비어 있는 행: "
                f"{int(missing_mask.sum())}개"
            )

    # 같은 이미지가 여러 행에 등록되어 있으면 중단
    duplicate_mask = df["Image Index"].duplicated(keep=False)

    if duplicate_mask.any():
        examples = df.loc[
            duplicate_mask, "Image Index"
        ].head(5).tolist()

        raise ValueError(f"메타데이터의 이미지 파일명 중복: {examples}")

    # 하위 폴더까지 탐색해 파일명 -> 실제 경로 연결표 생성
    image_paths = {}

    for path in IMAGE_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue

        if path.name in image_paths:
            raise ValueError(
                f"서로 다른 위치에 같은 이미지 파일명이 있습니다:\n"
                f"{image_paths[path.name]}\n"
                f"{path}"
            )

        image_paths[path.name] = path

    if not image_paths:
        raise ValueError(f"PNG 이미지가 없습니다: {IMAGE_DIR}")

    # 경로는 data/raw 기준 상대경로로 저장
    relative_paths = {
        name: path.relative_to(RAW_DIR).as_posix()
        for name, path in image_paths.items()
    }

    df["Image Path"] = df["Image Index"].map(relative_paths)

    missing_mask = df["Image Path"].isna()

    if missing_mask.any():
        examples = df.loc[
            missing_mask, "Image Index"
        ].head(5).tolist()

        raise ValueError(
            f"실제 파일이 없는 이미지: "
            f"{int(missing_mask.sum())}개. 예시: {examples}"
        )

    extra_count = len(
        set(image_paths) - set(df["Image Index"])
    )

    print("\n[이미지 파일 매칭]")
    print(f"발견한 PNG 파일: {len(image_paths):,}")
    print(f"메타데이터와 연결된 행: {len(df):,}")
    print(f"메타데이터에 없는 추가 PNG: {extra_count:,}")

    return df

def inspect_images(df: pd.DataFrame) -> pd.DataFrame:
    """이미지를 실제로 읽고 크기/밝기/읽기 오류를 기록"""
    records = []

    rows = df[["Image Index", "Image Path"]].itertuples(
        index=False, name=None
    )

    for image_name, relative_path in tqdm(
        rows, total=len(df), desc="이미지 검사"
    ):
        record = {
            "Image Index": image_name,
            "Image Path": relative_path,
            "Readable": False,
            "Width": None,
            "Height": None,
            "Mode": None,
            "Expected Size": False,
            "Mean Brightness": np.nan,
            "Brightness Std": np.nan,
            "Error": "",
        }

        image_path = RAW_DIR / relative_path

        try:
            with Image.open(image_path) as image:
                # 파일 정보뿐 아니라 픽셀까지 실제로 읽기
                image.load()

                record["Readable"] = True
                record["Width"], record["Height"] = image.size
                record["Mode"] = image.mode
                record["Expected Size"] = image.size == (224, 224)

                # 예상하지 않은 형식은 임의 변환하지 않고 기록
                if image.mode not in {"L", "RGB", "LA"}:
                    record["Error"] = (
                        f"확인이 필요한 이미지 모드: {image.mode}"
                    )

                elif (
                    image.mode == "LA"
                    and image.getchannel("A").getextrema() != (255, 255)
                ):
                    record["Error"] = "투명 또는 반투명 픽셀 포함"

                else:
                    # 완전 불투명한 LA는 밝기 채널만 사용
                    pixels = np.asarray(
                        image.convert("L"), dtype=np.float32
                    )

                    record["Mean Brightness"] = float(pixels.mean())
                    record["Brightness Std"] = float(pixels.std())

        except (OSError, ValueError, SyntaxError) as error:
            record["Error"] = f"{type(error).__name__}: {error}"

        records.append(record)

    quality_df = pd.DataFrame(records)

    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    quality_df.to_csv(
        QUALITY_REPORT_PATH, index=False, encoding="utf-8-sig"
    )

    print("\n[이미지 검사 요약]")
    print(f"검사한 이미지: {len(quality_df):,}")
    print(
        "읽기 실패: "
        f"{int((~quality_df['Readable']).sum()):,}"
    )
    print(
        "읽기는 성공했지만 크기가 다른 이미지: "
        f"{int((quality_df['Readable'] & ~quality_df['Expected Size']).sum()):,}"
    )
    print(
        "오류 또는 추가 확인 기록: "
        f"{int(quality_df['Error'].ne('').sum()):,}"
    )

    print("\n[이미지 모드별 개수]")
    print(quality_df["Mode"].value_counts(dropna=False).to_string())

    measured = quality_df.dropna(subset=["Mean Brightness"])

    print("\n[이미지별 평균 밝기 분포]")
    print(
        measured["Mean Brightness"].describe(
            percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
        ).to_string()
    )

    print("\n[평균 밝기가 가장 낮은 5장]")
    print(
        measured.nsmallest(5, "Mean Brightness")[
            ["Image Index", "Mean Brightness", "Brightness Std"]
        ].to_string(index=False)
    )

    print("\n[평균 밝기가 가장 높은 5장]")
    print(
        measured.nlargest(5, "Mean Brightness")[
            ["Image Index", "Mean Brightness", "Brightness Std"]
        ].to_string(index=False)
    )

    print(f"\n검사 결과 저장: {QUALITY_REPORT_PATH}")

    return quality_df

def split_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    """같은 환자의 이미지를 묶어서 Train/Test로 분할"""
    df = df.copy().reset_index(drop=True)

    if df["Patient ID"].isna().any():
        raise ValueError("Patient ID에 결측값이 있습니다.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=SEED,
    )

    train_indices, test_indices = next(
        splitter.split(df, groups=df["Patient ID"])
    )

    df["Split"] = ""
    df.loc[train_indices, "Split"] = "train"
    df.loc[test_indices, "Split"] = "test"

    train_patients = set(
        df.loc[df["Split"] == "train", "Patient ID"]
    )
    test_patients = set(
        df.loc[df["Split"] == "test", "Patient ID"]
    )

    overlap = train_patients & test_patients

    if overlap:
        raise ValueError(
            f"Train/Test에 중복된 환자가 있습니다: {len(overlap)}명"
        )

    # 이미 저장된 분할이 있다면 임의로 변경하지 않도록 확인
    if SPLIT_METADATA_PATH.is_file():
        previous = pd.read_csv(
            SPLIT_METADATA_PATH,
            usecols=["Image Index", "Patient ID", "Split"],
        )

        current = df[["Image Index", "Patient ID", "Split"]]

        previous = previous.sort_values("Image Index").reset_index(drop=True)
        current = current.sort_values("Image Index").reset_index(drop=True)

        if not previous.equals(current):
            raise ValueError(
                "기존에 저장된 분할과 다릅니다. "
                "원본 데이터 또는 분할 설정의 변경 여부를 확인하세요."
            )

    SPLIT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        SPLIT_METADATA_PATH, index=False, encoding="utf-8-sig"
    )

    print("\n[환자 기준 Train/Test 분할]")

    for split in ["train", "test"]:
        subset = df.loc[df["Split"] == split]
        patient_count = subset["Patient ID"].nunique()

        print(
            f"{split}: 이미지 {len(subset):,}장 / "
            f"환자 {patient_count:,}명"
        )

    print(f"중복 환자: {len(overlap)}명")
    print(f"분할 결과 저장: {SPLIT_METADATA_PATH}")

    return df


def flag_image_quality(df: pd.DataFrame) -> pd.DataFrame:
    """고정된 품질 기준으로 제외 사유를 표시"""
    df = df.copy()

    measurement_columns = ["Mean Brightness", "Brightness Std"]

    if df[measurement_columns].isna().any().any():
        raise ValueError("밝기 측정값에 결측이 있습니다.")

    df["Too Dark"] = (
        df["Mean Brightness"] < MIN_MEAN_BRIGHTNESS
    )
    df["Too Bright"] = (
        df["Mean Brightness"] > MAX_MEAN_BRIGHTNESS
    )
    df["Nearly Uniform"] = (
        df["Brightness Std"] < MIN_BRIGHTNESS_STD
    )

    def get_reason(row):
        reasons = []

        if row["Too Dark"]:
            reasons.append("too_dark")
        if row["Too Bright"]:
            reasons.append("too_bright")
        if row["Nearly Uniform"]:
            reasons.append("nearly_uniform")

        return "|".join(reasons)

    df["Quality Exclusion Reason"] = df.apply(get_reason, axis=1)
    df["Quality Pass"] = df["Quality Exclusion Reason"].eq("")

    train_mask = df["Split"].eq("train")
    excluded_train = df.loc[
        train_mask & ~df["Quality Pass"]
    ].copy()

    df.to_csv(
        QUALITY_METADATA_PATH, index=False, encoding="utf-8-sig"
    )
    excluded_train.to_csv(
        EXCLUDED_TRAIN_PATH, index=False, encoding="utf-8-sig"
    )

    print("\n[Train 품질 필터링]")
    print(f"필터링 전: {int(train_mask.sum()):,}장")

    for column in ["Too Dark", "Too Bright", "Nearly Uniform"]:
        count = int(df.loc[train_mask, column].sum())
        print(f"{column}: {count:,}장")

    print(f"중복 없이 제외할 이미지: {len(excluded_train):,}장")
    print(
        "학습에 사용할 이미지: "
        f"{int((train_mask & df['Quality Pass']).sum()):,}장"
    )

    if not excluded_train.empty:
        print("\n[제외 대상 — 최대 20장]")
        print(
            excluded_train[
                [
                    "Image Index",
                    "Mean Brightness",
                    "Brightness Std",
                    "Quality Exclusion Reason",
                ]
            ].head(20).to_string(index=False)
        )

    print(f"\n품질 표시 메타데이터: {QUALITY_METADATA_PATH}")
    print(f"Train 제외 목록: {EXCLUDED_TRAIN_PATH}")

    return df

def assign_folds(df: pd.DataFrame) -> pd.DataFrame:
    """품질 기준을 통과한 Train에 환자 기준 Fold를 배정한다."""
    df = df.copy().reset_index(drop=True)

    eligible_mask = df["Split"].eq("train") & df["Quality Pass"]
    eligible = df.loc[eligible_mask].copy()

    if eligible["Patient ID"].nunique() < N_FOLDS:
        raise ValueError("Fold 수보다 고유 환자 수가 적습니다.")

    # -1: Fold 배정 대상이 아님(Test 또는 품질 기준 탈락)
    df["Fold"] = -1

    splitter = GroupKFold(n_splits=N_FOLDS)
    summary_rows = []

    test_patients = set(
        df.loc[df["Split"] == "test", "Patient ID"]
    )

    for fold, (train_positions, val_positions) in enumerate(
        splitter.split(eligible, groups=eligible["Patient ID"])
    ):
        fold_train = eligible.iloc[train_positions]
        fold_val = eligible.iloc[val_positions]

        train_patients = set(fold_train["Patient ID"])
        val_patients = set(fold_val["Patient ID"])

        overlap = train_patients & val_patients

        if overlap:
            raise ValueError(f"Fold {fold}: 학습·검증 환자 중복")

        if (train_patients | val_patients) & test_patients:
            raise ValueError(f"Fold {fold}: Test 환자가 포함되어 있습니다.")

        # 검증으로 배정된 행에 해당 Fold 번호 기록
        df.loc[fold_val.index, "Fold"] = fold

        # 각 질환의 AUROC 등을 계산할 수 있는지 확인
        for subset_name, subset in [
            ("train", fold_train),
            ("val", fold_val),
        ]:
            positive_counts = subset[DISEASES].sum()

            invalid_classes = positive_counts[
                (positive_counts == 0)
                | (positive_counts == len(subset))
            ].index.tolist()

            if invalid_classes:
                raise ValueError(
                    f"Fold {fold} {subset_name}: "
                    f"양성 또는 음성이 없는 질환: {invalid_classes}"
                )

        summary_rows.append({
            "Fold": fold,
            "Train Images": len(fold_train),
            "Val Images": len(fold_val),
            "Train Patients": len(train_patients),
            "Val Patients": len(val_patients),
            "Patient Overlap": len(overlap),
        })

    # 모든 대상 이미지가 배정됐는지 확인
    if not df.loc[eligible_mask, "Fold"].between(0, N_FOLDS - 1).all():
        raise ValueError("Fold가 배정되지 않은 Train 이미지가 있습니다.")

    patient_fold_counts = (
        df.loc[eligible_mask]
        .groupby("Patient ID")["Fold"]
        .nunique()
    )

    if patient_fold_counts.gt(1).any():
        raise ValueError("같은 환자가 여러 Fold에 배정됐습니다.")

    # 재실행 시 기존 배정이 바뀌면 덮어쓰기 전에 중단
    comparison_columns = ["Image Index", "Patient ID", "Split", "Fold"]

    if FOLD_METADATA_PATH.is_file():
        previous = pd.read_csv(
            FOLD_METADATA_PATH, usecols=comparison_columns
        )
        current = df[comparison_columns]

        previous = previous.sort_values("Image Index").reset_index(drop=True)
        current = current.sort_values("Image Index").reset_index(drop=True)

        if not previous.equals(current):
            raise ValueError(
                "기존 Fold 배정과 다릅니다. "
                "데이터·품질 기준·분할 설정 변경 여부를 확인하세요."
            )

    summary_df = pd.DataFrame(summary_rows)

    df.to_csv(
        FOLD_METADATA_PATH, index=False, encoding="utf-8-sig"
    )
    summary_df.to_csv(
        FOLD_SUMMARY_PATH, index=False, encoding="utf-8-sig"
    )

    print("\n[5-Fold 배정 결과]")
    print(summary_df.to_string(index=False))
    print(f"\nFold 배정 이미지: {int(eligible_mask.sum()):,}장")
    print("각 Fold의 14개 질환 모두 양성·음성 존재 확인")
    print(f"메타데이터 저장: {FOLD_METADATA_PATH}")
    print(f"분할 요약 저장: {FOLD_SUMMARY_PATH}")

    return df


def load_or_inspect_images(df: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """검사표가 없으면 생성하고, 있으면 대상과 측정값을 확인해 재사용한다."""
    if refresh or not QUALITY_REPORT_PATH.is_file():
        quality_df = inspect_images(df)
    else:
        print(f"\n기존 이미지 검사표 재사용: {QUALITY_REPORT_PATH}")
        quality_df = pd.read_csv(QUALITY_REPORT_PATH, low_memory=False)

    required = {
        "Image Index", "Image Path", "Readable", "Expected Size",
        "Mean Brightness", "Brightness Std", "Error",
    }
    missing = required - set(quality_df.columns)
    if missing:
        raise ValueError(f"이미지 검사표 필수 열 누락: {sorted(missing)}")
    if (quality_df["Image Index"].isna().any()
            or quality_df["Image Index"].duplicated().any()
            or set(quality_df["Image Index"]) != set(df["Image Index"])):
        raise ValueError("이미지 검사표와 현재 메타데이터가 일치하지 않습니다.")

    current_paths = df.set_index("Image Index")["Image Path"].sort_index()
    saved_paths = quality_df.set_index("Image Index")["Image Path"].sort_index()
    if not current_paths.eq(saved_paths).all():
        raise ValueError("이미지 경로가 변경됐습니다. --refresh-image-audit로 재검사하세요.")

    quality_df["Error"] = quality_df["Error"].fillna("")
    for column in ["Readable", "Expected Size"]:
        values = quality_df[column].astype("string").str.lower()
        if not values.eq("true").fillna(False).all():
            raise ValueError(f"이미지 검사표의 {column}에 실패 또는 결측이 있습니다.")
        quality_df[column] = True
    stats = quality_df[["Mean Brightness", "Brightness Std"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if (quality_df["Error"].ne("").any()
            or not np.isfinite(stats.to_numpy(dtype=float)).all()
            or not stats["Mean Brightness"].between(0, 255).all()
            or stats["Brightness Std"].lt(0).any()):
        raise ValueError("이미지 검사에 미해결 항목 또는 잘못된 밝기 측정값이 있습니다.")
    quality_df[stats.columns] = stats
    return quality_df


def main():
    parser = argparse.ArgumentParser(description="메타데이터·품질 검사·환자 분할·Fold 생성")
    parser.add_argument("--refresh-image-audit", action="store_true",
                        help="저장된 검사표 대신 전체 이미지를 다시 검사")
    args = parser.parse_args()
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    df = load_metadata()
    df = preprocess_age(df)
    df = encode_labels(df)
    df = match_images(df)

    missing_count = int(df["Age Missing"].sum())
    invalid_count = int(
        (~df["Age Valid"] & ~df["Age Missing"]).sum()
    )

    print(f"메타데이터 행 수: {len(df):,}")
    print(f"고유 환자 수: {df['Patient ID'].nunique():,}")
    print(f"나이 결측: {missing_count:,}")
    print(f"나이 형식 오류 또는 범위 초과: {invalid_count:,}")

    # 월/일 단위가 올바르게 변환 됐는지 확인
    raw_age = df["Patient Age"].astype("string").str.strip().str.upper()
    month_day_mask = raw_age.str.endswith(("M", "D"), na=False)

    print("\n[월/일 단위 변환 예시]")
    print(
        df.loc[
            month_day_mask,
            ["Image Index", "Patient Age", "Patient Age Years"],
        ].head(10).to_string(index=False)
    )

    print("\n[유효하지 않은 나이 예시]")
    print(
        df.loc[
            ~df["Age Valid"],
            ["Image Index", "Patient Age", "Patient Age Years"],
        ].head(20).to_string(index=False)
    )

    print("\n[질환 라벨 변환 예시]")
    print(
        df[
            [
                "Image Index",
                "Finding Labels",
                "Cardiomegaly",
                "Effusion",
                "No Finding",
                "Positive Label Count",
            ]
        ].head(10).to_string(index=False)
    )

    print("\n[라벨 요약]")
    print(f"질환 정답 열 수: {len(DISEASES)}")
    print(f"No Finding 이미지: {int(df['No Finding'].sum()):,}")
    print(
        "2개 이상 질환이 표시된 이미지: "
        f"{int((df['Positive Label Count'] >= 2).sum()):,}"
    )

    print("\n[이미지 경로 예시]")
    print(
        df[
            ["Image Index", "Image Path"]
        ].head(5).to_string(index=False)
    )

    quality_df = load_or_inspect_images(df, refresh=args.refresh_image_audit)

    # 기존 Image Path와 중복되지 않도록 검사표 쪽 경로 열 제외
    df = df.merge(
        quality_df.drop(columns=["Image Path"]),
        on="Image Index",
        how="left",
        validate="one_to_one",
    )

    df = split_by_patient(df)
    df = flag_image_quality(df)
    df = assign_folds(df)


if __name__ == "__main__":
    main()
