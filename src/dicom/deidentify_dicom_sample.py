from copy import deepcopy
import json

import numpy as np
import pydicom
from pydicom.data import get_testdata_file


from src.paths import PROJECT_ROOT
REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "preprocessing"
    / "tables"
    / "dicom_sample_audit.json"
)

# 이 단일 프레임 CT 샘플의 픽셀 해석에 필요한 항목만 보존
KEEP_KEYWORDS = {
    "SOPClassUID",
    "Modality",
    "Rows",
    "Columns",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "PixelData",
    "RescaleIntercept",
    "RescaleSlope",
    "RescaleType",
    "WindowCenter",
    "WindowWidth",
    "PixelSpacing",
}

CHECK_KEYWORDS = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "AccessionNumber",
    "StudyDate",
    "InstitutionName",
    "ReferringPhysicianName",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
]


def main():
    sample_path = get_testdata_file("CT_small.dcm")

    if sample_path is None:
        raise FileNotFoundError("테스트 샘플을 찾을 수 없습니다.")

    original = pydicom.dcmread(sample_path)
    cleaned = deepcopy(original)

    before_count = len(cleaned)
    private_before = sum(
        element.tag.is_private for element in cleaned.iterall()
    )

    # 허용하지 않은 태그 제거. 시퀀스도 해당 항목 전체를 제거
    for tag in list(cleaned.keys()):
        element = cleaned[tag]

        if element.keyword not in KEEP_KEYWORDS:
            del cleaned[tag]

    # 파일 메타에는 전송 구문과 SOP 클래스만 보존
    keep_file_meta = {
        "TransferSyntaxUID",
        "MediaStorageSOPClassUID",
    }

    for tag in list(cleaned.file_meta.keys()):
        if cleaned.file_meta[tag].keyword not in keep_file_meta:
            del cleaned.file_meta[tag]

    # 파일 선두의 임의 데이터 영역도 제거
    cleaned.preamble = None

    private_after = sum(
        element.tag.is_private for element in cleaned.iterall()
    )

    fields_absent = {
        keyword: keyword not in cleaned
        for keyword in CHECK_KEYWORDS
    }

    pixels_equal = np.array_equal(
        original.pixel_array,
        cleaned.pixel_array,
    )

    only_allowed = all(
        element.keyword in KEEP_KEYWORDS
        for element in cleaned
    )

    if not all(fields_absent.values()):
        raise ValueError("제거 대상 태그가 남아 있습니다.")

    if private_after != 0 or not only_allowed:
        raise ValueError("허용하지 않은 태그가 남아 있습니다.")

    if not pixels_equal:
        raise ValueError("처리 전후 픽셀 값이 달라졌습니다.")

    report = {
        "sample": "pydicom CT_small.dcm",
        "scope": "sample-only metadata removal exercise",
        "original_modified": False,
        "dataset_elements_before": before_count,
        "dataset_elements_after": len(cleaned),
        "private_elements_before": int(private_before),
        "private_elements_after": int(private_after),
        "checked_fields_absent": fields_absent,
        "only_allowed_dataset_fields": only_allowed,
        "pixel_values_unchanged": pixels_equal,
        "dicom_file_saved": False,
        "pixel_identifiers_reviewed": False,
        "full_deidentification_verified": False,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[DICOM 샘플 메타데이터 제거]")
    print(f"최상위 태그 수: {before_count} → {len(cleaned)}")
    print(f"Private 태그 수: {private_before} → {private_after}")
    print(f"확인 대상 식별 태그 모두 제거: {all(fields_absent.values())}")
    print(f"허용된 영상 관련 태그만 남음: {only_allowed}")
    print(f"처리 전후 픽셀 동일: {pixels_equal}")
    print("원본 수정·DICOM 파일 저장: 수행하지 않음")
    print("완전한 비식별화 검증: 미완료")
    print(f"검사 요약 저장: {REPORT_PATH}")


if __name__ == "__main__":
    main()
