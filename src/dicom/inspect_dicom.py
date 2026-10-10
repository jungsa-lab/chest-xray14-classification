import numpy as np
import pydicom
from pydicom.data import get_testdata_file


def main():
    sample_path = get_testdata_file("CT_small.dcm")

    if sample_path is None:
        raise FileNotFoundError("pydicom 테스트 샘플을 찾을 수 없습니다.")

    # DICOM 파일을 읽어 Dataset 객체로 변환
    ds = pydicom.dcmread(sample_path)

    print("[DICOM 샘플]")
    print("출처: pydicom 테스트 데이터")
    print("파일: CT_small.dcm")
    print(f"pydicom 버전: {pydicom.__version__}")

    # 전체 Dataset을 출력하지 않고 기술 정보만 선택
    print("\n[영상 메타데이터]")

    technical_fields = [
        "Modality",
        "Rows",
        "Columns",
        "SamplesPerPixel",
        "PhotometricInterpretation",
        "BitsAllocated",
        "BitsStored",
        "PixelRepresentation",
    ]

    for keyword in technical_fields:
        print(f"{keyword}: {ds.get(keyword, '없음')}")

    print(
        "Transfer Syntax: "
        f"{ds.file_meta.TransferSyntaxUID.name}"
    )

    # 파일에 저장된 픽셀을 NumPy 배열로 해석
    pixels = ds.pixel_array

    print("\n[픽셀 추출]")
    print(f"배열 모양: {pixels.shape}")
    print(f"자료형: {pixels.dtype}")
    print(f"저장 픽셀 최솟값: {pixels.min()}")
    print(f"저장 픽셀 최댓값: {pixels.max()}")

    if pixels.shape != (int(ds.Rows), int(ds.Columns)):
        raise ValueError("메타데이터와 픽셀 배열 크기가 다릅니다.")

    if not np.isfinite(pixels).all():
        raise ValueError("픽셀에 NaN 또는 무한대가 있습니다.")

    # 식별 관련 필드는 실제 값을 출력하지 않음
    print("\n[식별 관련 필드 — 값은 출력하지 않음]")

    identifying_fields = [
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "AccessionNumber",
        "StudyDate",
        "InstitutionName",
        "ReferringPhysicianName",
    ]

    for keyword in identifying_fields:
        if keyword not in ds:
            status = "태그 없음"
        elif str(ds.get(keyword, "")).strip():
            status = "값 있음"
        else:
            status = "태그는 있으나 빈 값"

        print(f"{keyword}: {status}")

    private_count = sum(
        1 for element in ds.iterall()
        if element.tag.is_private
    )

    print(f"\nPrivate 태그 수: {private_count}")
    print("원본 수정·결과 파일 저장: 수행하지 않음")
    print("DICOM 메타데이터·픽셀 파싱: 정상")


if __name__ == "__main__":
    main()
