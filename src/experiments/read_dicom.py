
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.data import get_testdata_file
from pydicom.pixels import apply_modality_lut

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. DICOM 샘플 경로 가져오기
    dicom_path = get_testdata_file("CT_small.dcm")

    if dicom_path is None:
        raise FileNotFoundError("DICOM 샘플을 찾지 못했습니다.")

    # 2. DICOM 파일 읽기
    ds = pydicom.dcmread(dicom_path)

    print("===== DICOM 메타데이터 =====")
    print("Modality:", ds.get("Modality", "Unknown"))
    print("Rows:", ds.Rows)
    print("Columns:", ds.Columns)
    print("Bits Stored:", ds.get("BitsStored"))
    print("Photometric Interpretation:",
          ds.get("PhotometricInterpretation"))

    # 3. 픽셀 데이터를 NumPy 배열로 변환
    pixels = ds.pixel_array

    print("\n===== 픽셀 정보 =====")
    print("Shape:", pixels.shape)
    print("Dtype:", pixels.dtype)
    print("Min:", pixels.min())
    print("Max:", pixels.max())

    # 4. 영상 표현을 위한 픽셀 변환
    pixels = apply_modality_lut(pixels, ds)
    pixels = pixels.astype(np.float32)

    # 5. 화면 표시용 0~255 정규화
    min_value = pixels.min()
    max_value = pixels.max()

    if max_value > min_value:
        normalized = (
            (pixels - min_value)
            / (max_value - min_value)
            * 255
        )
    else:
        normalized = np.zeros_like(pixels)

    normalized = normalized.astype(np.uint8)

    # 6. 영상 저장
    output = OUTPUT_DIR / "dicom_sample.png"

    plt.imsave(
        output,
        normalized,
        cmap="gray",
        vmin=0,
        vmax=255,
    )

    print("\nDICOM 읽기 성공!")
    print("이미지 저장:", output)


if __name__ == "__main__":
    main()
