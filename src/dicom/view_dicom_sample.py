import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.data import get_testdata_file
from pydicom.pixels import apply_modality_lut


def main():
    sample_path = get_testdata_file("CT_small.dcm")

    if sample_path is None:
        raise FileNotFoundError("테스트 샘플을 찾을 수 없습니다.")

    ds = pydicom.dcmread(sample_path)

    pixels = ds.pixel_array
    values = np.asarray(
        apply_modality_lut(pixels, ds),
        dtype=np.float64,
    )

    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("예상과 다른 픽셀 배열입니다.")

    if ds.PhotometricInterpretation != "MONOCHROME2":
        raise ValueError("이 코드는 MONOCHROME2 샘플 전용입니다.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 전체 값 범위를 사용한 표시
    axes[0].imshow(
        values,
        cmap="gray",
        vmin=float(values.min()),
        vmax=float(values.max()),
        interpolation="nearest",
    )
    axes[0].set_title("Full value range")

    # 밝기 범위를 제한한 비교용 표시
    # 범용 DICOM 뷰어나 임상 판독 설정을 구현한 것은 아님
    axes[1].imshow(
        values,
        cmap="gray",
        vmin=-160,
        vmax=240,
        interpolation="nearest",
    )
    axes[1].set_title("Display range: -160 to 240")

    for ax in axes:
        ax.axis("off")

    fig.suptitle("pydicom CT test sample — local preview")
    fig.tight_layout()

    print("샘플 영상을 로컬 창에 표시합니다.")
    print("영상 안에 이름·환자번호·날짜 등의 문자가 있는지 확인하세요.")
    print("DICOM 및 이미지 파일은 저장하지 않습니다.")
    print("창을 닫으면 실행이 종료됩니다.")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
