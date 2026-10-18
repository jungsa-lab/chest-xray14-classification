from src.paths import RAW_DIR, PROCESSED_DIR, PREPROCESSING_REPORT_DIR

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


REPORT_TABLES_DIR = PREPROCESSING_REPORT_DIR / "tables"
REPORT_FIGURES_DIR = PREPROCESSING_REPORT_DIR / "figures"

BBOX_REFERENCE_SIZE = 1024
TILE_GRID_SIZE = (8, 8)
# 0은 CLAHE 미적용(원본)
CLIP_LIMITS = [0.0, 1.0, 2.0, 3.0, 4.0]
SELECTED_CLIP_LIMIT = 2.0
# 주변 영역 폭 = BBox 긴 변 × RING_RATIO
RING_RATIO = 0.5
METRICS = ["Contrast", "CNR", "Lesion Std", "Lesion Entropy"]


def load_gray(path) -> np.ndarray:
    """모델 입력 변환과 같은 규칙으로 회색조 uint8 배열을 읽는다."""
    with Image.open(path) as image:
        image.load()

        if image.mode == "LA":
            if image.getchannel("A").getextrema() != (255, 255):
                raise ValueError(f"투명 픽셀 포함: {path}")

        return np.array(image.convert("L"), dtype=np.uint8)


def lesion_masks(shape, left, top, right, bottom, ring_ratio=RING_RATIO):
    """BBox 내부와 그 바깥을 둘러싼 주변 영역(ring)의 마스크"""
    height, width = shape
    x0, y0 = int(np.floor(left)), int(np.floor(top))
    x1, y1 = int(np.ceil(right)), int(np.ceil(bottom))
    margin = max(2, int(round(ring_ratio * max(right - left, bottom - top))))

    inside = np.zeros(shape, dtype=bool)
    inside[max(y0, 0):min(y1, height), max(x0, 0):min(x1, width)] = True

    outer = np.zeros(shape, dtype=bool)
    outer[
        max(y0 - margin, 0):min(y1 + margin, height),
        max(x0 - margin, 0):min(x1 + margin, width),
    ] = True

    return inside, outer & ~inside


def entropy(pixels: np.ndarray) -> float:
    """uint8 밝기 분포의 섀넌 엔트로피(bit)"""
    hist = np.bincount(pixels.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def lesion_visibility(pixels, left, top, right, bottom) -> dict:
    """병변(BBox) 영역과 주변 영역의 시인성 지표

    - Contrast: |병변 평균 - 주변 평균| (밝기 단계)
    - CNR: Contrast / 주변 표준편차. 노이즈가 같이 커지면 증가하지 않는다.
    - Lesion Std: 병변 내부 밝기 표준편차(내부 구조의 선명도)
    - Lesion Entropy: 병변 내부 밝기 엔트로피(정보량)
    """
    inside, ring = lesion_masks(pixels.shape, left, top, right, bottom)

    if not inside.any() or not ring.any():
        raise ValueError("병변 또는 주변 영역이 비어 있습니다.")

    lesion = pixels[inside].astype(np.float64)
    background = pixels[ring].astype(np.float64)
    contrast = abs(lesion.mean() - background.mean())

    return {
        "Contrast": contrast,
        "CNR": contrast / (background.std() + 1e-6),
        "Lesion Std": lesion.std(),
        "Lesion Entropy": entropy(pixels[inside]),
    }


def scale_box(x, y, w, h, width, height):
    """NIH 1024 기준 좌표를 현재 영상 크기로 환산하고 영상 경계에 맞춘다."""
    left = float(np.clip(x * width / BBOX_REFERENCE_SIZE, 0, width))
    top = float(np.clip(y * height / BBOX_REFERENCE_SIZE, 0, height))
    right = float(np.clip((x + w) * width / BBOX_REFERENCE_SIZE, 0, width))
    bottom = float(np.clip((y + h) * height / BBOX_REFERENCE_SIZE, 0, height))
    return left, top, right, bottom


def summarize_change(metrics: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """원본(clip=0) 대비 BBox별 변화율의 중앙값과 개선된 BBox 비율"""
    base = metrics.loc[metrics["Clip Limit"].eq(0.0)].set_index("Box ID")
    rows = []

    for clip in [c for c in CLIP_LIMITS if c > 0]:
        current = metrics.loc[metrics["Clip Limit"].eq(clip)].set_index("Box ID")
        change = (current[METRICS] - base[METRICS]) / base[METRICS].replace(0, np.nan) * 100
        improved = current[METRICS] > base[METRICS]
        change[group_columns] = current[group_columns]
        improved[group_columns] = current[group_columns]

        median = change.groupby(group_columns)[METRICS].median()
        share = improved.groupby(group_columns)[METRICS].mean() * 100
        count = change.groupby(group_columns).size()

        for key in median.index:
            row = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
            row["Clip Limit"] = clip
            row["Boxes"] = int(count.loc[key])
            for metric in METRICS:
                row[f"{metric} Median Change (%)"] = median.loc[key, metric]
                row[f"{metric} Improved Boxes (%)"] = share.loc[key, metric]
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    boxes = pd.read_csv(
        RAW_DIR / "BBox_List_2017_Official_NIH.csv",
        header=0,
        usecols=range(6),
        names=["Image Index", "Finding Label", "x", "y", "w", "h"],
    )
    boxes["Image Index"] = boxes["Image Index"].str.strip()
    boxes["Finding Label"] = boxes["Finding Label"].str.strip()
    # BBox 파일만 "Infiltrate"로 표기하므로 프로젝트 질환명과 맞춘다
    boxes["Finding Label"] = boxes["Finding Label"].replace({"Infiltrate": "Infiltration"})

    metadata = pd.read_csv(
        PROCESSED_DIR / "fold_metadata.csv",
        usecols=["Image Index", "Image Path", "Patient ID", "Split", "Fold"],
    )

    # 파라미터 판단에 Test 정보가 섞이지 않도록 품질 통과 Train의 BBox만 사용
    eligible = metadata.loc[metadata["Split"].eq("train") & metadata["Fold"].ge(0)]
    candidates = boxes.merge(eligible, on="Image Index", how="inner", validate="many_to_one")
    candidates = candidates.reset_index(drop=True)
    candidates["Box ID"] = candidates.index

    if candidates.empty:
        raise ValueError("사용 가능한 Train BBox가 없습니다.")

    clahes = {
        clip: cv2.createCLAHE(clipLimit=clip, tileGridSize=TILE_GRID_SIZE)
        for clip in CLIP_LIMITS if clip > 0
    }

    records = []

    for _, row in candidates.iterrows():
        original = load_gray(RAW_DIR / row["Image Path"])
        height, width = original.shape
        left, top, right, bottom = scale_box(row["x"], row["y"], row["w"], row["h"], width, height)

        for clip in CLIP_LIMITS:
            pixels = original if clip == 0 else clahes[clip].apply(original)
            records.append({
                "Box ID": row["Box ID"],
                "Image Index": row["Image Index"],
                "Finding Label": row["Finding Label"],
                "Clip Limit": clip,
                **lesion_visibility(pixels, left, top, right, bottom),
            })

    metrics = pd.DataFrame(records)

    # 전체(Scope=All)와 질환별 요약을 한 표로 저장한다
    # BBox별 원자료는 스크립트 재실행으로 재현되므로 저장하지 않는다
    overall = summarize_change(metrics.assign(Scope="All"), ["Scope"])
    by_label = summarize_change(metrics.assign(Scope=metrics["Finding Label"]), ["Scope"])
    summary = pd.concat([overall, by_label], ignore_index=True)
    selected = by_label.loc[by_label["Clip Limit"].eq(SELECTED_CLIP_LIMIT)].sort_values(
        "Contrast Median Change (%)", ascending=False
    )
    medians = metrics.groupby("Clip Limit")[METRICS].median()

    summary_path = REPORT_TABLES_DIR / "clahe_bbox_metric_summary.csv"
    image_path = REPORT_FIGURES_DIR / "clahe_bbox_metrics.png"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 질환별 clipLimit에 따른 지표 중앙값 변화
    fig, axes = plt.subplots(1, len(METRICS), figsize=(20, 4.5))
    for ax, metric in zip(axes, METRICS):
        for label, group in metrics.groupby("Finding Label"):
            median = group.groupby("Clip Limit")[metric].median()
            ax.plot(median.index, median.to_numpy(), marker="o", label=label)
        ax.set_title(metric)
        ax.set_xlabel("clipLimit (0 = original)")
        ax.set_xticks(CLIP_LIMITS)
    axes[0].set_ylabel("Median over boxes")
    axes[-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(
        f"Lesion visibility vs CLAHE clipLimit (Train BBoxes n={len(candidates):,}, "
        f"tile {TILE_GRID_SIZE[0]}x{TILE_GRID_SIZE[1]})"
    )
    fig.tight_layout()
    fig.savefig(image_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    change_columns = [f"{m} Median Change (%)" for m in METRICS]
    print(f"분석 대상 Train BBox: {len(candidates):,}개 (이미지 {candidates['Image Index'].nunique():,}장)")
    print("\n[clipLimit별 지표 중앙값]")
    print(medians.round(3).to_string())
    print("\n[원본 대비 변화율 중앙값 (%)]")
    print(overall.set_index("Clip Limit")[change_columns].round(1).to_string())
    print(f"\n[질환별 변화율 중앙값 - clipLimit={SELECTED_CLIP_LIMIT}]")
    print(selected.set_index("Scope")[["Boxes", *change_columns]].round(1).to_string())
    print(f"\n전체·질환별 요약 저장: {summary_path}")
    print(f"그래프 저장: {image_path}")


if __name__ == "__main__":
    main()
