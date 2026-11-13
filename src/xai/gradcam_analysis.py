"""Grad-CAM 에러 분석: False Positive / False Negative 사례, 폐 영역 이탈(Shortcut Learning) 분석 (Test).

    python -m src.xai.gradcam_analysis --output-dir outputs/training --raw-dir data/raw

입력
    reports/evaluation/test_predictions_calibrated.csv  (calibration.py)
    reports/evaluation/thresholds.json                   (operating_point.py, 기본 = Youden 임계값)
    reports/evaluation/selected_runs.json                (tta_ensemble.py, 어떤 DenseNet 실험인지)
    <output-dir>/<DenseNet run_id>/fold_k/best.pt

판정과 설명의 역할 분리
    - FP/FN 판정: 최종 시스템 예측(Ensemble, TTA는 OOF 결정대로, 보정 확률)과 Youden 임계값
    - Grad-CAM : DenseNet-121 Fold 모델 하나 (검증 Mean AUROC가 가장 높은 Fold, API와 같은 모델),
                 원본 방향 이미지로 계산 (TTA 반전 이미지는 위치 해석에 쓰지 않음)

사례 선택 (자동, 재현 가능)
    FP: 정답 음성인데 확률이 임계값 이상 → 확률이 가장 높은 순, 질환마다 1건 (다양한 질환을 보도록)
    FN: 정답 양성인데 확률이 임계값 미만 → 확률이 가장 낮은 순, 질환마다 1건
    폐 영역 분석: 양성 예측(확률 ≥ 임계값) 중 질환별로 번갈아 무작위 표본 (기본 100건)

출력 (reports/evaluation/gradcam/)
    error_cases.csv            FP/FN 사례 + CAM 영역 비율 + 직접 채울 열(원인, 오류 유형)
    region_analysis.csv        표본별 기대 영역(폐 등) 안 CAM 비율, 이탈 여부, 이탈 방향 힌트
    figures/fp_cases.png, fn_cases.png, out_of_region_cases.png
    gradcam_analysis.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.constants import DISEASES
from src.data.image_transforms import ChestXrayTransform
from src.evaluation.common import (
    EVAL_DIR, METADATA_PATH, final_label, load_metadata, load_predictions, md_table,
)
from src.paths import PROJECT_ROOT
from src.training.trainer import BEST_FILE, N_FOLDS, select_device
from src.xai.anatomy import EXPECTED_REGIONS, dominant_outside, region_fractions
from src.xai.gradcam import GradCAM, overlay, peak_location, target_layer

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
CAM_MODEL = "densenet121"
OUT_OF_REGION = 0.5         # 기대 영역 안 CAM 에너지가 절반 미만이면 '이탈'
SEED = 42


# ---------------------------------------------------------------------------
# 사례 선택 (순수 함수, 테스트 대상)
# ---------------------------------------------------------------------------

def long_format(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """(이미지, 질환) 한 줄씩: 확률, 정답, 임계값"""
    parts = []
    for d in DISEASES:
        parts.append(pd.DataFrame({
            "Image Index": df["Image Index"].to_numpy(), "disease": d,
            "prob": df[f"prob_{d}"].to_numpy(), "label": df[d].to_numpy(), "threshold": thresholds[d],
        }))
    return pd.concat(parts, ignore_index=True)


def select_errors(long: pd.DataFrame, kind: str, n: int) -> pd.DataFrame:
    """kind = 'FP' | 'FN'. 질환마다 가장 확신한 오답 1건 → 확신도 순으로 n건"""
    if kind == "FP":
        cand = long[(long["label"] == 0) & (long["prob"] >= long["threshold"])]
        cand = cand.assign(margin=cand["prob"] - cand["threshold"]).sort_values("prob", ascending=False)
    elif kind == "FN":
        cand = long[(long["label"] == 1) & (long["prob"] < long["threshold"])]
        cand = cand.assign(margin=cand["threshold"] - cand["prob"]).sort_values("prob", ascending=True)
    else:
        raise ValueError(kind)
    picked = cand.drop_duplicates("disease").drop_duplicates("Image Index")
    picked = picked.sort_values("margin", ascending=False).head(n)
    return picked.assign(kind=kind).reset_index(drop=True)


def sample_positive_predictions(long: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """양성 예측 중 질환별로 번갈아 뽑은 n건 (흔한 질환에 쏠리지 않도록)"""
    pos = long[long["prob"] >= long["threshold"]].sample(frac=1.0, random_state=seed)
    pos = pos.assign(rank=pos.groupby("disease").cumcount())
    return pos.sort_values(["rank", "disease"]).head(n).drop(columns="rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 모델·이미지
# ---------------------------------------------------------------------------

def pick_cam_fold(output_dir: Path, run_id: str) -> tuple[int, Path]:
    best = None
    for k in range(N_FOLDS):
        path = output_dir / run_id / f"fold_{k}" / BEST_FILE
        auroc = torch.load(path, map_location="cpu", weights_only=True)["val_mean_auroc"]
        if best is None or auroc > best[1]:
            best = (k, auroc, path)
    print(f"Grad-CAM 모델: {run_id} Fold {best[0]} (검증 Mean AUROC {best[1]:.4f}, 5개 Fold 중 최고)")
    return best[0], best[2]


class CaseRenderer:
    def __init__(self, model, model_name, metadata, raw_dir, segmenter):
        self.cam = GradCAM(model, target_layer(model_name, model))
        self.paths = metadata.set_index("Image Index")["Image Path"]
        self.raw_dir = Path(raw_dir)
        self.transform = ChestXrayTransform(use_clahe=True)
        self.segmenter = segmenter
        self._cache: dict[str, tuple] = {}

    def image(self, name: str):
        if name not in self._cache:
            with Image.open(self.raw_dir / self.paths[name]) as im:
                im.load()
                gray = self.transform.gray_from_pil(im, source=name)
            masks = self.segmenter(gray) if self.segmenter else None
            self._cache = {name: (gray, self.transform.tensor_from_gray(gray), masks)}   # 직전 1장만 보관
        return self._cache[name]

    def analyze(self, row) -> dict:
        gray, tensor, masks = self.image(row["Image Index"])
        cam, probs = self.cam(tensor, [DISEASES.index(row["disease"])])
        cam = cam[0]
        x, y = peak_location(cam)
        out = {"cam_model_prob": float(probs[DISEASES.index(row["disease"])]), "peak_x": x, "peak_y": y,
               "_gray": gray, "_cam": cam, "_lung": masks["lung"] if masks else None}
        if masks is not None:
            fr = region_fractions(cam, masks, row["disease"])
            out.update({f"frac_{k}": v for k, v in fr.items()})
            out["out_of_region"] = bool(fr["expected"] < OUT_OF_REGION)
            out["outside_hint"] = dominant_outside(fr)
        return out


def run_cases(renderer: CaseRenderer, cases: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rows, visuals = [], []
    for i, row in cases.iterrows():
        res = renderer.analyze(row)
        visuals.append({k: res.pop(k) for k in ("_gray", "_cam", "_lung")})
        rows.append({**row.to_dict(), **res})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(cases)}", flush=True)
    return pd.DataFrame(rows), visuals


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------

def plot_cases(table: pd.DataFrame, visuals: list[dict], title: str, path: Path, gt_text: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(table)
    if n == 0:
        return
    cols = 2                                    # 사례 2개씩 한 줄 (사례당 원본 | 오버레이)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(4 * cols * 2, 4.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, ((_, row), vis) in enumerate(zip(table.iterrows(), visuals)):
        r, c = divmod(i, cols)
        a0, a1 = axes[r, c * 2], axes[r, c * 2 + 1]
        name = row["Image Index"]
        a0.imshow(vis["_gray"], cmap="gray")
        a0.set_title(f"{row['case']} {name}\nGT: {gt_text[name]}", fontsize=8)
        a1.imshow(overlay(vis["_gray"], vis["_cam"]))
        if vis["_lung"] is not None:
            a1.contour(vis["_lung"], levels=[0.5], colors="white", linewidths=0.8)
        extra = ""
        if not pd.isna(row.get("frac_expected", np.nan)):
            extra = f"\nin-region {row['frac_expected']:.0%} | outside: {row['outside_hint']}"
        a1.set_title(f"{row['disease']} p={row['prob']:.2f} (thr {row['threshold']:.2f}){extra}", fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))     # 전체 제목 자리 확보
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Grad-CAM FP/FN 및 폐 영역 이탈 분석")
    p.add_argument("--eval-dir", default=str(EVAL_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    p.add_argument("--metadata", default=str(METADATA_PATH))
    p.add_argument("--criterion", default="youden", choices=["youden", "sens90", "spec90"])
    p.add_argument("--n-cases", type=int, default=8, help="FP, FN 각각 사례 수 (요구사항 5건 이상)")
    p.add_argument("--n-region", type=int, default=100, help="폐 영역 분석 표본 수")
    p.add_argument("--no-anatomy", action="store_true", help="해부학 분할 없이 CAM만 (영역 비율 계산 안 함)")
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    eval_dir = Path(args.eval_dir)
    out_dir = eval_dir / "gradcam"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata)
    test = load_predictions(eval_dir / "test_predictions_calibrated.csv", metadata)
    thresholds = json.loads((eval_dir / "thresholds.json").read_text(encoding="utf-8"))[args.criterion]
    runs = json.loads((eval_dir / "selected_runs.json").read_text(encoding="utf-8"))["runs"]
    gt_text = {
        name: (", ".join(d for d in DISEASES if row[d]) or "No Finding")
        for name, row in test.set_index("Image Index")[DISEASES].iterrows()
    }

    device = select_device(args.device)
    fold, ckpt_path = pick_cam_fold(Path(args.output_dir), runs[CAM_MODEL]["run_id"])
    from src.evaluation.tta_ensemble import load_fold_model
    model = load_fold_model(ckpt_path, CAM_MODEL, device, torch.contiguous_format)
    segmenter = None
    if not args.no_anatomy:
        from src.xai.anatomy import AnatomySegmenter
        segmenter = AnatomySegmenter(device)
    renderer = CaseRenderer(model, CAM_MODEL, metadata, args.raw_dir, segmenter)

    long = long_format(test, thresholds)
    tables = {}
    for kind in ("FP", "FN"):
        cases = select_errors(long, kind, args.n_cases)
        cases.insert(0, "case", [f"{kind}-{i + 1}" for i in range(len(cases))])
        print(f"[{kind}] {len(cases)}건 Grad-CAM")
        table, visuals = run_cases(renderer, cases)
        plot_cases(table, visuals, f"{'False Positive' if kind == 'FP' else 'False Negative'} cases "
                   f"({final_label(eval_dir)} decision, DenseNet-121 fold {fold} Grad-CAM)",
                   fig_dir / f"{kind.lower()}_cases.png", gt_text)
        tables[kind] = table

    errors = pd.concat(tables.values(), ignore_index=True)
    errors["gt_labels"] = errors["Image Index"].map(gt_text)
    errors["원인 (직접 작성)"] = ""
    errors["오류 유형: 모델/라벨 (직접 작성)"] = ""
    errors.to_csv(out_dir / "error_cases.csv", index=False, float_format="%.4f")

    region = None
    if segmenter is not None:
        sample = sample_positive_predictions(long, args.n_region)
        sample.insert(0, "case", [f"R-{i + 1}" for i in range(len(sample))])
        print(f"[폐 영역 분석] 양성 예측 {len(sample)}건")
        region, visuals = run_cases(renderer, sample)
        region["Shortcut 판정 (직접 작성: 뼈/의료기기/텍스트/배경/정상)"] = ""
        region.to_csv(out_dir / "region_analysis.csv", index=False, float_format="%.4f")
        worst = region.sort_values("frac_expected").head(12)       # 이탈 상위 사례 (요구사항 5건 이상)
        plot_cases(worst, [visuals[i] for i in worst.index],
                   "Lowest in-region Grad-CAM (possible shortcut)", fig_dir / "out_of_region_cases.png", gt_text)

    (out_dir / "gradcam_analysis.md").write_text(
        markdown(errors, region, fold, runs[CAM_MODEL]["run_id"], args, final_label(eval_dir)), encoding="utf-8")
    print(f"저장 위치: {out_dir}")
    return 0


def markdown(errors: pd.DataFrame, region: pd.DataFrame | None, fold: int, run_id: str, args,
             method_label: str) -> str:
    cols = ["case", "Image Index", "disease", "prob", "threshold", "gt_labels"]
    if "frac_expected" in errors:
        cols += ["frac_expected", "outside_hint"]
    lines = [
        "# Grad-CAM 에러 분석 (Test)",
        "",
        f"- 판정: {method_label} 보정 확률, 임계값 = {args.criterion} (검증에서 정함)",
        f"- Grad-CAM: DenseNet-121 `{run_id}` Fold {fold}, 대상 레이어 `features` (마지막 Dense Block 출력, 7×7)",
        "- 영역 비율(frac_expected): CAM 에너지 중 질환별 기대 영역(기본 폐, 경계 7px 팽창) 안 비율. "
        f"{OUT_OF_REGION:.0%} 미만이면 '이탈'",
        "- 기대 영역 예외: " + ", ".join(f"{d} = {'+'.join(r)}" for d, r in EXPECTED_REGIONS.items()),
        "- outside_hint: 기대 영역 밖 에너지가 가장 많이 간 구조 (자동 힌트). 의료기기·텍스트는 분할 대상이 아니라 "
        "`outside_body`나 다른 구조로 잡히므로 그림을 보고 직접 판정한다.",
        "",
        "## False Positive",
        "",
        md_table(errors[errors["kind"].eq("FP")][[c for c in cols if c in errors]], "{:.2f}"),
        "",
        "![FP](figures/fp_cases.png)",
        "",
        "## False Negative",
        "",
        md_table(errors[errors["kind"].eq("FN")][[c for c in cols if c in errors]], "{:.2f}"),
        "",
        "![FN](figures/fn_cases.png)",
        "",
        "> TODO(팀): `error_cases.csv`의 '원인', '오류 유형(모델/라벨)' 열을 그림을 보고 채운다. "
        "분석 본문·방법 한계: [`../discussion.md`](../discussion.md) 3절.",
        "> NIH 라벨은 판독문에서 NLP로 추출해 오류가 있다. GT 목록과 영상을 보고 '라벨 오류 가능' 사례를 구분한다.",
        "",
    ]
    if region is not None:
        n = len(region)
        out = region[region["out_of_region"]]
        hint = out["outside_hint"].value_counts()
        by_disease = region.groupby("disease")["frac_expected"].agg(["count", "mean"]).reset_index()
        by_disease.columns = ["질환", "표본 수", "평균 기대 영역 비율"]
        lines += [
            "## 폐 영역 이탈 분석",
            "",
            f"- 분석: 양성 예측 {n}건 (질환별로 번갈아 무작위 추출)",
            f"- 기대 영역 안 활성화: {n - len(out)}건 ({(n - len(out)) / n:.0%})",
            f"- 이탈: {len(out)}건 ({len(out) / n:.0%})",
        ]
        lines += [f"  - {k}: {v}건" for k, v in hint.items()]
        lines += [
            "",
            md_table(by_disease, "{:.2f}"),
            "",
            "![Out of region](figures/out_of_region_cases.png)",
            "",
            "## Shortcut Learning 판정",
            "",
            "> TODO(팀): 이탈 사례 그림을 보고 `region_analysis.csv`의 'Shortcut 판정' 열을 채운다 "
            "(정리는 [`../discussion.md`](../discussion.md) 3절).",
            "> - 의료기기(심박조율기, 튜브, 카테터)·텍스트/마커 강조 → Shortcut 의심 (질환과 함께 나타나는 표식을 학습)",
            "> - 뼈(쇄골·늑골) 강조 → 겹친 구조의 오인 가능성",
            "> - 개선 방향: 폐 마스킹 입력, 장비·텍스트 영역 가리기 Augmentation, 장비 유무별 성능 비교",
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
