"""External Validation: 학습에 쓰지 않은 CheXpert에서 성능 측정, NIH Test와 비교.

    python -m src.evaluation.external_validation \
        --chexpert-csv /kaggle/input/chexpert/valid.csv --chexpert-root /kaggle/input/chexpert \
        --output-dir outputs/training

데이터
    CheXpert(Stanford, 2002–2017 단일 기관). Kaggle "CheXpert-v1.0-small"(약 11GB, 저해상도 jpg) 사용 가능.
    - valid.csv: 234장, 방사선 전문의 3명 합의 라벨 (불확실 라벨 없음) — 라벨 품질 높음, 표본 작음
    - train.csv: 22만 장, NLP 자동 라벨 + 불확실(-1) — --split-sample로 환자 단위 무작위 표본 사용
    정면(Frontal) 영상만 사용 (NIH는 정면만 있음).

라벨 매핑 (NIH → CheXpert, 같은 소견만 비교)
    Atelectasis, Cardiomegaly, Consolidation, Edema, Pneumonia, Pneumothorax: 이름 같음
    Effusion → Pleural Effusion
    Mass·Nodule → Lung Lesion (근사: 두 확률 중 큰 값, 정의가 달라 참고용)
    나머지 NIH 질환(Infiltration, Emphysema, Fibrosis, Pleural_Thickening, Hernia)은 CheXpert에 없음.

불확실 라벨(-1) 처리: U-Ignore(해당 질환 평가에서 제외, 기본) / U-Zeros(음성) / U-Ones(양성) 모두 보고.
빈칸은 '언급 없음' = 음성.

예측: Test와 같은 방식 (2종류 × 5 Fold × 원본/반전 평균) + calibration.json 보정.

출력 (reports/evaluation/external/)
    external_predictions.csv, external_metrics.csv, external_validation.md
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.constants import DISEASES
from src.data.image_transforms import ChestXrayTransform
from src.evaluation import calibration
from src.evaluation.common import (
    EVAL_DIR, METADATA_PATH, labels, load_metadata, load_predictions, md_table, probs, safe_auroc,
)
from src.evaluation.metrics import _binary_ece
from src.evaluation.tta_ensemble import (
    DEFAULT_OUTPUT_DIR, MODEL_ORDER, VIEWS, load_fold_model, predict, select_runs,
)
from src.training.trainer import BEST_FILE, N_FOLDS, select_device, sigmoid

MAPPING = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Pleural Effusion",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
}
LESION = ("Lung Lesion", ("Mass", "Nodule"))       # 근사 매핑
POLICIES = ("u_ignore", "u_zeros", "u_ones")
MIN_COUNT = 5


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def prepare_chexpert(csv: Path | str, sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df[df["Frontal/Lateral"].eq("Frontal")].copy()
    df["Patient ID"] = df["Path"].str.extract(r"(patient\d+)", expand=False)
    if sample:
        patients = df["Patient ID"].drop_duplicates().sample(frac=1.0, random_state=seed)
        keep, n = [], 0
        for pid, cnt in df.groupby("Patient ID").size().loc[patients].items():
            if n >= sample:
                break
            keep.append(pid)
            n += cnt
        df = df[df["Patient ID"].isin(keep)]
    return df.reset_index(drop=True)


def target_matrix(df: pd.DataFrame, policy: str) -> tuple[list[str], np.ndarray]:
    """비교 대상 CheXpert 라벨 (N, K): 1/0, 제외는 NaN"""
    names = [*MAPPING.values(), LESION[0]]
    raw = df[names].to_numpy(dtype=np.float64)
    raw = np.nan_to_num(raw, nan=0.0)                       # 빈칸 = 언급 없음 = 음성
    if policy == "u_ignore":
        raw[raw == -1] = np.nan
    elif policy == "u_zeros":
        raw[raw == -1] = 0
    elif policy == "u_ones":
        raw[raw == -1] = 1
    else:
        raise ValueError(policy)
    return names, raw


def mapped_scores(p: np.ndarray) -> np.ndarray:
    """NIH 14개 확률 (N, 14) → 비교 대상 순서 (N, K)"""
    cols = [p[:, DISEASES.index(d)] for d in MAPPING]
    cols.append(np.maximum(*(p[:, DISEASES.index(d)] for d in LESION[1])))
    return np.stack(cols, axis=1)


def resolve_path(root: Path, rel: str) -> Path:
    """CSV의 Path는 'CheXpert-v1.0-small/valid/...' 형태. Kaggle 데이터셋은 첫 폴더가 없을 수 있다."""
    p = root / rel
    if p.exists():
        return p
    return root / Path(*Path(rel).parts[1:])


class ExternalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path):
        self.paths = [resolve_path(root, r) for r in df["Path"]]
        self.transform = ChestXrayTransform(use_clahe=True)
        missing = [p for p in self.paths[:20] if not p.exists()]
        if missing:
            raise SystemExit(f"이미지를 찾지 못했습니다 (예: {missing[0]}). --chexpert-root 확인")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im.load()
            gray = self.transform.gray_from_pil(im, source=self.paths[i])
            size = im.size
        return {"image": self.transform.tensor_from_gray(gray), "label": torch.zeros(len(DISEASES)),
                "image_name": str(self.paths[i]), "width": size[0], "height": size[1],
                "brightness": float(gray.mean())}


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

def evaluate(names: list[str], y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(names):
        m = ~np.isnan(y[:, i])
        yy, ss = y[m, i].astype(int), s[m, i]
        rows.append({"label": name, "n": int(m.sum()), "n_pos": int(yy.sum()),
                     "auroc": safe_auroc(yy, ss, MIN_COUNT),
                     "ece": _binary_ece(yy, ss, 15) if m.any() else np.nan})
    return pd.DataFrame(rows)


def internal_reference(eval_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    test = load_predictions(eval_dir / "test_predictions_calibrated.csv", metadata)
    s = mapped_scores(probs(test))
    y_nih = labels(test)
    y = np.stack([y_nih[:, DISEASES.index(d)] for d in MAPPING]
                 + [np.maximum(*(y_nih[:, DISEASES.index(d)] for d in LESION[1]))], axis=1).astype(float)
    return evaluate([*MAPPING.values(), LESION[0]], y, s), test


def run(args) -> int:
    eval_dir = Path(args.eval_dir)
    out_dir = eval_dir / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata)
    calib = json.loads((eval_dir / "calibration.json").read_text(encoding="utf-8"))

    df = prepare_chexpert(args.chexpert_csv, args.sample)
    print(f"CheXpert 정면 영상: {len(df):,}장, 환자 {df['Patient ID'].nunique():,}명")

    cache = out_dir / "external_predictions.csv"
    if cache.exists() and not args.recompute:
        pred = pd.read_csv(cache)
        if pred["Path"].tolist() != df["Path"].tolist():
            pred = None
    else:
        pred = None

    if pred is None:
        runs = select_runs(Path(args.output_dir), args.runs)
        device = select_device(args.device)
        cuda = device.type == "cuda"
        memory_format = torch.channels_last if cuda else torch.contiguous_format
        models = {f"{runs[m]['run_id']}/fold{k}": load_fold_model(
            Path(args.output_dir) / runs[m]["run_id"] / f"fold_{k}" / BEST_FILE, m, device, memory_format)
            for m in MODEL_ORDER for k in range(N_FOLDS)}
        dataset = ExternalDataset(df, Path(args.chexpert_root))
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=cuda)
        started = time.time()
        logits, _, _ = predict(models, loader, device, cuda, memory_format)
        print(f"예측 완료: {time.time() - started:.0f}s")
        per_model = [np.mean([sigmoid(logits[f"{runs[m]['run_id']}/fold{k}/{v}"])
                              for k in range(N_FOLDS) for v in VIEWS], axis=0) for m in MODEL_ORDER]
        p = calibration.apply(np.mean(per_model, axis=0), calib)
        stats = pd.DataFrame([{k: dataset[i][k] for k in ("width", "height", "brightness")}
                              for i in range(0, len(dataset), max(1, len(dataset) // 500))])
        stats.to_csv(out_dir / "external_image_stats.csv", index=False)
        pred = pd.DataFrame(p, columns=DISEASES)
        pred.insert(0, "Path", df["Path"].to_numpy())
        pred.to_csv(cache, index=False, float_format="%.6f")

    p_ext = pred[DISEASES].to_numpy()
    s_ext = mapped_scores(p_ext)
    internal, test = internal_reference(eval_dir, metadata)

    results = []
    for policy in POLICIES:
        names, y = target_matrix(df, policy)
        r = evaluate(names, y, s_ext).assign(policy=policy)
        results.append(r)
    ext = pd.concat(results, ignore_index=True)
    ext.to_csv(out_dir / "external_metrics.csv", index=False)

    (out_dir / "external_validation.md").write_text(
        markdown(df, ext, internal, test, metadata, p_ext, out_dir, args), encoding="utf-8")
    print(ext[ext["policy"].eq("u_ignore")].to_string(index=False, float_format="%.4f"))
    return 0


def markdown(df, ext, internal, test, metadata, p_ext, out_dir, args) -> str:
    main = ext[ext["policy"].eq("u_ignore")].set_index("label")
    comp = internal.set_index("label")[["n_pos", "auroc"]].rename(columns={"n_pos": "NIH 양성", "auroc": "NIH AUROC"})
    comp["CheXpert 양성"] = main["n_pos"]
    comp["CheXpert AUROC"] = main["auroc"]
    comp["U-Zeros"] = ext[ext["policy"].eq("u_zeros")].set_index("label")["auroc"]
    comp["U-Ones"] = ext[ext["policy"].eq("u_ones")].set_index("label")["auroc"]
    comp["하락 (절대)"] = comp["CheXpert AUROC"] - comp["NIH AUROC"]
    comp["하락 (상대)"] = comp["하락 (절대)"] / comp["NIH AUROC"]
    core = comp.drop(index=LESION[0])
    valid = core.dropna(subset=["CheXpert AUROC", "NIH AUROC"])      # 양쪽 모두 계산된 질환만 비교
    comp.insert(0, "질환", [f"{n} (근사)" if n == LESION[0] else n for n in comp.index])

    # Domain shift 근거표
    names, y = target_matrix(df, "u_ignore")
    nih_y = labels(test)
    shift = pd.DataFrame({
        "질환": list(MAPPING.values()),
        "NIH Test 유병률": [nih_y[:, DISEASES.index(d)].mean() for d in MAPPING],
        "CheXpert 유병률": [np.nanmean(y[:, i]) for i in range(len(MAPPING))],
        "NIH 평균 예측": [probs(test)[:, DISEASES.index(d)].mean() for d in MAPPING],
        "CheXpert 평균 예측": [p_ext[:, DISEASES.index(d)].mean() for d in MAPPING],
    })
    nih_test_meta = metadata[metadata["Split"].eq("test")]
    stats_path = out_dir / "external_image_stats.csv"
    img = pd.read_csv(stats_path) if stats_path.exists() else None
    pop = pd.DataFrame([
        {"항목": "평균 나이", "NIH Test": nih_test_meta.loc[nih_test_meta["Age Valid"].astype(bool), "Patient Age Years"].mean(),
         "CheXpert": df["Age"].mean()},
        {"항목": "남성 비율", "NIH Test": nih_test_meta["Patient Gender"].eq("M").mean(), "CheXpert": df["Sex"].eq("Male").mean()},
        {"항목": "AP 비율", "NIH Test": nih_test_meta["View Position"].eq("AP").mean(), "CheXpert": df["AP/PA"].eq("AP").mean()},
        {"항목": "원본 가로/세로 비", "NIH Test": 1.0,
         "CheXpert": float((img["width"] / img["height"]).mean()) if img is not None else np.nan},
        {"항목": "평균 밝기 (224 흑백, CLAHE 전)", "NIH Test": nih_test_meta["Mean Brightness"].mean(),
         "CheXpert": float(img["brightness"].mean()) if img is not None else np.nan},
    ])

    return "\n".join([
        "# External Validation (CheXpert)",
        "",
        f"- 외부 데이터: `{Path(args.chexpert_csv).name}` 정면 영상 {len(df):,}장, 환자 {df['Patient ID'].nunique():,}명"
        + (f" (환자 단위 무작위 표본 {args.sample:,}장 목표)" if args.sample else ""),
        f"- 내부 기준: NIH Test {len(test):,}장. 모델·전처리·보정 모두 Test와 동일 (재학습·미세조정 없음)",
        "- 비교는 두 데이터셋에 공통으로 있는 질환만. 불확실 라벨 기본 처리 = U-Ignore",
        f"- AUROC는 양성·음성이 각각 {MIN_COUNT}개 이상일 때만 계산",
        "",
        "## Internal vs External",
        "",
        md_table(comp.reset_index(drop=True), "{:.3f}"),
        "",
        f"**Mean AUROC ({len(valid)}개 공통 질환, Lung Lesion 제외)**: NIH {valid['NIH AUROC'].mean():.4f} → "
        f"CheXpert {valid['CheXpert AUROC'].mean():.4f} "
        f"(하락 {valid['CheXpert AUROC'].mean() - valid['NIH AUROC'].mean():+.4f}, "
        f"{(valid['CheXpert AUROC'].mean() - valid['NIH AUROC'].mean()) / valid['NIH AUROC'].mean():+.1%})",
        "",
        "## Domain Shift 근거",
        "",
        "### 환자군·촬영 조건",
        "",
        md_table(pop, "{:.3f}"),
        "",
        "### 유병률과 예측 확률 (보정 후)",
        "",
        "보정은 NIH 검증 데이터로 했으므로, 외부에서 '평균 예측 ≠ 유병률'이면 확률 보정이 이식되지 않은 것이다.",
        "",
        md_table(shift, "{:.3f}"),
        "",
        "## Domain Shift 원인 분석",
        "",
        "> TODO(팀): 위 근거표 수치로 3가지 이상 작성. 확인할 가설:",
        "> 1. **기관·장비**: NIH(NIH Clinical Center) vs CheXpert(Stanford). 장비·촬영 프로토콜·후처리 차이 → 밝기·대비 분포(위 표).",
        "> 2. **라벨링 방식**: 둘 다 판독문 NLP 추출이지만 추출기·규칙이 다르다(CheXpert는 불확실 라벨 분리). "
        "valid.csv는 전문의 합의 라벨이라 '정답' 기준 자체가 다르다. U-Ignore/U-Zeros/U-Ones 차이가 크면 라벨 정의 차이가 큰 질환.",
        "> 3. **환자군**: CheXpert는 입원·응급 환자가 많아 AP 비율이 높고 중증도·의료기기 비율이 높다(위 표 AP 비율).",
        "> 4. **영상 형식**: CheXpert-small은 가로세로 비가 1이 아닌 저해상도 JPEG → 224×224로 늘리며 형태 왜곡, JPEG 압축.",
        "> 5. **유병률 차이**: 유병률이 다르면 AUPRC·PPV와 보정 확률이 달라진다(AUROC는 덜 민감).",
        "",
        "## 대응 전략",
        "",
        "- Fine-tuning: CheXpert 일부(환자 단위 분리)로 마지막 층 또는 전체 미세조정",
        "- Re-calibration: 외부 기관 소량 데이터로 Temperature만 다시 추정 (순위 성능은 유지, 확률만 맞춤)",
        "- 학습 데이터 다양화: NIH + CheXpert(+ PadChest) 공동 학습, 장비·밝기 Augmentation",
        "- 입력 정규화: 종횡비 유지 리사이즈(padding), 동일한 CLAHE·히스토그램 정규화",
        "",
    ])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="External Validation (CheXpert)")
    p.add_argument("--chexpert-csv", required=True, help="CheXpert valid.csv 또는 train.csv")
    p.add_argument("--chexpert-root", required=True, help="CSV의 Path가 가리키는 이미지 폴더의 상위 폴더")
    p.add_argument("--sample", type=int, default=None, help="train.csv 사용 시 환자 단위 표본 이미지 수 (예: 5000)")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--eval-dir", default=str(EVAL_DIR))
    p.add_argument("--metadata", default=str(METADATA_PATH))
    p.add_argument("--runs", nargs="+", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--recompute", action="store_true")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
