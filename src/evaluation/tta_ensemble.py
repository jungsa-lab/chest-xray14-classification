"""Test 데이터 최종 평가: Test-Time Augmentation(H-Flip)과 2-Model Soft Voting Ensemble.

    python -m src.evaluation.tta_ensemble --output-dir outputs/training --raw-dir data/raw

과정
    1. 모델 선택 (검증 성능 기준, Test는 사용하지 않음)
       모델 종류(densenet121, efficientnet_b0)마다 5-Fold를 모두 마친 실험 중
       검증 Mean AUROC가 가장 높은 실험 하나를 고른다. --runs로 직접 지정할 수도 있다.
    2. Test 이미지(Split == test, 학습·검증과 환자가 겹치지 않음)를 한 번씩만 읽고,
       10개 모델(2종류 × 5 Fold)에 원본과 좌우 반전(H-Flip) 이미지를 모두 넣어 logit을 저장한다.
    3. 확률(sigmoid) 단계에서 평균을 낸다.
       - TTA          : (원본 확률 + 반전 확률) / 2
       - Single Model : 한 모델 종류의 Fold별 성능 Mean ± Std, 그리고 5개 Fold 모델 확률 평균
       - Ensemble     : DenseNet-121과 EfficientNet-B0 확률의 단순 평균 (Soft Voting, 가중치 0.5:0.5)
       가중치·TTA 여부를 Test 점수로 고르지 않고 미리 정한 방식 그대로 보고한다.

출력 (--report-dir, 기본 reports/evaluation/)
    test_logits.npz           모델·Fold·시점(원본/반전)별 logit, 정답, 이미지명 (재계산 없이 다시 분석 가능)
    test_metrics.csv          모든 조합의 Mean AUROC / AUPRC / ECE
    test_per_disease_auroc.csv  질환별 AUROC (TTA 전후, 모델별)
    test_predictions.csv      최종(Ensemble + TTA) 질환별 확률
    oof_logits.npz            검증(OOF) logit: Fold k 이미지를 Fold k 모델(2종류 × 원본/반전)로 예측
    oof_predictions.csv       OOF 최종(Ensemble + TTA) 확률 — Temperature·임계값 추정용 (Test 대신)
    tta_comparison.md         TTA 적용 전후 Mean AUROC 비교 + 제한사항
    ensemble_comparison.md    Single Model 대비 Ensemble 향상 폭
    selected_runs.json        평가에 쓴 실험과 선택 기준

OOF 예측
    Temperature Scaling과 Operating Point 임계값은 Test가 아니라 OOF로 정해야 한다.
    각 검증 이미지는 자기를 학습에 쓰지 않은 Fold 모델 2개(DenseNet k, EfficientNet k)로만 예측한다.
    Test 최종 예측은 10개 모델 평균이라 OOF(2개 모델)보다 확률이 약간 덜 극단적일 수 있다 → 보고서에 명시.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.constants import DISEASES
from src.data.chest_xray_dataset import ChestXrayDataset, select_fold
from src.evaluation.metrics import evaluate_multilabel
from src.paths import PROJECT_ROOT
from src.training.model import build_model
from src.training.summarize import collect_results, summarize
from src.training.trainer import BEST_FILE, N_FOLDS, select_device, sigmoid, subsample

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "evaluation"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "fold_metadata.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

MODEL_ORDER = ("densenet121", "efficientnet_b0")       # 앙상블 2개 모델 (앞쪽이 기준 Single Model)
MODEL_LABELS = {"densenet121": "DenseNet-121", "efficientnet_b0": "EfficientNet-B0"}
VIEWS = ("orig", "flip")
METRICS = ("mean_auroc", "mean_auprc", "mean_ece")

# 좌우 위치가 진단 단서인 질환: 심장은 가슴 왼쪽에 치우쳐 있다
LATERALITY_DISEASES = ("Cardiomegaly",)

SELECTION_RULE = (
    "모델 종류마다 5-Fold를 모두 마친 실험 중 검증 Mean AUROC 평균이 가장 높은 실험 "
    "(같으면 AUPRC가 높은 쪽, 그다음 ECE가 낮은 쪽). Test 성능은 선택에 사용하지 않음."
)


# ---------------------------------------------------------------------------
# 1. 모델 선택
# ---------------------------------------------------------------------------

def select_runs(output_dir: Path, run_ids: list[str] | None) -> dict[str, dict]:
    """모델 종류 → {run_id, 검증 성능} (검증 성능 기준)"""
    results = collect_results(output_dir)
    if not results:
        raise SystemExit(f"학습 결과(result.json)가 없습니다: {output_dir}")
    _, summary, _ = summarize(results)

    def has_all_checkpoints(run_id: str) -> bool:
        return all((output_dir / run_id / f"fold_{k}" / BEST_FILE).exists() for k in range(N_FOLDS))

    complete = summary[
        summary["n_folds"].eq(N_FOLDS) & summary["run_id"].map(has_all_checkpoints)
    ]
    if run_ids:
        missing = sorted(set(run_ids) - set(complete["run_id"]))
        if missing:
            raise SystemExit(f"5-Fold best.pt가 모두 있는 실험이 아닙니다: {missing}")
        complete = complete[complete["run_id"].isin(run_ids)]

    selected = {}
    for model in MODEL_ORDER:
        group = complete[complete["model"].eq(model)]
        if group.empty:
            raise SystemExit(f"{model}: 5-Fold를 모두 마친 실험이 없습니다.")
        if run_ids and len(group) > 1:
            raise SystemExit(f"{model} 실험을 하나만 지정하세요: {list(group['run_id'])}")
        best = group.sort_values(
            ["mean_auroc_mean", "mean_auprc_mean", "mean_ece_mean"],
            ascending=[False, False, True],
        ).iloc[0]
        selected[model] = {
            "run_id": best["run_id"],
            "gamma": float(best["gamma"]),
            "val_mean_auroc": best["mean_auroc"],
            "val_mean_auprc": best["mean_auprc"],
            "val_mean_ece": best["mean_ece"],
        }
    return selected


def load_fold_model(path: Path, model_name: str, device: torch.device,
                    memory_format: torch.memory_format) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["model_name"] != model_name:
        raise ValueError(f"모델 종류가 다릅니다: {path} ({ckpt['model_name']} ≠ {model_name})")
    if list(ckpt["diseases"]) != list(DISEASES):
        raise ValueError(f"질환 순서가 다릅니다: {path}")

    model = build_model(model_name, pretrained=False)     # 가중치는 best.pt에서 불러오므로 다운로드 불필요
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device, memory_format=memory_format).eval()


# ---------------------------------------------------------------------------
# 2. 예측
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    models: dict[str, torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    memory_format: torch.memory_format,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """이미지를 한 번 읽어 모든 모델 × (원본, 좌우 반전) logit 계산"""
    outputs: dict[str, list[np.ndarray]] = {
        f"{key}/{view}": [] for key in models for view in VIEWS
    }
    labels, names = [], []
    done, started = 0, time.time()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True, memory_format=memory_format)
        flipped = torch.flip(images, dims=[3])               # (N, C, H, W)의 W축 = 좌우 반전

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            for key, model in models.items():
                outputs[f"{key}/orig"].append(model(images).float().cpu().numpy())
                outputs[f"{key}/flip"].append(model(flipped).float().cpu().numpy())

        labels.append(batch["label"].numpy())
        names.extend(batch["image_name"])
        done += images.size(0)
        if done % 2048 < images.size(0) or done == len(loader.dataset):
            rate = done / (time.time() - started)
            print(f"  {done:,}/{len(loader.dataset):,}장 | {rate:,.0f}장/s", flush=True)

    logits = {k: np.concatenate(v).astype(np.float32) for k, v in outputs.items()}
    return logits, np.concatenate(labels).astype(np.float32), names


# ---------------------------------------------------------------------------
# 3. 평가
# ---------------------------------------------------------------------------

def fold_probs(logits: dict[str, np.ndarray], run_id: str, fold: int, tta: bool) -> np.ndarray:
    p = sigmoid(logits[f"{run_id}/fold{fold}/orig"])
    if tta:
        p = (p + sigmoid(logits[f"{run_id}/fold{fold}/flip"])) / 2
    return p


def build_predictions(
    logits: dict[str, np.ndarray], runs: dict[str, dict],
) -> dict[tuple[str, str, bool], np.ndarray]:
    """(방식, 단위, TTA) → 확률 (N, 14)

    방식: densenet121 | efficientnet_b0 | ensemble
    단위: fold0~fold4 (Fold k 모델 하나, Ensemble은 두 종류의 Fold k 모델 평균)
          5fold (5개 Fold 모델 확률 평균, Ensemble은 10개 모델 = 두 종류 평균의 평균)
    """
    preds = {}
    for tta in (False, True):
        for fold in range(N_FOLDS):
            per_model = [fold_probs(logits, runs[m]["run_id"], fold, tta) for m in MODEL_ORDER]
            for m, p in zip(MODEL_ORDER, per_model):
                preds[(m, f"fold{fold}", tta)] = p
            preds[("ensemble", f"fold{fold}", tta)] = np.mean(per_model, axis=0)
        for m in (*MODEL_ORDER, "ensemble"):
            preds[(m, "5fold", tta)] = np.mean(
                [preds[(m, f"fold{k}", tta)] for k in range(N_FOLDS)], axis=0
            )
    return preds


def oof_final_probs(logits: dict[str, np.ndarray]) -> np.ndarray:
    """OOF 최종 확률: 모델 종류별 (원본 + 반전)/2 를 다시 모델 평균 (Test 최종 예측과 같은 방식)"""
    per_model = [
        (sigmoid(logits[f"{m}/orig"]) + sigmoid(logits[f"{m}/flip"])) / 2 for m in MODEL_ORDER
    ]
    return np.mean(per_model, axis=0)


def predict_oof(
    output_dir: Path, runs: dict[str, dict], metadata: pd.DataFrame, args,
    device: torch.device, amp: bool, memory_format: torch.memory_format,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], np.ndarray]:
    """Fold k 검증 이미지를 Fold k 모델로 예측해 이어 붙인다 (모델 키: densenet121/orig 등)"""
    parts, labels, names, folds = [], [], [], []
    for k in range(N_FOLDS):
        _, val_df = select_fold(metadata, k)
        if args.limit is not None:
            val_df = subsample(val_df, max(1, args.limit // N_FOLDS), seed=42)
        models = {
            m: load_fold_model(output_dir / runs[m]["run_id"] / f"fold_{k}" / BEST_FILE,
                               m, device, memory_format)
            for m in MODEL_ORDER
        }
        print(f"[OOF] Fold {k}: 검증 {len(val_df):,}장", flush=True)
        loader = DataLoader(
            ChestXrayDataset(val_df, args.raw_dir, use_clahe=True),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )
        logits, y, n = predict(models, loader, device, amp, memory_format)
        if n != list(val_df["Image Index"]):
            raise RuntimeError("OOF 예측 순서가 메타데이터와 다릅니다.")
        _check_against_training(output_dir, runs, k, n, logits)
        parts.append(logits)
        labels.append(y)
        names.extend(n)
        folds.append(np.full(len(n), k))
        del models

    merged = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    return merged, np.concatenate(labels), names, np.concatenate(folds)


def _check_against_training(output_dir, runs, fold, names, logits) -> None:
    """학습 때 저장한 val_predictions.csv의 logit과 다시 계산한 원본 logit이 같은지 확인 (모델·전처리 일치 검증)"""
    for m in MODEL_ORDER:
        path = output_dir / runs[m]["run_id"] / f"fold_{fold}" / "val_predictions.csv"
        if not path.exists():
            continue
        saved = pd.read_csv(path).set_index("Image Index").loc[names, [f"logit_{d}" for d in DISEASES]]
        diff = float(np.abs(saved.to_numpy() - logits[f"{m}/orig"]).max())
        status = "일치" if diff < 0.05 else "⚠ 차이 큼 — 모델·전처리가 학습 때와 다른지 확인"
        print(f"  {_label(m)} Fold {fold}: 학습 때 logit과 최대 차이 {diff:.4f} ({status})")


def score_all(preds, y_true) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, disease_rows = [], []
    for (method, unit, tta), prob in preds.items():
        m = evaluate_multilabel(y_true, prob)
        rows.append({"method": method, "unit": unit, "tta": tta,
                     "mean_auroc": m.mean_auroc, "mean_auprc": m.mean_auprc, "mean_ece": m.mean_ece})
        for d in DISEASES:
            disease_rows.append({"method": method, "unit": unit, "tta": tta, "disease": d,
                                 **{k.lower(): v for k, v in m.per_disease[d].items()}})
    return pd.DataFrame(rows), pd.DataFrame(disease_rows)


def _fold_values(metrics: pd.DataFrame, method: str, tta: bool, metric: str) -> np.ndarray:
    rows = metrics[metrics["method"].eq(method) & metrics["tta"].eq(tta)
                   & metrics["unit"].str.match(r"fold\d")].sort_values("unit")
    return rows[metric].to_numpy()


def _value(metrics: pd.DataFrame, method: str, tta: bool, metric: str) -> float:
    rows = metrics[metrics["method"].eq(method) & metrics["tta"].eq(tta) & metrics["unit"].eq("5fold")]
    return float(rows[metric].iloc[0])


def _ms(values: np.ndarray, signed: bool = False) -> str:
    fmt = "{:+.4f}" if signed else "{:.4f}"
    return f"{fmt.format(values.mean())} ± {values.std(ddof=1):.4f}"


def _label(method: str) -> str:
    return MODEL_LABELS.get(method, "2-Model Ensemble")


# ---------------------------------------------------------------------------
# 4. 보고서
# ---------------------------------------------------------------------------

def tta_markdown(metrics, per_disease, runs, n_test) -> str:
    lines = [
        "# Test-Time Augmentation (Horizontal Flip) 전후 비교",
        "",
        f"- 데이터: Test {n_test:,}장 (학습·검증과 환자가 겹치지 않음, 모델 선택·학습에 사용하지 않음)",
        "- TTA: 원본과 좌우 반전 이미지의 예측 확률 평균",
        f"- 모델: " + ", ".join(f"{_label(m)} `{runs[m]['run_id']}`" for m in MODEL_ORDER),
        "",
        "## Mean AUROC",
        "",
        "Fold별 = Fold 모델 하나씩 평가한 5개 값의 Mean ± Std, 5-Fold 평균 = 5개 Fold 모델 확률을 평균낸 예측",
        "",
        "| 모델 | 단위 | TTA 없음 | TTA (H-Flip) | 변화 | 개선된 Fold |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for method in (*MODEL_ORDER, "ensemble"):
        base = _fold_values(metrics, method, False, "mean_auroc")
        tta = _fold_values(metrics, method, True, "mean_auroc")
        lines.append(
            f"| {_label(method)} | Fold별 | {_ms(base)} | {_ms(tta)} | {_ms(tta - base, True)} "
            f"| {(tta > base).sum()}/{len(base)} |"
        )
        b, t = _value(metrics, method, False, "mean_auroc"), _value(metrics, method, True, "mean_auroc")
        lines.append(f"| {_label(method)} | 5-Fold 평균 | {b:.4f} | {t:.4f} | {t - b:+.4f} | |")

    lines += ["", "## 다른 지표 (5-Fold 평균 예측)", "",
              "| 모델 | TTA | Mean AUROC | Mean AUPRC | ECE |", "| --- | --- | --- | --- | --- |"]
    for method in (*MODEL_ORDER, "ensemble"):
        for tta in (False, True):
            vals = [_value(metrics, method, tta, k) for k in METRICS]
            lines.append(f"| {_label(method)} | {'O' if tta else 'X'} | "
                         + " | ".join(f"{v:.4f}" for v in vals) + " |")

    lines += ["", "## 질환별 AUROC 변화 (TTA − 원본, 5-Fold 평균 예측)", "",
              "| 질환 | " + " | ".join(_label(m) for m in (*MODEL_ORDER, "ensemble")) + " |",
              "| --- |" + " --- |" * (len(MODEL_ORDER) + 1)]
    sub = per_disease[per_disease["unit"].eq("5fold")]
    for d in DISEASES:
        cells = []
        for method in (*MODEL_ORDER, "ensemble"):
            rows = sub[sub["method"].eq(method) & sub["disease"].eq(d)].set_index("tta")["auroc"]
            cells.append(f"{rows[True] - rows[False]:+.4f}")
        mark = " (좌우 위치 의존)" if d in LATERALITY_DISEASES else ""
        lines.append(f"| {d}{mark} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 제한사항 (Grad-CAM 해석 시 주의)",
        "",
        "- 흉부 X-ray는 좌우가 대칭이 아니다. 심장은 왼쪽으로 치우쳐 있고(Cardiomegaly), "
        "대동맥궁·위 공기음영은 왼쪽, 간은 오른쪽에 있다. 좌우 반전 이미지는 실제로 존재하지 않는 해부학적 배치(우심증 형태)다.",
        "- 학습 때는 좌우 반전 Augmentation을 쓰지 않았으므로, 반전 이미지는 모델이 본 적 없는 분포다. "
        "TTA는 성능 개선 여부를 확인하는 목적으로만 적용했다.",
        "- 반전 이미지에서 만든 Grad-CAM은 병변 위치가 좌우로 바뀌어 나타난다. "
        "다시 반전해 원래 좌표로 되돌려도, 모델이 '왼쪽 심장 경계'를 보고 판단했는지 '오른쪽'을 봤는지는 원본과 의미가 달라진다.",
        "- 따라서 Grad-CAM 위치 해석은 **원본 이미지 예측에서만** 하고, TTA 평균 확률과 원본 Grad-CAM을 한 그림에 섞어 해석하지 않는다. "
        "좌측 폐 병변·Cardiomegaly처럼 위치가 진단 근거인 질환은 특히 그렇다.",
        "",
    ]
    return "\n".join(lines)


def ensemble_markdown(metrics, runs, n_test) -> str:
    base = MODEL_ORDER[0]
    lines = [
        "# 2-Model Soft Voting Ensemble",
        "",
        f"- 데이터: Test {n_test:,}장",
        f"- 구성: {_label(MODEL_ORDER[0])} + {_label(MODEL_ORDER[1])}, 질환별 확률의 단순 평균 (가중치 0.5 : 0.5, Test로 조정하지 않음)",
        "- 두 모델 모두 H-Flip TTA 적용 확률 사용 (TTA 없는 결과는 아래 표에 함께 표시)",
        "",
        "## 사용한 모델 (검증 성능으로 선택)",
        "",
        "| 모델 | 실험 | gamma | 검증 Mean AUROC |",
        "| --- | --- | --- | --- |",
    ]
    for m in MODEL_ORDER:
        r = runs[m]
        lines.append(f"| {_label(m)} | `{r['run_id']}` | {r['gamma']:g} | {r['val_mean_auroc']} |")

    for tta in (True, False):
        title = "TTA 적용" if tta else "TTA 없음"
        lines += ["", f"## Fold별 비교 ({title}, Mean ± Std)", "",
                  "Fold k의 Ensemble = DenseNet Fold k 모델 + EfficientNet Fold k 모델 (같은 학습 데이터로 학습한 두 모델)", "",
                  "| 방식 | Mean AUROC | Mean AUPRC | ECE |", "| --- | --- | --- | --- |"]
        for method in (*MODEL_ORDER, "ensemble"):
            lines.append(f"| {_label(method)} | "
                         + " | ".join(_ms(_fold_values(metrics, method, tta, k)) for k in METRICS) + " |")
        ens = {k: _fold_values(metrics, "ensemble", tta, k) for k in METRICS}
        for m in MODEL_ORDER:
            diff = {k: ens[k] - _fold_values(metrics, m, tta, k) for k in METRICS}
            better = (diff["mean_auroc"] > 0).sum()
            lines.append(f"| 향상 폭 (vs {_label(m)}) | {_ms(diff['mean_auroc'], True)} "
                         f"| {_ms(diff['mean_auprc'], True)} | {_ms(diff['mean_ece'], True)} |")
            lines.append(f"| └ AUROC가 오른 Fold | {better}/{N_FOLDS} | | |")

        lines += ["", f"## 5-Fold 평균 예측 ({title})", "",
                  "Single Model = 한 종류의 5개 Fold 모델 확률 평균, Ensemble = 두 종류 평균의 평균 (모델 10개)", "",
                  "| 방식 | Mean AUROC | Mean AUPRC | ECE | AUROC 향상 (vs DenseNet-121) |",
                  "| --- | --- | --- | --- | --- |"]
        ref = _value(metrics, base, tta, "mean_auroc")
        for method in (*MODEL_ORDER, "ensemble"):
            vals = [_value(metrics, method, tta, k) for k in METRICS]
            gain = "" if method == base else f"{vals[0] - ref:+.4f}"
            lines.append(f"| {_label(method)} | " + " | ".join(f"{v:.4f}" for v in vals) + f" | {gain} |")

    lines += ["", "ECE는 낮을수록 좋음. 향상 폭의 ECE가 음수면 Calibration이 좋아진 것.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test 평가: H-Flip TTA + 2-Model Soft Voting Ensemble")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="학습 결과 폴더 (<run_id>/fold_k/best.pt)")
    p.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    p.add_argument("--metadata", default=str(DEFAULT_METADATA))
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    p.add_argument("--runs", nargs="+", default=None,
                   help="평가할 실험 run_id (모델 종류마다 하나, 생략하면 검증 성능으로 자동 선택)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true", help="CUDA에서도 float32로 추론")
    p.add_argument("--limit", type=int, default=None, help="동작 확인용 Test 이미지 수")
    p.add_argument("--recompute", action="store_true", help="저장된 test_logits.npz가 있어도 다시 예측")
    p.add_argument("--skip-oof", action="store_true", help="검증(OOF) 예측 생략 (Calibration·임계값 분석 불가)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    runs = select_runs(output_dir, args.runs)
    print("[평가할 모델] " + SELECTION_RULE)
    for m in MODEL_ORDER:
        print(f"  {_label(m)}: {runs[m]['run_id']} (검증 AUROC {runs[m]['val_mean_auroc']})")

    metadata = pd.read_csv(args.metadata, low_memory=False)
    test_df = metadata[metadata["Split"].eq("test")]
    if args.limit is not None:
        test_df = subsample(test_df, args.limit, seed=42)
    print(f"Test 이미지: {len(test_df):,}장{' (동작 확인)' if args.limit else ''}")

    cache = report_dir / "test_logits.npz"
    expected = {f"{runs[m]['run_id']}/fold{k}/{v}" for m in MODEL_ORDER for k in range(N_FOLDS) for v in VIEWS}
    logits = None
    if cache.exists() and not args.recompute:
        data = np.load(cache, allow_pickle=False)
        if (expected <= set(data.files)
                and list(data["image_names"]) == list(test_df["Image Index"])):
            print(f"저장된 예측 사용: {cache} (다시 예측하려면 --recompute)")
            logits = {k: data[k] for k in expected}
            y_true, names = data["labels"], list(data["image_names"])

    device = select_device(args.device)
    cuda = device.type == "cuda"
    amp = cuda and not args.no_amp
    memory_format = torch.channels_last if cuda else torch.contiguous_format
    if cuda:
        torch.backends.cudnn.benchmark = True

    if logits is None:
        models = {}
        for m in MODEL_ORDER:
            for k in range(N_FOLDS):
                path = output_dir / runs[m]["run_id"] / f"fold_{k}" / BEST_FILE
                models[f"{runs[m]['run_id']}/fold{k}"] = load_fold_model(path, m, device, memory_format)
        print(f"모델 {len(models)}개 로드 | device={device} amp={amp}")

        loader = DataLoader(
            ChestXrayDataset(test_df, args.raw_dir, use_clahe=True),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=cuda,
        )
        started = time.time()
        logits, y_true, names = predict(models, loader, device, amp, memory_format)
        print(f"예측 완료: {time.time() - started:.0f}s")
        if names != list(test_df["Image Index"]):
            raise RuntimeError("예측 순서가 메타데이터와 다릅니다.")
        np.savez_compressed(cache, labels=y_true, image_names=np.array(names), **logits)
        del models

    if not args.skip_oof:
        oof_cache = report_dir / "oof_logits.npz"
        oof = None
        if oof_cache.exists() and not args.recompute:
            data = np.load(oof_cache, allow_pickle=False)
            if (data["run_ids"].tolist() == [runs[m]["run_id"] for m in MODEL_ORDER]
                    and bool(data["limited"]) == (args.limit is not None)):
                print(f"저장된 OOF 예측 사용: {oof_cache}")
                oof = ({f"{m}/{v}": data[f"{m}/{v}"] for m in MODEL_ORDER for v in VIEWS},
                       data["image_names"].tolist(), data["folds"])
        if oof is None:
            started = time.time()
            oof_logits, oof_y, oof_names, oof_folds = predict_oof(
                output_dir, runs, metadata, args, device, amp, memory_format)
            print(f"OOF 예측 완료: {time.time() - started:.0f}s")
            np.savez_compressed(
                oof_cache, labels=oof_y, image_names=np.array(oof_names), folds=oof_folds,
                run_ids=np.array([runs[m]["run_id"] for m in MODEL_ORDER]),
                limited=np.array(args.limit is not None), **oof_logits)
            oof = (oof_logits, oof_names, oof_folds)
        oof_logits, oof_names, oof_folds = oof
        table = pd.DataFrame(oof_final_probs(oof_logits), columns=DISEASES)
        table.insert(0, "Fold", oof_folds)
        table.insert(0, "Image Index", oof_names)
        table.to_csv(report_dir / "oof_predictions.csv", index=False, float_format="%.6f")

    preds = build_predictions(logits, runs)
    metrics, per_disease = score_all(preds, y_true)

    metrics.to_csv(report_dir / "test_metrics.csv", index=False)
    per_disease.to_csv(report_dir / "test_per_disease_auroc.csv", index=False)
    final = pd.DataFrame(preds[("ensemble", "5fold", True)], columns=DISEASES)
    final.insert(0, "Image Index", names)
    final.to_csv(report_dir / "test_predictions.csv", index=False, float_format="%.6f")

    n_test = len(y_true)
    (report_dir / "tta_comparison.md").write_text(
        tta_markdown(metrics, per_disease, runs, n_test), encoding="utf-8")
    (report_dir / "ensemble_comparison.md").write_text(
        ensemble_markdown(metrics, runs, n_test), encoding="utf-8")
    (report_dir / "selected_runs.json").write_text(json.dumps(
        {"criterion": SELECTION_RULE, "n_test": n_test, "limit": args.limit, "runs": runs},
        ensure_ascii=False, indent=2), encoding="utf-8")

    view = metrics[metrics["unit"].eq("5fold")].copy()
    view["method"] = view["method"].map(_label)
    print("\n[Test 결과: 5-Fold 평균 예측]")
    print(view[["method", "tta", *METRICS]].to_string(index=False, float_format="%.4f"))
    print(f"\n저장 위치: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
