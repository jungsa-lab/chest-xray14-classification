"""Operating Point 분석: Youden's J, Spec@Sens90, Sens@Spec90, 스크리닝 vs 확진.

    python -m src.evaluation.operating_point

두 가지를 구분해서 보고한다.
    (A) Test ROC 지표  : Test ROC 곡선에서 직접 읽은 Spec@Sens90, Sens@Spec90 (임계값과 무관한 모델 성능)
    (B) 임계값 적용 결과: 검증(OOF)에서 정한 임계값을 Test에 그대로 적용했을 때 Sens/Spec/PPV/NPV
                         실제 배포 상황과 같다. 임계값을 Test로 고르면 성능이 낙관적으로 보고된다.

임계값 기준 (OOF 보정 확률에서 질환별로 계산)
    youden  : Sensitivity + Specificity − 1 최대
    sens90  : Sensitivity ≥ 0.90을 만족하는 가장 높은 임계값 (그중 Specificity 최대)
    spec90  : Specificity ≥ 0.90을 만족하는 가장 낮은 임계값 (그중 Sensitivity 최대)

입력: oof/test_predictions_calibrated.csv (calibration.py 결과)
출력: thresholds.json (API 탐지 기준), operating_points.csv, operating_point.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.constants import DISEASES
from src.evaluation.common import (
    EVAL_DIR, METADATA_PATH, labels, load_metadata, load_predictions, md_table, probs, safe_auroc,
)

CRITERIA = ("youden", "sens90", "spec90")
CRITERION_LABELS = {"youden": "Youden's J", "sens90": "Sensitivity 90%", "spec90": "Specificity 90%"}
TARGET = 0.90
EXAMPLE_DISEASE = "Cardiomegaly"


def thresholds_for(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    fpr, tpr, thr = roc_curve(y, p)     # thr 내림차순, tpr·fpr 오름차순
    thr = np.minimum(thr, 1.0)          # 첫 값(inf) → 1.0
    youden = int(np.argmax(tpr - fpr))
    sens90 = int(np.argmax(tpr >= TARGET))                  # tpr ≥ 0.9가 처음 되는 지점 = 가장 높은 임계값
    spec90 = int(np.nonzero(fpr <= 1 - TARGET)[0][-1])      # fpr ≤ 0.1인 마지막 지점 = 가장 낮은 임계값
    return {"youden": float(thr[youden]), "sens90": float(thr[sens90]), "spec90": float(thr[spec90])}


def roc_targets(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """ROC 곡선에서 직접 읽은 Spec@Sens90, Sens@Spec90"""
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "spec_at_sens90": float(1 - fpr[np.argmax(tpr >= TARGET)]),
        "sens_at_spec90": float(tpr[np.nonzero(fpr <= 1 - TARGET)[0][-1]]),
    }


def confusion_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())

    def div(a, b):
        return a / b if b else float("nan")

    return {
        "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
        "flagged_per_1000": 1000 * (tp + fp) / len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def run(eval_dir: Path, metadata: pd.DataFrame) -> dict:
    oof = load_predictions(eval_dir / "oof_predictions_calibrated.csv", metadata)
    test = load_predictions(eval_dir / "test_predictions_calibrated.csv", metadata)
    p_oof, y_oof, p_test, y_test = probs(oof), labels(oof), probs(test), labels(test)

    thresholds = {c: {} for c in CRITERIA}
    rows, roc_rows = [], []
    for i, d in enumerate(DISEASES):
        thr = thresholds_for(y_oof[:, i], p_oof[:, i])
        roc_rows.append({"질환": d, "유병률": y_test[:, i].mean(), "AUROC": safe_auroc(y_test[:, i], p_test[:, i]),
                         **roc_targets(y_test[:, i], p_test[:, i])})
        for c in CRITERIA:
            thresholds[c][d] = thr[c]
            rows.append({"disease": d, "criterion": c, "threshold": thr[c],
                         **confusion_metrics(y_test[:, i], p_test[:, i], thr[c])})
    table = pd.DataFrame(rows)
    roc_table = pd.DataFrame(roc_rows).rename(
        columns={"spec_at_sens90": "Spec@Sens90", "sens_at_spec90": "Sens@Spec90"})

    calib = json.loads((eval_dir / "calibration.json").read_text(encoding="utf-8"))
    (eval_dir / "thresholds.json").write_text(json.dumps({
        "default": "youden",
        "probability": f"보정 확률 (calibration.json: {calib['method']})",
        "fit_on": "OOF (검증) 예측",
        **thresholds,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    table.to_csv(eval_dir / "operating_points.csv", index=False)
    roc_table.to_csv(eval_dir / "roc_operating_targets.csv", index=False)

    (eval_dir / "operating_point.md").write_text(
        markdown(table, roc_table, len(oof), len(test)), encoding="utf-8")
    print(roc_table.to_string(index=False, float_format="%.3f"))
    return thresholds


def _criterion_table(table: pd.DataFrame, disease: str) -> pd.DataFrame:
    sub = table[table["disease"].eq(disease)].set_index("criterion").loc[list(CRITERIA)]
    return pd.DataFrame({
        "기준": [CRITERION_LABELS[c] for c in CRITERIA],
        "Threshold": sub["threshold"].to_numpy(),
        "Sensitivity": sub["sensitivity"].to_numpy(),
        "Specificity": sub["specificity"].to_numpy(),
        "PPV": sub["ppv"].to_numpy(),
        "NPV": sub["npv"].to_numpy(),
        "양성 판정 /1000장": sub["flagged_per_1000"].to_numpy(),
    })


def markdown(table: pd.DataFrame, roc_table: pd.DataFrame, n_oof: int, n_test: int) -> str:
    wide = table.pivot(index="disease", columns="criterion", values=["threshold", "sensitivity", "specificity"])
    summary = pd.DataFrame({
        "질환": DISEASES,
        "Youden 임계값": [wide.loc[d, ("threshold", "youden")] for d in DISEASES],
        "Youden Sens": [wide.loc[d, ("sensitivity", "youden")] for d in DISEASES],
        "Youden Spec": [wide.loc[d, ("specificity", "youden")] for d in DISEASES],
        "Sens90 기준 → Test Sens": [wide.loc[d, ("sensitivity", "sens90")] for d in DISEASES],
        "Spec90 기준 → Test Spec": [wide.loc[d, ("specificity", "spec90")] for d in DISEASES],
    })
    ex = _criterion_table(table, EXAMPLE_DISEASE)
    s90 = table[table["criterion"].eq("sens90")]
    sp90 = table[table["criterion"].eq("spec90")]
    return "\n".join([
        "# Operating Point 분석",
        "",
        f"- 임계값 추정: 검증(OOF) {n_oof:,}장의 보정 확률 / 적용·보고: Test {n_test:,}장",
        "- 확률은 Calibration 보정 후 값이다. 보정은 질환별 단조 변환이라 ROC 곡선 자체는 보정 전과 같다.",
        "",
        "## (A) Test ROC 지표: Spec@Sens90, Sens@Spec90",
        "",
        "Test ROC 곡선에서 직접 읽은 값 (임계값을 고르지 않은 모델 고유 성능).",
        "",
        md_table(roc_table, "{:.3f}"),
        "",
        "## (B) 검증에서 정한 임계값을 Test에 적용",
        "",
        "실제 사용과 같은 조건. Sens90 기준을 적용한 Test Sensitivity가 0.90에서 벗어나는 정도가 "
        "임계값의 이식성(검증 → 새 데이터)을 보여준다.",
        "",
        md_table(summary, "{:.3f}"),
        "",
        f"- Sens90 기준 Test Sensitivity: 평균 {s90['sensitivity'].mean():.3f} "
        f"(최소 {s90['sensitivity'].min():.3f}, {s90.loc[s90['sensitivity'].idxmin(), 'disease']})",
        f"- Spec90 기준 Test Specificity: 평균 {sp90['specificity'].mean():.3f} "
        f"(최소 {sp90['specificity'].min():.3f}, {sp90.loc[sp90['specificity'].idxmin(), 'disease']})",
        "",
        f"## 예시: {EXAMPLE_DISEASE}",
        "",
        md_table(ex, "{:.3f}"),
        "",
        "## Operating Point 선택 근거",
        "",
        "### 1. 스크리닝 용도 (건강검진, 응급실 1차 분류) → **Sensitivity 90% 기준**",
        "",
        "- 목적: 질환이 있는 사람을 놓치지 않는 것. 위음성(FN)은 치료 지연으로 이어지고 되돌리기 어렵다.",
        "- 근거: NPV가 매우 높아(위 표) 'AI 음성'이면 질환이 없을 가능성이 높다 → 판독 우선순위를 낮춰도 안전한 영상을 고를 수 있다.",
        "- 대가: 임계값이 낮아 위양성(FP)이 늘고 PPV가 낮다. 양성 판정된 영상은 전문의가 다시 확인해야 하며, "
        "'양성 판정 /1000장' 열이 그 추가 판독량이다.",
        "",
        "### 2. 확진 보조 용도 (의심 환자 정밀 판독, 추가 검사 결정) → **Specificity 90% 기준**",
        "",
        "- 목적: 'AI 양성'을 근거로 CT·생검 같은 추가 검사를 할 때 불필요한 검사와 환자 불안을 줄이는 것.",
        "- 근거: PPV가 Sens90 기준보다 높다. 양성 판정의 신뢰도가 올라간다.",
        "- 대가: Sensitivity가 떨어져 일부 양성을 놓친다. 그래서 단독 진단이 아니라 전문의 판독의 보조로만 쓴다.",
        "",
        "### 3. 기본값 (API 탐지 결과) → **Youden's J**",
        "",
        "- 용도를 모를 때 Sensitivity와 Specificity를 함께 최대화하는 중립 기준. `thresholds.json`의 `default`.",
        "- 유병률이 낮은 질환(예: Hernia, Pneumonia)은 Specificity가 높아도 PPV가 낮다. "
        "AUROC만으로는 보이지 않는 점이라 PPV를 함께 보고한다.",
        "",
        "> 모든 수치는 NIH 자동 추출 라벨 기준이며, 실제 임상 성능과 다를 수 있다.",
        "",
    ])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Operating Point 분석")
    p.add_argument("--eval-dir", default=str(EVAL_DIR))
    p.add_argument("--metadata", default=str(METADATA_PATH))
    args = p.parse_args(argv)
    run(Path(args.eval_dir), load_metadata(args.metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
