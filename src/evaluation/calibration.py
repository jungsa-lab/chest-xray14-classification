"""Calibration 분석: Reliability Diagram, ECE/MCE, Temperature Scaling.

    python -m src.evaluation.calibration

원칙
    - 보정 파라미터는 검증(OOF) 예측으로만 추정하고, Test에는 적용만 한다.
    - 방법 선택도 OOF 안에서 Fold 교차 추정(4개 Fold로 추정 → 남은 Fold로 ECE 측정)으로 한다.

비교하는 방법 (z = logit(p), 질환 d)
    none                    보정 없음
    temperature             z / T                       (T 하나, 14개 질환 공통)
    temperature_per_disease z / T_d                     (질환별 T)
    prior_temperature       (z − log w_d) / T_d         w_d = 학습 데이터 pos_weight = 음성 수 / 양성 수
                            pos_weight 학습은 양성 손실을 w_d배 키워 logit을 약 log w_d만큼 위로 민다.
                            이 이동을 이론값으로 먼저 되돌린 뒤 Temperature만 추정한다 (추가 파라미터 없음).
    platt_per_disease       a_d·z + b_d                 (참고용: 질환별 기울기·절편 모두 추정)

선택 규칙 (결과를 보기 전에 정함)
    Temperature 계열(temperature_per_disease → prior_temperature) 중
    OOF 교차 추정 ECE ≤ 0.10인 첫 방법. 둘 다 넘으면 OOF ECE가 가장 낮은 방법.

출력 (reports/evaluation/)
    calibration.json                 선택한 방법과 파라미터 (API가 같은 보정을 적용)
    calibration_comparison.csv/.md   방법별 OOF·Test ECE / MCE
    test_predictions_calibrated.csv  보정된 Test 확률 (이후 분석의 입력)
    oof_predictions_calibrated.csv   보정된 OOF 확률 (임계값 추정의 입력, Test와 같은 보정 파라미터 적용)
    figures/reliability_diagram.png  질환별 + 전체 Reliability Diagram (보정 전후)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.constants import DISEASES
from src.evaluation.common import (
    EVAL_DIR, METADATA_PATH, labels, load_metadata, load_predictions, md_table, probs,
    to_logit, to_prob,
)

N_BINS = 15                      # 학습 보고서의 ECE와 같은 구간 수
ECE_TARGET = 0.10
METHODS = ("none", "temperature", "temperature_per_disease", "prior_temperature", "platt_per_disease")
PREFERENCE = ("temperature_per_disease", "prior_temperature")
METHOD_LABELS = {
    "none": "보정 없음",
    "temperature": "Temperature (공통 T)",
    "temperature_per_disease": "Temperature (질환별 T)",
    "prior_temperature": "Prior 보정 + 질환별 T",
    "platt_per_disease": "Platt (질환별 a, b, 참고)",
}
PLOT_LABELS = {      # 그림은 한글 폰트가 없는 환경(Kaggle 등)에서도 깨지지 않게 영어로
    "none": "No calibration",
    "temperature": "Temperature (global)",
    "temperature_per_disease": "Temperature (per disease)",
    "prior_temperature": "Prior shift + Temperature",
    "platt_per_disease": "Platt (per disease)",
}


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------

def bin_stats(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """동일 폭 구간별 개수·평균 확률·실제 양성 비율 (metrics._binary_ece와 같은 구간 규칙)"""
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "count": int(m.sum()),
                         "confidence": float(p[m].mean()), "accuracy": float(y[m].mean())})
    return pd.DataFrame(rows)


def ece_mce(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> tuple[float, float]:
    s = bin_stats(y, p, n_bins)
    gap = (s["accuracy"] - s["confidence"]).abs()
    return float((s["count"] / len(y) * gap).sum()), float(gap.max())


def mean_ece_mce(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """질환별 ECE·MCE의 평균 (학습 보고서의 mean_ece와 같은 정의)"""
    vals = np.array([ece_mce(y[:, i], p[:, i]) for i in range(y.shape[1])])
    return float(vals[:, 0].mean()), float(vals[:, 1].mean())


# ---------------------------------------------------------------------------
# 추정
# ---------------------------------------------------------------------------

def _nll(z: np.ndarray, y: np.ndarray) -> float:
    # log(1 + e^z) − y·z, 수치 안정형
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """NLL을 최소화하는 T (log T로 최적화해 T > 0 보장)"""
    res = minimize(lambda t: _nll(z / np.exp(t[0]), y), x0=[0.0], method="L-BFGS-B",
                   bounds=[(-4.0, 4.0)])
    return float(np.exp(res.x[0]))


def fit_platt(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # 기울기 a > 0으로 제한: 보정이 단조 증가여야 순위(AUROC)와 임계값 의미가 유지된다
    res = minimize(lambda w: _nll(w[0] * z + w[1], y), x0=[1.0, 0.0], method="L-BFGS-B",
                   bounds=[(1e-3, None), (None, None)])
    return float(res.x[0]), float(res.x[1])


def train_pos_weight(metadata: pd.DataFrame) -> dict[str, float]:
    """학습에 쓴 데이터(Split=train, Fold≥0) 전체의 pos_weight = 음성 수 / 양성 수"""
    train = metadata[metadata["Split"].eq("train") & metadata["Fold"].ge(0)]
    pos = train[DISEASES].sum()
    return {d: float((len(train) - pos[d]) / pos[d]) for d in DISEASES}


def fit(method: str, p: np.ndarray, y: np.ndarray, pos_weight: dict[str, float]) -> dict:
    """보정 파라미터 추정 → calibration.json에 저장되는 dict"""
    z = to_logit(p)
    params: dict = {"method": method}
    if method == "temperature":
        params["temperature"] = fit_temperature(z.ravel(), y.ravel())
    elif method == "temperature_per_disease":
        params["temperature"] = {d: fit_temperature(z[:, i], y[:, i]) for i, d in enumerate(DISEASES)}
    elif method == "prior_temperature":
        params["log_pos_weight"] = {d: float(np.log(pos_weight[d])) for d in DISEASES}
        params["temperature"] = {
            d: fit_temperature(z[:, i] - params["log_pos_weight"][d], y[:, i])
            for i, d in enumerate(DISEASES)
        }
    elif method == "platt_per_disease":
        params["platt"] = {d: list(fit_platt(z[:, i], y[:, i])) for i, d in enumerate(DISEASES)}
    elif method != "none":
        raise ValueError(f"알 수 없는 보정 방법: {method}")
    return params


def apply(p: np.ndarray, params: dict) -> np.ndarray:
    """확률 (N, 14) → 보정 확률. API에서도 이 함수를 그대로 쓴다."""
    method = params["method"]
    if method == "none":
        return np.asarray(p, dtype=np.float64)
    z = to_logit(np.asarray(p, dtype=np.float64))
    if method == "temperature":
        return to_prob(z / params["temperature"])
    out = np.empty_like(z)
    for i, d in enumerate(DISEASES):
        if method == "temperature_per_disease":
            out[:, i] = z[:, i] / params["temperature"][d]
        elif method == "prior_temperature":
            out[:, i] = (z[:, i] - params["log_pos_weight"][d]) / params["temperature"][d]
        elif method == "platt_per_disease":
            a, b = params["platt"][d]
            out[:, i] = a * z[:, i] + b
        else:
            raise ValueError(f"알 수 없는 보정 방법: {method}")
    return to_prob(out)


def cross_fit(method: str, p: np.ndarray, y: np.ndarray, folds: np.ndarray,
              pos_weight: dict[str, float]) -> np.ndarray:
    """Fold k 예측은 나머지 Fold로 추정한 보정으로 변환 → 과적합 없는 OOF 보정 확률"""
    out = np.empty_like(p, dtype=np.float64)
    for k in np.unique(folds):
        m = folds == k
        out[m] = apply(p[m], fit(method, p[~m], y[~m], pos_weight))
    return out


def choose(oof_ece: dict[str, float]) -> str:
    for method in PREFERENCE:
        if oof_ece[method] <= ECE_TARGET:
            return method
    candidates = [m for m in METHODS if m != "none"]
    return min(candidates, key=lambda m: oof_ece[m])


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------

def plot_reliability(y: np.ndarray, before: np.ndarray, after: np.ndarray,
                     after_label: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    panels = [(d, y[:, i], before[:, i], after[:, i]) for i, d in enumerate(DISEASES)]
    panels.append(("All (14 diseases pooled)", y.ravel(), before.ravel(), after.ravel()))
    for ax, (title, yy, pb, pa) in zip(axes.ravel(), panels):
        ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
        for p_, lab, c in ((pb, "Before", "#d62728"), (pa, after_label, "#1f77b4")):
            s = bin_stats(yy, p_)
            e, _ = ece_mce(yy, p_)
            ax.plot(s["confidence"], s["accuracy"], marker="o", ms=3, c=c, label=f"{lab} (ECE {e:.3f})")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("Mean predicted probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed positive rate")
    fig.suptitle("Reliability Diagram (Test, Ensemble + TTA)", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def run(eval_dir: Path, metadata: pd.DataFrame) -> dict:
    oof = load_predictions(eval_dir / "oof_predictions.csv", metadata)
    test = load_predictions(eval_dir / "test_predictions.csv", metadata)
    p_oof, y_oof, folds = probs(oof), labels(oof), oof["Fold"].to_numpy()
    p_test, y_test = probs(test), labels(test)
    pos_weight = train_pos_weight(metadata)

    rows, oof_cal, test_cal, fitted = [], {}, {}, {}
    for method in METHODS:
        oof_cal[method] = cross_fit(method, p_oof, y_oof, folds, pos_weight)
        fitted[method] = fit(method, p_oof, y_oof, pos_weight)       # Test 적용용: OOF 전체로 추정
        test_cal[method] = apply(p_test, fitted[method])
        o_ece, o_mce = mean_ece_mce(y_oof, oof_cal[method])
        t_ece, t_mce = mean_ece_mce(y_test, test_cal[method])
        rows.append({"method": method, "설명": METHOD_LABELS[method],
                     "OOF ECE": o_ece, "OOF MCE": o_mce, "Test ECE": t_ece, "Test MCE": t_mce})
    table = pd.DataFrame(rows)
    chosen = choose(dict(zip(table["method"], table["OOF ECE"])))

    per_disease = pd.DataFrame({
        "질환": DISEASES,
        "유병률(Test)": y_test.mean(axis=0),
        "평균 예측(보정 전)": p_test.mean(axis=0),
        "평균 예측(보정 후)": test_cal[chosen].mean(axis=0),
        "ECE 보정 전": [ece_mce(y_test[:, i], p_test[:, i])[0] for i in range(len(DISEASES))],
        "ECE 보정 후": [ece_mce(y_test[:, i], test_cal[chosen][:, i])[0] for i in range(len(DISEASES))],
    })
    params = fitted[chosen]
    if "temperature" in params and isinstance(params["temperature"], dict):
        per_disease["T"] = [params["temperature"][d] for d in DISEASES]

    # 저장
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "calibration.json").write_text(json.dumps(
        {**params, "fit_on": "OOF (검증) 예측 전체", "n_bins": N_BINS,
         "selection_rule": "Temperature 계열 중 OOF 교차 추정 ECE ≤ 0.10인 첫 방법, 없으면 OOF ECE 최소"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    table.to_csv(eval_dir / "calibration_comparison.csv", index=False)
    # OOF도 Test와 똑같은 변환(OOF 전체로 추정한 파라미터)을 적용해 저장 → 임계값을 같은 확률 공간에서 정한다.
    # (Fold 교차 추정 값은 방법 선택·ECE 보고에만 사용)
    oof_same = apply(p_oof, params)
    for name, frame, p_cal in (("test", test, test_cal[chosen]), ("oof", oof, oof_same)):
        out = pd.DataFrame(p_cal, columns=DISEASES)
        out.insert(0, "Image Index", frame["Image Index"].to_numpy())
        if name == "oof":
            out.insert(1, "Fold", folds)
        out.to_csv(eval_dir / f"{name}_predictions_calibrated.csv", index=False, float_format="%.6f")
    plot_reliability(y_test, p_test, test_cal[chosen], PLOT_LABELS[chosen],
                     eval_dir / "figures" / "reliability_diagram.png")

    before = table.set_index("method").loc["none"]
    after = table.set_index("method").loc[chosen]
    lines = [
        "# Calibration (Reliability Diagram, ECE, Temperature Scaling)",
        "",
        f"- 데이터: 보정 추정 = 검증(OOF) {len(oof):,}장, 보고 = Test {len(test):,}장",
        f"- ECE: 질환별 Binary ECE({N_BINS}개 동일 폭 구간)의 평균. MCE: 구간 오차 최댓값의 질환 평균",
        "- OOF 값은 Fold 교차 추정(4개 Fold로 보정 추정 → 남은 Fold에 적용)이라 과적합되지 않은 값이다.",
        "",
        "## 방법별 비교",
        "",
        md_table(table.drop(columns="method")),
        "",
        f"## 선택: **{METHOD_LABELS[chosen]}**",
        "",
        "선택 규칙(결과를 보기 전에 정함): Temperature 계열(질환별 T → Prior 보정 + 질환별 T) 중 "
        "OOF ECE ≤ 0.10인 첫 방법, 둘 다 넘으면 OOF ECE가 가장 낮은 방법.",
        "",
        f"| 지표 | 보정 전 | 보정 후 | 목표 |",
        f"| --- | --- | --- | --- |",
        f"| Test ECE | {before['Test ECE']:.4f} | {after['Test ECE']:.4f} | ≤ {ECE_TARGET} "
        f"{'✓' if after['Test ECE'] <= ECE_TARGET else '✗'} |",
        f"| Test MCE | {before['Test MCE']:.4f} | {after['Test MCE']:.4f} | |",
        "",
        "## 질환별 (Test)",
        "",
        md_table(per_disease),
        "",
        "![Reliability Diagram](figures/reliability_diagram.png)",
        "",
        "## 해석 참고",
        "",
        "- 학습 때 pos_weight(음성 수/양성 수)로 양성 손실을 키웠기 때문에, 모델은 logit을 약 log(pos_weight)만큼 "
        "높게 내도록 학습된다. 그래서 보정 전 평균 예측 확률이 실제 유병률보다 크게 높다(위 표 비교).",
        "- 이 오차는 '확률을 0.5 쪽으로 모으거나 펼치는' 스케일 문제가 아니라 '전체를 위로 민' 이동 문제라서, "
        "Temperature 하나로는 다 못 고칠 수 있다. Prior 보정은 이 이동을 이론값(log pos_weight)으로 되돌린다.",
        "- 보정은 질환별 단조 변환이라 AUROC·AUPRC(순위 지표)는 바뀌지 않는다.",
        "- Test 최종 예측은 10개 모델 평균, OOF는 2개 모델 평균이라 확률 분포가 약간 다르다. "
        "Test ECE가 OOF ECE와 다르면 이 차이 때문일 수 있다.",
        "",
    ]
    (eval_dir / "calibration.md").write_text("\n".join(lines), encoding="utf-8")

    print(table.to_string(index=False, float_format="%.4f"))
    print(f"\n선택: {chosen} | Test ECE {before['Test ECE']:.4f} → {after['Test ECE']:.4f}")
    return params


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Calibration: Reliability Diagram, ECE, Temperature Scaling")
    p.add_argument("--eval-dir", default=str(EVAL_DIR))
    p.add_argument("--metadata", default=str(METADATA_PATH))
    args = p.parse_args(argv)
    run(Path(args.eval_dir), load_metadata(args.metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
