"""Subgroup 분석: 성별, 연령대, View Position(PA/AP)별 Mean AUROC (Test).

    python -m src.evaluation.subgroup

정의
    연령대: 0–40 = [0, 40), 40–60 = [40, 60), 60+ = [60, ∞). 나이가 유효하지 않은 이미지(Age Valid=False)는 제외
    질환별 AUROC: 그룹 안에 양성·음성이 각각 MIN_COUNT개 이상일 때만 계산
    Mean AUROC: 같은 속성(예: 성별)의 모든 그룹에서 계산 가능한 질환만 평균 → 그룹 간 같은 질환 집합으로 비교
    95% CI: 환자 단위 Bootstrap (같은 환자 이미지를 함께 뽑음, 이미지 간 상관 반영)
    성능 차이: 상대 차이 = (최고 − 최저) / 최고. 10% 이상이면 원인 분석 대상으로 표시
               (Mean AUROC와 질환별 AUROC 모두 확인)

출력 (reports/evaluation/)
    subgroup_metrics.csv     속성·그룹별 Mean AUROC와 CI
    subgroup_per_disease.csv 속성·그룹·질환별 AUROC, 양성 수
    subgroup_context.csv     그룹 특성 (나이, AP 비율, 유병률 등) — 원인 분석 근거
    subgroup.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import DISEASES
from src.evaluation.common import (
    EVAL_DIR, METADATA_PATH, load_metadata, load_predictions, md_table, safe_auroc,
)

MIN_COUNT = 10
GAP_THRESHOLD = 0.10
N_BOOTSTRAP = 200
SEED = 42
AGE_BINS = [0, 40, 60, np.inf]
AGE_LABELS = ["0-40", "40-60", "60+"]


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Gender"] = df["Patient Gender"].map({"M": "Male", "F": "Female"})
    age = df["Patient Age Years"].where(df["Age Valid"].astype(bool))
    df["Age Group"] = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=False).astype("object")
    df["View"] = df["View Position"].where(df["View Position"].isin(["PA", "AP"]))
    return df


ATTRIBUTES = {"Gender": ["Male", "Female"], "Age Group": AGE_LABELS, "View": ["PA", "AP"]}


def per_disease_auroc(df: pd.DataFrame) -> dict[str, float]:
    return {d: safe_auroc(df[d].to_numpy(), df[f"prob_{d}"].to_numpy(), MIN_COUNT) for d in DISEASES}


def bootstrap_mean_auroc(df: pd.DataFrame, diseases: list[str], rng: np.random.Generator,
                         n: int = N_BOOTSTRAP) -> tuple[float, float]:
    """환자 단위 Bootstrap으로 Mean AUROC 95% CI"""
    index_lists = list(df.groupby("Patient ID").indices.values())
    y = df[diseases].to_numpy()
    p = df[[f"prob_{d}" for d in diseases]].to_numpy()
    values = []
    for _ in range(n):
        picked = rng.integers(0, len(index_lists), size=len(index_lists))
        idx = np.concatenate([index_lists[j] for j in picked])
        aucs = [safe_auroc(y[idx, i], p[idx, i], 1) for i in range(len(diseases))]
        values.append(np.nanmean(aucs))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def relative_gap(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) < 2:
        return float("nan")
    return float((values.max() - values.min()) / values.max())


def run(eval_dir: Path, metadata: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP) -> dict:
    test = add_groups(load_predictions(eval_dir / "test_predictions_calibrated.csv", metadata))
    rng = np.random.default_rng(SEED)

    metric_rows, disease_rows, context_rows = [], [], []
    for attr, groups in ATTRIBUTES.items():
        aucs = {}
        for g in groups:
            sub = test[test[attr].eq(g)]
            aucs[g] = per_disease_auroc(sub)
            for d in DISEASES:
                disease_rows.append({"attribute": attr, "group": g, "disease": d, "auroc": aucs[g][d],
                                     "n_pos": int(sub[d].sum()), "prevalence": float(sub[d].mean())})
            context_rows.append({
                "attribute": attr, "group": g, "n_images": len(sub), "n_patients": sub["Patient ID"].nunique(),
                "mean_age": float(sub.loc[sub["Age Valid"].astype(bool), "Patient Age Years"].mean()),
                "male_ratio": float(sub["Patient Gender"].eq("M").mean()),
                "ap_ratio": float(sub["View Position"].eq("AP").mean()),
                "no_finding_ratio": float((sub[DISEASES].sum(axis=1) == 0).mean()),
                "labels_per_image": float(sub[DISEASES].sum(axis=1).mean()),
            })
        common = [d for d in DISEASES if all(not np.isnan(aucs[g][d]) for g in groups)]
        for g in groups:
            sub = test[test[attr].eq(g)]
            lo, hi = bootstrap_mean_auroc(sub, common, rng, n_bootstrap) if n_bootstrap else (np.nan, np.nan)
            metric_rows.append({"attribute": attr, "group": g, "n_images": len(sub),
                                "n_diseases": len(common), "mean_auroc": float(np.mean([aucs[g][d] for d in common])),
                                "ci_low": lo, "ci_high": hi,
                                "excluded_diseases": ", ".join(d for d in DISEASES if d not in common)})

    metrics = pd.DataFrame(metric_rows)
    per_disease = pd.DataFrame(disease_rows)
    context = pd.DataFrame(context_rows)
    metrics.to_csv(eval_dir / "subgroup_metrics.csv", index=False)
    per_disease.to_csv(eval_dir / "subgroup_per_disease.csv", index=False)
    context.to_csv(eval_dir / "subgroup_context.csv", index=False)

    flags = find_flags(metrics, per_disease)
    (eval_dir / "subgroup.md").write_text(
        markdown(metrics, per_disease, context, flags, len(test)), encoding="utf-8")
    print(metrics.to_string(index=False, float_format="%.4f"))
    print(f"\n10% 이상 차이: {len(flags)}건")
    return {"metrics": metrics, "flags": flags}


def find_flags(metrics: pd.DataFrame, per_disease: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attr, sub in metrics.groupby("attribute", sort=False):
        gap = relative_gap(sub.set_index("group")["mean_auroc"])
        if gap >= GAP_THRESHOLD:
            rows.append({"attribute": attr, "disease": "Mean", "relative_gap": gap})
    for (attr, d), sub in per_disease.groupby(["attribute", "disease"], sort=False):
        s = sub.set_index("group")["auroc"]
        gap = relative_gap(s)
        if gap >= GAP_THRESHOLD:
            pos = sub.set_index("group")["n_pos"]
            rows.append({"attribute": attr, "disease": d, "relative_gap": gap,
                         "best": f"{s.idxmax()} {s.max():.3f} (양성 {pos[s.idxmax()]})",
                         "worst": f"{s.idxmin()} {s.min():.3f} (양성 {pos[s.idxmin()]})"})
    return pd.DataFrame(rows, columns=["attribute", "disease", "relative_gap", "best", "worst"])


def markdown(metrics, per_disease, context, flags, n_test) -> str:
    lines = [
        "# Subgroup Analysis (Test)",
        "",
        f"- 데이터: Test {n_test:,}장, Ensemble + TTA (보정 확률, AUROC는 보정과 무관)",
        f"- 질환별 AUROC는 그룹 안에 양성·음성이 각각 {MIN_COUNT}개 이상일 때만 계산. "
        "Mean AUROC는 같은 속성의 모든 그룹에서 계산 가능한 질환만 평균",
        f"- 95% CI: 환자 단위 Bootstrap {N_BOOTSTRAP}회",
        f"- 성능 차이 = (최고 − 최저) / 최고. {GAP_THRESHOLD:.0%} 이상이면 원인 분석 대상",
        "",
    ]
    names = {"Gender": "성별", "Age Group": "연령대", "View": "View Position"}
    for attr in ATTRIBUTES:
        m = metrics[metrics["attribute"].eq(attr)]
        c = context[context["attribute"].eq(attr)]
        gap = relative_gap(m.set_index("group")["mean_auroc"])
        view = pd.DataFrame({
            "그룹": m["group"].to_numpy(), "이미지 수": m["n_images"].to_numpy(),
            "Mean AUROC": m["mean_auroc"].to_numpy(),
            "95% CI": [f"{lo:.3f}–{hi:.3f}" for lo, hi in zip(m["ci_low"], m["ci_high"])],
            "평균 나이": c["mean_age"].to_numpy(), "남성 비율": c["male_ratio"].to_numpy(),
            "AP 비율": c["ap_ratio"].to_numpy(), "정상(No Finding) 비율": c["no_finding_ratio"].to_numpy(),
        })
        excluded = m["excluded_diseases"].iloc[0]
        lines += [f"## {names[attr]}", "", md_table(view, "{:.3f}"), "",
                  f"- 상대 차이: **{gap:.1%}** {'→ 10% 이상, 원인 분석 필요' if gap >= GAP_THRESHOLD else '(10% 미만)'}"]
        if excluded:
            lines.append(f"- 평균에서 제외한 질환 (양성 수 부족): {excluded}")
        pd_view = per_disease[per_disease["attribute"].eq(attr)].pivot(
            index="disease", columns="group", values="auroc").loc[DISEASES, ATTRIBUTES[attr]]
        pd_view.insert(0, "질환", pd_view.index)
        lines += ["", "<details><summary>질환별 AUROC</summary>", "", md_table(pd_view.reset_index(drop=True), "{:.3f}"),
                  "", "</details>", ""]

    lines += ["## 10% 이상 차이 (원인 분석 대상)", ""]
    if flags.empty:
        lines.append("없음")
    else:
        lines += [md_table(flags.fillna(""), "{:.1%}"), "",
                  "질환별 차이는 양성 수가 적은 그룹에서 우연히 커질 수 있다(괄호 안 양성 수 확인). "
                  "아래 원인 분석은 그룹 특성표(평균 나이, AP 비율, 유병률)를 근거로 작성한다."]
    lines += [
        "",
        "## 원인 분석",
        "",
        "> TODO(팀): 위 표의 그룹 특성을 근거로 작성. 확인할 가설:",
        "> - View Position: AP는 누워서·이동형 장비로 찍는 중환자 영상이 많다(화질 저하, 심장 확대 왜곡, 의료기기). "
        "AP 비율과 나이·동반 질환 수가 함께 다르므로 단일 원인으로 단정하지 않는다.",
        "> - 연령대: 젊은 층은 유병률이 낮아 양성 수가 적고(CI 넓음), 고령층은 여러 질환이 겹쳐 판별이 어렵다.",
        "> - 성별: 유방 음영, 체형 차이. 학습 데이터 성비(남성 약 56%)도 확인.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Subgroup Analysis (성별/연령대/View Position)")
    p.add_argument("--eval-dir", default=str(EVAL_DIR))
    p.add_argument("--metadata", default=str(METADATA_PATH))
    p.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP, help="0이면 CI 생략")
    args = p.parse_args(argv)
    run(Path(args.eval_dir), load_metadata(args.metadata), args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
