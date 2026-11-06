"""Fold별 result.json을 모아 교차검증 요약표와 gamma 비교표를 만든다.

    python -m src.training.summarize

출력 (기본: reports/training/)
    fold_results.csv       실험·Fold별 최고 검증 성능
    cv_summary.csv         실험별 Mean ± Std (Fold 간 표본 표준편차, Fold 1개면 비어 있음)
    per_disease_auroc.csv  실험·질환별 AUROC Mean ± Std
    gamma_comparison.csv   gamma 외 설정이 같은 실험끼리 gamma 0/1/2 비교
    gamma_comparison.md    요구사항 형식의 비교표 (해석 열은 직접 작성)
    selected_gamma.json    5-Fold를 모두 마친 gamma 중 최적값과 선택 기준

모든 수치는 검증 데이터 기준이며 Test 데이터는 사용하지 않는다.
동작 확인용(_smoke) 결과는 포함하지 않는다.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.constants import DISEASES
from src.paths import PROJECT_ROOT
from src.training.config import RUNTIME_FIELDS
from src.training.trainer import N_FOLDS, RESULT_FILE

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "training"

METRICS = ("mean_auroc", "mean_auprc", "mean_ece")
FOLD_COLUMNS = (
    "run_id", "display_name", "model", "gamma", "comparison_group",
    "fold", "best_epoch", "epochs_run", "stopped_early",
    *METRICS, "n_train", "n_val",
)

SELECTION_RULE = (
    "5-Fold 검증 Mean AUROC 평균이 가장 높은 gamma. "
    "같으면 Mean AUPRC가 높은 쪽, 그다음 ECE가 낮은 쪽."
)


def comparison_group(config: dict) -> str:
    """gamma만 다르고 나머지 설정이 같은 실험을 묶는 키 (예: densenet121-1a2b3c)"""
    payload = {
        k: v for k, v in sorted(config.items())
        if k not in RUNTIME_FIELDS and k != "gamma"
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:6]
    return f"{config['model']}-{digest}"


def collect_results(output_dir: Path) -> list[dict]:
    # <output_dir>/<run_id>/fold_k/ 만 찾는다. _smoke/<run_id>/fold_k/ 는 한 단계 더 깊어 포함되지 않음
    results = []
    for path in sorted(Path(output_dir).glob(f"*/fold_*/{RESULT_FILE}")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("smoke", False):
            continue
        result["model"] = result["config"]["model"]
        result["gamma"] = result["config"]["gamma"]
        result["comparison_group"] = comparison_group(result["config"])
        results.append(result)
    return results


def _mean_std(series: pd.Series) -> str:
    mean, std = series.mean(), series.std(ddof=1)
    return f"{mean:.4f}" if pd.isna(std) else f"{mean:.4f} ± {std:.4f}"


def summarize(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = (
        pd.DataFrame([{k: r[k] for k in FOLD_COLUMNS} for r in results])
        .sort_values(["run_id", "fold"])
        .reset_index(drop=True)
    )

    rows = []
    keys = ["run_id", "display_name", "model", "gamma", "comparison_group"]
    for key, group in folds.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row["n_folds"] = len(group)
        row["folds"] = " ".join(map(str, group["fold"]))
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
            row[metric] = _mean_std(group[metric])
        rows.append(row)
    summary = pd.DataFrame(rows)

    disease_rows = [
        {"run_id": r["run_id"], "fold": r["fold"], "disease": d,
         "auroc": r["per_disease"][d]["AUROC"]}
        for r in results for d in DISEASES
    ]
    per_disease = (
        pd.DataFrame(disease_rows)
        .groupby(["run_id", "disease"], sort=False)["auroc"]
        .agg(auroc_mean="mean", auroc_std=lambda s: s.std(ddof=1))
        .reset_index()
    )
    return folds, summary, per_disease


def select_gamma(summary: pd.DataFrame) -> dict[str, dict]:
    """비교 그룹마다 5-Fold를 모두 마친 gamma 중 최적값 선택 (2개 이상 있을 때만)"""
    selected = {}
    complete = summary[summary["n_folds"].eq(N_FOLDS)]
    for group_key, group in complete.groupby("comparison_group"):
        if group["gamma"].nunique() < 2:
            continue
        ranked = group.sort_values(
            ["mean_auroc_mean", "mean_auprc_mean", "mean_ece_mean"],
            ascending=[False, False, True],
        )
        best = ranked.iloc[0]
        selected[group_key] = {
            "gamma": float(best["gamma"]),
            "run_id": best["run_id"],
            "mean_auroc": best["mean_auroc"],
            "mean_auprc": best["mean_auprc"],
            "mean_ece": best["mean_ece"],
            "compared_gammas": sorted(float(g) for g in group["gamma"]),
            "criterion": SELECTION_RULE,
            "note": "검증 데이터만 사용. Test 성능은 선택에 사용하지 않음.",
        }
    return selected


def gamma_markdown(summary: pd.DataFrame, selected: dict[str, dict]) -> str:
    """요구사항 형식의 gamma 비교표 (해석 열은 결과를 보고 직접 작성)"""
    lines = ["# Gamma 비교 (5-Fold 검증, Mean ± Std)", ""]
    for group_key, group in summary.sort_values("gamma").groupby("comparison_group", sort=True):
        lines += [f"## {group_key}", ""]
        lines += ["| gamma | Mean AUROC | Mean AUPRC | ECE | Fold 수 | 해석 한 줄 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for _, row in group.iterrows():
            gamma = f"{row['gamma']:g}" + (" (BCE 동일)" if row["gamma"] == 0 else "")
            lines.append(
                f"| {gamma} | {row['mean_auroc']} | {row['mean_auprc']} | {row['mean_ece']} "
                f"| {row['n_folds']} | |"
            )
        lines.append("")
        if group_key in selected:
            lines.append(f"**선택: gamma = {selected[group_key]['gamma']:g}** — {SELECTION_RULE}")
        else:
            lines.append("아직 선택 불가: 5-Fold를 모두 마친 gamma가 2개 이상 필요합니다.")
        lines.append("")
    return "\n".join(lines)


def write_summary(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> pd.DataFrame | None:
    results = collect_results(output_dir)
    if not results:
        print(f"요약할 결과가 없습니다: {output_dir}")
        return None

    folds, summary, per_disease = summarize(results)
    selected = select_gamma(summary)

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(report_dir / "fold_results.csv", index=False)
    summary.to_csv(report_dir / "cv_summary.csv", index=False)
    per_disease.to_csv(report_dir / "per_disease_auroc.csv", index=False)

    gamma_table = summary[["comparison_group", "model", "gamma", "n_folds", *METRICS,
                           *[f"{m}_{s}" for m in METRICS for s in ("mean", "std")]]]
    gamma_table.sort_values(["comparison_group", "gamma"]).to_csv(
        report_dir / "gamma_comparison.csv", index=False
    )
    (report_dir / "gamma_comparison.md").write_text(
        gamma_markdown(summary, selected), encoding="utf-8"
    )
    (report_dir / "selected_gamma.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[교차검증 요약]")
    print(summary[["run_id", "n_folds", *METRICS]].to_string(index=False))
    for group_key, choice in selected.items():
        print(f"\n[{group_key}] 선택 gamma = {choice['gamma']:g} (AUROC {choice['mean_auroc']})")
    print(f"\n저장 위치: {report_dir}")
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="교차검증 결과 요약")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = p.parse_args(argv)
    write_summary(Path(args.output_dir), Path(args.report_dir))


if __name__ == "__main__":
    main()
