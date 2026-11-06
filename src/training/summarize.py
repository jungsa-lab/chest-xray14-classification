"""Fold별 result.json을 모아 교차검증 요약표를 만든다.

    python -m src.training.summarize

출력 (기본: reports/training/)
    fold_results.csv       실험·Fold별 최고 검증 성능
    cv_summary.csv         실험별 Mean ± Std (Fold 간 표본 표준편차, Fold 1개면 비어 있음)
    per_disease_auroc.csv  실험·질환별 AUROC Mean ± Std

동작 확인용(_smoke) 결과는 포함하지 않는다.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.constants import DISEASES
from src.paths import PROJECT_ROOT
from src.training.trainer import RESULT_FILE

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "training"

METRICS = ("mean_auroc", "mean_auprc", "mean_ece")
FOLD_COLUMNS = (
    "run_id", "display_name", "fold", "best_epoch", "epochs_run", "stopped_early",
    *METRICS, "n_train", "n_val",
)


def collect_results(output_dir: Path) -> list[dict]:
    # <output_dir>/<run_id>/fold_k/ 만 찾는다. _smoke/<run_id>/fold_k/ 는 한 단계 더 깊어 포함되지 않음
    results = []
    for path in sorted(Path(output_dir).glob(f"*/fold_*/{RESULT_FILE}")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if not result.get("smoke", False):
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
    for (run_id, name), group in folds.groupby(["run_id", "display_name"], sort=True):
        row = {
            "run_id": run_id,
            "display_name": name,
            "n_folds": len(group),
            "folds": " ".join(map(str, group["fold"])),
        }
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


def write_summary(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> pd.DataFrame | None:
    results = collect_results(output_dir)
    if not results:
        print(f"요약할 결과가 없습니다: {output_dir}")
        return None

    folds, summary, per_disease = summarize(results)

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(report_dir / "fold_results.csv", index=False)
    summary.to_csv(report_dir / "cv_summary.csv", index=False)
    per_disease.to_csv(report_dir / "per_disease_auroc.csv", index=False)

    print("\n[교차검증 요약]")
    print(summary[["run_id", "n_folds", *METRICS]].to_string(index=False))
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
