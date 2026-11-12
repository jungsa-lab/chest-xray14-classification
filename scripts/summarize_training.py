"""완료된 run_training_v2 실험 결과를 리포트로 내보내기
    python -m scripts.summarize_training
    python -m scripts.summarize_training --bundle-weights

대상 실험: fold 0..4의 gamma=0 및 fold 0의 gamma=1, 2.
6 에폭(epoch) 학습이 모두 완료된 실행 결과만 포함됨, 중단된 백업본과 
이전 v1 출력물은 무시.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import zipfile
from pathlib import Path

from src.constants import DISEASES
from src.paths import PROJECT_ROOT

EXPECTED_RUNS = [(0, fold) for fold in range(5)] + [(1, 0), (2, 0)]
METRICS = ("mean_auroc", "mean_auprc", "mean_ece")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def mean_std(values: list[float]) -> tuple[float, float | str]:
    if not values:
        raise ValueError("평균을 계산할 값이 없습니다.")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else ""


def load_run(input_dir: Path, gamma: int, fold: int, expected_epochs: int) -> dict:
    folder = input_dir / f"gamma{gamma}" / f"fold{fold}"
    hist_path = folder / "history.csv"
    metrics_path = folder / "best_metrics.json"
    for required in (hist_path, metrics_path):
        if not required.is_file():
            raise FileNotFoundError(f"필수 학습 결과가 없습니다: {required}")

    history = read_csv(hist_path)
    epochs = [int(row["epoch"]) for row in history]
    if epochs != list(range(1, expected_epochs + 1)):
        raise ValueError(
            f"gamma={gamma} Fold {fold}: history.csv가 {expected_epochs} Epoch를 "
            f"정상 완료하지 않았습니다. 기록된 Epoch: {epochs}"
        )

    best = json.loads(metrics_path.read_text(encoding="utf-8"))
    if float(best["gamma"]) != float(gamma) or int(best["fold"]) != fold:
        raise ValueError(f"gamma={gamma} Fold {fold}: best_metrics.json 실험 정보 불일치")

    best_epoch = int(best["epoch"])
    if best_epoch not in epochs:
        raise ValueError(f"gamma={gamma} Fold {fold}: 최적 Epoch가 history에 없습니다")

    # 체크포인트는 검증(Validation) 평균 AUROC가 가장 높은 시점의 것이어야 함.
    best_history = history[best_epoch - 1]
    aurocs = [float(row["mean_auroc"]) for row in history]
    if not math.isclose(float(best["mean_auroc"]), max(aurocs), abs_tol=1e-8):
        raise ValueError(f"gamma={gamma} Fold {fold}: best_metrics의 AUROC가 최고값과 다릅니다")
    for name in METRICS:
        if not math.isclose(float(best[name]), float(best_history[name]), abs_tol=1e-8):
            raise ValueError(f"gamma={gamma} Fold {fold}: {name} 값 불일치")

    per_disease = best.get("per_disease", {})
    if set(per_disease) != set(DISEASES):
        raise ValueError(f"gamma={gamma} Fold {fold}: 질환별 지표가 14개가 아닙니다")
    for disease in DISEASES:
        for key in ("AUROC", "AUPRC", "ECE"):
            if not math.isfinite(float(per_disease[disease][key])):
                raise ValueError(f"gamma={gamma} Fold {fold}: {disease}/{key} 값 오류")

    return {
        "gamma": gamma,
        "fold": fold,
        "folder": folder,
        "history_path": hist_path,
        "metrics_path": metrics_path,
        "history": history,
        "best": best,
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_label(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def export(input_dir: Path, report_dir: Path, expected_epochs: int,
           bundle_weights: bool = False) -> None:
    # 리포트 파일을 생성하기 전에 모든 실행(run)을 먼저 검증함.
    runs = [load_run(input_dir, g, f, expected_epochs) for g, f in EXPECTED_RUNS]
    report_dir.mkdir(parents=True, exist_ok=True)

    # 원본 에폭(epoch)별 원시 기록 및 최고 성능 지표(best-metrics) JSON 파일을 보존함.
    for run in runs:
        tag = f"gamma{run['gamma']}_fold{run['fold']}"
        for source, target in (
            (run["history_path"], report_dir / "history" / f"{tag}.csv"),
            (run["metrics_path"], report_dir / "metrics" / f"{tag}.json"),
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    fold_rows = []
    for run in runs:
        g, f, best = run["gamma"], run["fold"], run["best"]
        fold_rows.append({
            "gamma": g, "fold": f, "best_epoch": int(best["epoch"]),
            "epochs_run": len(run["history"]),
            **{key: float(best[key]) for key in METRICS},
            "history_file": f"history/gamma{g}_fold{f}.csv",
            "metrics_file": f"metrics/gamma{g}_fold{f}.json",
        })
    write_csv(report_dir / "fold_results.csv", [
        "gamma", "fold", "best_epoch", "epochs_run", *METRICS,
        "history_file", "metrics_file",
    ], fold_rows)

    # 평균 ± 표본 표준편차는 gamma=0인 경우에만 5-fold CV로 표기함.
    gamma0 = [r for r in runs if r["gamma"] == 0]
    summary = {"gamma": 0, "n_folds": len(gamma0), "folds": "0 1 2 3 4"}
    for metric in METRICS:
        avg, std = mean_std([float(r["best"][metric]) for r in gamma0])
        summary[f"{metric}_mean"] = avg
        summary[f"{metric}_std"] = std
        summary[metric] = f"{avg:.4f} ± {std:.4f}"
    write_csv(report_dir / "cv_summary.csv", [
        "gamma", "n_folds", "folds",
        *[field for metric in METRICS for field in (
            f"{metric}_mean", f"{metric}_std", metric)],
    ], [summary])

    # fold 0에서만 gamma를 비교함. gamma 1/2에 대한 5-fold 요약이 있다고 표기하지 않음.
    gamma_fold0 = [r for r in runs if r["fold"] == 0]
    comparison = [{
        "comparison_scope": "fold0_only", "gamma": r["gamma"], "fold": 0,
        "best_epoch": int(r["best"]["epoch"]),
        **{metric: float(r["best"][metric]) for metric in METRICS},
    } for r in gamma_fold0]
    write_csv(report_dir / "gamma_comparison.csv", [
        "comparison_scope", "gamma", "fold", "best_epoch", *METRICS,
    ], comparison)

    # 질환(disease) 단위로 결과를 그룹화함. 단일 fold 항목은 표준편차(std) 열을 비워둠.
    disease_rows = []
    for gamma in (0, 1, 2):
        subset = [r for r in runs if r["gamma"] == gamma]
        for disease in DISEASES:
            row = {"gamma": gamma, "disease": disease, "n_folds": len(subset)}
            for metric in ("AUROC", "AUPRC", "ECE"):
                avg, std = mean_std([
                    float(r["best"]["per_disease"][disease][metric])
                    for r in subset
                ])
                row[f"{metric.lower()}_mean"] = avg
                row[f"{metric.lower()}_std"] = std
            disease_rows.append(row)
    write_csv(report_dir / "per_disease_metrics.csv", [
        "gamma", "disease", "n_folds",
        "auroc_mean", "auroc_std", "auprc_mean", "auprc_std", "ece_mean", "ece_std",
    ], disease_rows)

    selected = max(gamma_fold0, key=lambda r: (
        float(r["best"]["mean_auroc"]),
        float(r["best"]["mean_auprc"]),
        -float(r["best"]["mean_ece"]),
    ))
    write_json(report_dir / "selected_gamma.json", {
        "selected_gamma": selected["gamma"],
        "selection_scope": "Fold 0 validation only",
        "selection_rule": "highest Mean AUROC; tie: higher Mean AUPRC; then lower Mean ECE",
        "warning": (
            "Gamma 1 and 2 were evaluated only on Fold 0. "
            "Including Fold 0 in the selected-gamma 5-fold summary may introduce selection bias. "
            "These are validation metrics, not independent test metrics."
        ),
    })

    checkpoints = []
    for run in gamma0:
        f = run["fold"]
        path = run["folder"] / "best.pt"
        if not path.is_file():
            if bundle_weights:
                raise FileNotFoundError(f"가중치 ZIP 생성에 필요한 파일이 없습니다: {path}")
            print(f"[경고] 가중치 파일 없음: {path}")
            continue
        checkpoints.append({
            "gamma": 0, "fold": f,
            "best_epoch": run["best"]["epoch"],
            "checkpoint_path": relative_label(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    write_csv(report_dir / "checkpoint_manifest.csv", [
        "gamma", "fold", "best_epoch", "checkpoint_path", "size_bytes", "sha256",
    ], checkpoints)

    if bundle_weights:
        bundle_path = PROJECT_ROOT / "outputs" / "sharing" / "densenet121_gamma0_best_5fold.zip"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
        try:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
                for run in gamma0:
                    f = run["fold"]
                    z.write(run["folder"] / "best.pt", arcname=f"gamma0/fold{f}/best.pt")
                z.write(report_dir / "checkpoint_manifest.csv", arcname="checkpoint_manifest.csv")
                z.writestr("README.txt", (
                    "DenseNet121 ImageNet pretrained, ChestX-ray14, gamma=0, 5 folds.\n"
                    "Validation-selected checkpoints; not independent test results.\n"
                    "Load with build_densenet121(pretrained=False), then "
                    "checkpoint['model_state_dict'].\n"
                    "Related report: reports/training/README.md\n"
                ))
            tmp.replace(bundle_path)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"[모델 공유용 ZIP] {bundle_path}")
        print("  ZIP은 GitHub에 커밋하지 말고 팀 공유 저장소에 직접 업로드하세요.")

    print(f"[완료] {len(runs)}개 학습 결과를 보고서용 파일로 정리했습니다: {report_dir}")
    print("  Fold 0 gamma 0/1/2 비교, gamma=0의 5-Fold Mean ± Std")
    for metric in METRICS:
        print(f"  {metric}: {summary[metric]}")
    print("  README.md는 덮어쓰지 않았습니다. 생성된 CSV/JSON을 검토한 뒤 커밋하세요.")


def main() -> None:
    p = argparse.ArgumentParser(description="완료된 v2 학습 결과를 reports/training에 내보냅니다.")
    p.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "outputs" / "training")
    p.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports" / "training")
    p.add_argument("--expected-epochs", type=int, default=6)
    p.add_argument("--bundle-weights", action="store_true",
                   help="gamma=0의 5개 best.pt를 outputs/sharing ZIP으로 묶기")
    args = p.parse_args()
    if args.expected_epochs < 1:
        p.error("--expected-epochs는 1 이상이어야 합니다")
    export(args.input_dir, args.report_dir, args.expected_epochs, args.bundle_weights)


if __name__ == "__main__":
    main()
