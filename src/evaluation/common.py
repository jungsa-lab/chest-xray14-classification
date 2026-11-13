"""평가 단계(Calibration, Operating Point, Subgroup, Grad-CAM 분석)가 공유하는 입출력.

tta_ensemble.py가 만든 예측 CSV만 읽으므로 GPU·이미지·모델 파일 없이 실행된다.
    reports/evaluation/oof_predictions.csv   검증(OOF) 최종 확률 → 보정·임계값을 '정하는' 데이터
    reports/evaluation/test_predictions.csv  Test 최종 확률      → 정한 값을 '적용·보고하는' 데이터
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.constants import DISEASES
from src.paths import PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "reports" / "evaluation"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "fold_metadata.csv"
EPS = 1e-6


def load_metadata(path: Path | str = METADATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def load_predictions(path: Path | str, metadata: pd.DataFrame) -> pd.DataFrame:
    """예측 CSV(Image Index + 14개 확률)에 정답과 환자 정보를 붙인다.

    반환 열: Image Index, prob_<질환> 14개, <질환> 정답 14개, 환자 정보
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"예측 파일이 없습니다: {path} (먼저 python -m src.evaluation.tta_ensemble 실행)")
    pred = pd.read_csv(path)
    pred = pred.rename(columns={d: f"prob_{d}" for d in DISEASES})
    info_cols = ["Image Index", "Patient ID", "Patient Gender", "Patient Age Years", "Age Valid",
                 "View Position", "Split", *DISEASES]
    merged = pred.merge(metadata[info_cols], on="Image Index", how="left", validate="one_to_one")
    if merged["Patient ID"].isna().any():
        raise ValueError("메타데이터에 없는 이미지가 예측 파일에 있습니다.")
    return merged


def use_tta(eval_dir: Path | str) -> bool:
    """최종 예측에 TTA를 썼는지 (tta_ensemble.py가 OOF로 정해 selected_runs.json에 기록)"""
    path = Path(eval_dir) / "selected_runs.json"
    if not path.exists():
        raise SystemExit(f"{path}가 없습니다 (먼저 python -m src.evaluation.tta_ensemble 실행)")
    tta = json.loads(path.read_text(encoding="utf-8")).get("tta")
    if tta is None:
        raise SystemExit(f"{path}에 TTA 결정이 없습니다 (tta_ensemble.py를 다시 실행)")
    return bool(tta["use"])


def final_label(eval_dir: Path | str) -> str:
    """보고서·그림에 쓰는 최종 예측 방식 이름 (그림 글꼴 문제로 영문)"""
    return "Ensemble + TTA" if use_tta(eval_dir) else "Ensemble"


def probs(df: pd.DataFrame) -> np.ndarray:
    return df[[f"prob_{d}" for d in DISEASES]].to_numpy(dtype=np.float64)


def labels(df: pd.DataFrame) -> np.ndarray:
    return df[DISEASES].to_numpy(dtype=np.int64)


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def to_prob(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def safe_auroc(y: np.ndarray, p: np.ndarray, min_count: int = 1) -> float:
    """양성·음성이 각각 min_count개 이상일 때만 AUROC, 아니면 NaN"""
    n_pos = int(y.sum())
    if n_pos < min_count or len(y) - n_pos < min_count:
        return float("nan")
    return float(roc_auc_score(y, p))


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """tabulate 없이 Markdown 표 만들기"""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])
