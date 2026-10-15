"""프로젝트 경로는 이 파일의 위치를 기준으로 한 곳에서 계산한다."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREPROCESSING_REPORT_DIR = PROJECT_ROOT / "reports" / "preprocessing"
