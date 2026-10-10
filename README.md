# NIH ChestX-ray14 Classification

NIH ChestX-ray14의 224×224 PNG를 사용해 14개 흉부 질환을 다중 라벨로 분류하는 학부 프로젝트입니다.

## 현재 상태

메타데이터 및 이미지 검사, 환자 단위 분할, Train 5-Fold 배정, 라벨 통계,
CLAHE 비교, 모델 입력 변환·DataLoader 점검, 공개 DICOM 샘플 실습을 수행했습니다.
**모델 학습, API, 대시보드 및 최종 성능 평가는 아직 구현·검증하지 않았습니다.**

| 항목 | 기존 실행 결과 |
|---|---:|
| 전체 이미지 / 환자 | 112,120 / 30,805 |
| Train 이미지 (품질 제외 전 → 후) | 95,396 → 95,394 |
| 고정 Test 이미지 | 16,724 |
| Train / Test 환자 | 26,184 / 4,621 |
| Train/Test 및 Fold 내 환자 중복 | 0 |

환자 그룹을 85:15로 나누며, Train 내부에서 5-Fold GroupKFold를 사용합니다.
모델 선택과 임계값 조정은 Train 내부에서 수행하고 Test는 최종 평가에 사용합니다.
팀원 간 비교에는 동일한 `data/processed/fold_metadata.csv`를 사용합니다.

## 현재 구조

| 경로 | 역할 |
|---|---|
| `src/constants.py` | 고정된 14개 질환 출력 순서 |
| `src/paths.py` | 프로젝트 기준 경로 |
| `src/data/` | Dataset, 이미지 변환 |
| `src/preprocessing/` | 메타데이터·이미지 검사, 품질 표시, 분할 및 Fold 생성 |
| `src/analysis/` | 질환 분포·동시 출현·품질 후보·CLAHE 비교 |
| `src/dicom/` | 공개 DICOM 샘플 실습 |
| `scripts/checks/` | GPU·이미지 변환·DataLoader 점검 |
| `legacy/` | 과거 구현 기록. 현재 파이프라인에서 실행하지 않음 |
| `data/raw/` | 다운로드한 원본, Git 제외 |
| `data/processed/` | 공통 학습 목록과 검사 CSV |
| `reports/preprocessing/` | 전처리 설명·표·그림 |
| `notebooks/` | 기존 탐색 노트북 |

[전처리 결과 및 해석](reports/preprocessing/README.md) ·
[구조 정리 변경점](docs/project_structure.md) ·
[기존 데이터 안내](data/README.md)

## 실행 환경

기존 Windows 환경에서 Python 3.14.7, PyTorch 2.10.0+cu128,
torchvision 0.25.0+cu128, RTX 4060의 CUDA 연산을 확인했습니다.
이는 사용자의 기존 환경 기록이며 이 정리 작업에서 GPU를 다시 검증한 결과는 아닙니다.

이미 작동하는 `.venv`에서는 구조 정리를 위해 패키지를 재설치할 필요가 없습니다.
새 환경에서는 플랫폼에 맞는 PyTorch를 설치한 뒤 필요한 패키지를 설치합니다.
`requirements.txt`는 직접 사용하는 패키지 목록이고,
`requirements-preprocessing-snapshot.txt`는 **PyTorch 설치 전**의 전처리 환경 기록입니다.
전체 학습 환경을 재현하는 잠금 파일로 취급하지 않습니다.
macOS에서는 CUDA를 사용할 수 없으며 현재 점검 코드는 CUDA가 없으면 CPU를 사용합니다.

## 데이터 준비

- `data/raw/Data_Entry_2017.csv`
- `data/raw/images/images-224/*.png` (하위 폴더를 재귀 탐색)
- `data/raw/BBox_List_2017_Official_NIH.csv` (BBox 비교 시)

`Image Path`는 `data/raw/` 기준 상대 경로입니다.
원본 이미지·DICOM 및 모델 가중치를 일반 Git 커밋에 포함하지 않습니다.

## 실행 순서

모든 명령은 **프로젝트 루트**에서 실행합니다. 파일 경로 실행 대신 `-m` 모듈 실행을 사용합니다.

### 기존 결과를 받아 학습 입력만 점검

원본 이미지와 공유된 `fold_metadata.csv`가 있다면 전체 전처리를 다시 실행할 필요가 없습니다.

```powershell
.\.venv\Scripts\python.exe -m scripts.checks.check_torch
.\.venv\Scripts\python.exe -m scripts.checks.check_image_transforms
.\.venv\Scripts\python.exe -m scripts.checks.check_dataloader
```

### 전처리 및 보고서 재생성

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.prepare_training_data
.\.venv\Scripts\python.exe -m src.analysis.summarize_labels
.\.venv\Scripts\python.exe -m src.analysis.plot_cooccurrence
.\.venv\Scripts\python.exe -m src.analysis.review_image_quality
.\.venv\Scripts\python.exe -m src.analysis.compare_clahe
.\.venv\Scripts\python.exe -m src.analysis.compare_clahe_bbox
```

검사 CSV가 없으면 전체 이미지를 읽어 검사합니다. 있으면 검사표를 확인해 재사용합니다.
같은 이름·경로의 이미지 내용 교체는 캐시 확인으로 감지하지 못합니다.
원본을 교체했을 때에는 다음 명령으로 재검사합니다.

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.prepare_training_data --refresh-image-audit
```

기존 환자 분할이나 Fold가 달라지면 중단합니다. 오류를 없애려고 기존 분할 파일을 삭제하지 않습니다.
중간 `split_metadata.csv`, `quality_metadata.csv`는 로컬에 남으며 학습 기준은 `fold_metadata.csv`입니다.
품질 후보 그림·표는 이제 `reports/preprocessing/figures`, `tables`에 생성됩니다.

### DICOM 실습

```powershell
.\.venv\Scripts\python.exe -m src.dicom.inspect_dicom
.\.venv\Scripts\python.exe -m src.dicom.deidentify_dicom_sample
.\.venv\Scripts\python.exe -m src.dicom.view_dicom_sample
```

공개 `CT_small.dcm` 샘플만 사용합니다. 태그 제거 실습은 범용 비식별화나 유효한
의료용 DICOM 출력 구현이 아닙니다. 처리 영상·DICOM 파일은 저장하지 않습니다.

## 최종 제출물에 맞춘 향후 확장

아래는 **구현 예정**입니다. 빈 파일은 미리 만들지 않습니다.

| 경로 | 구현할 기능 |
|---|---|
| `src/training/` | 모델, Focal Loss, 학습, 5-Fold CV |
| `src/inference/` | 공통 모델 로딩, 앙상블, 14개 점수·탐지 결과, Top-1 Grad-CAM |
| `src/evaluation/` | 외부 검증, Domain Shift, 성별/연령 분석, FP/FN 사례, Operating Point 선택 |
| `api/` | FastAPI GET `/health`, POST `/predict`, Swagger UI |
| `dashboard/` | Streamlit 업로드, 14개 점수 차트, Grad-CAM 오버레이, 색상 표시 |
| `configs/` | 학습·추론 설정 |
| `artifacts/` | `.pth` 가중치, 앙상블 구성, 질환 순서·전처리·임계값 정보 |
| `reports/training/`, `reports/evaluation/`, `reports/error_analysis/` | 실험 및 기술 보고서 |
| 루트 `Dockerfile` | 실행 환경 구성 |

모델 가중치는 추후 별도 다운로드 경로·버전·해시와 로딩 방법을 문서화합니다.
학습과 API는 동일한 이미지 변환과 질환 순서를 사용하도록 구성합니다.
