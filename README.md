# NIH ChestX-ray14 Classification

NIH ChestX-ray14 데이터셋을 활용하여 흉부 X-ray 이미지에서 14개 질환을 분류하는 프로젝트입니다.

하나의 이미지에 여러 질환이 포함될 수 있으므로 다중 라벨 분류(Multi-label Classification) 방식을 사용합니다.

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 데이터셋 | NIH ChestX-ray14 |
| 전체 이미지 | 112,120장 |
| 전체 환자 | 30,805명 |
| 이미지 크기 | 224 × 224 |
| 분류 대상 | 14개 흉부 질환 |
| 개발 언어 | Python |
| 주요 라이브러리 | PyTorch, scikit-learn, Pandas, NumPy |

데이터는 환자 ID를 기준으로 분할하여 동일한 환자의 이미지가 학습과 평가에 중복 사용되지 않도록 구성합니다.

## 2. 프로젝트 구조

| 경로 | 역할 |
|---|---|
| `src/` | 데이터 전처리, 분석, 모델 학습 및 평가 |
| `data/` | 원본 데이터 및 전처리 결과 |
| `scripts/` | 실행 환경 및 데이터 입력 검사 |
| `tests/` | 코드 단위 테스트 |
| `reports/` | 데이터 분석 및 실험 결과 |
| `notebooks/` | 데이터 탐색 및 실험용 노트북 |
| `legacy/` | 이전 구현 코드 보관 |

소스 코드의 세부 구성과 파일별 역할은 [src 디렉터리 구조](src/README.md)에서 확인할 수 있습니다.

## 3. 데이터 준비

원본 데이터셋은 저장소에 포함되어 있지 않으며, 별도로 다운로드하여 `data/raw/`에 배치해야 합니다.

기본 데이터 구조는 다음과 같습니다.

```text
data/
├── raw/
│   ├── Data_Entry_2017.csv
│   └── images/
│       └── images-224/
│           └── *.png
└── processed/
    └── fold_metadata.csv
```

- `Data_Entry_2017.csv`: 이미지별 질환 라벨 및 환자 정보
- `images-224/`: 흉부 X-ray 이미지
- `fold_metadata.csv`: 전처리 결과와 학습·검증 데이터 구분 정보

이미지는 `images-224/` 하위 디렉터리에 나누어 저장되어 있어도 됩니다.

일부 분석 작업에는 별도의 BBox 데이터가 필요할 수 있습니다.

원본 데이터와 모델 가중치는 Git 저장소에 포함하지 않습니다.

## 4. 실행 방법

모든 명령어는 프로젝트 루트 디렉터리에서 실행합니다.

### 의존성 설치

Python 가상환경을 구성한 후 필요한 패키지를 설치합니다.

```bash
python -m pip install -r requirements.txt
```

GPU를 사용하는 경우 운영체제 및 CUDA 환경에 맞는 PyTorch 설치가 필요합니다.

### 데이터 전처리

원본 데이터를 검사하고 학습에 필요한 메타데이터를 생성합니다.

```bash
python -m src.preprocessing.prepare_training_data
```

기존 전처리 결과인 `fold_metadata.csv`가 준비되어 있다면 다시 실행할 필요는 없습니다.

### 데이터 입력 확인

```bash
python -m scripts.checks.check_dataloader
```

이미지와 라벨이 모델 입력에 맞는 형태로 불러와지는지 확인합니다.

### 단위 테스트

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

프로젝트의 단위 테스트를 실행합니다.

## 5. 관련 문서

- [소스 코드 구조](src/README.md) — 디렉터리 및 파일별 역할
- [데이터 전처리](reports/preprocessing/README.md) — 전처리 과정 및 분석 결과