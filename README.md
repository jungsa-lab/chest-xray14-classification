# NIH ChestX-ray14 Classification

NIH ChestX-ray14 데이터셋을 활용한 흉부 X-ray 다중 질환 분류 프로젝트입니다.

[1] Dataset

* NIH ChestX-ray14
* 224×224 resized version
* Kaggle: `khanfashee / NIH Chest X ray 14 224x224 resized`
* 전체 이미지: 112,120장
* 전체 환자: 30,805명
* 분류 대상: 14개 흉부 질환

실제 데이터셋 파일은 용량 문제로 저장소에 업로드하지 않습니다.

자세한 데이터 관련 정보는 `data/README.md`를 참고합니다.

[2] Goal

Chest X-ray 이미지를 입력으로 사용하여 14개 흉부 질환의 존재 여부를 동시에 예측하는
다중 라벨 분류(Multi-label Classification) 모델을 개발합니다.

[3] Project Structure

```text
.
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── Data_Entry_2017.csv
│   ├── processed/
│   │   ├── preprocessed_metadata.csv
│   │   ├── split_label_distribution.csv
│   │   ├── image_validation.csv
│   │   ├── missing_value_summary.csv
│   │   └── preprocessing_report.txt
│   └── README.md
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   └── preprocess_data.py
├── .gitignore
├── README.md
└── requirements.txt
```

* `data/raw/` : 원본 데이터셋 저장 위치
* `data/processed/` : 전처리 결과 저장 위치
* `notebooks/` : EDA 및 실험용 Jupyter Notebook
* `src/` : 데이터 전처리 및 모델 관련 Python 코드
* `requirements.txt` : Python 패키지 목록

[4] Workflow

1. Dataset preparation
2. Data preprocessing and validation
3. Patient-level Train / Validation / Test split
4. Baseline model development
5. Model training
6. Model evaluation
7. Model improvement and comparison

[5] Environment Setup

필요한 Python 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

Windows와 macOS 환경에서 모두 개발할 수 있도록 구성합니다.

[6] Data Preprocessing

1. 원본 데이터 준비

NIH ChestX-ray14 데이터를 다음 위치에 저장합니다.

```text
data/raw/
├── images/
│   ├── 00000001_000.png
│   ├── 00000001_001.png
│   └── ...
└── Data_Entry_2017.csv
```

전체 X-ray 이미지 112,120장을 `data/raw/images/`에 저장합니다.

2. 전처리 실행

프로젝트 루트 디렉터리에서 다음 명령어를 실행합니다.

```bash
python src/preprocess_data.py
```

3. 전처리 과정

`preprocess_data.py`에서는 다음 작업을 수행합니다.

* 메타데이터 로드 및 기본 구조 확인
* 결측치 및 중복 데이터 검사
* Patient Age 이상치 검사
* 14개 질환 Multi-label Encoding
* No Finding 및 라벨 이상 여부 검사
* 이미지와 메타데이터 매칭
* Patient ID 기준 Train / Validation / Test 분할
* Patient Leakage 검사
* 전체 X-ray 이미지 유효성 검사

4. 데이터 분할 결과

Patient ID를 기준으로 데이터를 분할하여 동일 환자의 이미지가 서로 다른 데이터셋에 포함되지 않도록 합니다.

| Dataset | Images | Patients |
|---|---:|---:|
| Train | 78,934 | 21,563 |
| Validation | 11,341 | 3,080 |
| Test | 21,845 | 6,162 |
| Total | 112,120 | 30,805 |

Patient Leakage는 `0`으로 확인되었습니다.

5. 전처리 결과

전처리가 완료되면 `data/processed/`에 다음 파일이 생성됩니다.

```text
data/processed/
├── preprocessed_metadata.csv
├── split_label_distribution.csv
├── image_validation.csv
├── missing_value_summary.csv
└── preprocessing_report.txt
```

* `preprocessed_metadata.csv` : 전처리된 메타데이터 및 데이터 분할 정보
* `split_label_distribution.csv` : Train / Validation / Test의 질환별 라벨 분포
* `image_validation.csv` : 전체 이미지 유효성 검사 결과
* `missing_value_summary.csv` : 메타데이터 결측치 검사 결과
* `preprocessing_report.txt` : 전체 전처리 결과 요약

현재 전처리 결과:

```text
Total Images     : 112,120
Unique Patients  : 30,805

Patient Leakage  : 0
Missing Images   : 0
Invalid Images   : 0
```

112,120장의 이미지가 모두 정상적으로 확인되었으며,
Train / Validation / Test 간 Patient Leakage가 발생하지 않았습니다.

[7] Next Step

데이터 전처리 완료 후 다음 단계에서는 DenseNet-121을 Baseline 모델로 사용하여
14개 흉부 질환에 대한 Multi-label Classification 모델을 학습합니다.