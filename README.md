# NIH ChestX-ray14 Classification

NIH ChestX-ray14 데이터셋을 활용한 흉부 X-ray 다중 질환 분류 프로젝트입니다.

## Dataset

* NIH ChestX-ray14
* 224×224 resized version
* Kaggle: `khanfashee / NIH Chest X ray 14 224x224 resized`

실제 데이터셋 파일은 저장소에 업로드하지 않습니다.

자세한 데이터 관련 정보는 `data/README.md`를 참고합니다.

## Goal

Chest X-ray 이미지를 입력으로 사용하여 14개 흉부 질환을 다중 라벨 분류(Multi-label Classification)하는 모델을 개발합니다.

## Project Structure

```text
.
├── data/
│   └── README.md
├── notebooks/
│   └── 01_eda.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

* `data/` : 데이터셋 저장 위치
* `notebooks/` : EDA 및 실험용 Jupyter Notebook
* `requirements.txt` : Python 패키지 목록

## Workflow

1. Dataset exploration and EDA
2. Data preprocessing
3. Train / Validation / Test split
4. Baseline model development
5. Model training
6. Model evaluation
7. Model improvement and comparison

## Environment Setup

```bash
pip install -r requirements.txt
```

Windows와 macOS 환경에서 모두 개발할 수 있도록 구성
