# 모델 학습 실행 스크립트 및 실험 진행 기록

> NIH ChestX-ray14 다중 라벨 분류 프로젝트 · `jyanie` 브랜치  
> 마지막 정리: 2026-11-08

이 문서는 `scripts/`의 학습 관련 실행 스크립트, 현재까지 확인한 결과, 그리고 앞으로 진행할 실험 계획을 기록합니다. **계획한 실험과 실제 완료한 실험은 구분**합니다.

## 1. 관련 파일

| 경로 | 역할 | 현재 상태 |
| --- | --- | --- |
| `scripts/checks/smoke_train.py` | 학습 128장·검증 64장으로 DenseNet121, Focal Loss, AdamW, GPU 학습·검증 연결을 빠르게 확인 | 실행 성공 |
| `scripts/run_training.py` | 초기 학습 실행 버전(v1). gamma=2.0 고정, AdamW, EarlyStopping, 검증 지표 기록 및 체크포인트 저장·재개. **Cosine Annealing 미적용** | Fold 0에서 1 Epoch 실행 성공 |
| `scripts/run_training_v2.py` | 실험용 버전(v2). gamma 선택, AdamW + CosineAnnealingLR, EarlyStopping, 체크포인트·학습률·질환별 최적 성능 기록 | 구현 완료, **6 Epoch 정식 실험은 아직 시작 전** |

공통 학습 기능은 `src/training/`, 모델 평가 함수는 `src/evaluation/`, 데이터 로딩은 `src/data/`에 있습니다. 현재 v1과 v2를 **서로 다른 파일로 유지**합니다. 추후 v2 검증 후 정식 스크립트로 정리할 수 있습니다.

## 2. 현재까지 확인한 사항

- 개발 환경: Windows, PyTorch `2.10.0+cu128`, CUDA 사용 가능, NVIDIA GeForce RTX 4060.
- `data/processed/fold_metadata.csv` 및 `data/raw/images/` 경로 확인 완료.
- 기존 단위 테스트 **23개 통과**.
- `python -m scripts.checks.check_dataloader` 정상 통과 (Fold 0 학습 76,315장 / 검증 19,079장).
- `smoke_train.py` 소규모 테스트 정상 완료:
  - Train Loss `0.427036`, Val Loss `0.444321`
  - 학습 3.22초, 전체 3.55초 (학습 128장·검증 64장)
- **v1 실제 전체 학습: Fold 0 / gamma=2 / 1 Epoch 완료**
  - Train Loss `0.2977`, Val Loss `0.2831`
  - Mean AUROC `0.7615`, Mean AUPRC `0.1591`, Mean ECE `0.4097`
  - 소요 시간 `1425.6초` (약 23분 46초)
  - 결과 경로: `outputs/training/fold0/`
  - 이후 추가 학습을 시도했으나 `Ctrl+C`로 중단함. **위 결과는 v1의 1 Epoch 결과이며 v2 실험 결과가 아님.**

이전 결과는 **1 Epoch의 중간 성능**입니다. 최종 수렴 또는 모델 성능을 확정한 결과로 해석하지 않습니다.

## 3. 실험 설정 및 현실적인 계획

학습 시간과 GPU 자원 부담을 고려하여, **각 실험의 최대 Epoch를 6으로 설정**합니다. 과제에서 10 Epoch가 필수로 지정된 것은 아니며, 6 Epoch는 자원 제약을 반영한 절충안입니다. 수렴이 충분하지 않을 수 있다는 한계는 결과 분석에 포함합니다.

- 모델: ImageNet 사전 학습 DenseNet121, 14개 질환 다중 라벨 분류
- 손실 함수: Focal Loss (gamma 후보 `0`, `1`, `2`; 질환별 양성 가중치 적용)
- Optimizer: AdamW, 기본 학습률 `1e-4`
- Scheduler: CosineAnnealingLR (`T_max=6`, `eta_min=1e-6`)
- 교차검증: 환자 단위로 나눈 5-Fold의 Train/Validation 구간 사용
- 기준 설정: `--epochs 6 --batch-size 8`, EarlyStopping `patience=5`, seed `42`
- 독립적인 Test 분할은 gamma 선택이나 학습 중 모델 선택에 사용하지 않음

### 예정된 실험 순서

1. **Fold 0**에서 gamma=0, 1, 2를 각각 최대 6 Epoch 학습하고 검증 성능을 비교합니다. (**아직 미실행**)
2. Fold 0의 비교 결과를 바탕으로 gamma를 하나 선정하고, 선정 이유를 기록합니다.
3. 선정한 gamma로 **Fold 1~4**를 각각 최대 6 Epoch 학습합니다.
4. Fold 0~4의 성능 및 평균·표준편차(Mean ± Std)를 정리합니다.
5. 결과를 해석하고, 별도 Test 평가 및 후속 TTA/Ensemble 실험을 준비합니다.

이 계획에서 학습량의 상한은 `3 gamma × Fold 0 × 6 Epoch + 4개 추가 Fold × 6 Epoch = 42 Epoch`입니다. v1의 기존 1 Epoch는 제외하며, EarlyStopping에 따라 실제 횟수는 줄어들 수 있습니다. v1의 실측 시간을 단순 적용하면 약 **16시간 38분**이지만, 실제 시간은 실행 환경에 따라 달라집니다.

> **평가상의 한계:** Fold 0 검증 결과로 gamma를 선정하고 같은 Fold 0을 최종 5-Fold 평균에도 포함하면 선택 편향이 발생할 수 있습니다. 따라서 이 결과는 계산 자원을 고려한 **절충형 실험**임을 명시해야 합니다. 모든 gamma의 5-Fold 성능을 동일하게 비교한 결과라고 주장하지 않습니다. 더 엄밀한 비교가 요구된다면 모든 gamma × Fold 조합을 학습하거나 별도의 하이퍼파라미터 선택 절차가 필요합니다.

## 4. 실행 방법 (프로젝트 루트에서 실행)

아래 명령어는 Windows PowerShell 및 프로젝트 `.venv` 기준입니다.

### 실행 옵션 확인

```powershell
.\.venv\Scripts\python.exe -m scripts.run_training_v2 --help
```

### ① Fold 0, gamma=0부터 시작 (다음 실행 예정)

```powershell
.\.venv\Scripts\python.exe -m scripts.run_training_v2 --fold 0 --gamma 0 --epochs 6 --batch-size 8
```

### ② 같은 조건으로 gamma=1, 2 비교

```powershell
.\.venv\Scripts\python.exe -m scripts.run_training_v2 --fold 0 --gamma 1 --epochs 6 --batch-size 8
.\.venv\Scripts\python.exe -m scripts.run_training_v2 --fold 0 --gamma 2 --epochs 6 --batch-size 8
```

### ③ gamma 선정 후 Fold 1~4 학습

아래 예시의 `1`은 **선정된 gamma 값으로 바꾸어** 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_training_v2 --fold 1 --gamma 1 --epochs 6 --batch-size 8
.\.venv\Scripts\python.exe -m scripts.run_training_v2 --fold 2 --gamma 1 --epochs 6 --batch-size 8
.\.venv\Scripts\python.exe -m scripts.run_training_v2 --fold 3 --gamma 1 --epochs 6 --batch-size 8
.\.venv\Scripts\python.exe -m scripts.run_training_v2 --fold 4 --gamma 1 --epochs 6 --batch-size 8
```

### 중단된 **v2 실험**을 재개할 때

예: Fold 0 / gamma=0 / 최대 6 Epoch 설정을 유지하여 재개

```powershell
.\.venv\Scripts\python.exe -m scripts.run_training_v2 --fold 0 --gamma 0 --epochs 6 --batch-size 8 --resume
```

- `--resume`은 해당 조합의 `latest.pt`가 이미 있을 때만 사용합니다.
- v2는 저장한 설정과 **동일한 옵션**으로 재개해야 합니다 (`epochs`, `gamma`, `batch-size`, `lr`, `eta-min`, `patience`, `num-workers`, `seed` 등).
- 체크포인트는 **Epoch 종료 후** 기록됩니다. Epoch 도중 `Ctrl+C`로 중단하면 해당 Epoch는 다시 수행해야 할 수 있습니다.
- v1에서 저장한 `outputs/training/fold0/` 체크포인트는 v2로 재개하지 않습니다. v2의 Cosine Scheduler 적용 실험은 처음부터 별도로 수행합니다.
- 이미 결과가 있는 v2 경로에 `--resume` 없이 새로 실행하면 파일 보호를 위해 오류가 발생합니다.

## 5. 결과 파일 위치

```text
outputs/training/
├── fold0/                 # 기존 v1 결과: gamma=2, Cosine Annealing 없음
├── gamma0/
│   └── fold0/            # v2: gamma=0, Fold 0
├── gamma1/
│   └── fold0/            # v2: gamma=1, Fold 0
└── gamma2/
    └── fold0/            # v2: gamma=2, Fold 0
```

v2의 각 `gamma{값}/fold{번호}/` 폴더에는 실행 후 다음 파일이 생성됩니다.

| 파일 | 내용 |
| --- | --- |
| `latest.pt` | 마지막으로 완료한 Epoch의 모델·Optimizer·Scheduler·EarlyStopping 상태 (재개용) |
| `best.pt` | 검증 Mean AUROC가 가장 높았던 Epoch의 모델 가중치 |
| `history.csv` | Epoch별 학습률, Train/Val Loss, Mean AUROC/AUPRC/ECE 및 실행 시간 |
| `best_metrics.json` | 최적 AUROC Epoch와 질환별 AUROC/AUPRC/ECE |

`outputs/`와 `*.pt`는 `.gitignore` 대상으로, **Git에 모델 가중치는 커밋하지 않습니다.** 학습 코드와 이 README만 버전 관리합니다.

## 6. 아직 남은 일

- [ ] v2에서 gamma=0, Fold 0, 최대 6 Epoch 실험 완료
- [ ] v2에서 gamma=1, Fold 0, 최대 6 Epoch 실험 완료
- [ ] v2에서 gamma=2, Fold 0, 최대 6 Epoch 실험 완료
- [ ] gamma 비교표 작성 및 선택 근거 기록
- [ ] 선정한 gamma로 Fold 1~4 실험 완료
- [ ] Fold별 최적 성능의 Mean ± Std 집계
- [ ] Epoch별 성능 추이와 수렴·과적합 여부 검토
- [ ] 독립 Test 평가 및 후속 TTA/Ensemble 작업 검토

**주의:** `patience=5`를 유지하면서 상한을 6 Epoch로 정했으므로, EarlyStopping은 구현되어 있어도 실제로 조기 종료될 여지가 매우 적습니다. 최종 보고 시 이 설정상의 한계와 학습 자원 제약을 함께 설명합니다.
