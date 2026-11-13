# DenseNet-121 모델 학습 및 gamma 비교 기록

NIH ChestX-ray14 데이터셋을 사용해 DenseNet-121 기반 14개 질환 다중 라벨 분류를 학습했다. 이 문서는 Kaggle에서 수행한 7회 실험과 현재 확인된 결과를 기록한다. 모든 성능 수치는 Train 내부 검증 데이터 기준이며, 고정 Test 성능이 아니다.

## 프로젝트 및 분할

- 저장소: `jungsa-lab/chest-xray14-classification`, `dulgix` 브랜치
- 학습 코드 기준 커밋: `b45867862e161dbbc9bc355852d38c3ce0653118`
- 입력: `data/processed/fold_metadata.csv` 및 `data/raw/`의 224×224 PNG 이미지
- 전체 데이터: 112,120장, 30,805명
- 환자 기준 Train/Test 분할 후 Train 내부 5-Fold GroupKFold 수행
- 품질 검사 통과 Train: 95,394장
- 고정 Test: 16,724장
- Train/Test 간 환자 중복: 0명
- Fold별 규모: 약 76,315장 학습 / 19,079장 검증
- Test 데이터는 감마 선택, 학습, Early Stopping 및 이번 결과 표 작성에 사용하지 않았다.

## 고정 실험 설정

- 모델: ImageNet 사전학습 DenseNet-121, 출력층 14 logits
- 손실 함수: 직접 구현한 멀티라벨 Focal Loss
- gamma 후보: 0, 1, 2. gamma=0은 질환별 pos_weight를 적용한 BCE 기준선
- pos_weight: 학습 음성 수 / 학습 양성 수. 각 Fold의 학습 데이터만 사용
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingLR, T_max=6
- 최대 Epoch: 6
- Early Stopping: patience=5, validation macro AUROC 기준
- Batch size: 16
- Workers: 2
- Seed: 42
- 입력 처리: CLAHE 적용(clipLimit=2.0, tileGridSize=8×8), grayscale을 3채널로 복제, ImageNet 정규화
- Train 증강: RandomRotation ±5°
- 실행 환경: Kaggle GPU, Tesla T4

각 Fold는 ImageNet 사전학습 가중치에서 새로 시작했다. 최고 모델은 해당 Fold의 validation macro AUROC 기준으로 저장했다.

## 실행 및 결과

Kaggle 노트북에서 gamma 0·1·2의 Fold 0 탐색을 수행한 뒤, 선택된 gamma 0으로 Fold 0~4 검증을 수행했다. 총 7회 학습했다.

- gamma 0, Fold 0: 학습 및 검증 완료
- gamma 1, Fold 0: 감마 탐색용 학습 및 검증 완료
- gamma 2, Fold 0: 감마 탐색용 학습 및 검증 완료
- gamma 0, Fold 1~4: 추가 5-Fold 검증 완료

### Gamma 탐색 결과 (Fold 0 검증)

| Gamma | Best Epoch | Mean AUROC | Mean AUPRC | ECE | 비고 |
|---:|---:|---:|---:|---:|---|
| 0 | 6 | 0.8283 | 0.2657 | 0.2241 | AUROC·AUPRC가 가장 높고 ECE가 가장 낮음 |
| 1 | 6 | 0.8271 | 0.2608 | 0.2682 |  |
| 2 | 6 | 0.8282 | 0.2510 | 0.3022 | gamma 0과 AUROC 차이가 매우 작음 |

표의 감마 비교는 **Fold 0 한 개의 검증 결과**다. 감마별 5-Fold 평균 비교 결과가 아니다.

### 선택 gamma 0의 5-Fold 검증 결과

| Fold | Best Epoch | 학습 이미지 | 검증 이미지 | Mean AUROC | Mean AUPRC | ECE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 76,315 | 19,079 | 0.8283 | 0.2657 | 0.2241 |
| 1 | 6 | 76,315 | 19,079 | 0.8330 | 0.2693 | 0.2228 |
| 2 | 5 | 76,315 | 19,079 | 0.8290 | 0.2522 | 0.2411 |
| 3 | 6 | 76,315 | 19,079 | 0.8358 | 0.2650 | 0.2379 |
| 4 | 6 | 76,316 | 19,078 | 0.8306 | 0.2553 | 0.2033 |
| **Mean ± Std** | — | — | — | **0.8313 ± 0.0031** | **0.2615 ± 0.0073** | **0.2259 ± 0.0150** |

표준편차는 다섯 Fold의 표본 표준편차(ddof=1)다.

## 최적 gamma 선택 근거

- Fold 0의 validation AUROC를 1차 기준으로 비교하고, 동률에 가까우면 AUPRC와 ECE를 함께 확인했다.
- Fold 0 결과에서 gamma 0은 AUROC 0.8283, AUPRC 0.2657, ECE 0.2241로 세 후보 중 가장 균형이 좋아 선택했다.
- gamma 0과 gamma 2의 AUROC 차이는 0.000155로 매우 작으므로 통계적으로 유의한 우위라고 주장하지 않는다.
- 선택된 gamma 0의 5-Fold 평균은 AUROC 0.8313 ± 0.0031, AUPRC 0.2615 ± 0.0073, ECE 0.2259 ± 0.0150이다.
- Fold 0에서 gamma를 선택한 뒤 같은 Fold 0을 5-Fold 평균에도 포함했으므로 선택 편향 가능성이 있다.
- 위 값은 교차검증 validation 성능이며 고정 Test 성능이 아니다.

## 요구사항 3: TTA 및 Ensemble 진행 상태

- EfficientNet-B0/B4 추가 학습: **미수행**. 요구사항 확인이 늦어 이번 실험에서 별도 학습하지 못했다.
- H-Flip TTA: **미수행**. 적용 전후 예측값을 만들지 않았다.
- TTA 전후 Mean AUROC 비교: **미수행**. 측정 결과 없음.
- DenseNet + EfficientNet Soft Voting: **미수행**. EfficientNet 결과가 없어 앙상블을 구성하지 않았다.
- Single Model 대비 Ensemble 향상 폭: **미측정**.
- 해석상 제한: H-Flip은 좌우 위치 의존 질환의 위치 정보를 바꿀 수 있다. 추후 적용할 경우 이 점을 결과와 Grad-CAM 해석에 명시해야 한다.

TTA(Test-Time Augmentation)는 추론 단계에서 원본 이미지와 좌우 반전 이미지를 각각 예측해 확률을 결합하는 방법이다. 이번 실험에서는 TTA 및 Ensemble 성능을 측정하지 않았으므로 수치를 기재하지 않는다.

## 기록된 실행 정보 및 결과 파일

- Kaggle 출력 폴더: `checkpoints/auto_gamma6/`
- 감마 탐색 결과: `gamma_screening.csv`
- Fold별 결과: `fold_results.csv`
- 집계 결과: `final_summary.json`
- 선택 gamma: `selected_gamma.json`
- 실행 설정: `run_config.json`
- 각 gamma/Fold 폴더: `best.pt`, `best_metrics.json`, `history.csv`, `pos_weight.json`, `completed.json`

`fold_results_partial.csv`는 `fold_results.csv`와 내용이 같고, `gamma_screening_partial.csv`는 `gamma_screening.csv`와 내용이 같아 최종 결과 정리에는 중복 파일을 사용하지 않는다.

## 유의사항 및 후속 작업

- 현재 감마별 비교는 gamma 0·1·2 각각의 Fold 0 결과만 포함한다. 요구사항에서 세 gamma 모두의 5-Fold 비교를 요구하면 gamma 1·2의 Fold 1~4 실험이 추가로 필요하다.
- ECE 0.2259는 목표 0.10을 초과한다. 보정 전 결과로 기록한다.
- 고정 Test 평가는 아직 수행하지 않았다.
- EfficientNet, TTA 및 Soft Voting은 수행하지 않아 해당 성능 비교는 미완료다.
- Kaggle 결과 ZIP에는 저장소 사본과 `.git`, 데이터, 코드가 함께 들어 있다. 저장소 반영 시 ZIP 전체를 덮어쓰지 말고 보고서와 경량 결과 파일만 복사한다.
- 대용량 `.pt` 가중치는 일반 Git 커밋에 넣지 않고 Kaggle Output 등 별도 위치에 보관한다. 저장소에는 가중치 보관 위치와 실행 설정을 기록한다.

