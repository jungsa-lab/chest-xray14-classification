# 모델 학습 결과 (5-Fold 교차검증)

NIH ChestX-ray14, 14개 질환 다중 라벨 분류. 모든 수치는 **검증 데이터** 기준이며 Test 데이터(15%)는 사용하지 않았다.

## 1. 학습 설정

| 항목 | 값 |
| --- | --- |
| 베이스라인 모델 | DenseNet-121 (ImageNet Pretrained, `IMAGENET1K_V1`) |
| 출력층 | `Linear(1024 → 14)`. Sigmoid는 Loss(`binary_cross_entropy_with_logits`)와 추론 단계에서 적용 |
| 추가 모델 | EfficientNet-B0 (ImageNet Pretrained, 마지막 Linear → 14) |
| 입력 | 224×224, CLAHE(clip 2.0, 8×8) → 3채널 복제 → ImageNet 정규화 |
| Loss | Focal Loss 직접 구현 (`src/training/losses.py`): `(1 − p_t)^γ × BCE(pos_weight)`, γ = 0이면 BCE와 동일 |
| pos_weight | 질환별 유병률 p로 `pos_weight = (1 − p) / p` (= 음성 수 / 양성 수). Fold마다 그 Fold의 학습 데이터로만 계산 (`src/training/weights.py`). 예: Infiltration p ≈ 17.8% → 약 4.6, Pneumonia p ≈ 1.2% → 약 82, Hernia p ≈ 0.19% → 약 552. 드문 질환일수록 양성 샘플의 Loss를 크게 키운다. |
| Optimizer | AdamW (lr 1e-4, weight decay 1e-5) |
| Scheduler | CosineAnnealingLR (T_max = 30, Epoch 단위 step) |
| 교차검증 | 5-Fold GroupKFold, 그룹 = Patient ID (학습 76,315장 / 검증 19,079장) |
| Early Stopping | monitor = 검증 Mean AUROC, patience = 5, 최대 30 Epoch |
| 기타 | batch 32, AMP(float16), seed 42 |
| 실행 환경 | Kaggle Notebook, Tesla T4 × 2 (`notebooks/kaggle_train.ipynb`) |

## 2. Gamma 비교 (DenseNet-121, 5-Fold Mean ± Std)

| gamma | Mean AUROC | Mean AUPRC | ECE | 해석 한 줄 |
| --- | --- | --- | --- | --- |
| 0 (BCE 동일) | **0.8216 ± 0.0047** | **0.2464 ± 0.0074** | **0.2284 ± 0.0140** | pos_weight만으로 불균형을 보정했을 때 세 지표 모두 가장 좋다. |
| 1 | 0.8186 ± 0.0049 | 0.2426 ± 0.0054 | 0.2790 ± 0.0389 | AUROC는 소폭(−0.003) 낮아지고 ECE가 크게 나빠진다. pos_weight와 Focal 가중이 겹쳐 확률이 과하게 치우친 것으로 보인다. |
| 2 | 0.8167 ± 0.0040 | 0.2383 ± 0.0058 | 0.3052 ± 0.0235 | 모든 지표가 가장 나쁘다. 쉬운 샘플의 기여를 너무 줄여 순위 성능과 Calibration이 함께 떨어진다. |

ECE는 낮을수록 좋다. 원본 표: `gamma_comparison.md`, Fold별 값: 5절 및 `fold_results.csv`.

## 3. 최적 gamma 선택: **gamma = 0**

**선택 규칙 (학습 전에 정함)**: 5-Fold 검증 Mean AUROC 평균이 가장 높은 gamma. 같으면 Mean AUPRC가 높은 쪽, 그다음 ECE가 낮은 쪽.

**근거**
1. **평균 성능**: gamma 0이 Mean AUROC, Mean AUPRC, ECE 세 지표 모두에서 가장 좋다.
2. **Fold별 일관성**: 같은 Fold(같은 학습·검증 데이터)끼리 짝지어 비교했다.

   | 비교 | AUROC 차이 (Mean ± Std) | gamma 0이 더 높은 Fold | ECE 차이 | gamma 0이 더 낮은 Fold |
   | --- | --- | --- | --- | --- |
   | gamma 0 − gamma 1 | +0.0030 ± 0.0030 | 4 / 5 | −0.0506 ± 0.0377 | 4 / 5 |
   | gamma 0 − gamma 2 | +0.0049 ± 0.0015 | 5 / 5 | −0.0768 ± 0.0250 | 5 / 5 |

3. **해석**: AUROC 차이(0.003~0.005)는 Fold 간 표준편차(약 0.005)와 비슷한 크기라 순위 성능 차이는 작다. 결정적인 차이는 Calibration(ECE)에서 나타난다. pos_weight가 이미 양성 샘플의 손실을 크게(Infiltration 약 4.6배 ~ Hernia 약 550배) 키운 상태에서 Focal Loss가 어려운 샘플을 한 번 더 강조하면, 예측 확률이 양성 쪽으로 과하게 밀려 ECE가 커진다. 따라서 이 설정(pos_weight 적용)에서는 Focal 항을 추가할 이점이 없다.

## 4. 추가 모델: EfficientNet-B0 (gamma 0)

| 모델 | Mean AUROC | Mean AUPRC | ECE |
| --- | --- | --- | --- |
| DenseNet-121 (gamma 0) | **0.8216 ± 0.0047** | **0.2464 ± 0.0074** | **0.2284 ± 0.0140** |
| EfficientNet-B0 (gamma 0) | 0.8148 ± 0.0017 | 0.2284 ± 0.0107 | 0.2572 ± 0.0193 |

EfficientNet-B0는 평균은 조금 낮지만 Fold 간 편차가 가장 작다. 두 모델은 3단계(TTA·Ensemble)에서 Soft Voting으로 결합한다.

## 5. Fold별 성능

각 Fold의 값은 Early Stopping 기준(검증 Mean AUROC)이 가장 높았던 Epoch(`best.pt` 저장 시점)의 검증 성능이다. Std는 Fold 5개의 표본 표준편차(ddof = 1).

**Mean AUROC**

| Fold | DenseNet γ0 | DenseNet γ1 | DenseNet γ2 | EfficientNet γ0 |
| --- | --- | --- | --- | --- |
| 0 | 0.8192 | 0.8176 | 0.8130 | 0.8141 |
| 1 | 0.8216 | 0.8228 | 0.8176 | 0.8146 |
| 2 | 0.8216 | 0.8151 | 0.8180 | 0.8127 |
| 3 | 0.8292 | 0.8245 | 0.8222 | 0.8173 |
| 4 | 0.8166 | 0.8131 | 0.8127 | 0.8152 |
| **Mean ± Std** | **0.8216 ± 0.0047** | **0.8186 ± 0.0049** | **0.8167 ± 0.0040** | **0.8148 ± 0.0017** |

**Mean AUPRC**

| Fold | DenseNet γ0 | DenseNet γ1 | DenseNet γ2 | EfficientNet γ0 |
| --- | --- | --- | --- | --- |
| 0 | 0.2519 | 0.2450 | 0.2330 | 0.2443 |
| 1 | 0.2498 | 0.2486 | 0.2470 | 0.2248 |
| 2 | 0.2461 | 0.2358 | 0.2390 | 0.2341 |
| 3 | 0.2506 | 0.2454 | 0.2395 | 0.2187 |
| 4 | 0.2337 | 0.2381 | 0.2330 | 0.2202 |
| **Mean ± Std** | **0.2464 ± 0.0074** | **0.2426 ± 0.0054** | **0.2383 ± 0.0058** | **0.2284 ± 0.0107** |

**ECE** (낮을수록 좋음)

| Fold | DenseNet γ0 | DenseNet γ1 | DenseNet γ2 | EfficientNet γ0 |
| --- | --- | --- | --- | --- |
| 0 | 0.2439 | 0.2999 | 0.3368 | 0.2421 |
| 1 | 0.2302 | 0.3051 | 0.2774 | 0.2644 |
| 2 | 0.2271 | 0.2116 | 0.2874 | 0.2319 |
| 3 | 0.2349 | 0.2990 | 0.3087 | 0.2721 |
| 4 | 0.2060 | 0.2792 | 0.3158 | 0.2757 |
| **Mean ± Std** | **0.2284 ± 0.0140** | **0.2790 ± 0.0389** | **0.3052 ± 0.0235** | **0.2572 ± 0.0193** |

**Best Epoch / 종료 Epoch** (Early Stopping: Best Epoch 이후 5 Epoch 동안 개선이 없어 종료)

| Fold | DenseNet γ0 | DenseNet γ1 | DenseNet γ2 | EfficientNet γ0 |
| --- | --- | --- | --- | --- |
| 0 | 4 / 9 | 4 / 9 | 4 / 9 | 4 / 9 |
| 1 | 5 / 10 | 4 / 9 | 6 / 11 | 4 / 9 |
| 2 | 5 / 10 | 7 / 12 | 6 / 11 | 5 / 10 |
| 3 | 5 / 10 | 5 / 10 | 6 / 11 | 3 / 8 |
| 4 | 6 / 11 | 5 / 10 | 5 / 10 | 3 / 8 |

## 6. 질환별 AUROC (5-Fold 평균)

| 질환 | DenseNet γ0 | DenseNet γ1 | DenseNet γ2 | EfficientNet γ0 |
| --- | --- | --- | --- | --- |
| Emphysema | 0.9115 | 0.9142 | 0.9098 | 0.9048 |
| Hernia | 0.9076 | 0.9067 | 0.9140 | 0.8867 |
| Cardiomegaly | 0.9004 | 0.8969 | 0.8957 | 0.8916 |
| Edema | 0.8890 | 0.8901 | 0.8884 | 0.8887 |
| Effusion | 0.8726 | 0.8726 | 0.8713 | 0.8668 |
| Pneumothorax | 0.8663 | 0.8682 | 0.8685 | 0.8632 |
| Mass | 0.8283 | 0.8196 | 0.8108 | 0.8044 |
| Consolidation | 0.7913 | 0.7867 | 0.7853 | 0.7870 |
| Atelectasis | 0.7912 | 0.7926 | 0.7898 | 0.7906 |
| Fibrosis | 0.7890 | 0.7812 | 0.7796 | 0.7910 |
| Pleural_Thickening | 0.7805 | 0.7763 | 0.7721 | 0.7757 |
| Nodule | 0.7410 | 0.7367 | 0.7326 | 0.7252 |
| Pneumonia | 0.7263 | 0.7148 | 0.7105 | 0.7273 |
| Infiltration | 0.7080 | 0.7042 | 0.7053 | 0.7039 |

형태가 뚜렷한 질환(Emphysema, Cardiomegaly, Edema)은 0.89 이상이고, 경계가 모호하거나 다른 소견과 겹치는 질환(Infiltration, Pneumonia, Nodule)은 0.71~0.74로 낮다. Std 포함 원본: `per_disease_auroc.csv`.

## 7. 관찰과 한계

- **이른 과적합**: 모든 Fold에서 최고 성능 Epoch는 3~7(평균 약 5)이고, 이후 5 Epoch 동안 개선이 없어 8~12 Epoch에서 Early Stopping으로 종료됐다. 학습 Loss는 계속 줄지만 검증 Loss는 4 Epoch 무렵부터 증가한다 (`history/`).
- **Cosine Annealing 효과 제한**: T_max = 30으로 설정했지만 실제 학습이 약 10 Epoch에서 끝나 학습률이 1e-4에서 약 7.5e-5(종료 시점)까지만 내려갔다. 코사인 곡선의 앞부분 1/3만 사용해 학습률이 작아지는 후반 구간에 도달하지 못했다. T_max를 실제 학습 길이에 맞추거나 학습 Augmentation을 추가하면 개선 여지가 있다.
- **높은 ECE**: pos_weight가 양성 확률을 위로 끌어올리므로 gamma 0에서도 ECE가 0.2대다. 확률 자체를 해석에 쓰려면 Temperature Scaling 같은 사후 보정이 필요하다.
- **gamma 간 AUROC 차이는 작다**: Fold 표준편차 수준이므로 "gamma 0이 순위 성능에서 확실히 우월하다"기보다 "Focal 항이 이득을 주지 않고 Calibration을 해친다"로 해석하는 것이 정확하다.

## 8. 파일 구성

| 파일 | 내용 |
| --- | --- |
| `gamma_comparison.md` | gamma 비교표 (자동 생성) |
| `selected_gamma.json` | 선택한 gamma와 선택 규칙 |
| `cv_summary.csv` | 실험별 Mean ± Std |
| `fold_results.csv` | 실험·Fold별 최고 Epoch와 성능 |
| `per_disease_auroc.csv` | 실험·질환별 AUROC Mean ± Std |
| `history/<run_id>_fold_<k>.csv` | Epoch별 학습률, Loss, 검증 지표 (학습 곡선) |

모델 가중치(`best.pt`, Fold별 DenseNet-121 약 28MB, EfficientNet-B0 약 16MB)는 용량 때문에 저장소에 포함하지 않았다. Kaggle 노트북 Output에 보관되어 있다.

**재현**: `python -m src.training.run_queue --gpus 0 1` (Kaggle 실행 방법은 `notebooks/kaggle_train.ipynb` 참고). 요약표만 다시 만들 때는 `python -m src.training.summarize`.
