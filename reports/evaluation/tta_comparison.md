# Test-Time Augmentation (Horizontal Flip) 전후 비교

- 데이터: Test 16,724장 (학습·검증과 환자가 겹치지 않음, 모델 선택·학습에 사용하지 않음)
- TTA: 원본과 좌우 반전 이미지의 예측 확률 평균
- 모델: DenseNet-121 `densenet121_g0_clahe-52362096`, EfficientNet-B0 `efficientnet_b0_g0_clahe-b70ecc0f`

## 최종 예측의 TTA 사용 여부 (OOF로 결정)

- 규칙: OOF(Fold k 검증 이미지를 Fold k 모델로 예측)에서 2-Model Ensemble의 Fold별 Mean AUROC를 TTA 있음/없음으로 계산해, 5개 Fold 평균 차이(TTA − 없음)가 0보다 크면 TTA 사용. Test 성능은 사용하지 않음.

| Fold | TTA 없음 | TTA (H-Flip) | 변화 |
| --- | --- | --- | --- |
| 0 | 0.8270 | 0.8262 | -0.0007 |
| 1 | 0.8315 | 0.8291 | -0.0024 |
| 2 | 0.8290 | 0.8251 | -0.0039 |
| 3 | 0.8336 | 0.8330 | -0.0006 |
| 4 | 0.8290 | 0.8261 | -0.0029 |
| 평균 | 0.8300 | 0.8279 | -0.0021 |

- TTA로 오른 Fold: 0/5
- **결정: TTA 사용 안 함** → `test_predictions.csv`·`oof_predictions.csv`와 이후 Calibration·Operating Point·Subgroup·Grad-CAM·External 모두 이 방식의 확률을 사용

## Test Mean AUROC (참고 보고, 결정에는 사용하지 않음)

Fold별 = Fold 모델 하나씩 평가한 5개 값의 Mean ± Std, 5-Fold 평균 = 5개 Fold 모델 확률을 평균낸 예측

| 모델 | 단위 | TTA 없음 | TTA (H-Flip) | 변화 | 개선된 Fold |
| --- | --- | --- | --- | --- | --- |
| DenseNet-121 | Fold별 | 0.8203 ± 0.0041 | 0.8182 ± 0.0037 | -0.0021 ± 0.0015 | 1/5 |
| DenseNet-121 | 5-Fold 평균 | 0.8389 | 0.8364 | -0.0025 | |
| EfficientNet-B0 | Fold별 | 0.8139 ± 0.0025 | 0.8151 ± 0.0010 | +0.0013 ± 0.0019 | 4/5 |
| EfficientNet-B0 | 5-Fold 평균 | 0.8304 | 0.8299 | -0.0005 | |
| 2-Model Ensemble | Fold별 | 0.8294 ± 0.0011 | 0.8287 ± 0.0013 | -0.0007 ± 0.0004 | 0/5 |
| 2-Model Ensemble | 5-Fold 평균 | 0.8408 | 0.8388 | -0.0021 | |

## 다른 지표 (5-Fold 평균 예측)

| 모델 | TTA | Mean AUROC | Mean AUPRC | ECE |
| --- | --- | --- | --- | --- |
| DenseNet-121 | X | 0.8389 | 0.2585 | 0.2316 |
| DenseNet-121 | O | 0.8364 | 0.2468 | 0.2531 |
| EfficientNet-B0 | X | 0.8304 | 0.2436 | 0.2598 |
| EfficientNet-B0 | O | 0.8299 | 0.2404 | 0.2757 |
| 2-Model Ensemble | X | 0.8408 | 0.2591 | 0.2457 |
| 2-Model Ensemble | O | 0.8388 | 0.2508 | 0.2644 |

## 질환별 AUROC 변화 (TTA − 원본, 5-Fold 평균 예측)

| 질환 | DenseNet-121 | EfficientNet-B0 | 2-Model Ensemble |
| --- | --- | --- | --- |
| Atelectasis | -0.0029 | -0.0005 | -0.0024 |
| Cardiomegaly (좌우 위치 의존) | -0.0268 | -0.0102 | -0.0155 |
| Effusion | -0.0015 | +0.0000 | -0.0011 |
| Infiltration | -0.0044 | -0.0006 | -0.0034 |
| Mass | -0.0060 | -0.0010 | -0.0038 |
| Nodule | +0.0029 | -0.0002 | +0.0008 |
| Pneumonia | +0.0078 | +0.0095 | +0.0074 |
| Pneumothorax | -0.0004 | -0.0006 | +0.0001 |
| Consolidation | -0.0053 | -0.0023 | -0.0044 |
| Edema | +0.0009 | +0.0003 | +0.0000 |
| Emphysema | -0.0027 | +0.0003 | -0.0016 |
| Fibrosis | +0.0005 | -0.0002 | -0.0008 |
| Pleural_Thickening | -0.0005 | -0.0036 | -0.0022 |
| Hernia | +0.0038 | +0.0020 | -0.0020 |

## 제한사항 (Grad-CAM 해석 시 주의)

- 흉부 X-ray는 좌우가 대칭이 아니다. 심장은 왼쪽으로 치우쳐 있고(Cardiomegaly), 대동맥궁·위 공기음영은 왼쪽, 간은 오른쪽에 있다. 좌우 반전 이미지는 실제로 존재하지 않는 해부학적 배치(우심증 형태)다.
- 학습 때는 좌우 반전 Augmentation을 쓰지 않았으므로, 반전 이미지는 모델이 본 적 없는 분포다. TTA는 성능 개선 여부를 확인하는 목적으로만 적용했다.
- 반전 이미지에서 만든 Grad-CAM은 병변 위치가 좌우로 바뀌어 나타난다. 다시 반전해 원래 좌표로 되돌려도, 모델이 '왼쪽 심장 경계'를 보고 판단했는지 '오른쪽'을 봤는지는 원본과 의미가 달라진다.
- 따라서 Grad-CAM 위치 해석은 **원본 이미지 예측에서만** 하고, TTA 평균 확률과 원본 Grad-CAM을 한 그림에 섞어 해석하지 않는다. 좌측 폐 병변·Cardiomegaly처럼 위치가 진단 근거인 질환은 특히 그렇다.
