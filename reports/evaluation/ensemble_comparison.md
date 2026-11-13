# 2-Model Soft Voting Ensemble

- 데이터: Test 16,724장
- 구성: DenseNet-121 + EfficientNet-B0, 질환별 확률의 단순 평균 (가중치 0.5 : 0.5, Test로 조정하지 않음)
- 최종 예측: TTA 없이 원본 이미지 확률 사용 (OOF로 결정, `tta_comparison.md` 참고). TTA 적용 결과는 아래 표에 함께 표시

## 사용한 모델 (검증 성능으로 선택)

| 모델 | 실험 | gamma | 검증 Mean AUROC |
| --- | --- | --- | --- |
| DenseNet-121 | `densenet121_g0_clahe-52362096` | 0 | 0.8216 ± 0.0047 |
| EfficientNet-B0 | `efficientnet_b0_g0_clahe-b70ecc0f` | 0 | 0.8148 ± 0.0017 |

## Fold별 비교 (TTA 없음, 최종, Mean ± Std)

Fold k의 Ensemble = DenseNet Fold k 모델 + EfficientNet Fold k 모델 (같은 학습 데이터로 학습한 두 모델)

| 방식 | Mean AUROC | Mean AUPRC | ECE |
| --- | --- | --- | --- |
| DenseNet-121 | 0.8203 ± 0.0041 | 0.2354 ± 0.0064 | 0.2316 ± 0.0140 |
| EfficientNet-B0 | 0.8139 ± 0.0025 | 0.2216 ± 0.0053 | 0.2598 ± 0.0199 |
| 2-Model Ensemble | 0.8294 ± 0.0011 | 0.2439 ± 0.0028 | 0.2457 ± 0.0095 |
| 향상 폭 (vs DenseNet-121) | +0.0091 ± 0.0044 | +0.0085 ± 0.0049 | +0.0141 ± 0.0144 |
| └ AUROC가 오른 Fold | 5/5 | | |
| 향상 폭 (vs EfficientNet-B0) | +0.0156 ± 0.0018 | +0.0222 ± 0.0058 | -0.0141 ± 0.0144 |
| └ AUROC가 오른 Fold | 5/5 | | |

## 5-Fold 평균 예측 (TTA 없음, 최종)

Single Model = 한 종류의 5개 Fold 모델 확률 평균, Ensemble = 두 종류 평균의 평균 (모델 10개)

| 방식 | Mean AUROC | Mean AUPRC | ECE | AUROC 향상 (vs DenseNet-121) |
| --- | --- | --- | --- | --- |
| DenseNet-121 | 0.8389 | 0.2585 | 0.2316 |  |
| EfficientNet-B0 | 0.8304 | 0.2436 | 0.2598 | -0.0086 |
| 2-Model Ensemble | 0.8408 | 0.2591 | 0.2457 | +0.0019 |

## Fold별 비교 (TTA 적용, Mean ± Std)

Fold k의 Ensemble = DenseNet Fold k 모델 + EfficientNet Fold k 모델 (같은 학습 데이터로 학습한 두 모델)

| 방식 | Mean AUROC | Mean AUPRC | ECE |
| --- | --- | --- | --- |
| DenseNet-121 | 0.8182 ± 0.0037 | 0.2271 ± 0.0061 | 0.2531 ± 0.0193 |
| EfficientNet-B0 | 0.8151 ± 0.0010 | 0.2198 ± 0.0053 | 0.2757 ± 0.0235 |
| 2-Model Ensemble | 0.8287 ± 0.0013 | 0.2367 ± 0.0041 | 0.2644 ± 0.0146 |
| 향상 폭 (vs DenseNet-121) | +0.0105 ± 0.0038 | +0.0097 ± 0.0027 | +0.0113 ± 0.0158 |
| └ AUROC가 오른 Fold | 5/5 | | |
| 향상 폭 (vs EfficientNet-B0) | +0.0136 ± 0.0015 | +0.0169 ± 0.0051 | -0.0113 ± 0.0158 |
| └ AUROC가 오른 Fold | 5/5 | | |

## 5-Fold 평균 예측 (TTA 적용)

Single Model = 한 종류의 5개 Fold 모델 확률 평균, Ensemble = 두 종류 평균의 평균 (모델 10개)

| 방식 | Mean AUROC | Mean AUPRC | ECE | AUROC 향상 (vs DenseNet-121) |
| --- | --- | --- | --- | --- |
| DenseNet-121 | 0.8364 | 0.2468 | 0.2531 |  |
| EfficientNet-B0 | 0.8299 | 0.2404 | 0.2757 | -0.0066 |
| 2-Model Ensemble | 0.8388 | 0.2508 | 0.2644 | +0.0023 |

ECE는 낮을수록 좋음. 향상 폭의 ECE가 음수면 Calibration이 좋아진 것.
