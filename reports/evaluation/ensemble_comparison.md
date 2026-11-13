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

## 14개 질환별 AUROC · AUPRC (Test, 5-Fold 평균 예측, TTA 없음)

최종 예측은 2-Model Ensemble. AUPRC는 유병률에 따라 기준값(무작위 = 유병률)이 달라 질환 간 직접 비교하지 않는다. 모든 수치는 NIH 자동 추출 라벨 기준이다.

| 질환 | DenseNet-121 AUROC | DenseNet-121 AUPRC | EfficientNet-B0 AUROC | EfficientNet-B0 AUPRC | 2-Model Ensemble AUROC | 2-Model Ensemble AUPRC |
| --- | --- | --- | --- | --- | --- | --- |
| Atelectasis | 0.8030 | 0.3219 | 0.7932 | 0.3049 | 0.8018 | 0.3184 |
| Cardiomegaly | 0.9231 | 0.3002 | 0.9148 | 0.2698 | 0.9246 | 0.2930 |
| Effusion | 0.8835 | 0.5109 | 0.8738 | 0.4895 | 0.8821 | 0.5096 |
| Infiltration | 0.6979 | 0.3273 | 0.6899 | 0.3273 | 0.6970 | 0.3301 |
| Mass | 0.8439 | 0.3169 | 0.8128 | 0.2703 | 0.8380 | 0.3021 |
| Nodule | 0.7526 | 0.2267 | 0.7326 | 0.1880 | 0.7510 | 0.2137 |
| Pneumonia | 0.7652 | 0.0441 | 0.7532 | 0.0416 | 0.7674 | 0.0452 |
| Pneumothorax | 0.8666 | 0.2951 | 0.8724 | 0.3369 | 0.8742 | 0.3191 |
| Consolidation | 0.8224 | 0.1614 | 0.8132 | 0.1511 | 0.8212 | 0.1598 |
| Edema | 0.8954 | 0.1539 | 0.8951 | 0.1687 | 0.8987 | 0.1669 |
| Emphysema | 0.9281 | 0.3711 | 0.9142 | 0.3435 | 0.9281 | 0.3651 |
| Fibrosis | 0.8232 | 0.1063 | 0.8161 | 0.1064 | 0.8270 | 0.1125 |
| Pleural_Thickening | 0.8087 | 0.1393 | 0.7999 | 0.1312 | 0.8096 | 0.1394 |
| Hernia | 0.9313 | 0.3432 | 0.9438 | 0.2808 | 0.9510 | 0.3522 |
| **평균** | **0.8389** | **0.2585** | **0.8304** | **0.2436** | **0.8408** | **0.2591** |
