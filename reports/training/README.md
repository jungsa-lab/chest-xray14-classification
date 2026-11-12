# 모델 학습 실험 결과

> NIH ChestX-ray14 다중 라벨 분류 프로젝트 · `jyanie` 브랜치  
> 정리일: 2026-11-11  
> 상태: **Fold 0의 gamma 비교 및 선정 gamma의 5-Fold 학습 완료** / **독립 Test 평가 미실시**

이 문서는 모델 학습의 **실험 조건, 실제 수행 내역, 검증 결과 및 해석**을 기록합니다. 학습 스크립트의 사용법과 진행 이력은 [`../../scripts/README.md`](../../scripts/README.md)를 참고하세요.

## 1. 실험 목적 및 방법

- **데이터:** NIH ChestX-ray14, 14개 질환 다중 라벨 분류
- **모델:** ImageNet 사전학습 가중치를 사용한 DenseNet121, 출력 14개 Logit
- **입력:** 224×224, 3채널, CLAHE 적용 및 ImageNet 정규화
- **데이터 분할:** 환자 단위로 분리한 5-Fold 학습·검증 분할. 독립적인 Test 데이터는 학습, gamma 선택 및 아래 검증 지표 계산에 사용하지 않음
- **각 Fold의 데이터량:** 전체 Train/Validation 대상 95,394장. Fold 0~3은 학습 76,315장·검증 19,079장, Fold 4는 학습 76,316장·검증 19,078장
- **손실 함수:** Focal Loss, `gamma ∈ {0, 1, 2}` 비교; 질환별 `pos_weight`는 **각 Fold의 학습 데이터에서만** 계산
- **Optimizer:** AdamW (`lr=1e-4`, PyTorch 기본 `weight_decay=0.01`)
- **Scheduler:** CosineAnnealingLR (`T_max=6`, `eta_min=1e-6`, Epoch마다 갱신)
- **학습 조건:** `batch_size=8`, `seed=42`, Fold별 최대 6 Epoch, EarlyStopping `patience=5`
- **모델 선택:** 각 Fold에서 검증 **Mean AUROC가 가장 높은 Epoch**의 체크포인트(`best.pt`) 선택. 해당 Epoch의 AUPRC·ECE를 함께 보고
- **실행 코드:** `scripts/run_training_v2.py` (`src/training/` 및 `src/evaluation/`의 공통 모듈 사용)

`gamma=0`은 Focal Loss의 focusing 항이 1이므로, 이 구현에서는 질환별 양성 가중치를 적용한 BCE와 동등합니다.

## 2. 실험 수행 범위

계산 자원 및 학습 시간을 고려하여 다음과 같은 **절충형 실험 계획**을 적용했습니다.

1. **Fold 0**에서 gamma=0, 1, 2를 각각 최대 6 Epoch 학습해 비교
2. Fold 0의 최고 검증 Mean AUROC 기준으로 **gamma=0 선정**
3. gamma=0을 유지하여 Fold 1~4를 각각 6 Epoch 학습
4. gamma=0의 5개 Fold별 최고 AUROC 체크포인트로 교차검증 결과 집계

정식 실험에서 수행한 학습량은 총 **42 Epoch**입니다. (`gamma 3종 × Fold 0 × 6 = 18`, 선정 gamma의 나머지 4개 Fold × 6 = 24.) 초기 v1의 별도 1 Epoch 테스트는 이 결과에 포함하지 않았습니다.

Fold 1은 Epoch 6 도중 시스템 메모리 할당 오류로 한 번 중단됐으나, Epoch 5의 `latest.pt`에서 `--resume`으로 복구해 정상 완료했습니다. Fold 3은 첫 Epoch 완료 전 취소했던 실행 폴더를 백업한 뒤 처음부터 재실행했습니다. 최종 실험 결과는 모두 별도 폴더에 저장됐습니다.

## 3. gamma 비교: Fold 0

다음 표는 gamma별로 **검증 Mean AUROC가 최고인 체크포인트**를 기준으로 작성했습니다. 따라서 AUPRC나 ECE가 해당 gamma에서 기록한 개별 최고·최저값과 다를 수 있습니다.

| gamma | 선정 Epoch | Mean AUROC ↑ | Mean AUPRC ↑ | Mean ECE ↓ |
| ---: | ---: | ---: | ---: | ---: |
| **0** | 4 | **0.8262** | 0.2502 | **0.2693** |
| 1 | 5 | 0.8241 | **0.2526** | 0.2956 |
| 2 | 6 | 0.8221 | 0.2494 | 0.2883 |

**선정 결과:** Fold 0에서 Mean AUROC가 가장 높고, 해당 체크포인트의 ECE도 가장 낮은 gamma=0을 선택했습니다. gamma=1의 AUPRC가 약간 높으므로 모든 평가 지표에서 gamma=0이 우수한 것은 아닙니다.

- gamma=0과 gamma=1의 AUROC 차이: **0.0021**
- gamma=0과 gamma=2의 AUROC 차이: **0.0041**

차이가 작고 단일 Fold에서 비교했으므로, gamma=0의 일반적인 우수성이나 통계적 유의성을 입증한 결과로 해석하지 않습니다.

## 4. gamma=0의 5-Fold 교차검증 결과

각 Fold에서 **최고 검증 Mean AUROC를 기록한 `best.pt`** 기준입니다.

| Fold | 최적 Epoch | Mean AUROC ↑ | Mean AUPRC ↑ | Mean ECE ↓ |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 4 | 0.8262 | 0.2502 | 0.2693 |
| 1 | 5 | 0.8286 | 0.2611 | 0.2427 |
| 2 | 5 | 0.8233 | 0.2437 | 0.2575 |
| 3 | 6 | **0.8321** | 0.2583 | 0.2300 |
| 4 | 6 | 0.8223 | 0.2428 | **0.2292** |
| **Mean ± Std** | — | **0.8265 ± 0.0040** | **0.2512 ± 0.0083** | **0.2457 ± 0.0175** |

- `Mean`: 5개 Fold 검증 지표의 산술평균
- `Std`: 5개 Fold 간 **표본 표준편차** (`ddof=1`)
- 계산에는 콘솔 로그에서 제공한 소수점 넷째 자리 수치를 사용했습니다. `best_metrics.json`의 원래 정밀도 값으로 재집계할 경우 마지막 자리에서 차이가 있을 수 있습니다.
- 위 수치는 **교차검증의 검증(Validation) 성능**이며, 독립 Test 성능이 아닙니다.

### 해석

- AUROC가 Fold별 **0.8223~0.8321** 범위에 있고, Fold 간 표본 표준편차는 약 **0.0040**입니다. 본 분할에서는 검증 AUROC의 Fold 간 변동 폭이 비교적 작았습니다.
- Fold 0~2에서는 최고 AUROC가 Epoch 4~5에 나타났으며, Fold 3~4에서는 마지막 Epoch 6에 나타났습니다. 따라서 **모든 Fold가 충분히 수렴했다고 단정할 수 없습니다.**
- 여러 Fold에서 후반 Train Loss는 지속 감소한 반면 Val Loss가 상승하는 양상이 나타났습니다. 과적합 가능성을 검토할 신호지만, 검증 지표와 함께 해석해야 합니다.
- `patience=5`로 최대 6 Epoch를 학습했기 때문에, EarlyStopping이 실제 조기 종료를 일으킬 여지는 매우 제한적이었습니다.

## 5. 실험 설계상의 한계

1. **gamma 선택 편향:** Fold 0 검증 성능을 이용해 gamma=0을 선정하고, 같은 Fold 0을 최종 5-Fold 평균에도 포함했습니다. 따라서 최종 평균은 하이퍼파라미터 선정 과정으로 인해 다소 낙관적일 수 있습니다.
2. **gamma별 5-Fold 비교를 수행하지 않음:** gamma=1, 2는 Fold 0에서만 학습했습니다. 위 비교는 gamma별 5-Fold Mean ± Std 비교가 아닌 **Fold 0의 예비 비교**입니다.
3. **계산 자원 제약:** Fold마다 최대 6 Epoch만 학습했으며, 모든 Fold의 충분한 수렴을 보장하지 않습니다.
4. **아직 독립 Test 성능이 없음:** 일반화 성능을 최종 판단하려면 보존된 Test 데이터로 별도 평가해야 합니다.

이 한계를 명시한 상태에서, 본 실험은 **제한된 계산 자원으로 gamma를 예비 비교하고 선정한 설정의 환자 단위 5-Fold 검증 결과를 확보한 실험**으로 보고합니다.

## 6. 결과 파일 위치 및 재현 방법

```text
outputs/training/
├── fold0/                 # 별도 초기 v1 실험: gamma=2, Cosine Annealing 미적용
├── gamma0/
│   ├── fold0/
│   ├── fold1/
│   ├── fold2/
│   ├── fold3/
│   └── fold4/
├── gamma1/
│   └── fold0/
└── gamma2/
    └── fold0/
```

v2의 각 실험 폴더는 `best.pt`(최고 AUROC 모델), `latest.pt`(마지막 Epoch에서 재개할 체크포인트), `history.csv`(Epoch별 성능), `best_metrics.json`(선정 Epoch의 질환별·전체 성능)을 저장합니다.

프로젝트 루트에서 사용하는 실행 명령 예시:

```powershell
# gamma 비교 예시
.\.venv\Scripts\python.exe -m scripts.run_training_v2 --fold 0 --gamma 0 --epochs 6 --batch-size 8

# 선정 gamma의 다른 Fold 실행 예시
.\.venv\Scripts\python.exe -m scripts.run_training_v2 --fold 4 --gamma 0 --epochs 6 --batch-size 8

# 중단 후 재개 시: 기존과 동일한 설정 및 --resume 사용
.\.venv\Scripts\python.exe -m scripts.run_training_v2 --fold 4 --gamma 0 --epochs 6 --batch-size 8 --resume
```

> **주의:** 위 명령어는 재현을 위한 예시입니다. 이미 결과가 존재하는 폴더에 `--resume` 없이 실행하면 기존 결과 보호를 위해 `FileExistsError`가 발생합니다. 체크포인트와 모델 가중치는 `.gitignore` 정책에 따라 Git에 커밋하지 않습니다.

## 7. 후속 작업

- [x] gamma=0, 1, 2의 Fold 0 비교 완료
- [x] gamma=0 선정 및 5-Fold 학습 완료
- [x] 5-Fold AUROC·AUPRC·ECE의 Mean ± Std 정리
- [ ] `best_metrics.json` 원본 정밀도 값으로 결과표 자동 집계·검증
- [ ] Epoch별 학습 곡선 및 질환별 AUROC/AUPRC 분석
- [ ] 독립 Test 데이터 평가 및 최종 성능 보고
- [ ] 필요시 앙상블·TTA 및 예측 확률 Calibration 검토
