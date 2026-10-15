# DenseNet-121 모델 학습 및 gamma 비교 기록

## 프로젝트 및 분할
- 저장소: jungsa-lab/chest-xray14-classification, dulgix 브랜치
- 입력: `data/processed/fold_metadata.csv` 및 `data/raw/` PNG 이미지
- 환자 단위 GroupShuffleSplit(train/test) 후, Train 내부 5-Fold GroupKFold; 기존 CSV를 수정하지 않는다.
- Train(품질 통과) 95,394장, Test 16,724장 (기존 전처리 보고서 기준)
- Fold별 학습 약 76,315장 / 검증 약 19,079장.
- Test 데이터는 모델·gamma 선택이나 Early Stopping에 사용하지 않는다.

## 고정 실험 설정
- ImageNet pretrained DenseNet-121, 14 logits; 평가 시 sigmoid
- 멀티라벨 Focal Loss를 직접 구현. gamma=0은 pos_weight BCEWithLogitsLoss와 동일
- pos_weight=(학습 음성 수)/(학습 양성 수), **각 Fold train만** 사용
- AdamW lr=1e-4, weight_decay=1e-4, CosineAnnealingLR(T_max=30)
- 최대 30 epoch, patience=5, monitor=validation macro AUROC (엄격 개선 기준)
- batch=16(메모리에 따라 수정), seed=42, CLAHE=False, train augmentation RandomRotation(5°)
- 14개 질환별 AUROC/AUPRC 산출 후 macro 평균. ECE: 질환별 15개 동일폭 bin ECE 평균
- Fold 통계는 5개 Fold의 표본 표준편차(ddof=1)를 사용.

## 실행 및 결과
1. 단위 테스트 실행: `python -m unittest discover -s tests -v`
2. 모델 출력 점검: `python src/models/densenet.py` (이 명령은 다운로드 없이 구조만 확인)
3. 작은 시험 학습: `python src/training/train.py --gamma 0 --fold 0 --epochs 1 --max-batches 2 --batch-size 4 --device cpu`
   - smoke test는 지표가 NaN이며 유효 성능 결과가 아니므로 공식 결과에 포함하지 않는다.
4. 실제 단일 Fold: `python src/training/train.py --gamma 0 --fold 0 --epochs 30 --batch-size 16`
5. 전체 15회 실험: `python src/training/cross_validation.py --epochs 30 --batch-size 16`
6. 출력: `reports/training/fold_results.csv`, `gamma_comparison.csv`, `selected_gamma.json`
7. 가중치: `checkpoints/gamma_*/fold_*/best.pt` (Git ignore 대상)

## 결과 표 (실행 후 작성)
| Gamma | Mean AUROC ± Std | Mean AUPRC ± Std | ECE ± Std | 비고 |
|---:|---|---|---|---|
| 0 | 미실행 | 미실행 | 미실행 | pos_weight BCE 기준선 |
| 1 | 미실행 | 미실행 | 미실행 | |
| 2 | 미실행 | 미실행 | 미실행 | |

## 최적 gamma 선택 근거 (실행 후 작성)
- 사전에 정한 1차 기준은 **5개 Fold의 평균 validation AUROC가 가장 큰 gamma**.
- 동률이면 AUPRC가 높은 gamma, 그다음 ECE가 낮은 gamma를 선택한다.
- 선택한 gamma: **(실행 후 기입)**
- 다른 gamma와의 차이(평균/표준편차, 질환별 특징, 한계): **(실행 후 기입)**
- 이 결과는 교차검증 결과이며 고정 Test 성능이 아님. 최종 평가 과정은 별도 수행.

## 기록해야 할 실행 정보
- Git commit SHA, Python/torch/torchvision 버전, 장비(가속기/VRAM), 이미지 위치/원본 버전
- batch size, seed, CLAHE 사용 여부, max epochs, 각 Fold best epoch와 환자 누출 여부
- 다운로드된 사전학습 가중치 종류(DenseNet121_Weights.DEFAULT)

## 유의사항
- 기본 코드에서 `--max-batches` 옵션은 단지 실행 경로 확인을 위한 것.
- 5개 Fold는 매번 ImageNet pretrained 가중치에서 **새로 시작**한다.
- 새 데이터 분할을 생성하거나 환자별 Fold를 다시 섞지 않는다.
- 모델 학습에 GPU가 없어도 CPU 실행은 가능하지만 실제 15회 학습은 매우 무거울 수 있다.
- 이 안내서의 수치는 기존 전처리 보고서에서 가져온 것이며 실험 지표는 아직 비어 있다.
