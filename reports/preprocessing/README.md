# 데이터 전처리 진행 현황 및 결과 안내

## 1. 목적과 현재 범위

NIH ChestX-ray14 데이터의 메타데이터와 이미지 상태를 검사하고,
환자 단위로 학습·검증·최종 평가 데이터를 구분하였다.

팀원별 모델 실험은 동일한 `fold_metadata.csv`를 기준으로 수행한다.
원본 이미지 데이터는 저장소에 포함하지 않는다.

현재 전처리 전체가 완료된 상태는 아니다.
CLAHE 적용 및 비교 분석, ImageNet 정규화, DICOM 실습 등은 후속 작업이다.

## 2. 주요 결과

| 항목 | 결과 |
|---|---:|
| 전체 이미지 | 112,120장 |
| 고유 환자 | 30,805명 |
| 메타데이터와 연결되지 않은 이미지 행 | 0 |
| 메타데이터에 없는 추가 PNG | 0 |
| 이미지 읽기 실패 | 0 |
| 224×224 크기 불일치 | 0 |
| 나이 결측 | 0 |
| 나이 형식 오류 또는 허용 범위 초과 | 16 |
| 원본 전체 No Finding 이미지 | 60,412장 |
| 원본 전체에서 질환 라벨 2개 이상인 이미지 | 20,735장 |
| 품질 제외 전 Train | 95,396장 |
| Train 품질 제외 | 2장 |
| 교차검증에 사용하는 Train | 95,394장 |
| 고정 Test | 16,724장 |

## 3. 메타데이터 처리

### 나이

원본 `Patient Age`는 보존한다.

- Y: 연 단위 그대로 사용
- M: 12로 나누어 연 단위로 환산
- D: 365.25로 나누어 연 단위로 환산
- 프로젝트 처리 기준: 0~120년
- 형식 오류 또는 범위 초과 값은 `Patient Age Years`에서 결측 처리

나이 이상치가 있다는 이유만으로 해당 이미지를 제외하지는 않았다.

### 질환 라벨

14개 질환을 각각 0 또는 1로 표현하는 Multi-hot 방식으로 변환하였다.

- 한 이미지에 여러 질환이 표시될 수 있다.
- `No Finding`은 별도 보조 열로 보관한다.
- `No Finding` 이미지의 14개 질환 값은 모두 0이다.
- 모델의 질환 정답 열은 `src/prepare_training_data.py`의
  `DISEASES` 순서를 기준으로 한다.
- `No Finding`과 `Positive Label Count`는 14개 출력에 포함하지 않는다.

여기서 0과 1은 데이터셋의 라벨을 표현하며,
임상적으로 질환 유무를 새로 판정한 결과가 아니다.

## 4. 이미지 검사 및 품질 처리

전체 PNG의 읽기 가능 여부, 크기, 이미지 모드와 밝기 통계를 검사하였다.

- L 이미지: 111,601장
- LA 이미지: 519장
- LA 이미지 519장은 알파 채널의 모든 값이 255로 완전 불투명했다.
- LA 이미지의 밝기도 회색조로 변환하여 측정하였다.
- 원본 이미지 파일은 수정하지 않았다.

현재 품질 기준은 다음과 같다.

| 조건 | 기준 |
|---|---|
| Too Dark | 평균 밝기 < 5 |
| Too Bright | 평균 밝기 > 250 |
| Nearly Uniform | 픽셀 밝기 표준편차 < 3 |

이는 프로젝트의 보수적인 공학적 선별 기준이며 임상 판독 기준이 아니다.

Train에서 Too Dark와 Too Bright 해당 이미지는 없었다.
Nearly Uniform 이미지 2장을 학습·검증 대상에서 제외하였다.

| 이미지 | 제외 사유 |
|---|---|
| 00007160_002.png | nearly_uniform |
| 00010007_121.png | nearly_uniform |

품질 플래그와 실제 제외는 구분한다.
고정 Test는 품질 플래그가 있더라도 주 평가 대상에서 임의로 제외하지 않는다.

## 5. 데이터 분할

### Train/Test

Patient ID를 기준으로 GroupShuffleSplit을 사용하였다.

- random_state: 42
- Test 환자 비율 설정: 15%
- Train 환자: 26,184명
- Test 환자: 4,621명
- Train/Test 중복 환자: 0명

85:15는 환자 그룹을 기준으로 한 비율이므로,
이미지 수의 비율은 정확히 85:15와 일치하지 않을 수 있다.

### Train 내부 교차검증

품질 검사를 통과한 Train에 5-Fold GroupKFold를 적용하였다.

| Fold | 학습 이미지 | 검증 이미지 | 학습 환자 | 검증 환자 | 중복 환자 |
|---|---:|---:|---:|---:|---:|
| 0 | 76,315 | 19,079 | 20,947 | 5,237 | 0 |
| 1 | 76,315 | 19,079 | 20,947 | 5,237 | 0 |
| 2 | 76,315 | 19,079 | 20,947 | 5,237 | 0 |
| 3 | 76,315 | 19,079 | 20,947 | 5,237 | 0 |
| 4 | 76,316 | 19,078 | 20,948 | 5,236 | 0 |

각 Fold의 학습·검증 데이터에서 14개 질환 모두 양성과 음성이 존재한다.
질환별 비율이 동일하도록 보장하는 분할은 아니다.

Test와 품질 제외 Train에는 Fold=-1이 기록된다.
따라서 Fold=-1만으로 Test를 선택하면 안 된다.

## 6. 팀원 사용 방법

### 데이터 배치

저장소 루트를 기준으로 원본 데이터를 다음 위치에 준비한다.

- 메타데이터: `data/raw/Data_Entry_2017.csv`
- 이미지 예시: `data/raw/images/images-224/00000001_000.png`

`fold_metadata.csv`의 `Image Path`는 `data/raw/`를 기준으로 한 상대 경로다.

공유된 결과물을 사용하는 팀원은 전체 이미지 검사를 다시 수행할 필요 없이
공통 메타데이터를 읽어 모델 입력 코드를 작성할 수 있다.
단, 자신의 이미지 파일이 해당 경로에 존재하는지는 확인해야 한다.

### Fold 선택 예시

다음 코드는 저장소 루트에서 실행하는 경우를 기준으로 한다.

```python
from pathlib import Path

import pandas as pd

df = pd.read_csv(
    "data/processed/fold_metadata.csv",
    low_memory=False,
)

k = 0

eligible_train = df["Split"].eq("train") & df["Fold"].ge(0)

train_df = df.loc[
    eligible_train & df["Fold"].ne(k)
].copy()

val_df = df.loc[
    eligible_train & df["Fold"].eq(k)
].copy()

test_df = df.loc[df["Split"].eq("test")].copy()

image_path = (
    Path("data/raw") / train_df.iloc[0]["Image Path"]
)
```

`Fold != k` 조건만 사용하면 Test와 제외 이미지까지 포함될 수 있으므로
반드시 Train 및 유효 Fold 조건을 함께 적용한다.

각 Fold는 모델을 새로 초기화하여 학습한다.
이전 Fold에서 학습한 가중치를 다음 Fold의 시작점으로 사용하지 않는다.

모델 선택과 설정 비교는 Train 내부 교차검증을 사용한다.
Test는 설정을 확정한 후 최종 평가에 사용한다.

## 7. 공유 파일 안내

### 학습 기준 및 검사 기록

| 파일 | 용도 |
|---|---|
| [fold_metadata.csv](../../data/processed/fold_metadata.csv) | 정답·경로·Split·Fold를 포함한 공통 목록 |
| [excluded_train_images.csv](../../data/processed/excluded_train_images.csv) | Train 품질 제외 목록 |
| [image_quality_audit.csv](../../data/processed/image_quality_audit.csv) | 이미지 읽기·크기·모드·밝기 검사 기록 |
| [image_alpha_audit.csv](../../data/processed/image_alpha_audit.csv) | LA 이미지 투명도 검사 기록 |

### 분석 결과

| 파일 | 용도 |
|---|---|
| [fold_summary.csv](tables/fold_summary.csv) | Fold별 이미지·환자 수 및 중복 수 |
| [train_label_summary.csv](tables/train_label_summary.csv) | Train 질환별 양성·음성 개수와 비율 |
| [fold_label_rates.csv](tables/fold_label_rates.csv) | 검증 Fold별 질환 양성 비율 |
| [train_cooccurrence.csv](tables/train_cooccurrence.csv) | Train 질환 라벨 동시 출현 개수 |

![Train 질환 라벨 동시 출현](figures/train_cooccurrence.png)

대각선은 해당 질환의 양성 이미지 수다.
대각선 밖은 두 질환 라벨이 함께 표시된 이미지 수다.

색상에는 log(1 + 개수)를 사용하며, 각 칸의 숫자는 실제 개수다.
동시 출현 횟수만으로 인과관계나 연관성의 강도를 판단하지 않는다.

## 8. 스크립트 역할

| 스크립트 | 역할 |
|---|---|
| `src/prepare_training_data.py` | 메타데이터 처리, 이미지 연결, 품질 표시, 환자 분할 및 Fold 배정 |
| `src/check_alpha_images.py` | LA 이미지 알파 검사 및 밝기 검사 결과 보완 |
| `src/review_image_quality.py` | Train의 밝기·대비 검토 후보 그림 생성 |
| `src/summarize_labels.py` | 질환별 개수와 Fold별 비율 집계 |
| `src/plot_cooccurrence.py` | 동시 출현 행렬 및 히트맵 생성 |

현재 작업 과정에서는 전체 이미지 검사를 먼저 실행하고,
저장한 검사 CSV를 이후 실행에서 재사용했다.

따라서 현재 실행 경로의 `prepare_training_data.py`는
저장된 `image_quality_audit.csv`를 전제로 할 수 있다.
원본 데이터만 있는 환경에서 검사 CSV까지 만드는 최초 실행 절차는
후속 작업에서 정리·검증할 예정이다.

공유된 검사 CSV와 원본 이미지가 준비된 Windows 환경에서
분할·요약 결과를 갱신하는 명령은 다음과 같다.

```powershell
.\.venv\Scripts\python.exe src/prepare_training_data.py
.\.venv\Scripts\python.exe src/summarize_labels.py
.\.venv\Scripts\python.exe src/plot_cooccurrence.py
```

기존 분할과 다른 결과가 감지되면 원인을 확인한다.
오류를 없애기 위해 기존 분할 파일을 임의로 삭제하지 않는다.

## 9. 남은 작업

- [ ] CLAHE 적용 전후 비교
- [ ] BBox를 활용한 병변 영역 비교 및 문서화
- [ ] 모델 입력용 이미지 변환 및 ImageNet 정규화
- [ ] DICOM 샘플 파싱 및 비식별 처리 실습
- [ ] 원본 데이터부터 결과물까지 최초 실행 절차 정리
- [ ] 모델 학습 및 교차검증
- [ ] 모델·설정 확정 후 고정 Test 최종 평가

현재 문서는 전처리 중간 진행 결과를 설명한다.
전체 전처리 또는 모델 학습이 완료됐음을 의미하지 않는다.