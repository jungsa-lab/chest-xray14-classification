# 데이터 전처리 진행 현황 및 결과 안내

## 1. 목적과 현재 범위

NIH ChestX-ray14 데이터의 메타데이터와 이미지 상태를 검사하고,
환자 단위로 학습·검증·최종 평가 데이터를 구분하였다.

팀원별 모델 실험은 동일한 `fold_metadata.csv`를 기준으로 수행한다.
원본 이미지 데이터는 저장소에 포함하지 않는다.

메타데이터·이미지 검사, 환자 분할, CLAHE 비교, ImageNet 정규화 및 DICOM 샘플 실습을 수행했다.
모델 학습·성능 평가는 아직 수행하지 않았다. 구조 정리본의 Windows 실행은 적용 후 확인한다.

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
- 모델의 질환 정답 열은 `src/constants.py`의
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
| `src/preprocessing/prepare_training_data.py` | 메타데이터 처리, 이미지 연결, 품질 표시, 환자 분할 및 Fold 배정 |
| `legacy/check_alpha_images.py` | 과거 LA 검사표 보완 기록. 현재 파이프라인에서는 실행하지 않음 |
| `src/analysis/review_image_quality.py` | Train의 밝기·대비 검토 후보 그림 생성 |
| `src/analysis/summarize_labels.py` | 질환별 개수와 Fold별 비율 집계 |
| `src/analysis/plot_cooccurrence.py` | 동시 출현 행렬 및 히트맵 생성 |

현재 전처리 실행은 검사 CSV가 없으면 전체 이미지를 검사하고, 있으면 재사용한다.
파일명·경로·읽기 성공·크기·밝기 측정값을 검사한 뒤 병합한다.
동일한 이름과 경로로 파일 내용만 교체한 경우는 캐시 비교로 감지하지 못한다.
원본을 교체했거나 재검사가 필요할 때만 `--refresh-image-audit` 옵션을 사용한다.
기존 환자 분할과 Fold가 바뀌면 중단하는 검사는 유지한다.

공유된 검사 CSV와 원본 이미지가 준비된 Windows 환경에서
분할·요약 결과를 갱신하는 명령은 다음과 같다.

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.prepare_training_data
.\.venv\Scripts\python.exe -m src.analysis.summarize_labels
.\.venv\Scripts\python.exe -m src.analysis.plot_cooccurrence
```

기존 분할과 다른 결과가 감지되면 원인을 확인한다.
오류를 없애기 위해 기존 분할 파일을 임의로 삭제하지 않는다.

## CLAHE 적용 전후 비교

### 비교 방법

- 설정: clipLimit=2.0, tileGridSize=(8, 8)
- 품질 검사를 통과한 Train에서만 비교 대상을 선정했다.
- Nodule BBox가 있는 고유 Train 환자 65명 중
  서로 다른 환자 4명을 random_state=42로 선택했다.
- NIH 배포 영상의 1024×1024 좌표를 224×224에 맞게 환산했다.
- 좌표 변환은 전체 영상이 자르기 없이 축소됐다는 전제에 따른다.
- 전체 영상에 CLAHE를 적용한 다음, 원본과 처리 결과에서
  동일한 BBox 주변 영역을 잘라 비교했다.
- 모든 영상은 동일한 밝기 표시 범위(0~255)를 사용했다.
- 원본 이미지 파일은 수정하지 않았다.

### 비교 그림

전체 영상 비교:

![CLAHE 전체 영상 비교](figures/clahe_comparison.png)

BBox 및 관심 영역 비교:

![CLAHE BBox 비교](figures/clahe_bbox_comparison.png)

노란 사각형은 데이터셋의 Nodule 주석이며 모델 예측 결과가 아니다.

### 관찰 결과

일부 예시에서는 주석 영역 내부의 밝은 부분과 주변 배경의
명암 차이가 증가하여 시각적으로 구분하기 쉬워졌다.

그러나 모든 예시에서 대상이 명확해진 것은 아니며,
주변 구조와 영상의 거친 질감도 함께 강조됐다.

### 해석의 한계

- 4개 예시에 대한 정성적 관찰이며 전체 데이터의 효과를 보장하지 않는다.
- 224×224 영상의 확대는 소실된 세부 정보를 복원하지 않는다.
- BBox 위치의 정확성은 다운로드 데이터의 기하학적 변환 방식에 의존한다.
- 임상적 검출 능력이나 모델 성능 향상을 입증한 결과는 아니다.
- 모델 성능에 미치는 영향은 이후 동일한 분할을 사용한
  교차검증으로 평가해야 한다.

### 재현 자료

- 전체 영상 비교: `src/analysis/compare_clahe.py`
- BBox 비교: `src/analysis/compare_clahe_bbox.py`
- [전체 영상 비교 대상·설정](tables/clahe_samples.csv)
- [BBox 비교 대상·좌표·설정](tables/clahe_bbox_samples.csv)


## DICOM 샘플 처리 실습

### 대상과 목적

pydicom 3.0.2의 공개 테스트 샘플 `CT_small.dcm`을 사용했다.
DICOM의 메타데이터 접근, 픽셀 추출 및 식별 관련 태그 제거를
실습하기 위한 자료이며, NIH 흉부 X-ray 학습 데이터에는 포함하지 않았다.

### 파싱 결과

| 항목 | 결과 |
|---|---|
| 촬영 방식 | CT |
| 영상 크기 | 128×128 |
| 픽셀 자료형 | int16 |
| Photometric Interpretation | MONOCHROME2 |
| Transfer Syntax | Explicit VR Little Endian |
| 저장 픽셀 범위 | 128~2191 |

### 메타데이터 제거 결과

원본을 메모리에 복사한 뒤, 샘플의 영상 처리에 필요한
허용 목록에 없는 태그를 제거했다.

| 항목 | 결과 |
|---|---|
| 최상위 태그 수 | 258 → 14 |
| Private 태그 수 | 179 → 0 |
| 확인 대상 식별 관련 태그 | 모두 제거 |
| 처리 전후 픽셀 값 | 동일 |
| 원본 파일 수정 | 수행하지 않음 |
| 처리된 DICOM·영상 파일 저장 | 수행하지 않음 |

Private 태그가 모두 개인정보를 의미하는 것은 아니지만,
이번 실습에서는 보수적으로 제거했다.

### 로컬 영상 확인

메타데이터의 Modality LUT 또는 Rescale 정보를 반영한 후,
전체 값 범위 및 제한된 표시 범위로 샘플 영상을 확인했다.

표시한 영상에서 이름·환자번호·날짜 등의 식별 문자는
육안으로 관찰되지 않았다. 영상은 파일로 저장하지 않았다.

### 범위와 한계

본 작업은 특정 공개 테스트 샘플에 대한 파싱·태그 제거 실습이다.
범용 DICOM 비식별화 도구나 DICOM 비식별화 표준 준수 검증을
구현한 것은 아니다.

육안 확인만으로 픽셀 내 식별 정보의 부재를 완전히 보장하지 않는다.
따라서 완전한 비식별화 검증 상태는 미완료로 유지한다.

필수 UID 등도 제거하므로 처리 객체를 의료 시스템에서 사용할
DICOM 파일로 저장하는 용도로 사용하지 않는다.

### 실행 스크립트

- `src/dicom/inspect_dicom.py`: 메타데이터 및 픽셀 파싱
- `src/dicom/deidentify_dicom_sample.py`: 허용 목록 기반 태그 제거 및 검사
- `src/dicom/view_dicom_sample.py`: 로컬 영상 표시

## 9. 완료 및 남은 작업

- [x] CLAHE 적용 전후 비교
- [x] BBox를 활용한 병변 영역 비교 및 문서화
- [x] 모델 입력용 이미지 변환 및 ImageNet 정규화
- [x] DICOM 샘플 파싱 및 비식별 처리 실습
- [x] 최초 실행·검사 캐시 재사용 절차 구현 및 문서화
- [ ] 구조 정리본을 Windows 실제 데이터로 실행 확인
- [ ] 모델 학습 및 교차검증
- [ ] 모델·설정 확정 후 고정 Test 최종 평가

현재 문서는 실행 로그와 관찰로 확인한 전처리 결과를 설명한다.
DICOM 실습은 완전한 비식별화 검증을 의미하지 않으며 모델 성능 결과는 아직 없다.

## 모델 입력 및 점검

`src/data/image_transforms.py`에서 전체 영상의 224×224 크기, 선택적 CLAHE,
회색조 3채널 복제 및 ImageNet 평균/표준편차 정규화를 적용한다.
`src/data/chest_xray_dataset.py`가 이 변환과 14개 정답을 연결한다.

```powershell
.\.venv\Scripts\python.exe -m scripts.checks.check_torch
.\.venv\Scripts\python.exe -m scripts.checks.check_image_transforms
.\.venv\Scripts\python.exe -m scripts.checks.check_dataloader
```

기존 `image_alpha_audit.csv`는 과거 519장 보완 검사 기록이다. 현재 전체 이미지
검사는 LA 불투명 여부를 직접 확인하므로 이 별도 CSV가 없어도 실행할 수 있다.
DICOM JSON의 `pixel_identifiers_reviewed: false`는 스크립트가 픽셀 식별 문자 검사를
수행하지 않았음을 뜻한다. 위의 육안 관찰 기록과 자동 검증 결과를 구분한다.
