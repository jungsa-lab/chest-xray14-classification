# 프로젝트 구조 정리 기록

## 변경 목적

전처리·분석·DICOM 실습·학습 입력 코드가 섞인 `src/`를 역할별로 나눈다.
모델 학습과 FastAPI/Streamlit 기능은 이후 같은 데이터 변환·질환 순서를 재사용한다.

## 주요 변경

- `src/paths.py`에 루트 경로를 정의해 하위 폴더 이동에 따른 `parents[1]` 오류를 해소.
- `src/constants.py`에 기존 DISEASES 순서를 그대로 이동. 실행용 전처리 파일에 대한 import 의존 제거.
- 절대 패키지 import와 프로젝트 루트의 `python -m ...` 실행으로 통일.
- `summarize_labels.py`를 `src/analysis/`로 이동.
- 팀원 초기 전처리와 일회성 LA 보완 스크립트는 원문을 `legacy/`에 보관.
- 첫 실행에 이미지 검사표를 생성하고 이후에는 재사용. 명시적 재검사 옵션 추가.
- 캐시의 필수 열·대상·경로·읽기/크기 성공·유한한 밝기 측정값을 검사.
- 품질 후보 분석은 최종 fold_metadata의 전체 Train(제외 2장 포함)을 읽도록 변경.
- 품질 후보 결과는 reports에 저장. 과거 data/processed의 후보 결과는 자동 삭제하지 않음.
- 보고서·루트 README의 경로, 진행 상태, 분할 기준을 현재 구현으로 정리.
- 누락된 CLAHE/BBox/DICOM 결과물을 .gitignore 허용 목록에 추가.
- requirements에 opencv-python, pydicom을 추가. 과거 환경 스냅샷을 UTF-8로 변환.

## 점검 스크립트

입력 ZIP에는 이동한 `scripts/`가 포함되지 않았다. 따라서 제공된 세 점검 파일은
현재 Dataset/Transform 인터페이스에 맞춰 새로 작성한 기본 점검본이다.
설치기는 기존 파일이 있으면 먼저 백업하며, ZIP에 없던 다른 파일은 유지한다.

## 유지한 조건

14개 질환 순서, 환자 분할 seed=42 및 Test 15%, GroupKFold 5개,
Train 품질 기준 mean<5/mean>250/std<3, CLAHE clip=2/grid=8,
ImageNet 평균·표준편차, 기존 분할·Fold 변경 감지 조건을 유지한다.
실제 데이터·CSV·이미지·모델 가중치는 이 정리 패키지에 포함하지 않는다.

## 과거 파일 처리

`legacy/`의 스크립트는 참고 기록이며 이동 후 실행 경로를 지원하지 않는다.
현재 전처리 결과와 다른 70/10/20 분할을 만드는 초기 코드를 혼용하지 않는다.
과거 `image_alpha_audit.csv`는 당시 LA 검사 근거로 유지한다.
`data/raw/pretrained_model.h5`는 이 작업에서 사용처를 확인하지 못했으므로 삭제하지 않았다.

## 적용 후 확인

먼저 `scripts.checks.check_torch`, `check_image_transforms`, `check_dataloader`를 실행한다.
Fold 0의 기존 기준은 Train 76,315장 / Val 19,079장이다.
그다음 요약·동시 출현·CLAHE 등 필요한 보고서만 재생성한다.
Windows의 실제 데이터와 CUDA 검증은 해당 PC에서 수행해야 한다.
