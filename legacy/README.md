# 과거 구현 보관

현재 실행 파이프라인은 `python -m src.preprocessing.prepare_training_data`입니다.

- `preprocess_data.py`: 팀원 초기 전처리 구현. 기존 70/10/20 분할을 사용하며 현재 85/15 + Train 5-Fold와 다릅니다. 참고용으로 원문을 보존했습니다.
- `check_alpha_images.py`: 초기 검사 CSV에서 LA 519장의 밝기를 보완했던 일회성 작업입니다. 현재 통합 검사에서도 불투명 LA를 처리하므로 재실행할 필요가 없습니다.

두 파일은 위치 이동에 따른 경로 수정을 하지 않은 과거 기록입니다. 이 폴더에서 실행하지 마세요.
기존 `image_alpha_audit.csv`는 과거 검사 근거로 보존하며 새 전처리 실행의 필수 입력은 아닙니다.
