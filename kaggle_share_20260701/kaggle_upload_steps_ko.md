# Kaggle 공유 순서

1. `input_dataset/dataset-metadata.json`의 `YOUR_KAGGLE_USERNAME`을 실제 Kaggle
   사용자명으로 바꿉니다.
2. `input_dataset` 폴더를 새 Kaggle Dataset으로 업로드합니다.
3. `27th-place-robust-oof-selection.ipynb`를 Kaggle Code에 업로드합니다.
4. Code 입력으로 대회 데이터와 2번에서 만든 Dataset을 추가합니다.
5. Internet은 끄고 CPU 환경에서 Run All을 실행합니다.
6. 출력에서 OOF 0.9706589608, train 변경 3행, test 변경 1행,
   positive fold rate 1.0이 나오는지 확인합니다.
7. Code 설명에는 `kaggle_code_description_en.md`를 사용합니다.
8. Discussion에는 `kaggle_discussion_post_en.md`를 붙여 넣고, 게시한 Code 링크를
   본문에 추가합니다.

이 노트북은 최종 선택 단계를 재현합니다. 모든 1단계 모델을 처음부터 재학습하는
노트북이라고 설명하면 안 됩니다. 사전 계산 OOF/test 확률을 사용한다는 사실을
제목 아래와 글 본문에 그대로 유지해야 합니다.
