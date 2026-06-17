# Kaggle Predicting Stellar Class

Kaggle Playground Series S6E6 `Predicting Stellar Class` 대회를 위한 연구 저장소입니다.

목표는 천체 관측 데이터를 이용해 각 관측치를 `GALAXY`, `QSO`, `STAR` 세 class 중 하나로 분류하는 것입니다. 대회 평가지표는 balanced accuracy이므로, 전체 정확도뿐 아니라 세 class를 고르게 맞히는 것이 중요합니다.

## 프로젝트 방향

이 저장소는 두 가지 제출 방향을 분리해서 관리합니다.

1. **Generalization track**
   - 공개 제출 CSV를 사용하지 않고, train/test data와 모델 학습만으로 만든 제출입니다.
   - private leaderboard와 재현 가능한 모델 성능을 가장 중요하게 봅니다.

2. **Public + generalization track**
   - 공개적으로 접근 가능한 submission bank와 모델 확률을 분석해 public leaderboard 성능을 함께 고려한 제출입니다.
   - research-only reference output은 분석 신호로만 보고, 최종 제출 파일을 그대로 복사하는 방식은 사용하지 않습니다.

## 폴더 구조

```text
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── external_preds/
│   └── 공개 submission bank 또는 공개 모델 예측 파일
├── src/
│   └── stellar_features.py
├── scripts/
│   ├── train_lgbm_cv.py
│   ├── train_catboost_cv.py
│   ├── build_pure_model_ensemble.py
│   ├── analyze_pure_model_errors.py
│   ├── analyze_submission_bank.py
│   ├── build_bank_ridge_flip_candidates.py
│   ├── build_final_submission_tracks.py
│   └── make_research_dashboard.py
└── artifacts/
    ├── pure_model_ensemble/
    ├── pure_model_diagnostics/
    ├── bank_ridge_flip_v5/
    ├── final_submissions/
    ├── research_dashboard/
    └── submit_queue_clean/
```

`data/`, `external_preds/`, `artifacts/`는 용량이 크거나 대회 데이터가 포함될 수 있어 Git에 포함하지 않습니다.

## 주요 파일 설명

### `src/stellar_features.py`

모델에 사용할 feature engineering을 정의합니다. 원본 magnitude feature 외에도 color index, magnitude statistics, redshift interaction, sky coordinate encoding 등을 생성합니다.

### `scripts/train_lgbm_cv.py`

LightGBM 5-fold cross validation을 실행하고 OOF probability와 test probability를 저장합니다.

### `scripts/train_catboost_cv.py`

CatBoost 5-fold cross validation을 실행하고 OOF probability와 test probability를 저장합니다.

### `scripts/build_pure_model_ensemble.py`

LightGBM과 CatBoost의 OOF/test probability를 조합해 anchor-free pure ensemble submission을 만듭니다.

### `scripts/analyze_pure_model_errors.py`

순수 모델이 어떤 class, feature 구간, subset에서 주로 틀리는지 분석합니다. `pure_model_diagnostics/`에 confusion matrix, class recall, error pair, feature-bin error 등을 저장합니다.

### `scripts/analyze_submission_bank.py`

`external_preds/`에 있는 공개 submission bank를 분석합니다. 파일 형식, class 분포, bank consensus, row-level disagreement를 확인합니다.

### `scripts/build_bank_ridge_flip_candidates.py`

submission bank의 public score와 row-level disagreement를 이용해 ridge 기반 flip candidate를 추정합니다. public+generalization track 후보를 만드는 핵심 스크립트입니다.

### `scripts/build_final_submission_tracks.py`

최종 제출 후보 2개를 생성합니다.

```text
artifacts/final_submissions/final_generalization_model.csv
artifacts/final_submissions/final_public_generalization.csv
```

### `scripts/make_research_dashboard.py`

모델 성능, subset error, public probe 결과를 시각화하기 위한 research dashboard 산출물을 생성합니다.

## 기본 실행 순서

가상환경을 만든 뒤 필요한 패키지를 설치합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kaggle 대회 데이터를 `data/`에 넣습니다.

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

순수 모델을 학습합니다.

```bash
python scripts/train_lgbm_cv.py
python scripts/train_catboost_cv.py
python scripts/build_pure_model_ensemble.py
python scripts/analyze_pure_model_errors.py
```

공개 submission bank를 분석하고 public+generalization 후보를 만듭니다.

```bash
python scripts/analyze_submission_bank.py
python scripts/build_bank_ridge_flip_candidates.py
python scripts/build_final_submission_tracks.py
```

연구용 dashboard를 생성합니다.

```bash
python scripts/make_research_dashboard.py
```

## 최종 산출물

현재 최종 제출 후보는 다음 위치에 생성됩니다.

```text
artifacts/final_submissions/final_generalization_model.csv
artifacts/final_submissions/final_public_generalization.csv
```

`final_generalization_model.csv`는 모델 학습 기반 제출이고, `final_public_generalization.csv`는 public leaderboard 성능까지 고려한 제출입니다.

## 주의사항

- Kaggle competition data와 submission artifact는 Git에 올리지 않습니다.
- 공개 notebook 또는 공개 submission file은 연구용으로 분석할 수 있지만, 최종 output을 그대로 복사하는 방식은 사용하지 않습니다.
- public leaderboard 점수와 private/generalization 성능은 같은 문제가 아니므로, 두 track을 분리해서 판단합니다.
