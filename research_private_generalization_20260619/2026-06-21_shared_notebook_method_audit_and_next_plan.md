# Shared Notebook Method Audit And Kaggle Next Plan

작성 시점: 2026-06-21

목적은 공유받은 상위권 노트북과 prediction bank에서 얻은 기법을 다시 고정 정리하고, 우리 Kaggle repo에 무엇이 이미 들어갔고 무엇이 아직 빠졌는지 헷갈리지 않게 남기는 것이다.

## 현재 기준

- 우선순위는 private/generalization 성능이다.
- public LB 점수는 참고값이다. public이 내려간다고 private이 반드시 내려간다고 보지 않는다.
- 최종 선택은 OOF/CV, fold 안정성, class recall, SDSS external stress-test, source diversity를 같이 본다.
- 단순 submission CSV 복사는 연구가 아니다. OOF/test proba를 생산하거나, OOF 근거로 stacker와 optimizer에서 채택되는 방식이어야 한다.

## 공유 파일별 핵심 기법

### realmlp-v5-for-s6e6

- 5-fold RealMLP 단일 모델.
- 공개 설명 기준 local CV는 약 0.969대.
- 핵심은 단순 MLP가 아니라 RealMLP식 tabular neural net 구조다.
- `n_ens=8` ensemble head를 쓴다.
- categorical embedding 또는 one-hot을 feature별로 다르게 처리한다.
- numerical feature에 PBLD/periodic style embedding을 쓴다.
- dropout, class weight, sample weight power, robust preprocessing 설정이 들어간다.
- raw numerical을 floor category로 바꾼다.
- `delta` quantile bin을 만든다.
- `alpha_floor x delta_floor`, `u_floor x z_floor` interaction category를 만든다.
- 위 interaction category에 대해 fold-safe multiclass TargetEncoder를 쓴다.
- TargetEncoder는 반드시 fold 내부 train으로만 fit되어야 한다.

### cat-v3-for-s6e6

- 5-fold CatBoost 단일 모델.
- 핵심은 CatBoost native categorical feature를 매우 많이 만들어 넣는 것이다.
- floor category, rounded category, fractional/decimal category, quantile category를 쓴다.
- color index, magnitude, redshift 기반 interaction category를 만든다.
- class weight는 STAR/QSO를 더 강하게 주는 방향이다.
- 외부 SDSS 원본 데이터 `star_classification.csv`를 낮은 weight로 추가 학습하는 구조가 있다.
- 외부 데이터는 private 일반화에 도움이 될 수 있지만, weight가 너무 크면 competition 분포와 어긋날 수 있다.

### ps6e6-one-vs-rest-xgb

- multiclass XGBoost 하나가 아니라 class별 one-vs-rest 이진 모델 구조다.
- 각 class boundary를 따로 학습한다.
- `TargetEncoder(cv=5)`, `KBinsDiscretizer`, floor/category view를 쓴다.
- class별 binary prediction을 다시 LogisticRegressionCV로 섞는다.
- 상위권 XGB는 단순 multiclass XGB가 아니라 이 구조가 핵심이다.

### ps6e6-one-vs-rest-tabm

- class별 one-vs-rest TabM 구조다.
- `TabM_D_Classifier`, `tabm_k=32`, PWL numerical embedding, `d_embedding=12`, dropout을 쓴다.
- TabM 자체는 아직 우리 로컬 직접 학습 코드에 없다.
- 기존 prediction bank의 TabM OOF/test는 stacker 재료로 사용 중이다.

### gpu-logistic-regression-stacker

- OOF/test prediction bank를 입력으로 받는 multinomial logistic regression stacker다.
- raw probability를 logit feature로 변환해서 쓴다.
- balanced accuracy 대회라 class weight와 STAR boost를 둔다.
- 5-fold OOF stacker로 OOF score를 직접 확인한다.
- 모델별 coefficient importance와 confusion matrix를 본다.
- 이 방식은 public 점수만 보고 blend하는 것이 아니라 OOF 기반 stacker라는 점이 중요하다.

### logistic-regression-stacker-43-models-0-97105

- 같은 logistic stacker 구조를 43개 source로 확장한 버전이다.
- 여러 RealMLP, TabM, XGB, CatBoost, LGBM, NN source를 사용한다.
- 핵심은 source 수 자체가 아니라 source diversity와 OOF 기반 coefficient 선택이다.
- 우리 쪽에서는 모든 source를 직접 재현하지 않았고, 확보한 OOF/test pair만 사용 중이다.

### s6e6-weighted-blend-meta-stacker

- base model OOF/test를 모아 weight를 최적화한다.
- `scipy.optimize.minimize` 계열로 balanced accuracy가 높은 weight를 탐색한다.
- LogisticRegression/LGBM level-2 meta stacker도 실험한다.
- class bias 또는 class weight 보정도 같이 본다.
- 우리 쪽에는 greedy OOF optimizer가 있지만, constrained weighted blend optimizer는 아직 production path로 완성하지 않았다.

### stellar-catb-hgbc-xgb-lgbm-realmlp-baseline

- 최고점 solution이라기보다 잘 정리된 실험 template다.
- physics-based feature engineering이 많다.
- color index, magnitude 통계, spectral curvature, redshift transform, redshift interaction, sky coordinate encoding을 만든다.
- preprocessing을 fold 안에서 fit하는 구조가 좋다.
- categorical/bucket별 target statistics를 만들 수 있게 구성되어 있다.
- subset별 error analysis, train/valid/test distribution graph가 핵심 참고 포인트다.

## 우리 repo에 이미 들어간 것

- `src/stellar_features.py`
  - base/advanced feature.
  - RealMLP-style non-leaky feature.
  - floor category, delta quantile bin, `alpha_floor_x_delta_floor`, `u_floor_x_z_floor`.
- `scripts/train_lgbm_foldsafe_te_cv.py`
  - fold-safe multiclass target encoding LightGBM.
  - RealMLP interaction category 기반 TE.
- `scripts/train_catboost_cv.py`
  - CatBoost 5-fold, valid-BAC early stopping, prediction iteration policy, train/valid diagnostic graph.
- `scripts/train_xgboost_cv.py`
  - XGBoost 5-fold, train/valid diagnostics.
- `scripts/train_ovr_xgboost_cv.py`
  - one-vs-rest XGBoost 계열 구현.
- `scripts/train_ovr_catboost_cv.py`
  - one-vs-rest CatBoost 계열 구현.
- `scripts/train_classwise_logistic_blender.py`
  - class-wise logistic blender.
- `scripts/build_available_prediction_stacker.py`
  - 확보된 OOF/test prediction pair를 자동 로드해서 logistic stacker 생성.
- `scripts/optimize_oof_generalization_stack.py`
  - public LB 없이 OOF 기준 greedy blend와 class bias 탐색.
- `scripts/build_private_cv_stable_submissions.py`
  - OOF 개선 row를 guard 조건으로 제한해 private 후보 CSV 생성.

## 현재 주요 수치

- `07_lr_stacker_v9_direct`: OOF 0.970279, public 0.97101.
- `35_PRIVATE_CV_greedy_cat_realmlp_plus_realmlp0`: OOF 0.970479, public 0.97104.
- `51_PRIVATE_CV_greedy_with_foldsafe_te`: OOF 0.970528.
- `56-63_PRIVATE_CV_te_disagree...`: OOF 0.970573 계열, public 확인값 중 하나는 0.97097.
- `catboost_cv_realmlp_features`: OOF 0.969746. 우리 직접 학습 단일 모델 중 핵심 성과.
- `lgbm_foldsafe_te_realmlp`: OOF 0.967900. 단일로는 낮지만 stacker 후보로는 검증됨.
- `ovr_xgboost_realmlp_features`: OOF 0.964203. 현재 설정으로는 직접 성능이 낮아 우선순위 낮음.

## 아직 제대로 안 넣은 것

- RealMLP 직접 학습.
  - 기존 RealMLP OOF/test source는 사용하지만, RealMLP v5 구조 자체를 재학습하지 않았다.
- TabM 직접 학습.
  - 기존 TabM OOF/test source는 사용하지만, `TabM_D_Classifier`, PWL embedding, `tabm_k=32` 직접 학습은 없다.
- CatBoost v3 full categorical view.
  - RealMLP floor/quantile combo 일부만 있었다.
  - rounded/frac/quantile/category view와 color/redshift interaction category가 부족했다.
- CatBoost v3 external SDSS low-weight 학습.
  - SDSS external validation은 했지만, CatBoost train pool에 external을 낮은 weight로 붙이는 구조는 아직 없다.
- target statistics feature bank.
  - fold-safe TE는 구현했지만, baseline notebook의 category/bucket별 mean/std/skew/median/min/max/count 통계 bank는 아직 없다.
- OOF source correlation selector.
  - source별 단일 점수와 greedy gain은 보지만, class별 correlation/diversity를 정식 선택 기준으로 쓰지는 않는다.
- constrained weighted blend optimizer.
  - greedy OOF optimizer는 있지만, Nelder-Mead/simplex 기반 weight optimizer는 아직 없다.
- class-wise logit intercept calibration.
  - balanced accuracy용 bias search는 있으나, transition/subset별 안정적인 logit intercept calibration은 아직 실험 단계다.

## 이번에 추가한 구현

- `src/stellar_features.py`에 `catv3` feature set을 추가한다.
- target을 보지 않는 categorical view만 먼저 추가한다.
- 추가 feature:
  - raw numeric rounded category.
  - raw numeric fractional category.
  - raw/color/magnitude/redshift quantile bins.
  - redshift sign category.
  - color-redshift, magnitude-redshift, sky-color compact interaction category.
- `scripts/train_catboost_cv.py`에서 `--feature-set catv3`를 받을 수 있게 한다.
- `scripts/build_available_prediction_stacker.py`가 `artifacts/catboost_cv_catv3_views`를 자동 source로 읽도록 한다.

## 다음 구현 우선순위

1. CatBoost catv3 categorical-view 5-fold 학습.
2. 해당 OOF/test proba를 available prediction stacker와 OOF greedy optimizer에 추가.
3. OOF gain, fold 안정성, class recall, changed-row subset 진단 확인.
4. 괜찮으면 private guarded candidate CSV 생성.
5. 다음 라운드에서 CatBoost v3 external SDSS low-weight 학습을 추가.
6. 그 다음 constrained weighted blend optimizer와 source correlation selector를 구현.
7. RealMLP/TabM 직접 학습은 dependency와 시간 비용이 크므로, 먼저 기존 OOF/test source의 contribution을 완전히 사용한 뒤 진행.

