# 2026-06-19 Single Model Import Notes

## 현재 판단

상위권 단일 모델 노트북을 확인한 결과, 우리 단일 모델 성능이 낮은 핵심 이유는 알고리즘 이름 자체가 아니라 feature 구조와 학습 구조 차이다.

현재 우리 직접 학습 결과:

- LightGBM: OOF 약 `0.965704`
- XGBoost: OOF 약 `0.965721`
- CatBoost 짧은 run: OOF 약 `0.965082`
- CatBoost long depth8: OOF 약 `0.965103`
- CatBoost long depth7: OOF 약 `0.965247`

상위권 단일 모델 기준:

- RealMLP v5: CV 약 `0.96904`
- CatBoost v3: CV 약 `0.96897`
- one-vs-rest XGB: CV 약 `0.96862`
- one-vs-rest TabM: CV 약 `0.96862`

따라서 지금 차이는 반복 수만으로 해결되지 않는다.

## 상위권 단일 모델에서 확인한 핵심 장치

1. 숫자를 구간값으로 다시 보기

`alpha`, `delta`, `u`, `g`, `r`, `i`, `z`, `redshift`를 그대로만 쓰지 않고 정수 구간, 분위수 구간, 조합 구간으로 바꾼다.

이유:

- 나무 모델과 신경망 모두 특정 경계 구간을 더 쉽게 잡는다.
- 별/은하/퀘이사 경계는 연속적인 숫자 크기보다 특정 색과 적색편이 구간에서 갈리는 경우가 많다.

2. 색 차이와 밝기 형태를 더 강하게 쓰기

중요 feature:

- `u-g`, `g-r`, `r-i`, `i-z`
- `u-r`, `g-i`, `r-z`
- `mag_mean`, `mag_range`
- `g / redshift`, `i / redshift`

이유:

- 천문 데이터에서는 단일 band 밝기보다 band 사이 차이가 물리적으로 더 직접적인 신호다.
- 같은 색이라도 redshift 위치에 따라 의미가 달라진다.

3. 위치와 밝기의 조합 범주 만들기

중요 조합:

- `alpha_floor x delta_floor`
- `u_floor x z_floor`

이유:

- 각각 따로 보면 약한 feature라도 조합하면 class별 밀집 구간이 생긴다.
- 상위권 RealMLP는 이 조합을 fold-safe target encoding 재료로 사용했다.

4. fold-safe target encoding

범주값마다 class 비율을 숫자로 바꾸는 방식이다.

주의:

- 전체 train 정답을 한 번에 보면 누수다.
- 반드시 fold 안에서만 통계를 만들고 validation에는 transform만 해야 한다.

현재 상태:

- 아직 우리 코드에 완전 이식하지 않았다.
- 이번 밤샘 run은 먼저 target을 쓰지 않는 non-leaky feature부터 넣는다.
- 다음 단계에서 fold-safe target encoding을 별도 helper로 추가한다.

5. one-vs-rest 구조

3개 클래스를 한 모델이 한 번에 맞히는 대신, class별로 “이 class인가 아닌가”를 따로 학습한다.

이유:

- `GALAXY`, `QSO`, `STAR`의 오류 경계가 서로 다르다.
- 특히 `GALAXY <-> STAR`, `QSO <-> GALAXY` 경계는 다른 모델이 필요할 수 있다.

현재 상태:

- CatBoost one-vs-rest는 이미 간단 구현이 있다.
- XGBoost/TabM one-vs-rest는 아직 직접 구현하지 않았다.
- 이번 밤샘 run에는 CatBoost one-vs-rest + RealMLP식 feature를 먼저 넣는다.

## CatBoost long run 해석

CatBoost long depth7이 가장 좋았다.

- OOF: `0.965247`
- depth8 long보다 약간 좋음
- 기존 짧은 run보다 약간 좋음

하지만 그래프상 검증 균형정확도는 중반 이후 거의 평평하다.

해석:

- 최고 지점 저장은 제대로 된다.
- 더 긴 반복만으로 `0.968+`까지 가기는 어렵다.
- 상위권 CatBoost v3와의 차이는 feature, 범주화, 외부 SDSS 낮은 가중치 학습 구조다.

## 오늘 밤 실험 목표

이번 run의 목적은 public 점수 올리기가 아니다.

목표:

1. RealMLP v5식 non-leaky feature를 공통 feature builder에 추가한다.
2. LightGBM, XGBoost, CatBoost에 같은 feature를 넣어 직접 비교한다.
3. CatBoost one-vs-rest에도 같은 feature를 넣는다.
4. 새로 만든 OOF/test probability를 기존 prediction bank와 함께 stacker에 넣는다.
5. 결과는 OOF, fold 안정성, class recall, 추후 SDSS external 기준으로 판단한다.

## 다음 단계

밤샘 run 이후 확인할 것:

- `artifacts/lgbm_cv_realmlp_features/lgbm_baseline_report.json`
- `artifacts/xgboost_cv_realmlp_features/report.json`
- `artifacts/catboost_cv_realmlp_features/catboost_baseline_report.json`
- `artifacts/ovr_catboost_realmlp_features/report.json`
- `artifacts/available_prediction_stacker_realmlp_feature_bank_c010/report.json`

승격 기준:

- OOF가 기존 reliable anchor `0.9703513342698414`를 넘거나,
- OOF가 비슷하더라도 fold 안정성과 class recall 균형이 좋아야 한다.

보류 기준:

- OOF가 올라도 특정 class recall이 크게 깨지는 경우
- SDSS external에서 확실히 악화되는 경우
- public LB만 좋아지고 OOF 근거가 약한 경우

