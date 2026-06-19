# Private Generalization Research Snapshot

작성 시점: 2026-06-19

이 폴더는 Kaggle Playground S6E6 `Predicting Stellar Class`에서 public LB가 아니라 private/generalization 성능을 기준으로 연구한 내용을 따로 고정 기록하기 위한 공간이다.

이 폴더는 사용자가 명시적으로 요청할 때만 다시 정리한다.

## 현재 결론

1. Public LB 전용 row patch, boundary patch만 계속 밀면 private/generalization 근거가 약하다.
2. 현재 SDSS external validation에서는 CatBoost 단일 모델이 가장 강했다.
3. OOF 기준 최고 후보는 `19_PRIVATE_CV_guarded...`지만, 이것은 Kaggle test CSV 후보라 SDSS external에 직접 채점할 수 없다.
4. 따라서 다음 연구는 `OOF/test proba를 생산하는 실제 모델`을 추가하고, stacker에 넣은 뒤 OOF, fold 안정성, class recall, SDSS external을 함께 보는 방식으로 진행한다.

## 왜 XGBoost, LightGBM, CatBoost부터 다시 하는가

우리가 받은 상위권 stacker 이미지에는 더 많은 모델이 있었다.

- RealMLP
- TabM
- TabICL
- NN
- Logistic Regression
- XGBoost 여러 버전
- LightGBM 여러 버전
- CatBoost 여러 버전

하지만 이들 중 상당수는 다른 참가자가 만든 OOF/test prediction bank 형태로 확보한 재료였다. 즉, 우리가 직접 학습 과정을 재현하고 train/valid curve, fold별 BAC, SDSS external을 같이 볼 수 있는 상태는 아니었다.

그래서 현재 우선순위는 다음과 같다.

1. 직접 재현 가능한 GBDT 3종을 같은 진단 포맷으로 만든다.
2. XGBoost, LightGBM, CatBoost 각각의 OOF/test proba와 diagnostic curve를 확보한다.
3. 이 재료를 기존 `07_lr_v9` stacker와 합쳐서 OOF-only stacker를 다시 만든다.
4. 그 뒤 RealMLP, TabM, TabICL 계열은 직접 학습 가능성 또는 확보한 OOF/test proba의 품질을 따로 평가해서 추가한다.

## 현재 구현 상태

세 모델 모두 아래 형식으로 저장되도록 맞췄다.

- OOF probability
- test probability
- submission CSV
- fold score CSV
- train/valid loss diagnostic CSV
- train sample BAC / valid BAC diagnostic CSV
- logloss curve SVG
- balanced accuracy curve SVG
- prediction iteration policy

지원하는 prediction iteration policy:

- `logloss-best`
- `valid-bac-best`
- `fixed`

## 현재 가장 중요한 실험 질문

1. XGBoost를 추가하면 `07_lr_v9`보다 OOF/CV가 오르는가?
2. 오른다면 SDSS external에서도 악화되지 않는가?
3. CatBoost가 SDSS external에서 강한 이유가 private에도 연결될 수 있는가?
4. LGBM은 내부 OOF 대비 external STAR recall이 약한데, stacker에서 비중을 낮추는 것이 맞는가?
5. boundary/row patch는 OOF 개선이 있어도 external에서 약해졌으므로, 최종 private 후보에서는 제한적으로만 써야 하는가?

