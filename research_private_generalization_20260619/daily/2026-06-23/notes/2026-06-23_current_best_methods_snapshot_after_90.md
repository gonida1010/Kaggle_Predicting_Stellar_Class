# Current Best Methods And Hyperparameters

작성일: 2026-06-22

목적은 public leaderboard 점수만 올리는 것이 아니라, OOF/CV 기준으로 private/generalization 후보를 단단하게 만드는 것이다. 아래 내용은 지금까지 연구하면서 실제로 의미가 있었던 설정과 알고리즘만 정리한 것이다.

## 현재 최고 수치

| 구분 | 최고 수치 | 근거 |
|---|---:|---|
| 순수 직접 학습 단일 모델 OOF | 0.969746 | CatBoost + RealMLP-style feature |
| OOF stacker 기준 최고 | 0.970528 | class-wise blender + CatBoost RealMLP feature + RealMLP source |
| 이전 private/CV 후보 최고 | 0.970573 | TE disagreement patch 56번 |
| 현재 private/CV 후보 최고 | 0.970603 | research material stack 68번 |
| 56번 public 확인값 | 0.97097 | 56번 제출 결과 |

현재 최고 후보:

- `outputs/68_PRIVATE_CV_research_material_stack_oof0970603.csv`
- `outputs/56_PRIVATE_CV_te_disagree_high_gi_low_rz_base_galaxy_to_star_c0_55_m0_15_b0_60_oof0970573.csv`

56번 OOF class report:

| class | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| GALAXY | 0.990829 | 0.959971 | 0.975156 | 377480 |
| QSO | 0.951441 | 0.977301 | 0.964198 | 117143 |
| STAR | 0.882952 | 0.974445 | 0.926445 | 82724 |
| macro avg | 0.941741 | 0.970573 | 0.955266 | 577347 |
| weighted avg | 0.967380 | 0.965561 | 0.965953 | 577347 |

## 가장 중요한 학습 판단 기준

1. 최종 판단은 logloss가 아니라 validation balanced accuracy와 OOF balanced accuracy로 한다.
2. logloss는 계속 떨어질 수 있다. 하지만 balanced accuracy는 중간에 plateau 또는 하락할 수 있다.
3. CatBoost, LightGBM, XGBoost 모두 최종 후보 저장 기준은 best validation balanced accuracy fallback이어야 한다.
4. public LB가 낮아져도 OOF/CV가 올라가면 private 후보로 유지한다.
5. 단, class recall이 한쪽으로 무너지거나 test 변경 row가 과도하면 guarded 후보로 제한한다.

## 1. CatBoost + RealMLP-style Feature

현재 직접 학습 단일 모델 중 최고다.

| 항목 | 값 |
|---|---:|
| OOF balanced accuracy | 0.969746 |
| fold 수 | 5 |
| feature set | `realmlp` |
| loss | `MultiClass` |
| eval metric | `CatBoostBalancedAccuracyMetric` |
| iterations | 9000 |
| learning rate | 0.03 |
| depth | 7 |
| l2_leaf_reg | 12 |
| random_strength | 1.0 |
| bagging_temperature | 0.55 |
| class weight | `auto_class_weights=Balanced` |
| seed | 20260611 |

fold별 OOF:

| fold | balanced accuracy |
|---:|---:|
| 1 | 0.968947 |
| 2 | 0.970799 |
| 3 | 0.970103 |
| 4 | 0.969856 |
| 5 | 0.969027 |

해석:

- RealMLP-style feature가 가장 큰 성능 상승 원인이었다.
- 기존 CatBoost 단순 feature보다 훨씬 강하다.
- 다만 기존 full run은 logloss 기준으로 오래 끌고 갈 위험이 있었다.
- 앞으로는 같은 feature를 쓰더라도 validation balanced accuracy fallback 방식으로 다시 학습하는 것이 맞다.

## 2. CatBoost v3-style Categorical View

현재 진행 중인 다음 핵심 방향이다. 아직 최고 후보로 완료된 것은 아니지만, 상위권 CatBoost v3 방식과 우리 연구 방향을 합친 설정이다.

| 항목 | 값 |
|---|---:|
| feature set | `catv3` |
| feature count | 157 |
| max iterations | 4500 |
| chunk size | 150 |
| BAC patience chunks | 6 |
| learning rate | 0.030 |
| depth | 7 |
| l2_leaf_reg | 12 |
| random_strength | 1.0 |
| bagging_temperature | 0.55 |
| class weight | `auto_class_weights=Balanced` |
| early stop 기준 | validation balanced accuracy |

핵심 feature view:

- raw numeric rounded category
- raw numeric fractional category
- raw, color, magnitude, redshift quantile bins
- redshift sign category
- color-redshift interaction category
- magnitude-redshift interaction category

중요:

- CatBoost 내부 `bestIteration=149`는 chunk 내부 local iteration이다.
- 실제 판단은 우리 로그의 `iter=150`, `iter=300`, `best_bac=...@global_iteration`을 본다.
- 상위권 CatBoost v3는 fold별 best iteration이 대략 1200~2500 근처였다.
- 따라서 chunked run은 최소 1200~2500 부근까지는 확인해야 한다.

## 3. Fold-safe Target Encoding LightGBM

단독 성능은 최고는 아니지만, disagreement patch에서 의미 있는 신호를 만들었다.

| 항목 | 값 |
|---|---:|
| OOF balanced accuracy | 0.967900 |
| fold 수 | 5 |
| learning rate | 0.028 |
| num_leaves | 96 |
| max_depth | -1 |
| min_child_samples | 90 |
| subsample | 0.88 |
| colsample_bytree | 0.84 |
| reg_alpha | 0.1 |
| reg_lambda | 2.4 |
| class_weight | `balanced` |
| n_estimators | 5000 |

핵심 원칙:

- target encoding은 반드시 fold 내부 train으로만 fit한다.
- validation fold와 test에는 transform만 적용한다.
- 잘못 넣으면 target leakage가 생기므로, 전체 train으로 fit하면 안 된다.

현재 역할:

- 전체 예측을 그대로 stack하면 중복 신호가 많다.
- 대신 기존 stacker와 disagreement가 나는 특정 row를 찾는 데 유용했다.
- 56번 최고 후보는 이 모델의 disagreement 신호에서 나왔다.

## 4. OOF Generalization Greedy Stack

현재 private/CV 스태킹 핵심이다.

accepted stages:

| 단계 | source | weight | OOF |
|---|---|---:|---:|
| base raw | `lr-stacker-v9-public-oof` | 0.00 | 0.970279 |
| base bias | `lr-stacker-v9-public-oof` | 0.00 | 0.970324 |
| blend round 1 | `our-classwise-logistic-blender` | 0.35 | 0.970456 |
| blend round 2 | `our-catboost-realmlp-features` | 0.281944 | 0.970511 |
| blend round 3 | `realmlp-2` | 0.029167 | 0.970528 |

현재 해석:

- 공개 OOF prediction bank를 무작정 쓰는 것이 아니라 OOF gain이 있는 source만 채택한다.
- CatBoost RealMLP feature source는 실제로 OOF를 올렸다.
- RealMLP source는 낮은 weight로만 추가되는 것이 안정적이었다.
- class-wise blender는 QSO/STAR recall을 보강하지만 GALAXY recall을 과하게 깎을 수 있으므로 guard가 필요하다.

## 5. TE Disagreement Patch

현재 최고 private/CV 후보를 만든 방법이다.

| 항목 | 값 |
|---|---:|
| base 후보 | classwise greedy stack |
| base OOF | 0.970528 |
| challenger | fold-safe TE LightGBM |
| challenger OOF | 0.967900 |
| best candidate OOF | 0.970573 |
| base 대비 상승 | +0.000045 |
| train 변경 row | 83 |
| test 변경 row | 40 |
| transition | `GALAXY -> STAR` |

최고 조건:

| 조건 | 값 |
|---|---:|
| feature mask | `high_gi_low_rz` |
| transition rule | `base_galaxy_to_star` |
| challenger_conf_min | 0.55 |
| challenger_margin_min | 0.15 |
| base_conf_max | 0.60 |

해석:

- fold-safe TE 모델은 전체적으로는 약하지만, 일부 높은 g-i / 낮은 redshift 경계 row에서 추가 신호를 줬다.
- 이 후보는 OOF 최고점이다.
- 다만 meta-fold minimum delta가 음수라서 최종 선택 시 안정 후보와 같이 비교해야 한다.

## 6. Guarded Candidate Rule

row-level 후보는 반드시 OOF에서 이득이 있었던 subset과 transition만 허용한다.

현재 유효했던 guard 방향:

- `high_gi_low_rz`
- `GALAXY -> STAR`
- challenger confidence 높음
- challenger margin 충분함
- base confidence는 너무 높지 않음
- test 변경 row는 작게 유지

현재 private/CV 후보 우선순위:

| 우선순위 | 후보 | OOF | 역할 |
|---:|---|---:|---|
| 1 | 56 TE disagreement patch | 0.970573 | OOF 최고 |
| 2 | 48 subset guard | 0.970556 | 안정성 후보 |
| 3 | 44 subset guard | 0.970558 | OOF 높은 보조 후보 |
| 4 | 35 greedy cat+realmlp | 0.970479 | stacker 기반 안정 후보 |

## 7. Logistic Stacker 설정

현재 유효했던 stacker 계열:

| stacker | OOF |
|---|---:|
| lr-stacker-v9 direct | 0.970279 |
| available prediction stacker + RealMLP feature bank, C=0.10 | 0.970447 |
| available prediction stacker + fold-safe TE, C=0.10 | 0.970408 |

유효 설정:

- 5-fold OOF 기반
- multinomial logistic regression 계열
- regularization `C=0.10`
- 여러 seed 평균
- raw probability를 그대로 넣고 stacker에서 calibration을 맡김

주의:

- source를 많이 넣는다고 자동으로 좋아지지 않는다.
- OOF score, class recall, source diversity, test changed row를 같이 봐야 한다.

## 8. 지금 기준 보류 또는 낮은 우선순위

| 방법 | 현재 수치 | 판단 |
|---|---:|---|
| OVR XGBoost RealMLP feature | 0.964203 | 설정이 아직 약함. 단독으로는 낮음 |
| fold-safe TE LightGBM direct | 0.967900 | direct 제출 후보 아님. disagreement source로 유효 |
| pure model ensemble | 0.966345 | 기준선 역할 |
| public-only row probing | public은 오를 수 있음 | private 목적에서는 보조 정보만 |

## 9. Research Material Stack

2026-06-23에 추가한 단계다. 기존 stacker가 직접 모델 OOF/test source 중심이었다면, 이 단계는 우리가 만든 guarded 후보와 TE disagreement 후보의 OOF/test probability까지 다시 stack 재료로 넣는다.

실행 스크립트:

- `scripts/optimize_research_material_stack.py`

기준 base:

- `56_te_disagreement_oof0970573`

선택된 stage:

| 단계 | source | weight | OOF |
|---|---|---:|---:|
| base raw | 56 TE disagreement | 0.000 | 0.970573 |
| base bias | 56 TE disagreement | 0.000 | 0.970574 |
| blend round 1 | `private_cv_guarded_03_all_changed_rz_0_2_safe_color_cand_margin_005` | 0.050 | 0.970591 |
| blend round 2 | `our-classwise-logistic-blender` | 0.095 | 0.970599 |
| blend round 3 | `oof_generalization_stack_realmlp_feature_bank_fast` | 0.005 | 0.970603 |

생성 후보:

| 후보 | OOF | 역할 |
|---|---:|---|
| `outputs/68_PRIVATE_CV_research_material_stack_oof0970603.csv` | 0.970603 | 현재 OOF 최고, 공격형 |
| `outputs/69_PRIVATE_CV_guarded_01_all_changed_rz_0_2_allconf_oof970595.csv` | 0.970595 | 68의 guarded 안정형 |
| `outputs/75_PRIVATE_CV_research_material_stack_wide_oof0970595.csv` | 0.970595 | wide 탐색 결과, 68보다 낮아 우선순위 낮음 |

68번 class report:

| class | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| GALAXY | 0.990754 | 0.960310 | 0.975295 | 377480 |
| QSO | 0.951994 | 0.977113 | 0.964390 | 117143 |
| STAR | 0.883517 | 0.974385 | 0.926729 | 82724 |
| macro avg | 0.942088 | 0.970603 | 0.955471 | 577347 |

해석:

- 68번은 56번 대비 OOF를 +0.000030 올렸다.
- 다만 test 변경 row가 604개라 public 반응은 낮을 수 있다.
- 69번은 OOF가 68보다 조금 낮지만 accuracy, precision, f1 쪽이 더 안정적이다.
- 제출 확인 우선순위는 68, 69 순서다.

## 다음에 계속 밀 설정

## 10. Class-wise Research Blend

2026-06-23에 추가한 단계다. 68번 후보는 전체 probability blend로는 강했지만 public 반응이 약했다. 그래서 전체 source를 다시 섞는 대신, class column별로만 source를 섞었다.

실행 스크립트:

- `scripts/optimize_classwise_research_blend.py`

핵심 결과:

| 후보 | 시작점 | 변경 방식 | OOF | test 변경 row |
|---|---|---|---:|---:|
| `outputs/84_PRIVATE_CV_classwise_research_blend_oof0970621.csv` | 68 | STAR column만 3단계 보정 | 0.970621 | 27 |
| `outputs/85_PRIVATE_CV_classwise_research_blend_69start_oof0970608.csv` | 69 | QSO/GALAXY/STAR 소폭 보정 | 0.970608 | 10 |
| `outputs/90_PRIVATE_CV_subset_guard_68_plus_84_good_union_oof0970627.csv` | 68 | 84의 good subset만 추가 | 0.970627 | 15 vs 68 |

84번 선택 stage:

| 단계 | source | class | alpha | OOF |
|---|---|---|---:|---:|
| start | 68 research material stack | - | 0.000 | 0.970603 |
| round 1 | available CatV3 stacker | STAR | 0.070 | 0.970610 |
| round 2 | classwise blender C=0.10 | STAR | 0.035 | 0.970616 |
| round 3 | 69 guarded research stack | STAR | 0.010 | 0.970621 |

84번 class recall:

| class | 68 recall | 84 recall | delta |
|---|---:|---:|---:|
| GALAXY | 0.960310 | 0.960350 | +0.000040 |
| QSO | 0.977113 | 0.977105 | -0.000009 |
| STAR | 0.974385 | 0.974409 | +0.000024 |

해석:

- 새 최고 OOF는 90번의 0.970627이다.
- 상승은 대부분 STAR column 확률 보정에서 나왔다.
- 68 대비 OOF changed row는 63개, test changed row는 27개라 과도한 흔들기가 아니다.
- 85번은 69를 시작점으로 한 더 보수적인 후보지만 OOF는 84보다 낮다.
- 90번은 68을 기준으로 84의 good subset 보정만 얹은 후보라 84보다 OOF가 더 높다.
- 단, 90번도 `good_m` subset과 일부 class recall floor가 약하므로 다음 단계는 raw OOF만 보지 말고 meta-fold min delta와 worst subset penalty를 목적함수에 넣어야 한다.

## 다음에 계속 밀 설정

1. 90번을 새 기준 후보로 둔다.
2. 90번의 `good_m`, `O/B`, `G/K_Red_Sequence` 등 약점 subset을 별도로 guard한다.
3. classwise blend 목적함수에 `meta_fold_min_delta`, `worst_subset_delta`, `worst_class_recall_delta` penalty를 넣는다.
4. hard-coded source 목록 대신 artifacts 전체에서 top/diverse OOF source를 자동 탐색해 classwise blend 후보로 넣는다.
5. CatBoost `catv3` feature set은 direct 제출 후보가 아니라 source material로만 쓴다.
6. 새 OOF/test source가 생기면 `optimize_classwise_research_blend.py` source 후보에 추가한다.
7. OOF source diversity 분석으로 중복 source를 제거한다.
8. logloss 기준 best iteration은 사용하지 않는다. 최종 판단은 validation balanced accuracy fallback이다.
