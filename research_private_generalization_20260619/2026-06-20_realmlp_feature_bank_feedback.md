# 2026-06-20 RealMLP Feature Bank 실험 기록

## 목적

이번 실험의 목적은 public 점수만 올리는 row patch가 아니라, OOF/CV 기준으로 private 일반화 후보를 더 단단하게 만드는 것이었다.

핵심 질문은 두 가지였다.

1. 우리가 새로 만든 단일 모델이 기존 스태커 재료보다 실제로 강한가?
2. 그 재료를 넣은 스태커가 OOF/CV 기준으로 기존 private 후보를 넘는가?

## 완료된 overnight 실험

실행 스크립트:

```bash
bash research_private_generalization_20260619/run_private_overnight_experiments.sh
```

실행된 작업:

1. LightGBM + RealMLP식 non-leaky feature
2. XGBoost + RealMLP식 non-leaky feature
3. CatBoost + RealMLP식 non-leaky feature
4. One-vs-rest CatBoost + RealMLP식 non-leaky feature
5. 새 모델 bank를 포함한 logistic stacker
6. OOF/subset/transition 진단

## 단일 모델 결과

| 모델 | OOF balanced accuracy | 판단 |
|---|---:|---|
| 기존 pure ensemble | 0.966345 | 이전 순수 모델 기준선 |
| 기존 LightGBM | 0.965704 | 이전 LGBM |
| 새 LightGBM realmlp feature | 0.965842 | 소폭 개선 |
| 기존 XGBoost | 0.965721 | 이전 XGB |
| 새 XGBoost realmlp feature | 0.966051 | 개선 |
| 기존 CatBoost depth7 | 0.965247 | 이전 CatBoost |
| 새 CatBoost realmlp feature | 0.969746 | 이번 실험의 핵심 성과 |
| 새 one-vs-rest CatBoost realmlp feature | 0.969152 | 단독은 낮지만 스태커 재료로 중요 |

가장 중요한 변화는 CatBoost였다.

기존 CatBoost는 OOF 0.965247 수준이었는데, RealMLP식 feature를 넣은 CatBoost는 OOF 0.969746까지 올라갔다. 이것은 단순한 튜닝 차이가 아니라 feature 표현이 모델 성능을 크게 바꾼 결과다.

CatBoost fold별 결과:

| fold | OOF balanced accuracy |
|---:|---:|
| 1 | 0.968947 |
| 2 | 0.970799 |
| 3 | 0.970103 |
| 4 | 0.969856 |
| 5 | 0.969027 |

저장은 마지막 iteration 기준이 아니라 validation balanced accuracy 최고 지점 기준으로 이루어졌다.

## 새 스태커 결과

| 후보 | OOF | public |
|---|---:|---:|
| 07_lr_stacker_v9_direct | 0.970279 | 0.97101 |
| 18_oof_generalization_lr_v9_bias_realmlp0 | 0.970345 | 0.97069 |
| 19_PRIVATE_CV_guarded_01 | 0.970351 | 0.97090 |
| 새 realmlp feature bank stacker | 0.970447 | 0.97081 |

새 스태커는 OOF 기준으로 기존 후보를 넘었다. 하지만 public 점수는 0.97081로 낮았다.

이 결과는 public과 OOF/CV가 같은 방향으로 움직이지 않는다는 것을 다시 보여준다. private를 목표로 하면 OOF 상승 자체는 의미가 있지만, public 하락의 원인이 어떤 변경 방향인지 분석해야 한다.

## 새 스태커가 public에서 내려간 이유

07 기준 test 변경:

| 후보 | test 변경 row |
|---|---:|
| aggressive 새 스태커 | 936 |
| 23 guarded | 265 |
| 25 guarded | 349 |

aggressive 새 스태커의 test transition:

| transition | count |
|---|---:|
| GALAXY → QSO | 146 |
| GALAXY → STAR | 312 |
| QSO → GALAXY | 96 |
| QSO → STAR | 34 |
| STAR → GALAXY | 303 |
| STAR → QSO | 45 |

OOF에서는 STAR → GALAXY, QSO → GALAXY 쪽이 강한 이득을 만들었지만, GALAXY → STAR, GALAXY → QSO는 손실이 컸다.

OOF changed row 기준:

| transition | fixed | broken | 순이득 |
|---|---:|---:|---:|
| STAR → GALAXY | 718 | 140 | +578 |
| QSO → GALAXY | 242 | 73 | +169 |
| STAR → QSO | 69 | 43 | +26 |
| QSO → STAR | 43 | 65 | -22 |
| GALAXY → QSO | 93 | 313 | -220 |
| GALAXY → STAR | 188 | 754 | -566 |

따라서 새 스태커는 전체 OOF를 올리지만, GALAXY를 다른 클래스로 바꾸는 방향은 위험하다.

## 제출 피드백

| 제출 파일 | OOF | public | 해석 |
|---|---:|---:|---|
| aggressive available_prediction_stacker_submission | 0.970447 | 0.97081 | OOF는 상승, public은 하락 |
| 23_PRIVATE_CV_guarded_01 | 0.970439 | 0.97081 | 변경 row를 줄였지만 public 회복 없음 |
| 25_PRIVATE_CV_guarded_03 | 0.970459 | 0.97086 | OOF 최고, public은 약간 회복 |

23은 test 변경을 265개까지 줄였는데도 public은 aggressive와 같은 0.97081이었다. 즉 public 하락은 변경량만의 문제가 아니라, 변경된 row의 방향과 public split의 정답 분포가 맞지 않았기 때문으로 보인다.

25는 OOF가 가장 높고 public도 0.97086으로 조금 회복했다. public 관점에서는 여전히 07보다 낮지만, private/CV 관점에서는 가장 의미 있는 후보였다.

## 현재 판단

1. 단일 모델 개선은 성공했다.
2. CatBoost realmlp feature는 지금까지 만든 우리 단일 모델 중 가장 강하다.
3. OVR CatBoost는 단독 점수보다 스태커 재료로서 가치가 크다.
4. 새 스태커는 OOF 기준 최고를 갱신했다.
5. public은 새 후보를 좋게 평가하지 않았지만, 이것만으로 private 성능이 낮다고 단정할 수 없다.
6. 다만 GALAXY → STAR, GALAXY → QSO 변경은 강하게 제한해야 한다.

## 다음 연구 방향

다음 실험은 세 축으로 간다.

1. OOF greedy optimizer 재실행
   - 새 CatBoost, OVR CatBoost, XGBoost, LightGBM을 모두 넣고 greedy blending을 다시 수행한다.
   - 현재 logistic stacker는 전체 모델을 동시에 쓰지만, greedy 방식은 어떤 모델이 실제 OOF를 올리는지 단계별로 드러난다.

2. GALAXY 보호형 후보
   - GALAXY → STAR, GALAXY → QSO 변경을 거의 막는다.
   - STAR → GALAXY, QSO → GALAXY처럼 OOF에서 순이득이 큰 방향만 제한적으로 허용한다.

3. 외부 검증
   - SDSS external validation으로 public과 다른 검증축을 만든다.
   - public이 낮아도 외부 검증과 OOF가 같이 좋으면 private 후보로 유지한다.

## 현재 사용해야 할 기준 파일

private/CV 후보 기준:

- outputs/25_PRIVATE_CV_guarded_03_all_changed_rz_0_2_safe_color_cand_margin_005_oof970459.csv

public 기준 참고 파일:

- outputs/07_lr_stacker_v9_direct_oof970279.csv

새 스태커 원본:

- artifacts/available_prediction_stacker_realmlp_feature_bank_c010/available_prediction_stacker_submission.csv

분석 자료:

- artifacts/oof_generalization_diagnostics_realmlp_feature_bank_c010/subset_delta_metrics.csv
- artifacts/oof_generalization_diagnostics_realmlp_feature_bank_c010/transition_counts.csv
- artifacts/oof_generalization_diagnostics_realmlp_feature_bank_c010/oof_changed_row_outcomes.csv
- artifacts/private_cv_stable_submissions_realmlp_feature_bank_after_public/candidate_summary.csv

## 추가 진행: 빠른 OOF greedy optimizer

23번과 25번 제출 이후, 새 CatBoost/OVR 재료를 어떻게 섞어야 하는지 보기 위해 빠른 greedy optimizer를 다시 실행했다.

결과:

| 단계 | 선택 재료 | weight | OOF |
|---|---|---:|---:|
| base raw | lr-stacker-v9-public-oof | 0.00 | 0.970279 |
| base bias | lr-stacker-v9-public-oof | 0.00 | 0.970307 |
| blend round 1 | our-catboost-realmlp-features | 0.27 | 0.970432 |
| blend round 2 | realmlp-0 | 0.06 | 0.970479 |

이 결과는 새 CatBoost realmlp feature가 실제로 기존 lr-stacker-v9에 추가 가치가 있음을 보여준다. 다만 full grid가 아니라 빠른 탐색이므로, 다음 실험에서는 one-vs-rest XGBoost까지 추가한 뒤 다시 스태킹해야 한다.

빠른 optimizer 후보의 특징:

- OOF: 0.970479
- 07 대비 OOF 변화 row: 1269
- 07 대비 test 변화 row: 413
- GALAXY recall 상승
- STAR recall 소폭 상승
- QSO recall 소폭 하락

이 후보에서 만든 안정 후보:

| 파일 | OOF | test 변경 row | 판단 |
|---|---:|---:|---|
| outputs/27_PRIVATE_CV_guarded_01_all_changed_rz_0_2_safe_color_allconf_oof970398.csv | 0.970398 | 218 | 보수형 |
| outputs/28_PRIVATE_CV_guarded_02_all_changed_rz_0_2_safe_color_uncertain_base_oof970390.csv | 0.970390 | 216 | 더 보수형 |
| outputs/29_PRIVATE_CV_guarded_03_all_changed_rz_0_cand_margin_010_oof970331.csv | 0.970331 | 8 | 매우 보수형 |
| outputs/30_PRIVATE_CV_guarded_04_all_changed_rz_0_2_gi_ge3_not_ob_allconf_oof970393.csv | 0.970393 | 225 | 보수형 |

## 다음 수면용 큐

다음 큐는 아직 직접 학습하지 않은 one-vs-rest XGBoost를 추가한다.

실행 파일:

```bash
bash research_private_generalization_20260619/run_private_next_queue_20260620.sh
```

이 큐가 남기는 것:

1. one-vs-rest XGBoost OOF/test probability
2. class별 binary balanced accuracy
3. multiclass fold balanced accuracy
4. train/valid logloss CSV
5. PNG/SVG 그래프
6. 새 logistic stacker
7. 새 OOF greedy optimizer
8. OOF/subset/transition 진단 그래프
9. private 안정 후보 CSV
10. 최종 요약 로그

아직 남아 있는 연구:

- fold-safe target encoding
- RealMLP 직접 학습
- TabM 직접 학습
- CatBoost v3식 SDSS external low-weight 학습
- class-wise logistic blending

이 중 가장 조심해야 할 것은 fold-safe target encoding이다. 잘못 넣으면 정답 누수가 생길 수 있으므로, 다음 구현에서는 반드시 fold 안에서 fit하고 validation/test에는 transform만 적용해야 한다.

## 2026-06-20 one-vs-rest XGBoost 큐 결과

수면용 큐 실행:

```bash
bash research_private_generalization_20260619/run_private_next_queue_20260620.sh
```

로그:

- artifacts/overnight_logs/private_next_queue_20260620_025357.log

이번 큐는 RealMLP-style feature를 넣은 one-vs-rest XGBoost를 새 재료로 만들고, 기존 available prediction stacker와 OOF greedy optimizer에 추가하는 목적이었다.

결론부터 보면 one-vs-rest XGBoost 단독 모델은 실패했다.

| 항목 | 결과 |
|---|---:|
| OVR XGBoost OOF | 0.964203 |
| 새 logistic stacker OOF | 0.970466 |
| 새 greedy optimizer OOF | 0.970479 |

OVR XGBoost의 class별 이진 모델 자체는 완전히 나쁘지 않았다. 하지만 class별 binary probability를 그대로 합쳐 multiclass 예측으로 바꾸는 과정에서 성능이 무너졌다. 즉 문제는 XGBoost가 모든 경계를 못 배운 것이 아니라, class별 이진 점수를 multiclass 결정으로 변환하는 방식에 있다.

그래서 OVR XGBoost는 단독 제출 후보가 아니다.

새 logistic stacker는 OOF가 0.970466으로 기존 stacker보다 조금 올랐다. 하지만 07 기준 test 변경 row가 948개라서 너무 공격적이다.

반면 greedy optimizer는 OVR XGBoost를 선택하지 않았다. 최종 선택은 기존과 같다.

## 2026-06-20 fold-safe target encoding 결과

이번에는 RealMLP v5 계열에서 중요하게 보였던 조합 범주 통계를 LightGBM에 넣었다.

중요한 점은 target encoding을 전체 train에 한 번에 fit하지 않았다는 것이다. 각 fold마다 train fold에서만 조합 범주의 class 통계를 만들고, validation fold와 test에는 그 통계만 transform했다. 이 구조가 아니면 정답 누수가 생긴다.

실행 큐:

```bash
bash research_private_generalization_20260619/run_foldsafe_te_queue_20260620.sh
```

주요 결과:

| 항목 | 결과 |
|---|---:|
| fold-safe TE LightGBM OOF | 0.967900 |
| 이전 LightGBM realmlp feature OOF | 0.965842 |
| 개선폭 | +0.002058 |
| available prediction stacker + TE OOF | 0.970408 |
| greedy optimizer + TE OOF | 0.970528 |

fold별 결과:

| fold | balanced accuracy | best iteration |
|---:|---:|---:|
| 1 | 0.967708 | 232 |
| 2 | 0.967947 | 265 |
| 3 | 0.967749 | 266 |
| 4 | 0.967629 | 358 |
| 5 | 0.968466 | 256 |

해석:

1. fold-safe target encoding은 단일 LightGBM 성능을 확실히 올렸다.
2. 하지만 현재 강한 스태커에는 통째로 넣었을 때 추가 선택되지 않았다.
3. greedy optimizer의 최종 선택은 이전 classwise blender 실험과 동일했다.
4. `outputs/51_PRIVATE_CV_greedy_with_foldsafe_te_oof0970528.csv`는 `outputs/37_PRIVATE_CV_greedy_with_classwise_blender_oof0970528.csv`와 완전히 같은 파일이다.
5. direct TE 제출 파일인 `outputs/50_PRIVATE_CV_lgbm_foldsafe_te_direct_oof0967900.csv`는 단독 모델 검증용이지 최종 제출 후보가 아니다.

새로 나온 파일 판단:

| 파일 | 판단 |
|---|---|
| outputs/50_PRIVATE_CV_lgbm_foldsafe_te_direct_oof0967900.csv | 제출 비추천. 단일 모델 검증용 |
| outputs/51_PRIVATE_CV_greedy_with_foldsafe_te_oof0970528.csv | 37번과 동일. 새 제출 의미 없음 |
| outputs/52_PRIVATE_CV_guarded_01_star_to_qso_rz_2_cand_margin_005_oof970338.csv | 기존 보수 후보와 유사. 우선순위 낮음 |
| outputs/53_PRIVATE_CV_guarded_03_star_to_qso_rz_2_allconf_oof970351.csv | 기존 보수 후보와 유사. 우선순위 낮음 |
| outputs/54_PRIVATE_CV_guarded_05_star_to_galaxy_rz_0_uncertain_base_oof970371.csv | 기존 보수 후보와 유사. 우선순위 낮음 |
| outputs/55_PRIVATE_CV_guarded_06_star_to_qso_all_cand_margin_010_oof970334.csv | 기존 보수 후보와 유사. 우선순위 낮음 |

현재 private/CV 기준 우선 후보는 그대로 아래 두 개다.

1. `outputs/44_PRIVATE_CV_subset_guard_guard_weak_core_base_galaxy_to_star_oof0970558.csv`
2. `outputs/48_PRIVATE_CV_subset_guard_guard_weak_core_base_galaxy_to_non_galaxy_oof0970556.csv`

단, 44는 OOF가 가장 높고 48은 meta-fold 안정성이 더 좋아 보였다. public 점수만으로 둘 중 하나를 버리면 안 된다.

## 다음 실험 방향

fold-safe TE를 전체 스태커에 그냥 넣는 방식은 효과가 작았다. 다음은 TE 모델을 독립 모델로 쓰는 것이 아니라, disagreement row 분석용으로 써야 한다.

구체적으로는 다음을 본다.

1. classwise greedy 후보는 A라고 판단하고 TE LightGBM은 B라고 판단한 row
2. 그중 TE의 확신도와 margin이 높은 row
3. OOF에서 실제로 그 변경 방향이 이득이었던 subset
4. test에서도 같은 물리 feature 구간에 있는 row

즉 다음 후보는 “TE 전체 예측을 섞는 후보”가 아니라 “TE가 강하게 반대하는 일부 row만 허용하는 후보”가 되어야 한다.

이 방향이 필요한 이유는 명확하다. TE LightGBM은 단독 OOF를 올렸지만 스태커는 선택하지 않았다. 그러면 전체 모델로는 중복 신호가 많다는 뜻이고, 남은 가치는 boundary disagreement row 안에 있다.

## 2026-06-20 TE disagreement patch 결과

위 가설을 바로 검증했다.

실행:

```bash
.venv/bin/python scripts/build_te_disagreement_patch.py --output-dir artifacts/te_disagreement_patch_classwise37 --top-k 8 --output-rank-start 56
```

기준:

| 항목 | 값 |
|---|---:|
| base 후보 | classwise greedy stack |
| base OOF | 0.970528 |
| challenger | fold-safe TE LightGBM |
| challenger OOF | 0.967900 |
| OOF disagreement row | 5,760 |
| test disagreement row | 2,289 |

결과:

| 후보 | OOF | base 대비 | train 변경 row | test 변경 row |
|---|---:|---:|---:|---:|
| outputs/56_PRIVATE_CV_te_disagree_high_gi_low_rz_base_galaxy_to_star_c0_55_m0_15_b0_60_oof0970573.csv | 0.970573 | +0.000045 | 83 | 40 |

56번 후보는 현재까지의 OOF 최고점이다.

비교:

| 후보 | OOF |
|---|---:|
| 37 classwise greedy | 0.970528 |
| 44 subset guard | 0.970558 |
| 48 subset guard | 0.970556 |
| 56 TE disagreement patch | 0.970573 |

해석:

1. fold-safe TE 모델은 전체로 섞으면 약하지만, 특정 경계 row에서는 추가 신호가 있었다.
2. 잡힌 조건은 `high_gi_low_rz` 구간의 `GALAXY -> STAR` 변경이다.
3. 즉 TE 모델은 전체적으로 GALAXY 쪽으로 강하게 치우쳤지만, 일부 높은 g-i / 낮은 redshift 경계에서는 기존 스태커가 GALAXY로 둔 것을 STAR로 바꾸는 신호를 줬다.
4. 다만 meta-fold 최소 delta는 음수라서 48번보다 안정성은 약하다.

중요한 주의:

57~63번은 이름만 다르고 사실상 같은 83개 train row, 40개 test row를 잡은 중복 후보들이다. 제출한다면 56번 하나만 의미 있다.

현재 private/CV 후보 우선순위:

1. `outputs/56_PRIVATE_CV_te_disagree_high_gi_low_rz_base_galaxy_to_star_c0_55_m0_15_b0_60_oof0970573.csv`
2. `outputs/48_PRIVATE_CV_subset_guard_guard_weak_core_base_galaxy_to_non_galaxy_oof0970556.csv`
3. `outputs/44_PRIVATE_CV_subset_guard_guard_weak_core_base_galaxy_to_star_oof0970558.csv`

56번은 OOF 최고점 후보, 48번은 안정성 후보, 44번은 기존 최고 OOF에 가까운 후보로 유지한다.

| 단계 | 선택 재료 | weight | OOF |
|---|---|---:|---:|
| base raw | lr-stacker-v9-public-oof | 0.00 | 0.970279 |
| base bias | lr-stacker-v9-public-oof | 0.00 | 0.970307 |
| blend round 1 | our-catboost-realmlp-features | 0.27 | 0.970432 |
| blend round 2 | realmlp-0 | 0.06 | 0.970479 |

따라서 현재 private/CV 기준 핵심 후보는 아래 파일이다.

- outputs/35_PRIVATE_CV_greedy_cat_realmlp_plus_realmlp0_oof970479.csv

07 기준 test 변경은 413개다.

전이 방향:

| 방향 | row |
|---|---:|
| STAR -> GALAXY | 176 |
| GALAXY -> STAR | 85 |
| QSO -> GALAXY | 84 |
| GALAXY -> QSO | 38 |
| QSO -> STAR | 17 |
| STAR -> QSO | 13 |

이 후보는 이전 aggressive stacker보다 test 변경이 작고, OOF에서 GALAXY recall을 올리면서 STAR도 소폭 유지했다. public 점수는 낮게 나올 수 있지만, private/CV 목적에는 가장 논리가 맞는 후보다.

이번 큐가 새로 만든 보수 후보:

| 파일 | OOF | test 변경 row | 판단 |
|---|---:|---:|---|
| outputs/31_PRIVATE_CV_guarded_01_all_changed_rz_0_2_safe_color_allconf_oof970398.csv | 0.970398 | 218 | 보수형 |
| outputs/32_PRIVATE_CV_guarded_02_all_changed_rz_0_2_safe_color_uncertain_base_oof970390.csv | 0.970390 | 216 | 보수형 |
| outputs/33_PRIVATE_CV_guarded_03_all_changed_rz_0_cand_margin_010_oof970331.csv | 0.970331 | 8 | 초보수형 |
| outputs/34_PRIVATE_CV_guarded_04_all_changed_rz_0_2_gi_ge3_not_ob_allconf_oof970393.csv | 0.970393 | 225 | 보수형 |

31-34는 35보다 OOF가 낮다. 제출 후보 우선순위는 아니다.

다음 연구 방향:

1. OVR XGBoost를 raw multiclass 변환으로 쓰지 말고 class-wise logistic blending으로 다시 보정한다.
2. fold-safe target encoding을 구현한다.
3. CatBoost v3식 SDSS external low-weight 학습을 별도 검증한다.
4. RealMLP/TabM 직접 학습은 의존성과 실행 시간이 크므로, 위 두 실험 뒤에 진행한다.

가장 먼저 해야 할 일은 class-wise logistic blending이다. 이유는 OVR XGBoost의 이진 경계 점수에는 정보가 있는데, 최종 multiclass 변환에서 정보가 깨졌기 때문이다.

## 35번 제출 피드백

제출 파일:

- outputs/35_PRIVATE_CV_greedy_cat_realmlp_plus_realmlp0_oof970479.csv

결과:

| 항목 | 값 |
|---|---:|
| OOF | 0.970479 |
| Public LB | 0.97104 |

해석:

- public은 0.97104로 높지는 않지만, 23/25번의 0.97081/0.97086보다 낫다.
- OOF 개선 방향이 public에서 완전히 깨지지는 않았다.
- public 0.972 계열과 비교하면 낮지만, private/CV용 후보로는 현재 가장 논리가 맞다.
- 35번은 최종 선택 후보군에 남긴다.

다음 실험은 class-wise logistic blending이다.

목적:

1. OVR XGBoost/OVR CatBoost의 class별 이진 경계 신호를 raw multiclass 변환으로 버리지 않는다.
2. class별 binary logistic meta-model로 다시 보정한다.
3. 그 결과를 다시 OOF greedy optimizer에 넣는다.
4. OOF 상승, class recall 변화, test 변경 row를 동시에 확인한다.

추가한 실행 파일:

- scripts/train_classwise_logistic_blender.py
- research_private_generalization_20260619/run_classwise_blender_queue_20260620.sh

수면용 실행 명령:

```bash
bash research_private_generalization_20260619/run_classwise_blender_queue_20260620.sh
```

이 큐가 생성하는 핵심 파일:

- artifacts/classwise_logistic_blender_c010/report.json
- artifacts/classwise_logistic_blender_c010/classwise_blender_submission.csv
- artifacts/classwise_logistic_blender_c010/classwise_blender_oof.npy
- artifacts/classwise_logistic_blender_c010/classwise_blender_test.npy
- artifacts/classwise_logistic_blender_c010/classwise_recall_vs_reference.png
- artifacts/classwise_logistic_blender_c010/classwise_confusion_delta.png
- artifacts/classwise_logistic_blender_c010/classwise_model_importance.png
- artifacts/oof_generalization_stack_with_classwise_blender_fast/report.json
- artifacts/private_cv_stable_with_classwise_blender_fast/report.json
- outputs/36_PRIVATE_CV_classwise_blender_direct_oof*.csv
- outputs/37_PRIVATE_CV_greedy_with_classwise_blender_oof*.csv
- outputs/40_PRIVATE_CV_guarded_*.csv 이후 안정 후보

smoke run에서는 2개 모델만 사용했는데도 reference 대비 OOF가 소폭 상승했다. full run은 모든 사용 가능한 예측 재료를 넣기 때문에 결과를 확인해야 한다.

## class-wise blender 결과

실행 결과:

| 후보 | OOF | 07 대비 test 변경 row | 해석 |
|---|---:|---:|---|
| 36 classwise direct | 0.970394 | 810 | 직접 모델은 너무 공격적 |
| 37 greedy with classwise | 0.970528 | 603 | 현재 최고 OOF |

37번 greedy optimizer가 선택한 흐름:

| 단계 | 선택 재료 | weight | OOF |
|---|---|---:|---:|
| base raw | lr-stacker-v9-public-oof | 0.00 | 0.970279 |
| base bias | lr-stacker-v9-public-oof | 0.00 | 0.970324 |
| blend round 1 | our-classwise-logistic-blender | 0.35 | 0.970456 |
| blend round 2 | our-catboost-realmlp-features | 0.281944 | 0.970511 |
| blend round 3 | realmlp-2 | 0.029167 | 0.970528 |

37번의 class recall:

| class | 07 recall | 37 recall | 변화 |
|---|---:|---:|---:|
| GALAXY | 0.960689 | 0.960128 | -0.000562 |
| QSO | 0.976840 | 0.977301 | +0.000461 |
| STAR | 0.973309 | 0.974155 | +0.000846 |

37번은 OOF 최고지만 GALAXY recall을 깎고 QSO/STAR를 올리는 형태다. 35번은 GALAXY를 늘리는 방향이었고 public 0.97104가 나왔다. 따라서 37번은 private/CV용 핵심 후보지만, public에서는 낮게 나올 가능성이 있다.

37번의 약한 subset:

- g_i_bin 0
- g_i_bin 2
- mag_range_bin 0
- O/B_Blue_Cloud
- A/F_Blue_Cloud

이 구간에서 특히 GALAXY -> STAR 전환이 손해를 만든다.

## subset guard 결과

37번을 기준으로 약한 구간의 전환을 막는 subset guard를 만들었다.

생성 스크립트:

- scripts/build_subset_guarded_candidate.py

실행 결과 핵심:

| 후보 | OOF | 07 대비 test 변경 row | meta-fold min delta | positive rate | 판단 |
|---|---:|---:|---:|---:|---|
| 44 | 0.970558 | 538 | -0.000017 | 0.96 | 최고 OOF |
| 48 | 0.970556 | 454 | +0.000024 | 1.00 | 가장 안정적 |

44번:

- outputs/44_PRIVATE_CV_subset_guard_guard_weak_core_base_galaxy_to_star_oof0970558.csv
- 37에서 weak_core 구간의 GALAXY -> STAR 전환을 제거
- OOF가 37보다 0.000030 더 상승
- 다만 meta-fold min delta가 아주 작게 음수라 안정성은 48보다 낮다.

48번:

- outputs/48_PRIVATE_CV_subset_guard_guard_weak_core_base_galaxy_to_non_galaxy_oof0970556.csv
- weak_core 구간에서 GALAXY가 다른 class로 빠지는 전환을 더 넓게 제거
- OOF는 44보다 0.000002 낮지만, meta-fold positive rate가 1.00이고 min delta도 양수
- private 안정성 기준으로는 48이 더 낫다.

제출 우선순위:

1. 48번: 안정성 확인용 private 후보
2. 44번: 최고 OOF 확인용 private 후보
3. 37번: guard 없는 원본 classwise greedy 후보

36번 direct classwise는 test 변경이 810개로 너무 공격적이라 우선 제출하지 않는다.
