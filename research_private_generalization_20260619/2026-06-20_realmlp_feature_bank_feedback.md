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
