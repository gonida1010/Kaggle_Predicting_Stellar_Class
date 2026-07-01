# Kaggle Playground S6E6 최종 결과, public보다 OOF를 선택한 이유

안녕하세요.

Kaggle Playground Series Season 6 Episode 6 대회가 끝났습니다.

최종 결과는 2,824팀 중 27등이었습니다. private leaderboard 점수는 0.97029였고, 상위 약 1%로 대회를 마무리했습니다.

[이미지 삽입 위치: 2026-07-01_final_leaderboard_rank27.png - 최종 27등 리더보드 화면]

대회 중간에는 public leaderboard 점수를 높이는 실험도 별도로 진행했습니다. 가장 높은 public용 파일은 다음 결과를 기록했습니다.

- public score: 0.97254
- private score: 0.97005

반면 최종 제출로 선택한 193번 후보는 public 점수를 선택 기준으로 사용하지 않았습니다.

- OOF Balanced Accuracy: 0.9706589608
- private score: 0.97029
- 최종 순위: 27 / 2,824

public에서는 공격적으로 맞춘 파일이 더 좋아 보였지만, 숨겨진 private test에서는 OOF와 반복 검증으로 선택한 파일이 0.00024 높았습니다. 대회 내내 걱정했던 public leaderboard 과적합이 실제 최종 결과에서 확인된 셈입니다.

## 최종 193번 후보는 어떻게 만들었는가

193번은 새로운 대형 모델 하나가 갑자기 만든 결과가 아니었습니다.

LightGBM, XGBoost, CatBoost, one-vs-rest CatBoost, fold-safe target encoding, RealMLP 계열과 공개 OOF/test probability를 연구 재료로 모았습니다. 각 모델의 단독 점수만 비교하지 않고, 현재 후보가 틀리는 경계에서 다른 정보를 제공하는지 확인했습니다.

최종 후보로 이어진 흐름은 다음과 같습니다.

1. 95번에서 target encoding disagreement와 subset guard를 결합했습니다.
2. 171번에서 취약한 O/B 구간을 안정 후보로 되돌렸습니다.
3. 181번에서 CatBoost + RealMLP식 feature 모델의 QSO 확률을 0.5% 추가했습니다.
4. 193번에서 one-vs-rest CatBoost + RealMLP식 feature 모델의 GALAXY 확률을 0.5% 추가했습니다.

점수 변화는 다음과 같았습니다.

```text
95   0.9706450214
171  0.9706515037
181  0.9706563117
193  0.9706589608
```

마지막 변화는 매우 작았습니다.

```text
new_GALAXY = 0.995 * current_GALAXY
           + 0.005 * OVR_CatBoost_GALAXY
```

확률을 다시 정규화한 뒤 최종 class를 선택했습니다. 181번과 비교하면 train 예측은 3개, test 예측은 1개만 달라졌습니다.

one-vs-rest CatBoost의 단독 OOF는 0.9691523963으로 최종 후보보다 낮았습니다. 하지만 단독 점수가 낮다고 쓸모없는 모델은 아니었습니다. GALAXY와 나머지 class를 따로 학습한 구조라서 기존 stacker와 다른 경계 정보를 가지고 있었습니다.

강한 모델을 크게 섞은 것이 아니라, 현재 모델이 부족한 class 신호만 아주 작게 빌려온 방식이었습니다.

[이미지 삽입 위치: 2026-07-01_source_oof_score_rank.png - 단독 모델과 누적 후보의 OOF 비교]

## 가장 높은 OOF를 그대로 선택하지 않았습니다

후보는 총 1,199개를 만들었고, 상위 120개를 다시 정밀 검사했습니다.

전체 OOF 하나만 보면 193번보다 점수가 높은 후보도 있었습니다. 192번은 OOF 0.970664로 더 높았습니다. 하지만 192번은 특정 subset 규칙으로 더 많은 row를 변경한 공격적인 후보였습니다.

- 192번 test 변경: 130 rows
- 193번 test 변경: 85 rows
- 193번은 직전 181번에서 test 1 row만 변경

후보를 많이 탐색한 뒤 최고 OOF만 고르면 OOF 자체에 과적합될 수 있습니다. 그래서 다음 조건을 함께 확인했습니다.

- class별 recall
- 취약 subset의 최악 손실
- 변경한 train/test row 수
- 10개 seed와 5-fold를 조합한 50개 meta validation 구간
- 기준 후보보다 좋아진 fold 비율

193번은 56번 기준 후보와 비교한 50개 meta-fold에서 모두 양수였습니다.

```text
평균 meta-fold delta   +0.0000863677
최소 meta-fold delta   +0.0000102667
positive fold rate      1.0
```

한두 fold에서 크게 오른 결과가 아니었습니다. 모든 반복 구간에서 기준보다 좋아졌고, 직전 안정 후보에서 거의 움직이지 않았습니다.

[이미지 삽입 위치: 2026-07-01_robust_candidate_rank.png - 상위 120개 후보의 안정성 재정렬]

## class별 성능과 취약 구간도 확인했습니다

193번의 OOF class recall은 다음과 같습니다.

```text
GALAXY  0.960276
QSO     0.977207
STAR    0.974493
```

Balanced Accuracy는 세 class recall의 평균입니다. 데이터가 많은 GALAXY만 잘 맞히는 것으로는 좋은 점수를 만들 수 없습니다.

subset별로는 weak O/B, mag_range bin 2, g-i bin 2, A/F Blue Cloud와 낮은 magnitude 구간이 개선됐습니다. 반대로 weak u-r bin 6, high g-i, M 계열 일부에서는 작은 손실이 있었습니다.

모든 구간이 좋아졌다고 포장하지 않았습니다. 최악 subset 손실을 penalty로 넣고, 상승 폭보다 손실이 큰 후보는 제외했습니다.

이 방식은 마지막까지 이어졌습니다. 이후 SDSS17 외부 데이터를 낮은 가중치로 추가한 CatBoost, 확장 target statistics, nested blend도 실험했습니다. 일부 단일 모델과 stacker OOF는 개선됐지만, 반복 fold와 subset 안정성을 포함하면 193번을 확실히 넘지 못했습니다.

그래서 마지막 순간에 새 결과를 억지로 제출하지 않고 193번을 유지했습니다.

## public 파일과 private 파일을 분리한 선택

public 전용 실험은 대회 test의 공개 구간에서 점수를 올리는 목적이었습니다. ridge flip과 consensus, 공개 제출 파일의 disagreement를 이용해 public 0.97254까지 올렸습니다.

하지만 public 점수는 test 전체가 아니라 일부 공개 구간의 결과입니다. 같은 규칙이 private 구간에서도 맞는다는 보장은 없었습니다.

최종 결과는 다음처럼 갈렸습니다.

```text
public 최적화 파일
public   0.97254
private  0.97005

OOF 일반화 파일
OOF      0.970659
private  0.97029
rank     27 / 2,824
```

public 점수가 더 높은 파일을 최종 선택했다면 순위가 내려갔을 가능성이 높았습니다. 공개 점수와 일반화 후보를 끝까지 분리한 판단이 맞았습니다.

## 마무리

이번 대회에서 가장 크게 배운 것은 모델 수가 많다고 자동으로 강한 앙상블이 되지 않는다는 점이었습니다.

좋은 보조 모델은 단독 점수가 가장 높은 모델이 아니라, 현재 후보의 오류와 다른 오류를 가진 모델이었습니다. 그리고 보조 신호가 유효하더라도 크게 섞을 필요는 없었습니다. 최종 성능을 만든 마지막 가중치는 0.5%였습니다.

또한 OOF가 높다는 사실만으로 일반화를 증명할 수는 없었습니다. 후보를 많이 탐색할수록 OOF에도 과적합할 수 있습니다. 그래서 반복 fold, class recall, subset 손실, 변경 row 수를 함께 봤습니다.

최종적으로 남은 기준은 다음과 같습니다.

1. public leaderboard 점수를 최종 선택 기준으로 사용하지 않습니다.
2. train fold에서 학습하고 validation fold에서만 평가한 OOF를 사용합니다.
3. class별 recall과 취약 subset을 함께 확인합니다.
4. 여러 seed와 fold에서 같은 방향으로 개선되는지 확인합니다.
5. 점수가 비슷하면 기존 안정 후보를 덜 바꾸는 쪽을 선택합니다.

대회 종료 후 확인한 private score는 0.97029였습니다.

2,824팀 중 27등.

public 점수를 끝까지 쫓기보다 우리가 만든 검증 환경을 믿고 선택한 결과였습니다.
