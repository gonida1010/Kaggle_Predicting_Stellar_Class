# CatBoost SDSS 0.03 stacker source ablation

검증 목적은 새 `CatBoost + SDSS17 weight 0.03` 모델이 단순 단일 모델 개선을 넘어 기존 OOF 스태커에 독립적인 신호를 추가하는지 확인하는 것이었습니다.

- 제외 스태커 OOF balanced accuracy: 0.970540578
- 포함 스태커 OOF balanced accuracy: 0.970556447
- OOF 변화: +0.000015870
- 동일 seed/fold 개선: 13/25
- 폴드 변화 평균: +0.000020404
- 폴드 변화 최솟값: -0.000135014
- 부트스트랩 95% 구간: [-0.000063144, +0.000091007]
- 부트스트랩 개선 확률: 66.800%
- 예측 변경 행: 372
- 새로 정답이 된 행 / 오답이 된 행: 186 / 176
- 클래스 재현율 변화: GALAXY +0.000010597, QSO +0.000085366, STAR -0.000048354
- 새 소스 중요도 순위: 14 / 24
- 판정: `research_only`

판정 이유: OOF increased, but paired stability evidence is not yet strong enough for final promotion.

그래프는 `stacker_source_ablation.png`에 저장했습니다. 이 비교는 두 실행의 입력 모델, 폴드, 시드를 고정하고 새 소스 하나만 추가한 인과적 ablation입니다.
