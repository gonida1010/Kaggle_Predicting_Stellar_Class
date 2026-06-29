# 남은 일반화 연구의 실행 순서

이 문서는 아직 검증하지 않은 방법을 순서대로 실행하고, 완료 결과를 잊지 않도록 남기는 기준 기록입니다.

1. 강한 CatBoost와 SDSS 저가중치 학습
- 현재 직접 학습 CatBoost 최고인 RealMLP식 비누수 피처 모델을 기준으로 사용합니다.
- SDSS는 학습 fold에만 추가하고 competition validation fold에는 넣지 않습니다.
- weight 0.00, 0.03, 0.06, 0.10을 같은 fold로 비교합니다.
- paired fold 평균, 최악 fold, class recall을 통과한 weight만 5-fold 전체 검증으로 보냅니다.

2. 상관성과 다양성을 제한조건으로 둔 class-wise stacker
- 단순 OOF 최대화가 아니라 fold별 개선 일관성과 모델 간 상관성을 함께 사용합니다.
- 현재 최고 `193_PRIVATE_CV_oof970659.csv`를 넘어도 최악 fold가 무너지면 채택하지 않습니다.

3. 확장 fold-safe target-stat feature bank
- 범주별 class mean뿐 아니라 count, frequency, entropy와 compact interaction을 fold 내부에서만 계산합니다.
- validation과 test 변환에는 해당 fold train 통계만 사용합니다.

4. RealMLP v5 직접 학습
- 공개 OOF를 복사하지 않고 로컬 5-fold OOF와 test probability를 직접 생성합니다.
- embedding, robust preprocessing, ensemble, EMA, label smoothing을 단계별로 분리 검증합니다.

5. one-vs-rest TabM 직접 학습
- class별 이진 OOF를 생성하고 class-wise logistic 결합을 검증합니다.
- 직접 RealMLP와의 오류 상관성이 낮을 때만 stacker 재료로 채택합니다.

6. nested transition/subset logit calibration
- 기존 OOF를 같은 OOF로 다시 튜닝하는 낙관 편향을 막기 위해 바깥 fold에서만 평가합니다.
- 전체 OOF뿐 아니라 class recall과 취약 subset의 최악 성능도 함께 제한합니다.

각 단계의 결과는 해당 날짜 `notes` 폴더와 실험 `artifacts` 폴더에 CSV, JSON, 그래프, 판단 메모로 저장합니다. 한 단계가 채택 또는 폐기된 뒤 다음 단계로 이동합니다.
