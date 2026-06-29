# CatBoost + SDSS 저가중치 선별 기록

- 생성 시각: 2026-06-29T15:26:36.671836+09:00
- 기준 모델: RealMLP식 비누수 피처를 사용한 CatBoost
- 검증 원칙: SDSS는 각 학습 fold에만 추가하고 validation은 competition row만 사용
- 비교 원칙: 같은 seed와 같은 fold의 weight=0 대조군과 paired 비교

## 결과

- weight=0.000: mean BAC=0.969494, delta=+0.000000, worst fold delta=+0.000000, positive folds=0/2, full-CV 후보=False
- weight=0.030: mean BAC=0.970163, delta=+0.000668, worst fold delta=+0.000667, positive folds=2/2, full-CV 후보=True
- weight=0.060: mean BAC=0.970067, delta=+0.000573, worst fold delta=+0.000521, positive folds=2/2, full-CV 후보=True

## 판단

- 결정: `promote_to_full_cv`
- 선택 weight: `0.030`
- 대조군 대비 평균 변화: `+0.000668`
- 최악 fold 변화: `+0.000667`
- 제한: This screen is not final generalization evidence. A promoted weight must pass full 5-fold OOF and external/generalization diagnostics before entering the stacker.

## 다음 단계

- promote_to_full_cv이면 같은 설정으로 5-fold 전체 OOF를 재학습한다.
- 5-fold OOF, class recall, SDSS 외부 검증, 기존 stacker 추가 이득을 모두 통과해야 채택한다.
- reject_external_weights이면 이 외부 데이터 혼합은 폐기하고 다음 미실험 항목으로 이동한다.
