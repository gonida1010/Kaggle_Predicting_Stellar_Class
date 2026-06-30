# 확장 fold-safe target-stat 검증

기존 basic target encoding과 동일한 LightGBM 설정, 동일한 5개 fold를 사용했습니다. 차이는 fold train에서 만드는 통계 피처뿐입니다.

- basic target-stat 피처: 44개
- extended target-stat 피처: 110개
- basic OOF: 0.967899835
- extended OOF: 0.967887034
- OOF 변화: -0.000012802
- 개선 fold: 3/5
- 최악 fold 변화: -0.000166484
- 부트스트랩 95% 구간: [-0.000189364, +0.000152890]
- 부트스트랩 개선 확률: 42.250%
- 클래스 재현율 변화: GALAXY +0.000569567, QSO +0.000008537, STAR -0.000616508
- 변경 예측: 1,853행
- 새로 정답 / 새로 오답: 990 / 825
- 판정: `reject`

누수 방지 조건:

- 모든 통계는 각 fold의 train 행과 label만 사용했습니다.
- validation과 test는 해당 fold train에서 만든 mapping으로 transform했습니다.
- unseen 또는 최소 빈도 미만 범주는 fold train의 class prior로 대체했습니다.
