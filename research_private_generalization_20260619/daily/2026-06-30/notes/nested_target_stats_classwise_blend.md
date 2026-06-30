# Nested class-wise target-stat blend

확장 target-stat 모델은 전체 OOF에서는 basic 모델보다 낮았지만 GALAXY/QSO 경계에 다른 신호가 있었습니다. 전체 OOF에서 가중치를 고르는 낙관 편향을 피하기 위해 10시드 x 5폴드 바깥 검증을 사용했습니다.

- 선택 클래스: QSO
- 최종 고정 alpha: 0.880
- basic OOF: 0.967899835
- 최종 OOF: 0.967924659
- 최종 OOF 변화: +0.000024824
- nested valid 평균 변화: -0.000005278
- nested valid 최악 변화: -0.000248798
- nested 개선 비율: 25/50
- 최종 설정 부트스트랩 95% 구간: [-0.000051181, +0.000095777]
- 판정: `research_only`
