# CatBoost + SDSS 0.03 전체 5-fold 결과

- 생성 시각: 2026-06-29T18:06:13.639131+09:00
- 기존 CatBoost OOF BAC: 0.969746378
- SDSS 0.03 CatBoost OOF BAC: 0.969966101
- OOF 변화: +0.000219723
- 개선 fold: 5/5
- 최악 fold 변화: +0.000087424
- 최악 class 평균 recall 변화: +0.000127159
- 판단: `promote_to_stacker_screen`

## fold별 변화

- fold 1: 0.968947 -> 0.969436 (+0.000489)
- fold 2: 0.970799 -> 0.971011 (+0.000212)
- fold 3: 0.970103 -> 0.970264 (+0.000161)
- fold 4: 0.969856 -> 0.969943 (+0.000087)
- fold 5: 0.969027 -> 0.969177 (+0.000149)

## 다음 단계

- promote_to_stacker_screen이면 이 모델의 OOF/test probability를 기존 private stacker source bank에 추가한다.
- 추가 후 전체 OOF뿐 아니라 반복 meta-fold, class recall, 취약 subset의 최악 변화까지 다시 검사한다.
- reject_or_research이면 외부 데이터 혼합을 최종 후보에 사용하지 않는다.
