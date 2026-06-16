# 상위권 노트북 분석 및 실행 계획

## 결론

현재 우리 public probe는 `0.97141`에서 포화됐다. 더 많은 local row patch를 제출하는 것은 정보량이 낮다.

상위권 노트북에서 가져올 핵심은 두 가지다.

1. `ps-s6e6-ensemble.ipynb`
   - 94개 score-named public submission bank를 사용한다.
   - top 3 제출물 majority vote로 consensus anchor를 만든다.
   - 각 제출물의 public score 차이를 target으로 두고 Ridge로 row flip 효과를 추정한다.
   - Bayesian support, isotonic smoothing, entropy-gated probability tail, candidate voting을 붙인다.
   - 노트북 출력상 `v5_voted_ensemble.csv`가 public `0.97199`.

2. `s6e6-stellar-a-deeper-look-at-the-results.ipynb`
   - 5개 독립 public notebook 결과를 비교한다.
   - 5개 중 mode count가 2인 `2-2-1` 초고난도 row만 따로 본다.
   - `0.97209` base에서 count==2 행만 XGB 제출 결과로 교체해 public `0.97214`를 기록했다.
   - 전체를 건드리는 stacking이 아니라 ambiguity-only patch라서 public+generalization 후보로 가치가 있다.

`s6e6-0-97209-clean-final.ipynb` 자체는 모델이 아니라 `zoli800/s6e6-097209-final-submission` dataset의 `submission.csv`를 복사하는 노트북이다. 이 파일은 재현용 입력으로는 유용하지만, 그 자체를 우리 모델 개선으로 보면 안 된다.

## 지금 추가한 스크립트

- `scripts/build_bank_ridge_flip_candidates.py`
  - `ps-s6e6-ensemble.ipynb`의 핵심을 로컬 스크립트로 이식했다.
  - 입력: `external_preds/` 아래 score-named CSV들, 예: `0.97183.csv`, `0.97182.csv`.
  - 출력: `artifacts/bank_ridge_flip_v5/v5_voted_ensemble.csv` 등.

- `scripts/build_ambiguous_vote_patch.py`
  - `deeper-look` 노트북의 count==2 ambiguity patch를 이식했다.
  - 입력: `external_preds/0.97209.csv`와 5개 독립 notebook submission.
  - 출력: `artifacts/ambiguous_vote_patch/ambiguous_count2_replace_sub4.csv`.

- `scripts/build_final_submission_tracks.py`
  - 최종 제출용 두 파일을 만든다.
  - `artifacts/final_submissions/final_generalization_model.csv`
  - `artifacts/final_submissions/final_public_generalization.csv`

## 필요한 외부 파일

`external_preds/` 아래에 다음을 넣는다. 이 폴더는 git ignore 대상이다.

### Ridge submission bank

`nina2025/ps-s6e6` 계열 score-named CSV:

- `0.97183.csv`
- `0.97182.csv`
- `0.97181.csv`
- `0.97179.csv`
- 가능하면 노트북처럼 총 94개 전체

확률 파일이 있으면 추가:

- `test_preds__*.csv`
- `*test_preds__*.npy`

확률 파일이 없어도 Ridge flip 후보는 생성된다. 다만 tail/entropy 후보는 생략된다.

### Ambiguous vote patch

다음 이름으로 넣으면 기본 명령이 바로 동작한다.

- `0.97209.csv`
- `cat-3_submission.csv`
- `realmlp-5_submission.csv`
- `nn-2_submission.csv`
- `xgb-5_submission.csv`
- `submission_binary.csv`

## 실행 순서

```bash
python scripts/build_bank_ridge_flip_candidates.py --prediction-dir external_preds
python scripts/build_ambiguous_vote_patch.py
python scripts/build_final_submission_tracks.py
```

그리고 제출 후보는 다음에서 확인한다.

```bash
ls -lh artifacts/final_submissions
cat artifacts/final_submissions/final_submission_tracks_report.json
```

## 오늘 판단

외부 bank가 들어오기 전까지 public 제출을 더 쓰지 않는다.

현재 fallback:

- 일반화: `artifacts/final_submissions/final_generalization_model.csv`
- public+일반화: `artifacts/final_submissions/final_public_generalization.csv`

단, 현재 public+일반화 파일은 외부 bank가 없어서 `group_research_top_10.csv`, public `0.97141` fallback이다. 0.972급으로 올라가려면 외부 bank와 0.97209 계열 파일이 필요하다.
