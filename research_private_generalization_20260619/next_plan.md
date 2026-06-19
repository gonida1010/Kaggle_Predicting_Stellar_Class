# Next Plan

## Updated Direction

현재 단순 GBDT 반복 수 튜닝만으로는 상위권 단일 모델과의 차이를 줄이기 어렵다.

다음 실험은 상위권 단일 모델에서 확인한 구조를 우리 코드에 이식하는 방향이다.

1. RealMLP v5식 non-leaky feature를 공통 feature builder에 추가한다.
2. LightGBM, XGBoost, CatBoost를 같은 feature set으로 다시 돌린다.
3. CatBoost one-vs-rest도 같은 feature set으로 돌린다.
4. 새 OOF/test probability를 기존 prediction bank와 함께 stacker에 넣는다.
5. OOF, fold 안정성, class recall, 이후 SDSS external 기준으로 판단한다.

## Commands

전체 밤샘 실행:

```bash
bash research_private_generalization_20260619/run_private_overnight_experiments.sh
```

밤샘 실행이 끝난 뒤 후속 검증/최적화:

```bash
bash research_private_generalization_20260619/run_after_overnight_validation_and_stack.sh
```

개별 결과 확인:

```bash
cat artifacts/lgbm_cv_realmlp_features/lgbm_baseline_report.json
cat artifacts/xgboost_cv_realmlp_features/report.json
cat artifacts/catboost_cv_realmlp_features/catboost_baseline_report.json
cat artifacts/ovr_catboost_realmlp_features/report.json
cat artifacts/available_prediction_stacker_realmlp_feature_bank_c010/report.json
```

## Decision Rules

### Promote to private candidate

Promote if:

- OOF BAC beats 0.9703513342698414, or
- OOF is close but fold stability and SDSS external are clearly better, or
- class recall balance improves without large worst-subset damage.

### Hold

Hold if:

- OOF improves only through many row-level changes.
- SDSS external drops sharply.
- STAR recall worsens without clear QSO/GALAXY gain.

### Reject

Reject if:

- OOF improves but meta-fold positive rate is weak.
- external validation shows the same pattern as boundary patch: local OOF gain but external degradation.
- it mostly reproduces public-blend behavior without model-level evidence.

## Next Implementation After Overnight Run

If `realmlp` feature set helps:

1. Add fold-safe target encoding for `alpha_floor_x_delta_floor` and `u_floor_x_z_floor`.
2. Add XGBoost one-vs-rest specialist based on the Kirill-style notebook structure.
3. Add CatBoost v3-style external SDSS low-weight training.

If it does not help:

1. Keep the feature set only for stacker/boundary analysis.
2. Move directly to one-vs-rest XGBoost and existing RealMLP/TabM prediction bank selection.
