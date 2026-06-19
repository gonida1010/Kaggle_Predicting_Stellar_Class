# Algorithm Inventory

## Directly Implemented Or Being Implemented

### LightGBM

Status: implemented with diagnostics.

Current direct result on SDSS external:

- external BAC: 0.9101638725728026
- GALAXY recall: 0.9147783665573219
- QSO recall: 0.9158799641369126
- STAR recall: 0.8998332870241734

Interpretation:

- Internal OOF is decent, but SDSS external STAR recall is weak.
- It is useful as a stacker component, but external evidence says it should not dominate private-generalization selection.

### CatBoost

Status: implemented with diagnostics.

Current direct result on SDSS external:

- external BAC: 0.9205782928437469
- GALAXY recall: 0.8932795020607284
- QSO recall: 0.9226834027741153
- STAR recall: 0.9457719736963971

Interpretation:

- Best external model so far.
- Strong STAR recall is the main signal.
- Next step is not blind CatBoost-only submission, but adding robust CatBoost variants to the stacker.

### XGBoost

Status: newly implemented with diagnostics.

Reason for adding:

- Existing high-rank stacker used multiple XGBoost variants.
- We had XGBoost prediction-bank inputs, but not a local fold-safe diagnostic training pipeline.
- Now we can generate OOF/test proba and compare against LGBM/CatBoost under the same private/generalization framework.

Pending:

- Full 5-fold run result.
- SDSS external result with full run.
- Stacker impact after adding `our-xgboost`.

## Already Used As External OOF/Test Prediction Inputs

These were not all retrained locally, but were loaded as OOF/test proba bank inputs.

- realmlp-0
- realmlp-2
- realmlp-5
- tabm-0
- tabm-1
- xgb-6
- lgbm-5
- lr-stacker-v9-public-oof
- our-pure
- our-meta

Known OOF stack optimizer result:

- raw `lr-stacker-v9`: 0.9702794714614861
- best OOF stack after bias + realmlp-0 blend: 0.9703454236206706
- accepted added model: `realmlp-0` with weight 0.06

## Not Yet Directly Rebuilt Locally

### RealMLP

Why not yet:

- It is deep-learning based, dependency and training setup are heavier.
- We already have some RealMLP OOF/test predictions.
- The immediate gain is to use them correctly in OOF stacker, then decide if direct retraining is worth the time.

Next possible action:

- Try direct RealMLP only if current XGBoost/LGBM/CatBoost stacker plateaus.

### TabM

Why not yet:

- Same as RealMLP, but potentially more important because stacker feature importance previously ranked `tabm-1` very high.
- We have OOF/test prediction bank inputs.

Next possible action:

- First verify how much existing `tabm-1` contributes in current stacker.
- Direct TabM training is a later heavier experiment.

### TabICL / TabPFN

Why not yet:

- Dataset has 577k train rows. TabPFN-style models are not always practical at this scale without special scaling/subsampling.
- The realistic path is either using available OOF/test predictions or using them as feature generators on sampled folds.

Next possible action:

- Only test after GBDT + existing OOF bank stacker is stable.

### NN / Logistic Regression

Why not yet:

- Logistic Regression is already used as stacker/meta model.
- Standalone NN is lower priority unless OOF/test prediction bank shows unique complementary errors.

## Research Rule Going Forward

No model is accepted because its public score is high.

A model or stacker stage is accepted only if it helps at least one of:

- OOF balanced accuracy
- fold stability
- class recall balance
- SDSS external stability
- complementary error pattern versus existing stacker

