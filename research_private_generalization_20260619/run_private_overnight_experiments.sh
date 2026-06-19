#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/parkyeonggon/Projects/kaggle/Kaggle_Predicting_Stellar_Class"
cd "$ROOT"

mkdir -p artifacts/overnight_logs
LOG_PATH="artifacts/overnight_logs/private_overnight_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[overnight] log: $LOG_PATH"
echo "[overnight] started: $(date)"

echo "[1/6] LightGBM with RealMLP-style non-leaky features"
.venv/bin/python scripts/train_lgbm_cv.py \
  --output-dir artifacts/lgbm_cv_realmlp_features \
  --feature-set realmlp \
  --fold-limit 5 \
  --n-estimators 6500 \
  --early-stopping-rounds 260 \
  --early-stop-metric valid-bac \
  --learning-rate 0.025 \
  --num-leaves 80 \
  --max-depth -1 \
  --min-child-samples 120 \
  --subsample 0.90 \
  --colsample-bytree 0.88 \
  --reg-alpha 0.15 \
  --reg-lambda 3.0 \
  --prediction-iteration-policy early-stop-best \
  --diagnostic-period 100 \
  --diagnostic-train-sample 50000 \
  --log-period 250

echo "[2/6] XGBoost with RealMLP-style non-leaky features"
.venv/bin/python scripts/train_xgboost_cv.py \
  --output-dir artifacts/xgboost_cv_realmlp_features \
  --feature-set realmlp \
  --fold-limit 5 \
  --num-boost-round 12000 \
  --early-stopping-rounds 500 \
  --early-stop-metric valid-bac \
  --learning-rate 0.024 \
  --max-depth 6 \
  --min-child-weight 10 \
  --subsample 0.90 \
  --colsample-bytree 0.88 \
  --reg-alpha 0.10 \
  --reg-lambda 6.0 \
  --max-bin 256 \
  --prediction-iteration-policy early-stop-best \
  --diagnostic-period 200 \
  --diagnostic-train-sample 50000 \
  --log-period 250

echo "[3/6] CatBoost with RealMLP-style non-leaky features"
.venv/bin/python scripts/train_catboost_cv.py \
  --output-dir artifacts/catboost_cv_realmlp_features \
  --feature-set realmlp \
  --fold-limit 5 \
  --iterations 9000 \
  --early-stopping-rounds 900 \
  --early-stop-metric valid-bac \
  --learning-rate 0.030 \
  --depth 7 \
  --l2-leaf-reg 12 \
  --random-strength 1.0 \
  --bagging-temperature 0.55 \
  --prediction-iteration-policy early-stop-best \
  --diagnostic-period 100 \
  --diagnostic-train-sample 50000 \
  --log-period 250

echo "[4/6] One-vs-rest CatBoost with RealMLP-style non-leaky features"
.venv/bin/python scripts/train_ovr_catboost_cv.py \
  --output-dir artifacts/ovr_catboost_realmlp_features \
  --feature-set realmlp \
  --fold-limit 5 \
  --iterations 5000 \
  --early-stopping-rounds 500 \
  --learning-rate 0.030 \
  --depth 7 \
  --l2-leaf-reg 12 \
  --random-strength 0.9 \
  --bagging-temperature 0.55 \
  --log-period 250

echo "[5/6] Logistic stacker with new private model bank"
.venv/bin/python scripts/build_available_prediction_stacker.py \
  --output-dir artifacts/available_prediction_stacker_realmlp_feature_bank_c010 \
  --c 0.1 \
  --seeds 5 \
  --epochs 1200 \
  --include-own-models

echo "[6/6] OOF diagnostics for new stacker"
.venv/bin/python scripts/make_oof_generalization_diagnostics.py \
  --candidate-name stacker_realmlp_feature_bank_c010 \
  --candidate-oof artifacts/available_prediction_stacker_realmlp_feature_bank_c010/available_prediction_stacker_oof.npy \
  --candidate-test artifacts/available_prediction_stacker_realmlp_feature_bank_c010/available_prediction_stacker_test.npy \
  --output-dir artifacts/oof_generalization_diagnostics_realmlp_feature_bank_c010

echo "[overnight] finished: $(date)"
echo "[overnight] log: $LOG_PATH"
