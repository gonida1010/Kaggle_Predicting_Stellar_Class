#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/parkyeonggon/Projects/kaggle/Kaggle_Predicting_Stellar_Class"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
TRAIN_DIR="$ROOT/artifacts/repleafgbm_sdss_bag_20260626"
STACK_DIR="$ROOT/artifacts/available_prediction_stacker_with_repleaf_20260629"
OPT_DIR="$ROOT/artifacts/oof_generalization_stack_with_repleaf_20260629"

mkdir -p "$TRAIN_DIR" "$STACK_DIR" "$OPT_DIR"

echo "[1/3] Full 5-fold, 3-seed RepLeafGBM + fold-safe SDSS17 augmentation"
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/train_repleafgbm_cv_sdss_bag.py \
  --output-dir "$TRAIN_DIR" \
  --fold-limit 5 \
  --use-sdss17 \
  --sdss-zip "/Users/parkyeonggon/Downloads/archive (12).zip" \
  --sdss-weight 0.3 \
  --extra-seeds 0 1 \
  --n-estimators 3000 \
  --learning-rate 0.05 \
  --num-leaves 128 \
  --min-samples-leaf 20 \
  --l2-leaf 5.0 \
  --early-stopping-rounds 50 \
  --heartbeat-seconds 60 \
  2>&1 | tee "$TRAIN_DIR/run.log"

echo "[2/3] Rebuild the available-prediction logistic stacker with RepLeaf OOF/test probabilities"
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/build_available_prediction_stacker.py \
  --output-dir "$STACK_DIR" \
  --c 0.10 \
  --seeds 5 \
  2>&1 | tee "$STACK_DIR/run.log"

echo "[3/3] Run the OOF-only greedy generalization optimizer with the new source"
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/optimize_oof_generalization_stack.py \
  --output-dir "$OPT_DIR" \
  --base-model lr-stacker-v9-public-oof \
  --max-added-weight 0.35 \
  --weight-steps 36 \
  --rounds 3 \
  2>&1 | tee "$OPT_DIR/run.log"

echo "[done] Reports:"
echo "- $TRAIN_DIR/report.json"
echo "- $STACK_DIR/report.json"
echo "- $OPT_DIR/report.json"
