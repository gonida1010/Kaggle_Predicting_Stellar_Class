#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR=/private/tmp/matplotlib
export XDG_CACHE_HOME=/private/tmp

SCREEN_DIR="artifacts/catboost_sdss_weight_screen_20260629"
MEMO_PATH="research_private_generalization_20260619/daily/2026-06-29/notes/catboost_sdss_weight_screen.md"
SDSS_ZIP="/Users/parkyeonggon/Downloads/archive (12).zip"
LOG_DIR="artifacts/overnight_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/catboost_sdss_weight_screen_${STAMP}.log"

mkdir -p "$SCREEN_DIR" "$(dirname "$MEMO_PATH")" "$LOG_DIR"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[screen] started: $(date)"
echo "[screen] control and candidates share identical folds, seed, feature mapping, and capacity"
echo "[screen] SDSS is appended only to training folds; validation is competition-only"
echo "[screen] log: $LOG_PATH"

run_weight() {
  local label="$1"
  local weight="$2"
  echo
  echo "[screen] training ${label}: sdss_weight=${weight}"
  .venv/bin/python scripts/train_catboost_cv.py \
    --output-dir "${SCREEN_DIR}/${label}" \
    --feature-set realmlp \
    --fold-limit 2 \
    --iterations 5500 \
    --early-stopping-rounds 700 \
    --early-stop-metric valid-bac \
    --learning-rate 0.030 \
    --depth 7 \
    --l2-leaf-reg 12 \
    --random-strength 1.0 \
    --bagging-temperature 0.55 \
    --prediction-iteration-policy early-stop-best \
    --diagnostic-period 100 \
    --diagnostic-train-sample 50000 \
    --log-period 250 \
    --heartbeat-seconds 60 \
    --sdss-zip "$SDSS_ZIP" \
    --sdss-weight "$weight"
}

run_weight "w000" "0.00"
run_weight "w003" "0.03"
run_weight "w006" "0.06"
run_weight "w010" "0.10"

echo
echo "[screen] paired fold analysis"
.venv/bin/python scripts/analyze_catboost_sdss_weight_screen.py \
  --screen-dir "$SCREEN_DIR" \
  --memo-path "$MEMO_PATH"

echo
echo "[screen] completed: $(date)"
echo "[screen] summary: ${SCREEN_DIR}/catboost_sdss_weight_summary.csv"
echo "[screen] decision: ${SCREEN_DIR}/catboost_sdss_weight_decision.json"
echo "[screen] graph: ${SCREEN_DIR}/catboost_sdss_weight_screen.png"
echo "[screen] memo: ${MEMO_PATH}"
