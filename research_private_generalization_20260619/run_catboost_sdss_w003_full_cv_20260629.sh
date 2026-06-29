#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR=/private/tmp/matplotlib
export XDG_CACHE_HOME=/private/tmp

OUTPUT_DIR="artifacts/catboost_sdss_w003_full_cv_20260629"
LOG_DIR="artifacts/overnight_logs"
MEMO_PATH="research_private_generalization_20260619/daily/2026-06-29/notes/catboost_sdss_w003_full_cv.md"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/catboost_sdss_w003_full_cv_${STAMP}.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$(dirname "$MEMO_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[full-cv] started: $(date)"
echo "[full-cv] selected from paired screen: SDSS weight 0.03"
echo "[full-cv] validation remains competition-only"
echo "[full-cv] snapshots are enabled every 300 seconds"
echo "[full-cv] log: $LOG_PATH"

.venv/bin/python scripts/train_catboost_cv.py \
  --output-dir "$OUTPUT_DIR" \
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
  --log-period 250 \
  --heartbeat-seconds 60 \
  --save-snapshot \
  --snapshot-interval 300 \
  --sdss-zip "/Users/parkyeonggon/Downloads/archive (12).zip" \
  --sdss-weight 0.03

.venv/bin/python scripts/analyze_catboost_sdss_full_cv.py \
  --baseline-report artifacts/catboost_cv_realmlp_features/catboost_baseline_report.json \
  --candidate-report "$OUTPUT_DIR/catboost_baseline_report.json" \
  --output-dir "$OUTPUT_DIR" \
  --memo-path "$MEMO_PATH"

echo "[full-cv] completed: $(date)"
echo "[full-cv] report: ${OUTPUT_DIR}/catboost_sdss_full_decision.json"
echo "[full-cv] graph: ${OUTPUT_DIR}/catboost_sdss_full_fold_delta.png"
echo "[full-cv] memo: ${MEMO_PATH}"
