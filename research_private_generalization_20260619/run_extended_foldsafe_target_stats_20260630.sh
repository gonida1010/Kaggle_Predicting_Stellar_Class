#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
BASELINE_DIR="artifacts/lgbm_foldsafe_te_realmlp"
BASE_DIR="artifacts/extended_foldsafe_target_stats_20260630"
CANDIDATE_DIR="$BASE_DIR/full_extended_stats"
ANALYSIS_DIR="$BASE_DIR/analysis"
LOG_DIR="artifacts/overnight_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-$LOG_DIR/extended_foldsafe_target_stats_${STAMP}.log}"
MEMO_PATH="research_private_generalization_20260619/daily/2026-06-30/notes/extended_foldsafe_target_stats.md"

mkdir -p "$CANDIDATE_DIR" "$ANALYSIS_DIR" "$LOG_DIR" "$(dirname "$MEMO_PATH")"

if [[ "${TARGET_STATS_LOGGED:-0}" != "1" ]]; then
  export TARGET_STATS_LOGGED=1 LOG_PATH
  set +e
  bash "$0" "$@" 2>&1 | tee -a "$LOG_PATH"
  status="${PIPESTATUS[0]}"
  exit "$status"
fi

echo "[target-stats] started: $(date)"
echo "[target-stats] purpose: basic vs extended fold-safe target-stat 5-fold ablation"
echo "[target-stats] validation metric: balanced accuracy"
echo "[target-stats] public leaderboard score is not used"
echo "[target-stats] log: $LOG_PATH"

echo "[1/2] Train extended target-stat LightGBM"
PYTHONUNBUFFERED=1 MPLCONFIGDIR="$ROOT/artifacts/.mplconfig" "$PYTHON" \
  scripts/train_lgbm_foldsafe_te_cv.py \
  --output-dir "$CANDIDATE_DIR" \
  --fold-limit 5 \
  --n-estimators 1200 \
  --early-stopping-rounds 220 \
  --learning-rate 0.028 \
  --num-leaves 96 \
  --max-depth -1 \
  --min-child-samples 90 \
  --subsample 0.88 \
  --colsample-bytree 0.84 \
  --reg-alpha 0.10 \
  --reg-lambda 2.4 \
  --class-weight balanced \
  --te-stat-mode extended \
  --te-interaction-mode base \
  --te-smoothing 40 \
  --te-min-count 1 \
  --early-stop-metric valid-bac \
  --prediction-iteration-policy early-stop-best \
  --log-period 100 \
  --diagnostic-period 50 \
  --diagnostic-train-sample 50000

echo "[2/2] Analyze paired folds and bootstrap stability"
PYTHONUNBUFFERED=1 MPLCONFIGDIR="$ROOT/artifacts/.mplconfig" "$PYTHON" \
  scripts/analyze_extended_foldsafe_target_stats.py \
  --baseline-dir "$BASELINE_DIR" \
  --candidate-dir "$CANDIDATE_DIR" \
  --output-dir "$ANALYSIS_DIR" \
  --memo-path "$MEMO_PATH" \
  --bootstrap-repeats 2000

echo "[target-stats] completed: $(date)"
echo "[target-stats] report: $ANALYSIS_DIR/decision.json"
echo "[target-stats] graph: $ANALYSIS_DIR/extended_target_stats_ablation.png"
echo "[target-stats] memo: $MEMO_PATH"
