#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SOURCE_NAME="our-lgbm-extended-target-stats"
BASE_DIR="artifacts/extended_target_stats_stacker_ablation_20260630"
WITHOUT_DIR="$BASE_DIR/without_extended_stats"
WITH_DIR="$BASE_DIR/with_extended_stats"
LOG_DIR="artifacts/overnight_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-$LOG_DIR/extended_target_stats_stacker_ablation_${STAMP}.log}"
MEMO_PATH="research_private_generalization_20260619/daily/2026-06-30/notes/extended_target_stats_stacker_ablation.md"

mkdir -p "$WITHOUT_DIR" "$WITH_DIR" "$LOG_DIR" "$(dirname "$MEMO_PATH")"

if [[ "${EXTENDED_STACKER_LOGGED:-0}" != "1" ]]; then
  export EXTENDED_STACKER_LOGGED=1 LOG_PATH
  set +e
  bash "$0" "$@" 2>&1 | tee -a "$LOG_PATH"
  status="${PIPESTATUS[0]}"
  exit "$status"
fi

echo "[ablation] started: $(date)"
echo "[ablation] source: $SOURCE_NAME"
echo "[ablation] identical 5-seed x 5-fold logistic stacker comparison"
echo "[ablation] public leaderboard score is not used"
echo "[ablation] log: $LOG_PATH"

echo "[1/3] Train stacker WITHOUT extended target-stat source"
PYTHONUNBUFFERED=1 "$PYTHON" scripts/build_available_prediction_stacker.py \
  --output-dir "$WITHOUT_DIR" \
  --seeds 5 \
  --folds 5 \
  --epochs 650 \
  --c 0.1 \
  --exclude-models "$SOURCE_NAME"

echo "[2/3] Train stacker WITH extended target-stat source"
PYTHONUNBUFFERED=1 "$PYTHON" scripts/build_available_prediction_stacker.py \
  --output-dir "$WITH_DIR" \
  --seeds 5 \
  --folds 5 \
  --epochs 650 \
  --c 0.1

echo "[3/3] Analyze paired stability"
PYTHONUNBUFFERED=1 MPLCONFIGDIR="$ROOT/artifacts/.mplconfig" "$PYTHON" \
  scripts/analyze_stacker_source_ablation.py \
  --without-dir "$WITHOUT_DIR" \
  --with-dir "$WITH_DIR" \
  --source-name "$SOURCE_NAME" \
  --output-dir "$BASE_DIR/analysis" \
  --memo-path "$MEMO_PATH" \
  --bootstrap-repeats 2000

echo "[ablation] completed: $(date)"
echo "[ablation] report: $BASE_DIR/analysis/decision.json"
echo "[ablation] graph: $BASE_DIR/analysis/stacker_source_ablation.png"
echo "[ablation] memo: $MEMO_PATH"
