#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SOURCE_NAME="our-catboost-sdss003-realmlp"
BASE_DIR="artifacts/catboost_sdss_stacker_ablation_20260630"
WITHOUT_DIR="$BASE_DIR/without_sdss003"
WITH_DIR="$BASE_DIR/with_sdss003"
LOG_DIR="artifacts/overnight_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-$LOG_DIR/catboost_sdss_stacker_ablation_${STAMP}.log}"
MEMO_PATH="research_private_generalization_20260619/daily/2026-06-30/notes/catboost_sdss_stacker_ablation.md"

mkdir -p "$WITHOUT_DIR" "$WITH_DIR" "$LOG_DIR" "$(dirname "$MEMO_PATH")"

if [[ "${ABLATION_LOGGED:-0}" != "1" ]]; then
  export ABLATION_LOGGED=1 LOG_PATH
  set +e
  bash "$0" "$@" 2>&1 | tee -a "$LOG_PATH"
  status="${PIPESTATUS[0]}"
  exit "$status"
fi

echo "[ablation] started: $(date)"
echo "[ablation] purpose: identical 5-seed x 5-fold stacker comparison with one source removed/added"
echo "[ablation] source: $SOURCE_NAME"
echo "[ablation] log: $LOG_PATH"

echo "[1/3] Train stacker WITHOUT new CatBoost SDSS source"
PYTHONUNBUFFERED=1 "$PYTHON" scripts/build_available_prediction_stacker.py \
  --output-dir "$WITHOUT_DIR" \
  --seeds 5 \
  --folds 5 \
  --epochs 650 \
  --c 0.1 \
  --exclude-models "$SOURCE_NAME"

echo "[2/3] Train stacker WITH new CatBoost SDSS source"
PYTHONUNBUFFERED=1 "$PYTHON" scripts/build_available_prediction_stacker.py \
  --output-dir "$WITH_DIR" \
  --seeds 5 \
  --folds 5 \
  --epochs 650 \
  --c 0.1

echo "[3/3] Analyze paired folds, class recall, and bootstrap stability"
PYTHONUNBUFFERED=1 "$PYTHON" scripts/analyze_stacker_source_ablation.py \
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
