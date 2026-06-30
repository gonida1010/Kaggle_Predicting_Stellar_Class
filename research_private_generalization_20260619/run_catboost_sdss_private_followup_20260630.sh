#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
BASE_DIR="artifacts/catboost_sdss_private_followup_20260630"
LOG_DIR="artifacts/overnight_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-$LOG_DIR/catboost_sdss_private_followup_${STAMP}.log}"
START_DIR="artifacts/robust_private_cv_after181_20260624"
START_STEM="193_PRIVATE_CV_after181_classblend_our-ovr-catboost-realmlp-features_GALAXY_a0p0050_oof0970659"

mkdir -p "$BASE_DIR" "$LOG_DIR"

if [[ "${FOLLOWUP_LOGGED:-0}" != "1" ]]; then
  export FOLLOWUP_LOGGED=1 LOG_PATH
  set +e
  bash "$0" "$@" 2>&1 | tee -a "$LOG_PATH"
  status="${PIPESTATUS[0]}"
  exit "$status"
fi

echo "[followup] started: $(date)"
echo "[followup] purpose: OOF-only robust search starting from private candidate 193"
echo "[followup] new sources: CatBoost SDSS 0.03 and its validated logistic stacker"
echo "[followup] public leaderboard score is not used"
echo "[followup] log: $LOG_PATH"

PYTHONUNBUFFERED=1 MPLCONFIGDIR="$ROOT/artifacts/.mplconfig" "$PYTHON" \
  scripts/build_robust_private_cv_next_candidates.py \
  --output-dir "$BASE_DIR" \
  --start-oof "$START_DIR/${START_STEM}_oof.npy" \
  --start-test "$START_DIR/${START_STEM}_test.npy" \
  --start-name "193_private_cv" \
  --alpha-max 0.10 \
  --alpha-steps 20 \
  --scan-top 120 \
  --top-k 8 \
  --folds 5 \
  --seeds 10 \
  --max-sources 42 \
  --min-source-oof 0.965 \
  --output-rank-start 220 \
  --output-prefix "PRIVATE_CV_sdss_followup"

echo "[followup] completed: $(date)"
echo "[followup] report: $BASE_DIR/report.json"
echo "[followup] ranked candidates: $BASE_DIR/audited_candidates.csv"
echo "[followup] outputs: $BASE_DIR/output_candidates.csv"
