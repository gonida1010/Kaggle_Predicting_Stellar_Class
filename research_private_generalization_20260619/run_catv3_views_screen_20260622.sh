#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p artifacts/overnight_logs
LOG="artifacts/overnight_logs/catv3_screen_$(date +%Y%m%d_%H%M%S).log"
PY=".venv/bin/python"
export PYTHONUNBUFFERED=1

exec > >(tee -a "$LOG") 2>&1

echo "[catv3-screen] log: $LOG"
echo "[catv3-screen] started: $(date)"
echo "[catv3-screen] purpose: 1-fold timing/heartbeat/iteration screen before full catv3 queue"

"$PY" scripts/train_catboost_bac_chunked_cv.py \
  --output-dir artifacts/catboost_cv_catv3_bac_chunked_screen \
  --feature-set catv3 \
  --fold-limit 1 \
  --max-iterations 2500 \
  --chunk-size 150 \
  --bac-patience-chunks 5 \
  --bac-min-delta 0.0 \
  --learning-rate 0.030 \
  --depth 7 \
  --l2-leaf-reg 12 \
  --random-strength 1.0 \
  --bagging-temperature 0.55 \
  --diagnostic-train-sample 50000 \
  --log-period 50

echo "[catv3-screen] report"
"$PY" - <<'PY'
import json
from pathlib import Path

p = Path("artifacts/catboost_cv_catv3_bac_chunked_screen/catboost_baseline_report.json")
d = json.loads(p.read_text())
print("covered_oof_balanced_accuracy:", d.get("covered_oof_balanced_accuracy"))
print("fold_scores:")
for row in d.get("fold_scores", []):
    print(row)
print("outputs:", d.get("outputs"))
PY

echo "[catv3-screen] finished: $(date)"
