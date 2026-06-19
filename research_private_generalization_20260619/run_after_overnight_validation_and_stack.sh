#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/parkyeonggon/Projects/kaggle/Kaggle_Predicting_Stellar_Class"
cd "$ROOT"

mkdir -p artifacts/overnight_logs
LOG_PATH="artifacts/overnight_logs/after_overnight_validation_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[after-overnight] log: $LOG_PATH"
echo "[after-overnight] started: $(date)"

echo "[1/5] SDSS external validation with RealMLP-style features"
.venv/bin/python scripts/evaluate_sdss_external_generalization.py \
  --output-dir artifacts/sdss_external_generalization_realmlp_features \
  --feature-set realmlp \
  --models lgbm catboost xgboost \
  --fold-limit 5 \
  --external-sample 0 \
  --skip-boundary \
  --log-period 250

echo "[2/5] OOF-only optimizer using newly available own models"
.venv/bin/python scripts/optimize_oof_generalization_stack.py \
  --output-dir artifacts/oof_generalization_stack_realmlp_feature_bank \
  --base-model lr-stacker-v9-public-oof \
  --max-added-weight 0.35 \
  --weight-steps 36 \
  --bias-low 0.82 \
  --bias-high 1.22 \
  --bias-steps 33 \
  --rounds 3 \
  --include-own-models

echo "[3/5] Diagnostics for optimized OOF stack"
.venv/bin/python scripts/make_oof_generalization_diagnostics.py \
  --candidate-name optimized_stack_realmlp_feature_bank \
  --candidate-oof artifacts/oof_generalization_stack_realmlp_feature_bank/generalization_stack_oof.npy \
  --candidate-test artifacts/oof_generalization_stack_realmlp_feature_bank/generalization_stack_test.npy \
  --output-dir artifacts/oof_generalization_diagnostics_optimized_realmlp_feature_bank

echo "[4/5] Stable private-candidate CSVs from optimized OOF stack"
.venv/bin/python scripts/build_private_cv_stable_submissions.py \
  --output-dir artifacts/private_cv_stable_submissions_realmlp_feature_bank \
  --candidate-oof artifacts/oof_generalization_stack_realmlp_feature_bank/generalization_stack_oof.npy \
  --candidate-test artifacts/oof_generalization_stack_realmlp_feature_bank/generalization_stack_test.npy \
  --stack-report artifacts/oof_generalization_stack_realmlp_feature_bank/report.json \
  --folds 5 \
  --seeds 5 \
  --bins 10 \
  --top-k 4

echo "[5/5] Print key reports"
.venv/bin/python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("artifacts/lgbm_cv_realmlp_features/lgbm_baseline_report.json"),
    Path("artifacts/xgboost_cv_realmlp_features/report.json"),
    Path("artifacts/catboost_cv_realmlp_features/catboost_baseline_report.json"),
    Path("artifacts/ovr_catboost_realmlp_features/report.json"),
    Path("artifacts/available_prediction_stacker_realmlp_feature_bank_c010/report.json"),
    Path("artifacts/oof_generalization_stack_realmlp_feature_bank/report.json"),
    Path("artifacts/sdss_external_generalization_realmlp_features/report.json"),
    Path("artifacts/private_cv_stable_submissions_realmlp_feature_bank/report.json"),
]

for path in paths:
    print(f"\n## {path}")
    if not path.exists():
        print("missing")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in [
        "oof_balanced_accuracy",
        "full_oof_balanced_accuracy",
        "covered_oof_balanced_accuracy",
        "score_on_covered_rows",
        "best_oof_balanced_accuracy",
        "delta_vs_raw_base",
        "best_external",
        "candidate_summary",
    ]:
        if key in data:
            print(f"{key}: {data[key]}")
PY

echo "[after-overnight] finished: $(date)"
echo "[after-overnight] log: $LOG_PATH"
