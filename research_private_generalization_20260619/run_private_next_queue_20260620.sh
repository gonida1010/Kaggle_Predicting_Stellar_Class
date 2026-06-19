#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/parkyeonggon/Projects/kaggle/Kaggle_Predicting_Stellar_Class"
cd "$ROOT"

mkdir -p artifacts/overnight_logs
LOG_PATH="artifacts/overnight_logs/private_next_queue_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[next-queue] log: $LOG_PATH"
echo "[next-queue] started: $(date)"
echo "[next-queue] cwd: $(pwd)"
echo "[next-queue] python: $(.venv/bin/python --version)"

echo
echo "[1/7] One-vs-rest XGBoost with RealMLP-style non-leaky features"
.venv/bin/python scripts/train_ovr_xgboost_cv.py \
  --output-dir artifacts/ovr_xgboost_realmlp_features \
  --feature-set realmlp \
  --fold-limit 5 \
  --num-boost-round 9000 \
  --early-stopping-rounds 700 \
  --learning-rate 0.025 \
  --max-depth 5 \
  --min-child-weight 9 \
  --subsample 0.90 \
  --colsample-bytree 0.88 \
  --reg-alpha 0.12 \
  --reg-lambda 8 \
  --max-bin 256 \
  --log-period 250

echo
echo "[2/7] Logistic stacker with OVR XGBoost added"
.venv/bin/python scripts/build_available_prediction_stacker.py \
  --output-dir artifacts/available_prediction_stacker_with_ovr_xgb_realmlp_c010 \
  --c 0.1 \
  --seeds 5 \
  --epochs 1200 \
  --include-own-models

echo
echo "[3/7] OOF-only greedy optimizer with OVR XGBoost added"
.venv/bin/python scripts/optimize_oof_generalization_stack.py \
  --output-dir artifacts/oof_generalization_stack_with_ovr_xgb_realmlp_fast \
  --base-model lr-stacker-v9-public-oof \
  --max-added-weight 0.30 \
  --weight-steps 10 \
  --bias-low 0.90 \
  --bias-high 1.10 \
  --bias-steps 9 \
  --rounds 2 \
  --include-own-models

echo
echo "[4/7] Diagnostics for logistic stacker candidate"
.venv/bin/python scripts/make_oof_generalization_diagnostics.py \
  --candidate-name stacker_with_ovr_xgb_realmlp_c010 \
  --candidate-oof artifacts/available_prediction_stacker_with_ovr_xgb_realmlp_c010/available_prediction_stacker_oof.npy \
  --candidate-test artifacts/available_prediction_stacker_with_ovr_xgb_realmlp_c010/available_prediction_stacker_test.npy \
  --output-dir artifacts/oof_diagnostics_stacker_with_ovr_xgb_realmlp_c010

echo
echo "[5/7] Diagnostics for greedy optimized candidate"
.venv/bin/python scripts/make_oof_generalization_diagnostics.py \
  --candidate-name optimized_with_ovr_xgb_realmlp_fast \
  --candidate-oof artifacts/oof_generalization_stack_with_ovr_xgb_realmlp_fast/generalization_stack_oof.npy \
  --candidate-test artifacts/oof_generalization_stack_with_ovr_xgb_realmlp_fast/generalization_stack_test.npy \
  --output-dir artifacts/oof_diagnostics_optimized_with_ovr_xgb_realmlp_fast

echo
echo "[6/7] Stable private candidate CSVs from greedy optimized candidate"
.venv/bin/python scripts/build_private_cv_stable_submissions.py \
  --output-dir artifacts/private_cv_stable_with_ovr_xgb_realmlp_fast \
  --candidate-oof artifacts/oof_generalization_stack_with_ovr_xgb_realmlp_fast/generalization_stack_oof.npy \
  --candidate-test artifacts/oof_generalization_stack_with_ovr_xgb_realmlp_fast/generalization_stack_test.npy \
  --stack-report artifacts/oof_generalization_stack_with_ovr_xgb_realmlp_fast/report.json \
  --folds 5 \
  --seeds 7 \
  --bins 10 \
  --top-k 4 \
  --output-rank-start 31

echo
echo "[7/7] Print key reports and generated artifacts"
.venv/bin/python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("artifacts/ovr_xgboost_realmlp_features/report.json"),
    Path("artifacts/available_prediction_stacker_with_ovr_xgb_realmlp_c010/report.json"),
    Path("artifacts/oof_generalization_stack_with_ovr_xgb_realmlp_fast/report.json"),
    Path("artifacts/oof_diagnostics_stacker_with_ovr_xgb_realmlp_c010/report.json"),
    Path("artifacts/oof_diagnostics_optimized_with_ovr_xgb_realmlp_fast/report.json"),
    Path("artifacts/private_cv_stable_with_ovr_xgb_realmlp_fast/report.json"),
]

score_keys = [
    "oof_balanced_accuracy",
    "full_oof_balanced_accuracy",
    "covered_oof_balanced_accuracy",
    "best_oof_balanced_accuracy",
    "raw_base_oof_balanced_accuracy",
    "delta_vs_raw_base",
    "base_raw_oof",
    "base_bias_oof",
    "aggressive_candidate_oof",
]

for path in paths:
    print(f"\n## {path}")
    if not path.exists():
        print("missing")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in score_keys:
        if key in data:
            print(f"{key}: {data[key]}")
    if "metrics" in data:
        metrics = data["metrics"]
        for key in ["base_oof_bac", "candidate_oof_bac", "oof_delta", "oof_changed_rows", "test_changed_rows"]:
            if key in metrics:
                print(f"metrics.{key}: {metrics[key]}")
    if "accepted_stages" in data:
        print("accepted_stages:")
        for stage in data["accepted_stages"]:
            print(f"  - {stage['stage']}: score={stage['score']} model={stage.get('model')} weight={stage.get('weight')}")
    if "selected_outputs" in data:
        print("selected_outputs:")
        for row in data["selected_outputs"]:
            print(f"  - {row['path']} OOF={row['oof_balanced_accuracy']} robust={row['robust_rank_score']}")
    if "outputs" in data:
        print("outputs:")
        for item in data["outputs"]:
            print(f"  - {item}")

print("\nGenerated output CSVs:")
for path in sorted(Path("outputs").glob("3*_PRIVATE_CV_*.csv")):
    print(f"  - {path}")
PY

echo "[next-queue] finished: $(date)"
echo "[next-queue] log: $LOG_PATH"
