#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p artifacts/overnight_logs outputs
LOG="artifacts/overnight_logs/foldsafe_te_$(date +%Y%m%d_%H%M%S).log"
PY=".venv/bin/python"

exec > >(tee -a "$LOG") 2>&1

echo "[te] log: $LOG"
echo "[te] started: $(date)"

echo "[1/6] Train fold-safe target-encoded LightGBM"
"$PY" scripts/train_lgbm_foldsafe_te_cv.py \
  --output-dir artifacts/lgbm_foldsafe_te_realmlp \
  --fold-limit 5 \
  --n-estimators 5000 \
  --early-stopping-rounds 220 \
  --learning-rate 0.028 \
  --num-leaves 96 \
  --max-depth -1 \
  --min-child-samples 90 \
  --subsample 0.88 \
  --colsample-bytree 0.84 \
  --reg-alpha 0.10 \
  --reg-lambda 2.4 \
  --early-stop-metric valid-bac \
  --prediction-iteration-policy early-stop-best \
  --te-smoothing 40 \
  --log-period 200 \
  --diagnostic-period 100 \
  --diagnostic-train-sample 50000

echo "[2/6] Build available prediction stacker with TE model registered"
"$PY" scripts/build_available_prediction_stacker.py \
  --output-dir artifacts/available_prediction_stacker_with_foldsafe_te_c010 \
  --c 0.10 \
  --epochs 800 \
  --seeds 5 \
  --folds 5

echo "[3/6] OOF-only greedy optimizer with TE model registered"
"$PY" scripts/optimize_oof_generalization_stack.py \
  --output-dir artifacts/oof_generalization_stack_with_foldsafe_te_fast \
  --base-model lr-stacker-v9-public-oof \
  --max-added-weight 0.35 \
  --weight-steps 36 \
  --bias-low 0.88 \
  --bias-high 1.14 \
  --bias-steps 27 \
  --rounds 3

echo "[4/6] Diagnose optimized TE stack"
"$PY" scripts/make_oof_generalization_diagnostics.py \
  --output-dir artifacts/oof_diagnostics_optimized_with_foldsafe_te_fast \
  --base-name 07_lr_v9 \
  --candidate-name optimized_with_foldsafe_te \
  --candidate-oof artifacts/oof_generalization_stack_with_foldsafe_te_fast/generalization_stack_oof.npy \
  --candidate-test artifacts/oof_generalization_stack_with_foldsafe_te_fast/generalization_stack_test.npy

echo "[5/6] Build guarded private-CV stable submissions"
"$PY" scripts/build_private_cv_stable_submissions.py \
  --output-dir artifacts/private_cv_stable_with_foldsafe_te_fast \
  --candidate-oof artifacts/oof_generalization_stack_with_foldsafe_te_fast/generalization_stack_oof.npy \
  --candidate-test artifacts/oof_generalization_stack_with_foldsafe_te_fast/generalization_stack_test.npy \
  --stack-report artifacts/oof_generalization_stack_with_foldsafe_te_fast/report.json \
  --output-rank-start 52 \
  --top-k 4

echo "[6/6] Copy direct high-CV candidates to outputs"
"$PY" - <<'PY'
import json
import shutil
from pathlib import Path

root = Path.cwd()
jobs = [
    (
        root / "artifacts/lgbm_foldsafe_te_realmlp/lgbm_te_report.json",
        root / "artifacts/lgbm_foldsafe_te_realmlp/lgbm_te_submission.csv",
        "50_PRIVATE_CV_lgbm_foldsafe_te_direct",
        "oof_balanced_accuracy",
    ),
    (
        root / "artifacts/oof_generalization_stack_with_foldsafe_te_fast/report.json",
        root / "artifacts/oof_generalization_stack_with_foldsafe_te_fast/generalization_stack_submission.csv",
        "51_PRIVATE_CV_greedy_with_foldsafe_te",
        "best_oof_balanced_accuracy",
    ),
]
for report_path, src, prefix, key in jobs:
    report = json.loads(report_path.read_text())
    score = report[key]
    if score is None:
        score = report.get("covered_oof_balanced_accuracy")
    score = float(score)
    score_tag = f"{score:.6f}".replace(".", "")
    dst = root / "outputs" / f"{prefix}_oof{score_tag}.csv"
    shutil.copyfile(src, dst)
    print(f"[te] copied {src.relative_to(root)} -> {dst.relative_to(root)} score={score:.9f}")
PY

echo "[te] reports"
"$PY" - <<'PY'
import json
from pathlib import Path

paths = [
    "artifacts/lgbm_foldsafe_te_realmlp/lgbm_te_report.json",
    "artifacts/available_prediction_stacker_with_foldsafe_te_c010/report.json",
    "artifacts/oof_generalization_stack_with_foldsafe_te_fast/report.json",
    "artifacts/private_cv_stable_with_foldsafe_te_fast/report.json",
]
for p in paths:
    print(f"\n### {p}")
    d = json.loads(Path(p).read_text())
    for key in [
        "oof_balanced_accuracy",
        "covered_oof_balanced_accuracy",
        "oof_balanced_accuracy",
        "raw_base_oof_balanced_accuracy",
        "best_oof_balanced_accuracy",
        "delta_vs_raw_base",
    ]:
        if key in d:
            print(f"{key}: {d[key]}")
    if "accepted_stages" in d:
        print("accepted_stages:")
        for row in d["accepted_stages"]:
            print(row)
    if "selected_outputs" in d:
        print("selected_outputs:")
        for row in d["selected_outputs"]:
            print(row)
PY

echo "[te] finished: $(date)"
