#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p artifacts/overnight_logs outputs
LOG="artifacts/overnight_logs/catv3_views_$(date +%Y%m%d_%H%M%S).log"
PY=".venv/bin/python"

exec > >(tee -a "$LOG") 2>&1

echo "[catv3] log: $LOG"
echo "[catv3] started: $(date)"
echo "[catv3] purpose: CatBoost v3-style categorical views, then OOF-only stack/guarded candidates"

echo "[1/7] Train CatBoost with catv3 categorical views"
"$PY" scripts/train_catboost_cv.py \
  --output-dir artifacts/catboost_cv_catv3_views \
  --feature-set catv3 \
  --fold-limit 5 \
  --iterations 9000 \
  --early-stopping-rounds 1000 \
  --early-stop-metric valid-bac \
  --learning-rate 0.030 \
  --depth 7 \
  --l2-leaf-reg 12 \
  --random-strength 1.0 \
  --bagging-temperature 0.55 \
  --prediction-iteration-policy early-stop-best \
  --diagnostic-period 250 \
  --diagnostic-train-sample 50000 \
  --log-period 250

echo "[2/7] Build available prediction stacker with catv3 model registered"
"$PY" scripts/build_available_prediction_stacker.py \
  --output-dir artifacts/available_prediction_stacker_with_catv3_c010 \
  --c 0.10 \
  --epochs 800 \
  --seeds 5 \
  --folds 5

echo "[3/7] OOF source quality/diversity analysis with catv3 registered"
"$PY" scripts/analyze_oof_source_diversity.py \
  --output-dir artifacts/oof_source_diversity_with_catv3 \
  --top-n 25

echo "[4/7] OOF-only greedy optimizer with catv3 model registered"
"$PY" scripts/optimize_oof_generalization_stack.py \
  --output-dir artifacts/oof_generalization_stack_with_catv3_fast \
  --base-model lr-stacker-v9-public-oof \
  --max-added-weight 0.35 \
  --weight-steps 36 \
  --bias-low 0.88 \
  --bias-high 1.14 \
  --bias-steps 27 \
  --rounds 3

echo "[5/7] Diagnose optimized catv3 stack"
"$PY" scripts/make_oof_generalization_diagnostics.py \
  --output-dir artifacts/oof_diagnostics_optimized_with_catv3_fast \
  --base-name 07_lr_v9 \
  --candidate-name optimized_with_catv3 \
  --candidate-oof artifacts/oof_generalization_stack_with_catv3_fast/generalization_stack_oof.npy \
  --candidate-test artifacts/oof_generalization_stack_with_catv3_fast/generalization_stack_test.npy

echo "[6/7] Build guarded private-CV stable submissions"
"$PY" scripts/build_private_cv_stable_submissions.py \
  --output-dir artifacts/private_cv_stable_with_catv3_fast \
  --candidate-oof artifacts/oof_generalization_stack_with_catv3_fast/generalization_stack_oof.npy \
  --candidate-test artifacts/oof_generalization_stack_with_catv3_fast/generalization_stack_test.npy \
  --stack-report artifacts/oof_generalization_stack_with_catv3_fast/report.json \
  --output-rank-start 64 \
  --top-k 4

echo "[7/7] Copy direct high-CV candidates to outputs"
"$PY" - <<'PY'
import json
import shutil
from pathlib import Path

root = Path.cwd()
jobs = [
    (
        root / "artifacts/catboost_cv_catv3_views/catboost_baseline_report.json",
        root / "artifacts/catboost_cv_catv3_views/catboost_baseline_submission.csv",
        "64_PRIVATE_CV_catboost_catv3_direct",
        "oof_balanced_accuracy",
    ),
    (
        root / "artifacts/available_prediction_stacker_with_catv3_c010/report.json",
        root / "artifacts/available_prediction_stacker_with_catv3_c010/available_prediction_stacker_submission.csv",
        "65_PRIVATE_CV_available_stacker_with_catv3",
        "biased_oof_balanced_accuracy",
    ),
    (
        root / "artifacts/oof_generalization_stack_with_catv3_fast/report.json",
        root / "artifacts/oof_generalization_stack_with_catv3_fast/generalization_stack_submission.csv",
        "66_PRIVATE_CV_greedy_with_catv3",
        "best_oof_balanced_accuracy",
    ),
]
for report_path, src, prefix, key in jobs:
    report = json.loads(report_path.read_text())
    score = report.get(key)
    if score is None:
        score = report.get("covered_oof_balanced_accuracy")
    score = float(score)
    score_tag = f"{score:.6f}".replace(".", "")
    dst = root / "outputs" / f"{prefix}_oof{score_tag}.csv"
    shutil.copyfile(src, dst)
    print(f"[catv3] copied {src.relative_to(root)} -> {dst.relative_to(root)} score={score:.9f}")
PY

echo "[catv3] reports"
"$PY" - <<'PY'
import json
from pathlib import Path

paths = [
    "artifacts/catboost_cv_catv3_views/catboost_baseline_report.json",
    "artifacts/available_prediction_stacker_with_catv3_c010/report.json",
    "artifacts/oof_source_diversity_with_catv3/report.json",
    "artifacts/oof_generalization_stack_with_catv3_fast/report.json",
    "artifacts/oof_diagnostics_optimized_with_catv3_fast/report.json",
    "artifacts/private_cv_stable_with_catv3_fast/report.json",
]
for p in paths:
    print(f"\n### {p}")
    d = json.loads(Path(p).read_text())
    for key in [
        "feature_set",
        "oof_balanced_accuracy",
        "covered_oof_balanced_accuracy",
        "raw_oof_balanced_accuracy",
        "biased_oof_balanced_accuracy",
        "raw_base_oof_balanced_accuracy",
        "best_oof_balanced_accuracy",
        "delta_vs_raw_base",
        "delta_vs_reference",
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

echo "[catv3] finished: $(date)"
