#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p artifacts/overnight_logs outputs
LOG="artifacts/overnight_logs/classwise_blender_$(date +%Y%m%d_%H%M%S).log"
PY=".venv/bin/python"

exec > >(tee -a "$LOG") 2>&1

echo "[classwise] log: $LOG"
echo "[classwise] started: $(date)"

echo "[1/6] Train class-wise logistic blender"
"$PY" scripts/train_classwise_logistic_blender.py \
  --output-dir artifacts/classwise_logistic_blender_c010 \
  --folds 5 \
  --seeds 5 \
  --c 0.10 \
  --max-iter 800 \
  --positive-weight 1.0 \
  --boost-star 1.0 \
  --bias-low 0.88 \
  --bias-high 1.14 \
  --bias-steps 27 \
  --log-period 1

echo "[2/6] Diagnose class-wise blender against 07 reference"
"$PY" scripts/make_oof_generalization_diagnostics.py \
  --output-dir artifacts/oof_diagnostics_classwise_blender_c010 \
  --base-name 07_lr_v9 \
  --candidate-name classwise_blender_c010 \
  --candidate-oof artifacts/classwise_logistic_blender_c010/classwise_blender_oof.npy \
  --candidate-test artifacts/classwise_logistic_blender_c010/classwise_blender_test.npy

echo "[3/6] OOF-only greedy optimizer with class-wise blender registered"
"$PY" scripts/optimize_oof_generalization_stack.py \
  --output-dir artifacts/oof_generalization_stack_with_classwise_blender_fast \
  --base-model lr-stacker-v9-public-oof \
  --max-added-weight 0.35 \
  --weight-steps 36 \
  --bias-low 0.88 \
  --bias-high 1.14 \
  --bias-steps 27 \
  --rounds 3

echo "[4/6] Diagnose optimized class-wise stack"
"$PY" scripts/make_oof_generalization_diagnostics.py \
  --output-dir artifacts/oof_diagnostics_optimized_with_classwise_blender_fast \
  --base-name 07_lr_v9 \
  --candidate-name optimized_with_classwise_blender \
  --candidate-oof artifacts/oof_generalization_stack_with_classwise_blender_fast/generalization_stack_oof.npy \
  --candidate-test artifacts/oof_generalization_stack_with_classwise_blender_fast/generalization_stack_test.npy

echo "[5/6] Build guarded private-CV stable submissions"
"$PY" scripts/build_private_cv_stable_submissions.py \
  --output-dir artifacts/private_cv_stable_with_classwise_blender_fast \
  --candidate-oof artifacts/oof_generalization_stack_with_classwise_blender_fast/generalization_stack_oof.npy \
  --candidate-test artifacts/oof_generalization_stack_with_classwise_blender_fast/generalization_stack_test.npy \
  --stack-report artifacts/oof_generalization_stack_with_classwise_blender_fast/report.json \
  --output-rank-start 40 \
  --top-k 4

echo "[6/6] Copy direct high-CV candidates to outputs"
"$PY" - <<'PY'
import json
import shutil
from pathlib import Path

root = Path.cwd()
jobs = [
    (
        root / "artifacts/classwise_logistic_blender_c010/report.json",
        root / "artifacts/classwise_logistic_blender_c010/classwise_blender_submission.csv",
        "36_PRIVATE_CV_classwise_blender_direct",
        "biased_oof_balanced_accuracy",
    ),
    (
        root / "artifacts/oof_generalization_stack_with_classwise_blender_fast/report.json",
        root / "artifacts/oof_generalization_stack_with_classwise_blender_fast/generalization_stack_submission.csv",
        "37_PRIVATE_CV_greedy_with_classwise_blender",
        "best_oof_balanced_accuracy",
    ),
]
for report_path, src, prefix, key in jobs:
    report = json.loads(report_path.read_text())
    score = float(report[key])
    score_tag = f"{score:.6f}".replace(".", "")
    dst = root / "outputs" / f"{prefix}_oof{score_tag}.csv"
    shutil.copyfile(src, dst)
    print(f"[classwise] copied {src.relative_to(root)} -> {dst.relative_to(root)} score={score:.9f}")
PY

echo "[classwise] reports"
"$PY" - <<'PY'
import json
from pathlib import Path

paths = [
    "artifacts/classwise_logistic_blender_c010/report.json",
    "artifacts/oof_generalization_stack_with_classwise_blender_fast/report.json",
    "artifacts/private_cv_stable_with_classwise_blender_fast/report.json",
]
for p in paths:
    print(f"\n### {p}")
    d = json.loads(Path(p).read_text())
    for key in [
        "reference_oof_balanced_accuracy",
        "raw_oof_balanced_accuracy",
        "biased_oof_balanced_accuracy",
        "delta_vs_reference",
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

echo "[classwise] finished: $(date)"
