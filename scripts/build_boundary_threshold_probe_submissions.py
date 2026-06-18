from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import add_advanced_features  # noqa: E402
from scripts.train_boundary_pair_calibrator import (  # noqa: E402
    apply_pair_override,
    balanced_accuracy,
    class_recalls,
    load_pure_arrays,
    segment_masks,
    transition_counts,
)


DATA = ROOT / "data"
EXPERIMENT_ROOT = ROOT / "artifacts" / "boundary_pair_experiments"
OUT_DIR = ROOT / "artifacts" / "boundary_threshold_probes"
LABELS = ["GALAXY", "QSO", "STAR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build forced-threshold GALAXY->STAR probe submissions from saved boundary probabilities."
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--pair", default="GALAXY:STAR")
    parser.add_argument("--mask", default="mid_redshift_and_pairsum085")
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--to-left-threshold", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def parse_pair(pair: str) -> tuple[str, str]:
    parts = pair.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid pair: {pair}")
    return parts[0], parts[1]


def safe_threshold_name(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def main() -> None:
    args = parse_args()
    run_dir = EXPERIMENT_ROOT / args.run
    report_path = run_dir / "boundary_pair_calibrator_report.json"
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pair_payload = report.get("pair_outputs", {}).get(args.pair)
    if not pair_payload:
        raise ValueError(f"Pair {args.pair} missing from {report_path}")

    left, right = parse_pair(args.pair)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["class"].astype(str))
    classes = encoder.classes_.tolist()
    if classes != LABELS:
        raise ValueError(f"Unexpected classes: {classes}")

    cal_oof, cal_test, raw_oof, raw_test, _ = load_pure_arrays(classes)
    base_oof_pred = cal_oof.argmax(axis=1)
    base_test_pred = cal_test.argmax(axis=1)
    base_score = balanced_accuracy(y, base_oof_pred, len(classes))

    right_oof = np.load(run_dir / pair_payload["oof_path"]).astype(np.float32)
    right_test = np.load(run_dir / pair_payload["test_path"]).astype(np.float32)
    train_masks = segment_masks(add_advanced_features(train), cal_oof, classes, left, right)
    test_masks = segment_masks(add_advanced_features(test), cal_test, classes, left, right)
    if args.mask not in train_masks or args.mask not in test_masks:
        raise ValueError(f"Unknown mask {args.mask}. Available: {sorted(train_masks)}")

    left_idx = classes.index(left)
    right_idx = classes.index(right)
    rows = []
    for threshold in args.thresholds:
        oof_pred = apply_pair_override(
            base_oof_pred,
            right_oof,
            train_masks[args.mask],
            left_idx,
            right_idx,
            threshold,
            args.to_left_threshold,
        )
        test_pred = apply_pair_override(
            base_test_pred,
            right_test,
            test_masks[args.mask],
            left_idx,
            right_idx,
            threshold,
            args.to_left_threshold,
        )
        score = balanced_accuracy(y, oof_pred, len(classes))
        delta = score - base_score
        submission = sample.copy()
        submission["class"] = encoder.inverse_transform(test_pred)
        filename = f"{args.run}_{args.mask}_thr{safe_threshold_name(threshold)}.csv"
        path = args.output_dir / filename
        submission.to_csv(path, index=False)
        row = {
            "submission_path": str(path.relative_to(ROOT)),
            "run": args.run,
            "pair": args.pair,
            "mask": args.mask,
            "to_right_threshold": threshold,
            "to_left_threshold": args.to_left_threshold,
            "oof_score": score,
            "oof_delta": delta,
            "oof_transition_counts": json.dumps(transition_counts(base_oof_pred, oof_pred, classes), ensure_ascii=False),
            "test_transition_counts": json.dumps(transition_counts(base_test_pred, test_pred, classes), ensure_ascii=False),
            "class_recalls": json.dumps(class_recalls(y, oof_pred, classes), ensure_ascii=False),
        }
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("oof_delta", ascending=False)
    out.to_csv(args.output_dir / f"{args.run}_threshold_probe_report.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
