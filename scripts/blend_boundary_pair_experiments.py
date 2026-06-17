from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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
    display_path,
    load_pure_arrays,
    search_pair_rule,
    segment_masks,
    transition_counts,
)


DATA = ROOT / "data"
DEFAULT_ROOT = ROOT / "artifacts" / "boundary_pair_experiments"
DEFAULT_OUT = ROOT / "artifacts" / "boundary_pair_blends"
LABELS = ["GALAXY", "QSO", "STAR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend saved boundary-pair experiment probabilities and search leak-safe OOF override rules."
    )
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--pair", default="GALAXY:STAR")
    parser.add_argument("--to-right-min", type=float, default=0.40)
    parser.add_argument("--to-right-max", type=float, default=0.58)
    parser.add_argument("--to-left-min", type=float, default=0.02)
    parser.add_argument("--to-left-max", type=float, default=0.24)
    parser.add_argument("--threshold-steps", type=int, default=37)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--write-even-if-worse", action="store_true")
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_pair(pair: str) -> tuple[str, str]:
    parts = pair.split(":")
    if len(parts) != 2 or parts[0] == parts[1]:
        raise ValueError(f"Invalid pair: {pair}. Use LEFT:RIGHT.")
    return parts[0], parts[1]


def read_report(run_dir: Path) -> dict:
    path = run_dir / "boundary_pair_calibrator_report.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_probability(run_dir: Path, pair: str) -> tuple[np.ndarray, np.ndarray, dict]:
    report = read_report(run_dir)
    pair_payload = report.get("pair_outputs", {}).get(pair)
    if not pair_payload:
        raise ValueError(f"Pair {pair} is missing from {run_dir}")
    oof_path = run_dir / pair_payload["oof_path"]
    test_path = run_dir / pair_payload["test_path"]
    if not oof_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing probability arrays for {run_dir}")
    return np.load(oof_path).astype(np.float32), np.load(test_path).astype(np.float32), report


def normalize_weights(weights: list[float], n: int) -> np.ndarray:
    if len(weights) != n:
        raise ValueError(f"weights length {len(weights)} != runs length {n}")
    arr = np.array(weights, dtype=np.float64)
    if np.any(arr < 0) or arr.sum() <= 0:
        raise ValueError("--weights must be non-negative and sum to a positive value.")
    return arr / arr.sum()


def main() -> None:
    args = parse_args()
    args.experiment_root = args.experiment_root.resolve()
    args.output_dir = (args.output_dir / args.name).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    left, right = parse_pair(args.pair)
    weights = normalize_weights(args.weights or [1.0] * len(args.runs), len(args.runs))

    progress("Loading train/test/sample data")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["class"].astype(str))
    classes = encoder.classes_.tolist()
    if classes != LABELS:
        raise ValueError(f"Unexpected class order: {classes}")
    if left not in classes or right not in classes:
        raise ValueError(f"Pair {args.pair} is not valid for classes {classes}")

    progress("Loading pure model probabilities")
    cal_oof, cal_test, raw_oof, raw_test, pure_report = load_pure_arrays(classes)
    base_oof_pred = cal_oof.argmax(axis=1)
    base_test_pred = cal_test.argmax(axis=1)
    base_score = balanced_accuracy(y, base_oof_pred, len(classes))

    progress("Loading boundary experiment probabilities")
    blend_oof = np.zeros(len(train), dtype=np.float64)
    blend_test = np.zeros(len(test), dtype=np.float64)
    run_rows = []
    for run_name, weight in zip(args.runs, weights):
        run_dir = args.experiment_root / run_name
        right_oof, right_test, report = load_run_probability(run_dir, args.pair)
        blend_oof += float(weight) * right_oof
        blend_test += float(weight) * right_test
        best = report.get("pair_outputs", {}).get(args.pair, {}).get("best", {})
        run_rows.append(
            {
                "run": run_name,
                "weight": float(weight),
                "run_combined_delta": report.get("combined_delta"),
                "run_pair_delta": best.get("delta"),
                "run_changed_rows": best.get("changed_rows"),
                "run_to_right_threshold": best.get("to_right_threshold"),
                "run_mask": best.get("mask"),
            }
        )

    progress("Building subset masks")
    train_fe = add_advanced_features(train)
    test_fe = add_advanced_features(test)
    train_masks = segment_masks(train_fe, cal_oof, classes, left, right)
    test_masks = segment_masks(test_fe, cal_test, classes, left, right)

    progress("Searching blended OOF override thresholds")
    covered = np.isfinite(blend_oof)
    best, search_top = search_pair_rule(
        y,
        base_oof_pred,
        blend_oof.astype(np.float32),
        train_masks,
        classes,
        left,
        right,
        covered,
        args.to_right_min,
        args.to_right_max,
        args.to_left_min,
        args.to_left_max,
        args.threshold_steps,
    )
    search_df = pd.DataFrame(search_top)
    if not search_df.empty:
        search_df["class_recalls"] = search_df["class_recalls"].apply(json.dumps, ensure_ascii=False)
        search_df["transition_counts"] = search_df["transition_counts"].apply(json.dumps, ensure_ascii=False)
        search_df.to_csv(args.output_dir / "blend_search_top.csv", index=False)

    left_idx = classes.index(left)
    right_idx = classes.index(right)
    combined_oof_pred = base_oof_pred.copy()
    combined_test_pred = base_test_pred.copy()
    if best["mask"] != "none" and best["delta"] > 0:
        combined_oof_pred = apply_pair_override(
            base_oof_pred,
            blend_oof,
            train_masks[best["mask"]],
            left_idx,
            right_idx,
            best["to_right_threshold"],
            best["to_left_threshold"],
        )
        combined_test_pred = apply_pair_override(
            base_test_pred,
            blend_test,
            test_masks[best["mask"]],
            left_idx,
            right_idx,
            best["to_right_threshold"],
            best["to_left_threshold"],
        )

    combined_score = balanced_accuracy(y, combined_oof_pred, len(classes))
    accepted = combined_score > base_score + args.min_delta or args.write_even_if_worse
    report = {
        "purpose": "Weighted blend of saved boundary-pair OOF/test probabilities. No public submission CSV is used.",
        "pair": args.pair,
        "runs": run_rows,
        "threshold_search": {
            "to_right_min": args.to_right_min,
            "to_right_max": args.to_right_max,
            "to_left_min": args.to_left_min,
            "to_left_max": args.to_left_max,
            "threshold_steps": args.threshold_steps,
        },
        "base_oof_balanced_accuracy": float(base_score),
        "combined_oof_balanced_accuracy": float(combined_score),
        "combined_delta": float(combined_score - base_score),
        "accepted_as_candidate": bool(accepted),
        "best_override": best,
        "oof_transition_counts": transition_counts(base_oof_pred, combined_oof_pred, classes),
        "class_recalls": class_recalls(y, combined_oof_pred, classes),
        "pure_report_best_config": pure_report.get("best_config"),
    }

    np.save(args.output_dir / "blend_right_oof.npy", blend_oof.astype(np.float32))
    np.save(args.output_dir / "blend_right_test.npy", blend_test.astype(np.float32))
    pd.DataFrame(run_rows).to_csv(args.output_dir / "blend_runs.csv", index=False)

    if accepted:
        submission = sample.copy()
        submission["class"] = encoder.inverse_transform(combined_test_pred)
        submission_path = args.output_dir / "boundary_pair_blend_submission.csv"
        submission.to_csv(submission_path, index=False)
        report["submission_path"] = display_path(submission_path)
        report["submission_class_share"] = submission["class"].value_counts(normalize=True).sort_index().to_dict()

    (args.output_dir / "boundary_pair_blend_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
