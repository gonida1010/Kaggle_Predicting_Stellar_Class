from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "artifacts" / "boundary_pair_experiments"
OUT_DIR = ROOT / "artifacts" / "boundary_honest_oof_validation"
DATA = ROOT / "data"
PURE_DIR = ROOT / "artifacts" / "pure_model_ensemble"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate boundary rule candidates with an honest outer-fold rule-selection loop. "
            "This separates threshold/mask selection from the fold used for scoring."
        )
    )
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--pair", default="GALAXY:STAR")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional run directory names to evaluate. Defaults to all reports.",
    )
    parser.add_argument(
        "--mode",
        choices=["reported", "select", "narrow_select"],
        default="reported",
        help=(
            "reported evaluates the final reported rule fold-by-fold. "
            "select reruns threshold/mask selection on non-holdout folds and is much slower. "
            "narrow_select searches only near the reported rule."
        ),
    )
    parser.add_argument("--select-window", type=float, default=0.02)
    parser.add_argument("--select-steps", type=int, default=17)
    return parser.parse_args()


def load_boundary_module():
    path = ROOT / "scripts" / "train_boundary_pair_calibrator.py"
    spec = importlib.util.spec_from_file_location("boundary_pair_calibrator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def threshold_config(report: dict) -> tuple[float, float, float, float, int]:
    config = report.get("threshold_search", {})
    return (
        float(config.get("to_right_min", 0.52)),
        float(config.get("to_right_max", 0.94)),
        float(config.get("to_left_min", 0.06)),
        float(config.get("to_left_max", 0.48)),
        int(config.get("threshold_steps", 22)),
    )


def quick_search_rule(
    module,
    y: np.ndarray,
    base_pred: np.ndarray,
    right_oof: np.ndarray,
    masks: dict[str, np.ndarray],
    classes: list[str],
    left_idx: int,
    right_idx: int,
    select_mask: np.ndarray,
    reported_best: dict,
    window: float,
    steps: int,
) -> dict:
    base_score = balanced_accuracy_score(y[select_mask], base_pred[select_mask])
    reported_mask = reported_best.get("mask", "none")
    if reported_mask not in masks:
        return {"mask": "none", "score": base_score, "delta": 0.0}
    center = float(reported_best.get("to_right_threshold") or 0.5)
    to_left = float(reported_best.get("to_left_threshold") or 0.02)
    thresholds = np.linspace(max(0.0, center - window), min(1.0, center + window), max(3, steps))
    best = {
        "mask": reported_mask,
        "to_right_threshold": center,
        "to_left_threshold": to_left,
        "score": base_score,
        "delta": 0.0,
    }
    for threshold in thresholds:
        pred = module.apply_pair_override(
            base_pred,
            right_oof,
            masks[reported_mask],
            left_idx,
            right_idx,
            float(threshold),
            to_left,
        )
        score = balanced_accuracy_score(y[select_mask], pred[select_mask])
        if score > best["score"]:
            best = {
                "mask": reported_mask,
                "to_right_threshold": float(threshold),
                "to_left_threshold": to_left,
                "score": float(score),
                "delta": float(score - base_score),
            }
    return best


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    module = load_boundary_module()
    left, right = args.pair.split(":")

    train = pd.read_csv(DATA / "train.csv")
    classes = sorted(train["class"].astype(str).unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y = train["class"].astype(str).map(label_to_idx).to_numpy()

    cal_oof = np.load(PURE_DIR / "pure_model_ensemble_oof_proba.npy")
    base_pred = cal_oof.argmax(axis=1)
    base_score = balanced_accuracy_score(y, base_pred)

    train_fe = module.add_advanced_features(train)
    masks = module.segment_masks(train_fe, cal_oof, classes, left, right)
    left_idx = classes.index(left)
    right_idx = classes.index(right)

    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(cv.split(train_fe, y))

    summary_rows = []
    detail_rows = []

    for report_path in sorted(args.experiment_root.glob("*/boundary_pair_calibrator_report.json")):
        run_dir = report_path.parent
        if args.runs is not None and run_dir.name not in set(args.runs):
            continue
        right_path = run_dir / f"{left}_{right}_right_oof.npy"
        if not right_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        pair_payload = report.get("pair_outputs", {}).get(args.pair)
        if not pair_payload:
            continue
        reported_best = pair_payload.get("best", {})

        right_oof = np.load(right_path)
        honest_pred = base_pred.copy()
        fold_deltas = []
        selected_rules = []

        for fold, (_, hold_idx) in enumerate(splits, start=1):
            hold_mask = np.zeros(len(y), dtype=bool)
            hold_mask[hold_idx] = True
            select_mask = ~hold_mask
            select_mask &= ~np.isnan(right_oof)
            hold_mask &= ~np.isnan(right_oof)

            if args.mode == "select":
                best, _ = module.search_pair_rule(
                    y,
                    base_pred,
                    right_oof,
                    masks,
                    classes,
                    left,
                    right,
                    select_mask,
                    *threshold_config(report),
                )
            elif args.mode == "narrow_select":
                best = quick_search_rule(
                    module,
                    y,
                    base_pred,
                    right_oof,
                    masks,
                    classes,
                    left_idx,
                    right_idx,
                    select_mask,
                    reported_best,
                    args.select_window,
                    args.select_steps,
                )
            else:
                best = reported_best

            if best["mask"] == "none":
                fold_pred = base_pred.copy()
            else:
                fold_pred = module.apply_pair_override(
                    base_pred,
                    right_oof,
                    masks[best["mask"]],
                    left_idx,
                    right_idx,
                    best["to_right_threshold"],
                    best["to_left_threshold"],
                )

            honest_pred[hold_mask] = fold_pred[hold_mask]
            fold_base = balanced_accuracy_score(y[hold_mask], base_pred[hold_mask])
            fold_score = balanced_accuracy_score(y[hold_mask], fold_pred[hold_mask])
            fold_delta = float(fold_score - fold_base)
            fold_deltas.append(fold_delta)
            selected_rules.append(
                (
                    best["mask"],
                    round(float(best["to_right_threshold"] or 0.0), 6),
                    round(float(best["to_left_threshold"] or 0.0), 6),
                )
            )
            detail_rows.append(
                {
                    "run": run_dir.name,
                    "fold": fold,
                    "selected_mask": best["mask"],
                    "selected_to_right": best["to_right_threshold"],
                    "selected_to_left": best["to_left_threshold"],
                    "selector_delta": best.get("delta"),
                    "holdout_base_score": fold_base,
                    "holdout_score": fold_score,
                    "holdout_delta": fold_delta,
                    "holdout_changed_rows": int((fold_pred[hold_mask] != base_pred[hold_mask]).sum()),
                }
            )

        honest_score = balanced_accuracy_score(y, honest_pred)
        optimistic = float(report.get("combined_delta", np.nan))
        summary_rows.append(
            {
                "run": run_dir.name,
                "mode": args.mode,
                "accepted": bool(report.get("accepted_as_candidate")),
                "optimistic_oof": report.get("combined_eval_oof_balanced_accuracy"),
                "optimistic_delta": optimistic,
                "honest_oof": honest_score,
                "honest_delta": float(honest_score - base_score),
                "optimism_gap": float(optimistic - (honest_score - base_score)),
                "mean_fold_holdout_delta": float(np.mean(fold_deltas)),
                "min_fold_holdout_delta": float(np.min(fold_deltas)),
                "max_fold_holdout_delta": float(np.max(fold_deltas)),
                "rule_count": json.dumps(Counter(selected_rules).most_common(), ensure_ascii=False),
                "reported_mask": best.get("mask"),
                "reported_to_right": best.get("to_right_threshold"),
                "reported_changed_rows": best.get("changed_rows"),
                "submission_path": report.get("submission_path"),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["honest_delta", "optimistic_delta"], ascending=False)
    detail = pd.DataFrame(detail_rows)
    summary_path = args.output_dir / "honest_boundary_oof_summary.csv"
    detail_path = args.output_dir / "honest_boundary_oof_folds.csv"
    summary.to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)

    print(f"base_oof={base_score:.9f}")
    print(summary[["run", "optimistic_delta", "honest_delta", "optimism_gap", "min_fold_holdout_delta"]].to_string(index=False))
    print(f"wrote {summary_path}")
    print(f"wrote {detail_path}")


if __name__ == "__main__":
    main()
