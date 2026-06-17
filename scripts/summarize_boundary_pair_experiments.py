from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "artifacts" / "boundary_pair_experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize boundary pair calibrator experiment reports.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "experiment_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = sorted(args.root.glob("*/boundary_pair_calibrator_report.json"))
    rows = []
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        row = {
            "run": report_path.parent.name,
            "accepted": report.get("accepted_as_candidate"),
            "base_oof": report.get("base_eval_oof_balanced_accuracy"),
            "combined_oof": report.get("combined_eval_oof_balanced_accuracy"),
            "combined_delta": report.get("combined_delta"),
            "prediction_iteration_policy": report.get("prediction_iteration_policy"),
            "n_estimators": report.get("model_params", {}).get("n_estimators"),
            "learning_rate": report.get("model_params", {}).get("learning_rate"),
            "num_leaves": report.get("model_params", {}).get("num_leaves"),
            "max_depth": report.get("model_params", {}).get("max_depth"),
            "min_child_samples": report.get("model_params", {}).get("min_child_samples"),
            "reg_alpha": report.get("model_params", {}).get("reg_alpha"),
            "reg_lambda": report.get("model_params", {}).get("reg_lambda"),
            "submission_path": report.get("submission_path", ""),
        }
        for pair, payload in report.get("pair_outputs", {}).items():
            best = payload.get("best", {})
            prefix = pair.replace(":", "_")
            row[f"{prefix}_delta"] = best.get("delta")
            row[f"{prefix}_changed_rows"] = best.get("changed_rows")
            row[f"{prefix}_mask"] = best.get("mask")
            row[f"{prefix}_to_right"] = best.get("to_right_threshold")
            row[f"{prefix}_to_left"] = best.get("to_left_threshold")
            row[f"{prefix}_transitions"] = json.dumps(best.get("transition_counts", {}), ensure_ascii=False)
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No boundary pair reports found under {args.root}")

    df = pd.DataFrame(rows).sort_values("combined_delta", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.head(20).to_string(index=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
