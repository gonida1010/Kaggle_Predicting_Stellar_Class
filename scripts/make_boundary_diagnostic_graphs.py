from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.train_boundary_pair_calibrator import (  # noqa: E402
    write_pair_panel_metric_svg,
    write_train_valid_panel_svg,
)


OUT_DIR = ROOT / "artifacts" / "boundary_pair_calibrator"


def main() -> None:
    diagnostics_path = OUT_DIR / "boundary_pair_training_diagnostics.csv"
    if not diagnostics_path.exists():
        raise FileNotFoundError(
            f"Missing diagnostics file: {diagnostics_path}. "
            "Run train_boundary_pair_calibrator.py with --diagnostic-period first."
        )
    rows = pd.read_csv(diagnostics_path).to_dict("records")
    write_pair_panel_metric_svg(
        OUT_DIR / "boundary_pair_valid_logloss_by_pair.svg",
        "Boundary Pair Valid Binary Logloss By Pair",
        rows,
        "valid_binary_logloss",
        "valid logloss",
    )
    write_pair_panel_metric_svg(
        OUT_DIR / "boundary_pair_valid_balanced_accuracy_by_pair.svg",
        "Boundary Pair Valid Binary Balanced Accuracy By Pair",
        rows,
        "valid_binary_balanced_accuracy",
        "valid balanced accuracy",
    )
    write_train_valid_panel_svg(
        OUT_DIR / "boundary_pair_train_valid_accuracy_by_pair.svg",
        "Boundary Pair Train vs Valid Balanced Accuracy",
        rows,
    )
    print("wrote corrected boundary diagnostic graphs:")
    print(f"- {OUT_DIR / 'boundary_pair_valid_logloss_by_pair.svg'}")
    print(f"- {OUT_DIR / 'boundary_pair_valid_balanced_accuracy_by_pair.svg'}")
    print(f"- {OUT_DIR / 'boundary_pair_train_valid_accuracy_by_pair.svg'}")


if __name__ == "__main__":
    main()
