from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

TARGET = "class"
ID = "id"
CLASSES = ["GALAXY", "QSO", "STAR"]

ENSEMBLE_FILES = [
    "0.97047.b.csv",
    "0.97101.csv",
    "0.97108.csv",
    "0.97111.csv",
    "0.97122.csv",
]

PUBLIC_BLEND_MICRO_PATCH = {
    665223: "GALAXY",
    676483: "GALAXY",
    755752: "GALAXY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the public 0.97137 protected-anchor blend from the shared recipe."
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=ROOT / "external_preds",
        help="Directory containing the public submission bank csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS / "public_blend_anchor",
        help="Directory where rebuilt submission and diagnostics will be written.",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=DATA / "submission.csv",
        help="Optional existing submission to compare with the rebuilt output.",
    )
    return parser.parse_args()


def read_prediction(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != [ID, TARGET]:
        raise ValueError(f"{path.name} columns must be {[ID, TARGET]}, got {df.columns.tolist()}")
    if not df[ID].equals(sample[ID]):
        raise ValueError(f"{path.name} id order differs from sample_submission.csv")
    bad_labels = sorted(set(df[TARGET].dropna()) - set(CLASSES))
    if bad_labels:
        raise ValueError(f"{path.name} has invalid labels: {bad_labels}")
    return df.rename(columns={TARGET: path.name})


def build_blend(prediction_dir: Path, sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [name for name in ENSEMBLE_FILES if not (prediction_dir / name).exists()]
    if missing:
        message = [
            "Missing public submission-bank files.",
            f"prediction_dir: {prediction_dir}",
            "Required files:",
            *[f"- {name}" for name in ENSEMBLE_FILES],
            "Put the shared Kaggle dataset files under external_preds/ or pass --prediction-dir.",
        ]
        raise FileNotFoundError("\n".join(message))

    selected = [read_prediction(prediction_dir / name, sample) for name in ENSEMBLE_FILES]
    blend = selected[0]
    for df in selected[1:]:
        blend = blend.merge(df, on=ID, how="inner")

    anchor_col = ENSEMBLE_FILES[-1]
    companion_cols = ENSEMBLE_FILES[:-1]

    first_four_unanimous = blend[companion_cols].nunique(axis=1).eq(1)
    blend["protected_anchor_prediction"] = blend[anchor_col]
    blend.loc[first_four_unanimous, "protected_anchor_prediction"] = blend.loc[
        first_four_unanimous, companion_cols[-1]
    ]

    blend["prediction"] = blend["protected_anchor_prediction"]
    patch_mask = blend[ID].isin(PUBLIC_BLEND_MICRO_PATCH)
    blend.loc[patch_mask, "prediction"] = blend.loc[patch_mask, ID].map(PUBLIC_BLEND_MICRO_PATCH)

    blend["all_five_agree"] = blend[ENSEMBLE_FILES].nunique(axis=1).eq(1)
    blend["changed_from_anchor"] = blend["prediction"].ne(blend[anchor_col])
    blend["changed_from_protected_anchor"] = blend["prediction"].ne(blend["protected_anchor_prediction"])

    submission = sample.copy()
    submission[TARGET] = blend["prediction"].to_numpy()
    return submission, blend


def compare_submission(rebuilt: pd.DataFrame, compare_path: Path) -> dict:
    if not compare_path.exists():
        return {"compare_path": str(compare_path), "exists": False}

    existing = pd.read_csv(compare_path)
    if not rebuilt[ID].equals(existing[ID]):
        return {"compare_path": str(compare_path), "exists": True, "id_order_equal": False}

    diff = rebuilt[TARGET].ne(existing[TARGET])
    diff_rows = rebuilt.loc[diff, [ID, TARGET]].copy()
    diff_rows["existing"] = existing.loc[diff, TARGET].to_numpy()
    table = pd.crosstab(diff_rows["existing"], diff_rows[TARGET]).to_dict()
    return {
        "compare_path": str(compare_path),
        "exists": True,
        "id_order_equal": True,
        "exact_match": bool(not diff.any()),
        "diff_rows": int(diff.sum()),
        "diff_table_existing_to_rebuilt": table,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    submission, blend = build_blend(args.prediction_dir, sample)

    output_path = args.output_dir / "submission_rebuilt_097137.csv"
    submission.to_csv(output_path, index=False)

    diagnostics = {
        "recipe": "protected-anchor blend from 0.97137 reference md",
        "prediction_dir": str(args.prediction_dir),
        "ensemble_files": ENSEMBLE_FILES,
        "public_blend_micro_patch": PUBLIC_BLEND_MICRO_PATCH,
        "rows": int(len(blend)),
        "all_five_agree": int(blend["all_five_agree"].sum()),
        "companions_unanimously_override_anchor": int(
            blend["protected_anchor_prediction"].ne(blend[ENSEMBLE_FILES[-1]]).sum()
        ),
        "independent_micro_patch_rows": int(blend["changed_from_protected_anchor"].sum()),
        "final_differs_from_anchor": int(blend["changed_from_anchor"].sum()),
        "submission_class_share": submission[TARGET].value_counts(normalize=True).sort_index().to_dict(),
        "comparison": compare_submission(submission, args.compare),
    }
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    blend.to_csv(args.output_dir / "blend_row_diagnostics.csv", index=False)

    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
