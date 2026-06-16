from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXTERNAL = ROOT / "external_preds"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "ambiguous_vote_patch"
CLASSES = ["GALAXY", "QSO", "STAR"]


DEFAULT_VOTE_SUBMISSIONS = [
    EXTERNAL / "cat-3_submission.csv",
    EXTERNAL / "realmlp-5_submission.csv",
    EXTERNAL / "nn-2_submission.csv",
    EXTERNAL / "xgb-5_submission.csv",
    EXTERNAL / "submission_binary.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch only high-ambiguity rows where independent public notebooks split 2-2-1. "
            "This reproduces the useful idea from the deeper-look notebook without hardcoding Kaggle paths."
        )
    )
    parser.add_argument(
        "--base-submission",
        type=Path,
        default=EXTERNAL / "0.97209.csv",
        help="High-public base submission to patch, for example zoli800 0.97209.",
    )
    parser.add_argument(
        "--vote-submissions",
        type=Path,
        nargs="+",
        default=DEFAULT_VOTE_SUBMISSIONS,
        help="Independent model submissions used to define agreement count.",
    )
    parser.add_argument(
        "--replacement-index",
        type=int,
        default=4,
        help="1-based index of vote-submission used on count==2 rows. The notebook used sub4, XGB.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def read_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    sub = pd.read_csv(path)
    if list(sub.columns) != ["id", "class"]:
        sub = sub[["id", "class"]].copy()
    if not sub["id"].equals(sample["id"]):
        raise ValueError(f"ID order mismatch: {path}")
    invalid = sorted(set(sub["class"].dropna()) - set(CLASSES))
    if invalid:
        raise ValueError(f"{path} has invalid labels: {invalid}")
    return sub


def agreement_counts(label_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    modes = []
    counts = []
    for col in label_matrix.T:
        values, value_counts = np.unique(col, return_counts=True)
        max_count = value_counts.max()
        counts.append(int(max_count))
        tied = set(values[value_counts == max_count])
        mode = next(label for label in col if label in tied)
        modes.append(mode)
    return np.array(modes), np.array(counts)


def write_patch(
    base: pd.DataFrame,
    replacement: pd.DataFrame,
    mask: np.ndarray,
    output_dir: Path,
    name: str,
) -> Path:
    out = base.copy()
    out.loc[mask, "class"] = replacement.loc[mask, "class"].to_numpy()
    path = output_dir / name
    out.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(DATA / "sample_submission.csv")

    if len(args.vote_submissions) < 3:
        raise ValueError("At least 3 vote submissions are required.")
    if not 1 <= args.replacement_index <= len(args.vote_submissions):
        raise ValueError("--replacement-index is 1-based and must point into --vote-submissions.")

    base = read_submission(args.base_submission, sample)
    votes = [read_submission(path, sample) for path in args.vote_submissions]
    matrix = np.vstack([sub["class"].to_numpy() for sub in votes])
    mode, count = agreement_counts(matrix)

    vote_frame = sample[["id"]].copy()
    for idx, path in enumerate(args.vote_submissions, start=1):
        vote_frame[f"sub{idx}"] = matrix[idx - 1]
        vote_frame[f"sub{idx}_file"] = path.name
    vote_frame["mode"] = mode
    vote_frame["mode_count"] = count
    vote_frame["base"] = base["class"].to_numpy()
    vote_frame.to_csv(args.output_dir / "vote_agreement_by_row.csv", index=False)

    count_summary = (
        vote_frame["mode_count"]
        .value_counts()
        .sort_index()
        .rename_axis("mode_count")
        .reset_index(name="rows")
    )
    count_summary["share"] = count_summary["rows"] / len(vote_frame)
    count_summary.to_csv(args.output_dir / "vote_agreement_summary.csv", index=False)

    generated = []
    count2_mask = vote_frame["mode_count"].eq(2).to_numpy()
    count3_mask = vote_frame["mode_count"].eq(3).to_numpy()
    replacement = votes[args.replacement_index - 1]
    generated.append(
        write_patch(
            base,
            replacement,
            count2_mask,
            args.output_dir,
            f"ambiguous_count2_replace_sub{args.replacement_index}.csv",
        )
    )
    generated.append(
        write_patch(
            base,
            replacement,
            count2_mask | count3_mask,
            args.output_dir,
            f"ambiguous_count2_3_replace_sub{args.replacement_index}.csv",
        )
    )
    for idx, sub in enumerate(votes, start=1):
        generated.append(
            write_patch(
                base,
                sub,
                count2_mask,
                args.output_dir,
                f"ambiguous_count2_replace_sub{idx}.csv",
            )
        )

    ambiguous_rows = vote_frame[vote_frame["mode_count"].le(2)].copy()
    ambiguous_rows.to_csv(args.output_dir / "ambiguous_count2_rows.csv", index=False)

    report = {
        "base_submission": str(args.base_submission),
        "vote_submissions": [str(path) for path in args.vote_submissions],
        "replacement_index": args.replacement_index,
        "mode_count_summary": count_summary.to_dict(orient="records"),
        "count2_rows": int(count2_mask.sum()),
        "count3_rows": int(count3_mask.sum()),
        "generated_files": [path.name for path in generated],
        "recommended_first": f"ambiguous_count2_replace_sub{args.replacement_index}.csv",
        "note": (
            "The deeper-look notebook reported that replacing only count==2 rows with sub4 "
            "lifted a 0.97209 base to 0.97214. Treat count2_3 as exploratory and riskier."
        ),
    }
    (args.output_dir / "ambiguous_vote_patch_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "ACTIVE_AMBIGUOUS_PATCHES.txt").write_text(
        "\n".join(["# Ambiguous vote patch candidates", *[path.name for path in generated]]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
