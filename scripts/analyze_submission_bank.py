from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "submission_bank_analysis"
CLASSES = ["GALAXY", "QSO", "STAR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the public hard-label submission bank used by top public blends."
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=ROOT / "external_preds",
        help="Directory containing public submission-bank CSVs.",
    )
    parser.add_argument(
        "--anchor-file",
        type=Path,
        default=DATA / "submission.csv",
        help="Submission to compare against the public bank.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for bank diagnostics.",
    )
    return parser.parse_args()


def score_from_filename(path: Path) -> float | None:
    match = re.match(r"(0\.\d+)", path.name)
    return float(match.group(1)) if match else None


def read_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "class"]:
        raise ValueError(f"{path.name} columns must be ['id', 'class']")
    if not df["id"].equals(sample["id"]):
        raise ValueError(f"{path.name} id order differs from sample_submission.csv")
    invalid = sorted(set(df["class"].dropna()) - set(CLASSES))
    if invalid:
        raise ValueError(f"{path.name} has invalid labels: {invalid}")
    return df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    anchor = read_submission(args.anchor_file, sample)

    files = sorted(args.prediction_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No submission-bank CSVs found in {args.prediction_dir}. "
            "Download or place the public nina2025/ps-s6e6 files there."
        )

    manifest_rows = []
    bank = sample[["id"]].copy()
    for path in files:
        df = read_submission(path, sample)
        bank[path.name] = df["class"].to_numpy()
        manifest_rows.append(
            {
                "file": path.name,
                "public_score_from_filename": score_from_filename(path),
                "size_mb": path.stat().st_size / 1024**2,
                "diff_vs_anchor": int(df["class"].ne(anchor["class"]).sum()),
            }
        )

    pred_cols = [path.name for path in files]
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["public_score_from_filename", "file"],
        ascending=[False, True],
        na_position="last",
    )

    votes = sample[["id"]].copy()
    votes["anchor"] = anchor["class"].to_numpy()
    votes["bank_nunique"] = bank[pred_cols].nunique(axis=1)
    votes["bank_consensus"] = bank[pred_cols].mode(axis=1)[0]
    for cls in CLASSES:
        votes[f"vote_{cls}"] = bank[pred_cols].eq(cls).sum(axis=1)
    votes["bank_consensus_count"] = votes[[f"vote_{cls}" for cls in CLASSES]].max(axis=1)
    votes["bank_consensus_share"] = votes["bank_consensus_count"] / len(pred_cols)
    votes["anchor_matches_consensus"] = votes["anchor"].eq(votes["bank_consensus"])
    votes["change"] = votes["anchor"] + "->" + votes["bank_consensus"]

    disagreement = votes[~votes["anchor_matches_consensus"]].copy()
    disagreement = disagreement.sort_values(
        ["bank_consensus_share", "bank_nunique"],
        ascending=[False, True],
    )

    manifest.to_csv(args.output_dir / "submission_bank_manifest.csv", index=False)
    votes.to_csv(args.output_dir / "submission_bank_row_votes.csv", index=False)
    disagreement.to_csv(args.output_dir / "anchor_vs_bank_consensus_disagreements.csv", index=False)

    report = {
        "prediction_dir": str(args.prediction_dir),
        "anchor_file": str(args.anchor_file),
        "n_submission_files": len(files),
        "n_rows": int(len(votes)),
        "rows_all_bank_agree": int(votes["bank_nunique"].eq(1).sum()),
        "rows_anchor_differs_from_bank_consensus": int(len(disagreement)),
        "disagreement_change_counts": disagreement["change"].value_counts().to_dict(),
        "top_files": manifest.head(20).to_dict(orient="records"),
    }
    (args.output_dir / "submission_bank_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
