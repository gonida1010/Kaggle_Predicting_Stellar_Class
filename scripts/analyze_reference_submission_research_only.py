from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "reference_only_submission_analysis"
CLASSES = {"GALAXY", "QSO", "STAR"}


COMPARE_FILES = {
    "anchor_097137": DATA / "submission.csv",
    "public_097141": ARTIFACTS / "star_to_galaxy_research" / "group_research_top_10.csv",
    "final_public_generalization_current": ARTIFACTS / "final_submissions" / "final_public_generalization.csv",
    "final_generalization_model": ARTIFACTS / "final_submissions" / "final_generalization_model.csv",
    "pure_model": ARTIFACTS / "pure_model_ensemble" / "pure_model_ensemble_submission.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only analysis of a public/reference submission. "
            "Writes aggregate diagnostics only and does not create submission candidates."
        )
    )
    parser.add_argument("reference_submission", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "class"]:
        raise ValueError(f"{path} columns must be ['id', 'class']")
    if not df["id"].equals(sample["id"]):
        raise ValueError(f"{path} id order differs from sample_submission.csv")
    invalid = sorted(set(df["class"].dropna()) - CLASSES)
    if invalid:
        raise ValueError(f"{path} has invalid labels: {invalid}")
    return df


def transition_table(reference: pd.DataFrame, other: pd.DataFrame, other_name: str) -> pd.DataFrame:
    mask = reference["class"].ne(other["class"])
    transitions = (
        (other.loc[mask, "class"] + "->" + reference.loc[mask, "class"])
        .value_counts()
        .rename_axis("transition")
        .reset_index(name="rows")
    )
    transitions.insert(0, "compare_file", other_name)
    transitions.insert(1, "diff_rows_total", int(mask.sum()))
    return transitions


def segment_summary(reference: pd.DataFrame, other: pd.DataFrame, test: pd.DataFrame, other_name: str) -> pd.DataFrame:
    mask = reference["class"].ne(other["class"])
    frame = test.loc[mask, ["spectral_type", "galaxy_population", "redshift", "u", "g", "r", "i", "z"]].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["transition"] = other.loc[mask, "class"].to_numpy() + "->" + reference.loc[mask, "class"].to_numpy()
    frame["g_i"] = frame["g"] - frame["i"]
    frame["mag_range"] = frame[["u", "g", "r", "i", "z"]].max(axis=1) - frame[["u", "g", "r", "i", "z"]].min(axis=1)
    grouped = (
        frame.groupby(["transition", "spectral_type", "galaxy_population"], observed=True)
        .agg(
            rows=("redshift", "size"),
            avg_redshift=("redshift", "mean"),
            med_redshift=("redshift", "median"),
            avg_g_i=("g_i", "mean"),
            avg_mag_range=("mag_range", "mean"),
        )
        .reset_index()
        .sort_values(["rows", "transition"], ascending=[False, True])
    )
    grouped.insert(0, "compare_file", other_name)
    return grouped


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    test = pd.read_csv(DATA / "test.csv")
    reference = read_submission(args.reference_submission, sample)

    transition_tables = []
    segment_tables = []
    compare_summary = []
    for name, path in COMPARE_FILES.items():
        if not path.exists():
            continue
        other = read_submission(path, sample)
        mask = reference["class"].ne(other["class"])
        compare_summary.append({"compare_file": name, "diff_rows": int(mask.sum())})
        transition_tables.append(transition_table(reference, other, name))
        segment_tables.append(segment_summary(reference, other, test, name))

    class_counts = reference["class"].value_counts().sort_index()
    class_share = reference["class"].value_counts(normalize=True).sort_index()
    pd.DataFrame(
        {
            "class": class_counts.index,
            "rows": class_counts.to_numpy(),
            "share": class_share.to_numpy(),
        }
    ).to_csv(args.output_dir / "reference_class_mix.csv", index=False)
    pd.DataFrame(compare_summary).to_csv(args.output_dir / "reference_compare_summary.csv", index=False)
    if transition_tables:
        pd.concat(transition_tables, ignore_index=True).to_csv(
            args.output_dir / "reference_transition_counts.csv",
            index=False,
        )
    if segment_tables:
        pd.concat(segment_tables, ignore_index=True).to_csv(
            args.output_dir / "reference_segment_summary.csv",
            index=False,
        )

    report = {
        "reference_submission": str(args.reference_submission),
        "sha256": sha256_file(args.reference_submission),
        "rows": int(len(reference)),
        "class_counts": class_counts.to_dict(),
        "compare_summary": compare_summary,
        "research_only": True,
        "note": "Reference output is for aggregate diagnostics only. It is not used as a final submission or training target.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote aggregate research-only analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
