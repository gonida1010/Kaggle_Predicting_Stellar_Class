from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
BANK_ANALYSIS = ARTIFACTS / "submission_bank_analysis"
OUT_DIR = ARTIFACTS / "bank_informed_probes"
DEFAULT_ANCHOR = ARTIFACTS / "star_to_galaxy_research" / "group_research_top_10.csv"
CLASSES = ["GALAXY", "QSO", "STAR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score anchor-vs-submission-bank disagreements and build public probes."
    )
    parser.add_argument(
        "--bank-votes",
        type=Path,
        default=BANK_ANALYSIS / "submission_bank_row_votes.csv",
        help="Output from scripts/analyze_submission_bank.py.",
    )
    parser.add_argument(
        "--anchor-file",
        type=Path,
        default=DEFAULT_ANCHOR,
        help="Current public anchor submission.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Where to write ranked candidates and probe submissions.",
    )
    return parser.parse_args()


def read_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "class"]:
        raise ValueError(f"{path} columns must be ['id', 'class']")
    if not df["id"].equals(sample["id"]):
        raise ValueError(f"{path} id order differs from sample_submission.csv")
    invalid = sorted(set(df["class"].dropna()) - set(CLASSES))
    if invalid:
        raise ValueError(f"{path} has invalid labels: {invalid}")
    return df


def load_proba(report_path: Path, proba_path: Path) -> tuple[list[str], np.ndarray]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report["classes"], np.load(proba_path)


def align_proba(classes: list[str], proba: np.ndarray) -> np.ndarray:
    class_to_col = {cls: idx for idx, cls in enumerate(classes)}
    return np.column_stack([proba[:, class_to_col[cls]] for cls in CLASSES])


def load_model_evidence() -> tuple[np.ndarray, np.ndarray]:
    pure_classes, pure_proba = load_proba(
        ARTIFACTS / "pure_model_ensemble" / "pure_model_ensemble_report.json",
        ARTIFACTS / "pure_model_ensemble" / "pure_model_ensemble_test_proba.npy",
    )
    pure = align_proba(pure_classes, pure_proba)

    base_parts = []
    for model in ["lgbm", "catboost"]:
        classes, proba = load_proba(
            ARTIFACTS / f"{model}_baseline_report.json",
            ARTIFACTS / f"{model}_test_proba.npy",
        )
        base_parts.append(align_proba(classes, proba))
    base_mean = np.mean(base_parts, axis=0)
    return pure, base_mean


def write_submission(anchor: pd.DataFrame, changes: pd.DataFrame, output_dir: Path, name: str) -> Path:
    submission = anchor.copy()
    id_to_label = dict(zip(changes["id"], changes["bank_consensus"]))
    mask = submission["id"].isin(id_to_label)
    submission.loc[mask, "class"] = submission.loc[mask, "id"].map(id_to_label)
    path = output_dir / name
    submission.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.bank_votes.exists():
        raise FileNotFoundError(
            f"{args.bank_votes} does not exist. First run:\n"
            "python scripts/analyze_submission_bank.py --prediction-dir external_preds "
            "--anchor-file artifacts/star_to_galaxy_research/group_research_top_10.csv"
        )

    sample = pd.read_csv(DATA / "sample_submission.csv")
    anchor = read_submission(args.anchor_file, sample)
    votes = pd.read_csv(args.bank_votes)
    if not votes["id"].equals(sample["id"]):
        raise ValueError(f"{args.bank_votes} id order differs from sample_submission.csv")

    pure_proba, base_model_proba = load_model_evidence()
    candidates = votes[~votes["anchor_matches_consensus"]].copy()
    candidates["anchor"] = anchor["class"].to_numpy()[~votes["anchor_matches_consensus"].to_numpy()]
    candidates["transition"] = candidates["anchor"] + "->" + candidates["bank_consensus"]

    for idx, cls in enumerate(CLASSES):
        candidates[f"pure_p_{cls}"] = pure_proba[candidates.index, idx]
        candidates[f"model_p_{cls}"] = base_model_proba[candidates.index, idx]

    label_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
    row_idx = candidates.index.to_numpy()
    anchor_idx = candidates["anchor"].map(label_to_idx).to_numpy()
    consensus_idx = candidates["bank_consensus"].map(label_to_idx).to_numpy()
    candidates["pure_p_anchor"] = pure_proba[row_idx, anchor_idx]
    candidates["pure_p_consensus"] = pure_proba[row_idx, consensus_idx]
    candidates["model_p_anchor"] = base_model_proba[row_idx, anchor_idx]
    candidates["model_p_consensus"] = base_model_proba[row_idx, consensus_idx]
    candidates["pure_margin"] = candidates["pure_p_consensus"] - candidates["pure_p_anchor"]
    candidates["model_margin"] = candidates["model_p_consensus"] - candidates["model_p_anchor"]

    vote_cols = [f"vote_{cls}" for cls in CLASSES]
    candidates["bank_vote_anchor"] = candidates.apply(lambda row: row[f"vote_{row['anchor']}"], axis=1)
    candidates["bank_vote_consensus"] = candidates.apply(lambda row: row[f"vote_{row['bank_consensus']}"], axis=1)
    candidates["bank_vote_gap"] = candidates["bank_vote_consensus"] - candidates["bank_vote_anchor"]
    candidates["bank_vote_gap_share"] = candidates["bank_vote_gap"] / candidates[vote_cols].sum(axis=1)
    candidates["bank_score"] = (
        2.20 * candidates["bank_consensus_share"]
        + 0.45 * candidates["bank_vote_gap_share"]
        + 0.65 * candidates["model_margin"].clip(-1, 1)
        + 0.30 * candidates["pure_margin"].clip(-1, 1)
        - 0.08 * (candidates["bank_nunique"] - 1)
    )
    candidates = candidates.sort_values(
        ["bank_score", "bank_consensus_share", "model_margin", "pure_margin"],
        ascending=False,
    ).reset_index(drop=True)
    candidates["bank_rank"] = np.arange(1, len(candidates) + 1)
    candidates.to_csv(args.output_dir / "bank_disagreement_scored_candidates.csv", index=False)

    strict = candidates[
        candidates["bank_consensus_share"].ge(0.85)
        & candidates["bank_vote_gap_share"].gt(0)
        & candidates["model_margin"].gt(-0.05)
    ].copy()
    strict.to_csv(args.output_dir / "bank_strict_candidates.csv", index=False)

    generated = []
    for n in [1, 3, 5, 10]:
        if len(candidates) >= n:
            generated.append(write_submission(anchor, candidates.head(n), args.output_dir, f"bank_global_top_{n:02d}.csv"))
        if len(strict) >= n:
            generated.append(write_submission(anchor, strict.head(n), args.output_dir, f"bank_strict_top_{n:02d}.csv"))

    for transition, group in candidates.groupby("transition", sort=False):
        safe_transition = transition.replace("->", "_to_")
        group = group.sort_values(
            ["bank_score", "bank_consensus_share", "model_margin"],
            ascending=False,
        )
        group.to_csv(args.output_dir / f"bank_{safe_transition}_candidates.csv", index=False)
        for n in [1, 3, 5]:
            if len(group) >= n:
                generated.append(
                    write_submission(anchor, group.head(n), args.output_dir, f"bank_{safe_transition}_top_{n:02d}.csv")
                )

    transition_summary = (
        candidates.groupby("transition", observed=True)
        .agg(
            candidates=("id", "size"),
            max_bank_score=("bank_score", "max"),
            avg_consensus_share=("bank_consensus_share", "mean"),
            avg_model_margin=("model_margin", "mean"),
        )
        .reset_index()
        .sort_values(["max_bank_score", "candidates"], ascending=False)
    )
    transition_summary.to_csv(args.output_dir / "bank_transition_summary.csv", index=False)

    report = {
        "anchor_file": str(args.anchor_file.relative_to(ROOT) if args.anchor_file.is_relative_to(ROOT) else args.anchor_file),
        "bank_votes": str(args.bank_votes.relative_to(ROOT) if args.bank_votes.is_relative_to(ROOT) else args.bank_votes),
        "candidate_count": int(len(candidates)),
        "strict_candidate_count": int(len(strict)),
        "transition_counts": candidates["transition"].value_counts().to_dict(),
        "recommended_first_wave": [
            "bank_strict_top_03.csv",
            "bank_global_top_03.csv",
            "best transition-specific top_01 after reading bank_transition_summary.csv",
        ],
        "generated_files": [path.name for path in generated],
    }
    (args.output_dir / "bank_informed_probe_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "ACTIVE_BANK_PROBES.txt").write_text(
        "\n".join(["# Bank-informed probe files", *[path.name for path in generated]]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
