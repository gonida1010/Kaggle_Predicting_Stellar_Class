from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
EXTERNAL = ROOT / "external_preds"
OUT_DIR = ARTIFACTS / "final_submissions"
CLASSES = {"GALAXY", "QSO", "STAR"}


GENERALIZATION_CANDIDATES = [
    ARTIFACTS / "pure_model_ensemble" / "pure_model_ensemble_submission.csv",
    ARTIFACTS / "catboost_baseline_submission.csv",
    ARTIFACTS / "lgbm_baseline_submission.csv",
]


PUBLIC_GENERALIZATION_CANDIDATES = [
    ARTIFACTS / "ambiguous_vote_patch" / "ambiguous_count2_replace_sub4.csv",
    ARTIFACTS / "bank_ridge_flip_v5" / "v5_voted_ensemble.csv",
    EXTERNAL / "0.97209.csv",
    ARTIFACTS / "bank_informed_probes" / "bank_strict_top_03.csv",
    ARTIFACTS / "star_to_galaxy_research" / "group_research_top_10.csv",
    ARTIFACTS / "probe_queue" / "group_minconf_095_star_to_galaxy_except_rank_01.csv",
    DATA / "submission.csv",
]


def validate_submission(path: Path, sample: pd.DataFrame) -> dict:
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "class"]:
        raise ValueError(f"{path} columns must be ['id', 'class']")
    if not df["id"].equals(sample["id"]):
        raise ValueError(f"{path} id order differs from sample_submission.csv")
    invalid = sorted(set(df["class"].dropna()) - CLASSES)
    if invalid:
        raise ValueError(f"{path} has invalid labels: {invalid}")
    return {
        "path": str(path),
        "rows": int(len(df)),
        "counts": df["class"].value_counts().sort_index().to_dict(),
    }


def choose_first_existing(candidates: list[Path], sample: pd.DataFrame) -> tuple[Path, dict]:
    missing = []
    for path in candidates:
        if path.exists():
            return path, validate_submission(path, sample)
        missing.append(str(path))
    raise FileNotFoundError("No candidate submission exists. Checked:\n" + "\n".join(missing))


def copy_candidate(src: Path, dst: Path, sample: pd.DataFrame) -> dict:
    info = validate_submission(src, sample)
    shutil.copyfile(src, dst)
    copied_info = validate_submission(dst, sample)
    return {**info, "output": str(dst), "output_counts": copied_info["counts"]}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(DATA / "sample_submission.csv")

    generalization_src, generalization_info = choose_first_existing(GENERALIZATION_CANDIDATES, sample)
    public_src, public_info = choose_first_existing(PUBLIC_GENERALIZATION_CANDIDATES, sample)

    generalization = copy_candidate(
        generalization_src,
        OUT_DIR / "final_generalization_model.csv",
        sample,
    )
    public_generalization = copy_candidate(
        public_src,
        OUT_DIR / "final_public_generalization.csv",
        sample,
    )
    report = {
        "generalization_track": {
            "selected": str(generalization_src),
            "why": "Best available anchor-free model submission. This is the private/generalization candidate.",
            **generalization,
            "selection_candidates": [str(path) for path in GENERALIZATION_CANDIDATES],
        },
        "public_generalization_track": {
            "selected": str(public_src),
            "why": (
                "Best available high-public candidate. Priority is ambiguous-vote patch, "
                "then bank ridge flip ensemble, then known 0.97141 anchor fallback."
            ),
            **public_generalization,
            "selection_candidates": [str(path) for path in PUBLIC_GENERALIZATION_CANDIDATES],
        },
    }
    (OUT_DIR / "final_submission_tracks_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Final Submission Tracks",
                "",
                "Two files are generated for Kaggle final selection:",
                "",
                "- `final_generalization_model.csv`: anchor-free model track.",
                "- `final_public_generalization.csv`: high-public track with generalization safeguards where available.",
                "",
                "Read `final_submission_tracks_report.json` before selecting final submissions.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
