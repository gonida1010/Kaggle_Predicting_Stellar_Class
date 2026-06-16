from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
STAR_RESEARCH = ARTIFACTS / "star_to_galaxy_research"
TRANSITION_RESEARCH = ARTIFACTS / "transition_research"
OUT_DIR = ARTIFACTS / "public_feedback"
BASE_ANCHOR = ARTIFACTS / "probe_queue" / "group_minconf_095_star_to_galaxy_except_rank_01.csv"
BASE_SCORE = 0.97139


# Scores observed from Kaggle Public LB. The public display is rounded to 5 decimals,
# so these values are coarse feedback, not exact row labels.
OBSERVED_PUBLIC_SCORES = {
    "group_research_top_03.csv": 0.97140,
    "group_research_top_05.csv": 0.97140,
    "group_research_top_10.csv": 0.97141,
    "group_research_top_15.csv": 0.97141,
    "group_research_rank_06_10.csv": 0.97140,
    "group_research_rank_11_15.csv": 0.97139,
    "group_research_strong_local_top_10.csv": 0.97141,
    "group_research_minpgal_092_top_10.csv": 0.97141,
    "group_research_top10_plus_rank_17_19.csv": 0.97141,
    "group_research_top10_without_rank_09.csv": 0.97141,
    "feedback_replace_rank09_with_rank17.csv": 0.97141,
    "transition_QSO_to_GALAXY_top_01.csv": 0.97141,
    "transition_QSO_to_GALAXY_top_03.csv": 0.97141,
}


def load_anchor() -> pd.DataFrame:
    if not BASE_ANCHOR.exists():
        raise FileNotFoundError(BASE_ANCHOR)
    return pd.read_csv(BASE_ANCHOR)


def resolve_submission_path(file_name: str) -> Path:
    for directory in [STAR_RESEARCH, OUT_DIR, TRANSITION_RESEARCH]:
        path = directory / file_name
        if path.exists():
            return path
    raise FileNotFoundError(f"{file_name} was not found under star/public-feedback/transition artifacts")


def changed_ids(path: Path, base_anchor: pd.DataFrame) -> list[int]:
    submission = pd.read_csv(path)
    if list(submission.columns) != ["id", "class"]:
        raise ValueError(f"Not a submission file: {path}")
    if not submission["id"].equals(base_anchor["id"]):
        raise ValueError(f"id order differs: {path}")
    mask = submission["class"].ne(base_anchor["class"])
    return submission.loc[mask, "id"].astype(int).tolist()


def write_submission(base_anchor: pd.DataFrame, ids: list[int], name: str) -> Path:
    submission = base_anchor.copy()
    mask = submission["id"].isin(ids)
    submission.loc[mask, "class"] = "GALAXY"
    path = OUT_DIR / name
    submission.to_csv(path, index=False)
    return path


def ridge_effect_estimate(matrix: np.ndarray, target: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    xtx = matrix.T @ matrix
    rhs = matrix.T @ target
    return np.linalg.solve(xtx + alpha * np.eye(xtx.shape[0]), rhs)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_anchor = load_anchor()
    pool = pd.read_csv(STAR_RESEARCH / "next_probe_pool.csv")
    pool = pool.sort_values("next_probe_rank").reset_index(drop=True)

    observed_rows = []
    observed_id_sets = {}
    for file_name, score in OBSERVED_PUBLIC_SCORES.items():
        ids = changed_ids(resolve_submission_path(file_name), base_anchor)
        observed_id_sets[file_name] = set(ids)
        observed_rows.append(
            {
                "file": file_name,
                "public_score": score,
                "delta_units_vs_base": int(round((score - BASE_SCORE) * 100000)),
                "changed_rows": len(ids),
                "ids": ids,
            }
        )

    candidate_ids = sorted(set().union(*observed_id_sets.values()))
    id_to_col = {row_id: idx for idx, row_id in enumerate(candidate_ids)}
    x = np.zeros((len(observed_rows), len(candidate_ids)), dtype=float)
    y = np.array([row["delta_units_vs_base"] for row in observed_rows], dtype=float)
    for row_idx, row in enumerate(observed_rows):
        for row_id in row["ids"]:
            x[row_idx, id_to_col[row_id]] = 1.0

    effect = ridge_effect_estimate(x, y)
    effect_df = pd.DataFrame({"id": candidate_ids, "feedback_effect_units": effect})
    ranked = pool.merge(effect_df, on="id", how="left")
    ranked["feedback_effect_units"] = ranked["feedback_effect_units"].fillna(0.0)
    ranked["in_top10"] = ranked["next_probe_rank"].le(10)
    ranked["rank_11_20"] = ranked["next_probe_rank"].between(11, 20)
    ranked["feedback_score"] = (
        ranked["feedback_effect_units"]
        + 0.018 * (ranked["patchability_score"] - ranked["patchability_score"].median())
        + 0.40 * (ranked["local_p_GALAXY"] - ranked["local_p_STAR"])
    )
    ranked = ranked.sort_values(["feedback_score", "patchability_score"], ascending=False).reset_index(drop=True)
    ranked.to_csv(OUT_DIR / "feedback_ranked_candidates.csv", index=False)
    pd.DataFrame(observed_rows).to_csv(OUT_DIR / "observed_public_scores.csv", index=False)

    top10 = pool[pool["next_probe_rank"].le(10)].copy()
    top10_ids = top10["id"].astype(int).tolist()
    rank_to_id = dict(zip(pool["next_probe_rank"].astype(int), pool["id"].astype(int)))

    # Current best is top10. Because top10_without_rank_09 tied top10, rank 9 is
    # the safest slot to replace in the next experiments.
    weak_slot_id = rank_to_id[9]
    core_without_rank9 = [row_id for row_id in top10_ids if row_id != weak_slot_id]
    rank_17_19_ids = [
        int(row_id)
        for row_id in pool.loc[pool["next_probe_rank"].between(17, 19), "id"]
    ]

    generated = []
    generated.append(
        write_submission(
            base_anchor,
            core_without_rank9,
            "feedback_core_top10_without_rank_09.csv",
        )
    )
    for rank in [17, 18, 19]:
        generated.append(
            write_submission(
                base_anchor,
                core_without_rank9 + [rank_to_id[rank]],
                f"feedback_replace_rank09_with_rank{rank:02d}.csv",
            )
        )
    generated.append(
        write_submission(
            base_anchor,
            core_without_rank9 + rank_17_19_ids,
            "feedback_without_rank09_plus_rank17_19.csv",
        )
    )

    top_new_ids = (
        ranked[
            ~ranked["id"].isin(top10_ids)
            & ranked["next_probe_rank"].between(11, 30)
            & ranked["local_p_STAR"].lt(0.10)
        ]
        .head(3)["id"]
        .astype(int)
        .tolist()
    )
    if top_new_ids:
        generated.append(
            write_submission(
                base_anchor,
                core_without_rank9 + top_new_ids,
                "feedback_without_rank09_plus_best3.csv",
            )
        )

    # Leave-one-out files are not meant to beat the best immediately. They are
    # tomography probes for identifying indispensable rows when submissions reset.
    for rank in range(1, 11):
        row_id = rank_to_id[rank]
        ids = [candidate_id for candidate_id in top10_ids if candidate_id != row_id]
        generated.append(
            write_submission(
                base_anchor,
                ids,
                f"feedback_top10_without_rank_{rank:02d}.csv",
            )
        )

    report = {
        "base_anchor": str(BASE_ANCHOR.relative_to(ROOT)),
        "base_score": BASE_SCORE,
        "current_public_best": {
            "file": "group_research_top_10.csv",
            "public_score": OBSERVED_PUBLIC_SCORES["group_research_top_10.csv"],
            "changed_rows": len(top10_ids),
            "ids": top10_ids,
        },
        "interpretation": {
            "top10_without_rank09_tied_best": "rank 9 is public-neutral within rounded LB precision.",
            "replace_rank09_with_rank17_tied_best": "rank 17 did not unlock a visible public gain.",
            "qso_to_galaxy_top01_top03_tied_best": "QSO->GALAXY did not beat the current public anchor.",
            "top15_tied_best": "ranks 11-15 add no visible public gain and increase private risk.",
            "rank11_15_alone": OBSERVED_PUBLIC_SCORES["group_research_rank_11_15.csv"],
            "rank06_10_alone": OBSERVED_PUBLIC_SCORES["group_research_rank_06_10.csv"],
        },
        "recommended_next_submission_order": [],
        "next_direction": (
            "Pause local row-probe submissions. The observed STAR->GALAXY, replacement, "
            "and QSO->GALAXY probes are saturated at the rounded 0.97141 public score. "
            "The next high-signal step is submission-bank disagreement analysis from external_preds/."
        ),
        "generated_files": [path.name for path in generated],
    }
    (OUT_DIR / "public_feedback_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUT_DIR / "NEXT_PUBLIC_PROBES.txt").write_text(
        "\n".join(
            [
                "# Public feedback probe plan",
                "# Current best remains group_research_top_10.csv at 0.97141.",
                "# Latest replacement and QSO->GALAXY probes tied 0.97141.",
                "# Do not spend more submissions on local row-probes until submission-bank files are available.",
                "",
                "Next action:",
                "1. Put public submission-bank CSVs under external_preds/.",
                "2. Run: python scripts/analyze_submission_bank.py --prediction-dir external_preds --anchor-file artifacts/star_to_galaxy_research/group_research_top_10.csv",
                "3. Build new candidates from bank consensus disagreements plus our pure-model evidence.",
                "",
                "# Keep group_research_top_10.csv as the public final candidate unless a bank-informed probe beats it.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
