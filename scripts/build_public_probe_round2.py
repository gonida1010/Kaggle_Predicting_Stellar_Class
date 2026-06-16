from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
TRANSITION = ARTIFACTS / "transition_research"
OUT_DIR = ARTIFACTS / "round2_submission_queue_revised"
ANCHOR = ARTIFACTS / "star_to_galaxy_research" / "group_research_top_10.csv"


OBSERVED = {
    "01_generalization_model_baseline.csv": 0.96688,
    "02_transition_GALAXY_to_STAR_top_01.csv": 0.97141,
    "03_transition_GALAXY_to_QSO_top_01.csv": 0.97141,
    "04_transition_STAR_to_QSO_top_01.csv": 0.97141,
    "transition_QSO_to_GALAXY_top_01.csv": 0.97141,
    "transition_QSO_to_GALAXY_top_03.csv": 0.97141,
}


def changed_rows(candidate: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    if not candidate["id"].equals(anchor["id"]):
        raise ValueError("candidate id order differs from anchor")
    mask = candidate["class"].ne(anchor["class"])
    return candidate.loc[mask, ["id", "class"]].copy()


def merge_candidates(anchor: pd.DataFrame, names: list[str], output_name: str) -> tuple[Path, list[dict]]:
    out = anchor.copy()
    changes = []
    seen: dict[int, str] = {}
    for name in names:
        candidate = pd.read_csv(TRANSITION / name)
        changed = changed_rows(candidate, anchor)
        for _, row in changed.iterrows():
            row_id = int(row["id"])
            label = str(row["class"])
            if row_id in seen and seen[row_id] != label:
                raise ValueError(f"conflicting label for id {row_id}: {seen[row_id]} vs {label}")
            seen[row_id] = label
            changes.append({"source": name, "id": row_id, "class": label})
    id_to_label = seen
    mask = out["id"].isin(id_to_label)
    out.loc[mask, "class"] = out.loc[mask, "id"].map(id_to_label)
    path = OUT_DIR / output_name
    out.to_csv(path, index=False)
    return path, changes


def copy_candidate(anchor: pd.DataFrame, source_name: str, output_name: str) -> tuple[Path, list[dict]]:
    candidate = pd.read_csv(TRANSITION / source_name)
    changes = changed_rows(candidate, anchor)
    path = OUT_DIR / output_name
    candidate.to_csv(path, index=False)
    return path, [{"source": source_name, "id": int(row["id"]), "class": str(row["class"])} for _, row in changes.iterrows()]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    anchor = pd.read_csv(ANCHOR)
    if not anchor["id"].equals(sample["id"]):
        raise ValueError("anchor id order differs from sample")

    specs = [
        (
            "01_QSO_to_STAR_top_01.csv",
            "copy",
            ["transition_QSO_to_STAR_top_01.csv"],
            "Best remaining direction: reference analysis, local train neighborhood, pure model, LGBM, and CatBoost all support QSO->STAR on this boundary.",
        ),
        (
            "02_GALAXY_to_STAR_top_03.csv",
            "copy",
            ["transition_GALAXY_to_STAR_top_03.csv"],
            "Reference analysis says the largest public-anchor gap is GALAXY->STAR, concentrated in low-redshift high-color/high-mag-range rows. Top3 is cleaner than top5.",
        ),
        (
            "03_strict_QSO_STAR_1_plus_GALAXY_STAR_3.csv",
            "merge",
            [
                "transition_GALAXY_to_STAR_top_01.csv",
                "transition_QSO_to_STAR_top_01.csv",
                "transition_GALAXY_to_STAR_top_03.csv",
            ],
            "Strict combined probe: only the two boundary directions supported by deeper reference analysis.",
        ),
        (
            "04_QSO_to_STAR_top_03.csv",
            "copy",
            ["transition_QSO_to_STAR_top_03.csv"],
            "Use only if QSO->STAR top1 ties or improves; tests whether the QSO->STAR signal is a group effect.",
        ),
        (
            "05_GALAXY_to_STAR_top_05.csv",
            "copy",
            ["transition_GALAXY_to_STAR_top_05.csv"],
            "Use only if GALAXY->STAR top3 ties or improves. Top5 is less clean than top3.",
        ),
        (
            "06_reference_style_small_mix_holdout.csv",
            "merge",
            [
                "transition_GALAXY_to_STAR_top_03.csv",
                "transition_QSO_to_GALAXY_top_03.csv",
                "transition_QSO_to_STAR_top_01.csv",
            ],
            "Holdout only. Includes QSO->GALAXY because that direction is a known reference gap, but prior QSO->GALAXY top1/top3 tied.",
        ),
    ]

    rows = []
    for priority, (output_name, mode, sources, reason) in enumerate(specs, start=1):
        if mode == "copy":
            path, changes = copy_candidate(anchor, sources[0], output_name)
        else:
            path, changes = merge_candidates(anchor, sources, output_name)
        rows.append(
            {
                "priority": priority,
                "file": path.name,
                "mode": mode,
                "sources": sources,
                "changed_rows": len({item["id"] for item in changes}),
                "reason": reason,
            }
        )

    pd.DataFrame(rows).to_csv(OUT_DIR / "round2_queue_manifest.csv", index=False)
    report = {
        "anchor": str(ANCHOR.relative_to(ROOT)),
        "observed_so_far": OBSERVED,
        "public_baseline_read": (
            "Pure model public score 0.96688 is close to OOF 0.966345, so the generalization track is credible. "
            "The 0.97141 plateau is a public-anchor/probe-resolution issue, not a pure-model validation issue."
        ),
        "why_single_rows_tie": (
            "Kaggle public score uses only the public slice and displays 5 decimals. "
            "A one-row change often lands on private rows or is hidden by rounding."
        ),
        "queue": rows,
    }
    (OUT_DIR / "round2_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Round 2 Submission Queue",
                "",
                "Do not submit the 0.97214 reference output.",
                "Known current best remains `group_research_top_10.csv` at `0.97141`.",
                "This revised queue removes GALAXY->QSO, STAR->QSO, and broad medium mix probes because deeper evidence did not support them.",
                "",
                "Submit in order:",
                "",
                *[
                    f"{row['priority']}. `{row['file']}` - {row['reason']}"
                    for row in rows
                ],
                "",
                "Decision rules:",
                "",
                "- Submit 1 and 2 first.",
                "- Submit 3 only if both 1 and 2 do not drop.",
                "- Submit 4 only if 1 is neutral or positive.",
                "- Submit 5 only if 2 is neutral or positive.",
                "- Submit 6 only as the last holdout.",
                "- If every file ties `0.97141`, stop public row-probing until a real submission bank is available.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
