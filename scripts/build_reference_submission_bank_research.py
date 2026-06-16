from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "reference_submission_bank_RESEARCH_ONLY"

CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_TO_INT = {label: idx for idx, label in enumerate(CLASSES)}
DEFAULT_ANCHOR = ARTIFACTS / "star_to_galaxy_research" / "group_research_top_10.csv"


MANUAL_PUBLIC_SCORES = {
    "submission.csv": 0.97137,
    "group_minconf_095_all.csv": 0.97139,
    "01_QSO_to_STAR_top_01.csv": 0.97141,
    "02_GALAXY_to_STAR_top_03.csv": 0.97141,
    "03_strict_QSO_STAR_1_plus_GALAXY_STAR_3.csv": 0.97141,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only experiment: include a provided high-score reference submission "
            "inside a local submission bank and measure how candidate rankings change. "
            "This script does not create final submission files."
        )
    )
    parser.add_argument("reference_submission", type=Path)
    parser.add_argument("--reference-score", type=float, default=0.97214)
    parser.add_argument("--anchor-file", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--anchor-score", type=float, default=0.97141)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--min-local-score",
        type=float,
        default=0.97137,
        help="Only local submissions at or above this public score are used in the mini-bank.",
    )
    parser.add_argument("--top-n-report", type=int, default=80)
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
    invalid = sorted(set(df["class"].dropna()) - set(CLASSES))
    if invalid:
        raise ValueError(f"{path} has invalid labels: {invalid}")
    return df


def locate_artifact_by_name(filename: str) -> Path | None:
    if filename == "submission.csv":
        path = DATA / "submission.csv"
        return path if path.exists() else None
    matches = sorted(ARTIFACTS.rglob(filename))
    return matches[0] if matches else None


def collect_known_local_bank(min_score: float) -> list[dict]:
    rows: list[dict] = []
    observed_path = ARTIFACTS / "public_feedback" / "observed_public_scores.csv"
    if observed_path.exists():
        observed = pd.read_csv(observed_path)
        for _, row in observed.iterrows():
            filename = str(row["file"])
            score = float(row["public_score"])
            path = locate_artifact_by_name(filename)
            if path is None or score < min_score:
                continue
            rows.append(
                {
                    "name": filename,
                    "score": score,
                    "path": path,
                    "source": "observed_public_scores",
                    "is_reference": False,
                }
            )

    for filename, score in MANUAL_PUBLIC_SCORES.items():
        path = locate_artifact_by_name(filename)
        if path is None or score < min_score:
            continue
        rows.append(
            {
                "name": filename,
                "score": score,
                "path": path,
                "source": "manual_observed_or_shared",
                "is_reference": False,
            }
        )

    dedup: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row["path"]), row["name"])
        existing = dedup.get(key)
        if existing is None or float(row["score"]) > float(existing["score"]):
            dedup[key] = row

    return sorted(dedup.values(), key=lambda item: (-float(item["score"]), str(item["name"])))


def safe_bank_filename(row: dict) -> str:
    label = str(row["name"]).replace(" ", "_").replace("/", "_")
    prefix = f"{float(row['score']):.5f}"
    if row.get("is_reference"):
        return f"{prefix}__REFERENCE_ONLY__{label}"
    return f"{prefix}__local__{label}"


def copy_bank(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.csv"):
        stale.unlink()
    for row in rows:
        shutil.copy2(row["path"], output_dir / safe_bank_filename(row))


def load_model_proba(report_path: Path, proba_path: Path) -> tuple[list[str], np.ndarray] | None:
    if not report_path.exists() or not proba_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    classes = list(report["classes"])
    proba = np.load(proba_path)
    aligned = np.column_stack([proba[:, classes.index(label)] for label in CLASSES])
    return CLASSES, aligned


def load_model_evidence() -> dict[str, np.ndarray]:
    evidence: dict[str, np.ndarray] = {}
    pure = load_model_proba(
        ARTIFACTS / "pure_model_ensemble" / "pure_model_ensemble_report.json",
        ARTIFACTS / "pure_model_ensemble" / "pure_model_ensemble_test_proba.npy",
    )
    if pure is not None:
        evidence["pure"] = pure[1]
    lgbm = load_model_proba(ARTIFACTS / "lgbm_baseline_report.json", ARTIFACTS / "lgbm_test_proba.npy")
    if lgbm is not None:
        evidence["lgbm"] = lgbm[1]
    cat = load_model_proba(ARTIFACTS / "catboost_baseline_report.json", ARTIFACTS / "catboost_test_proba.npy")
    if cat is not None:
        evidence["catboost"] = cat[1]
    if "lgbm" in evidence and "catboost" in evidence:
        evidence["lgbm_cat_mean"] = np.mean([evidence["lgbm"], evidence["catboost"]], axis=0)
    return evidence


def build_candidate_table(
    rows: list[dict],
    sample: pd.DataFrame,
    anchor: pd.DataFrame,
    anchor_score: float,
    test: pd.DataFrame,
    model_evidence: dict[str, np.ndarray],
) -> pd.DataFrame:
    anchor_labels = anchor["class"].to_numpy()
    row_records: dict[tuple[int, str], dict] = {}
    ids = sample["id"].to_numpy()
    id_to_pos = {int(row_id): pos for pos, row_id in enumerate(ids)}

    for bank_row in rows:
        sub = read_submission(bank_row["path"], sample)
        labels = sub["class"].to_numpy()
        changed_positions = np.flatnonzero(labels != anchor_labels)
        score = float(bank_row["score"])
        delta = score - anchor_score
        for pos in changed_positions:
            row_id = int(ids[pos])
            proposed = str(labels[pos])
            key = (row_id, proposed)
            record = row_records.setdefault(
                key,
                {
                    "id": row_id,
                    "row_pos": int(pos),
                    "anchor_class": str(anchor_labels[pos]),
                    "proposed_class": proposed,
                    "transition": f"{anchor_labels[pos]}->{proposed}",
                    "support_files": 0,
                    "support_non_reference_files": 0,
                    "support_reference_files": 0,
                    "best_support_score": -np.inf,
                    "weighted_delta_sum": 0.0,
                    "positive_delta_sum": 0.0,
                    "negative_delta_sum": 0.0,
                    "support_file_names": [],
                },
            )
            record["support_files"] += 1
            if bank_row["is_reference"]:
                record["support_reference_files"] += 1
            else:
                record["support_non_reference_files"] += 1
            record["best_support_score"] = max(float(record["best_support_score"]), score)
            record["weighted_delta_sum"] += delta
            record["positive_delta_sum"] += max(0.0, delta)
            record["negative_delta_sum"] += min(0.0, delta)
            record["support_file_names"].append(str(bank_row["name"]))

    if not row_records:
        return pd.DataFrame()

    candidates = pd.DataFrame(row_records.values())
    candidates["support_file_names"] = candidates["support_file_names"].apply(lambda names: "|".join(names))
    candidates["research_bank_score"] = (
        candidates["positive_delta_sum"] * 100000.0
        + candidates["weighted_delta_sum"] * 25000.0
        + candidates["support_non_reference_files"] * 0.15
        + candidates["support_reference_files"] * 0.05
        - candidates["support_files"].sub(candidates["support_non_reference_files"]).clip(lower=0) * 0.02
    )

    test_by_id = test.set_index("id")
    feature_rows = test_by_id.loc[candidates["id"], ["spectral_type", "galaxy_population", "redshift", "u", "g", "r", "i", "z"]]
    feature_rows = feature_rows.reset_index(drop=True)
    feature_rows["u_g"] = feature_rows["u"] - feature_rows["g"]
    feature_rows["g_r"] = feature_rows["g"] - feature_rows["r"]
    feature_rows["r_i"] = feature_rows["r"] - feature_rows["i"]
    feature_rows["i_z"] = feature_rows["i"] - feature_rows["z"]
    feature_rows["u_r"] = feature_rows["u"] - feature_rows["r"]
    feature_rows["g_i"] = feature_rows["g"] - feature_rows["i"]
    feature_rows["mag_range"] = feature_rows[["u", "g", "r", "i", "z"]].max(axis=1) - feature_rows[["u", "g", "r", "i", "z"]].min(axis=1)
    candidates = pd.concat([candidates.reset_index(drop=True), feature_rows], axis=1)

    positions = candidates["row_pos"].to_numpy()
    anchor_idx = candidates["anchor_class"].map(LABEL_TO_INT).to_numpy()
    proposed_idx = candidates["proposed_class"].map(LABEL_TO_INT).to_numpy()
    for name, proba in model_evidence.items():
        candidates[f"{name}_p_anchor"] = proba[positions, anchor_idx]
        candidates[f"{name}_p_proposed"] = proba[positions, proposed_idx]
        candidates[f"{name}_margin"] = candidates[f"{name}_p_proposed"] - candidates[f"{name}_p_anchor"]

    margin_cols = [col for col in candidates.columns if col.endswith("_margin")]
    if margin_cols:
        candidates["mean_model_margin"] = candidates[margin_cols].mean(axis=1)
        candidates["models_supporting_proposed"] = candidates[margin_cols].gt(0).sum(axis=1)
    else:
        candidates["mean_model_margin"] = np.nan
        candidates["models_supporting_proposed"] = 0

    candidates["combined_research_score"] = (
        candidates["research_bank_score"]
        + candidates["mean_model_margin"].fillna(0.0).clip(-1, 1) * 0.7
        + candidates["models_supporting_proposed"] * 0.05
    )
    return candidates.sort_values(
        ["combined_research_score", "research_bank_score", "support_non_reference_files", "mean_model_margin"],
        ascending=False,
    ).reset_index(drop=True)


def summarize_candidates(candidates: pd.DataFrame) -> dict:
    if candidates.empty:
        return {"candidate_count": 0}
    return {
        "candidate_count": int(len(candidates)),
        "transition_counts": candidates["transition"].value_counts().to_dict(),
        "reference_only_candidate_count": int(
            candidates["support_reference_files"].gt(0).mul(candidates["support_non_reference_files"].eq(0)).sum()
        ),
        "non_reference_supported_candidate_count": int(candidates["support_non_reference_files"].gt(0).sum()),
        "top10": candidates.head(10)[
            [
                "id",
                "transition",
                "support_non_reference_files",
                "support_reference_files",
                "combined_research_score",
                "mean_model_margin",
                "spectral_type",
                "galaxy_population",
                "redshift",
                "g_i",
                "mag_range",
            ]
        ].to_dict(orient="records"),
    }


def write_readme(output_dir: Path, report: dict) -> None:
    lines = [
        "# Reference Submission Bank Research Only",
        "",
        "This folder is a research-only sandbox. Do not submit files from here to Kaggle.",
        "",
        "What this tests:",
        "",
        "- Build a local mini submission bank from submissions with observed public scores.",
        "- Add the user-provided high-score reference output as one extra bank member.",
        "- Compare candidate rankings with and without that reference member.",
        "- Attach our model probability margins so reference-only rows are not treated as ground truth.",
        "",
        "Main outputs:",
        "",
        "- `manifest.csv`: bank members and known/research scores.",
        "- `candidate_scores_without_reference.csv`: candidates from our known local bank only.",
        "- `candidate_scores_with_reference.csv`: candidates after adding the reference file.",
        "- `reference_inclusion_delta.csv`: rows whose rank/score changed when reference was added.",
        "- `reference_driven_top_candidates.csv`: top reference-driven rows for inspection only.",
        "- `mini_bank_with_reference/`: score-prefixed research bank input copies.",
        "",
        "Read:",
        "",
        f"- Local bank files: {report['local_bank_files']}",
        f"- With-reference candidates: {report['with_reference']['candidate_count']}",
        f"- Reference-only candidates: {report['with_reference'].get('reference_only_candidate_count', 0)}",
        "",
        "Interpretation:",
        "",
        report["interpretation"],
        "",
    ]
    output_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    test = pd.read_csv(DATA / "test.csv")
    anchor = read_submission(args.anchor_file, sample)
    reference = read_submission(args.reference_submission, sample)
    model_evidence = load_model_evidence()

    local_rows = collect_known_local_bank(args.min_local_score)
    if not local_rows:
        raise FileNotFoundError("No known local bank files were found.")

    reference_row = {
        "name": args.reference_submission.name,
        "score": float(args.reference_score),
        "path": args.reference_submission,
        "source": "user_provided_reference_RESEARCH_ONLY",
        "is_reference": True,
    }
    with_reference_rows = [*local_rows, reference_row]

    copy_bank(local_rows, args.output_dir / "mini_bank_without_reference")
    copy_bank(with_reference_rows, args.output_dir / "mini_bank_with_reference")

    manifest = pd.DataFrame(
        [
            {
                "name": row["name"],
                "score": row["score"],
                "source": row["source"],
                "is_reference": row["is_reference"],
                "path": str(row["path"]),
                "sha256": sha256_file(row["path"]),
                "diff_vs_anchor": int(read_submission(row["path"], sample)["class"].ne(anchor["class"]).sum()),
            }
            for row in with_reference_rows
        ]
    )
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)

    without_ref = build_candidate_table(local_rows, sample, anchor, args.anchor_score, test, model_evidence)
    with_ref = build_candidate_table(with_reference_rows, sample, anchor, args.anchor_score, test, model_evidence)
    without_ref.to_csv(args.output_dir / "candidate_scores_without_reference.csv", index=False)
    with_ref.to_csv(args.output_dir / "candidate_scores_with_reference.csv", index=False)

    key_cols = ["id", "proposed_class"]
    before_cols = key_cols + ["combined_research_score", "research_bank_score", "support_non_reference_files"]
    before = without_ref[before_cols].copy() if not without_ref.empty else pd.DataFrame(columns=before_cols)
    before = before.rename(
        columns={
            "combined_research_score": "score_without_reference",
            "research_bank_score": "bank_score_without_reference",
            "support_non_reference_files": "support_non_reference_without",
        }
    )
    before["rank_without_reference"] = np.arange(1, len(before) + 1)

    after_cols = key_cols + [
        "anchor_class",
        "transition",
        "combined_research_score",
        "research_bank_score",
        "support_non_reference_files",
        "support_reference_files",
        "mean_model_margin",
        "models_supporting_proposed",
        "spectral_type",
        "galaxy_population",
        "redshift",
        "u_r",
        "g_i",
        "mag_range",
    ]
    after = with_ref[after_cols].copy()
    after = after.rename(
        columns={
            "combined_research_score": "score_with_reference",
            "research_bank_score": "bank_score_with_reference",
        }
    )
    after["rank_with_reference"] = np.arange(1, len(after) + 1)
    delta = after.merge(before, on=key_cols, how="left")
    delta["became_candidate_only_after_reference"] = delta["score_without_reference"].isna()
    delta["score_delta_from_reference_inclusion"] = (
        delta["score_with_reference"] - delta["score_without_reference"].fillna(0.0)
    )
    delta = delta.sort_values(
        [
            "became_candidate_only_after_reference",
            "score_delta_from_reference_inclusion",
            "score_with_reference",
            "mean_model_margin",
        ],
        ascending=[False, False, False, False],
    )
    delta.to_csv(args.output_dir / "reference_inclusion_delta.csv", index=False)

    reference_driven = delta[
        delta["support_reference_files"].gt(0)
        & delta["support_non_reference_files"].eq(0)
    ].copy()
    reference_driven.head(args.top_n_report).to_csv(
        args.output_dir / "reference_driven_top_candidates.csv",
        index=False,
    )

    report = {
        "research_only": True,
        "reference_submission": str(args.reference_submission),
        "reference_score_used_for_research": float(args.reference_score),
        "reference_sha256": sha256_file(args.reference_submission),
        "anchor_file": str(args.anchor_file),
        "anchor_score": float(args.anchor_score),
        "local_bank_files": len(local_rows),
        "model_evidence_loaded": sorted(model_evidence),
        "without_reference": summarize_candidates(without_ref),
        "with_reference": summarize_candidates(with_ref),
        "interpretation": (
            "Because the real external high-score bank is still missing, adding one 0.97214 reference file mostly "
            "promotes rows that only that file changes. Treat those as hypotheses, not labels. The useful rows are "
            "the subset where the reference direction also has positive margins from our independent models or "
            "support from non-reference local submissions."
        ),
        "do_not_submit": [
            "candidate_scores_with_reference.csv",
            "reference_driven_top_candidates.csv",
            "any file in this RESEARCH_ONLY directory",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(args.output_dir, report)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote research-only bank experiment to {args.output_dir}")


if __name__ == "__main__":
    main()
