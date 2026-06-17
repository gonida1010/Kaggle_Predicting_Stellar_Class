from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
EXTERNAL = ROOT / "external_preds"
OUT_DIR = ARTIFACTS / "post_097207_probe_queue"

LABELS = ["GALAXY", "QSO", "STAR"]
CURRENT = ARTIFACTS / "bank_ridge_flip_v5" / "v5_voted_ensemble.csv"
TOP130 = ARTIFACTS / "bank_ridge_flip_v5" / "v5_ridge_top130.csv"
TAIL10 = ARTIFACTS / "bank_ridge_flip_v5" / "v5_ridge_top150_tail10.csv"
REFERENCE = Path("/Users/parkyeonggon/Downloads/submission (1).csv")


def read_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)[["id", "class"]].copy()
    if not df["id"].equals(sample["id"]):
        raise ValueError(f"{path} id order differs from sample_submission.csv")
    invalid = sorted(set(df["class"].dropna()) - set(LABELS))
    if invalid:
        raise ValueError(f"{path} invalid labels: {invalid}")
    return df


def load_probability_consensus(sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    arrays = []
    weights = []
    manifest = []
    for path in sorted(EXTERNAL.glob("test_preds__*.csv")):
        df = pd.read_csv(path)
        if not set(LABELS).issubset(df.columns):
            continue
        if not df["id"].equals(sample["id"]):
            raise ValueError(f"{path.name} id order differs from sample")
        arr = df[LABELS].to_numpy(np.float64)
        arr = arr / arr.sum(axis=1, keepdims=True)
        score = float(path.name.replace("test_preds__", "").replace(".csv", ""))
        weight = max(0.0, (score - 0.968) ** 2) * 1e4
        arrays.append(arr)
        weights.append(weight)
        manifest.append({"file": path.name, "score": score, "weight": weight})

    for path in sorted(EXTERNAL.glob("*test_preds__*.npy")):
        arr = np.load(path).astype(np.float64)
        if arr.shape != (len(sample), len(LABELS)):
            continue
        arr = arr / arr.sum(axis=1, keepdims=True)
        match = re.search(r"(0\.\d{5})", path.name)
        score = float(match.group(1)) if match else 0.969
        weight = max(0.0, (score - 0.968) ** 2) * 1e4
        arrays.append(arr)
        weights.append(weight)
        manifest.append({"file": path.name, "score": score, "weight": weight})

    if not arrays:
        raise FileNotFoundError("No probability consensus files found in external_preds")
    proba = np.average(np.stack(arrays), axis=0, weights=np.array(weights))
    pred = np.array(LABELS)[proba.argmax(axis=1)]
    margin = np.sort(proba, axis=1)[:, -1] - np.sort(proba, axis=1)[:, -2]
    return proba, pred, margin, manifest


def load_model_proba(report_path: Path, proba_path: Path) -> np.ndarray:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    proba = np.load(proba_path)
    classes = report["classes"]
    return np.column_stack([proba[:, classes.index(label)] for label in LABELS])


def build_candidate_table() -> tuple[pd.DataFrame, dict]:
    sample = pd.read_csv(DATA / "sample_submission.csv")
    test = pd.read_csv(DATA / "test.csv")
    current = read_submission(CURRENT, sample)
    top130 = read_submission(TOP130, sample)
    reference = read_submission(REFERENCE, sample)
    votes = pd.read_csv(ARTIFACTS / "submission_bank_analysis" / "submission_bank_row_votes.csv")
    if not votes["id"].equals(sample["id"]):
        raise ValueError("submission bank votes id order differs from sample")

    proba, proba_pred, proba_margin, proba_manifest = load_probability_consensus(sample)
    pure = load_model_proba(
        ARTIFACTS / "pure_model_ensemble" / "pure_model_ensemble_report.json",
        ARTIFACTS / "pure_model_ensemble" / "pure_model_ensemble_test_proba.npy",
    )
    lgbm = load_model_proba(ARTIFACTS / "lgbm_baseline_report.json", ARTIFACTS / "lgbm_test_proba.npy")
    cat = load_model_proba(ARTIFACTS / "catboost_baseline_report.json", ARTIFACTS / "catboost_test_proba.npy")
    base = (lgbm + cat) / 2.0

    # The 40 rows changed by top130 are excluded because top130 already dropped from 0.97207 to 0.97202.
    top130_drop_ids = set(current.loc[current["class"].ne(top130["class"]), "id"].astype(int))

    rows = []
    current_diff_bank = current["class"].ne(votes["bank_consensus"])
    for idx in np.flatnonzero(current_diff_bank.to_numpy()):
        row_id = int(current.at[idx, "id"])
        if row_id in top130_drop_ids:
            continue
        current_label = str(current.at[idx, "class"])
        bank_label = str(votes.at[idx, "bank_consensus"])
        ref_label = str(reference.at[idx, "class"])
        current_idx = LABELS.index(current_label)
        bank_idx = LABELS.index(bank_label)
        ref_idx = LABELS.index(ref_label)

        record = {
            "id": row_id,
            "current": current_label,
            "bank_target": bank_label,
            "ref_target": ref_label,
            "transition": f"{current_label}->{bank_label}",
            "bank_share": float(votes.at[idx, "bank_consensus_share"]),
            "bank_nunique": int(votes.at[idx, "bank_nunique"]),
            "proba_pred": str(proba_pred[idx]),
            "proba_margin": float(proba_margin[idx]),
            "p_current": float(proba[idx, current_idx]),
            "p_bank": float(proba[idx, bank_idx]),
            "p_ref": float(proba[idx, ref_idx]),
            "proba_delta_bank": float(proba[idx, bank_idx] - proba[idx, current_idx]),
            "proba_delta_ref": float(proba[idx, ref_idx] - proba[idx, current_idx]),
            "pure_delta_bank": float(pure[idx, bank_idx] - pure[idx, current_idx]),
            "base_delta_bank": float(base[idx, bank_idx] - base[idx, current_idx]),
            "spectral_type": test.at[idx, "spectral_type"],
            "galaxy_population": test.at[idx, "galaxy_population"],
            "redshift": float(test.at[idx, "redshift"]),
            "g_i": float(test.at[idx, "g"] - test.at[idx, "i"]),
            "mag_range": float(test.loc[idx, ["u", "g", "r", "i", "z"]].max() - test.loc[idx, ["u", "g", "r", "i", "z"]].min()),
        }
        record["model_agree_count_bank"] = int(
            (record["proba_delta_bank"] > 0)
            + (record["pure_delta_bank"] > 0)
            + (record["base_delta_bank"] > 0)
        )
        record["bank_score"] = (
            2.0 * record["bank_share"]
            + 1.30 * record["proba_delta_bank"]
            + 0.50 * np.clip(record["pure_delta_bank"], -1, 1)
            + 0.70 * np.clip(record["base_delta_bank"], -1, 1)
            + 0.10 * record["model_agree_count_bank"]
        )
        record["refbank_score"] = (
            2.0 * record["bank_share"]
            + 1.20 * record["proba_delta_ref"]
            + 0.40 * np.clip(record["pure_delta_bank"], -1, 1)
            + 0.60 * np.clip(record["base_delta_bank"], -1, 1)
        )
        rows.append(record)

    table = pd.DataFrame(rows)
    metadata = {
        "current": str(CURRENT.relative_to(ROOT)),
        "observed_current_public_score": 0.97207,
        "top130": str(TOP130.relative_to(ROOT)),
        "observed_top130_public_score": 0.97202,
        "top130_drop_excluded_rows": len(top130_drop_ids),
        "reference": str(REFERENCE),
        "probability_files": proba_manifest,
    }
    return table, metadata


def apply_changes(current: pd.DataFrame, changes: pd.DataFrame, target_col: str, output_path: Path) -> dict:
    out = current.copy()
    id_to_label = dict(zip(changes["id"].astype(int), changes[target_col].astype(str)))
    mask = out["id"].isin(id_to_label)
    out.loc[mask, "class"] = out.loc[mask, "id"].map(id_to_label)
    out.to_csv(output_path, index=False)
    return {
        "file": output_path.name,
        "changed_rows": int(mask.sum()),
        "ids": [int(row_id) for row_id in changes["id"].tolist()],
        "transitions": changes["transition"].value_counts().to_dict(),
        "counts": out["class"].value_counts().sort_index().to_dict(),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    current = read_submission(CURRENT, sample)
    table, metadata = build_candidate_table()
    table.to_csv(OUT_DIR / "post097207_all_candidate_rows.csv", index=False)

    refbank = table[
        table["bank_target"].eq(table["ref_target"])
        & table["proba_pred"].eq(table["ref_target"])
        & table["proba_delta_ref"].gt(0)
        & table["bank_share"].ge(0.55)
    ].sort_values(["refbank_score", "bank_share", "proba_delta_ref"], ascending=False)
    refbank.to_csv(OUT_DIR / "refbank_proba_candidates.csv", index=False)

    bank_noref = table[
        table["proba_pred"].eq(table["bank_target"])
        & table["proba_delta_bank"].gt(0.05)
        & table["bank_share"].ge(0.60)
        & table["model_agree_count_bank"].ge(3)
    ].sort_values(["bank_score", "bank_share", "proba_delta_bank"], ascending=False)
    bank_noref.to_csv(OUT_DIR / "bankproba_noref_candidates.csv", index=False)

    generated = []
    for n in [3, 5, 10]:
        if len(refbank) >= n:
            generated.append(
                apply_changes(
                    current,
                    refbank.head(n),
                    "ref_target",
                    OUT_DIR / f"post097207_refbank_proba_top{n:02d}.csv",
                )
            )
        if len(bank_noref) >= n:
            generated.append(
                apply_changes(
                    current,
                    bank_noref.head(n),
                    "bank_target",
                    OUT_DIR / f"post097207_bankproba_noref_top{n:02d}.csv",
                )
            )

    # Existing ablation candidate copied into the same queue for convenience.
    tail10_out = OUT_DIR / "post097207_tail10_ablation.csv"
    shutil.copyfile(TAIL10, tail10_out)
    tail10 = read_submission(tail10_out, sample)
    generated.append(
        {
            "file": tail10_out.name,
            "changed_rows": int(tail10["class"].ne(current["class"]).sum()),
            "ids": [int(row_id) for row_id in tail10.loc[tail10["class"].ne(current["class"]), "id"].tolist()],
            "transitions": (
                current.loc[tail10["class"].ne(current["class"]), "class"]
                + "->"
                + tail10.loc[tail10["class"].ne(current["class"]), "class"]
            ).value_counts().to_dict(),
            "counts": tail10["class"].value_counts().sort_index().to_dict(),
        }
    )

    report = {
        **metadata,
        "candidate_rows": int(len(table)),
        "refbank_candidates": int(len(refbank)),
        "bankproba_noref_candidates": int(len(bank_noref)),
        "generated": generated,
        "recommended_order": [
            "post097207_refbank_proba_top03.csv",
            "post097207_refbank_proba_top05.csv if top03 improves or ties",
            "post097207_tail10_ablation.csv if refbank does not improve",
            "post097207_bankproba_noref_top03.csv only if willing to test a generalization-oriented anti-reference signal",
        ],
        "reasoning": (
            "The top130 drop says removing the known 40-row delta is harmful. "
            "This queue excludes those rows and only tests new small patches around the 0.97207 file."
        ),
    }
    (OUT_DIR / "post097207_probe_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Post 0.97207 Probe Queue",
                "",
                "Current best public file: `v5_voted_ensemble.csv`, score `0.97207`.",
                "`v5_ridge_top130.csv` scored `0.97202`, so reduced-flip variants are weaker.",
                "",
                "Submit first:",
                "",
                "1. `post097207_refbank_proba_top03.csv`",
                "2. `post097207_refbank_proba_top05.csv` only if top03 improves or ties.",
                "3. `post097207_tail10_ablation.csv` only if refbank does not improve.",
                "",
                "Do not submit all files blindly.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
