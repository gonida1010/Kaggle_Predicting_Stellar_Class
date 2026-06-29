from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
LABELS = np.array(["GALAXY", "QSO", "STAR"], dtype=object)

sys.path.insert(0, str(ROOT / "scripts"))
from build_advanced_ridge_v3_candidates import (  # noqa: E402
    load_probability_arrays,
    probability_consensus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the external 0.97252 public reference against the prior "
            "0.97244 anchor, probability consensus, and high-OOF local candidates."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/Users/parkyeonggon/Downloads/submission (14).csv"),
    )
    parser.add_argument(
        "--old-anchor",
        type=Path,
        default=OUTPUTS / "211_PUBLIC_V3_anchor097244_direct.csv",
    )
    parser.add_argument(
        "--bank-archive",
        type=Path,
        default=Path("/Users/parkyeonggon/Downloads/archive (11).zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "public_097252_analysis_20260629",
    )
    parser.add_argument("--private-oof-min", type=float, default=0.970652)
    parser.add_argument(
        "--direct-output",
        type=Path,
        default=OUTPUTS / "219_PUBLIC_REF097252_direct_external.csv",
    )
    return parser.parse_args()


def align(df: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    if np.array_equal(df["id"].to_numpy(), ids):
        return df.reset_index(drop=True)
    out = df.set_index("id").reindex(ids).reset_index()
    if out["class"].isna().any():
        raise ValueError("Submission IDs do not align.")
    return out


def parse_oof_score(name: str) -> float | None:
    match = re.search(r"oof0?(\d{6})", name)
    return int(match.group(1)) / 1_000_000 if match else None


def load_private_consensus(ids: np.ndarray, minimum: float) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    arrays = []
    records = []
    seen = set()
    for path in sorted(OUTPUTS.glob("*PRIVATE*.csv")):
        score = parse_oof_score(path.name)
        if score is None or score < minimum:
            continue
        frame = align(pd.read_csv(path, usecols=["id", "class"]), ids)
        labels = frame["class"].to_numpy(dtype=object)
        key = labels.astype("U8").tobytes()
        if key in seen:
            continue
        seen.add(key)
        arrays.append(labels)
        records.append({"file": path.name, "oof_score": score})
    if not arrays:
        raise RuntimeError("No high-OOF private candidates were found.")
    matrix = np.stack(arrays)
    votes = np.stack([(matrix == label).sum(axis=0) for label in LABELS], axis=1)
    return LABELS[votes.argmax(axis=1)], votes.max(axis=1) / len(matrix), records


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = (ROOT / args.output_dir).resolve()
    if not args.direct_output.is_absolute():
        args.direct_output = (ROOT / args.direct_output).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.direct_output.parent.mkdir(parents=True, exist_ok=True)

    ids = pd.read_csv(DATA / "sample_submission.csv", usecols=["id"])["id"].to_numpy()
    reference = align(pd.read_csv(args.reference, usecols=["id", "class"]), ids)
    old_anchor = align(pd.read_csv(args.old_anchor, usecols=["id", "class"]), ids)
    private_pred, private_support, private_records = load_private_consensus(ids, args.private_oof_min)

    with zipfile.ZipFile(args.bank_archive) as bank:
        proba_arrays, proba_weights, proba_records = load_probability_arrays(bank, ids)
    proba, proba_pred, proba_margin = probability_consensus(proba_arrays, proba_weights, 1.0)

    old_labels = old_anchor["class"].to_numpy(dtype=object)
    new_labels = reference["class"].to_numpy(dtype=object)
    changed_positions = np.flatnonzero(old_labels != new_labels)
    test = pd.read_csv(DATA / "test.csv").set_index("id")
    rows = []
    for pos in changed_positions:
        row_id = int(ids[pos])
        old_label = str(old_labels[pos])
        new_label = str(new_labels[pos])
        old_idx = int(np.flatnonzero(LABELS == old_label)[0])
        new_idx = int(np.flatnonzero(LABELS == new_label)[0])
        row = {
            "id": row_id,
            "old_097244": old_label,
            "new_097252": new_label,
            "transition": f"{old_label}->{new_label}",
            "probability_consensus": str(proba_pred[pos]),
            "probability_margin": float(proba_margin[pos]),
            "probability_delta_new_minus_old": float(proba[pos, new_idx] - proba[pos, old_idx]),
            "private_consensus": str(private_pred[pos]),
            "private_support": float(private_support[pos]),
            "probability_supports_new": bool(proba_pred[pos] == new_label),
            "private_supports_new": bool(private_pred[pos] == new_label),
            "both_support_new": bool(proba_pred[pos] == new_label and private_pred[pos] == new_label),
            "both_support_old": bool(proba_pred[pos] == old_label and private_pred[pos] == old_label),
        }
        for feature in ["spectral_type", "galaxy_population", "redshift", "u", "g", "r", "i", "z"]:
            if feature in test.columns:
                row[feature] = test.loc[row_id, feature]
        rows.append(row)

    changes = pd.DataFrame(rows).sort_values(
        ["both_support_new", "probability_delta_new_minus_old"],
        ascending=[False, False],
    )
    transition_summary = (
        changes.groupby("transition", as_index=False)
        .agg(
            rows=("id", "size"),
            probability_supports_new=("probability_supports_new", "sum"),
            private_supports_new=("private_supports_new", "sum"),
            both_support_new=("both_support_new", "sum"),
            both_support_old=("both_support_old", "sum"),
            mean_probability_delta=("probability_delta_new_minus_old", "mean"),
        )
        .sort_values("rows", ascending=False)
    )
    changes.to_csv(args.output_dir / "changed_rows_vs_097244.csv", index=False)
    transition_summary.to_csv(args.output_dir / "transition_summary.csv", index=False)
    pd.DataFrame(private_records).to_csv(args.output_dir / "private_consensus_sources.csv", index=False)
    pd.DataFrame(proba_records).to_csv(args.output_dir / "probability_sources.csv", index=False)
    shutil.copyfile(args.reference, args.direct_output)

    report = {
        "purpose": "Analyze the external public reference without treating it as OOF/private evidence.",
        "reference_file": str(args.reference),
        "old_anchor_file": str(args.old_anchor),
        "changed_rows_vs_097244": int(len(changes)),
        "transition_counts": changes["transition"].value_counts().to_dict(),
        "probability_supports_new": int(changes["probability_supports_new"].sum()),
        "private_supports_new": int(changes["private_supports_new"].sum()),
        "both_support_new": int(changes["both_support_new"].sum()),
        "both_support_old": int(changes["both_support_old"].sum()),
        "both_support_new_ids": changes.loc[changes["both_support_new"], "id"].astype(int).tolist(),
        "both_support_old_ids": changes.loc[changes["both_support_old"], "id"].astype(int).tolist(),
        "private_consensus_source_count": len(private_records),
        "probability_source_count": len(proba_records),
        "class_counts": reference["class"].value_counts().to_dict(),
        "direct_external_output": str(args.direct_output.relative_to(ROOT)),
        "warning": (
            "The supplied notebook selected zero flips in every round. Its output is the input "
            "0.97254-named anchor, and the hard-coded 0.97255/0.97256 feedback scores are not "
            "observed scores for changed predictions."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
