from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import add_advanced_features  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
EXTERNAL = ROOT / "external_preds"
PURE_DIR = ARTIFACTS / "pure_model_ensemble"
CURRENT_PUBLIC = ARTIFACTS / "final_submissions" / "final_public_generalization.csv"
OUT_DIR = ARTIFACTS / "oof_proba_boundary_analysis"
LABELS = ["GALAXY", "QSO", "STAR"]


def parse_args() -> argparse.Namespace:
    default_reference = Path("/Users/parkyeonggon/Downloads/submission (1).csv")
    parser = argparse.ArgumentParser(
        description="Analyze OOF/proba errors and current-vs-reference boundary disagreements by subset."
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--current-submission", type=Path, default=CURRENT_PUBLIC)
    parser.add_argument(
        "--reference-submission",
        type=Path,
        default=default_reference if default_reference.exists() else None,
    )
    parser.add_argument("--top-rows", type=int, default=300)
    return parser.parse_args()


def progress(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def read_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)[["id", "class"]].copy()
    if not df["id"].equals(sample["id"]):
        raise ValueError(f"ID order mismatch: {path}")
    invalid = sorted(set(df["class"].dropna()) - set(LABELS))
    if invalid:
        raise ValueError(f"Invalid labels in {path}: {invalid}")
    return df


def score_from_name(path: Path) -> float | None:
    match = re.match(r"^(0\.\d{5})", path.name)
    return float(match.group(1)) if match else None


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    out = {}
    for label in labels:
        mask = y_true == label
        out[f"recall_{label}"] = float((y_pred[mask] == label).mean()) if mask.any() else np.nan
        out[f"support_{label}"] = int(mask.sum())
    return out


def balanced_accuracy_present(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> float:
    recalls = []
    for label in labels:
        mask = y_true == label
        if mask.any():
            recalls.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(recalls)) if recalls else np.nan


def add_bin_columns(train_fe: pd.DataFrame, test_fe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_out = train_fe.copy()
    test_out = test_fe.copy()
    specs = {
        "redshift": 10,
        "g-i": 10,
        "u-r": 10,
        "mag_range": 10,
        "redshift_x_g-i": 10,
    }
    bin_cols = []
    for col, q in specs.items():
        if col not in train_out.columns or col not in test_out.columns:
            continue
        quantiles = train_out[col].quantile(np.linspace(0, 1, q + 1)).to_numpy()
        edges = np.unique(quantiles)
        if len(edges) < 3:
            continue
        edges[0] = -np.inf
        edges[-1] = np.inf
        bin_col = f"{col}_bin"
        train_out[bin_col] = pd.cut(train_out[col], bins=edges, include_lowest=True).astype(str)
        test_out[bin_col] = pd.cut(test_out[col], bins=edges, include_lowest=True).astype(str)
        bin_cols.append(bin_col)
    return train_out, test_out, bin_cols


def probability_frame(proba: np.ndarray, labels: list[str], prefix: str) -> pd.DataFrame:
    sorted_probs = np.sort(proba, axis=1)
    pred_idx = proba.argmax(axis=1)
    out = pd.DataFrame(
        {
            f"{prefix}_pred": np.array(labels)[pred_idx],
            f"{prefix}_top_prob": sorted_probs[:, -1],
            f"{prefix}_margin": sorted_probs[:, -1] - sorted_probs[:, -2],
        }
    )
    for idx, label in enumerate(labels):
        out[f"{prefix}_p_{label}"] = proba[:, idx]
    return out


def top_error_pair(group: pd.DataFrame) -> tuple[str, int]:
    wrong = group[group["true_class"] != group["pure_pred"]]
    if wrong.empty:
        return "", 0
    counts = wrong.groupby(["true_class", "pure_pred"], observed=True).size().sort_values(ascending=False)
    pair = counts.index[0]
    return f"{pair[0]}->{pair[1]}", int(counts.iloc[0])


def subset_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for group_col in group_cols:
        if group_col not in df.columns:
            continue
        grouped = df.groupby(group_col, observed=True, dropna=False)
        for value, group in grouped:
            if len(group) < 50:
                continue
            pair, pair_count = top_error_pair(group)
            recalls = class_recalls(
                group["true_class"].to_numpy(),
                group["pure_pred"].to_numpy(),
                LABELS,
            )
            row = {
                "group_col": group_col,
                "group_value": str(value),
                "count": int(len(group)),
                "balanced_accuracy_present": balanced_accuracy_present(
                    group["true_class"].to_numpy(),
                    group["pure_pred"].to_numpy(),
                    LABELS,
                ),
                "error_rate": float((group["true_class"] != group["pure_pred"]).mean()),
                "avg_margin": float(group["pure_margin"].mean()),
                "top_error_pair": pair,
                "top_error_pair_count": pair_count,
                **recalls,
            }
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["error_rate", "count"], ascending=[False, False])


def build_bank_consensus(sample: pd.DataFrame) -> pd.DataFrame | None:
    files = []
    for path in EXTERNAL.glob("*.csv"):
        score = score_from_name(path)
        if score is not None:
            files.append((path, score))
    files.sort(key=lambda item: (item[1], item[0].name), reverse=True)
    if not files:
        return None

    votes = {label: np.zeros(len(sample), dtype=np.int16) for label in LABELS}
    used = 0
    for path, _ in files:
        try:
            sub = read_submission(path, sample)
        except Exception:
            continue
        labels = sub["class"].to_numpy()
        for label in LABELS:
            votes[label] += labels == label
        used += 1
    if used == 0:
        return None

    vote_matrix = np.column_stack([votes[label] for label in LABELS])
    idx = vote_matrix.argmax(axis=1)
    top = vote_matrix.max(axis=1)
    nonzero = (vote_matrix > 0).sum(axis=1)
    return pd.DataFrame(
        {
            "id": sample["id"],
            "bank_consensus": np.array(LABELS)[idx],
            "bank_consensus_share": top / used,
            "bank_nunique": nonzero,
            "bank_used_files": used,
        }
    )


def disagreement_segments(df: pd.DataFrame, left: str, right: str, group_cols: list[str]) -> pd.DataFrame:
    work = df[df[left] != df[right]].copy()
    if work.empty:
        return pd.DataFrame()
    work["transition"] = work[left] + "->" + work[right]
    rows = []
    for group_col in group_cols:
        if group_col not in work.columns:
            continue
        grouped = work.groupby([group_col, "transition"], observed=True, dropna=False)
        for (value, transition), group in grouped:
            if len(group) < 2:
                continue
            rows.append(
                {
                    "comparison": f"{left}_vs_{right}",
                    "group_col": group_col,
                    "group_value": str(value),
                    "transition": transition,
                    "count": int(len(group)),
                    "avg_pure_margin": float(group["pure_margin"].mean()),
                    "avg_redshift": float(group["redshift"].mean()),
                    "avg_g_i": float(group["g-i"].mean()),
                    "avg_mag_range": float(group["mag_range"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["count", "avg_pure_margin"], ascending=[False, True])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    progress("Loading train/test/sample data")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    progress("Building advanced train/test features and bins")
    train_fe = add_advanced_features(train)
    test_fe = add_advanced_features(test)
    train_fe, test_fe, bin_cols = add_bin_columns(train_fe, test_fe)

    encoder = LabelEncoder()
    y = encoder.fit_transform(train["class"].astype(str))
    classes = encoder.classes_.tolist()
    if classes != LABELS:
        raise ValueError(f"Unexpected class order: {classes}")

    progress("Loading pure OOF/test probabilities")
    pure_oof = np.load(PURE_DIR / "pure_model_ensemble_oof_proba.npy")
    pure_test = np.load(PURE_DIR / "pure_model_ensemble_test_proba.npy")

    progress("Building train/test diagnostic frames")
    train_diag = pd.concat(
        [
            train_fe.reset_index(drop=True),
            pd.DataFrame({"true_class": train["class"].astype(str)}),
            probability_frame(pure_oof, LABELS, "pure"),
        ],
        axis=1,
    )
    test_diag = pd.concat(
        [
            test_fe.reset_index(drop=True),
            probability_frame(pure_test, LABELS, "pure"),
        ],
        axis=1,
    )
    test_diag["id"] = sample["id"].to_numpy()

    progress("Loading current public candidate")
    current = read_submission(args.current_submission, sample)
    test_diag["current"] = current["class"].to_numpy()

    reference_used = None
    if args.reference_submission and args.reference_submission.exists():
        progress(f"Loading reference submission: {args.reference_submission}")
        reference = read_submission(args.reference_submission, sample)
        test_diag["reference"] = reference["class"].to_numpy()
        reference_used = str(args.reference_submission)

    progress("Building submission-bank consensus")
    bank = build_bank_consensus(sample)
    if bank is not None:
        test_diag = test_diag.merge(bank, on="id", how="left")

    group_cols = [
        "spectral_type",
        "galaxy_population",
        "spectral_population",
        *bin_cols,
    ]
    progress("Computing OOF subset metrics")
    train_subset = subset_metrics(train_diag, group_cols)
    train_subset.to_csv(args.output_dir / "oof_subset_metrics.csv", index=False)

    progress("Computing test disagreement segments")
    segment_tables = [
        disagreement_segments(test_diag, "current", "pure_pred", group_cols),
    ]
    if "reference" in test_diag.columns:
        segment_tables.append(disagreement_segments(test_diag, "current", "reference", group_cols))
        ref_gap = test_diag[test_diag["current"] != test_diag["reference"]].copy()
        ref_gap["transition"] = ref_gap["current"] + "->" + ref_gap["reference"]
        keep = [
            "id",
            "transition",
            "current",
            "reference",
            "pure_pred",
            "pure_margin",
            "pure_p_GALAXY",
            "pure_p_QSO",
            "pure_p_STAR",
            "spectral_type",
            "galaxy_population",
            "spectral_population",
            "redshift",
            "g-i",
            "u-r",
            "mag_range",
        ]
        if "bank_consensus" in ref_gap.columns:
            keep += ["bank_consensus", "bank_consensus_share", "bank_nunique"]
        ref_gap[keep].sort_values(["transition", "pure_margin"]).head(args.top_rows).to_csv(
            args.output_dir / "reference_gap_rows.csv",
            index=False,
        )
    if "bank_consensus" in test_diag.columns:
        segment_tables.append(disagreement_segments(test_diag, "current", "bank_consensus", group_cols))

    segments = pd.concat([table for table in segment_tables if table is not None and not table.empty], ignore_index=True)
    if not segments.empty:
        segments.to_csv(args.output_dir / "test_disagreement_segments.csv", index=False)

    current_vs_pure = test_diag[test_diag["current"] != test_diag["pure_pred"]].copy()
    current_vs_pure["transition"] = current_vs_pure["current"] + "->" + current_vs_pure["pure_pred"]
    keep_cols = [
        "id",
        "transition",
        "current",
        "pure_pred",
        "pure_margin",
        "pure_p_GALAXY",
        "pure_p_QSO",
        "pure_p_STAR",
        "spectral_type",
        "galaxy_population",
        "spectral_population",
        "redshift",
        "g-i",
        "u-r",
        "mag_range",
    ]
    current_vs_pure[keep_cols].sort_values(["pure_margin"], ascending=False).head(args.top_rows).to_csv(
        args.output_dir / "current_vs_pure_high_margin_rows.csv",
        index=False,
    )

    report = {
        "purpose": "OOF/proba subset error analysis and test boundary disagreement mining.",
        "current_submission": str(args.current_submission),
        "reference_submission": reference_used,
        "rows": {
            "train": int(len(train)),
            "test": int(len(test)),
            "current_vs_pure": int(len(current_vs_pure)),
            "current_vs_reference": int((test_diag["current"] != test_diag["reference"]).sum())
            if "reference" in test_diag.columns
            else None,
            "current_vs_bank_consensus": int((test_diag["current"] != test_diag["bank_consensus"]).sum())
            if "bank_consensus" in test_diag.columns
            else None,
        },
        "group_columns": group_cols,
        "outputs": [
            "oof_subset_metrics.csv",
            "test_disagreement_segments.csv",
            "current_vs_pure_high_margin_rows.csv",
            "reference_gap_rows.csv" if "reference" in test_diag.columns else None,
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    progress("Wrote boundary subset analysis report")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
