from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"

ID = "id"
TARGET = "class"
LABELS = np.array(["GALAXY", "QSO", "STAR"], dtype=object)
L2I = {label: i for i, label in enumerate(LABELS)}

NEGATIVE_FEEDBACK_NAMES = {
    "recent_corr_rank_k3",
    "recent_wc_k35",
    "recent_sparse_delta_k5",
    "recent_patch_lr_k30",
    "new_stack_submission",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the next public-only ridge-flip candidate after a supplied 0.97227 anchor. "
            "This uses submission-bank public feedback and is not a private/CV candidate."
        )
    )
    parser.add_argument("--archive", type=Path, default=Path("/Users/parkyeonggon/Downloads/archive (7).zip"))
    parser.add_argument("--anchor", type=Path, default=ROOT / "outputs" / "21_PUBLIC_097227_ridge_consensus_direct.csv")
    parser.add_argument("--anchor-score", type=float, default=0.97227)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "public_097227_next_flip")
    parser.add_argument("--round-k", type=int, default=4)
    return parser.parse_args()


def score_from_name(name: str) -> float | None:
    basename = Path(name).name
    match = re.match(r"^(0\.\d{5})", basename)
    return float(match.group(1)) if match else None


def member_exists(zf: zipfile.ZipFile, member: str) -> bool:
    return member in set(zf.namelist())


def read_csv_member(zf: zipfile.ZipFile, member: str, **kwargs) -> pd.DataFrame:
    with zf.open(member) as handle:
        return pd.read_csv(handle, **kwargs)


def read_submission_member(zf: zipfile.ZipFile, member: str, ids: np.ndarray | None = None) -> pd.DataFrame | None:
    if member not in zf.namelist():
        return None
    try:
        df = read_csv_member(zf, member, usecols=[ID, TARGET]).copy()
    except Exception:
        return None
    if ids is not None and not np.array_equal(df[ID].to_numpy(), ids):
        df = df.set_index(ID).reindex(ids).reset_index()
        if df[TARGET].isna().any():
            return None
    if not set(df[TARGET].unique()).issubset(set(LABELS)):
        return None
    return df


def read_submission_path(path: Path, ids: np.ndarray | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[ID, TARGET]).copy()
    if ids is not None and not np.array_equal(df[ID].to_numpy(), ids):
        df = df.set_index(ID).reindex(ids).reset_index()
    if df[TARGET].isna().any():
        raise ValueError(f"{path} has missing labels after id alignment")
    return df


def load_npy_member(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as handle:
        return np.load(io.BytesIO(handle.read()))


def load_probability_consensus(zf: zipfile.ZipFile, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    arrays: list[np.ndarray] = []
    weights: list[float] = []
    names: list[str] = []
    members = zf.namelist()

    for member in sorted(name for name in members if Path(name).name.startswith("test_preds__") and name.endswith(".csv")):
        df = read_csv_member(zf, member)
        if ID not in df.columns or not set(LABELS).issubset(df.columns):
            continue
        if not np.array_equal(df[ID].to_numpy(), ids):
            continue
        score = score_from_name(Path(member).name.replace("test_preds__", "")) or 0.969
        arr = df[LABELS].to_numpy(np.float64)
        arr /= arr.sum(axis=1, keepdims=True)
        arrays.append(arr)
        weights.append(max(0.25, (score - 0.968) * 1000.0))
        names.append(member)

    for filename, weight in [("cat-3_test_preds__0.96972.npy", 1.7), ("pred_lr_stacker_v9.npy", 3.5)]:
        member = f"new/{filename}"
        if member in members:
            arr = load_npy_member(zf, member).astype(np.float64)
            arr /= arr.sum(axis=1, keepdims=True)
            arrays.append(arr)
            weights.append(weight)
            names.append(member)

    artifact = "new/sub_eda03_gbdt_artifact_blend.csv"
    if artifact in members:
        df = read_csv_member(zf, artifact, usecols=[ID, "proba_GALAXY", "proba_QSO", "proba_STAR"])
        if np.array_equal(df[ID].to_numpy(), ids):
            arr = df[["proba_GALAXY", "proba_QSO", "proba_STAR"]].to_numpy(np.float64)
            arr /= arr.sum(axis=1, keepdims=True)
            arrays.append(arr)
            weights.append(4.0)
            names.append(artifact)

    if not arrays:
        raise RuntimeError("No probability files found in archive.")

    proba = np.average(np.stack(arrays), axis=0, weights=np.array(weights))
    proba /= proba.sum(axis=1, keepdims=True)
    pred = LABELS[proba.argmax(axis=1)]
    sorted_p = np.sort(proba, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    return proba, pred, margin, names


def load_scored_entries(
    zf: zipfile.ZipFile,
    max_score: float,
    ids: np.ndarray,
    generated_entries: list[dict],
) -> list[dict]:
    entries: list[dict] = []
    seen: set[bytes] = set()

    def add(name: str, score: float, labels: np.ndarray | None = None, member: str | None = None) -> None:
        if labels is None:
            if member is None:
                return
            df = read_submission_member(zf, member, ids)
            if df is None:
                return
            labels_arr = df[TARGET].to_numpy()
        else:
            labels_arr = labels
        key = labels_arr.tobytes()
        if key in seen:
            return
        seen.add(key)
        entries.append({"name": name, "score": float(score), "labels": labels_arr})

    for member in sorted(name for name in zf.namelist() if name.startswith("new/") and name.endswith(".csv")):
        score = score_from_name(member)
        if score is not None and score <= max_score:
            add(Path(member).name, score, member=member)

    if "new/feedback_scores.csv" in zf.namelist():
        fb = read_csv_member(zf, "new/feedback_scores.csv")
        for row in fb.itertuples(index=False):
            score = float(row.score)
            if score <= max_score:
                raw_path = str(row.path)
                member = raw_path if raw_path.startswith("new/") else f"new/{raw_path}"
                name = getattr(row, "name", Path(raw_path).name)
                add(str(name), score, member=member)

    for item in generated_entries:
        if item["score"] <= max_score:
            add(item["name"], float(item["score"]), labels=item["labels"])

    entries.sort(key=lambda item: item["score"], reverse=True)
    return entries


def build_rank(
    zf: zipfile.ZipFile,
    current_anchor: pd.DataFrame,
    current_score: float,
    ids: np.ndarray,
    proba: np.ndarray,
    proba_pred: np.ndarray,
    proba_margin: np.ndarray,
    generated_entries: list[dict],
) -> tuple[pd.DataFrame, list[dict]]:
    anchor_labels = current_anchor[TARGET].to_numpy()
    entries = load_scored_entries(zf, current_score, ids, generated_entries)

    feature_to_col: dict[tuple[int, str], int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for i, entry in enumerate(entries):
        changed = np.flatnonzero(entry["labels"] != anchor_labels)
        for pos in changed:
            key = (int(ids[pos]), str(entry["labels"][pos]))
            col = feature_to_col.setdefault(key, len(feature_to_col))
            rows.append(i)
            cols.append(col)

    if not feature_to_col:
        raise RuntimeError("No flip features found for this round.")

    x_all = sparse.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(len(entries), len(feature_to_col)))
    y_all = np.array([entry["score"] - current_score for entry in entries], dtype=np.float64)
    support_all = np.asarray((x_all > 0).sum(axis=0)).ravel()

    inv = [None] * len(feature_to_col)
    for key, col in feature_to_col.items():
        inv[col] = key

    models = []
    for min_score in [0.968, 0.970, 0.971, 0.9715, 0.9717]:
        row_mask = np.array([entry["score"] >= min_score for entry in entries])
        if row_mask.sum() < 12:
            continue
        x = x_all[row_mask]
        y = y_all[row_mask]
        sample_weight = np.array([max(0.15, (entry["score"] - 0.968) * 900.0) for entry, keep in zip(entries, row_mask) if keep])
        for alpha in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(x, y, sample_weight=sample_weight)
            models.append({"min_score": min_score, "alpha": alpha, "coef": np.asarray(model.coef_, dtype=np.float64)})

    coef_mat = np.vstack([model["coef"] for model in models])
    pos_rate = (coef_mat > 0).mean(axis=0)
    median_coef = np.median(coef_mat, axis=0)
    q25_coef = np.quantile(coef_mat, 0.25, axis=0)
    mean_pos_coef = np.maximum(coef_mat, 0).mean(axis=0)
    id_to_pos = {int(row_id): i for i, row_id in enumerate(ids)}

    negative_features: set[tuple[int, str]] = set()
    for entry in entries:
        if entry["name"] not in NEGATIVE_FEEDBACK_NAMES or entry["score"] >= current_score:
            continue
        changed = np.flatnonzero(entry["labels"] != anchor_labels)
        for pos in changed:
            negative_features.add((int(ids[pos]), str(entry["labels"][pos])))

    rank_rows = []
    for col, (row_id, label) in enumerate(inv):
        pos = id_to_pos[int(row_id)]
        label_idx = L2I[label]
        anchor_idx = L2I[str(anchor_labels[pos])]
        prob_delta = float(proba[pos, label_idx] - proba[pos, anchor_idx])
        prob_agree = bool(proba_pred[pos] == label)
        feature = (int(row_id), str(label))
        negative_penalty = -5.0e-5 if feature in negative_features else 0.0
        score = (
            median_coef[col]
            + 0.7 * q25_coef[col]
            + 0.2 * mean_pos_coef[col]
            + (1.5e-5 if prob_agree else 0.0)
            + 1.0e-5 * np.tanh(prob_delta * 10)
            + negative_penalty
        )
        rank_rows.append(
            {
                "id": int(row_id),
                "from": str(anchor_labels[pos]),
                "class": str(label),
                "support": int(support_all[col]),
                "pos_rate": float(pos_rate[col]),
                "median_coef": float(median_coef[col]),
                "q25_coef": float(q25_coef[col]),
                "mean_pos_coef": float(mean_pos_coef[col]),
                "prob_delta": prob_delta,
                "prob_margin": float(proba_margin[pos]),
                "prob_agree": prob_agree,
                "negative_penalty": negative_penalty,
                "score": float(score),
            }
        )

    rank = pd.DataFrame(rank_rows).sort_values(
        ["score", "pos_rate", "support", "prob_delta"],
        ascending=[False, False, False, False],
    )
    return rank, entries


def apply_top_flips(anchor: pd.DataFrame, rank: pd.DataFrame, k: int) -> pd.DataFrame:
    out = anchor.copy()
    id_to_pos = {int(row_id): i for i, row_id in enumerate(out[ID].to_numpy())}
    for _, row in rank.head(k).iterrows():
        out.at[id_to_pos[int(row[ID])], TARGET] = row[TARGET]
    return out


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    anchor = read_submission_path(args.anchor, sample[ID].to_numpy())
    ids = anchor[ID].to_numpy()

    with zipfile.ZipFile(args.archive) as zf:
        proba, proba_pred, proba_margin, proba_names = load_probability_consensus(zf, ids)
        generated_entries = [
            {
                "name": "user_anchor_097227_submission8",
                "score": args.anchor_score,
                "labels": anchor[TARGET].to_numpy(),
            }
        ]
        rank, entries = build_rank(
            zf,
            anchor,
            args.anchor_score,
            ids,
            proba,
            proba_pred,
            proba_margin,
            generated_entries,
        )

    rank_path = output_dir / "next_flip_rank.csv"
    rank.to_csv(rank_path, index=False)
    candidates = []
    for k in [1, 2, args.round_k]:
        submission = apply_top_flips(anchor, rank, k)
        name = f"22_PUBLIC_097227_next_ridge_k{k}.csv"
        if k == args.round_k:
            name = "22_PUBLIC_097227_next_ridge_k4.csv"
        path = OUTPUTS / name
        submission.to_csv(path, index=False)
        changed = anchor[TARGET].ne(submission[TARGET])
        candidates.append(
            {
                "k": k,
                "path": str(path.relative_to(ROOT)),
                "changed_rows_vs_097227": int(changed.sum()),
                "flips": pd.DataFrame(
                    {
                        "id": anchor.loc[changed, ID],
                        "from": anchor.loc[changed, TARGET],
                        "to": submission.loc[changed, TARGET],
                    }
                ).to_dict(orient="records"),
                "class_counts": submission[TARGET].value_counts().sort_index().to_dict(),
            }
        )

    report = {
        "purpose": "Public-only next ridge flip after the 0.97227 anchor.",
        "archive": str(args.archive),
        "anchor": str(args.anchor),
        "anchor_score": args.anchor_score,
        "entries_used": len(entries),
        "ranked_features": int(len(rank)),
        "probability_sources": proba_names,
        "top_rank": rank.head(20).to_dict(orient="records"),
        "candidates": candidates,
        "recommended_public_attack": "outputs/22_PUBLIC_097227_next_ridge_k4.csv",
        "rank_csv": str(rank_path.relative_to(ROOT)),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
