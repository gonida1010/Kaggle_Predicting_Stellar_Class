from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import Counter
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
LABEL_TO_INT = {label: idx for idx, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Public-LB-only exploration from archive (9). This intentionally uses "
            "public-score submission banks and must not be treated as a private/CV candidate."
        )
    )
    parser.add_argument("--archive", type=Path, default=Path("/Users/parkyeonggon/Downloads/archive (9).zip"))
    parser.add_argument("--public-anchor", type=Path, default=OUTPUTS / "21_PUBLIC_097227_ridge_consensus_direct.csv")
    parser.add_argument("--public-anchor-score", type=float, default=0.97227)
    parser.add_argument("--private-candidates", type=Path, nargs="*", default=[
        OUTPUTS / "95_PRIVATE_CV_robust_next_56_te_disagreement_plus90_good_core_oof0970645.csv",
        OUTPUTS / "96_PRIVATE_CV_robust_next_rollback_weak_gk_red_to_56_te_disagreement_oof0970638.csv",
        OUTPUTS / "90_PRIVATE_CV_subset_guard_68_plus_84_good_union_oof0970627.csv",
    ])
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "public_archive9_explore_20260623")
    parser.add_argument("--output-prefix", default="PUBLIC_EXPLORE_archive9")
    return parser.parse_args()


def score_from_name(name: str) -> float | None:
    match = re.match(r"^(0\.\d{5})", Path(name).name)
    return float(match.group(1)) if match else None


def read_submission_path(path: Path, sample_ids: np.ndarray) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[ID, TARGET])
    if not np.array_equal(df[ID].to_numpy(), sample_ids):
        df = df.set_index(ID).reindex(sample_ids).reset_index()
    if df[TARGET].isna().any():
        raise ValueError(f"{path} has missing labels after id alignment")
    invalid = sorted(set(df[TARGET].unique()) - set(LABELS))
    if invalid:
        raise ValueError(f"{path} has invalid labels: {invalid}")
    return df[[ID, TARGET]].copy()


def read_submission_member(zf: zipfile.ZipFile, member: str, sample_ids: np.ndarray) -> pd.DataFrame | None:
    try:
        with zf.open(member) as handle:
            df = pd.read_csv(handle, usecols=[ID, TARGET])
    except Exception:
        return None
    if not np.array_equal(df[ID].to_numpy(), sample_ids):
        df = df.set_index(ID).reindex(sample_ids).reset_index()
    if df[TARGET].isna().any():
        return None
    if not set(df[TARGET].unique()).issubset(set(LABELS)):
        return None
    return df[[ID, TARGET]].copy()


def load_npy_member(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as handle:
        return np.load(io.BytesIO(handle.read()))


def load_archive_entries(zf: zipfile.ZipFile, sample_ids: np.ndarray) -> list[dict]:
    entries: list[dict] = []
    seen: set[bytes] = set()

    def add(name: str, score: float, member: str) -> None:
        df = read_submission_member(zf, member, sample_ids)
        if df is None:
            return
        labels = df[TARGET].to_numpy(dtype=object)
        key = labels.astype("U8").tobytes()
        if key in seen:
            return
        seen.add(key)
        entries.append(
            {
                "name": name,
                "member": member,
                "score": float(score),
                "labels": labels,
            }
        )

    for member in sorted(zf.namelist()):
        if not (member.startswith("new/") and member.endswith(".csv")):
            continue
        score = score_from_name(Path(member).name)
        if score is not None:
            add(Path(member).name, score, member)

    if "new/feedback_scores.csv" in zf.namelist():
        with zf.open("new/feedback_scores.csv") as handle:
            feedback = pd.read_csv(handle)
        for row in feedback.itertuples(index=False):
            member = str(row.path)
            if not member.startswith("new/"):
                member = f"new/{member}"
            if member not in zf.namelist():
                continue
            add(str(row.name), float(row.score), member)

    entries.sort(key=lambda item: (item["score"], item["name"]), reverse=True)
    return entries


def load_probability_consensus(zf: zipfile.ZipFile, sample_ids: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[dict]]:
    arrays: list[np.ndarray] = []
    weights: list[float] = []
    manifest: list[dict] = []

    for member in sorted(zf.namelist()):
        basename = Path(member).name
        if not (member.startswith("new/") and basename.startswith("test_preds__") and basename.endswith(".csv")):
            continue
        try:
            with zf.open(member) as handle:
                df = pd.read_csv(handle)
        except Exception:
            continue
        if ID not in df.columns or not set(LABELS).issubset(df.columns):
            continue
        if not np.array_equal(df[ID].to_numpy(), sample_ids):
            continue
        arr = df[list(LABELS)].to_numpy(np.float64)
        arr /= arr.sum(axis=1, keepdims=True)
        score = score_from_name(basename.replace("test_preds__", "")) or 0.969
        weight = max(0.25, (score - 0.968) * 1000.0)
        arrays.append(arr)
        weights.append(weight)
        manifest.append({"member": member, "score": score, "weight": weight})

    for basename, weight in [("cat-3_test_preds__0.96972.npy", 1.7), ("pred_lr_stacker_v9.npy", 3.5)]:
        member = f"new/{basename}"
        if member not in zf.namelist():
            continue
        arr = load_npy_member(zf, member).astype(np.float64)
        if arr.shape != (len(sample_ids), len(LABELS)):
            continue
        arr /= arr.sum(axis=1, keepdims=True)
        arrays.append(arr)
        weights.append(weight)
        manifest.append({"member": member, "score": None, "weight": weight})

    if not arrays:
        return None, None, None, manifest
    proba = np.average(np.stack(arrays), axis=0, weights=np.array(weights, dtype=np.float64))
    proba /= proba.sum(axis=1, keepdims=True)
    pred = LABELS[proba.argmax(axis=1)]
    sorted_proba = np.sort(proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    return proba, pred, margin, manifest


def weighted_consensus(entries: list[dict], sample_ids: np.ndarray, min_score: float, top_n: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    selected = [entry for entry in entries if entry["score"] >= min_score]
    if top_n is not None:
        selected = selected[:top_n]
    if not selected:
        raise ValueError(f"No entries for min_score={min_score}, top_n={top_n}")

    vote = np.zeros((len(sample_ids), len(LABELS)), dtype=np.float64)
    source_rows = []
    for entry in selected:
        weight = max(0.05, (entry["score"] - 0.9700) * 1000.0)
        labels = entry["labels"]
        for idx, label in enumerate(LABELS):
            vote[:, idx] += weight * (labels == label)
        source_rows.append({"name": entry["name"], "score": entry["score"], "weight": weight})

    total = vote.sum(axis=1, keepdims=True)
    share = vote / total
    pred = LABELS[share.argmax(axis=1)]
    sorted_share = np.sort(share, axis=1)
    margin = sorted_share[:, -1] - sorted_share[:, -2]
    return pred, share.max(axis=1), margin, source_rows


def transition_counts(base: np.ndarray, candidate: np.ndarray) -> dict[str, int]:
    changed = base != candidate
    return dict(Counter((base[changed] + "->" + candidate[changed]).tolist()))


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {label: int((labels == label).sum()) for label in LABELS}


def save_submission(sample_ids: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    pd.DataFrame({ID: sample_ids, TARGET: labels}).to_csv(output_path, index=False)


def build_ridge_rank(entries: list[dict], anchor_labels: np.ndarray, anchor_score: float, sample_ids: np.ndarray, proba_pred: np.ndarray | None, proba_margin: np.ndarray | None) -> pd.DataFrame:
    feature_to_col: dict[tuple[int, str], int] = {}
    rows: list[int] = []
    cols: list[int] = []
    used_entries = [entry for entry in entries if entry["score"] <= anchor_score]
    for row_idx, entry in enumerate(used_entries):
        changed = np.flatnonzero(entry["labels"] != anchor_labels)
        for pos in changed:
            key = (int(sample_ids[pos]), str(entry["labels"][pos]))
            col = feature_to_col.setdefault(key, len(feature_to_col))
            rows.append(row_idx)
            cols.append(col)

    if not feature_to_col:
        return pd.DataFrame()
    x = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(len(used_entries), len(feature_to_col)),
    )
    y = np.array([entry["score"] - anchor_score for entry in used_entries], dtype=np.float64)
    support = np.asarray((x > 0).sum(axis=0)).ravel()
    inv = [None] * len(feature_to_col)
    for key, col in feature_to_col.items():
        inv[col] = key

    model_specs = []
    for min_score in [0.968, 0.970, 0.971, 0.9715, 0.9720]:
        mask = np.array([entry["score"] >= min_score for entry in used_entries])
        if mask.sum() < 6:
            continue
        xx = x[mask]
        yy = y[mask]
        sample_weight = np.array([max(0.15, (entry["score"] - 0.968) * 900.0) for entry, keep in zip(used_entries, mask) if keep])
        for alpha in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(xx, yy, sample_weight=sample_weight)
            model_specs.append(np.asarray(model.coef_, dtype=np.float64))

    if not model_specs:
        return pd.DataFrame()
    coef_mat = np.vstack(model_specs)
    id_to_pos = {int(row_id): pos for pos, row_id in enumerate(sample_ids)}
    rows_out = []
    for col, (row_id, label) in enumerate(inv):
        pos = id_to_pos[int(row_id)]
        prob_agree = bool(proba_pred is not None and proba_pred[pos] == label)
        margin = float(proba_margin[pos]) if proba_margin is not None else 0.0
        median_coef = float(np.median(coef_mat[:, col]))
        q25 = float(np.quantile(coef_mat[:, col], 0.25))
        pos_rate = float((coef_mat[:, col] > 0).mean())
        robust_score = median_coef + 0.7 * q25 + (2.0e-5 if prob_agree else 0.0) + 1.0e-5 * np.tanh(margin * 10.0)
        rows_out.append(
            {
                "id": int(row_id),
                "from": str(anchor_labels[pos]),
                "to": str(label),
                "support": int(support[col]),
                "median_coef": median_coef,
                "q25_coef": q25,
                "pos_rate": pos_rate,
                "proba_agree": prob_agree,
                "proba_margin": margin,
                "robust_score": robust_score,
            }
        )
    return pd.DataFrame(rows_out).sort_values(
        ["robust_score", "pos_rate", "support", "proba_margin"],
        ascending=[False, False, False, False],
    )


def apply_rank(anchor_labels: np.ndarray, sample_ids: np.ndarray, rank: pd.DataFrame, k: int) -> np.ndarray:
    out = anchor_labels.copy()
    id_to_pos = {int(row_id): pos for pos, row_id in enumerate(sample_ids)}
    for row in rank.head(k).itertuples(index=False):
        out[id_to_pos[int(row.id)]] = str(row.to)
    return out


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    sample_ids = sample[ID].to_numpy()
    anchor = read_submission_path(args.public_anchor, sample_ids)
    anchor_labels = anchor[TARGET].to_numpy(dtype=object)

    private_refs = []
    for path in args.private_candidates:
        if path.exists():
            df = read_submission_path(path, sample_ids)
            private_refs.append({"name": path.name, "path": path, "labels": df[TARGET].to_numpy(dtype=object)})

    with zipfile.ZipFile(args.archive) as zf:
        entries = load_archive_entries(zf, sample_ids)
        proba, proba_pred, proba_margin, proba_manifest = load_probability_consensus(zf, sample_ids)

    if not entries:
        raise RuntimeError(f"No hard-label submissions found in {args.archive}")

    source_rows = []
    for entry in entries:
        source_rows.append(
            {
                "name": entry["name"],
                "member": entry["member"],
                "score": entry["score"],
                "diff_vs_public_anchor": int(np.sum(entry["labels"] != anchor_labels)),
                "class_GALAXY": int(np.sum(entry["labels"] == "GALAXY")),
                "class_QSO": int(np.sum(entry["labels"] == "QSO")),
                "class_STAR": int(np.sum(entry["labels"] == "STAR")),
            }
        )
    source_df = pd.DataFrame(source_rows).sort_values(["score", "name"], ascending=[False, True])
    source_df.to_csv(output_dir / "source_summary.csv", index=False)

    candidates: list[dict] = []

    def add_candidate(name: str, labels: np.ndarray, method: str, details: dict) -> None:
        output_name = f"{name}.csv"
        output_path = OUTPUTS / output_name
        save_submission(sample_ids, labels, output_path)
        changed = labels != anchor_labels
        candidate = {
            "candidate": name,
            "file": str(output_path.relative_to(ROOT)),
            "method": method,
            "changed_vs_public_anchor": int(changed.sum()),
            "changed_vs_private95": None,
            "class_counts": class_counts(labels),
            "transition_counts_vs_public_anchor": transition_counts(anchor_labels, labels),
            **details,
        }
        for ref in private_refs:
            candidate[f"diff_vs_{ref['name']}"] = int(np.sum(labels != ref["labels"]))
        candidates.append(candidate)

    # 1. Direct public-bank baselines. They are not recommended unless a user explicitly wants public-only probing.
    top_entry = entries[0]
    add_candidate(
        f"{args.output_prefix}_direct_top_{str(top_entry['score']).replace('.', 'p')}",
        top_entry["labels"],
        "direct_archive_top_public_baseline",
        {"source": top_entry["name"], "source_score": top_entry["score"]},
    )

    # 2. Score-weighted hard-label consensus families.
    for label, min_score, top_n in [
        ("top4_ge097220", 0.97220, None),
        ("top8_ge097214", 0.97214, None),
        ("top15_ge097207", 0.97207, None),
        ("top20_by_score", 0.0, 20),
    ]:
        pred, share, margin, used = weighted_consensus(entries, sample_ids, min_score=min_score, top_n=top_n)
        add_candidate(
            f"{args.output_prefix}_consensus_{label}",
            pred,
            "weighted_hard_label_consensus",
            {
                "min_score": min_score,
                "top_n": top_n,
                "sources_used": len(used),
                "mean_consensus_share": float(np.mean(share)),
                "changed_share_ge_070": int(np.sum((pred != anchor_labels) & (share >= 0.70))),
            },
        )

        # Public anchor patched only on high-confidence consensus disagreements.
        patched = anchor_labels.copy()
        patch_mask = (pred != anchor_labels) & (share >= 0.70) & (margin >= 0.20)
        patched[patch_mask] = pred[patch_mask]
        add_candidate(
            f"{args.output_prefix}_anchorpatch_{label}_share070_margin020",
            patched,
            "public_anchor_plus_high_confidence_consensus_patch",
            {
                "min_score": min_score,
                "top_n": top_n,
                "sources_used": len(used),
                "patch_rows": int(patch_mask.sum()),
            },
        )

    # 3. Private/generalization candidates only where public high-bank consensus agrees.
    for ref in private_refs:
        pred, share, margin, used = weighted_consensus(entries, sample_ids, min_score=0.97207, top_n=None)
        change_mask = ref["labels"] != anchor_labels
        agree_mask = change_mask & (pred == ref["labels"]) & (share >= 0.62)
        patched = anchor_labels.copy()
        patched[agree_mask] = ref["labels"][agree_mask]
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", ref["name"]).strip("_")[:48]
        add_candidate(
            f"{args.output_prefix}_public_private_agree_{safe_name}",
            patched,
            "public_anchor_plus_private_change_when_high_bank_agrees",
            {
                "private_source": ref["name"],
                "sources_used": len(used),
                "private_changed_rows": int(change_mask.sum()),
                "patch_rows": int(agree_mask.sum()),
            },
        )

    # 4. Ridge-style row flip inference from public-score bank. Use as public-only probe material.
    rank = build_ridge_rank(entries, anchor_labels, args.public_anchor_score, sample_ids, proba_pred, proba_margin)
    rank.to_csv(output_dir / "ridge_flip_rank.csv", index=False)
    if len(rank):
        for k in [1, 2, 4, 8]:
            labels = apply_rank(anchor_labels, sample_ids, rank, k)
            add_candidate(
                f"{args.output_prefix}_ridgeflip_k{k}",
                labels,
                "ridge_inferred_public_row_flip",
                {
                    "k": k,
                    "top_rows": rank.head(k).to_dict(orient="records"),
                },
            )

    summary_rows = []
    for candidate in candidates:
        row = {
            "candidate": candidate["candidate"],
            "file": candidate["file"],
            "method": candidate["method"],
            "changed_vs_public_anchor": candidate["changed_vs_public_anchor"],
            "transition_counts_vs_public_anchor": json.dumps(candidate["transition_counts_vs_public_anchor"], ensure_ascii=False),
            "class_counts": json.dumps(candidate["class_counts"], ensure_ascii=False),
        }
        for key, value in candidate.items():
            if key.startswith("diff_vs_"):
                row[key] = value
        for key in ["source_score", "sources_used", "patch_rows", "mean_consensus_share", "changed_share_ge_070"]:
            if key in candidate:
                row[key] = candidate[key]
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["method", "changed_vs_public_anchor", "candidate"])
    summary.to_csv(output_dir / "candidate_summary.csv", index=False)

    report = {
        "purpose": "PUBLIC-LEADERBOARD exploration only. Do not use this as private/generalization evidence.",
        "archive": str(args.archive),
        "public_anchor": str(args.public_anchor),
        "public_anchor_score": args.public_anchor_score,
        "entries_loaded": len(entries),
        "top_archive_sources": source_df.head(20).to_dict(orient="records"),
        "probability_sources": proba_manifest,
        "private_reference_files": [ref["name"] for ref in private_refs],
        "candidate_summary": summary_rows,
        "recommended_public_probe_order": [
            f"{args.output_prefix}_ridgeflip_k4.csv",
            f"{args.output_prefix}_anchorpatch_top15_ge097207_share070_margin020.csv",
            f"{args.output_prefix}_public_private_agree_95_PRIVATE_CV_robust_next_56_te_disagreement_plus90.csv",
        ],
        "notes": [
            "submission (9).csv is identical to outputs/21_PUBLIC_097227_ridge_consensus_direct.csv, so the new information is archive (9).zip.",
            "Public candidates are intentionally separated with PUBLIC_EXPLORE names.",
            "Higher public score from these files would not prove better private/generalization performance.",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
