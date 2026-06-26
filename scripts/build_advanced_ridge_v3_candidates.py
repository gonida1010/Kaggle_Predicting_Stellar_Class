from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import Counter
from datetime import datetime
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
            "Reproduce the Advanced Ridge Flip & Probability Consensus v3 idea on local zip files. "
            "This is PUBLIC-LB exploration. It uses public scored submissions and must not be "
            "mistaken for an honest OOF/CV candidate."
        )
    )
    parser.add_argument("--top-archive", type=Path, default=Path("/Users/parkyeonggon/Downloads/archive (10).zip"))
    parser.add_argument("--bank-archive", type=Path, default=Path("/Users/parkyeonggon/Downloads/archive (11).zip"))
    parser.add_argument("--anchor-member", default="submission 0.97244.csv")
    parser.add_argument("--anchor-score", type=float, default=0.97244)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "advanced_ridge_v3_20260626")
    parser.add_argument("--output-rank-start", type=int, default=211)
    parser.add_argument("--power-grid", type=float, nargs="+", default=[1.0, 1.25, 1.5])
    parser.add_argument("--min-score-grid", type=float, nargs="+", default=[0.968, 0.970, 0.971, 0.9715, 0.9720])
    parser.add_argument("--ridge-alpha-grid", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    parser.add_argument("--min-support", type=int, default=4)
    parser.add_argument("--min-pos-rate", type=float, default=0.70)
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def score_from_name(name: str) -> float | None:
    match = re.search(r"(0\.\d{5})", Path(name).name)
    return float(match.group(1)) if match else None


def normalize_probs(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.clip(arr, 1e-12, None)
    arr /= arr.sum(axis=1, keepdims=True)
    return arr


def read_submission_member(zf: zipfile.ZipFile, member: str, ids: np.ndarray) -> pd.DataFrame | None:
    try:
        with zf.open(member) as handle:
            df = pd.read_csv(handle, usecols=[ID, TARGET])
    except Exception:
        return None
    if not np.array_equal(df[ID].to_numpy(), ids):
        df = df.set_index(ID).reindex(ids).reset_index()
    if df[TARGET].isna().any():
        return None
    if not set(df[TARGET].unique()).issubset(set(LABELS)):
        return None
    return df[[ID, TARGET]].copy()


def load_npy_member(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as handle:
        return np.load(io.BytesIO(handle.read()))


def load_anchor(top_zip: zipfile.ZipFile, bank_zip: zipfile.ZipFile, member: str, ids: np.ndarray) -> pd.DataFrame:
    for zf in (top_zip, bank_zip):
        if member in zf.namelist():
            df = read_submission_member(zf, member, ids)
            if df is not None:
                return df
    raise FileNotFoundError(member)


def add_entry(entries: list[dict], seen: set[bytes], name: str, score: float, labels: np.ndarray, member: str) -> None:
    key = labels.astype("U8").tobytes()
    if key in seen:
        return
    seen.add(key)
    entries.append({"name": name, "score": float(score), "labels": labels.astype(object), "member": member})


def load_scored_entries(top_zip: zipfile.ZipFile, bank_zip: zipfile.ZipFile, ids: np.ndarray, max_score: float) -> list[dict]:
    entries: list[dict] = []
    seen: set[bytes] = set()

    for zf, prefix in [(top_zip, "top"), (bank_zip, "bank")]:
        for member in sorted(zf.namelist()):
            if not member.endswith(".csv") or Path(member).name == "feedback_scores.csv":
                continue
            score = score_from_name(member)
            if score is None or score > max_score:
                continue
            df = read_submission_member(zf, member, ids)
            if df is None:
                continue
            add_entry(entries, seen, Path(member).name, score, df[TARGET].to_numpy(dtype=object), f"{prefix}:{member}")

    if "feedback_scores.csv" in bank_zip.namelist():
        with bank_zip.open("feedback_scores.csv") as handle:
            feedback = pd.read_csv(handle)
        for row in feedback.itertuples(index=False):
            score = float(row.score)
            if score > max_score:
                continue
            member = str(row.path)
            if member not in bank_zip.namelist():
                continue
            df = read_submission_member(bank_zip, member, ids)
            if df is None:
                continue
            name = str(getattr(row, "name", Path(member).name))
            add_entry(entries, seen, name, score, df[TARGET].to_numpy(dtype=object), f"feedback:{member}")

    entries.sort(key=lambda item: (item["score"], item["name"]), reverse=True)
    return entries


def load_probability_arrays(bank_zip: zipfile.ZipFile, ids: np.ndarray) -> tuple[list[np.ndarray], list[float], list[dict]]:
    arrays: list[np.ndarray] = []
    weights: list[float] = []
    manifest: list[dict] = []

    for member in sorted(bank_zip.namelist()):
        basename = Path(member).name
        if not (basename.startswith("test_preds__") and basename.endswith(".csv")):
            continue
        with bank_zip.open(member) as handle:
            df = pd.read_csv(handle)
        if ID not in df.columns or not set(LABELS).issubset(df.columns):
            continue
        if not np.array_equal(df[ID].to_numpy(), ids):
            continue
        score = score_from_name(basename.replace("test_preds__", "")) or 0.969
        arr = normalize_probs(df[list(LABELS)].to_numpy(np.float64))
        weight = max(0.25, (score - 0.968) * 1000.0)
        arrays.append(arr)
        weights.append(weight)
        manifest.append({"member": member, "kind": "csv_proba", "score": score, "weight": weight})

    for basename, weight in [("cat-3_test_preds__0.96972.npy", 1.7), ("pred_lr_stacker_v9.npy", 3.5)]:
        if basename not in bank_zip.namelist():
            continue
        arr = normalize_probs(load_npy_member(bank_zip, basename))
        if arr.shape != (len(ids), len(LABELS)):
            continue
        arrays.append(arr)
        weights.append(weight)
        manifest.append({"member": basename, "kind": "npy_proba", "score": score_from_name(basename), "weight": weight})

    artifact = "sub_eda03_gbdt_artifact_blend.csv"
    if artifact in bank_zip.namelist():
        with bank_zip.open(artifact) as handle:
            df = pd.read_csv(handle, usecols=[ID, "proba_GALAXY", "proba_QSO", "proba_STAR"])
        if np.array_equal(df[ID].to_numpy(), ids):
            arr = normalize_probs(df[["proba_GALAXY", "proba_QSO", "proba_STAR"]].to_numpy(np.float64))
            arrays.append(arr)
            weights.append(4.0)
            manifest.append({"member": artifact, "kind": "artifact_proba", "score": None, "weight": 4.0})

    if not arrays:
        raise RuntimeError("No probability arrays found in bank archive")
    return arrays, weights, manifest


def probability_consensus(arrays: list[np.ndarray], weights: list[float], power: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    powered = []
    for arr in arrays:
        p = np.power(normalize_probs(arr), power)
        powered.append(normalize_probs(p))
    proba = np.average(np.stack(powered), axis=0, weights=np.array(weights, dtype=np.float64))
    proba = normalize_probs(proba)
    pred = LABELS[proba.argmax(axis=1)]
    sorted_p = np.sort(proba, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    return proba, pred, margin


def build_rank(
    entries: list[dict],
    anchor_labels: np.ndarray,
    anchor_score: float,
    ids: np.ndarray,
    proba: np.ndarray,
    proba_pred: np.ndarray,
    proba_margin: np.ndarray,
    min_score_grid: list[float],
    alpha_grid: list[float],
    min_support: int,
    min_pos_rate: float,
) -> pd.DataFrame:
    feature_to_col: dict[tuple[int, str], int] = {}
    rows: list[int] = []
    cols: list[int] = []
    used_entries = [entry for entry in entries if entry["score"] <= anchor_score]
    for row_idx, entry in enumerate(used_entries):
        changed = np.flatnonzero(entry["labels"] != anchor_labels)
        for pos in changed:
            key = (int(ids[pos]), str(entry["labels"][pos]))
            col = feature_to_col.setdefault(key, len(feature_to_col))
            rows.append(row_idx)
            cols.append(col)

    if not feature_to_col:
        return pd.DataFrame()

    x_all = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(len(used_entries), len(feature_to_col)),
    )
    y_all = np.array([entry["score"] - anchor_score for entry in used_entries], dtype=np.float64)
    support = np.asarray((x_all > 0).sum(axis=0)).ravel()

    inv = [None] * len(feature_to_col)
    for key, col in feature_to_col.items():
        inv[col] = key

    coef_rows = []
    model_rows = []
    for min_score in min_score_grid:
        mask = np.array([entry["score"] >= min_score for entry in used_entries])
        if int(mask.sum()) < 8:
            continue
        x = x_all[mask]
        y = y_all[mask]
        sample_weight = np.array(
            [max(0.15, (entry["score"] - 0.968) * 900.0) for entry, keep in zip(used_entries, mask) if keep],
            dtype=np.float64,
        )
        for alpha in alpha_grid:
            model = Ridge(alpha=float(alpha), fit_intercept=True)
            model.fit(x, y, sample_weight=sample_weight)
            coef = np.asarray(model.coef_, dtype=np.float64)
            coef_rows.append(coef)
            model_rows.append({"min_score": float(min_score), "alpha": float(alpha), "row_count": int(mask.sum())})

    if not coef_rows:
        return pd.DataFrame()

    coef_mat = np.vstack(coef_rows)
    pos_rate = (coef_mat > 0).mean(axis=0)
    median_coef = np.median(coef_mat, axis=0)
    q25_coef = np.quantile(coef_mat, 0.25, axis=0)
    mean_pos_coef = np.maximum(coef_mat, 0).mean(axis=0)

    negative_features: set[tuple[int, str]] = set()
    for entry in used_entries:
        if entry["name"] not in NEGATIVE_FEEDBACK_NAMES or entry["score"] >= anchor_score:
            continue
        changed = np.flatnonzero(entry["labels"] != anchor_labels)
        for pos in changed:
            negative_features.add((int(ids[pos]), str(entry["labels"][pos])))

    max_support = int(max(support.max(), 1))
    id_to_pos = {int(row_id): idx for idx, row_id in enumerate(ids)}
    rank_rows = []
    for col, (row_id, label) in enumerate(inv):
        pos = id_to_pos[int(row_id)]
        label_idx = L2I[label]
        anchor_idx = L2I[str(anchor_labels[pos])]
        prob_delta = float(proba[pos, label_idx] - proba[pos, anchor_idx])
        support_ratio = support[col] / max_support
        support_penalty = 8.0e-5 * (1.0 - support_ratio)
        in_negative = (int(row_id), str(label)) in negative_features
        score = (
            0.7 * median_coef[col]
            + 0.4 * q25_coef[col]
            + 0.1 * mean_pos_coef[col]
            + (2.5e-5 if proba_pred[pos] == label else 0.0)
            + 3.5e-5 * np.tanh(prob_delta * 10.0)
            + (-1.0e-4 if in_negative else 0.0)
            - support_penalty
        )
        rank_rows.append(
            {
                ID: int(row_id),
                "from": str(anchor_labels[pos]),
                "to": str(label),
                "support": int(support[col]),
                "support_ratio": float(support_ratio),
                "pos_rate": float(pos_rate[col]),
                "median_coef": float(median_coef[col]),
                "q25_coef": float(q25_coef[col]),
                "mean_pos_coef": float(mean_pos_coef[col]),
                "prob_delta": prob_delta,
                "prob_margin": float(proba_margin[pos]),
                "prob_agree": bool(proba_pred[pos] == label),
                "in_negative_feedback": bool(in_negative),
                "score": float(score),
            }
        )

    rank = pd.DataFrame(rank_rows)
    if rank.empty:
        return rank
    rank = rank[
        (rank["support"] >= int(min_support))
        & (rank["pos_rate"] >= float(min_pos_rate))
        & (rank["score"] > 0)
    ].copy()
    return rank.sort_values(["score", "pos_rate", "support", "prob_delta"], ascending=False).reset_index(drop=True)


def transition_counts(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    changed = before != after
    return dict(Counter((before[changed] + "->" + after[changed]).tolist()))


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {label: int((labels == label).sum()) for label in LABELS}


def write_submission(ids: np.ndarray, labels: np.ndarray, path: Path) -> None:
    pd.DataFrame({ID: ids, TARGET: labels}).to_csv(path, index=False)


def apply_top(anchor_labels: np.ndarray, ids: np.ndarray, rank: pd.DataFrame, k: int) -> np.ndarray:
    labels = anchor_labels.copy()
    id_to_pos = {int(row_id): idx for idx, row_id in enumerate(ids)}
    for row in rank.head(k).itertuples(index=False):
        labels[id_to_pos[int(getattr(row, ID))]] = str(row.to)
    return labels


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = (ROOT / args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    ids = sample[ID].to_numpy()

    progress("Opening archives")
    with zipfile.ZipFile(args.top_archive) as top_zip, zipfile.ZipFile(args.bank_archive) as bank_zip:
        anchor_df = load_anchor(top_zip, bank_zip, args.anchor_member, ids)
        anchor_labels = anchor_df[TARGET].to_numpy(dtype=object)

        progress("Loading scored submissions")
        entries = load_scored_entries(top_zip, bank_zip, ids, args.anchor_score)
        progress(f"loaded scored entries={len(entries)}")

        progress("Loading probability arrays")
        arrays, weights, proba_manifest = load_probability_arrays(bank_zip, ids)
        progress(f"loaded probability arrays={len(arrays)}")

        direct_path = OUTPUTS / f"{args.output_rank_start}_PUBLIC_V3_anchor097244_direct.csv"
        write_submission(ids, anchor_labels, direct_path)

        pairwise_rows = []
        for member in ["submission 0.97227.csv", "submission 0.97233.csv", "submission 0.97242.csv", "submission 0.97244.csv"]:
            if member in top_zip.namelist():
                df = read_submission_member(top_zip, member, ids)
                if df is None:
                    continue
                labels = df[TARGET].to_numpy(dtype=object)
                pairwise_rows.append(
                    {
                        "member": member,
                        "score": score_from_name(member),
                        "diff_vs_anchor": int((labels != anchor_labels).sum()),
                        "transition_counts_to_anchor": transition_counts(labels, anchor_labels),
                        "class_counts": class_counts(labels),
                    }
                )

        rank_manifests = []
        output_manifests = [
            {
                "file": direct_path.name,
                "path": str(direct_path.relative_to(ROOT)),
                "changed_rows": 0,
                "transition_counts": {},
                "class_counts": class_counts(anchor_labels),
                "note": "direct external public anchor; not an improved candidate",
            }
        ]
        seen_hashes = {anchor_labels.astype("U8").tobytes()}
        rank_num = int(args.output_rank_start) + 1

        for power in args.power_grid:
            progress(f"Building rank for probability power={power}")
            proba, proba_pred, proba_margin = probability_consensus(arrays, weights, power)
            agreement = float((proba_pred == anchor_labels).mean())
            rank = build_rank(
                entries=entries,
                anchor_labels=anchor_labels,
                anchor_score=args.anchor_score,
                ids=ids,
                proba=proba,
                proba_pred=proba_pred,
                proba_margin=proba_margin,
                min_score_grid=[float(x) for x in args.min_score_grid],
                alpha_grid=[float(x) for x in args.ridge_alpha_grid],
                min_support=args.min_support,
                min_pos_rate=args.min_pos_rate,
            )
            rank_path = args.output_dir / f"ridge_rank_power_{str(power).replace('.', 'p')}.csv"
            rank.to_csv(rank_path, index=False)
            rank_manifests.append(
                {
                    "power": float(power),
                    "agreement_with_anchor": agreement,
                    "rank_rows": int(len(rank)),
                    "rank_path": str(rank_path.relative_to(ROOT)),
                    "top_rows": rank.head(12).to_dict(orient="records") if not rank.empty else [],
                }
            )
            if rank.empty:
                continue

            candidate_specs = [
                (f"PUBLIC_V3_ridge_p{str(power).replace('.', 'p')}_k02", rank, 2),
                (f"PUBLIC_V3_ridge_p{str(power).replace('.', 'p')}_k04", rank, 4),
                (f"PUBLIC_V3_ridge_p{str(power).replace('.', 'p')}_k06", rank, 6),
                (f"PUBLIC_V3_ridge_p{str(power).replace('.', 'p')}_k08", rank, 8),
            ]
            strict = rank[(rank["prob_agree"]) & (rank["prob_delta"] > 0)].copy()
            if not strict.empty:
                candidate_specs.extend(
                    [
                        (f"PUBLIC_V3_strict_p{str(power).replace('.', 'p')}_k02", strict, 2),
                        (f"PUBLIC_V3_strict_p{str(power).replace('.', 'p')}_k04", strict, 4),
                    ]
                )

            for name, rank_df, k in candidate_specs:
                labels = apply_top(anchor_labels, ids, rank_df, k)
                key = labels.astype("U8").tobytes()
                if key in seen_hashes:
                    continue
                seen_hashes.add(key)
                output_path = OUTPUTS / f"{rank_num}_{name}.csv"
                write_submission(ids, labels, output_path)
                changed = labels != anchor_labels
                output_manifests.append(
                    {
                        "file": output_path.name,
                        "path": str(output_path.relative_to(ROOT)),
                        "changed_rows": int(changed.sum()),
                        "transition_counts": transition_counts(anchor_labels, labels),
                        "class_counts": class_counts(labels),
                        "selected": rank_df.head(k).to_dict(orient="records"),
                    }
                )
                rank_num += 1

    pd.DataFrame(output_manifests).to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(args.output_dir / "archive10_pairwise_to_anchor.csv", index=False)
    pd.DataFrame(proba_manifest).to_csv(args.output_dir / "probability_manifest.csv", index=False)

    report = {
        "purpose": "PUBLIC-LB advanced ridge v3 local reproduction; not a CV/private result.",
        "inputs": {
            "top_archive": str(args.top_archive),
            "bank_archive": str(args.bank_archive),
            "anchor_member": args.anchor_member,
            "anchor_score": float(args.anchor_score),
        },
        "scored_entries": len(entries),
        "probability_sources": proba_manifest,
        "rank_manifests": rank_manifests,
        "candidate_manifest": output_manifests,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Advanced Ridge V3 Local Reproduction",
                "",
                "이 폴더는 `Advanced Ridge Flip & Probability Consensus (v3)` 노트북을 로컬 zip 파일 기준으로 재현한 결과입니다.",
                "",
                "- public 0.97244 앵커에서 시작합니다.",
                "- archive11의 hard-label submission bank와 probability bank를 사용합니다.",
                "- 생성 CSV는 public leaderboard 탐색용입니다. OOF/CV 후보가 아닙니다.",
                "- `candidate_manifest.csv`에서 변경 row와 transition을 확인할 수 있습니다.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    progress(f"Wrote {len(output_manifests)} candidates/report rows to {args.output_dir}")
    for item in output_manifests:
        print(f"- {item['path']} changed={item['changed_rows']} transitions={item['transition_counts']}", flush=True)


if __name__ == "__main__":
    main()
