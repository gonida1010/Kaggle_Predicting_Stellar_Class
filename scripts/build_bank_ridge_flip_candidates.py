from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneOut


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "bank_ridge_flip_v5"
LABELS = np.array(["GALAXY", "QSO", "STAR"])
LABEL_TO_INT = {label: idx for idx, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build high-public-score flip candidates from score-named public submission-bank CSVs. "
            "This adapts the public Ridge flip-search idea into a reproducible local script."
        )
    )
    parser.add_argument("--prediction-dir", type=Path, default=ROOT / "external_preds")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--anchor-top-n", type=int, default=3)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--bayesian-prior", type=float, default=5.0)
    parser.add_argument("--coef-threshold-frac", type=float, default=0.025)
    parser.add_argument("--loo-influence-sigma", type=float, default=3.0)
    parser.add_argument("--tail-min-margin", type=float, default=0.12)
    parser.add_argument("--tail-max-entropy", type=float, default=0.85)
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    )
    return parser.parse_args()


def score_from_name(path: Path) -> float | None:
    match = re.match(r"^(0\.\d{5})", path.name)
    return float(match.group(1)) if match else None


def load_scored_submissions(prediction_dir: Path) -> list[tuple[Path, float]]:
    rows = []
    for path in prediction_dir.glob("*.csv"):
        score = score_from_name(path)
        if score is not None:
            rows.append((path, score))
    rows.sort(key=lambda item: (item[1], item[0].name), reverse=True)
    if not rows:
        raise FileNotFoundError(
            f"No score-named submission CSVs found in {prediction_dir}. "
            "Put files such as 0.97183.csv, 0.97182.csv, ... under external_preds/."
        )
    return rows


def read_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    sub = pd.read_csv(path)[["id", "class"]].copy()
    if not sub["id"].equals(sample["id"]):
        raise ValueError(f"ID order mismatch: {path.name}")
    invalid = sorted(set(sub["class"].dropna()) - set(LABELS))
    if invalid:
        raise ValueError(f"{path.name} has invalid labels: {invalid}")
    return sub


def majority_vote(labels: np.ndarray) -> str:
    counts: dict[str, int] = {}
    for value in labels:
        counts[str(value)] = counts.get(str(value), 0) + 1
    max_count = max(counts.values())
    for value in labels:
        value = str(value)
        if counts[value] == max_count:
            return value
    raise RuntimeError("unreachable majority vote state")


def build_consensus_anchor(
    scored_submissions: list[tuple[Path, float]],
    sample: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, Path, float, int]:
    anchor_path, anchor_score = scored_submissions[0]
    base_anchor = read_submission(anchor_path, sample)
    n = min(top_n, len(scored_submissions))
    label_matrix = np.vstack(
        [read_submission(path, sample)["class"].to_numpy() for path, _ in scored_submissions[:n]]
    )
    consensus = np.array([majority_vote(label_matrix[:, idx]) for idx in range(label_matrix.shape[1])])
    anchor = base_anchor.copy()
    anchor["class"] = consensus
    diff_from_best = int((consensus != label_matrix[0]).sum())
    return anchor, anchor_path, anchor_score, diff_from_best


def build_design(
    rows: list[tuple[Path, float]],
    anchor: pd.DataFrame,
    sample: pd.DataFrame,
    anchor_score: float,
) -> tuple[sparse.csr_matrix, np.ndarray, list[tuple[int, str]]]:
    anchor_ids = anchor["id"].to_numpy()
    anchor_labels = anchor["class"].to_numpy()
    feature_index: dict[tuple[int, str], int] = {}
    row_indices: list[int] = []
    col_indices: list[int] = []

    for row_idx, (path, _) in enumerate(rows):
        sub = read_submission(path, sample)
        labels = sub["class"].to_numpy()
        changed = np.flatnonzero(labels != anchor_labels)
        for pos in changed:
            key = (int(anchor_ids[pos]), str(labels[pos]))
            if key not in feature_index:
                feature_index[key] = len(feature_index)
            row_indices.append(row_idx)
            col_indices.append(feature_index[key])

    data = np.ones(len(row_indices), dtype=np.float32)
    x = sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(rows), len(feature_index)),
        dtype=np.float32,
    )
    y = np.array([score - anchor_score for _, score in rows], dtype=np.float64)
    inverse_features = [None] * len(feature_index)
    for key, idx in feature_index.items():
        inverse_features[idx] = key
    return x, y, inverse_features


def detect_outliers_stratified(
    x: sparse.csr_matrix,
    y: np.ndarray,
    scores: np.ndarray,
    sigma_thresh: float,
    alpha: float = 0.1,
    top_k_features: int = 50,
    n_quintiles: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    if n < 8 or x.shape[1] == 0:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=float)

    model_full = Ridge(alpha=alpha, fit_intercept=True)
    model_full.fit(x, y)
    coef_full = model_full.coef_.copy()
    top_idx = np.argsort(np.abs(coef_full))[::-1][: min(top_k_features, len(coef_full))]

    loo_diffs = np.zeros(n, dtype=np.float64)
    for idx in range(n):
        keep = np.r_[0:idx, idx + 1 : n]
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(x[keep], y[keep])
        loo_diffs[idx] = np.abs(model.coef_[top_idx] - coef_full[top_idx]).mean()

    qlabels = pd.qcut(scores, q=min(n_quintiles, n), labels=False, duplicates="drop")
    z_scores = np.zeros(n, dtype=np.float64)
    for q in np.unique(qlabels):
        mask = np.asarray(qlabels == q)
        if mask.sum() < 2:
            continue
        vals = loo_diffs[mask]
        z_scores[mask] = (vals - vals.mean()) / (vals.std() + 1e-12)
    return z_scores > sigma_thresh, z_scores


def fit_multi_alpha_ridge(x: sparse.csr_matrix, y: np.ndarray, alphas: list[float]) -> np.ndarray:
    coefs = []
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
        model.fit(x, y)
        coefs.append(model.coef_)
    return np.mean(coefs, axis=0)


def choose_alpha_cv(x: sparse.csr_matrix, y: np.ndarray) -> tuple[float, float]:
    alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    n = x.shape[0]
    if n <= 30:
        splits = list(LeaveOneOut().split(np.arange(n)))
    else:
        splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(n)))

    best_alpha = alphas[0]
    best_rmse = float("inf")
    for alpha in alphas:
        preds = []
        actual = []
        for train_idx, valid_idx in splits:
            model = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
            model.fit(x[train_idx], y[train_idx])
            preds.extend(model.predict(x[valid_idx]).tolist())
            actual.extend(y[valid_idx].tolist())
        rmse = float(np.sqrt(np.mean((np.array(preds) - np.array(actual)) ** 2)))
        if rmse < best_rmse:
            best_alpha = alpha
            best_rmse = rmse
    return best_alpha, best_rmse


def build_flip_candidates(
    x_clean: sparse.csr_matrix,
    coef: np.ndarray,
    inverse_features: list[tuple[int, str]],
    min_support: int,
    bayesian_prior: float,
    threshold_frac: float,
) -> tuple[list[dict], dict]:
    support = np.asarray((x_clean > 0).sum(axis=0)).ravel()
    n_clean = x_clean.shape[0]
    bayesian_support = support / (support + bayesian_prior)
    support_scale = np.log1p(bayesian_support * n_clean)

    pos_mask = coef > 0
    if pos_mask.sum() > 2:
        sorted_idx = np.argsort(coef[pos_mask])[::-1]
        iso_input = coef[pos_mask][sorted_idx]
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso_fit = iso.fit_transform(np.arange(len(iso_input), dtype=float), iso_input)
        iso_coef = np.zeros_like(coef)
        pos_indices = np.where(pos_mask)[0]
        iso_coef[pos_indices[sorted_idx]] = iso_fit
    else:
        iso_coef = coef.copy()

    combined_coef = (coef * support_scale * 0.7) + (iso_coef * support_scale * 0.3)
    raw = []
    for idx, value in enumerate(combined_coef):
        if value <= 0 or support[idx] < min_support:
            continue
        row_id, label = inverse_features[idx]
        raw.append(
            {
                "id": int(row_id),
                "class": str(label),
                "raw_coef": float(coef[idx]),
                "rescaled_coef": float(value),
                "iso_coef": float(iso_coef[idx]),
                "support": int(support[idx]),
                "bayes_support": float(bayesian_support[idx]),
            }
        )

    id_to_candidates: dict[int, list[dict]] = defaultdict(list)
    for item in raw:
        id_to_candidates[item["id"]].append(item)

    resolved = []
    recovered_conflicts = 0
    conflicted_ids = 0
    for _, group in id_to_candidates.items():
        if len({item["class"] for item in group}) <= 1:
            resolved.extend(group)
        else:
            conflicted_ids += 1
            resolved.append(max(group, key=lambda item: item["rescaled_coef"]))
            recovered_conflicts += 1

    class_groups: dict[str, list[dict]] = defaultdict(list)
    for item in resolved:
        class_groups[item["class"]].append(item)

    final = []
    class_thresholds = {}
    for cls, group in class_groups.items():
        group = sorted(group, key=lambda item: item["rescaled_coef"], reverse=True)
        threshold = group[0]["rescaled_coef"] * threshold_frac
        class_thresholds[cls] = threshold
        final.extend([item for item in group if item["rescaled_coef"] >= threshold])
    final.sort(key=lambda item: item["rescaled_coef"], reverse=True)
    diagnostics = {
        "positive_features": int(pos_mask.sum()),
        "raw_flip_candidates": len(raw),
        "conflicted_ids": conflicted_ids,
        "recovered_conflicts": recovered_conflicts,
        "class_thresholds": class_thresholds,
        "usable_flips": len(final),
    }
    return final, diagnostics


def apply_flips(anchor: pd.DataFrame, flips: list[dict]) -> pd.DataFrame:
    out = anchor.copy()
    id_to_label = {int(item["id"]): str(item["class"]) for item in flips}
    mask = out["id"].isin(id_to_label)
    out.loc[mask, "class"] = out.loc[mask, "id"].map(id_to_label)
    return out


def save_candidate(
    anchor: pd.DataFrame,
    anchor_labels: np.ndarray,
    flips: list[dict],
    output_dir: Path,
    name: str,
    metadata: dict,
) -> Path:
    sub = apply_flips(anchor, flips)
    path = output_dir / name
    sub.to_csv(path, index=False)
    meta = {
        **metadata,
        "num_flips": len(flips),
        "changes_vs_anchor": int((sub["class"].to_numpy() != anchor_labels).sum()),
        "sum_rescaled_coef": float(sum(item["rescaled_coef"] for item in flips)),
        "counts": sub["class"].value_counts().to_dict(),
        "flips": flips,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def greedy_flip_search(
    candidates: list[dict],
    min_marginal_coef: float = 5e-7,
    gain_ratio_threshold: float = 0.015,
    warmup: int = 20,
) -> list[dict]:
    selected = []
    values = []
    for item in candidates:
        marginal = item["rescaled_coef"]
        if marginal < min_marginal_coef:
            break
        selected.append(item)
        values.append(marginal)
        if len(values) >= warmup:
            if marginal / (float(np.mean(values)) + 1e-12) < gain_ratio_threshold:
                break
    return selected


def load_probability_consensus(
    prediction_dir: Path,
    ids: np.ndarray,
    labels: np.ndarray = LABELS,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    arrays = []
    weights = []
    loaded = []
    for path in sorted(prediction_dir.glob("test_preds__*.csv")):
        df = pd.read_csv(path)
        if not set(labels).issubset(df.columns):
            continue
        if not np.array_equal(df["id"].to_numpy(), ids):
            raise ValueError(f"ID order mismatch: {path.name}")
        arr = df[list(labels)].to_numpy(np.float64)
        arr = arr / arr.sum(axis=1, keepdims=True)
        score = score_from_name(Path(path.name.replace("test_preds__", ""))) or 0.969
        arrays.append(arr)
        weights.append(max(0.1, score - 0.968))
        loaded.append(path.name)

    for path in sorted(prediction_dir.glob("*test_preds__*.npy")):
        arr = np.load(path).astype(np.float64)
        if arr.shape != (len(ids), len(labels)):
            continue
        arr = arr / arr.sum(axis=1, keepdims=True)
        arrays.append(arr)
        weights.append(0.1)
        loaded.append(path.name)

    if not arrays:
        return None, None, None, None, loaded

    proba = np.average(np.stack(arrays), axis=0, weights=np.array(weights))
    pred = labels[proba.argmax(axis=1)]
    sorted_proba = np.sort(proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    entropy = -(proba * np.log(proba + 1e-12)).sum(axis=1) / np.log(len(labels))
    return proba, pred, margin, entropy, loaded


def save_tail_candidate(
    anchor: pd.DataFrame,
    anchor_labels: np.ndarray,
    output_dir: Path,
    base_k: int,
    tail_start: int,
    tail_stop: int,
    take: int,
    candidates: list[dict],
    proba: np.ndarray,
    proba_pred: np.ndarray,
    proba_margin: np.ndarray,
    proba_entropy: np.ndarray,
    metadata: dict,
    min_margin: float,
    max_entropy: float,
) -> Path:
    id_to_pos = {int(row_id): pos for pos, row_id in enumerate(anchor["id"].to_numpy())}
    base = candidates[:base_k]
    tail_pool = candidates[tail_start:tail_stop]
    supported = []
    for item in tail_pool:
        pos = id_to_pos[int(item["id"])]
        label = str(item["class"])
        if proba_pred[pos] != label:
            continue
        if proba_margin[pos] < min_margin or proba_entropy[pos] > max_entropy:
            continue
        enriched = dict(item)
        label_prob = float(proba[pos, LABEL_TO_INT[label]])
        enriched["label_prob"] = label_prob
        enriched["proba_margin"] = float(proba_margin[pos])
        enriched["entropy"] = float(proba_entropy[pos])
        enriched["tail_score"] = float(
            np.log1p(label_prob) * proba_margin[pos] * (1.0 - proba_entropy[pos]) * item["rescaled_coef"]
        )
        supported.append(enriched)
    supported.sort(key=lambda item: item["tail_score"], reverse=True)
    return save_candidate(
        anchor,
        anchor_labels,
        base + supported[:take],
        output_dir,
        f"v5_ridge_top{base_k}_tail{take}.csv",
        {**metadata, "tail_pool_size": len(tail_pool), "tail_gate_passed": len(supported)},
    )


def vote_candidate_files(anchor_labels: np.ndarray, files: list[Path], output_path: Path) -> Path | None:
    frames = [pd.read_csv(path) for path in files if path.exists()]
    if len(frames) < 2:
        return None
    ids = frames[0]["id"].to_numpy()
    matrix = np.vstack([frame["class"].to_numpy() for frame in frames])
    voted = np.array([majority_vote(matrix[:, idx]) for idx in range(matrix.shape[1])])
    out = frames[0].copy()
    out["class"] = voted
    out.to_csv(output_path, index=False)
    meta = {
        "candidate_files": [path.name for path in files],
        "changes_vs_anchor": int((voted != anchor_labels).sum()),
        "counts": out["class"].value_counts().to_dict(),
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(DATA / "sample_submission.csv")

    scored_submissions = load_scored_submissions(args.prediction_dir)
    scores = np.array([score for _, score in scored_submissions], dtype=np.float64)
    anchor, anchor_path, anchor_score, diff_from_best = build_consensus_anchor(
        scored_submissions,
        sample,
        args.anchor_top_n,
    )
    anchor_labels = anchor["class"].to_numpy()
    anchor.to_csv(args.output_dir / "consensus_anchor.csv", index=False)

    x, y, inverse_features = build_design(scored_submissions, anchor, sample, anchor_score)
    outlier_mask, outlier_z = detect_outliers_stratified(
        x,
        y,
        scores,
        sigma_thresh=args.loo_influence_sigma,
    )
    clean_idx = np.where(~outlier_mask)[0]
    x_clean = x[clean_idx]
    y_clean = y[clean_idx]
    coef = fit_multi_alpha_ridge(x_clean, y_clean, args.ridge_alphas)
    best_alpha, best_rmse = choose_alpha_cv(x_clean, y_clean)
    candidates, candidate_diagnostics = build_flip_candidates(
        x_clean,
        coef,
        inverse_features,
        min_support=args.min_support,
        bayesian_prior=args.bayesian_prior,
        threshold_frac=args.coef_threshold_frac,
    )
    pd.DataFrame(candidates).to_csv(args.output_dir / "ridge_flip_candidates.csv", index=False)

    base_meta = {
        "prediction_dir": str(args.prediction_dir),
        "anchor_file": anchor_path.name,
        "anchor_score": anchor_score,
        "anchor_top_n": args.anchor_top_n,
        "consensus_rows_diff_from_best": diff_from_best,
        "clean_submissions": int(len(clean_idx)),
        "outliers_removed": int(outlier_mask.sum()),
        "best_alpha_cv": best_alpha,
        "best_alpha_rmse": best_rmse,
        "ridge_alphas": args.ridge_alphas,
        "min_support": args.min_support,
        "bayesian_prior": args.bayesian_prior,
        **candidate_diagnostics,
    }

    generated: dict[str, str] = {}
    for k in [20, 35, 50, 75, 100, 130, 150, 180, 220, 260, 300, 350, 400, 450]:
        if candidates:
            path = save_candidate(
                anchor,
                anchor_labels,
                candidates[: min(k, len(candidates))],
                args.output_dir,
                f"v5_ridge_top{k}.csv",
                base_meta,
            )
            generated[path.name] = str(path)

    greedy = greedy_flip_search(candidates)
    greedy_path = save_candidate(anchor, anchor_labels, greedy, args.output_dir, "v5_ridge_greedy.csv", base_meta)
    generated[greedy_path.name] = str(greedy_path)

    proba, proba_pred, proba_margin, proba_entropy, proba_files = load_probability_consensus(
        args.prediction_dir,
        anchor["id"].to_numpy(),
    )
    tail_paths: dict[str, Path] = {}
    if proba is not None:
        for base_k, takes in [
            (100, [10, 20, 30, 40]),
            (130, [10, 20, 30, 40]),
            (150, [10, 20, 30]),
            (180, [10, 20, 30]),
            (220, [10, 20, 30]),
            (260, [10, 20]),
            (300, [10, 20]),
        ]:
            tail_stop = min(base_k + 300, len(candidates))
            for take in takes:
                path = save_tail_candidate(
                    anchor,
                    anchor_labels,
                    args.output_dir,
                    base_k,
                    base_k,
                    tail_stop,
                    take,
                    candidates,
                    proba,
                    proba_pred,
                    proba_margin,
                    proba_entropy,
                    base_meta,
                    args.tail_min_margin,
                    args.tail_max_entropy,
                )
                tail_paths[path.name] = path
                generated[path.name] = str(path)

    if {"v5_ridge_top130_tail20.csv", "v5_ridge_top150_tail20.csv"}.issubset(tail_paths):
        vote_inputs = [greedy_path, tail_paths["v5_ridge_top130_tail20.csv"], tail_paths["v5_ridge_top150_tail20.csv"]]
    else:
        vote_inputs = [
            greedy_path,
            args.output_dir / "v5_ridge_top130.csv",
            args.output_dir / "v5_ridge_top150.csv",
        ]
    voted_path = vote_candidate_files(anchor_labels, vote_inputs, args.output_dir / "v5_voted_ensemble.csv")
    if voted_path is not None:
        generated[voted_path.name] = str(voted_path)

    manifest = pd.DataFrame(
        [
            {
                "file": path.name,
                "score_from_filename": score,
                "outlier": bool(outlier_mask[idx]),
                "outlier_z": float(outlier_z[idx]),
            }
            for idx, (path, score) in enumerate(scored_submissions)
        ]
    )
    manifest.to_csv(args.output_dir / "submission_bank_manifest.csv", index=False)

    report = {
        **base_meta,
        "scored_submissions": len(scored_submissions),
        "design_shape": list(x.shape),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "probability_files": proba_files,
        "voted_ensemble": voted_path.name if voted_path is not None else None,
        "generated_files": sorted(generated),
        "recommended_public_generalization": voted_path.name if voted_path is not None else "v5_ridge_top130.csv",
    }
    (args.output_dir / "bank_ridge_flip_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "ACTIVE_RIDGE_FLIP_SUBMISSIONS.txt").write_text(
        "\n".join(
            [
                "# Bank ridge flip submissions",
                f"# Recommended: {report['recommended_public_generalization']}",
                *sorted(generated),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
