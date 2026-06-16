from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import LeaveOneOut


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "bank_regularized_safety"
LABELS = np.array(["GALAXY", "QSO", "STAR"])
LABEL_TO_INT = {label: idx for idx, label in enumerate(LABELS)}
BAD_IDS = {665223, 676483, 755752}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build safer public+generalization candidates from the score-named submission bank "
            "using Ridge/Lasso/ElasticNet agreement and probability consensus margins."
        )
    )
    parser.add_argument("--prediction-dir", type=Path, default=ROOT / "external_preds")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--ridge-grid", type=int, default=40)
    parser.add_argument("--lasso-grid", type=int, default=40)
    parser.add_argument("--enet-grid", type=int, default=30)
    return parser.parse_args()


def score_from_name(path: Path) -> float | None:
    match = re.match(r"^(0\.\d{5})", path.name)
    return float(match.group(1)) if match else None


def score_from_probability_name(path: Path) -> float | None:
    match = re.search(r"(0\.\d{5})", path.name)
    return float(match.group(1)) if match else None


def read_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "class"]:
        raise ValueError(f"{path.name} is not a hard-label submission.")
    if not df["id"].equals(sample["id"]):
        raise ValueError(f"{path.name} id order differs from sample_submission.csv.")
    invalid = sorted(set(df["class"].dropna()) - set(LABELS))
    if invalid:
        raise ValueError(f"{path.name} has invalid labels: {invalid}")
    return df


def load_scored_submissions(prediction_dir: Path, sample: pd.DataFrame) -> list[tuple[Path, float]]:
    rows = []
    for path in prediction_dir.glob("*.csv"):
        score = score_from_name(path)
        if score is None:
            continue
        try:
            read_submission(path, sample)
        except ValueError:
            continue
        rows.append((path, score))
    rows.sort(key=lambda item: (item[1], item[0].name), reverse=True)
    if not rows:
        raise FileNotFoundError(f"No score-named hard-label submissions found in {prediction_dir}.")
    return rows


def build_design(
    scored_submissions: list[tuple[Path, float]],
    anchor: pd.DataFrame,
    sample: pd.DataFrame,
) -> tuple[sparse.csr_matrix, np.ndarray, list[tuple[int, str]]]:
    anchor_ids = anchor["id"].to_numpy()
    anchor_labels = anchor["class"].to_numpy()
    anchor_score = scored_submissions[0][1]
    feature_index: dict[tuple[int, str], int] = {}
    row_indices: list[int] = []
    col_indices: list[int] = []

    for row_idx, (path, _) in enumerate(scored_submissions):
        sub = read_submission(path, sample)
        labels = sub["class"].to_numpy()
        changed = np.flatnonzero(labels != anchor_labels)
        for pos in changed:
            key = (int(anchor_ids[pos]), str(labels[pos]))
            if key not in feature_index:
                feature_index[key] = len(feature_index)
            row_indices.append(row_idx)
            col_indices.append(feature_index[key])

    x = sparse.csr_matrix(
        (np.ones(len(row_indices), dtype=np.float32), (row_indices, col_indices)),
        shape=(len(scored_submissions), len(feature_index)),
        dtype=np.float32,
    )
    y = np.array([score - anchor_score for _, score in scored_submissions], dtype=np.float64)
    inverse_features = [None] * len(feature_index)
    for key, idx in feature_index.items():
        inverse_features[idx] = key
    return x, y, inverse_features


def loo_search(x: sparse.csr_matrix, y: np.ndarray, estimator_fn, alphas: list[float], label: str):
    loo = LeaveOneOut()
    best_alpha = alphas[0]
    best_rmse = float("inf")
    n = x.shape[0]
    for alpha in alphas:
        preds = np.zeros(n, dtype=np.float64)
        for train_idx, valid_idx in loo.split(np.arange(n)):
            model = estimator_fn(alpha)
            model.fit(x[train_idx], y[train_idx])
            preds[valid_idx[0]] = float(model.predict(x[valid_idx])[0])
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        if rmse < best_rmse:
            best_alpha = alpha
            best_rmse = rmse

    model = estimator_fn(best_alpha)
    model.fit(x, y)
    return best_alpha, best_rmse, model


def extract_flips(
    coef: np.ndarray,
    inverse_features: list[tuple[int, str]],
    support: np.ndarray,
    min_support: int,
) -> list[dict]:
    flips = []
    for idx, value in enumerate(coef):
        if value <= 0 or support[idx] < min_support:
            continue
        row_id, label = inverse_features[idx]
        if int(row_id) in BAD_IDS:
            continue
        flips.append(
            {
                "id": int(row_id),
                "class": str(label),
                "coef": float(value),
                "support": int(support[idx]),
            }
        )
    flips.sort(key=lambda item: item["coef"], reverse=True)
    return flips


def load_probability_consensus(prediction_dir: Path, ids: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[dict]]:
    arrays = []
    weights = []
    manifest = []

    for path in sorted(prediction_dir.glob("test_preds__*.csv")):
        df = pd.read_csv(path)
        if not set(LABELS).issubset(df.columns):
            continue
        if not np.array_equal(df["id"].to_numpy(), ids):
            raise ValueError(f"ID order mismatch: {path.name}")
        arr = df[list(LABELS)].to_numpy(np.float64)
        arr = arr / arr.sum(axis=1, keepdims=True)
        score = score_from_probability_name(path) or 0.969
        weight = max(0.0, (score - 0.968) ** 2) * 1e4
        arrays.append(arr)
        weights.append(weight)
        manifest.append({"file": path.name, "score": score, "weight": weight})

    for path in sorted(prediction_dir.glob("*test_preds__*.npy")):
        arr = np.load(path).astype(np.float64)
        if arr.shape != (len(ids), len(LABELS)):
            continue
        arr = arr / arr.sum(axis=1, keepdims=True)
        score = score_from_probability_name(path) or 0.969
        weight = max(0.0, (score - 0.968) ** 2) * 1e4
        arrays.append(arr)
        weights.append(weight)
        manifest.append({"file": path.name, "score": score, "weight": weight})

    if not arrays:
        return None, None, None, manifest

    weights_arr = np.array(weights, dtype=np.float64)
    if weights_arr.sum() == 0:
        weights_arr = np.ones_like(weights_arr)
    proba = np.average(np.stack(arrays), axis=0, weights=weights_arr)
    pred = LABELS[proba.argmax(axis=1)]
    sorted_proba = np.sort(proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    return proba, pred, margin, manifest


def apply_flips(anchor: pd.DataFrame, flips: list[dict]) -> pd.DataFrame:
    out = anchor.copy()
    id_to_label = {int(item["id"]): str(item["class"]) for item in flips}
    mask = out["id"].isin(id_to_label)
    out.loc[mask, "class"] = out.loc[mask, "id"].map(id_to_label)
    return out


def save_candidate(
    anchor: pd.DataFrame,
    anchor_labels: np.ndarray,
    output_dir: Path,
    name: str,
    flips: list[dict],
    metadata: dict,
) -> Path:
    sub = apply_flips(anchor, flips)
    path = output_dir / f"{name}.csv"
    sub.to_csv(path, index=False)
    meta = {
        **metadata,
        "candidate": name,
        "num_flips": len(flips),
        "changes_vs_anchor": int((sub["class"].to_numpy() != anchor_labels).sum()),
        "sum_coef": float(sum(item["coef"] for item in flips)),
        "counts": sub["class"].value_counts().sort_index().to_dict(),
        "flips": flips,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def proba_filter(
    flips: list[dict],
    id_to_pos: dict[int, int],
    proba: np.ndarray,
    proba_pred: np.ndarray,
    proba_margin: np.ndarray,
    min_margin: float,
) -> list[dict]:
    out = []
    for item in flips:
        pos = id_to_pos[int(item["id"])]
        label = str(item["class"])
        if proba_pred[pos] != label or proba_margin[pos] < min_margin:
            continue
        enriched = dict(item)
        enriched["proba_margin"] = float(proba_margin[pos])
        enriched["label_prob"] = float(proba[pos, LABEL_TO_INT[label]])
        out.append(enriched)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    scored = load_scored_submissions(args.prediction_dir, sample)
    anchor_path, anchor_score = scored[0]
    anchor = read_submission(anchor_path, sample)
    anchor_labels = anchor["class"].to_numpy()
    anchor.to_csv(args.output_dir / "anchor.csv", index=False)

    x, y, inverse_features = build_design(scored, anchor, sample)
    support = np.asarray((x > 0).sum(axis=0)).ravel()

    ridge_alpha, ridge_rmse, ridge_model = loo_search(
        x,
        y,
        lambda alpha: Ridge(alpha=alpha, fit_intercept=True, random_state=42),
        np.logspace(-3, 2, args.ridge_grid).tolist(),
        "Ridge",
    )
    lasso_alpha, lasso_rmse, lasso_model = loo_search(
        x,
        y,
        lambda alpha: Lasso(alpha=alpha, max_iter=50_000, random_state=42, fit_intercept=True),
        np.logspace(-6, -1, args.lasso_grid).tolist(),
        "Lasso",
    )
    enet_alpha, enet_rmse, enet_model = loo_search(
        x,
        y,
        lambda alpha: ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=50_000, random_state=42, fit_intercept=True),
        np.logspace(-6, -1, args.enet_grid).tolist(),
        "ElasticNet",
    )

    ridge_flips = extract_flips(np.asarray(ridge_model.coef_, dtype=np.float64), inverse_features, support, args.min_support)
    lasso_flips = extract_flips(np.asarray(lasso_model.coef_, dtype=np.float64), inverse_features, support, args.min_support)
    enet_flips = extract_flips(np.asarray(enet_model.coef_, dtype=np.float64), inverse_features, support, args.min_support)

    lasso_keys = {(item["id"], item["class"]) for item in lasso_flips}
    enet_keys = {(item["id"], item["class"]) for item in enet_flips}
    agreed_rl = [item for item in ridge_flips if (item["id"], item["class"]) in lasso_keys]
    agreed_all3 = [item for item in agreed_rl if (item["id"], item["class"]) in enet_keys]

    proba, proba_pred, proba_margin, proba_manifest = load_probability_consensus(args.prediction_dir, anchor["id"].to_numpy())
    id_to_pos = {int(row_id): pos for pos, row_id in enumerate(anchor["id"].to_numpy())}

    base_meta = {
        "prediction_dir": str(args.prediction_dir),
        "anchor_file": anchor_path.name,
        "anchor_score": float(anchor_score),
        "scored_submissions": len(scored),
        "design_shape": list(x.shape),
        "min_support": args.min_support,
        "bad_ids": sorted(BAD_IDS),
        "ridge": {"alpha": float(ridge_alpha), "loo_rmse": float(ridge_rmse), "positive_flips": len(ridge_flips)},
        "lasso": {
            "alpha": float(lasso_alpha),
            "loo_rmse": float(lasso_rmse),
            "positive_flips": len(lasso_flips),
            "nonzero": int(np.count_nonzero(lasso_model.coef_)),
        },
        "elasticnet": {
            "alpha": float(enet_alpha),
            "loo_rmse": float(enet_rmse),
            "positive_flips": len(enet_flips),
            "nonzero": int(np.count_nonzero(enet_model.coef_)),
        },
        "agreed_rl_flips": len(agreed_rl),
        "agreed_all3_flips": len(agreed_all3),
        "probability_files": proba_manifest,
    }

    candidates: dict[str, list[dict]] = {}
    for k in [20, 35, 75, 100, 150, 180, 220, 260]:
        candidates[f"ridge_top{k}"] = ridge_flips[: min(k, len(ridge_flips))]
    for k in [20, 50, 100]:
        candidates[f"lasso_top{k}"] = lasso_flips[: min(k, len(lasso_flips))]
    candidates["lasso_topall"] = lasso_flips
    for k in [20, 50]:
        candidates[f"agreed_rl_top{k}"] = agreed_rl[: min(k, len(agreed_rl))]
    candidates["agreed_rl_topall"] = agreed_rl
    candidates["agreed_all3"] = agreed_all3

    if proba is not None and proba_pred is not None and proba_margin is not None:
        for margin in [0.10, 0.15]:
            suffix = int(margin * 100)
            candidates[f"agreed_all3_proba{suffix:02d}"] = proba_filter(
                agreed_all3, id_to_pos, proba, proba_pred, proba_margin, margin
            )
            candidates[f"agreed_rl_proba{suffix:02d}"] = proba_filter(
                agreed_rl, id_to_pos, proba, proba_pred, proba_margin, margin
            )

    generated = []
    rows = []
    for name, flips in candidates.items():
        path = save_candidate(anchor, anchor_labels, args.output_dir, name, flips, base_meta)
        generated.append(path)
        rows.append(
            {
                "candidate": name,
                "file": path.name,
                "num_flips": len(flips),
                "changes_vs_anchor": len({item["id"] for item in flips}),
                "sum_coef": float(sum(item["coef"] for item in flips)),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["num_flips", "sum_coef"], ascending=[True, False])
    summary.to_csv(args.output_dir / "candidate_summary.csv", index=False)

    preferred_order = [
        "agreed_all3_proba15",
        "agreed_all3_proba10",
        "agreed_all3",
        "agreed_rl_proba15",
        "agreed_rl_proba10",
        "agreed_rl_top20",
        "lasso_topall",
    ]
    recommended = next((name for name in preferred_order if name in candidates and len(candidates[name]) > 0), None)
    report = {
        **base_meta,
        "generated_files": [path.name for path in generated],
        "candidate_summary": rows,
        "recommended_safe_public_generalization": f"{recommended}.csv" if recommended else None,
        "interpretation": (
            "These candidates are more conservative than the ridge-only public search because "
            "Lasso and ElasticNet remove unstable row flips, and the proba variants require an "
            "independent probability consensus margin."
        ),
    }
    (args.output_dir / "regularized_bank_safety_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "ACTIVE_SAFE_BANK_SUBMISSIONS.txt").write_text(
        "\n".join(
            [
                "# Safer bank-derived candidates",
                f"# Recommended: {report['recommended_safe_public_generalization']}",
                *[path.name for path in generated],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
