from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import make_xy  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "boundary_specialist"
PURE_ENSEMBLE_DIR = ARTIFACTS / "pure_model_ensemble"
SEED = 20260613
N_SPLITS = 5
SOURCE_MODELS = ["lgbm", "catboost"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a leak-safe GALAXY/STAR boundary specialist and test OOF override rules."
    )
    parser.add_argument(
        "--fold-limit",
        type=int,
        default=N_SPLITS,
        help="Number of CV folds to run. Use 1 for screening, 5 for final artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory where specialist artifacts are written.",
    )
    return parser.parse_args()


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for class_idx in range(n_classes):
        mask = y_true == class_idx
        if mask.any():
            recalls.append(float((y_pred[mask] == class_idx).mean()))
    return float(np.mean(recalls))


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, float]:
    recalls = {}
    for idx, cls in enumerate(classes):
        mask = y_true == idx
        recalls[cls] = float((y_pred[mask] == idx).mean()) if mask.any() else float("nan")
    return recalls


def load_base_proba(prefix: str) -> tuple[np.ndarray, np.ndarray]:
    oof_path = ARTIFACTS / f"{prefix}_oof_proba.npy"
    test_path = ARTIFACTS / f"{prefix}_test_proba.npy"
    if not oof_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing base proba artifacts for {prefix}.")
    return np.load(oof_path), np.load(test_path)


def add_source_probability_features(
    features: pd.DataFrame,
    proba_by_model: dict[str, np.ndarray],
    classes: list[str],
) -> pd.DataFrame:
    out = features.reset_index(drop=True).copy()
    for model, proba in proba_by_model.items():
        sorted_probs = np.sort(proba, axis=1)
        for idx, cls in enumerate(classes):
            out[f"{model}_p_{cls}"] = proba[:, idx]
        out[f"{model}_pred_idx"] = proba.argmax(axis=1).astype("int16")
        out[f"{model}_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    if len(proba_by_model) >= 2:
        preds = [proba.argmax(axis=1) for proba in proba_by_model.values()]
        out["base_models_agree"] = np.all(np.stack(preds, axis=1) == preds[0].reshape(-1, 1), axis=1).astype("int8")
    return out


def specialist_params() -> dict:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.035,
        "num_leaves": 72,
        "max_depth": -1,
        "min_child_samples": 90,
        "subsample": 0.88,
        "subsample_freq": 1,
        "colsample_bytree": 0.86,
        "reg_alpha": 0.12,
        "reg_lambda": 2.8,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_estimators": 3200,
        "n_jobs": -1,
        "verbosity": -1,
    }


def build_masks(features: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "all_gs_predictions": np.ones(len(features), dtype=bool),
        "low_redshift": features["redshift"].between(-0.02, 0.14).to_numpy(),
        "diagnostic_hard_bin": (
            features["redshift"].between(-0.02, 0.14)
            & features["g-i"].between(0.75, 1.35)
            & features["mag_range"].between(1.4, 3.1)
        ).to_numpy(),
        "redshift_color_product": (
            features["redshift_x_g-i"].between(0.012, 0.245)
            | features["redshift_x_u-r"].between(0.055, 0.410)
        ).to_numpy(),
    }


def apply_override(
    base_pred: np.ndarray,
    star_proba: np.ndarray,
    mask: np.ndarray,
    galaxy_idx: int,
    star_idx: int,
    to_star_threshold: float,
    to_galaxy_threshold: float,
) -> np.ndarray:
    pred = base_pred.copy()
    gs_pred = (base_pred == galaxy_idx) | (base_pred == star_idx)
    active = gs_pred & mask
    pred[active & (star_proba >= to_star_threshold)] = star_idx
    pred[active & (star_proba <= to_galaxy_threshold)] = galaxy_idx
    return pred


def search_override(
    y: np.ndarray,
    base_pred: np.ndarray,
    star_proba: np.ndarray,
    masks: dict[str, np.ndarray],
    classes: list[str],
    covered: np.ndarray,
) -> tuple[dict, list[dict]]:
    galaxy_idx = classes.index("GALAXY")
    star_idx = classes.index("STAR")
    n_classes = len(classes)
    base_score = balanced_accuracy(y[covered], base_pred[covered], n_classes)
    records = []
    best = {
        "score": base_score,
        "delta": 0.0,
        "mask": "none",
        "to_star_threshold": None,
        "to_galaxy_threshold": None,
        "changed_rows": 0,
        "class_recalls": class_recalls(y[covered], base_pred[covered], classes),
    }
    to_star_values = np.linspace(0.54, 0.90, 19)
    to_galaxy_values = np.linspace(0.10, 0.46, 19)

    for mask_name, mask in masks.items():
        for to_star in to_star_values:
            for to_galaxy in to_galaxy_values:
                if to_galaxy >= to_star:
                    continue
                pred = apply_override(
                    base_pred,
                    star_proba,
                    mask,
                    galaxy_idx,
                    star_idx,
                    float(to_star),
                    float(to_galaxy),
                )
                score = balanced_accuracy(y[covered], pred[covered], n_classes)
                changed = int((pred[covered] != base_pred[covered]).sum())
                record = {
                    "score": score,
                    "delta": score - base_score,
                    "mask": mask_name,
                    "to_star_threshold": float(to_star),
                    "to_galaxy_threshold": float(to_galaxy),
                    "changed_rows": changed,
                    "class_recalls": class_recalls(y[covered], pred[covered], classes),
                }
                records.append(record)
                if score > best["score"]:
                    best = record

    records.sort(key=lambda row: row["score"], reverse=True)
    return best, records[:30]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    x_base, y_raw, x_test_base, feature_names = make_xy(train, test)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    classes = encoder.classes_.tolist()
    if "GALAXY" not in classes or "STAR" not in classes:
        raise ValueError(f"Required classes missing: {classes}")
    galaxy_idx = classes.index("GALAXY")
    star_idx = classes.index("STAR")

    pure_oof = np.load(PURE_ENSEMBLE_DIR / "pure_model_ensemble_oof_proba.npy")
    pure_test = np.load(PURE_ENSEMBLE_DIR / "pure_model_ensemble_test_proba.npy")
    base_oof_pred = pure_oof.argmax(axis=1)
    base_test_pred = pure_test.argmax(axis=1)

    oof_by_model = {}
    test_by_model = {}
    for model in SOURCE_MODELS:
        oof_by_model[model], test_by_model[model] = load_base_proba(model)

    x = add_source_probability_features(x_base, oof_by_model, classes)
    x_test = add_source_probability_features(x_test_base, test_by_model, classes)
    binary_train_mask = (y == galaxy_idx) | (y == star_idx)
    binary_y = (y == star_idx).astype("int8")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(cv.split(x, y))[: args.fold_limit]
    if not splits:
        raise ValueError("--fold-limit must be at least 1.")

    star_oof = np.full(len(x), np.nan, dtype=np.float32)
    star_test = np.zeros(len(x_test), dtype=np.float32)
    fold_scores = []
    best_iterations = []
    params = specialist_params()

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        tr_binary = tr_idx[binary_train_mask[tr_idx]]
        model = lgb.LGBMClassifier(**params)
        model.fit(
            x.iloc[tr_binary],
            binary_y[tr_binary],
            eval_set=[(x.iloc[va_idx[binary_train_mask[va_idx]]], binary_y[va_idx[binary_train_mask[va_idx]]])],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(140, verbose=False),
                lgb.log_evaluation(250),
            ],
        )
        valid_star = model.predict_proba(x.iloc[va_idx])[:, 1]
        star_oof[va_idx] = valid_star
        binary_valid = va_idx[binary_train_mask[va_idx]]
        binary_score = balanced_accuracy_score(binary_y[binary_valid], (star_oof[binary_valid] >= 0.5).astype("int8"))
        fold_scores.append(float(binary_score))
        best_iterations.append(int(model.best_iteration_ or params["n_estimators"]))
        star_test += model.predict_proba(x_test)[:, 1] / len(splits)
        print(
            f"fold {fold}: binary_balanced_accuracy={binary_score:.6f}, "
            f"best_iteration={best_iterations[-1]}"
        )

    covered = ~np.isnan(star_oof)
    train_masks = build_masks(x_base)
    test_masks = build_masks(x_test_base)
    best, search_top = search_override(
        y,
        base_oof_pred,
        star_oof,
        train_masks,
        classes,
        covered,
    )

    base_score = balanced_accuracy(y[covered], base_oof_pred[covered], len(classes))
    report = {
        "purpose": "Pure-model GALAXY/STAR boundary specialist. No public submission CSV is used.",
        "fold_limit": len(splits),
        "covered_rows": int(covered.sum()),
        "base_score_on_covered_rows": base_score,
        "best_override": best,
        "search_top": search_top,
        "binary_fold_scores": fold_scores,
        "best_iterations": best_iterations,
        "params": params,
        "feature_count": len(x.columns),
        "base_feature_count": len(feature_names),
    }

    if covered.all() and best["mask"] != "none":
        final_test_pred = apply_override(
            base_test_pred,
            star_test,
            test_masks[best["mask"]],
            galaxy_idx,
            star_idx,
            best["to_star_threshold"],
            best["to_galaxy_threshold"],
        )
        submission = sample.copy()
        submission["class"] = encoder.inverse_transform(final_test_pred)
        submission.to_csv(args.output_dir / "boundary_specialist_submission.csv", index=False)
        np.save(args.output_dir / "boundary_specialist_star_oof.npy", star_oof)
        np.save(args.output_dir / "boundary_specialist_star_test.npy", star_test.astype(np.float32))
        report["submission_path"] = str((args.output_dir / "boundary_specialist_submission.csv").relative_to(ROOT))
        report["submission_class_share"] = submission["class"].value_counts(normalize=True).sort_index().to_dict()

    (args.output_dir / "boundary_specialist_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
