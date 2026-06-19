from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import categorical_columns_for_feature_set, make_xy  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "ovr_catboost"
SEED = 20260618
N_SPLITS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train fold-safe one-vs-rest CatBoost specialists and convert their binary probabilities "
            "into a multiclass submission. No public submission CSV is used."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--feature-set", choices=["base", "advanced", "realmlp"], default="advanced")
    parser.add_argument("--fold-limit", type=int, default=N_SPLITS)
    parser.add_argument("--iterations", type=int, default=2600)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    parser.add_argument("--random-strength", type=float, default=0.45)
    parser.add_argument("--bagging-temperature", type=float, default=0.45)
    parser.add_argument("--early-stopping-rounds", type=int, default=180)
    parser.add_argument("--log-period", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_probs(proba: np.ndarray) -> np.ndarray:
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return proba / row_sum


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for class_idx in range(n_classes):
        mask = y_true == class_idx
        if mask.any():
            recalls.append(float((y_pred[mask] == class_idx).mean()))
    return float(np.mean(recalls))


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, float]:
    out = {}
    for idx, label in enumerate(classes):
        mask = y_true == idx
        out[label] = float((y_pred[mask] == idx).mean()) if mask.any() else float("nan")
    return out


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    progress("Loading train/test data")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    progress(f"Building {args.feature_set} feature matrix")
    x, y_raw, x_test, features = make_xy(train, test, feature_set=args.feature_set)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    classes = encoder.classes_.tolist()
    n_classes = len(classes)
    cat_features = [features.index(col) for col in categorical_columns_for_feature_set(args.feature_set) if col in features]

    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_strength": float(args.random_strength),
        "bagging_temperature": float(args.bagging_temperature),
        "auto_class_weights": "Balanced",
        "random_seed": int(args.seed),
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": int(args.log_period),
    }

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)
    splits = list(cv.split(x, y))[: int(args.fold_limit)]
    if not splits:
        raise ValueError("--fold-limit must be at least 1")

    oof_binary = np.zeros((len(x), n_classes), dtype=np.float32)
    test_binary = np.zeros((len(x_test), n_classes), dtype=np.float32)
    covered_valid = np.zeros(len(x), dtype=bool)
    fold_rows = []
    binary_rows = []
    test_pool = Pool(x_test, cat_features=cat_features)

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        x_tr = x.iloc[tr_idx]
        x_va = x.iloc[va_idx]
        y_tr_full = y[tr_idx]
        y_va_full = y[va_idx]
        fold_test_binary = np.zeros((len(x_test), n_classes), dtype=np.float32)
        covered_valid[va_idx] = True

        for class_idx, class_name in enumerate(classes):
            y_tr = (y_tr_full == class_idx).astype(int)
            y_va = (y_va_full == class_idx).astype(int)
            train_pool = Pool(x_tr, y_tr, cat_features=cat_features)
            valid_pool = Pool(x_va, y_va, cat_features=cat_features)
            progress(
                f"Training fold {fold}/{len(splits)} one-vs-rest class={class_name} "
                f"pos_train={int(y_tr.sum())} pos_valid={int(y_va.sum())}"
            )
            model = CatBoostClassifier(**params)
            model.fit(
                train_pool,
                eval_set=valid_pool,
                use_best_model=True,
                early_stopping_rounds=int(args.early_stopping_rounds),
            )
            valid_prob = model.predict_proba(valid_pool)[:, 1]
            test_prob = model.predict_proba(test_pool)[:, 1]
            oof_binary[va_idx, class_idx] = valid_prob.astype(np.float32)
            fold_test_binary[:, class_idx] = test_prob.astype(np.float32)

            binary_pred = (valid_prob >= 0.5).astype(int)
            binary_score = balanced_accuracy_score(y_va, binary_pred)
            best_iteration = int(model.get_best_iteration() or args.iterations)
            binary_rows.append(
                {
                    "fold": fold,
                    "class": class_name,
                    "binary_balanced_accuracy": float(binary_score),
                    "best_iteration": best_iteration,
                }
            )
            progress(
                f"fold {fold} class={class_name} binary_BAC={binary_score:.6f} "
                f"best_iteration={best_iteration}"
            )

        fold_test_binary = normalize_probs(fold_test_binary)
        test_binary += fold_test_binary / len(splits)
        fold_oof = normalize_probs(oof_binary[va_idx])
        fold_pred = fold_oof.argmax(axis=1)
        fold_score = balanced_accuracy(y_va_full, fold_pred, n_classes)
        fold_rows.append(
            {
                "fold": fold,
                "balanced_accuracy": fold_score,
                "class_recalls": class_recalls(y_va_full, fold_pred, classes),
            }
        )
        progress(f"fold {fold} multiclass_BAC={fold_score:.6f}")

    oof_proba = normalize_probs(oof_binary)
    test_proba = normalize_probs(test_binary)
    oof_pred = oof_proba.argmax(axis=1)
    oof_score = balanced_accuracy(y, oof_pred, n_classes)
    covered_oof_score = balanced_accuracy(y[covered_valid], oof_pred[covered_valid], n_classes)
    progress(f"covered OOF balanced_accuracy={covered_oof_score:.6f}")
    if covered_valid.all():
        progress(f"full OOF balanced_accuracy={oof_score:.6f}")

    submission = sample.copy()
    submission["class"] = np.array(classes)[test_proba.argmax(axis=1)]
    submission_path = args.output_dir / "ovr_catboost_submission.csv"
    submission.to_csv(submission_path, index=False)
    np.save(args.output_dir / "ovr_catboost_oof_proba.npy", oof_proba.astype(np.float32))
    np.save(args.output_dir / "ovr_catboost_test_proba.npy", test_proba.astype(np.float32))
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_scores.csv", index=False)
    pd.DataFrame(binary_rows).to_csv(args.output_dir / "binary_fold_scores.csv", index=False)

    report = {
        "purpose": "One-vs-rest CatBoost specialist bank. No public submission CSV is used.",
        "classes": classes,
        "feature_set": args.feature_set,
        "feature_count": len(features),
        "cat_features": cat_features,
        "params": params,
        "fold_limit": len(splits),
        "fold_scores": fold_rows,
        "binary_fold_scores": binary_rows,
        "covered_valid_rows": int(covered_valid.sum()),
        "covered_oof_balanced_accuracy": covered_oof_score,
        "full_oof_balanced_accuracy": oof_score if covered_valid.all() else None,
        "covered_oof_class_recalls": class_recalls(y[covered_valid], oof_pred[covered_valid], classes),
        "submission_path": str(submission_path.relative_to(ROOT)),
        "submission_class_share": submission["class"].value_counts(normalize=True).sort_index().to_dict(),
        "outputs": [
            "ovr_catboost_submission.csv",
            "ovr_catboost_oof_proba.npy",
            "ovr_catboost_test_proba.npy",
            "fold_scores.csv",
            "binary_fold_scores.csv",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
