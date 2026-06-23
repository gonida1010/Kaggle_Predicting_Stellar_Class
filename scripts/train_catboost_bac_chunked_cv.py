from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import categorical_columns_for_feature_set, make_xy  # noqa: E402
from train_catboost_cv import class_recalls, normalize_probs, write_diagnostic_plots  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
SEED = 20260611
N_SPLITS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train CatBoost in chunks and stop by validation balanced accuracy. "
            "This is designed for cases where logloss keeps improving but BAC plateaus or degrades."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "catboost_bac_chunked")
    parser.add_argument("--feature-set", choices=["base", "advanced", "realmlp", "catv3"], default="catv3")
    parser.add_argument("--fold-limit", type=int, default=N_SPLITS)
    parser.add_argument("--max-iterations", type=int, default=4500)
    parser.add_argument("--chunk-size", type=int, default=150)
    parser.add_argument("--bac-patience-chunks", type=int, default=6)
    parser.add_argument("--bac-min-delta", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.030)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--l2-leaf-reg", type=float, default=12.0)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--bagging-temperature", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--diagnostic-train-sample", type=int, default=50000)
    parser.add_argument("--log-period", type=int, default=50)
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def diagnostic_train_indices(indices: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0 or len(indices) <= sample_size:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=sample_size, replace=False))


def predict_proba(pool: Pool, model: CatBoostClassifier, iteration: int | None = None) -> np.ndarray:
    if iteration is None:
        return normalize_probs(model.predict_proba(pool))
    return normalize_probs(model.predict_proba(pool, ntree_start=0, ntree_end=int(iteration)))


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.max_iterations < args.chunk_size:
        raise ValueError("--max-iterations must be >= --chunk-size")

    progress("Loading train/test data")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    progress(f"Building {args.feature_set} feature matrix")
    x, y_raw, x_test, features = make_xy(train, test, feature_set=args.feature_set)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    classes = encoder.classes_.tolist()
    cat_features = [features.index(col) for col in categorical_columns_for_feature_set(args.feature_set) if col in features]

    base_params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": int(args.chunk_size),
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
    if len(splits) != N_SPLITS:
        progress(f"Using fold_limit={len(splits)}; this is a partial OOF screen run.")

    oof = np.zeros((len(x), len(classes)), dtype=np.float32)
    test_pred = np.zeros((len(x_test), len(classes)), dtype=np.float32)
    test_pool = Pool(x_test, cat_features=cat_features)
    fold_rows: list[dict] = []
    diagnostic_rows: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        progress(f"Training chunked CatBoost fold {fold}/{len(splits)}")
        diag_tr_idx = diagnostic_train_indices(tr_idx, int(args.diagnostic_train_sample), int(args.seed) + fold)
        train_pool = Pool(x.iloc[tr_idx], y[tr_idx], cat_features=cat_features)
        valid_pool = Pool(x.iloc[va_idx], y[va_idx], cat_features=cat_features)
        diag_train_pool = Pool(x.iloc[diag_tr_idx], y[diag_tr_idx], cat_features=cat_features)

        model: CatBoostClassifier | None = None
        best_valid_bac = -np.inf
        best_iteration = 0
        best_valid_logloss = float("nan")
        stale_chunks = 0
        total_iterations = 0

        while total_iterations < int(args.max_iterations):
            this_chunk = min(int(args.chunk_size), int(args.max_iterations) - total_iterations)
            params = {**base_params, "iterations": this_chunk}
            chunk_model = CatBoostClassifier(**params)
            fit_kwargs = {"eval_set": valid_pool, "use_best_model": False}
            if model is not None:
                fit_kwargs["init_model"] = model

            progress(
                f"fold {fold}: chunk start {total_iterations + 1}-{total_iterations + this_chunk}; "
                f"current_best_bac={best_valid_bac if np.isfinite(best_valid_bac) else None}"
            )
            chunk_model.fit(train_pool, **fit_kwargs)
            model = chunk_model
            total_iterations = int(model.tree_count_)

            train_diag_pred = predict_proba(diag_train_pool, model)
            valid_diag_pred = predict_proba(valid_pool, model)
            train_bac = balanced_accuracy_score(y[diag_tr_idx], train_diag_pred.argmax(axis=1))
            valid_bac = balanced_accuracy_score(y[va_idx], valid_diag_pred.argmax(axis=1))
            train_loss = log_loss(y[diag_tr_idx], train_diag_pred, labels=list(range(len(classes))))
            valid_loss = log_loss(y[va_idx], valid_diag_pred, labels=list(range(len(classes))))

            improved = valid_bac > best_valid_bac + float(args.bac_min_delta)
            if improved:
                best_valid_bac = float(valid_bac)
                best_iteration = int(total_iterations)
                best_valid_logloss = float(valid_loss)
                stale_chunks = 0
            else:
                stale_chunks += 1

            row = {
                "fold": fold,
                "iteration": int(total_iterations),
                "train_mlogloss": float(train_loss),
                "valid_mlogloss": float(valid_loss),
                "train_balanced_accuracy_sample": float(train_bac),
                "valid_balanced_accuracy": float(valid_bac),
                "best_valid_balanced_accuracy_so_far": float(best_valid_bac),
                "best_iteration_so_far": int(best_iteration),
                "stale_chunks": int(stale_chunks),
            }
            diagnostic_rows.append(row)
            progress(
                f"fold {fold}: iter={total_iterations}, valid_bac={valid_bac:.6f}, "
                f"best_bac={best_valid_bac:.6f}@{best_iteration}, "
                f"valid_logloss={valid_loss:.6f}, stale_chunks={stale_chunks}/{args.bac_patience_chunks}"
            )

            pd.DataFrame(diagnostic_rows).to_csv(
                args.output_dir / "catboost_training_diagnostics_partial.csv",
                index=False,
            )
            if stale_chunks >= int(args.bac_patience_chunks):
                progress(f"fold {fold}: BAC early stop after {stale_chunks} stale chunks")
                break

        if model is None:
            raise RuntimeError("CatBoost model was not trained")

        prediction_iteration = max(1, int(best_iteration))
        va_pred = predict_proba(valid_pool, model, prediction_iteration)
        oof[va_idx] = va_pred
        pred_label = va_pred.argmax(axis=1)
        score = balanced_accuracy_score(y[va_idx], pred_label)
        fold_rows.append(
            {
                "fold": fold,
                "balanced_accuracy": float(score),
                "prediction_iteration": int(prediction_iteration),
                "tree_count": int(model.tree_count_),
                "best_valid_balanced_accuracy": float(best_valid_bac),
                "best_valid_logloss_at_best_bac": float(best_valid_logloss),
                "stopped_after_iterations": int(total_iterations),
                "stale_chunks_at_stop": int(stale_chunks),
                "class_recalls": class_recalls(y[va_idx], pred_label, classes),
            }
        )
        test_pred += predict_proba(test_pool, model, prediction_iteration) / len(splits)
        np.save(args.output_dir / "catboost_oof_proba_partial.npy", oof.astype(np.float32))
        np.save(args.output_dir / "catboost_test_proba_partial.npy", test_pred.astype(np.float32))
        pd.DataFrame(fold_rows).to_csv(args.output_dir / "catboost_fold_scores_partial.csv", index=False)
        progress(
            f"fold {fold}: selected prediction_iteration={prediction_iteration}, "
            f"fold_bac={score:.6f}, final_tree_count={int(model.tree_count_)}"
        )

    covered = oof.sum(axis=1) > 0
    oof = normalize_probs(oof)
    test_pred = normalize_probs(test_pred)
    oof_pred = oof.argmax(axis=1)
    covered_score = balanced_accuracy_score(y[covered], oof_pred[covered])
    full_score = balanced_accuracy_score(y, oof_pred) if covered.all() else None
    progress(f"covered OOF balanced_accuracy={covered_score:.6f}")
    if full_score is not None:
        progress(f"full OOF balanced_accuracy={full_score:.6f}")

    submission = sample.copy()
    submission["class"] = encoder.inverse_transform(test_pred.argmax(axis=1))

    np.save(args.output_dir / "catboost_oof_proba.npy", oof.astype(np.float32))
    np.save(args.output_dir / "catboost_test_proba.npy", test_pred.astype(np.float32))
    submission.to_csv(args.output_dir / "catboost_baseline_submission.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "catboost_fold_scores.csv", index=False)
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(args.output_dir / "catboost_training_diagnostics.csv", index=False)
    write_diagnostic_plots(diagnostics, args.output_dir)

    report = {
        "purpose": "Chunked CatBoost with validation balanced-accuracy early stopping.",
        "seed": int(args.seed),
        "n_splits": N_SPLITS,
        "fold_limit": len(splits),
        "classes": classes,
        "feature_set": args.feature_set,
        "feature_count": int(len(features)),
        "cat_feature_count": int(len(cat_features)),
        "params": base_params,
        "max_iterations": int(args.max_iterations),
        "chunk_size": int(args.chunk_size),
        "bac_patience_chunks": int(args.bac_patience_chunks),
        "bac_min_delta": float(args.bac_min_delta),
        "fold_scores": fold_rows,
        "covered_oof_balanced_accuracy": float(covered_score),
        "oof_balanced_accuracy": float(full_score) if full_score is not None else None,
        "submission_class_share": submission["class"].value_counts(normalize=True).sort_index().to_dict(),
        "outputs": [
            "catboost_oof_proba.npy",
            "catboost_test_proba.npy",
            "catboost_baseline_submission.csv",
            "catboost_fold_scores.csv",
            "catboost_training_diagnostics.csv",
            "catboost_logloss_curve.svg",
            "catboost_logloss_curve.png",
            "catboost_logloss_curve_zoom.svg",
            "catboost_logloss_curve_zoom.png",
            "catboost_balanced_accuracy_curve.svg",
            "catboost_balanced_accuracy_curve.png",
        ],
    }
    (args.output_dir / "catboost_baseline_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

