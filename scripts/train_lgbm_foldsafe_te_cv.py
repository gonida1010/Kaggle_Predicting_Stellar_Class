from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "artifacts" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / "artifacts" / ".cache"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from src.stellar_features import (  # noqa: E402
    add_realmlp_style_features,
    categorical_columns_for_feature_set,
    encode_categories,
)
from train_lgbm_cv import (  # noqa: E402
    class_recalls,
    diagnostic_iterations,
    diagnostic_train_indices,
    normalize_probs,
    write_diagnostic_plots,
)


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "lgbm_foldsafe_te_realmlp"
SEED = 20260620
N_SPLITS = 5
CLASSES = ["GALAXY", "QSO", "STAR"]
DEFAULT_TE_COLS = [
    "spectral_type",
    "galaxy_population",
    "spectral_population",
    "alpha_floor_x_delta_floor",
    "u_floor_x_z_floor",
    "delta_qbin_100",
    "delta_qbin_500",
    "u_floor_cat",
    "g_floor_cat",
    "z_floor_cat",
    "redshift_floor_cat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train LightGBM with fold-safe multiclass target encoding. "
            "Target statistics are fit only on the training fold and transformed into valid/test."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--fold-limit", type=int, default=N_SPLITS)
    parser.add_argument("--n-estimators", type=int, default=5000)
    parser.add_argument("--early-stopping-rounds", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.028)
    parser.add_argument("--num-leaves", type=int, default=96)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=90)
    parser.add_argument("--subsample", type=float, default=0.88)
    parser.add_argument("--colsample-bytree", type=float, default=0.84)
    parser.add_argument("--reg-alpha", type=float, default=0.10)
    parser.add_argument("--reg-lambda", type=float, default=2.4)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--te-cols", nargs="*", default=DEFAULT_TE_COLS)
    parser.add_argument("--te-smoothing", type=float, default=40.0)
    parser.add_argument("--te-min-count", type=int, default=1)
    parser.add_argument("--log-period", type=int, default=200)
    parser.add_argument(
        "--early-stop-metric",
        choices=["logloss", "valid-bac"],
        default="valid-bac",
    )
    parser.add_argument("--diagnostic-period", type=int, default=100)
    parser.add_argument("--diagnostic-train-sample", type=int, default=50000)
    parser.add_argument(
        "--prediction-iteration-policy",
        choices=["early-stop-best", "valid-bac-best", "fixed"],
        default="early-stop-best",
    )
    parser.add_argument("--fixed-iteration", type=int, default=0)
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def choose_prediction_iteration(args: argparse.Namespace, fold_diag: list[dict], early_stop_best_iteration: int) -> int:
    if args.prediction_iteration_policy == "fixed":
        if args.fixed_iteration <= 0:
            raise ValueError("--fixed-iteration must be positive when using fixed prediction iteration.")
        return int(args.fixed_iteration)
    if args.prediction_iteration_policy == "valid-bac-best" and fold_diag:
        best = max(fold_diag, key=lambda row: row["valid_balanced_accuracy"])
        return int(best["iteration"])
    return int(early_stop_best_iteration)


def lgb_balanced_accuracy_metric_factory(n_classes: int):
    def metric(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
        pred = np.asarray(y_pred)
        if pred.ndim == 1:
            pred = pred.reshape(len(y_true), n_classes)
        return "balanced_accuracy", float(balanced_accuracy_score(y_true, pred.argmax(axis=1))), True

    return metric


def prepare_base_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train_raw, test_raw = add_realmlp_style_features(train, test)
    train_encoded, test_encoded = encode_categories(train_raw, test_raw, feature_set="realmlp")
    drop_cols = ["id", "class"]
    features = [col for col in train_encoded.columns if col not in drop_cols]
    return train_raw, test_raw, train_encoded[features].copy(), test_encoded[features].copy(), features


def fit_target_encoding(
    frame: pd.DataFrame,
    y: np.ndarray,
    cols: list[str],
    n_classes: int,
    smoothing: float,
) -> dict[str, dict]:
    priors = np.bincount(y, minlength=n_classes).astype(np.float64)
    priors = priors / priors.sum()
    encoders = {}
    for col in cols:
        keys = frame[col].astype(str)
        tmp = pd.DataFrame({"key": keys, "target": y})
        counts = pd.crosstab(tmp["key"], tmp["target"]).astype(np.float64)
        for class_idx in range(n_classes):
            if class_idx not in counts.columns:
                counts[class_idx] = 0.0
        counts = counts[list(range(n_classes))]
        total = counts.sum(axis=1)
        encoded = counts.copy()
        for class_idx in range(n_classes):
            encoded[class_idx] = (counts[class_idx] + priors[class_idx] * smoothing) / (total + smoothing)
        encoded["count_log1p"] = np.log1p(total)
        encoders[col] = {
            "table": encoded,
            "priors": priors,
        }
    return encoders


def transform_target_encoding(frame: pd.DataFrame, encoders: dict[str, dict], cols: list[str], prefix: str) -> pd.DataFrame:
    pieces = []
    index = frame.index
    for col in cols:
        encoder = encoders[col]
        table = encoder["table"]
        priors = encoder["priors"]
        keys = frame[col].astype(str)
        part = pd.DataFrame(index=index)
        for class_idx, label in enumerate(CLASSES):
            part[f"{prefix}_{col}_{label}"] = keys.map(table[class_idx]).fillna(float(priors[class_idx])).astype("float32")
        part[f"{prefix}_{col}_count_log1p"] = keys.map(table["count_log1p"]).fillna(0.0).astype("float32")
        pieces.append(part)
    return pd.concat(pieces, axis=1)


def make_fold_matrices(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    x_base: pd.DataFrame,
    x_test_base: pd.DataFrame,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_cols: list[str],
    smoothing: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    encoders = fit_target_encoding(train_raw.iloc[tr_idx], y[tr_idx], te_cols, len(CLASSES), smoothing)
    te_train = transform_target_encoding(train_raw.iloc[tr_idx], encoders, te_cols, "te")
    te_valid = transform_target_encoding(train_raw.iloc[va_idx], encoders, te_cols, "te")
    te_test = transform_target_encoding(test_raw, encoders, te_cols, "te")

    x_tr = pd.concat([x_base.iloc[tr_idx].reset_index(drop=True), te_train.reset_index(drop=True)], axis=1)
    x_va = pd.concat([x_base.iloc[va_idx].reset_index(drop=True), te_valid.reset_index(drop=True)], axis=1)
    x_te = pd.concat([x_test_base.reset_index(drop=True), te_test.reset_index(drop=True)], axis=1)
    return x_tr, x_va, x_te, te_train.columns.tolist()


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    progress("Loading train/test data")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["class"])
    classes = encoder.classes_.tolist()
    if classes != CLASSES:
        raise RuntimeError(f"Unexpected class order: {classes}")

    progress("Building realmlp base features")
    train_raw, test_raw, x_base, x_test_base, base_features = prepare_base_features(train, test)
    te_cols = [col for col in args.te_cols if col in train_raw.columns and col in test_raw.columns]
    if not te_cols:
        raise ValueError("No valid --te-cols found.")
    progress(f"fold-safe TE columns ({len(te_cols)}): {te_cols}")

    params = {
        "objective": "multiclass",
        "num_class": len(classes),
        "metric": "None" if args.early_stop_metric == "valid-bac" else "multi_logloss",
        "learning_rate": float(args.learning_rate),
        "num_leaves": int(args.num_leaves),
        "max_depth": int(args.max_depth),
        "min_child_samples": int(args.min_child_samples),
        "subsample": float(args.subsample),
        "subsample_freq": 1,
        "colsample_bytree": float(args.colsample_bytree),
        "reg_alpha": float(args.reg_alpha),
        "reg_lambda": float(args.reg_lambda),
        "class_weight": "balanced" if args.class_weight == "balanced" else None,
        "random_state": int(args.seed),
        "n_estimators": int(args.n_estimators),
        "n_jobs": -1,
        "verbosity": -1,
    }

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=int(args.seed))
    splits = list(cv.split(x_base, y))[: int(args.fold_limit)]
    if len(splits) != N_SPLITS:
        progress(f"Using fold_limit={len(splits)}; this is a partial smoke/screen run.")

    oof = np.zeros((len(train), len(classes)), dtype=np.float32)
    test_pred = np.zeros((len(test), len(classes)), dtype=np.float32)
    fold_rows = []
    diagnostic_rows = []
    te_feature_names: list[str] | None = None

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        progress(f"Preparing fold-safe TE matrices fold {fold}/{len(splits)}")
        x_tr, x_va, x_te, te_cols_created = make_fold_matrices(
            train_raw,
            test_raw,
            x_base,
            x_test_base,
            y,
            tr_idx,
            va_idx,
            te_cols,
            float(args.te_smoothing),
        )
        if te_feature_names is None:
            te_feature_names = te_cols_created
            progress(f"created {len(te_feature_names)} target-encoding features")

        progress(f"Training LightGBM TE fold {fold}/{len(splits)} train={x_tr.shape} valid={x_va.shape}")
        model = lgb.LGBMClassifier(**params)
        eval_metric = lgb_balanced_accuracy_metric_factory(len(classes)) if args.early_stop_metric == "valid-bac" else "multi_logloss"
        model.fit(
            x_tr,
            y[tr_idx],
            eval_set=[(x_va, y[va_idx])],
            eval_metric=eval_metric,
            callbacks=[
                lgb.early_stopping(int(args.early_stopping_rounds), first_metric_only=True, verbose=False),
                lgb.log_evaluation(int(args.log_period)),
            ],
        )

        early_stop_best_iteration = int(model.best_iteration_ or params["n_estimators"])
        diag_tr_local = diagnostic_train_indices(np.arange(len(tr_idx)), int(args.diagnostic_train_sample), int(args.seed) + fold)
        fold_diag = []
        for iteration in diagnostic_iterations(params["n_estimators"], early_stop_best_iteration, int(args.diagnostic_period)):
            train_diag_pred = normalize_probs(model.predict_proba(x_tr.iloc[diag_tr_local], num_iteration=iteration))
            valid_diag_pred = normalize_probs(model.predict_proba(x_va, num_iteration=iteration))
            row = {
                "fold": fold,
                "iteration": int(iteration),
                "train_mlogloss": float(log_loss(y[tr_idx][diag_tr_local], train_diag_pred, labels=list(range(len(classes))))),
                "valid_mlogloss": float(log_loss(y[va_idx], valid_diag_pred, labels=list(range(len(classes))))),
                "train_balanced_accuracy_sample": float(
                    balanced_accuracy_score(y[tr_idx][diag_tr_local], train_diag_pred.argmax(axis=1))
                ),
                "valid_balanced_accuracy": float(balanced_accuracy_score(y[va_idx], valid_diag_pred.argmax(axis=1))),
            }
            diagnostic_rows.append(row)
            fold_diag.append(row)

        prediction_iteration = choose_prediction_iteration(args, fold_diag, early_stop_best_iteration)
        prediction_iteration = min(prediction_iteration, early_stop_best_iteration)
        va_pred = normalize_probs(model.predict_proba(x_va, num_iteration=prediction_iteration))
        oof[va_idx] = va_pred
        pred_label = va_pred.argmax(axis=1)
        score = float(balanced_accuracy_score(y[va_idx], pred_label))
        diagnostic_best = max(fold_diag, key=lambda row: row["valid_balanced_accuracy"]) if fold_diag else None
        test_pred += normalize_probs(model.predict_proba(x_te, num_iteration=prediction_iteration)) / len(splits)

        fold_rows.append(
            {
                "fold": fold,
                "balanced_accuracy": score,
                "early_stop_metric": args.early_stop_metric,
                "early_stop_best_iteration": early_stop_best_iteration,
                "prediction_iteration": int(prediction_iteration),
                "prediction_iteration_policy": args.prediction_iteration_policy,
                "diagnostic_best_valid_bac_iteration": int(diagnostic_best["iteration"]) if diagnostic_best else None,
                "diagnostic_best_valid_bac": float(diagnostic_best["valid_balanced_accuracy"]) if diagnostic_best else None,
                "class_recalls": class_recalls(y[va_idx], pred_label, classes),
            }
        )
        progress(
            f"fold {fold}: BAC={score:.6f}, early_stop_best_iteration={early_stop_best_iteration}, "
            f"prediction_iteration={prediction_iteration}"
        )

    covered = oof.sum(axis=1) > 0
    oof = normalize_probs(oof)
    test_pred = normalize_probs(test_pred)
    oof_pred = oof.argmax(axis=1)
    covered_score = float(balanced_accuracy_score(y[covered], oof_pred[covered]))
    full_score = float(balanced_accuracy_score(y, oof_pred)) if covered.all() else None
    progress(f"covered OOF balanced_accuracy={covered_score:.9f}")
    if full_score is not None:
        progress(f"full OOF balanced_accuracy={full_score:.9f}")

    submission = sample.copy()
    submission["class"] = encoder.inverse_transform(test_pred.argmax(axis=1))
    np.save(args.output_dir / "lgbm_te_oof_proba.npy", oof.astype(np.float32))
    np.save(args.output_dir / "lgbm_te_test_proba.npy", test_pred.astype(np.float32))
    submission.to_csv(args.output_dir / "lgbm_te_submission.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "lgbm_te_fold_scores.csv", index=False)
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(args.output_dir / "lgbm_te_training_diagnostics.csv", index=False)
    write_diagnostic_plots(diagnostics, args.output_dir)

    report = {
        "purpose": "Fold-safe target-encoded LightGBM. TE is fit only on each training fold.",
        "seed": int(args.seed),
        "n_splits": N_SPLITS,
        "fold_limit": len(splits),
        "classes": classes,
        "base_feature_set": "realmlp",
        "base_feature_count": len(base_features),
        "target_encoding_columns": te_cols,
        "target_encoding_feature_count": len(te_feature_names or []),
        "te_smoothing": float(args.te_smoothing),
        "params": params,
        "early_stop_metric": args.early_stop_metric,
        "prediction_iteration_policy": args.prediction_iteration_policy,
        "diagnostic_period": int(args.diagnostic_period),
        "diagnostic_train_sample": int(args.diagnostic_train_sample),
        "fold_scores": fold_rows,
        "covered_oof_balanced_accuracy": covered_score,
        "oof_balanced_accuracy": full_score,
        "submission_class_share": submission["class"].value_counts(normalize=True).sort_index().to_dict(),
        "outputs": [
            "lgbm_te_oof_proba.npy",
            "lgbm_te_test_proba.npy",
            "lgbm_te_submission.csv",
            "lgbm_te_fold_scores.csv",
            "lgbm_te_training_diagnostics.csv",
            "lgbm_logloss_curve.svg",
            "lgbm_logloss_curve.png",
            "lgbm_logloss_curve_zoom.svg",
            "lgbm_logloss_curve_zoom.png",
            "lgbm_balanced_accuracy_curve.svg",
            "lgbm_balanced_accuracy_curve.png",
        ],
    }
    (args.output_dir / "lgbm_te_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
