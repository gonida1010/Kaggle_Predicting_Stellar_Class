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

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import make_xy  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
SEED = 20260610
N_SPLITS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a fold-safe LightGBM multiclass model and save OOF/test probabilities plus diagnostics. "
            "No public submission CSV is used."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--feature-set", choices=["base", "advanced", "realmlp"], default="base")
    parser.add_argument("--fold-limit", type=int, default=N_SPLITS)
    parser.add_argument("--n-estimators", type=int, default=4500)
    parser.add_argument("--early-stopping-rounds", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=96)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--subsample", type=float, default=0.88)
    parser.add_argument("--colsample-bytree", type=float, default=0.86)
    parser.add_argument("--reg-alpha", type=float, default=0.08)
    parser.add_argument("--reg-lambda", type=float, default=1.8)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-period", type=int, default=200)
    parser.add_argument(
        "--early-stop-metric",
        choices=["logloss", "valid-bac"],
        default="logloss",
        help="Metric used by LightGBM early stopping. valid-bac stops on validation balanced accuracy.",
    )
    parser.add_argument("--diagnostic-period", type=int, default=100)
    parser.add_argument("--diagnostic-train-sample", type=int, default=50000)
    parser.add_argument(
        "--prediction-iteration-policy",
        choices=["early-stop-best", "logloss-best", "valid-bac-best", "fixed"],
        default="early-stop-best",
    )
    parser.add_argument("--fixed-iteration", type=int, default=0)
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_probs(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return (proba / row_sum).astype(np.float32)


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, float]:
    out = {}
    for idx, label in enumerate(classes):
        mask = y_true == idx
        out[label] = float((y_pred[mask] == idx).mean()) if mask.any() else float("nan")
    return out


def diagnostic_iterations(n_estimators: int, best_iteration: int, period: int) -> list[int]:
    if period <= 0:
        return []
    max_iteration = max(1, min(int(n_estimators), int(best_iteration)))
    points = set(range(1, max_iteration + 1, period))
    points.add(max_iteration)
    return sorted(points)


def diagnostic_train_indices(indices: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0 or len(indices) <= sample_size:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=sample_size, replace=False))


def choose_prediction_iteration(args: argparse.Namespace, fold_diag: list[dict], early_stop_best_iteration: int) -> int:
    if args.prediction_iteration_policy == "fixed":
        if args.fixed_iteration <= 0:
            raise ValueError("--fixed-iteration must be positive when using --prediction-iteration-policy fixed.")
        return int(args.fixed_iteration)
    if args.prediction_iteration_policy == "valid-bac-best" and fold_diag:
        best = max(fold_diag, key=lambda row: row["valid_balanced_accuracy"])
        return int(best["iteration"])
    return int(early_stop_best_iteration)


def write_matplotlib_diagnostic_plots(grouped: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception:
        return

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=150)
    ax.plot(grouped["iteration"], grouped["train_mlogloss"], label="train mlogloss", color="#d92d20", linewidth=2.2)
    ax.plot(grouped["iteration"], grouped["valid_mlogloss"], label="valid mlogloss", color="#1f5eff", linewidth=2.2)
    ax.set_title("LightGBM Train vs Valid Logloss")
    ax.set_xlabel("Boosting rounds")
    ax.set_ylabel("Multi logloss")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "lgbm_logloss_curve.svg", format="svg")
    fig.savefig(output_dir / "lgbm_logloss_curve.png", format="png")
    plt.close(fig)

    zoomed = grouped[grouped["iteration"] >= min(101, int(grouped["iteration"].max()))]
    if len(zoomed) >= 2:
        fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=150)
        ax.plot(zoomed["iteration"], zoomed["train_mlogloss"], label="train mlogloss", color="#d92d20", linewidth=2.2)
        ax.plot(zoomed["iteration"], zoomed["valid_mlogloss"], label="valid mlogloss", color="#1f5eff", linewidth=2.2)
        ax.set_title("LightGBM Train vs Valid Logloss Zoom")
        ax.set_xlabel("Boosting rounds")
        ax.set_ylabel("Multi logloss")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / "lgbm_logloss_curve_zoom.svg", format="svg")
        fig.savefig(output_dir / "lgbm_logloss_curve_zoom.png", format="png")
        plt.close(fig)

    best_row = grouped.loc[grouped["valid_balanced_accuracy"].idxmax()]
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=150)
    ax.plot(
        grouped["iteration"],
        grouped["train_balanced_accuracy_sample"],
        label="train BAC sample",
        color="#d92d20",
        linewidth=2.2,
    )
    ax.plot(grouped["iteration"], grouped["valid_balanced_accuracy"], label="valid BAC", color="#1f5eff", linewidth=2.2)
    ax.axvline(best_row["iteration"], color="#111827", linestyle="--", linewidth=1.3)
    ax.annotate(
        f"best valid BAC @ {int(best_row['iteration'])}",
        xy=(best_row["iteration"], best_row["valid_balanced_accuracy"]),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=9,
        color="#111827",
        arrowprops={"arrowstyle": "->", "color": "#111827", "lw": 0.8},
    )
    ax.set_title("LightGBM Train Sample vs Valid Balanced Accuracy")
    ax.set_xlabel("Boosting rounds")
    ax.set_ylabel("Balanced accuracy")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "lgbm_balanced_accuracy_curve.svg", format="svg")
    fig.savefig(output_dir / "lgbm_balanced_accuracy_curve.png", format="png")
    plt.close(fig)


def write_diagnostic_plots(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    if diagnostics.empty:
        return
    grouped = (
        diagnostics.groupby("iteration", as_index=False)
        .agg(
            train_mlogloss=("train_mlogloss", "mean"),
            valid_mlogloss=("valid_mlogloss", "mean"),
            train_balanced_accuracy_sample=("train_balanced_accuracy_sample", "mean"),
            valid_balanced_accuracy=("valid_balanced_accuracy", "mean"),
        )
        .sort_values("iteration")
    )
    write_matplotlib_diagnostic_plots(grouped, output_dir)


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

    def lgb_balanced_accuracy_metric(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
        pred = np.asarray(y_pred)
        if pred.ndim == 1:
            pred = pred.reshape(len(y_true), len(classes))
        return "balanced_accuracy", float(balanced_accuracy_score(y_true, pred.argmax(axis=1))), True

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

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)
    splits = list(cv.split(x, y))[: int(args.fold_limit)]
    if not splits:
        raise ValueError("--fold-limit must be at least 1")
    if len(splits) != N_SPLITS:
        progress(f"Using fold_limit={len(splits)}; this is a partial OOF smoke/screen run.")

    oof = np.zeros((len(x), len(classes)), dtype=np.float32)
    test_pred = np.zeros((len(x_test), len(classes)), dtype=np.float32)
    fold_rows = []
    diagnostic_rows = []
    best_iterations = []

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        progress(f"Training LightGBM fold {fold}/{len(splits)}")
        diag_tr_idx = diagnostic_train_indices(tr_idx, int(args.diagnostic_train_sample), int(args.seed) + fold)
        model = lgb.LGBMClassifier(**params)
        eval_metric = lgb_balanced_accuracy_metric if args.early_stop_metric == "valid-bac" else "multi_logloss"
        model.fit(
            x.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(x.iloc[va_idx], y[va_idx])],
            eval_metric=eval_metric,
            callbacks=[
                lgb.early_stopping(int(args.early_stopping_rounds), first_metric_only=True, verbose=False),
                lgb.log_evaluation(int(args.log_period)),
            ],
        )

        early_stop_best_iteration = int(model.best_iteration_ or params["n_estimators"])
        fold_diag = []
        for iteration in diagnostic_iterations(params["n_estimators"], early_stop_best_iteration, int(args.diagnostic_period)):
            train_diag_pred = normalize_probs(model.predict_proba(x.iloc[diag_tr_idx], num_iteration=iteration))
            valid_diag_pred = normalize_probs(model.predict_proba(x.iloc[va_idx], num_iteration=iteration))
            row = {
                "fold": fold,
                "iteration": int(iteration),
                "train_mlogloss": float(log_loss(y[diag_tr_idx], train_diag_pred, labels=list(range(len(classes))))),
                "valid_mlogloss": float(log_loss(y[va_idx], valid_diag_pred, labels=list(range(len(classes))))),
                "train_balanced_accuracy_sample": float(
                    balanced_accuracy_score(y[diag_tr_idx], train_diag_pred.argmax(axis=1))
                ),
                "valid_balanced_accuracy": float(balanced_accuracy_score(y[va_idx], valid_diag_pred.argmax(axis=1))),
            }
            diagnostic_rows.append(row)
            fold_diag.append(row)

        prediction_iteration = choose_prediction_iteration(args, fold_diag, early_stop_best_iteration)
        prediction_iteration = min(prediction_iteration, early_stop_best_iteration)
        va_pred = normalize_probs(model.predict_proba(x.iloc[va_idx], num_iteration=prediction_iteration))
        oof[va_idx] = va_pred
        pred_label = va_pred.argmax(axis=1)
        score = balanced_accuracy_score(y[va_idx], pred_label)
        diagnostic_best = max(fold_diag, key=lambda row: row["valid_balanced_accuracy"]) if fold_diag else None
        best_iterations.append(early_stop_best_iteration)
        fold_rows.append(
            {
                "fold": fold,
                "balanced_accuracy": float(score),
                "early_stop_metric": args.early_stop_metric,
                "early_stop_best_iteration": early_stop_best_iteration,
                "logloss_best_iteration": early_stop_best_iteration if args.early_stop_metric == "logloss" else None,
                "prediction_iteration": int(prediction_iteration),
                "prediction_iteration_policy": args.prediction_iteration_policy,
                "diagnostic_best_valid_bac_iteration": int(diagnostic_best["iteration"]) if diagnostic_best else None,
                "diagnostic_best_valid_bac": float(diagnostic_best["valid_balanced_accuracy"]) if diagnostic_best else None,
                "class_recalls": class_recalls(y[va_idx], pred_label, classes),
            }
        )
        test_pred += normalize_probs(model.predict_proba(x_test, num_iteration=prediction_iteration)) / len(splits)
        progress(
            f"fold {fold}: balanced_accuracy={score:.6f}, "
            f"early_stop_metric={args.early_stop_metric}, "
            f"early_stop_best_iteration={early_stop_best_iteration}, prediction_iteration={prediction_iteration}"
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

    np.save(args.output_dir / "lgbm_oof_proba.npy", oof.astype(np.float32))
    np.save(args.output_dir / "lgbm_test_proba.npy", test_pred.astype(np.float32))
    submission.to_csv(args.output_dir / "lgbm_baseline_submission.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "lgbm_fold_scores.csv", index=False)
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(args.output_dir / "lgbm_training_diagnostics.csv", index=False)
    write_diagnostic_plots(diagnostics, args.output_dir)

    report = {
        "seed": int(args.seed),
        "n_splits": N_SPLITS,
        "fold_limit": len(splits),
        "classes": classes,
        "feature_set": args.feature_set,
        "features": features,
        "params": params,
        "early_stop_metric": args.early_stop_metric,
        "prediction_iteration_policy": args.prediction_iteration_policy,
        "fixed_iteration": int(args.fixed_iteration),
        "diagnostic_period": int(args.diagnostic_period),
        "diagnostic_train_sample": int(args.diagnostic_train_sample),
        "fold_scores": fold_rows,
        "covered_oof_balanced_accuracy": float(covered_score),
        "oof_balanced_accuracy": float(full_score) if full_score is not None else None,
        "best_iterations": best_iterations,
        "submission_class_share": submission["class"].value_counts(normalize=True).sort_index().to_dict(),
        "outputs": [
            "lgbm_oof_proba.npy",
            "lgbm_test_proba.npy",
            "lgbm_baseline_submission.csv",
            "lgbm_fold_scores.csv",
            "lgbm_training_diagnostics.csv",
            "lgbm_logloss_curve.svg",
            "lgbm_logloss_curve.png",
            "lgbm_logloss_curve_zoom.svg",
            "lgbm_logloss_curve_zoom.png",
            "lgbm_balanced_accuracy_curve.svg",
            "lgbm_balanced_accuracy_curve.png",
        ],
    }
    (args.output_dir / "lgbm_baseline_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
