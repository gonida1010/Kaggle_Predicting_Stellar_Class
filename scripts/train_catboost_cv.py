from __future__ import annotations

import argparse
import json
import os
import sys
import threading
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


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
SEED = 20260611
N_SPLITS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a fold-safe CatBoost multiclass model and save OOF/test probabilities plus diagnostics. "
            "No public submission CSV is used."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--feature-set", choices=["base", "advanced", "realmlp", "catv3"], default="base")
    parser.add_argument("--fold-limit", type=int, default=N_SPLITS)
    parser.add_argument("--iterations", type=int, default=3200)
    parser.add_argument("--early-stopping-rounds", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2-leaf-reg", type=float, default=8.0)
    parser.add_argument("--random-strength", type=float, default=0.6)
    parser.add_argument("--bagging-temperature", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-period", type=int, default=250)
    parser.add_argument(
        "--early-stop-metric",
        choices=["logloss", "valid-bac"],
        default="logloss",
        help="Metric used by CatBoost early stopping. valid-bac stops on validation balanced accuracy.",
    )
    parser.add_argument("--diagnostic-period", type=int, default=100)
    parser.add_argument("--diagnostic-train-sample", type=int, default=50000)
    parser.add_argument(
        "--prediction-iteration-policy",
        choices=["early-stop-best", "logloss-best", "valid-bac-best", "fixed"],
        default="early-stop-best",
    )
    parser.add_argument("--fixed-iteration", type=int, default=0)
    parser.add_argument(
        "--save-snapshot",
        action="store_true",
        help="Enable CatBoost snapshots per fold so long runs can resume after interruption.",
    )
    parser.add_argument("--snapshot-interval", type=int, default=600)
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=60,
        help="Print Python-side heartbeat while CatBoost is inside model.fit. This avoids buffered CatBoost stdout looking frozen.",
    )
    parser.add_argument(
        "--use-best-model",
        action="store_true",
        help=(
            "Let CatBoost shrink the fitted model to its eval_metric best iteration. "
            "Do not use this with --prediction-iteration-policy valid-bac-best."
        ),
    )
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def read_catboost_metric_tail(train_dir: Path) -> str | None:
    for filename in ("test_error.tsv", "learn_error.tsv"):
        path = train_dir / filename
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8").strip().splitlines()
        except OSError:
            continue
        if len(lines) < 2:
            continue
        header = lines[0].split("\t")
        values = lines[-1].split("\t")
        pairs = []
        for key, value in zip(header, values):
            if key.lower() in {"iter", "iteration"} or len(pairs) < 4:
                pairs.append(f"{key}={value}")
        return f"{filename}: " + ", ".join(pairs)
    return None


def start_fit_heartbeat(train_dir: Path, fold: int, interval_seconds: int) -> tuple[threading.Event, threading.Thread | None]:
    stop_event = threading.Event()
    if interval_seconds <= 0:
        return stop_event, None

    def run() -> None:
        last_line = None
        while not stop_event.wait(interval_seconds):
            line = read_catboost_metric_tail(train_dir)
            if line and line != last_line:
                progress(f"fold {fold} heartbeat: {line}")
                last_line = line
            else:
                progress(f"fold {fold} heartbeat: CatBoost fit still running")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return stop_event, thread


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


def diagnostic_iterations(max_iteration: int, period: int) -> list[int]:
    if period <= 0:
        return []
    max_iteration = max(1, int(max_iteration))
    points = set(range(1, max_iteration + 1, period))
    points.add(max_iteration)
    return sorted(points)


def diagnostic_train_indices(indices: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0 or len(indices) <= sample_size:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=sample_size, replace=False))


def eval_metric_history(evals_result: dict, split_name: str) -> list[float]:
    if split_name in evals_result:
        metrics = evals_result[split_name]
    else:
        candidates = [key for key in evals_result if split_name.lower() in key.lower()]
        metrics = evals_result[candidates[0]] if candidates else {}
    for key in ("MultiClass", "Logloss"):
        if key in metrics:
            return metrics[key]
    if metrics:
        return next(iter(metrics.values()))
    return []


class CatBoostBalancedAccuracyMetric:
    def is_max_optimal(self) -> bool:
        return True

    def evaluate(self, approxes, target, weight):
        pred = np.vstack(approxes).T
        score = balanced_accuracy_score(np.asarray(target, dtype=np.int64), pred.argmax(axis=1))
        return float(score), 1.0

    def get_final_error(self, error, weight) -> float:
        return float(error)


def choose_prediction_iteration(args: argparse.Namespace, fold_diag: list[dict], early_stop_best_iteration: int) -> int:
    if args.prediction_iteration_policy == "fixed":
        if args.fixed_iteration <= 0:
            raise ValueError("--fixed-iteration must be positive when using --prediction-iteration-policy fixed.")
        return int(args.fixed_iteration)
    if args.prediction_iteration_policy == "valid-bac-best" and fold_diag:
        best = max(fold_diag, key=lambda row: row["valid_balanced_accuracy"])
        return int(best["iteration"])
    return int(early_stop_best_iteration) + 1


def predict_proba(pool: Pool, model: CatBoostClassifier, iteration: int) -> np.ndarray:
    return normalize_probs(model.predict_proba(pool, ntree_start=0, ntree_end=int(iteration)))


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
    ax.plot(grouped["iteration"], grouped["train_mlogloss"], label="train loss", color="#d92d20", linewidth=2.2)
    ax.plot(grouped["iteration"], grouped["valid_mlogloss"], label="valid loss", color="#1f5eff", linewidth=2.2)
    ax.set_title("CatBoost Train vs Valid Loss")
    ax.set_xlabel("Boosting rounds")
    ax.set_ylabel("MultiClass loss")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "catboost_logloss_curve.svg", format="svg")
    fig.savefig(output_dir / "catboost_logloss_curve.png", format="png")
    plt.close(fig)

    zoomed = grouped[grouped["iteration"] >= min(101, int(grouped["iteration"].max()))]
    if len(zoomed) >= 2:
        fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=150)
        ax.plot(zoomed["iteration"], zoomed["train_mlogloss"], label="train loss", color="#d92d20", linewidth=2.2)
        ax.plot(zoomed["iteration"], zoomed["valid_mlogloss"], label="valid loss", color="#1f5eff", linewidth=2.2)
        ax.set_title("CatBoost Train vs Valid Loss Zoom")
        ax.set_xlabel("Boosting rounds")
        ax.set_ylabel("MultiClass loss")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / "catboost_logloss_curve_zoom.svg", format="svg")
        fig.savefig(output_dir / "catboost_logloss_curve_zoom.png", format="png")
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
    ax.set_title("CatBoost Train Sample vs Valid Balanced Accuracy")
    ax.set_xlabel("Boosting rounds")
    ax.set_ylabel("Balanced accuracy")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "catboost_balanced_accuracy_curve.svg", format="svg")
    fig.savefig(output_dir / "catboost_balanced_accuracy_curve.png", format="png")
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
    if args.use_best_model and args.prediction_iteration_policy == "valid-bac-best":
        raise ValueError(
            "--use-best-model is unsafe with --prediction-iteration-policy valid-bac-best: "
            "CatBoost would shrink to eval_metric/logloss best before the BAC diagnostic can choose a later iteration."
        )
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
    cat_features = [features.index(col) for col in categorical_columns_for_feature_set(args.feature_set) if col in features]

    params = {
        "loss_function": "MultiClass",
        "eval_metric": CatBoostBalancedAccuracyMetric() if args.early_stop_metric == "valid-bac" else "MultiClass",
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_strength": float(args.random_strength),
        "bagging_temperature": float(args.bagging_temperature),
        "auto_class_weights": "Balanced",
        "random_seed": int(args.seed),
        "allow_writing_files": bool(args.save_snapshot or args.heartbeat_seconds > 0),
        "thread_count": -1,
        "verbose": int(args.log_period),
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
    test_pool = Pool(x_test, cat_features=cat_features)

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        progress(f"Training CatBoost fold {fold}/{len(splits)}")
        diag_tr_idx = diagnostic_train_indices(tr_idx, int(args.diagnostic_train_sample), int(args.seed) + fold)
        train_pool = Pool(x.iloc[tr_idx], y[tr_idx], cat_features=cat_features)
        valid_pool = Pool(x.iloc[va_idx], y[va_idx], cat_features=cat_features)
        diag_train_pool = Pool(x.iloc[diag_tr_idx], y[diag_tr_idx], cat_features=cat_features)
        fold_train_dir = args.output_dir / "catboost_train_dir" / f"fold_{fold}"
        fold_train_dir.mkdir(parents=True, exist_ok=True)
        fold_params = {**params, "train_dir": str(fold_train_dir)}
        model = CatBoostClassifier(**fold_params)
        fit_kwargs = {
            "eval_set": valid_pool,
            "use_best_model": bool(args.use_best_model),
            "early_stopping_rounds": int(args.early_stopping_rounds),
        }
        if args.save_snapshot:
            fit_kwargs.update(
                {
                    "save_snapshot": True,
                    "snapshot_file": str(args.output_dir / f"catboost_fold{fold}.snapshot"),
                    "snapshot_interval": int(args.snapshot_interval),
                }
            )
        heartbeat_stop, heartbeat_thread = start_fit_heartbeat(
            fold_train_dir,
            fold,
            int(args.heartbeat_seconds),
        )
        try:
            progress(
                f"fold {fold}: fit start; use_best_model={bool(args.use_best_model)}, "
                f"early_stop_metric={args.early_stop_metric}, "
                f"prediction_iteration_policy={args.prediction_iteration_policy}"
            )
            model.fit(
                train_pool,
                **fit_kwargs,
            )
        finally:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=2)

        tree_count = int(model.tree_count_)
        early_stop_best_iteration = int(model.get_best_iteration() if model.get_best_iteration() is not None else tree_count - 1)
        progress(
            f"fold {fold}: fit done; tree_count={tree_count}, "
            f"eval_metric_best_iteration={early_stop_best_iteration}, "
            f"model_was_not_shrunk={not bool(args.use_best_model)}"
        )
        evals_result = model.get_evals_result()
        if args.early_stop_metric == "valid-bac":
            learn_loss = []
            valid_loss = []
        else:
            learn_loss = eval_metric_history(evals_result, "learn")
            valid_loss = eval_metric_history(evals_result, "validation")
        fold_diag = []
        for iteration in diagnostic_iterations(tree_count, int(args.diagnostic_period)):
            train_diag_pred = predict_proba(diag_train_pool, model, iteration)
            valid_diag_pred = predict_proba(valid_pool, model, iteration)
            loss_idx = min(iteration - 1, len(learn_loss) - 1, len(valid_loss) - 1)
            row = {
                "fold": fold,
                "iteration": int(iteration),
                "train_mlogloss": float(learn_loss[loss_idx]) if learn_loss else float(log_loss(y[diag_tr_idx], train_diag_pred, labels=list(range(len(classes))))),
                "valid_mlogloss": float(valid_loss[loss_idx]) if valid_loss else float(log_loss(y[va_idx], valid_diag_pred, labels=list(range(len(classes))))),
                "train_balanced_accuracy_sample": float(
                    balanced_accuracy_score(y[diag_tr_idx], train_diag_pred.argmax(axis=1))
                ),
                "valid_balanced_accuracy": float(balanced_accuracy_score(y[va_idx], valid_diag_pred.argmax(axis=1))),
            }
            diagnostic_rows.append(row)
            fold_diag.append(row)

        prediction_iteration = choose_prediction_iteration(args, fold_diag, early_stop_best_iteration)
        prediction_iteration = min(prediction_iteration, tree_count)
        va_pred = predict_proba(valid_pool, model, prediction_iteration)
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
                "tree_count": tree_count,
                "prediction_iteration": int(prediction_iteration),
                "prediction_iteration_policy": args.prediction_iteration_policy,
                "diagnostic_best_valid_bac_iteration": int(diagnostic_best["iteration"]) if diagnostic_best else None,
                "diagnostic_best_valid_bac": float(diagnostic_best["valid_balanced_accuracy"]) if diagnostic_best else None,
                "class_recalls": class_recalls(y[va_idx], pred_label, classes),
            }
        )
        test_pred += predict_proba(test_pool, model, prediction_iteration) / len(splits)
        progress(
            f"fold {fold}: balanced_accuracy={score:.6f}, "
            f"early_stop_metric={args.early_stop_metric}, "
            f"early_stop_best_iteration={early_stop_best_iteration}, prediction_iteration={prediction_iteration}"
        )
        np.save(args.output_dir / "catboost_oof_proba_partial.npy", oof.astype(np.float32))
        np.save(args.output_dir / "catboost_test_proba_partial.npy", test_pred.astype(np.float32))
        pd.DataFrame(fold_rows).to_csv(args.output_dir / "catboost_fold_scores_partial.csv", index=False)
        pd.DataFrame(diagnostic_rows).to_csv(args.output_dir / "catboost_training_diagnostics_partial.csv", index=False)

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

    report_params = {
        key: (value.__class__.__name__ if key == "eval_metric" and not isinstance(value, str) else value)
        for key, value in params.items()
    }
    report = {
        "seed": int(args.seed),
        "n_splits": N_SPLITS,
        "fold_limit": len(splits),
        "classes": classes,
        "feature_set": args.feature_set,
        "features": features,
        "cat_features": cat_features,
        "params": report_params,
        "early_stop_metric": args.early_stop_metric,
        "prediction_iteration_policy": args.prediction_iteration_policy,
        "use_best_model": bool(args.use_best_model),
        "fixed_iteration": int(args.fixed_iteration),
        "diagnostic_period": int(args.diagnostic_period),
        "diagnostic_train_sample": int(args.diagnostic_train_sample),
        "fold_scores": fold_rows,
        "covered_oof_balanced_accuracy": float(covered_score),
        "oof_balanced_accuracy": float(full_score) if full_score is not None else None,
        "best_iterations": best_iterations,
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
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
