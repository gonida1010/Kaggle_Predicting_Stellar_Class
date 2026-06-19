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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    import xgboost as xgb
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    raise SystemExit(
        "xgboost is not installed in this virtualenv. Install it first with: "
        "python -m pip install xgboost"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import make_xy  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "xgboost_cv"
SEED = 20260619
N_SPLITS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a fold-safe XGBoost multiclass model and save OOF/test probabilities. "
            "No public leaderboard labels or submission-bank CSVs are used."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--feature-set", choices=["base", "advanced", "realmlp"], default="advanced")
    parser.add_argument("--fold-limit", type=int, default=N_SPLITS)
    parser.add_argument("--num-boost-round", type=int, default=4200)
    parser.add_argument("--early-stopping-rounds", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.032)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=8.0)
    parser.add_argument("--subsample", type=float, default=0.88)
    parser.add_argument("--colsample-bytree", type=float, default=0.86)
    parser.add_argument("--reg-alpha", type=float, default=0.08)
    parser.add_argument("--reg-lambda", type=float, default=5.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-period", type=int, default=200)
    parser.add_argument(
        "--early-stop-metric",
        choices=["logloss", "valid-bac"],
        default="logloss",
        help="Metric used by XGBoost early stopping. valid-bac stops on validation balanced accuracy.",
    )
    parser.add_argument(
        "--diagnostic-period",
        type=int,
        default=100,
        help="Iteration spacing for train/valid diagnostics and plots. 0 disables diagnostic scoring.",
    )
    parser.add_argument(
        "--diagnostic-train-sample",
        type=int,
        default=50000,
        help="Rows sampled from each train fold for train balanced-accuracy diagnostics.",
    )
    parser.add_argument(
        "--prediction-iteration-policy",
        choices=["early-stop-best", "logloss-best", "valid-bac-best", "fixed"],
        default="early-stop-best",
        help=(
            "Iteration used for OOF/test probabilities. early-stop-best uses XGBoost best_iteration from "
            "--early-stop-metric; logloss-best is kept as a backward-compatible alias; "
            "valid-bac-best uses the best diagnostic valid balanced accuracy; fixed uses --fixed-iteration."
        ),
    )
    parser.add_argument(
        "--fixed-iteration",
        type=int,
        default=0,
        help="Number of boosting rounds to use when --prediction-iteration-policy fixed.",
    )
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


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    class_values = np.unique(y)
    weights = compute_class_weight("balanced", classes=class_values, y=y)
    weight_map = dict(zip(class_values, weights))
    return np.array([weight_map[value] for value in y], dtype=np.float32)


def predict_proba(model: xgb.Booster, matrix: xgb.DMatrix, n_classes: int, iteration_end: int | None = None) -> np.ndarray:
    if iteration_end is None:
        best_iteration = int(getattr(model, "best_iteration", 0))
        iteration_end = best_iteration + 1 if best_iteration >= 0 else 0
    pred = model.predict(matrix, iteration_range=(0, iteration_end))
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 1:
        pred = pred.reshape(-1, n_classes)
    return normalize_probs(pred)


def diagnostic_iterations(num_boost_round: int, best_iteration: int, period: int) -> list[int]:
    if period <= 0:
        return []
    max_iteration = max(1, min(num_boost_round, best_iteration + 1))
    points = set(range(1, max_iteration + 1, period))
    points.add(max_iteration)
    return sorted(points)


def diagnostic_train_indices(indices: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0 or len(indices) <= sample_size:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=sample_size, replace=False))


def choose_prediction_iteration(
    args: argparse.Namespace,
    fold_diag: list[dict],
    early_stop_best_iteration: int,
) -> int:
    if args.prediction_iteration_policy == "fixed":
        if args.fixed_iteration <= 0:
            raise ValueError("--fixed-iteration must be positive when using --prediction-iteration-policy fixed.")
        return int(args.fixed_iteration)
    if args.prediction_iteration_policy == "valid-bac-best" and fold_diag:
        best = max(fold_diag, key=lambda row: row["valid_balanced_accuracy"])
        return int(best["iteration"])
    return int(early_stop_best_iteration) + 1


def write_matplotlib_diagnostic_plots(grouped: pd.DataFrame, output_dir: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception:
        return False

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
    ax.set_title("XGBoost Train vs Valid Logloss")
    ax.set_xlabel("Boosting rounds")
    ax.set_ylabel("Multi logloss")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "xgboost_logloss_curve.svg", format="svg")
    fig.savefig(output_dir / "xgboost_logloss_curve.png", format="png")
    plt.close(fig)

    zoomed = grouped[grouped["iteration"] >= min(101, int(grouped["iteration"].max()))]
    if len(zoomed) >= 2:
        fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=150)
        ax.plot(zoomed["iteration"], zoomed["train_mlogloss"], label="train mlogloss", color="#d92d20", linewidth=2.2)
        ax.plot(zoomed["iteration"], zoomed["valid_mlogloss"], label="valid mlogloss", color="#1f5eff", linewidth=2.2)
        ax.set_title("XGBoost Train vs Valid Logloss Zoom")
        ax.set_xlabel("Boosting rounds")
        ax.set_ylabel("Multi logloss")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / "xgboost_logloss_curve_zoom.svg", format="svg")
        fig.savefig(output_dir / "xgboost_logloss_curve_zoom.png", format="png")
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
    ax.set_title("XGBoost Train Sample vs Valid Balanced Accuracy")
    ax.set_xlabel("Boosting rounds")
    ax.set_ylabel("Balanced accuracy")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "xgboost_balanced_accuracy_curve.svg", format="svg")
    fig.savefig(output_dir / "xgboost_balanced_accuracy_curve.png", format="png")
    plt.close(fig)
    return True


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
    if write_matplotlib_diagnostic_plots(grouped, output_dir):
        return

    def render_svg(
        path: Path,
        title: str,
        ylabel: str,
        series: list[tuple[str, str, pd.Series]],
        best_iteration: int | None = None,
    ) -> None:
        width, height = 1200, 680
        left, right, top, bottom = 92, 40, 72, 86
        plot_w = width - left - right
        plot_h = height - top - bottom
        x_vals = grouped["iteration"].to_numpy(dtype=float)
        y_vals = np.concatenate([values.to_numpy(dtype=float) for _, _, values in series])
        x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
        y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
        if x_min == x_max:
            x_max = x_min + 1.0
        if y_min == y_max:
            y_max = y_min + 1.0
        y_pad = (y_max - y_min) * 0.08
        y_min -= y_pad
        y_max += y_pad

        def sx(x: float) -> float:
            return left + (x - x_min) / (x_max - x_min) * plot_w

        def sy(y: float) -> float:
            return top + (y_max - y) / (y_max - y_min) * plot_h

        def polyline(values: pd.Series) -> str:
            points = " ".join(f"{sx(float(x)):.2f},{sy(float(y)):.2f}" for x, y in zip(x_vals, values))
            return points

        x_ticks = np.linspace(x_min, x_max, 7)
        y_ticks = np.linspace(y_min, y_max, 6)
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<text x="{left}" y="36" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">{title}</text>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.2"/>',
        ]
        for tick in x_ticks:
            x = sx(float(tick))
            parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e5e7eb"/>')
            parts.append(
                f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="13" fill="#374151">{int(round(tick))}</text>'
            )
        for tick in y_ticks:
            y = sy(float(tick))
            parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
            parts.append(
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-family="Arial, sans-serif" font-size="13" fill="#374151">{tick:.5f}</text>'
            )
        if best_iteration is not None:
            x = sx(float(best_iteration))
            parts.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
                f'stroke="#111827" stroke-width="1.4" stroke-dasharray="7 5"/>'
            )
            parts.append(
                f'<text x="{x + 8:.2f}" y="{top + 20}" font-family="Arial, sans-serif" '
                f'font-size="13" fill="#111827">best valid BAC @ {best_iteration}</text>'
            )
        legend_x = left + plot_w - 240
        legend_y = top + 18
        for idx, (label, color, values) in enumerate(series):
            y = legend_y + idx * 24
            parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 34}" y2="{y}" stroke="{color}" stroke-width="3"/>')
            parts.append(
                f'<text x="{legend_x + 44}" y="{y + 5}" font-family="Arial, sans-serif" '
                f'font-size="14" fill="#111827">{label}</text>'
            )
            parts.append(
                f'<polyline points="{polyline(values)}" fill="none" stroke="{color}" '
                f'stroke-width="2.8" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        parts.append(
            f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="15" fill="#111827">Boosting rounds</text>'
        )
        parts.append(
            f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" '
            f'font-family="Arial, sans-serif" font-size="15" fill="#111827">{ylabel}</text>'
        )
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")

    render_svg(
        output_dir / "xgboost_logloss_curve.svg",
        "XGBoost Train vs Valid Logloss",
        "Multi logloss",
        [
            ("train mlogloss", "#d92d20", grouped["train_mlogloss"]),
            ("valid mlogloss", "#1f5eff", grouped["valid_mlogloss"]),
        ],
    )
    best_row = grouped.loc[grouped["valid_balanced_accuracy"].idxmax()]
    render_svg(
        output_dir / "xgboost_balanced_accuracy_curve.svg",
        "XGBoost Train Sample vs Valid Balanced Accuracy",
        "Balanced accuracy",
        [
            ("train BAC sample", "#d92d20", grouped["train_balanced_accuracy_sample"]),
            ("valid BAC", "#1f5eff", grouped["valid_balanced_accuracy"]),
        ],
        best_iteration=int(best_row["iteration"]),
    )


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

    def xgb_balanced_accuracy_metric(predt: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[str, float]:
        label = dmatrix.get_label().astype(np.int64)
        pred = np.asarray(predt, dtype=np.float32)
        if pred.ndim == 1:
            pred = pred.reshape(label.shape[0], n_classes)
        return "balanced_accuracy", float(balanced_accuracy_score(label, pred.argmax(axis=1)))

    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": n_classes,
        "eta": float(args.learning_rate),
        "max_depth": int(args.max_depth),
        "min_child_weight": float(args.min_child_weight),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "alpha": float(args.reg_alpha),
        "lambda": float(args.reg_lambda),
        "max_bin": int(args.max_bin),
        "tree_method": "hist",
        "seed": int(args.seed),
        "nthread": -1,
    }

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)
    splits = list(cv.split(x, y))[: int(args.fold_limit)]
    if not splits:
        raise ValueError("--fold-limit must be at least 1")
    if len(splits) != N_SPLITS:
        progress(f"Using fold_limit={len(splits)}; this is a partial OOF smoke/screen run.")

    oof = np.zeros((len(x), n_classes), dtype=np.float32)
    test_pred = np.zeros((len(x_test), n_classes), dtype=np.float32)
    fold_rows = []
    diagnostic_rows = []
    test_matrix = xgb.DMatrix(x_test, feature_names=features)

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        progress(f"Training XGBoost fold {fold}/{len(splits)}")
        diag_tr_idx = diagnostic_train_indices(tr_idx, int(args.diagnostic_train_sample), int(args.seed) + fold)
        train_matrix = xgb.DMatrix(
            x.iloc[tr_idx],
            label=y[tr_idx],
            weight=balanced_sample_weight(y[tr_idx]),
            feature_names=features,
        )
        valid_matrix = xgb.DMatrix(x.iloc[va_idx], label=y[va_idx], feature_names=features)
        diag_train_matrix = xgb.DMatrix(x.iloc[diag_tr_idx], label=y[diag_tr_idx], feature_names=features)
        evals_result: dict[str, dict[str, list[float]]] = {}
        train_kwargs = {
            "params": params,
            "dtrain": train_matrix,
            "num_boost_round": int(args.num_boost_round),
            "evals": [(train_matrix, "train"), (valid_matrix, "valid")],
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "verbose_eval": int(args.log_period),
            "evals_result": evals_result,
        }
        if args.early_stop_metric == "valid-bac":
            train_kwargs["custom_metric"] = xgb_balanced_accuracy_metric
            train_kwargs["maximize"] = True
        model = xgb.train(**train_kwargs)

        early_stop_best_iteration = int(model.best_iteration)
        fold_diag = []
        for iteration in diagnostic_iterations(
            int(args.num_boost_round), early_stop_best_iteration, int(args.diagnostic_period)
        ):
            train_diag_pred = predict_proba(model, diag_train_matrix, n_classes, iteration_end=iteration)
            valid_diag_pred = predict_proba(model, valid_matrix, n_classes, iteration_end=iteration)
            row = {
                "fold": fold,
                "iteration": int(iteration),
                "train_mlogloss": float(evals_result["train"]["mlogloss"][iteration - 1]),
                "valid_mlogloss": float(evals_result["valid"]["mlogloss"][iteration - 1]),
                "train_balanced_accuracy_sample": float(
                    balanced_accuracy_score(y[diag_tr_idx], train_diag_pred.argmax(axis=1))
                ),
                "valid_balanced_accuracy": float(balanced_accuracy_score(y[va_idx], valid_diag_pred.argmax(axis=1))),
            }
            diagnostic_rows.append(row)
            fold_diag.append(row)

        prediction_iteration = choose_prediction_iteration(args, fold_diag, early_stop_best_iteration)
        prediction_iteration = min(prediction_iteration, early_stop_best_iteration + 1)
        va_pred = predict_proba(model, valid_matrix, n_classes, iteration_end=prediction_iteration)
        oof[va_idx] = va_pred
        pred_label = va_pred.argmax(axis=1)
        score = balanced_accuracy_score(y[va_idx], pred_label)
        diagnostic_best = max(fold_diag, key=lambda row: row["valid_balanced_accuracy"]) if fold_diag else None
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
        test_pred += predict_proba(model, test_matrix, n_classes, iteration_end=prediction_iteration) / len(splits)
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
    submission_path = args.output_dir / "xgboost_submission.csv"
    submission.to_csv(submission_path, index=False)
    np.save(args.output_dir / "xgboost_oof_proba.npy", oof.astype(np.float32))
    np.save(args.output_dir / "xgboost_test_proba.npy", test_pred.astype(np.float32))
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_scores.csv", index=False)
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(args.output_dir / "xgboost_training_diagnostics.csv", index=False)
    write_diagnostic_plots(diagnostics, args.output_dir)

    report = {
        "purpose": "Fold-safe XGBoost OOF/test probability generator. No public submission CSV is used.",
        "classes": classes,
        "feature_set": args.feature_set,
        "feature_count": len(features),
        "params": params,
        "num_boost_round": int(args.num_boost_round),
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "early_stop_metric": args.early_stop_metric,
        "prediction_iteration_policy": args.prediction_iteration_policy,
        "fixed_iteration": int(args.fixed_iteration),
        "diagnostic_period": int(args.diagnostic_period),
        "diagnostic_train_sample": int(args.diagnostic_train_sample),
        "fold_limit": len(splits),
        "fold_scores": fold_rows,
        "covered_oof_balanced_accuracy": float(covered_score),
        "full_oof_balanced_accuracy": float(full_score) if full_score is not None else None,
        "covered_oof_class_recalls": class_recalls(y[covered], oof_pred[covered], classes),
        "submission_path": str(submission_path.relative_to(ROOT)),
        "submission_class_share": submission["class"].value_counts(normalize=True).sort_index().to_dict(),
        "outputs": [
            "xgboost_submission.csv",
            "xgboost_oof_proba.npy",
            "xgboost_test_proba.npy",
            "fold_scores.csv",
            "xgboost_training_diagnostics.csv",
            "xgboost_logloss_curve.svg",
            "xgboost_logloss_curve.png",
            "xgboost_logloss_curve_zoom.svg",
            "xgboost_logloss_curve_zoom.png",
            "xgboost_balanced_accuracy_curve.svg",
            "xgboost_balanced_accuracy_curve.png",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
