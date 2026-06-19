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
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    import xgboost as xgb
except ModuleNotFoundError as exc:
    raise SystemExit("xgboost is not installed in this virtualenv. Install it first with: python -m pip install xgboost") from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import make_xy  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "ovr_xgboost_realmlp_features"
SEED = 20260620
N_SPLITS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train fold-safe one-vs-rest XGBoost specialists and save OOF/test probabilities, "
            "fold metrics, training curves, and a JSON report. No public labels or submission-bank CSVs are used."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--feature-set", choices=["base", "advanced", "realmlp"], default="realmlp")
    parser.add_argument("--fold-limit", type=int, default=N_SPLITS)
    parser.add_argument("--num-boost-round", type=int, default=9000)
    parser.add_argument("--early-stopping-rounds", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-child-weight", type=float, default=9.0)
    parser.add_argument("--subsample", type=float, default=0.90)
    parser.add_argument("--colsample-bytree", type=float, default=0.88)
    parser.add_argument("--reg-alpha", type=float, default=0.12)
    parser.add_argument("--reg-lambda", type=float, default=8.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-period", type=int, default=250)
    parser.add_argument(
        "--class-threshold",
        type=float,
        default=0.5,
        help="Threshold used only for binary balanced-accuracy diagnostics.",
    )
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_probs(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return (proba / row_sum).astype(np.float32)


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


def binary_sample_weight(y_binary: np.ndarray) -> np.ndarray:
    class_values = np.unique(y_binary)
    weights = compute_class_weight("balanced", classes=class_values, y=y_binary)
    weight_map = dict(zip(class_values, weights))
    return np.array([weight_map[value] for value in y_binary], dtype=np.float32)


def predict_binary(model: xgb.Booster, matrix: xgb.DMatrix, iteration_end: int | None = None) -> np.ndarray:
    if iteration_end is None:
        best_iteration = int(getattr(model, "best_iteration", 0))
        iteration_end = best_iteration + 1 if best_iteration >= 0 else 0
    pred = model.predict(matrix, iteration_range=(0, iteration_end))
    return np.asarray(pred, dtype=np.float32)


def write_plots(output_dir: Path, binary_rows: pd.DataFrame, fold_rows: pd.DataFrame, loss_rows: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception as exc:
        progress(f"Skipping matplotlib plots: {exc}")
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

    fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=150)
    pivot = binary_rows.pivot(index="fold", columns="class", values="binary_balanced_accuracy")
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("One-vs-Rest XGBoost Binary Balanced Accuracy")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Binary balanced accuracy")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "ovr_xgboost_binary_bac_by_fold.svg", format="svg")
    fig.savefig(output_dir / "ovr_xgboost_binary_bac_by_fold.png", format="png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.8, 5.8), dpi=150)
    ax.plot(fold_rows["fold"], fold_rows["balanced_accuracy"], marker="o", linewidth=2.2, color="#1f5eff")
    ax.set_title("One-vs-Rest XGBoost Multiclass OOF by Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Balanced accuracy")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "ovr_xgboost_multiclass_bac_by_fold.svg", format="svg")
    fig.savefig(output_dir / "ovr_xgboost_multiclass_bac_by_fold.png", format="png")
    plt.close(fig)

    if not loss_rows.empty:
        grouped = loss_rows.groupby(["class", "iteration"], as_index=False)[["train_logloss", "valid_logloss"]].mean()
        fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=150)
        for class_name, sub in grouped.groupby("class"):
            ax.plot(sub["iteration"], sub["valid_logloss"], linewidth=1.8, label=f"{class_name} valid")
        ax.set_title("One-vs-Rest XGBoost Validation Logloss")
        ax.set_xlabel("Boosting rounds")
        ax.set_ylabel("Binary logloss")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / "ovr_xgboost_valid_logloss_by_class.svg", format="svg")
        fig.savefig(output_dir / "ovr_xgboost_valid_logloss_by_class.png", format="png")
        plt.close(fig)


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

    progress(f"Feature count={len(features)} rows train={len(x)} test={len(x_test)} classes={classes}")
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
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
        "verbosity": 1,
        "nthread": -1,
    }

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)
    splits = list(cv.split(x, y))[: int(args.fold_limit)]
    if not splits:
        raise ValueError("--fold-limit must be at least 1")

    oof_binary = np.zeros((len(x), n_classes), dtype=np.float32)
    test_binary = np.zeros((len(x_test), n_classes), dtype=np.float32)
    covered_valid = np.zeros(len(x), dtype=bool)
    fold_rows: list[dict] = []
    binary_rows: list[dict] = []
    loss_rows: list[dict] = []
    dtest = xgb.DMatrix(x_test)

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        x_tr = x.iloc[tr_idx]
        x_va = x.iloc[va_idx]
        y_tr_full = y[tr_idx]
        y_va_full = y[va_idx]
        fold_test_binary = np.zeros((len(x_test), n_classes), dtype=np.float32)
        covered_valid[va_idx] = True

        for class_idx, class_name in enumerate(classes):
            y_tr = (y_tr_full == class_idx).astype(np.int32)
            y_va = (y_va_full == class_idx).astype(np.int32)
            dtrain = xgb.DMatrix(x_tr, label=y_tr, weight=binary_sample_weight(y_tr))
            dvalid = xgb.DMatrix(x_va, label=y_va)
            evals_result: dict = {}
            progress(
                f"Training fold {fold}/{len(splits)} one-vs-rest class={class_name} "
                f"pos_train={int(y_tr.sum())} pos_valid={int(y_va.sum())}"
            )
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=int(args.num_boost_round),
                evals=[(dtrain, "train"), (dvalid, "valid")],
                early_stopping_rounds=int(args.early_stopping_rounds),
                verbose_eval=int(args.log_period),
                evals_result=evals_result,
            )
            best_iteration = int(getattr(model, "best_iteration", args.num_boost_round - 1))
            prediction_iteration = best_iteration + 1
            valid_prob = predict_binary(model, dvalid, prediction_iteration)
            test_prob = predict_binary(model, dtest, prediction_iteration)
            oof_binary[va_idx, class_idx] = valid_prob.astype(np.float32)
            fold_test_binary[:, class_idx] = test_prob.astype(np.float32)

            binary_pred = (valid_prob >= args.class_threshold).astype(np.int32)
            binary_score = balanced_accuracy_score(y_va, binary_pred)
            binary_logloss = log_loss(y_va, valid_prob, labels=[0, 1])
            binary_rows.append(
                {
                    "fold": fold,
                    "class": class_name,
                    "binary_balanced_accuracy": float(binary_score),
                    "valid_logloss": float(binary_logloss),
                    "best_iteration": best_iteration,
                    "prediction_iteration": prediction_iteration,
                }
            )
            for iteration in range(0, prediction_iteration, max(1, int(args.log_period))):
                if iteration < len(evals_result["valid"]["logloss"]):
                    loss_rows.append(
                        {
                            "fold": fold,
                            "class": class_name,
                            "iteration": iteration + 1,
                            "train_logloss": float(evals_result["train"]["logloss"][iteration]),
                            "valid_logloss": float(evals_result["valid"]["logloss"][iteration]),
                        }
                    )
            if prediction_iteration - 1 < len(evals_result["valid"]["logloss"]):
                idx = prediction_iteration - 1
                loss_rows.append(
                    {
                        "fold": fold,
                        "class": class_name,
                        "iteration": prediction_iteration,
                        "train_logloss": float(evals_result["train"]["logloss"][idx]),
                        "valid_logloss": float(evals_result["valid"]["logloss"][idx]),
                    }
                )
            progress(
                f"fold {fold} class={class_name} binary_BAC={binary_score:.6f} "
                f"valid_logloss={binary_logloss:.6f} best_iteration={best_iteration}"
            )

        fold_test_binary = normalize_probs(fold_test_binary)
        test_binary += fold_test_binary / len(splits)
        fold_oof = normalize_probs(oof_binary[va_idx])
        fold_pred = fold_oof.argmax(axis=1)
        fold_score = balanced_accuracy(y_va_full, fold_pred, n_classes)
        fold_rows.append(
            {
                "fold": fold,
                "balanced_accuracy": float(fold_score),
                "class_recalls": class_recalls(y_va_full, fold_pred, classes),
            }
        )
        progress(f"fold {fold} multiclass_BAC={fold_score:.6f}")

    oof_proba = normalize_probs(oof_binary)
    test_proba = normalize_probs(test_binary)
    oof_pred = oof_proba.argmax(axis=1)
    covered_oof_score = balanced_accuracy(y[covered_valid], oof_pred[covered_valid], n_classes)
    full_oof_score = balanced_accuracy(y, oof_pred, n_classes)
    progress(f"covered OOF balanced_accuracy={covered_oof_score:.6f}")
    if covered_valid.all():
        progress(f"full OOF balanced_accuracy={full_oof_score:.6f}")

    submission = sample.copy()
    submission["class"] = np.array(classes)[test_proba.argmax(axis=1)]
    submission_path = args.output_dir / "ovr_xgboost_submission.csv"
    submission.to_csv(submission_path, index=False)
    np.save(args.output_dir / "ovr_xgboost_oof_proba.npy", oof_proba.astype(np.float32))
    np.save(args.output_dir / "ovr_xgboost_test_proba.npy", test_proba.astype(np.float32))
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_scores.csv", index=False)
    pd.DataFrame(binary_rows).to_csv(args.output_dir / "binary_fold_scores.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(args.output_dir / "training_logloss.csv", index=False)
    write_plots(args.output_dir, pd.DataFrame(binary_rows), pd.DataFrame(fold_rows), pd.DataFrame(loss_rows))

    report = {
        "purpose": "One-vs-rest XGBoost specialist bank. No public submission CSV is used.",
        "classes": classes,
        "feature_set": args.feature_set,
        "feature_count": len(features),
        "params": params,
        "fold_limit": len(splits),
        "fold_scores": fold_rows,
        "binary_fold_scores": binary_rows,
        "covered_valid_rows": int(covered_valid.sum()),
        "covered_oof_balanced_accuracy": float(covered_oof_score),
        "full_oof_balanced_accuracy": float(full_oof_score) if covered_valid.all() else None,
        "covered_oof_class_recalls": class_recalls(y[covered_valid], oof_pred[covered_valid], classes),
        "submission_path": str(submission_path.relative_to(ROOT)),
        "submission_class_share": submission["class"].value_counts(normalize=True).sort_index().to_dict(),
        "outputs": [
            "ovr_xgboost_submission.csv",
            "ovr_xgboost_oof_proba.npy",
            "ovr_xgboost_test_proba.npy",
            "fold_scores.csv",
            "binary_fold_scores.csv",
            "training_logloss.csv",
            "ovr_xgboost_binary_bac_by_fold.svg",
            "ovr_xgboost_binary_bac_by_fold.png",
            "ovr_xgboost_multiclass_bac_by_fold.svg",
            "ovr_xgboost_multiclass_bac_by_fold.png",
            "ovr_xgboost_valid_logloss_by_class.svg",
            "ovr_xgboost_valid_logloss_by_class.png",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
