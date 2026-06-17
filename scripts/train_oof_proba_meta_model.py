from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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
PURE_DIR = ARTIFACTS / "pure_model_ensemble"
OUT_DIR = ARTIFACTS / "oof_proba_meta_model"
SEED = 20260617
N_SPLITS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a leak-safe OOF/proba meta model. The model uses train-row OOF probabilities "
            "and matching test probabilities, plus advanced stellar features."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--feature-set", choices=["base", "advanced"], default="advanced")
    parser.add_argument("--fold-limit", type=int, default=N_SPLITS)
    parser.add_argument("--n-estimators", type=int, default=2600)
    parser.add_argument("--early-stopping-rounds", type=int, default=140)
    parser.add_argument("--bias-search-steps", type=int, default=2500)
    parser.add_argument("--log-period", type=int, default=50)
    parser.add_argument(
        "--write-even-if-worse",
        action="store_true",
        help="Write a submission even when OOF balanced accuracy is worse than the pure base model.",
    )
    return parser.parse_args()


def progress(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def entropy(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(proba, 1e-8, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for class_idx in range(n_classes):
        mask = y_true == class_idx
        if mask.any():
            recalls.append(float((y_pred[mask] == class_idx).mean()))
    return float(np.mean(recalls))


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, float]:
    out = {}
    for idx, cls in enumerate(classes):
        mask = y_true == idx
        out[cls] = float((y_pred[mask] == idx).mean()) if mask.any() else float("nan")
    return out


def load_pure_arrays(classes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    report_path = PURE_DIR / "pure_model_ensemble_report.json"
    if not report_path.exists():
        raise FileNotFoundError("Run scripts/build_pure_model_ensemble.py before this script.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["classes"] != classes:
        raise ValueError(f"Class order mismatch: {report['classes']} != {classes}")

    paths = {
        "cal_oof": PURE_DIR / "pure_model_ensemble_oof_proba.npy",
        "cal_test": PURE_DIR / "pure_model_ensemble_test_proba.npy",
        "raw_oof": PURE_DIR / "pure_model_ensemble_raw_oof_proba.npy",
        "raw_test": PURE_DIR / "pure_model_ensemble_raw_test_proba.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing pure model probability files:\n" + "\n".join(missing))
    return (
        np.load(paths["cal_oof"]).astype(np.float32),
        np.load(paths["cal_test"]).astype(np.float32),
        np.load(paths["raw_oof"]).astype(np.float32),
        np.load(paths["raw_test"]).astype(np.float32),
        report,
    )


def add_proba_features(
    features: pd.DataFrame,
    calibrated: np.ndarray,
    raw: np.ndarray,
    classes: list[str],
) -> pd.DataFrame:
    out = features.reset_index(drop=True).copy()
    for prefix, proba in [("cal", calibrated), ("raw", raw)]:
        sorted_probs = np.sort(proba, axis=1)
        out[f"{prefix}_pred_idx"] = proba.argmax(axis=1).astype("int16")
        out[f"{prefix}_top_prob"] = sorted_probs[:, -1]
        out[f"{prefix}_second_prob"] = sorted_probs[:, -2]
        out[f"{prefix}_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
        out[f"{prefix}_entropy"] = entropy(proba)
        for idx, cls in enumerate(classes):
            out[f"{prefix}_p_{cls}"] = proba[:, idx]

    for idx, cls in enumerate(classes):
        out[f"bias_delta_p_{cls}"] = calibrated[:, idx] - raw[:, idx]

    class_index = {cls: idx for idx, cls in enumerate(classes)}
    pairs = [("GALAXY", "STAR"), ("GALAXY", "QSO"), ("QSO", "STAR")]
    for left, right in pairs:
        if left in class_index and right in class_index:
            li = class_index[left]
            ri = class_index[right]
            out[f"cal_margin_{left}_vs_{right}"] = calibrated[:, li] - calibrated[:, ri]
            out[f"raw_margin_{left}_vs_{right}"] = raw[:, li] - raw[:, ri]

    out["cal_raw_pred_agree"] = (calibrated.argmax(axis=1) == raw.argmax(axis=1)).astype("int8")
    return out


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(np.float32)
    return out


def params(n_classes: int, n_estimators: int) -> dict:
    return {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.025,
        "num_leaves": 48,
        "max_depth": 8,
        "min_child_samples": 180,
        "subsample": 0.88,
        "subsample_freq": 1,
        "colsample_bytree": 0.88,
        "reg_alpha": 0.25,
        "reg_lambda": 6.5,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_estimators": n_estimators,
        "n_jobs": -1,
        "verbosity": -1,
    }


def optimize_class_bias(
    y: np.ndarray,
    proba: np.ndarray,
    n_classes: int,
    steps: int,
) -> tuple[float, list[float]]:
    base_pred = proba.argmax(axis=1)
    best_score = balanced_accuracy(y, base_pred, n_classes)
    best_bias = np.ones(n_classes, dtype=np.float64)
    if steps <= 0:
        return best_score, best_bias.tolist()

    rng = np.random.default_rng(SEED)
    candidates = [np.ones(n_classes, dtype=np.float64)]
    for scale in [0.015, 0.03, 0.06, 0.10]:
        noise = rng.normal(0.0, scale, size=(max(1, steps // 4), n_classes))
        candidates.extend(np.exp(noise))

    for bias in candidates[:steps]:
        pred = (proba * bias.reshape(1, -1)).argmax(axis=1)
        score = balanced_accuracy(y, pred, n_classes)
        if score > best_score:
            best_score = score
            best_bias = bias.copy()
    best_bias = best_bias / best_bias.mean()
    return float(best_score), best_bias.tolist()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    progress("Loading train/test data")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    progress(f"Building {args.feature_set} feature matrix")
    x_base, y_raw, x_test_base, base_features = make_xy(train, test, feature_set=args.feature_set)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw.astype(str))
    classes = encoder.classes_.tolist()

    progress("Loading pure model OOF/test probabilities")
    cal_oof, cal_test, raw_oof, raw_test, pure_report = load_pure_arrays(classes)
    if cal_oof.shape[0] != len(train) or cal_test.shape[0] != len(test):
        raise ValueError("Pure probability row count differs from train/test data.")

    progress("Building meta features from advanced features and probabilities")
    x = clean_frame(add_proba_features(x_base, cal_oof, raw_oof, classes))
    x_test = clean_frame(add_proba_features(x_test_base, cal_test, raw_test, classes))
    progress(f"Meta train shape={x.shape}, test shape={x_test.shape}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(cv.split(x, y))[: args.fold_limit]
    if not splits:
        raise ValueError("--fold-limit must be at least 1.")

    model_params = params(len(classes), args.n_estimators)
    oof = np.zeros((len(train), len(classes)), dtype=np.float32)
    covered = np.zeros(len(train), dtype=bool)
    test_proba = np.zeros((len(test), len(classes)), dtype=np.float32)
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        progress(
            f"Starting fold {fold}/{len(splits)} "
            f"(train_rows={len(tr_idx)}, valid_rows={len(va_idx)})"
        )
        model = lgb.LGBMClassifier(**model_params)
        model.fit(
            x.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(x.iloc[va_idx], y[va_idx])],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(args.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(args.log_period),
            ],
        )
        progress(f"Finished fold {fold}; scoring validation predictions")
        valid_proba = model.predict_proba(x.iloc[va_idx])
        oof[va_idx] = valid_proba
        covered[va_idx] = True
        test_proba += model.predict_proba(x_test) / len(splits)

        base_pred = cal_oof[va_idx].argmax(axis=1)
        meta_pred = valid_proba.argmax(axis=1)
        base_score = balanced_accuracy_score(y[va_idx], base_pred)
        meta_score = balanced_accuracy_score(y[va_idx], meta_pred)
        row = {
            "fold": fold,
            "base_score": float(base_score),
            "meta_score": float(meta_score),
            "delta": float(meta_score - base_score),
            "best_iteration": int(model.best_iteration_ or args.n_estimators),
            "base_recalls": class_recalls(y[va_idx], base_pred, classes),
            "meta_recalls": class_recalls(y[va_idx], meta_pred, classes),
        }
        fold_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    progress("Optimizing class bias on covered OOF predictions")
    base_score_covered = balanced_accuracy(y[covered], cal_oof[covered].argmax(axis=1), len(classes))
    meta_score_covered = balanced_accuracy(y[covered], oof[covered].argmax(axis=1), len(classes))
    bias_score, class_bias = optimize_class_bias(
        y[covered],
        oof[covered],
        len(classes),
        args.bias_search_steps,
    )

    report = {
        "purpose": "Leak-safe OOF/proba meta model with advanced feature engineering.",
        "feature_set": args.feature_set,
        "classes": classes,
        "fold_limit": len(splits),
        "covered_rows": int(covered.sum()),
        "feature_count": int(x.shape[1]),
        "base_feature_count": len(base_features),
        "base_score_on_covered_rows": base_score_covered,
        "meta_score_on_covered_rows": meta_score_covered,
        "meta_delta_on_covered_rows": meta_score_covered - base_score_covered,
        "bias_optimized_score_on_covered_rows": bias_score,
        "bias_optimized_delta_on_covered_rows": bias_score - base_score_covered,
        "class_bias": dict(zip(classes, class_bias)),
        "folds": fold_rows,
        "params": model_params,
        "pure_model_score": pure_report.get("best_oof_balanced_accuracy"),
    }

    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_scores.csv", index=False)
    np.save(args.output_dir / "meta_oof_proba.npy", oof)
    np.save(args.output_dir / "meta_test_proba.npy", test_proba)

    should_write_submission = covered.all() and (
        args.write_even_if_worse or bias_score > base_score_covered
    )
    report["accepted_as_candidate"] = bool(should_write_submission)
    report["candidate_rule"] = (
        "Write a submission only when full OOF bias-optimized balanced accuracy beats the pure base model."
    )

    if should_write_submission:
        test_pred = (test_proba * np.array(class_bias).reshape(1, -1)).argmax(axis=1)
        submission = sample.copy()
        submission["class"] = encoder.inverse_transform(test_pred)
        submission_path = args.output_dir / "meta_model_submission.csv"
        submission.to_csv(submission_path, index=False)
        report["submission_path"] = str(submission_path.relative_to(ROOT))
        report["submission_class_counts"] = submission["class"].value_counts().sort_index().to_dict()
    else:
        stale_submission = args.output_dir / "meta_model_submission.csv"
        if stale_submission.exists() and not args.write_even_if_worse:
            stale_submission.unlink()
        report["submission_path"] = None
        if not covered.all():
            report["note"] = "Fold limit did not cover all rows; submission was not written."
        else:
            report["note"] = "Meta model did not beat the pure base OOF score; submission was intentionally not written."

    (args.output_dir / "meta_model_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    progress("Wrote meta model report")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
