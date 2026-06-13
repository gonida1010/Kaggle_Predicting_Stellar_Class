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
OUT_DIR = ARTIFACTS / "meta_stacker"
SEED = 20260612
N_SPLITS = 5
DEFAULT_MODELS = ["lgbm", "catboost"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a leak-safe second-stage stacker from base-model OOF probabilities and stable stellar features."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Base model prefixes with *_oof_proba.npy and *_test_proba.npy artifacts.",
    )
    parser.add_argument(
        "--fold-limit",
        type=int,
        default=N_SPLITS,
        help="Number of meta CV folds to run. Use 1 for quick screening, 5 for final artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory where stacker artifacts are written.",
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


def load_model_artifacts(model: str) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    report_path = ARTIFACTS / f"{model}_baseline_report.json"
    oof_path = ARTIFACTS / f"{model}_oof_proba.npy"
    test_path = ARTIFACTS / f"{model}_test_proba.npy"
    if not report_path.exists() or not oof_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing artifacts for {model}. Run its CV training script first.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report["classes"], np.load(oof_path), np.load(test_path), report


def entropy(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(proba, 1e-8, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def add_probability_features(
    out: pd.DataFrame,
    model: str,
    proba: np.ndarray,
    classes: list[str],
) -> pd.DataFrame:
    for idx, cls in enumerate(classes):
        out[f"{model}_p_{cls}"] = proba[:, idx]
    sorted_probs = np.sort(proba, axis=1)
    out[f"{model}_pred_idx"] = proba.argmax(axis=1).astype("int16")
    out[f"{model}_top_prob"] = sorted_probs[:, -1]
    out[f"{model}_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    out[f"{model}_entropy"] = entropy(proba)
    return out


def build_meta_features(
    base_features: pd.DataFrame,
    proba_by_model: dict[str, np.ndarray],
    classes: list[str],
) -> pd.DataFrame:
    out = base_features.reset_index(drop=True).copy()
    stacked = np.stack([proba_by_model[model] for model in proba_by_model], axis=0)
    for model, proba in proba_by_model.items():
        out = add_probability_features(out, model, proba, classes)
    mean_proba = stacked.mean(axis=0)
    std_proba = stacked.std(axis=0)
    for idx, cls in enumerate(classes):
        out[f"mean_p_{cls}"] = mean_proba[:, idx]
        out[f"std_p_{cls}"] = std_proba[:, idx]
    if len(proba_by_model) >= 2:
        preds = [proba.argmax(axis=1) for proba in proba_by_model.values()]
        out["base_models_agree"] = np.all(np.stack(preds, axis=1) == preds[0].reshape(-1, 1), axis=1).astype("int8")
    return out


def params(n_classes: int) -> dict:
    return {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.025,
        "num_leaves": 40,
        "max_depth": 7,
        "min_child_samples": 180,
        "subsample": 0.90,
        "subsample_freq": 1,
        "colsample_bytree": 0.92,
        "reg_alpha": 0.30,
        "reg_lambda": 5.0,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_estimators": 2600,
        "n_jobs": -1,
        "verbosity": -1,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = list(dict.fromkeys(args.models))
    if not models:
        raise ValueError("At least one base model prefix is required.")

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    x_base, y_raw, x_test_base, base_feature_names = make_xy(train, test)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    classes = encoder.classes_.tolist()

    oof_by_model: dict[str, np.ndarray] = {}
    test_by_model: dict[str, np.ndarray] = {}
    source_reports: dict[str, dict] = {}
    for model in models:
        model_classes, oof, test_pred, report = load_model_artifacts(model)
        if model_classes != classes:
            raise ValueError(f"{model} class order differs: {model_classes} != {classes}")
        oof_by_model[model] = oof.astype(np.float32)
        test_by_model[model] = test_pred.astype(np.float32)
        source_reports[model] = report

    x_meta = build_meta_features(x_base, oof_by_model, classes)
    x_test_meta = build_meta_features(x_test_base, test_by_model, classes)
    feature_names = x_meta.columns.tolist()

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(cv.split(x_meta, y))[: args.fold_limit]
    if not splits:
        raise ValueError("--fold-limit must be at least 1.")

    model_params = params(len(classes))
    oof = np.zeros((len(x_meta), len(classes)), dtype=np.float32)
    test_pred = np.zeros((len(x_test_meta), len(classes)), dtype=np.float32)
    fold_scores = []
    fold_recalls = []
    best_iterations = []

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        model = lgb.LGBMClassifier(**model_params)
        model.fit(
            x_meta.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(x_meta.iloc[va_idx], y[va_idx])],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(120, verbose=False),
                lgb.log_evaluation(250),
            ],
        )
        va_pred = model.predict_proba(x_meta.iloc[va_idx])
        oof[va_idx] = va_pred
        pred_label = va_pred.argmax(axis=1)
        score = balanced_accuracy_score(y[va_idx], pred_label)
        recalls = class_recalls(y[va_idx], pred_label, classes)
        fold_scores.append(float(score))
        fold_recalls.append(recalls)
        best_iterations.append(int(model.best_iteration_ or model_params["n_estimators"]))
        test_pred += model.predict_proba(x_test_meta) / len(splits)
        print(
            f"fold {fold}: balanced_accuracy={score:.6f}, "
            f"best_iteration={best_iterations[-1]}, recalls={recalls}"
        )

    covered = np.zeros(len(y), dtype=bool)
    for _, va_idx in splits:
        covered[va_idx] = True
    if covered.all():
        oof_score = balanced_accuracy(y, oof.argmax(axis=1), len(classes))
    else:
        oof_score = balanced_accuracy(y[covered], oof[covered].argmax(axis=1), len(classes))

    report = {
        "purpose": "Leak-safe pure-model stacker using base OOF probabilities plus stable train/test features.",
        "models": models,
        "classes": classes,
        "fold_limit": len(splits),
        "covered_rows": int(covered.sum()),
        "score_on_covered_rows": oof_score,
        "fold_scores": fold_scores,
        "fold_recalls": fold_recalls,
        "best_iterations": best_iterations,
        "params": model_params,
        "feature_count": len(feature_names),
        "base_feature_count": len(base_feature_names),
        "source_model_reports": {
            model: {
                "oof_balanced_accuracy": source_reports[model].get("oof_balanced_accuracy"),
                "seed": source_reports[model].get("seed"),
            }
            for model in models
        },
    }

    if covered.all():
        submission = sample.copy()
        submission["class"] = encoder.inverse_transform(test_pred.argmax(axis=1))
        submission.to_csv(args.output_dir / "meta_stacker_submission.csv", index=False)
        np.save(args.output_dir / "meta_stacker_oof_proba.npy", oof)
        np.save(args.output_dir / "meta_stacker_test_proba.npy", test_pred.astype(np.float32))
        report["submission_path"] = str((args.output_dir / "meta_stacker_submission.csv").relative_to(ROOT))
        report["submission_class_share"] = submission["class"].value_counts(normalize=True).sort_index().to_dict()

    (args.output_dir / "meta_stacker_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
