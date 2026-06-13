from __future__ import annotations

import argparse
import json
import sys
import time
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
OUT_DIR = ARTIFACTS / "lgbm_variant_screen"
SEED = 20260610
N_SPLITS = 5


BASE_PARAMS = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "learning_rate": 0.035,
    "num_leaves": 96,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.88,
    "subsample_freq": 1,
    "colsample_bytree": 0.86,
    "reg_alpha": 0.08,
    "reg_lambda": 1.8,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_estimators": 4500,
    "n_jobs": -1,
    "verbosity": -1,
}


def variant_params() -> dict[str, dict]:
    variants = {
        "baseline": {},
        "galaxy_precision": {
            "num_leaves": 72,
            "min_child_samples": 140,
            "colsample_bytree": 0.82,
            "subsample": 0.90,
            "reg_alpha": 0.18,
            "reg_lambda": 3.2,
            "class_weight": {0: 1.00, 1: 1.08, 2: 1.02},
        },
        "star_boundary": {
            "learning_rate": 0.028,
            "num_leaves": 128,
            "min_child_samples": 70,
            "colsample_bytree": 0.90,
            "subsample": 0.86,
            "reg_alpha": 0.04,
            "reg_lambda": 2.4,
            "class_weight": {0: 0.92, 1: 1.07, 2: 1.22},
            "n_estimators": 5200,
        },
        "shallow_regularized": {
            "learning_rate": 0.04,
            "num_leaves": 48,
            "max_depth": 8,
            "min_child_samples": 150,
            "colsample_bytree": 0.78,
            "subsample": 0.92,
            "reg_alpha": 0.35,
            "reg_lambda": 5.0,
            "class_weight": "balanced",
        },
        "deep_low_lr": {
            "learning_rate": 0.022,
            "num_leaves": 176,
            "min_child_samples": 55,
            "colsample_bytree": 0.92,
            "subsample": 0.84,
            "reg_alpha": 0.02,
            "reg_lambda": 2.8,
            "class_weight": "balanced",
            "n_estimators": 6500,
        },
        "extra_trees": {
            "learning_rate": 0.032,
            "num_leaves": 112,
            "min_child_samples": 90,
            "colsample_bytree": 0.88,
            "subsample": 0.88,
            "reg_alpha": 0.10,
            "reg_lambda": 2.2,
            "class_weight": "balanced",
            "extra_trees": True,
            "extra_seed": SEED + 17,
        },
        "no_class_weight": {
            "num_leaves": 128,
            "min_child_samples": 90,
            "colsample_bytree": 0.88,
            "subsample": 0.88,
            "reg_alpha": 0.06,
            "reg_lambda": 2.0,
            "class_weight": None,
        },
    }
    return variants


def parse_args() -> argparse.Namespace:
    presets = sorted(variant_params())
    parser = argparse.ArgumentParser(
        description="Screen LGBM parameter variants on deterministic CV folds before spending full 5-fold training time."
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["baseline", "no_class_weight", "deep_low_lr"],
        choices=presets,
        help="Variant presets to run.",
    )
    parser.add_argument(
        "--fold-limit",
        type=int,
        default=1,
        help="Number of CV folds to train for screening.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory where screen results are written.",
    )
    parser.add_argument(
        "--max-estimators",
        type=int,
        default=2600,
        help="Screening cap for n_estimators. Full CV scripts can use larger values later.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=120,
        help="Early stopping rounds for screening.",
    )
    return parser.parse_args()


def merge_params(name: str, n_classes: int) -> dict:
    params = BASE_PARAMS.copy()
    params.update(variant_params()[name])
    params["num_class"] = n_classes
    params["random_state"] = int(params.get("random_state", SEED))
    return params


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, float]:
    recalls = {}
    for idx, cls in enumerate(classes):
        mask = y_true == idx
        recalls[f"recall_{cls}"] = float((y_pred[mask] == idx).mean()) if mask.any() else float("nan")
    return recalls


def write_results(rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
    results = pd.DataFrame(rows).sort_values("mean_balanced_accuracy", ascending=False)
    csv_results = results.drop(columns=["params", "fold_scores", "best_iterations"])
    csv_results.to_csv(output_dir / "results.csv", index=False)
    (output_dir / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# LGBM Variant Screen",
                "",
                "This is a fast fixed-fold screen. Promote only variants that beat the baseline fold score and improve the target class recalls.",
                "",
                csv_results.to_string(index=False),
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    x, y_raw, _, features = make_xy(train, test)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    classes = encoder.classes_.tolist()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(cv.split(x, y))[: args.fold_limit]

    rows = []
    results_path = args.output_dir / "results.json"
    if results_path.exists():
        rows = json.loads(results_path.read_text(encoding="utf-8"))
    completed = {
        (row["preset"], int(row["fold_limit"]))
        for row in rows
    }
    for preset in args.presets:
        if (preset, len(splits)) in completed:
            print(f"skip preset={preset}: already screened for fold_limit={len(splits)}")
            continue
        params = merge_params(preset, len(classes))
        params["n_estimators"] = min(int(params["n_estimators"]), args.max_estimators)
        fold_scores = []
        fold_recalls = []
        best_iterations = []
        started = time.time()

        print(f"\n=== preset={preset} folds={len(splits)} ===")
        for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
            model = lgb.LGBMClassifier(**params)
            model.fit(
                x.iloc[tr_idx],
                y[tr_idx],
                eval_set=[(x.iloc[va_idx], y[va_idx])],
                eval_metric="multi_logloss",
                callbacks=[
                    lgb.early_stopping(args.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(250),
                ],
            )
            va_pred = model.predict_proba(x.iloc[va_idx])
            pred_label = va_pred.argmax(axis=1)
            score = balanced_accuracy_score(y[va_idx], pred_label)
            recalls = class_recalls(y[va_idx], pred_label, classes)
            fold_scores.append(float(score))
            fold_recalls.append(recalls)
            best_iterations.append(int(model.best_iteration_ or params["n_estimators"]))
            print(
                f"fold {fold}: balanced_accuracy={score:.6f}, "
                f"best_iteration={best_iterations[-1]}, recalls={recalls}"
            )

        row = {
            "preset": preset,
            "fold_limit": len(splits),
            "mean_balanced_accuracy": float(np.mean(fold_scores)),
            "std_balanced_accuracy": float(np.std(fold_scores)),
            "fold_scores": fold_scores,
            "best_iterations": best_iterations,
            "seconds": round(time.time() - started, 2),
            "params": params,
        }
        for cls in classes:
            key = f"recall_{cls}"
            row[key] = float(np.mean([recall[key] for recall in fold_recalls]))
        rows.append(row)
        write_results(rows, args.output_dir)

    results = pd.DataFrame(rows).sort_values("mean_balanced_accuracy", ascending=False)
    csv_results = results.drop(columns=["params", "fold_scores", "best_iterations"])
    print("\n=== sorted results ===")
    print(csv_results.to_string(index=False))
    print(f"wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
