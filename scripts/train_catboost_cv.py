from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import CAT_COLS, make_xy  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
SEED = 20260611
N_SPLITS = 5


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    x, y_raw, x_test, features = make_xy(train, test)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    classes = encoder.classes_.tolist()
    cat_features = [features.index(col) for col in CAT_COLS if col in features]

    params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": 3200,
        "learning_rate": 0.045,
        "depth": 8,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.6,
        "bagging_temperature": 0.35,
        "auto_class_weights": "Balanced",
        "random_seed": SEED,
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": 250,
    }

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(x), len(classes)), dtype=np.float32)
    test_pred = np.zeros((len(x_test), len(classes)), dtype=np.float32)
    fold_scores = []
    best_iterations = []
    test_pool = Pool(x_test, cat_features=cat_features)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(x, y), start=1):
        train_pool = Pool(x.iloc[tr_idx], y[tr_idx], cat_features=cat_features)
        valid_pool = Pool(x.iloc[va_idx], y[va_idx], cat_features=cat_features)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=180)

        va_pred = model.predict_proba(valid_pool)
        oof[va_idx] = va_pred
        pred_label = va_pred.argmax(axis=1)
        score = balanced_accuracy_score(y[va_idx], pred_label)
        fold_scores.append(float(score))
        best_iterations.append(int(model.get_best_iteration() or params["iterations"]))
        test_pred += model.predict_proba(test_pool) / N_SPLITS
        print(f"fold {fold}: balanced_accuracy={score:.6f}, best_iteration={best_iterations[-1]}")

    oof_pred = oof.argmax(axis=1)
    oof_score = balanced_accuracy_score(y, oof_pred)
    print(f"OOF balanced_accuracy={oof_score:.6f}")

    submission = sample.copy()
    submission["class"] = encoder.inverse_transform(test_pred.argmax(axis=1))

    np.save(ARTIFACTS / "catboost_oof_proba.npy", oof)
    np.save(ARTIFACTS / "catboost_test_proba.npy", test_pred)
    submission.to_csv(ARTIFACTS / "catboost_baseline_submission.csv", index=False)

    report = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "classes": classes,
        "features": features,
        "cat_features": cat_features,
        "params": params,
        "fold_scores": fold_scores,
        "oof_balanced_accuracy": float(oof_score),
        "best_iterations": best_iterations,
        "submission_class_share": submission["class"].value_counts(normalize=True).sort_index().to_dict(),
    }
    (ARTIFACTS / "catboost_baseline_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
