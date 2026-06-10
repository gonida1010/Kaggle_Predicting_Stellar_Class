from __future__ import annotations

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
SEED = 20260610
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

    params = {
        "objective": "multiclass",
        "num_class": len(classes),
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

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(x), len(classes)), dtype=np.float32)
    test_pred = np.zeros((len(x_test), len(classes)), dtype=np.float32)
    fold_scores = []
    best_iterations = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(x, y), start=1):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            x.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(x.iloc[va_idx], y[va_idx])],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(160, verbose=False),
                lgb.log_evaluation(200),
            ],
        )

        va_pred = model.predict_proba(x.iloc[va_idx])
        oof[va_idx] = va_pred
        pred_label = va_pred.argmax(axis=1)
        score = balanced_accuracy_score(y[va_idx], pred_label)
        fold_scores.append(float(score))
        best_iterations.append(int(model.best_iteration_ or params["n_estimators"]))
        test_pred += model.predict_proba(x_test) / N_SPLITS
        print(f"fold {fold}: balanced_accuracy={score:.6f}, best_iteration={best_iterations[-1]}")

    oof_pred = oof.argmax(axis=1)
    oof_score = balanced_accuracy_score(y, oof_pred)
    print(f"OOF balanced_accuracy={oof_score:.6f}")

    submission = sample.copy()
    submission["class"] = encoder.inverse_transform(test_pred.argmax(axis=1))

    np.save(ARTIFACTS / "lgbm_oof_proba.npy", oof)
    np.save(ARTIFACTS / "lgbm_test_proba.npy", test_pred)
    submission.to_csv(ARTIFACTS / "lgbm_baseline_submission.csv", index=False)

    report = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "classes": classes,
        "features": features,
        "params": params,
        "fold_scores": fold_scores,
        "oof_balanced_accuracy": float(oof_score),
        "best_iterations": best_iterations,
        "submission_class_share": submission["class"].value_counts(normalize=True).sort_index().to_dict(),
    }
    (ARTIFACTS / "lgbm_baseline_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
