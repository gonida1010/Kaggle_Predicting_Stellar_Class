from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, log_loss
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"

ID = "id"
TARGET = "class"
LABELS = np.array(["GALAXY", "QSO", "STAR"], dtype=object)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the public RepLeafGBM single-model idea locally and save OOF/test "
            "probabilities so it can be used as an honest CV/OOF stacking source."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "repleafgbm_cv_20260625")
    parser.add_argument("--fold-limit", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=128)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--l2-leaf", type=float, default=5.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta-grid", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 0.85, 1.0])
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def import_repleaf():
    try:
        import torch
        import repleafgbm
        from repleafgbm import RepLeafClassifier
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "RepLeafGBM is not installed in this venv.\n"
            "Install first, then rerun:\n"
            "  .venv/bin/python -m pip install 'repleafgbm[torch]' repleafgbm-native\n"
            f"Original error: {exc}"
        ) from exc
    return torch, repleafgbm, RepLeafClassifier


def cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def add_repleaf_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    return out


def make_submission(sample_ids: np.ndarray, labels: np.ndarray, path: Path) -> None:
    pd.DataFrame({ID: sample_ids, TARGET: labels}).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    torch, repleafgbm, RepLeafClassifier = import_repleaf()
    use_gpu = bool(torch.cuda.is_available()) and not args.force_cpu
    encoder_device = "cuda" if use_gpu else "cpu"
    split_backend = "cuda" if (use_gpu and cupy_available()) else "auto"

    progress(f"repleafgbm={getattr(repleafgbm, '__version__', 'unknown')}")
    progress(f"encoder_device={encoder_device}, split_backend={split_backend}")
    progress("Loading train/test data")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    y = train[TARGET].astype(str)
    if list(np.sort(y.unique())) != list(LABELS):
        raise ValueError(f"Unexpected labels: {sorted(y.unique())}")
    X = add_repleaf_features(train.drop(columns=[TARGET]))
    X_test = add_repleaf_features(test.copy())

    feature_cols = [col for col in X.columns if col != ID]
    X = X[feature_cols]
    X_test = X_test[feature_cols]

    oof = np.zeros((len(X), len(LABELS)), dtype=np.float64)
    test_proba = np.zeros((len(X_test), len(LABELS)), dtype=np.float64)
    fold_rows = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    splits = list(skf.split(X, y))
    if args.fold_limit < 1 or args.fold_limit > 5:
        raise ValueError("--fold-limit must be between 1 and 5")

    for fold, (tr_idx, va_idx) in enumerate(splits[: args.fold_limit], start=1):
        progress(f"Training RepLeafGBM fold {fold}/{args.fold_limit}")
        model = RepLeafClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_samples_leaf=args.min_samples_leaf,
            leaf_model="embedded_linear",
            encoder="torch_plr",
            encoder_params={"device": encoder_device, "random_state": args.seed},
            l2_leaf=args.l2_leaf,
            grow_policy="leafwise",
            split_backend=split_backend,
            early_stopping_rounds=args.early_stopping_rounds,
            eval_metric="balanced_accuracy",
            random_state=args.seed,
        )
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[va_idx], y.iloc[va_idx])])
        if list(model.classes_) != list(LABELS):
            raise ValueError(f"Fold {fold} class order mismatch: {model.classes_}")
        valid_proba = model.predict_proba(X.iloc[va_idx])
        oof[va_idx] = valid_proba
        test_proba += model.predict_proba(X_test) / args.fold_limit

        valid_pred = LABELS[valid_proba.argmax(axis=1)]
        fold_score = balanced_accuracy_score(y.iloc[va_idx], valid_pred)
        fold_loss = log_loss(y.iloc[va_idx], valid_proba, labels=list(LABELS))
        best_iter = getattr(model, "best_iteration_", None)
        progress(f"fold {fold}: raw_bac={fold_score:.6f}, logloss={fold_loss:.6f}, best_iteration={best_iter}")
        fold_rows.append(
            {
                "fold": fold,
                "raw_balanced_accuracy": float(fold_score),
                "logloss": float(fold_loss),
                "best_iteration": None if best_iter is None else int(best_iter),
            }
        )

    if args.fold_limit != 5:
        progress("fold-limit < 5: writing partial diagnostics only")
        covered = np.zeros(len(X), dtype=bool)
        for _, va_idx in splits[: args.fold_limit]:
            covered[va_idx] = True
        eval_y = y.iloc[covered]
        eval_oof = oof[covered]
    else:
        covered = np.ones(len(X), dtype=bool)
        eval_y = y
        eval_oof = oof

    prior = y.value_counts().reindex(LABELS).to_numpy(dtype=np.float64) / len(y)
    beta_rows = []
    best_beta = None
    best_score = -np.inf
    for beta in args.beta_grid:
        pred = LABELS[(eval_oof / (prior ** beta)).argmax(axis=1)]
        score = balanced_accuracy_score(eval_y, pred)
        beta_rows.append({"beta": float(beta), "balanced_accuracy": float(score)})
        if score > best_score:
            best_score = float(score)
            best_beta = float(beta)
    if best_beta is None:
        raise RuntimeError("No beta score was computed")

    full_pred = LABELS[(oof / (prior ** best_beta)).argmax(axis=1)] if args.fold_limit == 5 else None
    test_pred = LABELS[(test_proba / (prior ** best_beta)).argmax(axis=1)]

    np.save(args.output_dir / "repleaf_oof_proba.npy", oof)
    np.save(args.output_dir / "repleaf_test_proba.npy", test_proba)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_scores.csv", index=False)
    pd.DataFrame(beta_rows).to_csv(args.output_dir / "beta_scores.csv", index=False)
    make_submission(sample[ID].to_numpy(), test_pred, args.output_dir / "repleaf_submission.csv")

    report = {
        "purpose": "RepLeafGBM OOF/test proba source for honest OOF stacking.",
        "feature_cols": feature_cols,
        "fold_limit": int(args.fold_limit),
        "params": {
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "min_samples_leaf": int(args.min_samples_leaf),
            "l2_leaf": float(args.l2_leaf),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "seed": int(args.seed),
            "encoder_device": encoder_device,
            "split_backend": split_backend,
        },
        "fold_scores": fold_rows,
        "beta_scores": beta_rows,
        "best_beta": best_beta,
        "best_oof_balanced_accuracy": best_score,
        "outputs": {
            "oof_proba": str((args.output_dir / "repleaf_oof_proba.npy").relative_to(ROOT)),
            "test_proba": str((args.output_dir / "repleaf_test_proba.npy").relative_to(ROOT)),
            "submission": str((args.output_dir / "repleaf_submission.csv").relative_to(ROOT)),
        },
    }
    if full_pred is not None:
        report["classification_report"] = classification_report(y, full_pred, labels=list(LABELS), output_dict=True, zero_division=0)

    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
