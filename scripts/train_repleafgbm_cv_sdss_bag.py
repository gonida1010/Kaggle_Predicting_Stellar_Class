from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, log_loss
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

ID = "id"
TARGET = "class"
LABELS = np.array(["GALAXY", "QSO", "STAR"], dtype=object)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RepLeafGBM CV with the 0.96633 notebook upgrades: optional SDSS17 "
            "train-fold augmentation, beta prior correction, and seed-bagging."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "repleafgbm_sdss_bag_20260626")
    parser.add_argument("--fold-limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-seeds", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--sdss-zip", type=Path, default=Path("/Users/parkyeonggon/Downloads/archive (12).zip"))
    parser.add_argument("--use-sdss17", action="store_true")
    parser.add_argument("--sdss-weight", type=float, default=0.3)
    parser.add_argument("--n-estimators", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=128)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--l2-leaf", type=float, default=5.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--beta-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.15, 1.3, 1.5])
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
            "RepLeafGBM is not installed.\n"
            "For Python 3.14, install native-free first:\n"
            "  .venv/bin/python -m pip install 'repleafgbm[torch]'\n"
            "If you need repleafgbm-native, use Python 3.13 or set PyO3 forward compatibility manually.\n"
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


def load_sdss17(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            member = "star_classification.csv"
            if member not in zf.namelist():
                return None
            with zf.open(member) as handle:
                external = pd.read_csv(handle)
    else:
        external = pd.read_csv(path)
    keep = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift", TARGET]
    missing = [col for col in keep if col not in external.columns]
    if missing:
        raise ValueError(f"SDSS17 file missing columns: {missing}")
    external = external[keep].copy()
    external["spectral_type"] = "unknown"
    external["galaxy_population"] = "unknown"
    external = external[external[TARGET].isin(LABELS)].reset_index(drop=True)
    return external


def tune_beta(y_true: pd.Series, proba: np.ndarray, prior: np.ndarray, beta_grid: list[float]) -> tuple[float, float, list[dict]]:
    rows = []
    best_beta = float(beta_grid[0])
    best_score = -np.inf
    for beta in beta_grid:
        pred = LABELS[(proba / (prior ** beta)).argmax(axis=1)]
        score = balanced_accuracy_score(y_true, pred)
        rows.append({"beta": float(beta), "balanced_accuracy": float(score)})
        if score > best_score:
            best_beta = float(beta)
            best_score = float(score)
    return best_beta, float(best_score), rows


def run_cv(
    RepLeafClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    args: argparse.Namespace,
    seed: int,
    encoder_device: str,
    split_backend: str,
    augment: tuple[pd.DataFrame, pd.Series, float] | None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))
    oof = np.zeros((len(X), len(LABELS)), dtype=np.float64)
    test_proba = np.zeros((len(X_test), len(LABELS)), dtype=np.float64)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(splits[: args.fold_limit], start=1):
        progress(f"seed={seed} fold {fold}/{args.fold_limit}: training RepLeafGBM")
        X_train = X.iloc[tr_idx]
        y_train = y.iloc[tr_idx]
        sample_weight = None
        if augment is not None:
            X_extra, y_extra, extra_weight = augment
            X_train = pd.concat([X_train, X_extra], ignore_index=True)
            y_train = pd.concat([y_train, y_extra], ignore_index=True)
            sample_weight = np.concatenate([np.ones(len(tr_idx)), np.full(len(X_extra), extra_weight)])

        model = RepLeafClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_samples_leaf=args.min_samples_leaf,
            leaf_model="embedded_linear",
            encoder="torch_plr",
            encoder_params={"device": encoder_device, "random_state": seed},
            l2_leaf=args.l2_leaf,
            grow_policy="leafwise",
            split_backend=split_backend,
            early_stopping_rounds=args.early_stopping_rounds,
            eval_metric="balanced_accuracy",
            random_state=seed,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X.iloc[va_idx], y.iloc[va_idx])])
        if list(model.classes_) != list(LABELS):
            raise ValueError(f"class order mismatch: {model.classes_}")
        valid_proba = model.predict_proba(X.iloc[va_idx])
        oof[va_idx] = valid_proba
        test_proba += model.predict_proba(X_test) / args.fold_limit
        valid_pred = LABELS[valid_proba.argmax(axis=1)]
        raw_bac = balanced_accuracy_score(y.iloc[va_idx], valid_pred)
        valid_loss = log_loss(y.iloc[va_idx], valid_proba, labels=list(LABELS))
        best_iter = getattr(model, "best_iteration_", None)
        progress(f"seed={seed} fold={fold}: raw_bac={raw_bac:.6f}, logloss={valid_loss:.6f}, best_iteration={best_iter}")
        fold_rows.append(
            {
                "seed": int(seed),
                "fold": int(fold),
                "raw_balanced_accuracy": float(raw_bac),
                "logloss": float(valid_loss),
                "best_iteration": None if best_iter is None else int(best_iter),
                "train_rows": int(len(X_train)),
                "sdss_augmented": bool(augment is not None),
            }
        )
    return oof, test_proba, fold_rows


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = (ROOT / args.output_dir).resolve()
    if args.fold_limit < 1 or args.fold_limit > 5:
        raise ValueError("--fold-limit must be between 1 and 5")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch, repleafgbm, RepLeafClassifier = import_repleaf()
    use_gpu = bool(torch.cuda.is_available()) and not args.force_cpu
    encoder_device = "cuda" if use_gpu else "cpu"
    split_backend = "cuda" if (use_gpu and cupy_available()) else "auto"
    progress(f"repleafgbm={getattr(repleafgbm, '__version__', 'unknown')}")
    progress(f"encoder_device={encoder_device}, split_backend={split_backend}")

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    y = train[TARGET].astype(str)
    X = add_repleaf_features(train.drop(columns=[TARGET]))
    X_test = add_repleaf_features(test)
    feature_cols = [col for col in X.columns if col != ID]
    X = X[feature_cols]
    X_test = X_test[feature_cols]

    augment = None
    sdss_rows = 0
    if args.use_sdss17:
        progress(f"Loading SDSS17 augmentation from {args.sdss_zip}")
        external = load_sdss17(args.sdss_zip)
        if external is not None and not external.empty:
            y_extra = external[TARGET].astype(str)
            X_extra = add_repleaf_features(external.drop(columns=[TARGET]))
            X_extra = X_extra[feature_cols]
            augment = (X_extra, y_extra, float(args.sdss_weight))
            sdss_rows = int(len(X_extra))
            progress(f"SDSS17 rows={sdss_rows}, weight={args.sdss_weight}")
        else:
            progress("SDSS17 augmentation requested but file was not found/readable; using pure S6E6")

    seeds = [int(args.seed), *[int(seed) for seed in args.extra_seeds]]
    progress(f"Running seeds={seeds}")
    member_oofs = []
    member_tests = []
    fold_rows = []
    for seed in seeds:
        oof, test_proba, rows = run_cv(
            RepLeafClassifier,
            X,
            y,
            X_test,
            args,
            seed,
            encoder_device,
            split_backend,
            augment,
        )
        member_oofs.append(oof)
        member_tests.append(test_proba)
        fold_rows.extend(rows)

    if args.fold_limit != 5:
        covered = np.zeros(len(X), dtype=bool)
        splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(X, y))
        for _, va_idx in splits[: args.fold_limit]:
            covered[va_idx] = True
        eval_y = y.iloc[covered]
        eval_oof = np.mean([arr[covered] for arr in member_oofs], axis=0)
        full_oof = np.mean(member_oofs, axis=0)
    else:
        eval_y = y
        full_oof = np.mean(member_oofs, axis=0)
        eval_oof = full_oof

    test_bag = np.mean(member_tests, axis=0)
    prior = y.value_counts().reindex(LABELS).to_numpy(dtype=np.float64) / len(y)
    best_beta, best_score, beta_rows = tune_beta(eval_y, eval_oof, prior, [float(x) for x in args.beta_grid])
    test_pred = LABELS[(test_bag / (prior ** best_beta)).argmax(axis=1)]

    np.save(args.output_dir / "repleaf_sdss_bag_oof_proba.npy", full_oof)
    np.save(args.output_dir / "repleaf_sdss_bag_test_proba.npy", test_bag)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_scores.csv", index=False)
    pd.DataFrame(beta_rows).to_csv(args.output_dir / "beta_scores.csv", index=False)
    pd.DataFrame({ID: sample[ID].to_numpy(), TARGET: test_pred}).to_csv(args.output_dir / "repleaf_sdss_bag_submission.csv", index=False)

    report = {
        "purpose": "RepLeafGBM OOF/test proba with SDSS17 augmentation and seed bagging.",
        "fold_limit": int(args.fold_limit),
        "seeds": seeds,
        "sdss_augmented": bool(augment is not None),
        "sdss_rows": sdss_rows,
        "sdss_weight": float(args.sdss_weight),
        "params": {
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "min_samples_leaf": int(args.min_samples_leaf),
            "l2_leaf": float(args.l2_leaf),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "encoder_device": encoder_device,
            "split_backend": split_backend,
        },
        "best_beta": float(best_beta),
        "best_eval_balanced_accuracy": float(best_score),
        "beta_scores": beta_rows,
        "fold_scores": fold_rows,
        "outputs": {
            "oof_proba": str((args.output_dir / "repleaf_sdss_bag_oof_proba.npy").relative_to(ROOT)),
            "test_proba": str((args.output_dir / "repleaf_sdss_bag_test_proba.npy").relative_to(ROOT)),
            "submission": str((args.output_dir / "repleaf_sdss_bag_submission.csv").relative_to(ROOT)),
        },
    }
    if args.fold_limit == 5:
        pred = LABELS[(full_oof / (prior ** best_beta)).argmax(axis=1)]
        report["classification_report"] = classification_report(y, pred, labels=list(LABELS), output_dict=True, zero_division=0)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
