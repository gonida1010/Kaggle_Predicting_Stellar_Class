from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import CAT_COLS, add_advanced_features, make_xy  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
PURE_DIR = ARTIFACTS / "pure_model_ensemble"
OUT_DIR = ARTIFACTS / "sdss_external_generalization"
LABELS = ["GALAXY", "QSO", "STAR"]
REQUIRED_EXTERNAL = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift", "class"]
SEED = 20260617
N_SPLITS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train competition-only models and evaluate them on the external SDSS stellar classification dataset. "
            "This is an external generalization/stress validation, not a Kaggle submission builder."
        )
    )
    parser.add_argument("--external-data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--external-sample", type=int, default=0, help="0 means use all external rows.")
    parser.add_argument("--models", nargs="+", default=["lgbm", "catboost"], choices=["lgbm", "catboost"])
    parser.add_argument("--fold-limit", type=int, default=N_SPLITS)
    parser.add_argument(
        "--boundary-runs",
        nargs="*",
        default=[
            "gs_lr024_iter20_fallback",
            "gs_lr020_iter25_mid",
            "gs_lr024_iter20_mid",
            "gs_fixed050_mid_capacity",
        ],
    )
    parser.add_argument("--skip-boundary", action="store_true")
    parser.add_argument("--log-period", type=int, default=200)
    parser.add_argument(
        "--max-estimators",
        type=int,
        default=0,
        help="Debug/smoke-test cap for LGBM n_estimators and CatBoost iterations. 0 keeps production defaults.",
    )
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def find_external_data(user_path: Path | None) -> Path:
    candidates = []
    if user_path is not None:
        candidates.append(user_path)
    candidates.extend(
        [
            DATA / "star_classification.csv",
            DATA / "star_classification.csv.zip",
            ROOT / "external_data" / "star_classification.csv",
            ROOT / "external_data" / "star_classification.csv.zip",
            Path("/Users/parkyeonggon/Downloads/star_classification.csv"),
            Path("/Users/parkyeonggon/Downloads/star_classification.csv.zip"),
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing SDSS external data. Put star_classification.csv.zip under external_data/ "
        "or pass --external-data /path/to/star_classification.csv.zip."
    )


def read_external(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError(f"No CSV inside {path}")
            with zf.open(csv_names[0]) as handle:
                return pd.read_csv(handle)
    return pd.read_csv(path)


def normalize_external(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lower_to_original = {col.lower(): col for col in out.columns}
    rename = {}
    for col in REQUIRED_EXTERNAL:
        if col not in out.columns and col.lower() in lower_to_original:
            rename[lower_to_original[col.lower()]] = col
    if rename:
        out = out.rename(columns=rename)
    missing = [col for col in REQUIRED_EXTERNAL if col not in out.columns]
    if missing:
        raise ValueError(f"External data is missing required columns: {missing}")
    out = out[REQUIRED_EXTERNAL].copy()
    out["class"] = out["class"].astype(str).str.upper()
    out = out[out["class"].isin(LABELS)].copy()
    out["id"] = np.arange(len(out), dtype=np.int64) * -1 - 1
    out["spectral_type"] = "EXTERNAL_UNKNOWN"
    out["galaxy_population"] = "EXTERNAL_UNKNOWN"
    return out


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_probs(proba: np.ndarray) -> np.ndarray:
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return proba / row_sum


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, float]:
    out = {}
    for idx, label in enumerate(classes):
        mask = y_true == idx
        out[label] = float((y_pred[mask] == idx).mean()) if mask.any() else float("nan")
    return out


def transition_counts(before: np.ndarray, after: np.ndarray, classes: list[str]) -> dict[str, int]:
    changed = before != after
    counts = Counter(f"{classes[b]}->{classes[a]}" for b, a in zip(before[changed], after[changed]))
    return dict(sorted(counts.items()))


def score_record(
    name: str,
    y_ext: np.ndarray,
    pred_ext: np.ndarray,
    classes: list[str],
    extra: dict | None = None,
) -> dict:
    record = {
        "name": name,
        "external_balanced_accuracy": float(balanced_accuracy_score(y_ext, pred_ext)),
        "external_recalls": class_recalls(y_ext, pred_ext, classes),
        "external_pred_share": dict(
            sorted(
                {
                    classes[idx]: float((pred_ext == idx).mean())
                    for idx in range(len(classes))
                }.items()
            )
        ),
    }
    if extra:
        record.update(extra)
    return record


def lgbm_params() -> dict:
    report_path = ARTIFACTS / "lgbm_baseline_report.json"
    if report_path.exists():
        params = dict(load_json(report_path)["params"])
    else:
        params = {
            "objective": "multiclass",
            "num_class": 3,
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
            "random_state": 20260610,
            "n_estimators": 4500,
            "n_jobs": -1,
            "verbosity": -1,
        }
    params["random_state"] = int(params.get("random_state", SEED))
    return params


def catboost_params() -> dict:
    report_path = ARTIFACTS / "catboost_baseline_report.json"
    if report_path.exists():
        return dict(load_json(report_path)["params"])
    return {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": 3200,
        "learning_rate": 0.045,
        "depth": 8,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.6,
        "bagging_temperature": 0.35,
        "auto_class_weights": "Balanced",
        "random_seed": 20260611,
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": 250,
    }


def train_lgbm_external(
    x: pd.DataFrame,
    y: np.ndarray,
    x_ext: pd.DataFrame,
    classes: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    log_period: int,
    max_estimators: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    params = lgbm_params()
    if max_estimators > 0:
        params["n_estimators"] = min(int(params["n_estimators"]), int(max_estimators))
    oof = np.zeros((len(x), len(classes)), dtype=np.float32)
    ext = np.zeros((len(x_ext), len(classes)), dtype=np.float32)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        progress(f"Training LGBM fold {fold}/{len(splits)}")
        model = lgb.LGBMClassifier(**params)
        model.fit(
            x.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(x.iloc[va_idx], y[va_idx])],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(160, verbose=False),
                lgb.log_evaluation(log_period),
            ],
        )
        valid = model.predict_proba(x.iloc[va_idx])
        oof[va_idx] = valid
        ext += model.predict_proba(x_ext) / len(splits)
        rows.append(
            {
                "model": "lgbm",
                "fold": fold,
                "local_fold_balanced_accuracy": float(balanced_accuracy_score(y[va_idx], valid.argmax(axis=1))),
                "best_iteration": int(model.best_iteration_ or params["n_estimators"]),
            }
        )
    return oof, ext, rows


def train_catboost_external(
    x: pd.DataFrame,
    y: np.ndarray,
    x_ext: pd.DataFrame,
    classes: list[str],
    features: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    max_estimators: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    params = catboost_params()
    if max_estimators > 0:
        params["iterations"] = min(int(params["iterations"]), int(max_estimators))
    cat_features = [features.index(col) for col in CAT_COLS if col in features]
    oof = np.zeros((len(x), len(classes)), dtype=np.float32)
    ext = np.zeros((len(x_ext), len(classes)), dtype=np.float32)
    ext_pool = Pool(x_ext, cat_features=cat_features)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        progress(f"Training CatBoost fold {fold}/{len(splits)}")
        train_pool = Pool(x.iloc[tr_idx], y[tr_idx], cat_features=cat_features)
        valid_pool = Pool(x.iloc[va_idx], y[va_idx], cat_features=cat_features)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=180)
        valid = model.predict_proba(valid_pool)
        oof[va_idx] = valid
        ext += model.predict_proba(ext_pool) / len(splits)
        rows.append(
            {
                "model": "catboost",
                "fold": fold,
                "local_fold_balanced_accuracy": float(balanced_accuracy_score(y[va_idx], valid.argmax(axis=1))),
                "best_iteration": int(model.get_best_iteration() or params["iterations"]),
            }
        )
    return oof, ext, rows


def apply_pure_config(
    proba_by_model: dict[str, np.ndarray],
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    report = load_json(PURE_DIR / "pure_model_ensemble_report.json")
    config = report["best_config"]
    weights = config["model_weights"]
    bias = np.array([config["class_bias"][label] for label in classes], dtype=np.float64)
    usable = [model for model in weights if model in proba_by_model]
    if not usable:
        raise ValueError("No trained models available for pure ensemble config.")
    total_weight = sum(float(weights[model]) for model in usable)
    raw = sum(float(weights[model]) / total_weight * proba_by_model[model] for model in usable)
    raw = normalize_probs(raw)
    adjusted = normalize_probs(raw * bias.reshape(1, -1))
    return raw.astype(np.float32), adjusted.astype(np.float32)


def load_boundary_module():
    path = ROOT / "scripts" / "train_boundary_pair_calibrator.py"
    spec = importlib.util.spec_from_file_location("boundary_pair_calibrator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def evaluate_boundary_runs(
    runs: list[str],
    train: pd.DataFrame,
    external: pd.DataFrame,
    y: np.ndarray,
    y_ext: np.ndarray,
    classes: list[str],
    raw_ext: np.ndarray,
    cal_ext: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict], list[dict]]:
    module = load_boundary_module()
    cal_oof = np.load(PURE_DIR / "pure_model_ensemble_oof_proba.npy")
    raw_oof = np.load(PURE_DIR / "pure_model_ensemble_raw_oof_proba.npy")
    base_oof_pred = cal_oof.argmax(axis=1)
    base_ext_pred = cal_ext.argmax(axis=1)

    x_base, _, x_ext_base, _ = make_xy(train, external, feature_set="advanced")
    x = module.add_probability_features(x_base, cal_oof, raw_oof, classes)
    x_ext = module.add_probability_features(x_ext_base, cal_ext, raw_ext, classes)
    train_fe = add_advanced_features(train)
    ext_fe = add_advanced_features(external)

    rows = []
    fold_rows = []
    for run in runs:
        run_dir = ARTIFACTS / "boundary_pair_experiments" / run
        report_path = run_dir / "boundary_pair_calibrator_report.json"
        if not report_path.exists():
            progress(f"Skipping missing boundary run: {run}")
            continue
        report = load_json(report_path)
        if "GALAXY:STAR" not in report.get("pair_outputs", {}):
            progress(f"Skipping boundary run without GALAXY:STAR: {run}")
            continue
        best = report["pair_outputs"]["GALAXY:STAR"]["best"]
        fold_scores = report["pair_outputs"]["GALAXY:STAR"]["fold_scores"]
        left, right = "GALAXY", "STAR"
        left_idx = classes.index(left)
        right_idx = classes.index(right)
        pair_train_mask = (y == left_idx) | (y == right_idx)
        binary_y = (y == right_idx).astype("int8")
        right_ext = np.zeros(len(external), dtype=np.float32)
        right_oof = np.full(len(train), np.nan, dtype=np.float32)

        params = dict(report["model_params"])
        progress(f"Training external boundary evaluator for {run}")
        for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
            tr_pair_idx = tr_idx[pair_train_mask[tr_idx]]
            va_pair_idx = va_idx[pair_train_mask[va_idx]]
            pred_iter = int(fold_scores[fold_idx - 1].get("prediction_iteration") or report.get("fixed_iteration") or params["n_estimators"])
            model = lgb.LGBMClassifier(**params)
            model.fit(
                x.iloc[tr_pair_idx],
                binary_y[tr_pair_idx],
                eval_set=[(x.iloc[va_pair_idx], binary_y[va_pair_idx])],
                eval_metric="binary_logloss",
                callbacks=[
                    lgb.early_stopping(60, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            pred_iter = min(pred_iter, int(model.best_iteration_ or params["n_estimators"]))
            valid_right = model.predict_proba(x.iloc[va_idx], num_iteration=pred_iter)[:, 1]
            right_oof[va_idx] = valid_right
            right_ext += model.predict_proba(x_ext, num_iteration=pred_iter)[:, 1] / len(splits)
            fold_rows.append(
                {
                    "run": run,
                    "fold": fold_idx,
                    "prediction_iteration": pred_iter,
                    "local_boundary_fold_bacc": float(
                        balanced_accuracy_score(binary_y[va_pair_idx], (right_oof[va_pair_idx] >= 0.5).astype("int8"))
                    ),
                }
            )

        train_masks = module.segment_masks(train_fe, cal_oof, classes, left, right)
        ext_masks = module.segment_masks(ext_fe, cal_ext, classes, left, right)
        oof_pred = module.apply_pair_override(
            base_oof_pred,
            right_oof,
            train_masks[best["mask"]],
            left_idx,
            right_idx,
            best["to_right_threshold"],
            best["to_left_threshold"],
        )
        ext_pred = module.apply_pair_override(
            base_ext_pred,
            right_ext,
            ext_masks[best["mask"]],
            left_idx,
            right_idx,
            best["to_right_threshold"],
            best["to_left_threshold"],
        )
        rows.append(
            score_record(
                run,
                y_ext,
                ext_pred,
                classes,
                {
                    "kind": "boundary",
                    "reported_oof_delta": report.get("combined_delta"),
                    "recomputed_local_oof": float(balanced_accuracy_score(y, oof_pred)),
                    "external_delta_vs_pure": float(
                        balanced_accuracy_score(y_ext, ext_pred) - balanced_accuracy_score(y_ext, base_ext_pred)
                    ),
                    "external_changed_rows": int((ext_pred != base_ext_pred).sum()),
                    "external_transitions": json.dumps(
                        transition_counts(base_ext_pred, ext_pred, classes),
                        ensure_ascii=False,
                    ),
                    "mask": best["mask"],
                    "to_right_threshold": best["to_right_threshold"],
                    "to_left_threshold": best["to_left_threshold"],
                },
            )
        )
    return rows, fold_rows


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    progress("Loading competition train and SDSS external data")
    train = pd.read_csv(DATA / "train.csv")
    external_path = find_external_data(args.external_data)
    external_raw = read_external(external_path)
    external = normalize_external(external_raw)
    if args.external_sample and args.external_sample > 0 and args.external_sample < len(external):
        external = external.sample(n=args.external_sample, random_state=SEED).reset_index(drop=True)
        external["id"] = np.arange(len(external), dtype=np.int64) * -1 - 1

    encoder = LabelEncoder()
    y = encoder.fit_transform(train["class"].astype(str))
    classes = encoder.classes_.tolist()
    if classes != LABELS:
        progress(f"Class order from train is {classes}")
    y_ext = pd.Series(external["class"].astype(str)).map({label: idx for idx, label in enumerate(classes)}).to_numpy()

    progress("Building base feature matrices")
    x, _, x_ext, features = make_xy(train, external, feature_set="base")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(cv.split(x, y))[: args.fold_limit]
    if len(splits) != N_SPLITS:
        progress(f"Using fold_limit={len(splits)}; local OOF rows are partial and should be treated as smoke-test only.")

    proba_by_model = {}
    local_oof_by_model = {}
    fold_rows = []

    if "lgbm" in args.models:
        oof, ext, rows = train_lgbm_external(x, y, x_ext, classes, splits, args.log_period, args.max_estimators)
        proba_by_model["lgbm"] = ext
        local_oof_by_model["lgbm"] = oof
        fold_rows.extend(rows)
    if "catboost" in args.models:
        oof, ext, rows = train_catboost_external(x, y, x_ext, classes, features, splits, args.max_estimators)
        proba_by_model["catboost"] = ext
        local_oof_by_model["catboost"] = oof
        fold_rows.extend(rows)

    score_rows = []
    for model_name, proba in proba_by_model.items():
        score_rows.append(score_record(model_name, y_ext, proba.argmax(axis=1), classes, {"kind": "base_model"}))

    if len(proba_by_model) > 1:
        unweighted = normalize_probs(sum(proba_by_model.values()) / len(proba_by_model))
        score_rows.append(score_record("unweighted_external_ensemble", y_ext, unweighted.argmax(axis=1), classes, {"kind": "ensemble"}))

    raw_ext, cal_ext = apply_pure_config(proba_by_model, classes)
    pure_ext_pred = cal_ext.argmax(axis=1)
    score_rows.append(score_record("pure_config_external", y_ext, pure_ext_pred, classes, {"kind": "pure_config"}))

    boundary_fold_rows = []
    if not args.skip_boundary and set(["lgbm", "catboost"]).issubset(proba_by_model):
        boundary_rows, boundary_fold_rows = evaluate_boundary_runs(
            args.boundary_runs,
            train,
            external,
            y,
            y_ext,
            classes,
            raw_ext,
            cal_ext,
            splits,
        )
        score_rows.extend(boundary_rows)
    elif not args.skip_boundary:
        progress("Skipping boundary evaluation because both lgbm and catboost external probabilities are required.")

    summary = pd.DataFrame(score_rows).sort_values("external_balanced_accuracy", ascending=False)
    fold_df = pd.DataFrame([*fold_rows, *boundary_fold_rows])
    summary.to_csv(args.output_dir / "sdss_external_score_summary.csv", index=False)
    fold_df.to_csv(args.output_dir / "sdss_external_fold_diagnostics.csv", index=False)

    report = {
        "status": "ok",
        "external_path": str(external_path),
        "external_raw_shape": list(external_raw.shape),
        "external_used_shape": list(external.shape),
        "models": args.models,
        "fold_limit": len(splits),
        "summary_path": str((args.output_dir / "sdss_external_score_summary.csv").relative_to(ROOT)),
        "fold_diagnostics_path": str((args.output_dir / "sdss_external_fold_diagnostics.csv").relative_to(ROOT)),
        "best_external": summary.iloc[0].to_dict() if len(summary) else None,
        "note": (
            "This validation trains only on competition train labels and evaluates on labeled SDSS external rows. "
            "It measures external generalization, not Kaggle public/private score directly."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
