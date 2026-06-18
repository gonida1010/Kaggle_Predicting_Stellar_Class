from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "available_prediction_stacker"
DOWNLOADS = Path("/Users/parkyeonggon/Downloads")
EXTERNAL_OOF_TEST = ROOT / "external_sources" / "oof_test_predictions"
EXTERNAL_ARCHIVES = ROOT / "external_sources" / "prediction_archives"
EXTERNAL_PREDS = ROOT / "external_preds"
CLASSES = ["GALAXY", "QSO", "STAR"]
TARGET_MAP = {label: idx for idx, label in enumerate(CLASSES)}
INV_MAP = {idx: label for label, idx in TARGET_MAP.items()}
EPS = 1e-15
LOGIT_CLIP = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local logistic-regression stacker from whatever OOF/test prediction pairs are available. "
            "This avoids manually collecting all 19 public inputs before testing the stacker direction."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=650)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--boost-star", type=float, default=1.0)
    parser.add_argument("--include-own-models", action="store_true", default=True)
    parser.add_argument("--no-own-models", dest="include_own_models", action="store_false")
    parser.add_argument(
        "--only-models",
        nargs="*",
        default=None,
        help="Optional allowlist of model names to include after loading available prediction pairs.",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=[],
        help="Optional blocklist of model names to drop after loading available prediction pairs.",
    )
    parser.add_argument("--max-models", type=int, default=0, help="Debug cap. 0 uses all available pairs.")
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def prob_to_logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.clip(np.log(p / (1.0 - p)), -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32)


def normalize_probs(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return (proba / row_sum).astype(np.float32)


def load_csv_prediction(path: Path, expected_rows: int) -> np.ndarray:
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        vals = df.iloc[:, 0].to_numpy()
        if len(vals) != expected_rows * 3:
            raise ValueError(f"{path} has {len(vals)} flattened values, expected {expected_rows * 3}")
        return vals.reshape(-1, 3).astype(np.float32)
    return df.iloc[:, -3:].to_numpy(dtype=np.float32)[:expected_rows]


def load_zip_csv_prediction(zip_path: Path, member: str, expected_rows: int) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as handle:
            df = pd.read_csv(handle)
    if df.shape[1] == 1:
        vals = df.iloc[:, 0].to_numpy()
        if len(vals) != expected_rows * 3:
            raise ValueError(f"{zip_path}:{member} has {len(vals)} flattened values, expected {expected_rows * 3}")
        return vals.reshape(-1, 3).astype(np.float32)
    return df.iloc[:, -3:].to_numpy(dtype=np.float32)[:expected_rows]


def load_prediction(path: Path, expected_rows: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        arr = np.asarray(arr, dtype=np.float32)
    elif path.suffix.lower() == ".csv":
        arr = load_csv_prediction(path, expected_rows)
    else:
        raise ValueError(f"Unsupported prediction file: {path}")
    if arr.shape != (expected_rows, 3):
        raise ValueError(f"{path} shape {arr.shape}, expected {(expected_rows, 3)}")
    return normalize_probs(arr)


def archive4_pairs(n_train: int, n_test: int) -> list[dict]:
    zip_path = next(
        (
            path
            for path in [
                EXTERNAL_ARCHIVES / "archive (4).zip",
                DOWNLOADS / "archive (4).zip",
            ]
            if path.exists()
        ),
        None,
    )
    if zip_path is None:
        return []
    pairs = [
        ("realmlp-0", "oof_preds_realmlp0_v12.csv", "test_preds_realmlp0_v12.csv"),
        ("tabm-0", "oof_preds_tabm0_v2.csv", "test_preds_tabm0_v2.csv"),
        ("realmlp-2", "oof_preds_realmlp2_v10.csv", "test_preds_realmlp2_v10.csv"),
        ("lgbm-5", "oof_preds_lgbm5_v1.csv", "test_preds_lgbm5_v1.csv"),
        ("xgb-6", "oof_final_xgb6_v1.csv", "test_final_xgb6_v1.csv"),
        ("tabm-1", "oof_final_tabm1_v1.csv", "test_final_tabm1_v1.csv"),
    ]
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    for name, oof_member, test_member in pairs:
        if oof_member in names and test_member in names:
            out.append(
                {
                    "name": name,
                    "oof": load_zip_csv_prediction(zip_path, oof_member, n_train),
                    "test": load_zip_csv_prediction(zip_path, test_member, n_test),
                    "source": str(zip_path),
                }
            )
    return out


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def local_file_pairs(n_train: int, n_test: int) -> list[dict]:
    candidates = [
        (
            "realmlp-5",
            [EXTERNAL_OOF_TEST / "realmlp-5_oof.npy", DOWNLOADS / "realmlp-5_oof.npy"],
            [EXTERNAL_OOF_TEST / "realmlp-5_test_preds.npy", DOWNLOADS / "realmlp-5_test_preds.npy"],
        ),
        (
            "lr-stacker-v9-public-oof",
            [EXTERNAL_OOF_TEST / "oof_lr_stacker_v9.npy", DOWNLOADS / "oof_lr_stacker_v9.npy"],
            [EXTERNAL_PREDS / "pred_lr_stacker_v9.npy", DOWNLOADS / "pred_lr_stacker_v9.npy"],
        ),
        (
            "generic-oof-preds",
            [EXTERNAL_OOF_TEST / "oof_preds.npy", DOWNLOADS / "oof_preds.npy"],
            [EXTERNAL_OOF_TEST / "test_preds.npy", DOWNLOADS / "test_preds.npy"],
        ),
    ]
    out = []
    for name, oof_paths, test_paths in candidates:
        oof_path = first_existing(oof_paths)
        test_path = first_existing(test_paths)
        if oof_path is not None and test_path is not None:
            out.append(
                {
                    "name": name,
                    "oof": load_prediction(oof_path, n_train),
                    "test": load_prediction(test_path, n_test),
                    "source": f"{oof_path} + {test_path}",
                }
            )
        elif oof_path is not None:
            missing = " or ".join(path.name for path in test_paths)
            progress(f"Skipping {name}: OOF exists but matching test prediction is missing: {missing}")
    return out


def own_model_pairs(n_train: int, n_test: int) -> list[dict]:
    candidates = [
        (
            "our-pure",
            ARTIFACTS / "pure_model_ensemble" / "pure_model_ensemble_oof_proba.npy",
            ARTIFACTS / "pure_model_ensemble" / "pure_model_ensemble_test_proba.npy",
        ),
        (
            "our-meta",
            ARTIFACTS / "oof_proba_meta_model" / "meta_oof_proba.npy",
            ARTIFACTS / "oof_proba_meta_model" / "meta_test_proba.npy",
        ),
    ]
    out = []
    for name, oof_path, test_path in candidates:
        if oof_path.exists() and test_path.exists():
            out.append(
                {
                    "name": name,
                    "oof": load_prediction(oof_path, n_train),
                    "test": load_prediction(test_path, n_test),
                    "source": f"{oof_path} + {test_path}",
                }
            )
    return out


def train_stacker(
    x_blocks: list[np.ndarray],
    test_blocks: list[np.ndarray],
    y: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    n = len(y)
    m = len(test_blocks[0])
    x_all = np.concatenate([prob_to_logit(block) for block in x_blocks], axis=1).astype(np.float32)
    test_all = np.concatenate([prob_to_logit(block) for block in test_blocks], axis=1).astype(np.float32)
    progress(f"Training sklearn multinomial logistic stacker on {x_all.shape[1]} logit features")

    oof_sum = np.zeros((n, 3), dtype=np.float64)
    test_sum = np.zeros((m, 3), dtype=np.float64)
    fold_rows = []
    x_oof_matrix = np.zeros_like(x_all, dtype=np.float32)
    seeds = list(range(42, 42 + args.seeds))

    for seed in seeds:
        np.random.seed(seed)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        oof_seed = np.zeros((n, 3), dtype=np.float32)
        test_seed = np.zeros((m, 3), dtype=np.float32)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y), start=1):
            x_tr = x_all[tr_idx]
            x_va = x_all[va_idx]
            y_tr = y[tr_idx]
            y_va = y[va_idx]
            class_values = np.unique(y_tr)
            weights = compute_class_weight("balanced", classes=class_values, y=y_tr)
            class_weight = dict(zip(class_values, weights))
            class_weight[2] = class_weight[2] * float(args.boost_star)
            sample_weight = np.array([class_weight[value] for value in y_tr], dtype=np.float32)

            model = LogisticRegression(
                C=float(args.c),
                solver="lbfgs",
                max_iter=int(args.epochs),
                random_state=seed,
            )
            model.fit(x_tr, y_tr, sample_weight=sample_weight)
            va_prob = model.predict_proba(x_va)
            test_prob = model.predict_proba(test_all)
            oof_seed[va_idx] = va_prob
            test_seed += test_prob / args.folds
            if seed == seeds[-1]:
                x_oof_matrix[va_idx] = x_va
            fold_score = balanced_accuracy_score(y_va, va_prob.argmax(axis=1))
            fold_rows.append({"seed": seed, "fold": fold, "balanced_accuracy": float(fold_score)})
            progress(f"seed={seed} fold={fold} BAC={fold_score:.6f}")

        seed_score = balanced_accuracy_score(y, oof_seed.argmax(axis=1))
        progress(f"seed={seed} OOF BAC={seed_score:.6f}")
        oof_sum += oof_seed
        test_sum += test_seed

    oof = normalize_probs(oof_sum / len(seeds))
    test = normalize_probs(test_sum / len(seeds))
    return oof, test, fold_rows, x_oof_matrix


def feature_importance(
    x_oof_matrix: np.ndarray,
    y: np.ndarray,
    model_names: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    x_mean = x_oof_matrix.mean(axis=0, keepdims=True)
    x_std = x_oof_matrix.std(axis=0, keepdims=True)
    x_std[x_std == 0] = 1.0
    x_scaled = (x_oof_matrix - x_mean) / x_std

    class_values = np.unique(y)
    weights = compute_class_weight("balanced", classes=class_values, y=y)
    class_weight = dict(zip(class_values, weights))
    class_weight[2] = class_weight[2] * float(args.boost_star)
    sample_weight = np.array([class_weight[value] for value in y], dtype=np.float32)

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=max(200, int(args.epochs)),
    )
    model.fit(x_scaled, y, sample_weight=sample_weight)
    weights_np = model.coef_
    rows = []
    for idx, name in enumerate(model_names):
        block = weights_np[:, idx * 3 : (idx + 1) * 3]
        per_class = np.abs(block).sum(axis=1)
        rows.append(
            {
                "model": name,
                "total_importance": float(per_class.sum()),
                "GALAXY": float(per_class[0]),
                "QSO": float(per_class[1]),
                "STAR": float(per_class[2]),
            }
        )
    return pd.DataFrame(rows).sort_values("total_importance", ascending=False)


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["class"].map(TARGET_MAP).to_numpy()
    n_train = len(train)
    n_test = len(sample)

    records = []
    records.extend(archive4_pairs(n_train, n_test))
    records.extend(local_file_pairs(n_train, n_test))
    if args.include_own_models:
        records.extend(own_model_pairs(n_train, n_test))
    if args.only_models is not None and len(args.only_models) > 0:
        allowed = set(args.only_models)
        records = [record for record in records if record["name"] in allowed]
    if args.exclude_models:
        excluded = set(args.exclude_models)
        records = [record for record in records if record["name"] not in excluded]
    if args.max_models and args.max_models > 0:
        records = records[: args.max_models]
    if len(records) < 2:
        raise RuntimeError("Need at least two OOF/test prediction pairs to train a stacker.")

    model_names = [record["name"] for record in records]
    progress(f"Loaded {len(records)} prediction pairs: {model_names}")
    single_rows = []
    for record in records:
        pred = record["oof"].argmax(axis=1)
        single_rows.append(
            {
                "model": record["name"],
                "oof_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "source": record["source"],
            }
        )
    single_df = pd.DataFrame(single_rows).sort_values("oof_balanced_accuracy", ascending=False)
    single_df.to_csv(args.output_dir / "single_model_oof_scores.csv", index=False)

    oof, test, fold_rows, x_oof_matrix = train_stacker(
        [record["oof"] for record in records],
        [record["test"] for record in records],
        y,
        args,
    )
    pred = oof.argmax(axis=1)
    score = float(balanced_accuracy_score(y, pred))
    cm = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()

    submission = sample.copy()
    submission["class"] = np.array(CLASSES)[test.argmax(axis=1)]
    submission_path = args.output_dir / "available_prediction_stacker_submission.csv"
    submission.to_csv(submission_path, index=False)
    np.save(args.output_dir / "available_prediction_stacker_oof.npy", oof.astype(np.float32))
    np.save(args.output_dir / "available_prediction_stacker_test.npy", test.astype(np.float32))
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_scores.csv", index=False)
    imp = feature_importance(x_oof_matrix, y, model_names, args)
    imp.to_csv(args.output_dir / "model_importance.csv", index=False)

    report = {
        "purpose": "Local stacker from currently available OOF/test prediction pairs. Missing public inputs are skipped.",
        "models": model_names,
        "n_models": len(model_names),
        "epochs": args.epochs,
        "seeds": args.seeds,
        "folds": args.folds,
        "C": args.c,
        "lr": args.lr,
        "boost_star": args.boost_star,
        "oof_balanced_accuracy": score,
        "confusion_matrix": cm,
        "single_model_oof_scores": single_df.to_dict(orient="records"),
        "submission_path": str(submission_path.relative_to(ROOT)),
        "submission_class_share": submission["class"].value_counts(normalize=True).sort_index().to_dict(),
        "outputs": [
            "available_prediction_stacker_submission.csv",
            "available_prediction_stacker_oof.npy",
            "available_prediction_stacker_test.npy",
            "single_model_oof_scores.csv",
            "fold_scores.csv",
            "model_importance.csv",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
