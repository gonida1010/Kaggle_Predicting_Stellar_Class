from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import add_advanced_features, encode_categories  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "external_feature_experiment"
LABELS = ["GALAXY", "QSO", "STAR"]
REQUIRED_NUMERIC = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether the external SDSS stellar classification data is compatible with "
            "the competition distribution before using it for model training."
        )
    )
    parser.add_argument("--external-data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260617)
    return parser.parse_args()


def progress(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def find_external_data(user_path: Path | None) -> Path | None:
    if user_path is not None:
        return user_path if user_path.exists() else None
    candidates = [
        DATA / "star_classification.csv",
        DATA / "star_classification.csv.zip",
        ROOT / "external_data" / "star_classification.csv",
        ROOT / "external_data" / "star_classification.csv.zip",
        Path("/Users/parkyeonggon/Downloads/star_classification.csv"),
        Path("/Users/parkyeonggon/Downloads/star_classification.csv.zip"),
        Path("/Users/parkyeonggon/Downloads/archive.zip"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def read_external(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
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
    for col in [*REQUIRED_NUMERIC, "class"]:
        if col not in out.columns and col.lower() in lower_to_original:
            rename[lower_to_original[col.lower()]] = col
    if rename:
        out = out.rename(columns=rename)

    missing = [col for col in [*REQUIRED_NUMERIC, "class"] if col not in out.columns]
    if missing:
        raise ValueError(f"External data is missing required columns: {missing}")

    keep = [*REQUIRED_NUMERIC, "class"]
    out = out[keep].copy()
    out["class"] = out["class"].astype(str).str.upper()
    out = out[out["class"].isin(LABELS)].copy()
    out["id"] = np.arange(len(out), dtype=np.int64) * -1 - 1
    out["spectral_type"] = "EXTERNAL_UNKNOWN"
    out["galaxy_population"] = "EXTERNAL_UNKNOWN"
    return out


def class_distribution(df: pd.DataFrame, name: str) -> pd.DataFrame:
    counts = df["class"].value_counts().reindex(LABELS).fillna(0).astype(int)
    share = counts / max(1, counts.sum())
    return pd.DataFrame(
        {
            "dataset": name,
            "class": LABELS,
            "count": counts.to_numpy(),
            "share": share.to_numpy(),
        }
    )


def feature_shift(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> pd.DataFrame:
    rows = []
    numeric_cols = [col for col in left.columns if col in right.columns and pd.api.types.is_numeric_dtype(left[col])]
    for col in numeric_cols:
        if col == "id":
            continue
        a = left[col].replace([np.inf, -np.inf], np.nan).dropna()
        b = right[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(a) == 0 or len(b) == 0:
            continue
        stat = ks_2samp(a, b).statistic
        rows.append(
            {
                "left": left_name,
                "right": right_name,
                "feature": col,
                "left_mean": float(a.mean()),
                "right_mean": float(b.mean()),
                "mean_delta": float(b.mean() - a.mean()),
                "left_std": float(a.std()),
                "right_std": float(b.std()),
                "ks_stat": float(stat),
                "left_p05": float(a.quantile(0.05)),
                "right_p05": float(b.quantile(0.05)),
                "left_p50": float(a.quantile(0.50)),
                "right_p50": float(b.quantile(0.50)),
                "left_p95": float(a.quantile(0.95)),
                "right_p95": float(b.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows).sort_values("ks_stat", ascending=False)


def domain_screen(comp: pd.DataFrame, ext: pd.DataFrame, sample_size: int, seed: int) -> dict:
    n = min(sample_size, len(comp), len(ext))
    comp_sample = comp.sample(n=n, random_state=seed).copy()
    ext_sample = ext.sample(n=n, random_state=seed).copy()
    comp_sample["_domain"] = 0
    ext_sample["_domain"] = 1
    combined = pd.concat([comp_sample, ext_sample], ignore_index=True)

    labels = combined["_domain"].to_numpy()
    feature_df = combined.drop(
        columns=[
            "_domain",
            "class",
            "id",
            "spectral_type",
            "galaxy_population",
            "spectral_population",
        ],
        errors="ignore",
    )
    comp_part = feature_df.iloc[:n].copy()
    ext_part = feature_df.iloc[n:].copy()
    comp_part, ext_part = encode_categories(comp_part, ext_part)
    x = pd.concat([comp_part, ext_part], ignore_index=True).replace([np.inf, -np.inf], np.nan)
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            x[col] = x[col].astype(np.float32)

    tr_idx, va_idx = train_test_split(
        np.arange(len(x)),
        test_size=0.25,
        random_state=seed,
        stratify=labels,
    )
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        learning_rate=0.04,
        num_leaves=48,
        max_depth=7,
        min_child_samples=120,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=4.0,
        n_estimators=900,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x.iloc[tr_idx],
        labels[tr_idx],
        eval_set=[(x.iloc[va_idx], labels[va_idx])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    pred = model.predict_proba(x.iloc[va_idx])[:, 1]
    importance = pd.DataFrame(
        {
            "feature": x.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return {
        "sample_per_domain": int(n),
        "domain_auc": float(roc_auc_score(labels[va_idx], pred)),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "top_domain_features": importance.head(30).to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    progress("Finding external stellar data")
    external_path = find_external_data(args.external_data)
    if external_path is None:
        report = {
            "status": "missing_external_data",
            "message": (
                "Place star_classification.csv or star_classification.csv.zip under data/ "
                "or pass --external-data."
            ),
        }
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    progress("Loading competition train/test and external data")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    external_raw = read_external(external_path)
    progress("Normalizing external data columns")
    external = normalize_external(external_raw)

    progress("Building advanced features for train/test/external")
    train_fe = add_advanced_features(train)
    test_fe = add_advanced_features(test.assign(**{"class": "UNKNOWN"}))
    external_fe = add_advanced_features(external)

    progress("Writing class distribution")
    dist = pd.concat(
        [
            class_distribution(train, "competition_train"),
            class_distribution(external, "external"),
        ],
        ignore_index=True,
    )
    dist.to_csv(args.output_dir / "class_distribution.csv", index=False)

    progress("Computing feature shift tables")
    shift_ext = feature_shift(train_fe, external_fe, "competition_train", "external")
    shift_test = feature_shift(train_fe, test_fe, "competition_train", "competition_test")
    shift = pd.concat([shift_ext, shift_test], ignore_index=True)
    shift.to_csv(args.output_dir / "feature_shift.csv", index=False)

    progress(f"Training domain classifier with sample_size={args.sample_size}")
    domain = domain_screen(train_fe, external_fe, args.sample_size, args.seed)
    report = {
        "status": "ok",
        "external_path": str(external_path),
        "raw_external_shape": list(external_raw.shape),
        "normalized_external_shape": list(external.shape),
        "competition_train_shape": list(train.shape),
        "competition_test_shape": list(test.shape),
        "domain_screen": domain,
        "interpretation": (
            "High domain_auc means the external data is easy to distinguish from competition train. "
            "In that case, use it carefully: pretraining, subset-specific validation, or low-weight augmentation "
            "is safer than blindly concatenating it."
        ),
        "outputs": [
            "class_distribution.csv",
            "feature_shift.csv",
            "report.json",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    progress("Wrote external feature experiment report")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
