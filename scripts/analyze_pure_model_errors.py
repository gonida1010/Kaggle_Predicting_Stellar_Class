from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import add_features  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
DEFAULT_OOF = ARTIFACTS / "pure_model_ensemble" / "pure_model_ensemble_oof_proba.npy"
OUT_DIR = ARTIFACTS / "pure_model_diagnostics"
SOURCE_MODELS = ["lgbm", "catboost"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze anchor-free OOF model errors and write diagnostics for pure model research."
    )
    parser.add_argument(
        "--oof-proba",
        type=Path,
        default=DEFAULT_OOF,
        help="OOF probability numpy file to diagnose.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory where diagnostic tables and SVGs will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=300,
        help="Number of high-confidence wrong examples to export.",
    )
    return parser.parse_args()


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for class_idx in range(n_classes):
        mask = y_true == class_idx
        if mask.any():
            recalls.append(float((y_pred[mask] == class_idx).mean()))
    return float(np.mean(recalls))


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> pd.DataFrame:
    rows = []
    for idx, cls in enumerate(classes):
        true_mask = y_true == idx
        pred_mask = y_pred == idx
        tp = int((true_mask & pred_mask).sum())
        support = int(true_mask.sum())
        predicted = int(pred_mask.sum())
        rows.append(
            {
                "class": cls,
                "support": support,
                "predicted": predicted,
                "tp": tp,
                "recall": tp / support if support else np.nan,
                "precision": tp / predicted if predicted else np.nan,
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=classes, columns=classes, dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix.iat[int(true), int(pred)] += 1
    matrix.index.name = "true_class"
    matrix.columns.name = "pred_class"
    return matrix


def probability_frame(proba: np.ndarray, classes: list[str]) -> pd.DataFrame:
    sorted_probs = np.sort(proba, axis=1)
    pred_idx = proba.argmax(axis=1)
    top_prob = sorted_probs[:, -1]
    second_prob = sorted_probs[:, -2]
    frame = pd.DataFrame(
        {
            "pred_idx": pred_idx,
            "pred_class": np.array(classes)[pred_idx],
            "top_prob": top_prob,
            "margin": top_prob - second_prob,
        }
    )
    for idx, cls in enumerate(classes):
        frame[f"proba_{cls}"] = proba[:, idx]
    return frame


def svg_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_recall_svg(metrics: pd.DataFrame, path: Path) -> None:
    width = 760
    height = 330
    left = 110
    top = 40
    bar_h = 42
    gap = 34
    scale_w = 540
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="28" font-family="Arial" font-size="18" font-weight="700" fill="#1f2933">OOF Recall by Class</text>',
    ]
    for row_idx, row in metrics.reset_index(drop=True).iterrows():
        y = top + row_idx * (bar_h + gap)
        recall = float(row["recall"])
        bar_w = max(1, recall * scale_w)
        color = ["#2f80ed", "#27ae60", "#eb5757"][row_idx % 3]
        lines.extend(
            [
                f'<text x="24" y="{y + 28}" font-family="Arial" font-size="14" fill="#344054">{svg_escape(row["class"])}</text>',
                f'<rect x="{left}" y="{y}" width="{scale_w}" height="{bar_h}" rx="4" fill="#edf2f7"/>',
                f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="4" fill="{color}"/>',
                f'<text x="{left + scale_w + 16}" y="{y + 27}" font-family="Arial" font-size="14" fill="#111827">{recall:.5f}</text>',
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_confusion_svg(matrix: pd.DataFrame, path: Path) -> None:
    classes = matrix.index.tolist()
    max_value = max(1, int(matrix.to_numpy().max()))
    cell = 118
    left = 130
    top = 82
    width = left + cell * len(classes) + 38
    height = top + cell * len(classes) + 44
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="32" font-family="Arial" font-size="18" font-weight="700" fill="#1f2933">OOF Confusion Matrix</text>',
        '<text x="24" y="56" font-family="Arial" font-size="12" fill="#667085">rows=true, columns=predicted</text>',
    ]
    for col_idx, cls in enumerate(classes):
        x = left + col_idx * cell + cell / 2
        lines.append(
            f'<text x="{x:.1f}" y="{top - 18}" text-anchor="middle" font-family="Arial" font-size="13" fill="#344054">{svg_escape(cls)}</text>'
        )
    for row_idx, true_cls in enumerate(classes):
        y = top + row_idx * cell
        lines.append(
            f'<text x="{left - 18}" y="{y + cell / 2 + 5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#344054">{svg_escape(true_cls)}</text>'
        )
        row_sum = max(1, int(matrix.loc[true_cls].sum()))
        for col_idx, pred_cls in enumerate(classes):
            x = left + col_idx * cell
            value = int(matrix.loc[true_cls, pred_cls])
            intensity = value / max_value
            blue = int(245 - 120 * intensity)
            green = int(248 - 80 * intensity)
            red = int(255 - 190 * intensity)
            fill = f"rgb({red},{green},{blue})"
            share = value / row_sum
            lines.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell - 4}" height="{cell - 4}" rx="4" fill="{fill}" stroke="#ffffff"/>',
                    f'<text x="{x + cell / 2 - 2:.1f}" y="{y + 48}" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="#111827">{value}</text>',
                    f'<text x="{x + cell / 2 - 2:.1f}" y="{y + 72}" text-anchor="middle" font-family="Arial" font-size="12" fill="#475467">{share:.2%}</text>',
                ]
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df if max_rows is None else df.head(max_rows)
    if view.empty:
        return "_No rows._"
    columns = [str(col) for col in view.columns]
    rows = []
    for _, row in view.iterrows():
        values = []
        for col in view.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append(values)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(svg_escape(value) for value in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def build_feature_bin_errors(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        series = df[feature]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        binned = pd.qcut(series, q=10, duplicates="drop")
        grouped = df.assign(_bin=binned).groupby("_bin", observed=True)
        for interval, group in grouped:
            wrong = group[~group["correct"]]
            if wrong.empty:
                top_miss = ""
                top_miss_count = 0
            else:
                miss_counts = (
                    wrong.groupby(["true_class", "pred_class"], observed=True)
                    .size()
                    .sort_values(ascending=False)
                )
                top_pair = miss_counts.index[0]
                top_miss = f"{top_pair[0]}->{top_pair[1]}"
                top_miss_count = int(miss_counts.iloc[0])
            rows.append(
                {
                    "feature": feature,
                    "bin": str(interval),
                    "count": int(len(group)),
                    "error_rate": float((~group["correct"]).mean()),
                    "avg_margin": float(group["margin"].mean()),
                    "top_miss": top_miss,
                    "top_miss_count": top_miss_count,
                }
            )
    return pd.DataFrame(rows).sort_values(["error_rate", "count"], ascending=[False, False])


def load_source_disagreements(df: pd.DataFrame, classes: list[str], y: np.ndarray) -> pd.DataFrame:
    source_preds = {}
    for model in SOURCE_MODELS:
        path = ARTIFACTS / f"{model}_oof_proba.npy"
        if path.exists():
            source_preds[model] = np.load(path).argmax(axis=1)
    if len(source_preds) < 2:
        return pd.DataFrame()

    out = df[["id", "true_class", "pred_class", "top_prob", "margin", "correct"]].copy()
    for model, pred in source_preds.items():
        out[f"{model}_pred"] = np.array(classes)[pred]
        out[f"{model}_correct"] = pred == y
    model_names = list(source_preds)
    disagree = source_preds[model_names[0]] != source_preds[model_names[1]]
    return out[disagree].sort_values(["correct", "margin"], ascending=[True, False])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    train_features = add_features(train)
    classes = sorted(train["class"].astype(str).unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    y = train["class"].astype(str).map(class_to_idx).to_numpy()

    if not args.oof_proba.exists():
        raise FileNotFoundError(
            f"OOF proba not found: {args.oof_proba}. Run scripts/build_pure_model_ensemble.py first."
        )
    proba = np.load(args.oof_proba)
    if proba.shape != (len(train), len(classes)):
        raise ValueError(f"OOF shape mismatch: {proba.shape} vs {(len(train), len(classes))}")

    proba_info = probability_frame(proba, classes)
    pred = proba_info["pred_idx"].to_numpy()
    diagnostics = train_features.copy()
    diagnostics["true_idx"] = y
    diagnostics["true_class"] = train["class"].astype(str)
    diagnostics = pd.concat([diagnostics, proba_info], axis=1)
    diagnostics["correct"] = diagnostics["true_idx"] == diagnostics["pred_idx"]

    metrics = class_metrics(y, pred, classes)
    matrix = confusion_matrix(y, pred, classes)
    score = balanced_accuracy(y, pred, len(classes))

    error_pairs = (
        diagnostics[~diagnostics["correct"]]
        .groupby(["true_class", "pred_class"], observed=True)
        .agg(
            count=("id", "size"),
            avg_top_prob=("top_prob", "mean"),
            avg_margin=("margin", "mean"),
            avg_redshift=("redshift", "mean"),
            avg_u_minus_r=("u-r", "mean"),
            avg_g_minus_i=("g-i", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    keep_cols = [
        "id",
        "true_class",
        "pred_class",
        "top_prob",
        "margin",
        "redshift",
        "u",
        "g",
        "r",
        "i",
        "z",
        "u-r",
        "g-i",
        "mag_range",
        "spectral_type",
        "galaxy_population",
    ]
    hard_wrong = (
        diagnostics[~diagnostics["correct"]]
        .sort_values(["margin", "top_prob"], ascending=False)
        .loc[:, keep_cols]
        .head(args.top_n)
    )

    feature_bin_features = [
        "redshift",
        "redshift_abs",
        "u-g",
        "g-r",
        "r-i",
        "i-z",
        "u-r",
        "g-i",
        "mag_std",
        "mag_range",
        "redshift_x_u-r",
        "redshift_x_g-i",
    ]
    feature_bins = build_feature_bin_errors(diagnostics, feature_bin_features)
    disagreements = load_source_disagreements(diagnostics, classes, y)

    metrics.to_csv(args.output_dir / "class_metrics.csv", index=False)
    matrix.to_csv(args.output_dir / "confusion_matrix.csv")
    error_pairs.to_csv(args.output_dir / "error_pairs.csv", index=False)
    hard_wrong.to_csv(args.output_dir / "hard_wrong_examples.csv", index=False)
    feature_bins.to_csv(args.output_dir / "feature_bin_errors.csv", index=False)
    if not disagreements.empty:
        disagreements.head(args.top_n).to_csv(args.output_dir / "source_model_disagreements.csv", index=False)

    write_recall_svg(metrics, args.output_dir / "class_recall.svg")
    write_confusion_svg(matrix, args.output_dir / "confusion_matrix.svg")

    report = {
        "purpose": "Pure model OOF diagnostics. No public submission CSV is used.",
        "oof_proba": str(args.oof_proba.relative_to(ROOT)),
        "balanced_accuracy": score,
        "class_metrics": metrics.to_dict(orient="records"),
        "largest_error_pairs": error_pairs.head(6).to_dict(orient="records"),
        "worst_feature_bins": feature_bins.head(12).to_dict(orient="records"),
        "outputs": [
            "class_metrics.csv",
            "confusion_matrix.csv",
            "error_pairs.csv",
            "hard_wrong_examples.csv",
            "feature_bin_errors.csv",
            "source_model_disagreements.csv",
            "class_recall.svg",
            "confusion_matrix.svg",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    readme = [
        "# Pure Model Diagnostics",
        "",
        "This report diagnoses anchor-free OOF predictions. It does not use public leaderboard submissions.",
        "",
        f"- OOF balanced accuracy: `{score:.6f}`",
        "",
        "## Class Metrics",
        "",
        dataframe_to_markdown(metrics),
        "",
        "## Largest Error Pairs",
        "",
        dataframe_to_markdown(error_pairs, max_rows=8),
        "",
        "## Worst Feature Bins",
        "",
        dataframe_to_markdown(feature_bins, max_rows=12),
        "",
        "## Files",
        "",
        "- `class_recall.svg`",
        "- `confusion_matrix.svg`",
        "- `hard_wrong_examples.csv`",
        "- `feature_bin_errors.csv`",
        "- `source_model_disagreements.csv` when both source model OOF files exist",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
