from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np
import pandas as pd

from build_available_prediction_stacker import (
    ARTIFACTS,
    CLASSES,
    DATA,
    ROOT,
    TARGET_MAP,
    archive4_pairs,
    local_file_pairs,
    own_model_pairs,
)


OUT_DIR = ARTIFACTS / "oof_source_diversity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze OOF source quality and diversity for stacker research. "
            "This uses only train labels and OOF/test prediction pairs, never public LB scores."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--include-own-models", action="store_true", default=True)
    parser.add_argument("--no-own-models", dest="include_own_models", action="store_false")
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_probs(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return proba / row_sum


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for class_idx in range(len(CLASSES)):
        mask = y_true == class_idx
        if mask.any():
            recalls.append(float((y_pred[mask] == class_idx).mean()))
    return float(np.mean(recalls))


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {}
    for idx, label in enumerate(CLASSES):
        mask = y_true == idx
        out[f"recall_{label}"] = float((y_pred[mask] == idx).mean()) if mask.any() else float("nan")
    return out


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def load_records(include_own_models: bool, n_train: int, n_test: int) -> list[dict]:
    records = []
    records.extend(archive4_pairs(n_train, n_test))
    records.extend(local_file_pairs(n_train, n_test))
    if include_own_models:
        records.extend(own_model_pairs(n_train, n_test))
    names = set()
    unique = []
    for record in records:
        if record["name"] in names:
            continue
        names.add(record["name"])
        unique.append(record)
    return unique


def write_graphs(summary: pd.DataFrame, pairwise: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception as exc:
        progress(f"Skipping graphs because matplotlib is unavailable: {exc}")
        return

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    top = summary.sort_values("oof_balanced_accuracy", ascending=False).head(top_n).iloc[::-1]
    fig_height = max(6.0, 0.34 * len(top) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=160)
    ax.barh(top["name"], top["oof_balanced_accuracy"], color="#2563eb", alpha=0.9)
    ax.set_title("OOF Source Balanced Accuracy")
    ax.set_xlabel("OOF balanced accuracy")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    for y_idx, value in enumerate(top["oof_balanced_accuracy"]):
        ax.text(value + 0.00008, y_idx, f"{value:.6f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "oof_source_score_rank.svg", format="svg")
    fig.savefig(output_dir / "oof_source_score_rank.png", format="png")
    plt.close(fig)

    names = summary.sort_values("oof_balanced_accuracy", ascending=False).head(min(top_n, 20))["name"].tolist()
    matrix = np.full((len(names), len(names)), np.nan, dtype=np.float64)
    pair_map = {}
    for _, row in pairwise.iterrows():
        pair_map[(row["model_a"], row["model_b"])] = row
        pair_map[(row["model_b"], row["model_a"])] = row
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            matrix[i, j] = 1.0 if i == j else float(pair_map[(a, b)]["correct_corr"])
    fig, ax = plt.subplots(figsize=(13, 11), dpi=160)
    im = ax.imshow(matrix, cmap="coolwarm", vmin=0.0, vmax=1.0)
    ax.set_title("OOF Source Correctness Correlation")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right")
    ax.set_yticklabels(names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correctness correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "oof_source_correctness_correlation.svg", format="svg")
    fig.savefig(output_dir / "oof_source_correctness_correlation.png", format="png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["class"].map(TARGET_MAP).to_numpy()
    records = load_records(args.include_own_models, len(train), len(sample))
    if not records:
        raise RuntimeError("No OOF/test prediction pairs found.")
    progress(f"Loaded {len(records)} OOF/test prediction sources")

    prepared = []
    summary_rows = []
    for record in records:
        proba = normalize_probs(record["oof"])
        pred = proba.argmax(axis=1)
        correct = pred == y
        score = balanced_accuracy(y, pred)
        row = {
            "name": record["name"],
            "oof_balanced_accuracy": float(score),
            "accuracy": float(correct.mean()),
            "mean_max_probability": float(proba.max(axis=1).mean()),
            "pred_GALAXY_share": float((pred == 0).mean()),
            "pred_QSO_share": float((pred == 1).mean()),
            "pred_STAR_share": float((pred == 2).mean()),
            "source": record.get("source", ""),
        }
        row.update(class_recalls(y, pred))
        summary_rows.append(row)
        prepared.append({"name": record["name"], "proba": proba, "pred": pred, "correct": correct.astype(np.float32)})

    pair_rows = []
    for i, left in enumerate(prepared):
        for right in prepared[i + 1 :]:
            pred_agreement = float((left["pred"] == right["pred"]).mean())
            both_wrong = np.logical_and(left["pred"] != y, right["pred"] != y)
            either_wrong = np.logical_or(left["pred"] != y, right["pred"] != y)
            wrong_jaccard = float(both_wrong.sum() / max(1, either_wrong.sum()))
            class_prob_corrs = [
                safe_corr(left["proba"][:, class_idx], right["proba"][:, class_idx])
                for class_idx in range(len(CLASSES))
            ]
            correct_corr = safe_corr(left["correct"], right["correct"])
            pair_rows.append(
                {
                    "model_a": left["name"],
                    "model_b": right["name"],
                    "pred_agreement": pred_agreement,
                    "correct_corr": correct_corr,
                    "wrong_jaccard": wrong_jaccard,
                    "prob_corr_mean": float(np.mean(class_prob_corrs)),
                    "prob_corr_GALAXY": class_prob_corrs[0],
                    "prob_corr_QSO": class_prob_corrs[1],
                    "prob_corr_STAR": class_prob_corrs[2],
                    "diversity_score": float((1.0 - max(0.0, correct_corr)) * (1.0 - pred_agreement)),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values("oof_balanced_accuracy", ascending=False)
    pairwise = pd.DataFrame(pair_rows).sort_values(["diversity_score", "prob_corr_mean"], ascending=[False, True])
    summary.to_csv(args.output_dir / "oof_source_summary.csv", index=False)
    pairwise.to_csv(args.output_dir / "oof_source_pairwise_diversity.csv", index=False)
    write_graphs(summary, pairwise, args.output_dir, int(args.top_n))

    report = {
        "purpose": "OOF source quality/diversity analysis for generalization stack research.",
        "source_count": int(len(records)),
        "top_sources": summary.head(15).to_dict(orient="records"),
        "top_diverse_pairs": pairwise.head(20).to_dict(orient="records"),
        "outputs": [
            "oof_source_summary.csv",
            "oof_source_pairwise_diversity.csv",
            "oof_source_score_rank.svg",
            "oof_source_score_rank.png",
            "oof_source_correctness_correlation.svg",
            "oof_source_correctness_correlation.png",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    progress(f"Best source: {summary.iloc[0]['name']} OOF={summary.iloc[0]['oof_balanced_accuracy']:.9f}")
    progress(f"Wrote source diversity report to {args.output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

