from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "artifacts" / ".cache"))
sys.path.append(str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from scripts.build_available_prediction_stacker import CLASSES, DATA, TARGET_MAP  # noqa: E402
from src.stellar_features import add_advanced_features  # noqa: E402


ARTIFACTS = ROOT / "artifacts"
DEFAULT_CANDIDATES = [
    (
        "56_te_disagreement",
        ARTIFACTS / "te_disagreement_patch_classwise37" / "56_high_gi_low_rz_base_galaxy_to_star_c0_55_m0_15_b0_60_oof.npy",
        0.97097,
    ),
    (
        "68_research_material_stack",
        ARTIFACTS / "research_material_stack_20260623" / "research_material_stack_oof.npy",
        0.97096,
    ),
    (
        "69_guarded_research_stack",
        ARTIFACTS / "private_cv_stable_research_material_stack_20260623" / "private_cv_guarded_01_all_changed_rz_0_2_allconf_oof.npy",
        0.97103,
    ),
    (
        "84_classwise_research_blend",
        ARTIFACTS / "classwise_research_blend_20260623" / "84_PRIVATE_CV_classwise_research_blend_oof0970621_oof.npy",
        0.97096,
    ),
    (
        "90_subset_guard_68_plus_84",
        ARTIFACTS / "classwise_research_blend_84_guard_20260623" / "90_PRIVATE_CV_subset_guard_68_plus_84_good_union_oof0970627_oof.npy",
        0.97096,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit private/generalization evidence for current OOF candidate submissions."
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "private_candidate_audit_20260623")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_probs(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return proba / row_sum


def load_proba(path: Path, expected_rows: int) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    arr = np.asarray(arr, dtype=np.float64)
    expected_shape = (expected_rows, len(CLASSES))
    if arr.shape != expected_shape:
        raise ValueError(f"{path} shape {arr.shape}, expected {expected_shape}")
    return normalize_probs(arr)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for class_idx in range(len(CLASSES)):
        mask = y_true == class_idx
        if mask.any():
            recalls.append(float((y_pred[mask] == class_idx).mean()))
    return float(np.mean(recalls))


def class_report_rows(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    rows = []
    for idx, label in enumerate(CLASSES):
        tp = int(((y_true == idx) & (y_pred == idx)).sum())
        fp = int(((y_true != idx) & (y_pred == idx)).sum())
        fn = int(((y_true == idx) & (y_pred != idx)).sum())
        support = int((y_true == idx).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "candidate": name,
                "class": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return rows


def quantile_bins(series: pd.Series, bins: int) -> pd.Series:
    return pd.qcut(series, q=bins, labels=False, duplicates="drop").astype("int16")


def build_subset_frame(train: pd.DataFrame, bins: int) -> pd.DataFrame:
    frame = add_advanced_features(train).copy()
    for col in ["redshift", "g-i", "u-r", "mag_range"]:
        frame[f"{col}_bin"] = quantile_bins(frame[col], bins)
    frame["spectral_population"] = frame["spectral_type"].astype(str) + "_" + frame["galaxy_population"].astype(str)
    return frame


def subset_summary(
    frame: pd.DataFrame,
    y: np.ndarray,
    ref_pred: np.ndarray,
    pred: np.ndarray,
    name: str,
) -> dict:
    specs = [
        ("spectral_type", "spectral_type"),
        ("spectral_population", "spectral_population"),
        ("redshift_bin", "redshift_bin"),
        ("g_i_bin", "g-i_bin"),
        ("u_r_bin", "u-r_bin"),
        ("mag_range_bin", "mag_range_bin"),
    ]
    rows = []
    for group_name, col in specs:
        for value, group in frame.groupby(col, observed=True):
            if len(group) < 300:
                continue
            idx = group.index.to_numpy()
            ref = balanced_accuracy(y[idx], ref_pred[idx])
            score = balanced_accuracy(y[idx], pred[idx])
            rows.append(
                {
                    "candidate": name,
                    "group": group_name,
                    "value": str(value),
                    "count": int(len(group)),
                    "delta_vs_reference": float(score - ref),
                    "changed_rows_vs_reference": int((pred[idx] != ref_pred[idx]).sum()),
                }
            )
    if not rows:
        return {
            "worst_subset": "",
            "worst_subset_delta_vs_reference": 0.0,
            "best_subset": "",
            "best_subset_delta_vs_reference": 0.0,
            "rows": [],
        }
    worst = min(rows, key=lambda row: row["delta_vs_reference"])
    best = max(rows, key=lambda row: row["delta_vs_reference"])
    return {
        "worst_subset": f"{worst['group']}={worst['value']}",
        "worst_subset_delta_vs_reference": worst["delta_vs_reference"],
        "best_subset": f"{best['group']}={best['value']}",
        "best_subset_delta_vs_reference": best["delta_vs_reference"],
        "rows": rows,
    }


def meta_fold_rows(
    y: np.ndarray,
    reference_pred: np.ndarray,
    preds: dict[str, np.ndarray],
    folds: int,
    seeds: int,
) -> list[dict]:
    rows = []
    for seed in range(20260623, 20260623 + seeds):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold_idx, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
            ref_score = balanced_accuracy(y[valid_idx], reference_pred[valid_idx])
            for name, pred in preds.items():
                score = balanced_accuracy(y[valid_idx], pred[valid_idx])
                rows.append(
                    {
                        "candidate": name,
                        "seed": seed,
                        "fold": fold_idx,
                        "reference_score": ref_score,
                        "score": score,
                        "delta_vs_reference": score - ref_score,
                    }
                )
    return rows


def style_axes(ax) -> None:
    ax.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#d0d5dd")


def save_score_plot(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.sort_values("oof_balanced_accuracy")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    bars = ax.barh(ordered["candidate"], ordered["oof_balanced_accuracy"], color="#2f80ed")
    for bar, value in zip(bars, ordered["oof_balanced_accuracy"]):
        ax.text(value + 0.000002, bar.get_y() + bar.get_height() / 2, f"{value:.9f}", va="center", fontsize=9)
    ax.set_title("OOF Balanced Accuracy by Candidate", fontsize=13, weight="bold")
    ax.set_xlabel("OOF balanced accuracy")
    ax.set_xlim(ordered["oof_balanced_accuracy"].min() - 0.00002, ordered["oof_balanced_accuracy"].max() + 0.00004)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_meta_fold_plot(meta: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)
    candidates = list(meta["candidate"].drop_duplicates())
    data = [meta.loc[meta["candidate"].eq(name), "delta_vs_reference"].to_numpy() for name in candidates]
    ax.boxplot(data, labels=candidates, showmeans=True)
    ax.axhline(0, color="#d92d20", linewidth=1.2, linestyle="--")
    ax.set_title("Repeated Meta-Fold Delta vs 56 Reference", fontsize=13, weight="bold")
    ax.set_ylabel("Balanced accuracy delta")
    ax.tick_params(axis="x", rotation=25)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    progress("Loading train data")
    train = pd.read_csv(DATA / "train.csv")
    y = train["class"].map(TARGET_MAP).to_numpy()
    subset_frame = build_subset_frame(train, args.bins)

    progress("Loading candidate OOF probabilities")
    probas = {}
    public_scores = {}
    for name, path, public_score in DEFAULT_CANDIDATES:
        if not path.exists():
            progress(f"skip missing candidate {name}: {path}")
            continue
        probas[name] = load_proba(path, len(train))
        public_scores[name] = public_score

    preds = {name: proba.argmax(axis=1) for name, proba in probas.items()}
    reference_name = "56_te_disagreement"
    reference_pred = preds[reference_name]
    reference_score = balanced_accuracy(y, reference_pred)

    progress("Computing candidate summary")
    summary_rows = []
    class_rows = []
    subset_rows = []
    for name, pred in preds.items():
        score = balanced_accuracy(y, pred)
        subset = subset_summary(subset_frame, y, reference_pred, pred, name)
        summary_rows.append(
            {
                "candidate": name,
                "oof_balanced_accuracy": score,
                "delta_vs_56": score - reference_score,
                "changed_rows_vs_56": int((pred != reference_pred).sum()),
                "public_score_seen": public_scores.get(name),
                "worst_subset": subset["worst_subset"],
                "worst_subset_delta_vs_56": subset["worst_subset_delta_vs_reference"],
                "best_subset": subset["best_subset"],
                "best_subset_delta_vs_56": subset["best_subset_delta_vs_reference"],
            }
        )
        class_rows.extend(class_report_rows(name, y, pred))
        subset_rows.extend(subset["rows"])

    summary = pd.DataFrame(summary_rows).sort_values("oof_balanced_accuracy", ascending=False)
    class_report = pd.DataFrame(class_rows)
    subset_report = pd.DataFrame(subset_rows)
    meta = pd.DataFrame(meta_fold_rows(y, reference_pred, preds, args.folds, args.seeds))
    meta_summary = (
        meta.groupby("candidate", observed=True)["delta_vs_reference"]
        .agg(["mean", "std", "min", "max", lambda values: float((values > 0).mean())])
        .reset_index()
        .rename(columns={"<lambda_0>": "positive_rate"})
    )

    summary = summary.merge(meta_summary, on="candidate", how="left", suffixes=("", "_meta_fold"))
    summary.to_csv(output_dir / "candidate_audit_summary.csv", index=False)
    class_report.to_csv(output_dir / "candidate_class_report.csv", index=False)
    subset_report.to_csv(output_dir / "candidate_subset_deltas.csv", index=False)
    meta.to_csv(output_dir / "candidate_meta_fold_deltas.csv", index=False)

    save_score_plot(summary, output_dir / "candidate_oof_balanced_accuracy.svg")
    save_meta_fold_plot(meta, output_dir / "candidate_meta_fold_delta_box.svg")

    report = {
        "purpose": "Evidence audit for private/generalization candidate selection.",
        "reference": reference_name,
        "notes": [
            "These are OOF/test-probability meta candidates, not single raw-data models.",
            "OOF score is computed on out-of-fold probabilities where available.",
            "The meta blend itself is optimized on OOF labels, so repeated meta-fold stability and subset deltas are used as overfit checks.",
            "Public LB is recorded only as noisy feedback and is not used for optimization.",
        ],
        "summary": summary.to_dict(orient="records"),
        "outputs": [
            "candidate_audit_summary.csv",
            "candidate_class_report.csv",
            "candidate_subset_deltas.csv",
            "candidate_meta_fold_deltas.csv",
            "candidate_oof_balanced_accuracy.svg",
            "candidate_meta_fold_delta_box.svg",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
