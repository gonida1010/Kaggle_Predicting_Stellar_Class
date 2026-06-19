from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import add_advanced_features, make_xy  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
PURE_DIR = ARTIFACTS / "pure_model_ensemble"
OUT_DIR = ARTIFACTS / "boundary_pair_calibrator"
SEED = 20260617
N_SPLITS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train boundary-specific binary calibrators from OOF probabilities and stellar features. "
            "A submission is written only when full OOF validation improves the pure model."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["GALAXY:STAR", "QSO:GALAXY"],
        help="Boundary pairs to test, e.g. GALAXY:STAR QSO:GALAXY.",
    )
    parser.add_argument("--feature-set", choices=["base", "advanced", "realmlp"], default="advanced")
    parser.add_argument("--fold-limit", type=int, default=N_SPLITS)
    parser.add_argument("--n-estimators", type=int, default=2200)
    parser.add_argument("--early-stopping-rounds", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.028)
    parser.add_argument("--num-leaves", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-child-samples", type=int, default=120)
    parser.add_argument("--subsample", type=float, default=0.88)
    parser.add_argument("--colsample-bytree", type=float, default=0.88)
    parser.add_argument("--reg-alpha", type=float, default=0.18)
    parser.add_argument("--reg-lambda", type=float, default=4.5)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--cosine-min-lr", type=float, default=0.006)
    parser.add_argument(
        "--prediction-iteration-policy",
        choices=["logloss", "fold_valid_bacc", "fixed", "fixed_with_fallback"],
        default="logloss",
        help=(
            "Iteration used for OOF/test probabilities. logloss uses LightGBM best_iteration; "
            "fold_valid_bacc uses the best diagnostic valid balanced accuracy per fold; fixed uses --fixed-iteration; "
            "fixed_with_fallback uses --fixed-iteration unless a fold diagnostic point is better by --iteration-fallback-min-gain."
        ),
    )
    parser.add_argument("--fixed-iteration", type=int, default=None)
    parser.add_argument(
        "--iteration-fallback-min-gain",
        type=float,
        default=0.0,
        help="Minimum fold valid balanced-accuracy gain required for fixed_with_fallback to override --fixed-iteration.",
    )
    parser.add_argument("--to-right-min", type=float, default=0.52)
    parser.add_argument("--to-right-max", type=float, default=0.94)
    parser.add_argument("--to-left-min", type=float, default=0.06)
    parser.add_argument("--to-left-max", type=float, default=0.48)
    parser.add_argument("--threshold-steps", type=int, default=22)
    parser.add_argument("--log-period", type=int, default=100)
    parser.add_argument(
        "--diagnostic-period",
        type=int,
        default=100,
        help="Iteration interval for saved logloss/balanced-accuracy diagnostics.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0001,
        help="Minimum full-OOF balanced-accuracy improvement required to write a submission.",
    )
    parser.add_argument(
        "--write-even-if-worse",
        action="store_true",
        help="Write the best candidate even if full OOF score does not improve.",
    )
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_pair(pair: str) -> tuple[str, str]:
    parts = pair.split(":")
    if len(parts) != 2 or not parts[0] or not parts[1] or parts[0] == parts[1]:
        raise ValueError(f"Invalid pair: {pair}. Use LEFT:RIGHT.")
    return parts[0], parts[1]


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for class_idx in range(n_classes):
        mask = y_true == class_idx
        if mask.any():
            recalls.append(float((y_pred[mask] == class_idx).mean()))
    return float(np.mean(recalls))


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, float]:
    out = {}
    for idx, label in enumerate(classes):
        mask = y_true == idx
        out[label] = float((y_pred[mask] == idx).mean()) if mask.any() else float("nan")
    return out


def entropy(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(proba, 1e-8, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def normalize_probs(proba: np.ndarray) -> np.ndarray:
    denom = proba.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return proba / denom


def load_pure_arrays(classes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    report_path = PURE_DIR / "pure_model_ensemble_report.json"
    if not report_path.exists():
        raise FileNotFoundError("Missing pure model report. Run build_pure_model_ensemble.py first.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["classes"] != classes:
        raise ValueError(f"Class order mismatch: {report['classes']} != {classes}")

    paths = {
        "cal_oof": PURE_DIR / "pure_model_ensemble_oof_proba.npy",
        "cal_test": PURE_DIR / "pure_model_ensemble_test_proba.npy",
        "raw_oof": PURE_DIR / "pure_model_ensemble_raw_oof_proba.npy",
        "raw_test": PURE_DIR / "pure_model_ensemble_raw_test_proba.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing pure probability files:\n" + "\n".join(missing))
    return (
        np.load(paths["cal_oof"]).astype(np.float32),
        np.load(paths["cal_test"]).astype(np.float32),
        np.load(paths["raw_oof"]).astype(np.float32),
        np.load(paths["raw_test"]).astype(np.float32),
        report,
    )


def add_probability_features(
    features: pd.DataFrame,
    cal: np.ndarray,
    raw: np.ndarray,
    classes: list[str],
) -> pd.DataFrame:
    out = features.reset_index(drop=True).copy()
    for prefix, proba in [("cal", cal), ("raw", raw)]:
        sorted_probs = np.sort(proba, axis=1)
        out[f"{prefix}_pred_idx"] = proba.argmax(axis=1).astype("int16")
        out[f"{prefix}_top_prob"] = sorted_probs[:, -1]
        out[f"{prefix}_second_prob"] = sorted_probs[:, -2]
        out[f"{prefix}_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
        out[f"{prefix}_entropy"] = entropy(proba)
        for idx, label in enumerate(classes):
            out[f"{prefix}_p_{label}"] = proba[:, idx]

    for idx, label in enumerate(classes):
        out[f"bias_delta_p_{label}"] = cal[:, idx] - raw[:, idx]

    for left, right in [("GALAXY", "STAR"), ("GALAXY", "QSO"), ("QSO", "STAR")]:
        if left in classes and right in classes:
            li = classes.index(left)
            ri = classes.index(right)
            out[f"cal_margin_{left}_vs_{right}"] = cal[:, li] - cal[:, ri]
            out[f"raw_margin_{left}_vs_{right}"] = raw[:, li] - raw[:, ri]
    return out.replace([np.inf, -np.inf], np.nan)


def params(args: argparse.Namespace) -> dict:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": 1,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_estimators": args.n_estimators,
        "n_jobs": -1,
        "verbosity": -1,
    }


def learning_rate_callback(args: argparse.Namespace):
    if args.lr_schedule == "constant":
        return None

    max_iter = max(1, int(args.n_estimators))
    start_lr = float(args.learning_rate)
    end_lr = float(args.cosine_min_lr)

    def cosine_lr(current_round: int) -> float:
        progress_ratio = min(max(current_round / max_iter, 0.0), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress_ratio))
        return end_lr + (start_lr - end_lr) * cosine

    return lgb.reset_parameter(learning_rate=cosine_lr)


def segment_masks(fe: pd.DataFrame, cal: np.ndarray, classes: list[str], left: str, right: str) -> dict[str, np.ndarray]:
    li = classes.index(left)
    ri = classes.index(right)
    pair_sum = cal[:, li] + cal[:, ri]
    third_max = np.max(np.delete(cal, [li, ri], axis=1), axis=1)

    base = {
        "all_pair_base": np.ones(len(fe), dtype=bool),
        "pair_sum_ge_070": pair_sum >= 0.70,
        "pair_sum_ge_085": pair_sum >= 0.85,
        "third_prob_le_020": third_max <= 0.20,
        "low_redshift": fe["redshift"].between(-0.02, 0.14).to_numpy(),
        "mid_redshift": fe["redshift"].between(0.05, 0.244).to_numpy(),
    }
    if "g-i" in fe.columns and "mag_range" in fe.columns:
        base["gs_oof_hard_color_range"] = (
            fe["redshift"].between(-0.02, 0.244)
            & fe["g-i"].between(0.493, 1.21)
            & fe["mag_range"].between(1.485, 2.984)
        ).to_numpy()
        base["red_sequence_like"] = (
            fe["g-i"].between(1.2, 3.2)
            & fe["mag_range"].between(3.0, 7.5)
            & fe["redshift"].between(0.02, 0.30)
        ).to_numpy()
    if "redshift_x_g-i" in fe.columns:
        base["redshift_x_gi_hard"] = fe["redshift_x_g-i"].between(0.0127, 0.242).to_numpy()

    combined = {}
    for name, mask in base.items():
        combined[name] = mask
        if name != "all_pair_base":
            combined[f"{name}_and_pairsum085"] = mask & (pair_sum >= 0.85)
    return combined


def apply_pair_override(
    base_pred: np.ndarray,
    right_proba: np.ndarray,
    active_mask: np.ndarray,
    left_idx: int,
    right_idx: int,
    to_right_threshold: float,
    to_left_threshold: float,
) -> np.ndarray:
    pred = base_pred.copy()
    pair_base = (base_pred == left_idx) | (base_pred == right_idx)
    active = pair_base & active_mask
    pred[active & (base_pred == left_idx) & (right_proba >= to_right_threshold)] = right_idx
    pred[active & (base_pred == right_idx) & (right_proba <= to_left_threshold)] = left_idx
    return pred


def transition_counts(before: np.ndarray, after: np.ndarray, classes: list[str]) -> dict[str, int]:
    changed = before != after
    counts = Counter(f"{classes[b]}->{classes[a]}" for b, a in zip(before[changed], after[changed]))
    return dict(sorted(counts.items()))


def search_pair_rule(
    y: np.ndarray,
    base_pred: np.ndarray,
    right_oof: np.ndarray,
    masks: dict[str, np.ndarray],
    classes: list[str],
    left: str,
    right: str,
    covered: np.ndarray,
    to_right_min: float,
    to_right_max: float,
    to_left_min: float,
    to_left_max: float,
    threshold_steps: int,
) -> tuple[dict, list[dict]]:
    left_idx = classes.index(left)
    right_idx = classes.index(right)
    n_classes = len(classes)
    eval_mask = covered
    base_score = balanced_accuracy(y[eval_mask], base_pred[eval_mask], n_classes)
    best = {
        "pair": f"{left}:{right}",
        "score": base_score,
        "delta": 0.0,
        "mask": "none",
        "to_right_threshold": None,
        "to_left_threshold": None,
        "changed_rows": 0,
        "transition_counts": {},
        "class_recalls": class_recalls(y[eval_mask], base_pred[eval_mask], classes),
    }
    records = []

    to_right_values = np.linspace(to_right_min, to_right_max, threshold_steps)
    to_left_values = np.linspace(to_left_min, to_left_max, threshold_steps)
    for mask_name, mask in masks.items():
        active = mask & eval_mask
        if active.sum() < 100:
            continue
        for to_right in to_right_values:
            for to_left in to_left_values:
                if to_left >= to_right:
                    continue
                pred = apply_pair_override(
                    base_pred,
                    right_oof,
                    mask,
                    left_idx,
                    right_idx,
                    float(to_right),
                    float(to_left),
                )
                score = balanced_accuracy(y[eval_mask], pred[eval_mask], n_classes)
                changed = int((pred[eval_mask] != base_pred[eval_mask]).sum())
                if changed == 0:
                    continue
                record = {
                    "pair": f"{left}:{right}",
                    "score": float(score),
                    "delta": float(score - base_score),
                    "mask": mask_name,
                    "to_right_threshold": float(to_right),
                    "to_left_threshold": float(to_left),
                    "changed_rows": changed,
                    "transition_counts": transition_counts(base_pred[eval_mask], pred[eval_mask], classes),
                    "class_recalls": class_recalls(y[eval_mask], pred[eval_mask], classes),
                }
                records.append(record)
                if record["score"] > best["score"]:
                    best = record

    records.sort(key=lambda row: (row["score"], -row["changed_rows"]), reverse=True)
    return best, records[:40]


def write_bar_svg(path: Path, title: str, rows: list[dict], label_key: str, value_key: str) -> None:
    rows = rows[:20]
    width = 1120
    row_h = 32
    top = 58
    left = 330
    right = 80
    height = top + row_h * max(1, len(rows)) + 48
    values = [float(row[value_key]) for row in rows]
    max_abs = max([abs(v) for v in values] + [1e-9])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700" fill="#111827">{title}</text>',
        f'<line x1="{left}" y1="{top-20}" x2="{left}" y2="{height-28}" stroke="#9ca3af" stroke-width="1"/>',
    ]
    zero_x = left
    bar_w = width - left - right
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        label = str(row[label_key]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        value = float(row[value_key])
        length = abs(value) / max_abs * bar_w
        color = "#2563eb" if value >= 0 else "#dc2626"
        x = zero_x if value >= 0 else zero_x - length
        parts.extend(
            [
                f'<text x="24" y="{y+18}" font-family="Arial" font-size="13" fill="#374151">{label}</text>',
                f'<rect x="{x:.2f}" y="{y}" width="{length:.2f}" height="20" fill="{color}" opacity="0.86"/>',
                f'<text x="{left+bar_w+10}" y="{y+16}" font-family="Arial" font-size="13" fill="#111827">{value:+.7f}</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_metric_curve_svg(
    path: Path,
    title: str,
    rows: list[dict],
    metric: str,
) -> None:
    if not rows:
        return
    width = 1120
    height = 620
    left = 84
    right = 220
    top = 58
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    iterations = [int(row["iteration"]) for row in rows]
    values = [float(row[metric]) for row in rows if pd.notna(row[metric])]
    if not values:
        return
    x_min, x_max = min(iterations), max(iterations)
    y_min, y_max = min(values), max(values)
    if y_min == y_max:
        y_min -= 1e-6
        y_max += 1e-6
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    def x_pos(iteration: int) -> float:
        if x_max == x_min:
            return left + plot_w / 2
        return left + (iteration - x_min) / (x_max - x_min) * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if pd.isna(row[metric]):
            continue
        key = f"{row['pair']} fold {row['fold']}"
        grouped.setdefault(key, []).append(row)

    colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
        "#be123c",
        "#15803d",
        "#7c2d12",
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700" fill="#111827">{title}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#d1d5db"/>',
        f'<text x="{left}" y="{height-24}" font-family="Arial" font-size="13" fill="#374151">iteration</text>',
        f'<text x="18" y="{top+14}" font-family="Arial" font-size="13" fill="#374151">{metric}</text>',
        f'<text x="{left}" y="{height-50}" font-family="Arial" font-size="12" fill="#6b7280">{x_min}</text>',
        f'<text x="{left+plot_w-34}" y="{height-50}" font-family="Arial" font-size="12" fill="#6b7280">{x_max}</text>',
        f'<text x="28" y="{top+12}" font-family="Arial" font-size="12" fill="#6b7280">{y_max:.6f}</text>',
        f'<text x="28" y="{top+plot_h}" font-family="Arial" font-size="12" fill="#6b7280">{y_min:.6f}</text>',
    ]

    legend_x = left + plot_w + 24
    legend_y = top + 8
    for idx, (key, group) in enumerate(grouped.items()):
        color = colors[idx % len(colors)]
        group = sorted(group, key=lambda row: int(row["iteration"]))
        points = " ".join(
            f'{x_pos(int(row["iteration"])):.2f},{y_pos(float(row[metric])):.2f}'
            for row in group
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" opacity="0.9"/>'
        )
        if idx < 16:
            y = legend_y + idx * 24
            label = key.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            parts.extend(
                [
                    f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+22}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                    f'<text x="{legend_x+30}" y="{y+4}" font-family="Arial" font-size="12" fill="#374151">{label}</text>',
                ]
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_pair_panel_metric_svg(
    path: Path,
    title: str,
    rows: list[dict],
    metric: str,
    y_label: str,
) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if metric not in df.columns:
        return
    pairs = list(dict.fromkeys(df["pair"].astype(str).tolist()))
    if not pairs:
        return

    width = 1120
    panel_h = 330
    top_margin = 58
    bottom_margin = 54
    height = top_margin + panel_h * len(pairs) + bottom_margin
    left = 88
    right = 56
    plot_w = width - left - right
    plot_h = 236
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700" fill="#111827">{title}</text>',
    ]

    for panel_idx, pair in enumerate(pairs):
        pair_df = df[df["pair"] == pair].copy()
        pair_df = pair_df.sort_values(["fold", "iteration"])
        mean_df = pair_df.groupby("iteration", as_index=False)[metric].mean()
        best_row = mean_df.loc[mean_df[metric].idxmax()]
        if "logloss" in metric:
            best_row = mean_df.loc[mean_df[metric].idxmin()]

        x_min = int(pair_df["iteration"].min())
        x_max = int(pair_df["iteration"].max())
        y_values = pair_df[metric].dropna().astype(float).to_numpy()
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        if y_min == y_max:
            y_min -= 1e-6
            y_max += 1e-6
        pad = (y_max - y_min) * 0.12
        y_min -= pad
        y_max += pad

        panel_y = top_margin + panel_idx * panel_h
        plot_y = panel_y + 50

        def x_pos(iteration: int) -> float:
            if x_max == x_min:
                return left + plot_w / 2
            return left + (iteration - x_min) / (x_max - x_min) * plot_w

        def y_pos(value: float) -> float:
            return plot_y + (y_max - value) / (y_max - y_min) * plot_h

        parts.extend(
            [
                f'<text x="24" y="{panel_y+26}" font-family="Arial" font-size="17" font-weight="700" fill="#111827">{pair}</text>',
                f'<text x="24" y="{panel_y+48}" font-family="Arial" font-size="12" fill="#4b5563">best mean iteration={int(best_row["iteration"])} / mean {metric}={float(best_row[metric]):.6f}</text>',
                f'<rect x="{left}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#d1d5db"/>',
                f'<text x="{left}" y="{plot_y+plot_h+28}" font-family="Arial" font-size="12" fill="#4b5563">{x_min}</text>',
                f'<text x="{left+plot_w-36}" y="{plot_y+plot_h+28}" font-family="Arial" font-size="12" fill="#4b5563">{x_max}</text>',
                f'<text x="22" y="{plot_y+12}" font-family="Arial" font-size="12" fill="#4b5563">{y_max:.6f}</text>',
                f'<text x="22" y="{plot_y+plot_h}" font-family="Arial" font-size="12" fill="#4b5563">{y_min:.6f}</text>',
                f'<text x="{left+plot_w/2-28}" y="{plot_y+plot_h+32}" font-family="Arial" font-size="12" fill="#374151">iteration</text>',
                f'<text x="22" y="{plot_y-8}" font-family="Arial" font-size="12" fill="#374151">{y_label}</text>',
            ]
        )

        best_x = x_pos(int(best_row["iteration"]))
        parts.append(
            f'<line x1="{best_x:.2f}" y1="{plot_y}" x2="{best_x:.2f}" y2="{plot_y+plot_h}" stroke="#111827" stroke-width="1.5" stroke-dasharray="5 4"/>'
        )
        tick_iterations = mean_df["iteration"].astype(int).tolist()
        if len(tick_iterations) > 13:
            keep = np.linspace(0, len(tick_iterations) - 1, 13).round().astype(int)
            tick_iterations = [tick_iterations[idx] for idx in sorted(set(keep.tolist()))]
        for iteration in tick_iterations:
            x = x_pos(iteration)
            parts.append(
                f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y+plot_h}" stroke="#e5e7eb" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x-10:.2f}" y="{plot_y+plot_h+28}" font-family="Arial" font-size="11" fill="#4b5563">{iteration}</text>'
            )

        for color_idx, (fold, fold_df) in enumerate(pair_df.groupby("fold")):
            points = " ".join(
                f'{x_pos(int(row["iteration"])):.2f},{y_pos(float(row[metric])):.2f}'
                for _, row in fold_df.iterrows()
            )
            color = colors[color_idx % len(colors)]
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.3" opacity="0.28"/>'
            )

        mean_points = " ".join(
            f'{x_pos(int(row["iteration"])):.2f},{y_pos(float(row[metric])):.2f}'
            for _, row in mean_df.iterrows()
        )
        parts.append(
            f'<polyline points="{mean_points}" fill="none" stroke="#111827" stroke-width="3.2" opacity="0.95"/>'
        )
        for _, row in mean_df.iterrows():
            parts.append(
                f'<circle cx="{x_pos(int(row["iteration"])):.2f}" cy="{y_pos(float(row[metric])):.2f}" r="3.2" fill="#111827"/>'
            )
        parts.append(
            f'<circle cx="{best_x:.2f}" cy="{y_pos(float(best_row[metric])):.2f}" r="6" fill="#f59e0b" stroke="#111827" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{left+plot_w-205}" y="{plot_y+20}" font-family="Arial" font-size="12" fill="#111827">black=fold mean, color=fold</text>'
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_train_valid_panel_svg(path: Path, title: str, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    metric_cols = ["train_binary_balanced_accuracy", "valid_binary_balanced_accuracy"]
    if any(col not in df.columns for col in metric_cols):
        return
    pairs = list(dict.fromkeys(df["pair"].astype(str).tolist()))
    width = 1120
    panel_h = 330
    top_margin = 58
    bottom_margin = 54
    height = top_margin + panel_h * len(pairs) + bottom_margin
    left = 88
    right = 56
    plot_w = width - left - right
    plot_h = 236

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700" fill="#111827">{title}</text>',
    ]

    for panel_idx, pair in enumerate(pairs):
        pair_df = df[df["pair"] == pair].copy()
        mean_df = pair_df.groupby("iteration", as_index=False)[metric_cols].mean()
        mean_df["gap"] = mean_df["train_binary_balanced_accuracy"] - mean_df["valid_binary_balanced_accuracy"]
        x_min = int(pair_df["iteration"].min())
        x_max = int(pair_df["iteration"].max())
        y_values = pair_df[metric_cols].to_numpy().astype(float).ravel()
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        pad = max((y_max - y_min) * 0.12, 1e-6)
        y_min -= pad
        y_max += pad
        panel_y = top_margin + panel_idx * panel_h
        plot_y = panel_y + 50

        def x_pos(iteration: int) -> float:
            if x_max == x_min:
                return left + plot_w / 2
            return left + (iteration - x_min) / (x_max - x_min) * plot_w

        def y_pos(value: float) -> float:
            return plot_y + (y_max - value) / (y_max - y_min) * plot_h

        last_gap = float(mean_df["gap"].iloc[-1])
        parts.extend(
            [
                f'<text x="24" y="{panel_y+26}" font-family="Arial" font-size="17" font-weight="700" fill="#111827">{pair}</text>',
                f'<text x="24" y="{panel_y+48}" font-family="Arial" font-size="12" fill="#4b5563">last mean train-valid gap={last_gap:.6f}</text>',
                f'<rect x="{left}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#d1d5db"/>',
                f'<text x="22" y="{plot_y+12}" font-family="Arial" font-size="12" fill="#4b5563">{y_max:.6f}</text>',
                f'<text x="22" y="{plot_y+plot_h}" font-family="Arial" font-size="12" fill="#4b5563">{y_min:.6f}</text>',
            ]
        )
        tick_iterations = mean_df["iteration"].astype(int).tolist()
        if len(tick_iterations) > 13:
            keep = np.linspace(0, len(tick_iterations) - 1, 13).round().astype(int)
            tick_iterations = [tick_iterations[idx] for idx in sorted(set(keep.tolist()))]
        for iteration in tick_iterations:
            x = x_pos(iteration)
            parts.append(
                f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y+plot_h}" stroke="#e5e7eb" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x-10:.2f}" y="{plot_y+plot_h+28}" font-family="Arial" font-size="11" fill="#4b5563">{iteration}</text>'
            )
        for col, color, label in [
            ("train_binary_balanced_accuracy", "#dc2626", "train mean"),
            ("valid_binary_balanced_accuracy", "#2563eb", "valid mean"),
        ]:
            points = " ".join(
                f'{x_pos(int(row["iteration"])):.2f},{y_pos(float(row[col])):.2f}'
                for _, row in mean_df.iterrows()
            )
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
            ly = plot_y + (18 if col.startswith("train") else 40)
            parts.extend(
                [
                    f'<line x1="{left+plot_w-160}" y1="{ly}" x2="{left+plot_w-134}" y2="{ly}" stroke="{color}" stroke-width="3"/>',
                    f'<text x="{left+plot_w-126}" y="{ly+4}" font-family="Arial" font-size="12" fill="#374151">{label}</text>',
                ]
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def diagnostic_iterations(
    best_iteration: int,
    period: int,
    extra_iterations: list[int] | None = None,
) -> list[int]:
    best_iteration = max(1, int(best_iteration))
    period = max(1, int(period))
    extras = [] if extra_iterations is None else [int(value) for value in extra_iterations if value is not None]
    values = sorted(set([1, *range(period, best_iteration + 1, period), best_iteration, *extras]))
    return [value for value in values if value <= best_iteration]


def choose_prediction_iteration(
    args: argparse.Namespace,
    best_iteration: int,
    fold_diagnostics: list[dict],
) -> tuple[int, dict]:
    best_iteration = max(1, int(best_iteration))
    if args.prediction_iteration_policy in {"fixed", "fixed_with_fallback"}:
        if args.fixed_iteration is None:
            raise ValueError(
                "--fixed-iteration is required when --prediction-iteration-policy fixed or fixed_with_fallback."
            )
        iteration = min(max(1, int(args.fixed_iteration)), best_iteration)
        matched = next(
            (row for row in fold_diagnostics if int(row["iteration"]) == iteration),
            {},
        )
        if args.prediction_iteration_policy == "fixed_with_fallback":
            best_row = max(
                fold_diagnostics,
                key=lambda row: (
                    float(row["valid_binary_balanced_accuracy"]),
                    -int(row["iteration"]),
                ),
            )
            fixed_score = float(matched.get("valid_binary_balanced_accuracy", "-inf"))
            best_score = float(best_row["valid_binary_balanced_accuracy"])
            if best_score - fixed_score >= float(args.iteration_fallback_min_gain):
                return int(best_row["iteration"]), best_row
        return iteration, matched

    if args.prediction_iteration_policy == "fold_valid_bacc":
        best_row = max(
            fold_diagnostics,
            key=lambda row: (
                float(row["valid_binary_balanced_accuracy"]),
                -int(row["iteration"]),
            ),
        )
        return int(best_row["iteration"]), best_row

    matched = next(
        (row for row in fold_diagnostics if int(row["iteration"]) == best_iteration),
        {},
    )
    return best_iteration, matched


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = [parse_pair(pair) for pair in args.pairs]

    progress("Loading train/test/sample data")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    progress(f"Building {args.feature_set} feature matrix")
    x_base, y_raw, x_test_base, feature_names = make_xy(train, test, feature_set=args.feature_set)
    train_fe = add_advanced_features(train)
    test_fe = add_advanced_features(test)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw.astype(str))
    classes = encoder.classes_.tolist()
    for left, right in pairs:
        if left not in classes or right not in classes:
            raise ValueError(f"Pair {left}:{right} is not valid for classes {classes}")

    progress("Loading pure OOF/test probabilities")
    cal_oof, cal_test, raw_oof, raw_test, pure_report = load_pure_arrays(classes)
    base_oof_pred = cal_oof.argmax(axis=1)
    base_test_pred = cal_test.argmax(axis=1)
    base_full_score = balanced_accuracy(y, base_oof_pred, len(classes))

    progress("Adding probability features")
    x = add_probability_features(x_base, cal_oof, raw_oof, classes)
    x_test = add_probability_features(x_test_base, cal_test, raw_test, classes)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(cv.split(x, y))[: args.fold_limit]
    if not splits:
        raise ValueError("--fold-limit must be at least 1.")

    model_params = params(args)
    lr_callback = learning_rate_callback(args)
    pair_outputs = {}
    pair_bests = []
    pair_search_rows = []
    diagnostic_rows = []
    combined_pred = base_oof_pred.copy()
    combined_test_pred = base_test_pred.copy()
    combined_covered = np.ones(len(train), dtype=bool) if len(splits) == N_SPLITS else np.zeros(len(train), dtype=bool)

    for left, right in pairs:
        pair_name = f"{left}:{right}"
        progress(f"Training boundary calibrator {pair_name}")
        left_idx = classes.index(left)
        right_idx = classes.index(right)
        pair_train_mask = (y == left_idx) | (y == right_idx)
        binary_y = (y == right_idx).astype("int8")
        right_oof = np.full(len(train), np.nan, dtype=np.float32)
        right_test = np.zeros(len(test), dtype=np.float32)
        fold_rows = []

        for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
            tr_pair_idx = tr_idx[pair_train_mask[tr_idx]]
            va_pair_idx = va_idx[pair_train_mask[va_idx]]
            progress(
                f"{pair_name} fold {fold}/{len(splits)} "
                f"train_pair_rows={len(tr_pair_idx)} valid_pair_rows={len(va_pair_idx)}"
            )
            model = lgb.LGBMClassifier(**model_params)
            callbacks = [
                lgb.early_stopping(args.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(args.log_period),
            ]
            if lr_callback is not None:
                callbacks.append(lr_callback)
            model.fit(
                x.iloc[tr_pair_idx],
                binary_y[tr_pair_idx],
                eval_set=[(x.iloc[va_pair_idx], binary_y[va_pair_idx])],
                eval_metric="binary_logloss",
                callbacks=callbacks,
            )
            best_iteration = int(model.best_iteration_ or args.n_estimators)
            fold_diagnostics = []
            extra_iterations = []
            if args.fixed_iteration is not None:
                extra_iterations.append(args.fixed_iteration)
            for iteration in diagnostic_iterations(best_iteration, args.diagnostic_period, extra_iterations):
                train_right = model.predict_proba(x.iloc[tr_pair_idx], num_iteration=iteration)[:, 1]
                valid_right_pair = model.predict_proba(x.iloc[va_pair_idx], num_iteration=iteration)[:, 1]
                diag_row = {
                    "pair": pair_name,
                    "fold": fold,
                    "iteration": iteration,
                    "train_binary_logloss": float(
                        log_loss(binary_y[tr_pair_idx], train_right, labels=[0, 1])
                    ),
                    "valid_binary_logloss": float(
                        log_loss(binary_y[va_pair_idx], valid_right_pair, labels=[0, 1])
                    ),
                    "train_binary_balanced_accuracy": balanced_accuracy(
                        binary_y[tr_pair_idx],
                        (train_right >= 0.5).astype("int8"),
                        2,
                    ),
                    "valid_binary_balanced_accuracy": balanced_accuracy(
                        binary_y[va_pair_idx],
                        (valid_right_pair >= 0.5).astype("int8"),
                        2,
                    ),
                }
                fold_diagnostics.append(diag_row)
                diagnostic_rows.append(diag_row)
            prediction_iteration, prediction_diag = choose_prediction_iteration(
                args,
                best_iteration,
                fold_diagnostics,
            )
            valid_right = model.predict_proba(x.iloc[va_idx], num_iteration=prediction_iteration)[:, 1]
            right_oof[va_idx] = valid_right
            right_test += model.predict_proba(x_test, num_iteration=prediction_iteration)[:, 1] / len(splits)
            binary_pred = (right_oof[va_pair_idx] >= 0.5).astype("int8")
            binary_score = balanced_accuracy(binary_y[va_pair_idx], binary_pred, 2)
            fold_rows.append(
                {
                    "pair": pair_name,
                    "fold": fold,
                    "binary_balanced_accuracy": float(binary_score),
                    "logloss_best_iteration": best_iteration,
                    "prediction_iteration": prediction_iteration,
                    "prediction_iteration_policy": args.prediction_iteration_policy,
                    "prediction_iteration_valid_binary_balanced_accuracy": prediction_diag.get(
                        "valid_binary_balanced_accuracy"
                    ),
                    "prediction_iteration_valid_binary_logloss": prediction_diag.get("valid_binary_logloss"),
                }
            )
            progress(
                f"{pair_name} fold {fold} binary_balanced_accuracy={binary_score:.6f} "
                f"logloss_best_iteration={best_iteration} "
                f"prediction_iteration={prediction_iteration}"
            )

        covered = ~np.isnan(right_oof)
        if len(splits) != N_SPLITS:
            combined_covered |= covered
        train_masks = segment_masks(train_fe, cal_oof, classes, left, right)
        test_masks = segment_masks(test_fe, cal_test, classes, left, right)
        progress(f"{pair_name} searching OOF override thresholds")
        best, search_top = search_pair_rule(
            y,
            base_oof_pred,
            right_oof,
            train_masks,
            classes,
            left,
            right,
            covered,
            args.to_right_min,
            args.to_right_max,
            args.to_left_min,
            args.to_left_max,
            args.threshold_steps,
        )
        pair_bests.append(best)
        for row in search_top:
            row_flat = row.copy()
            row_flat["class_recalls"] = json.dumps(row_flat["class_recalls"], ensure_ascii=False)
            row_flat["transition_counts"] = json.dumps(row_flat["transition_counts"], ensure_ascii=False)
            pair_search_rows.append(row_flat)

        if best["mask"] != "none" and best["delta"] > 0:
            combined_pred = apply_pair_override(
                combined_pred,
                right_oof,
                train_masks[best["mask"]],
                left_idx,
                right_idx,
                best["to_right_threshold"],
                best["to_left_threshold"],
            )
            combined_test_pred = apply_pair_override(
                combined_test_pred,
                right_test,
                test_masks[best["mask"]],
                left_idx,
                right_idx,
                best["to_right_threshold"],
                best["to_left_threshold"],
            )

        safe_name = pair_name.replace(":", "_")
        np.save(args.output_dir / f"{safe_name}_right_oof.npy", right_oof.astype(np.float32))
        np.save(args.output_dir / f"{safe_name}_right_test.npy", right_test.astype(np.float32))
        pair_outputs[pair_name] = {
            "best": best,
            "fold_scores": fold_rows,
            "oof_path": f"{safe_name}_right_oof.npy",
            "test_path": f"{safe_name}_right_test.npy",
        }

    eval_mask = np.ones(len(train), dtype=bool) if len(splits) == N_SPLITS else combined_covered
    combined_score = balanced_accuracy(y[eval_mask], combined_pred[eval_mask], len(classes))
    base_eval_score = balanced_accuracy(y[eval_mask], base_oof_pred[eval_mask], len(classes))
    accepted = (len(splits) == N_SPLITS and combined_score > base_full_score + args.min_delta) or args.write_even_if_worse

    if pair_search_rows:
        search_df = pd.DataFrame(pair_search_rows).sort_values(
            ["delta", "score", "changed_rows"],
            ascending=[False, False, True],
        )
        search_df.to_csv(args.output_dir / "boundary_pair_search_top.csv", index=False)
        graph_rows = [
            {
                "label": f"{row['pair']} | {row['mask']} | {row['changed_rows']} rows",
                "delta": float(row["delta"]),
            }
            for row in search_df.head(30).to_dict("records")
        ]
        write_bar_svg(
            args.output_dir / "boundary_pair_oof_delta.svg",
            "Boundary Pair OOF Delta",
            graph_rows,
            "label",
            "delta",
        )

    if diagnostic_rows:
        diagnostic_df = pd.DataFrame(diagnostic_rows)
        diagnostic_df.to_csv(args.output_dir / "boundary_pair_training_diagnostics.csv", index=False)
        write_metric_curve_svg(
            args.output_dir / "boundary_pair_valid_logloss_curve.svg",
            "Boundary Pair Valid Binary Logloss",
            diagnostic_rows,
            "valid_binary_logloss",
        )
        write_metric_curve_svg(
            args.output_dir / "boundary_pair_valid_balanced_accuracy_curve.svg",
            "Boundary Pair Valid Binary Balanced Accuracy",
            diagnostic_rows,
            "valid_binary_balanced_accuracy",
        )
        write_metric_curve_svg(
            args.output_dir / "boundary_pair_train_valid_gap_curve.svg",
            "Boundary Pair Train Binary Balanced Accuracy",
            diagnostic_rows,
            "train_binary_balanced_accuracy",
        )
        write_pair_panel_metric_svg(
            args.output_dir / "boundary_pair_valid_logloss_by_pair.svg",
            "Boundary Pair Valid Binary Logloss By Pair",
            diagnostic_rows,
            "valid_binary_logloss",
            "valid logloss",
        )
        write_pair_panel_metric_svg(
            args.output_dir / "boundary_pair_valid_balanced_accuracy_by_pair.svg",
            "Boundary Pair Valid Binary Balanced Accuracy By Pair",
            diagnostic_rows,
            "valid_binary_balanced_accuracy",
            "valid balanced accuracy",
        )
        write_train_valid_panel_svg(
            args.output_dir / "boundary_pair_train_valid_accuracy_by_pair.svg",
            "Boundary Pair Train vs Valid Balanced Accuracy",
            diagnostic_rows,
        )

    report = {
        "purpose": "Boundary-specific OOF calibrators. No public submission CSV is used for training.",
        "classes": classes,
        "pairs": [f"{left}:{right}" for left, right in pairs],
        "fold_limit": len(splits),
        "feature_set": args.feature_set,
        "feature_count": int(x.shape[1]),
        "base_feature_count": len(feature_names),
        "model_params": model_params,
        "lr_schedule": args.lr_schedule,
        "cosine_min_lr": args.cosine_min_lr,
        "prediction_iteration_policy": args.prediction_iteration_policy,
        "fixed_iteration": args.fixed_iteration,
        "threshold_search": {
            "to_right_min": args.to_right_min,
            "to_right_max": args.to_right_max,
            "to_left_min": args.to_left_min,
            "to_left_max": args.to_left_max,
            "threshold_steps": args.threshold_steps,
        },
        "base_full_oof_balanced_accuracy": float(base_full_score),
        "base_eval_oof_balanced_accuracy": float(base_eval_score),
        "combined_eval_oof_balanced_accuracy": float(combined_score),
        "combined_delta": float(combined_score - base_eval_score),
        "accepted_as_candidate": bool(accepted),
        "pair_outputs": pair_outputs,
        "pure_report_best_config": pure_report.get("best_config"),
    }

    if accepted:
        submission = sample.copy()
        submission["class"] = encoder.inverse_transform(combined_test_pred)
        submission_path = args.output_dir / "boundary_pair_calibrated_submission.csv"
        submission.to_csv(submission_path, index=False)
        report["submission_path"] = display_path(submission_path)
        report["submission_class_share"] = submission["class"].value_counts(normalize=True).sort_index().to_dict()
    else:
        stale_submission = args.output_dir / "boundary_pair_calibrated_submission.csv"
        if stale_submission.exists() and not args.write_even_if_worse:
            stale_submission.unlink()
            report["removed_stale_submission_path"] = display_path(stale_submission)

    (args.output_dir / "boundary_pair_calibrator_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
