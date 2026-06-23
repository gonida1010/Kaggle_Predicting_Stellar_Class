from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "artifacts" / ".cache"))
sys.path.append(str(ROOT))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from scripts.build_available_prediction_stacker import (  # noqa: E402
    ARTIFACTS,
    CLASSES,
    DATA,
    ROOT,
    TARGET_MAP,
    archive4_pairs,
    local_file_pairs,
    own_model_pairs,
)
from src.stellar_features import add_advanced_features  # noqa: E402


OUTPUTS = ROOT / "outputs"

DEFAULT_REFERENCE_OOF = (
    ARTIFACTS
    / "te_disagreement_patch_classwise37"
    / "56_high_gi_low_rz_base_galaxy_to_star_c0_55_m0_15_b0_60_oof.npy"
)
DEFAULT_REFERENCE_TEST = (
    ARTIFACTS
    / "te_disagreement_patch_classwise37"
    / "56_high_gi_low_rz_base_galaxy_to_star_c0_55_m0_15_b0_60_test.npy"
)
DEFAULT_START_OOF = (
    ARTIFACTS
    / "classwise_research_blend_84_guard_20260623"
    / "90_PRIVATE_CV_subset_guard_68_plus_84_good_union_oof0970627_oof.npy"
)
DEFAULT_START_TEST = (
    ARTIFACTS
    / "classwise_research_blend_84_guard_20260623"
    / "90_PRIVATE_CV_subset_guard_68_plus_84_good_union_oof0970627_test.npy"
)

EXPLICIT_SOURCES = [
    (
        "56_te_disagreement",
        DEFAULT_REFERENCE_OOF,
        DEFAULT_REFERENCE_TEST,
    ),
    (
        "68_research_material_stack",
        ARTIFACTS / "research_material_stack_20260623" / "research_material_stack_oof.npy",
        ARTIFACTS / "research_material_stack_20260623" / "research_material_stack_test.npy",
    ),
    (
        "69_guarded_research_stack",
        ARTIFACTS
        / "private_cv_stable_research_material_stack_20260623"
        / "private_cv_guarded_01_all_changed_rz_0_2_allconf_oof.npy",
        ARTIFACTS
        / "private_cv_stable_research_material_stack_20260623"
        / "private_cv_guarded_01_all_changed_rz_0_2_allconf_test.npy",
    ),
    (
        "84_classwise_research_blend",
        ARTIFACTS
        / "classwise_research_blend_20260623"
        / "84_PRIVATE_CV_classwise_research_blend_oof0970621_oof.npy",
        ARTIFACTS
        / "classwise_research_blend_20260623"
        / "84_PRIVATE_CV_classwise_research_blend_oof0970621_test.npy",
    ),
    (
        "90_subset_guard_68_plus_84",
        DEFAULT_START_OOF,
        DEFAULT_START_TEST,
    ),
    (
        "85_classwise_research_blend_69start",
        ARTIFACTS
        / "classwise_research_blend_69start_20260623"
        / "85_PRIVATE_CV_classwise_research_blend_69start_oof0970608_oof.npy",
        ARTIFACTS
        / "classwise_research_blend_69start_20260623"
        / "85_PRIVATE_CV_classwise_research_blend_69start_oof0970608_test.npy",
    ),
]


@dataclass
class Source:
    name: str
    oof: np.ndarray
    test: np.ndarray
    score: float
    pred: np.ndarray
    test_pred: np.ndarray
    correct: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build next private/CV candidates from the current best OOF source. "
            "This uses only train labels, OOF probabilities, and test probabilities. "
            "No public LB score is read."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "robust_private_cv_next_20260623")
    parser.add_argument("--reference-oof", type=Path, default=DEFAULT_REFERENCE_OOF)
    parser.add_argument("--reference-test", type=Path, default=DEFAULT_REFERENCE_TEST)
    parser.add_argument("--reference-name", default="56_te_disagreement")
    parser.add_argument("--start-oof", type=Path, default=DEFAULT_START_OOF)
    parser.add_argument("--start-test", type=Path, default=DEFAULT_START_TEST)
    parser.add_argument("--start-name", default="90_subset_guard_68_plus_84")
    parser.add_argument("--max-sources", type=int, default=42)
    parser.add_argument("--min-source-oof", type=float, default=0.965)
    parser.add_argument("--alpha-max", type=float, default=0.055)
    parser.add_argument("--alpha-steps", type=int, default=11)
    parser.add_argument("--scan-top", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output-rank-start", type=int, default=94)
    parser.add_argument("--output-prefix", default="PRIVATE_CV_robust_next")
    parser.add_argument(
        "--include-lower-than-start",
        action="store_true",
        help="Write top robust candidates even if OOF does not beat the start source.",
    )
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


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        label: float((y_pred[y_true == idx] == idx).mean())
        for idx, label in enumerate(CLASSES)
    }


def class_report_rows(candidate: str, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
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
                "candidate": candidate,
                "class": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return rows


def transition_counts(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    changed = before != after
    counts = Counter(
        f"{CLASSES[int(old)]}->{CLASSES[int(new)]}"
        for old, new in zip(before[changed], after[changed])
    )
    return dict(sorted(counts.items()))


def changed_outcomes(y: np.ndarray, before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    changed = before != after
    counts: Counter[str] = Counter()
    for true, old, new in zip(y[changed], before[changed], after[changed]):
        old_ok = old == true
        new_ok = new == true
        if (not old_ok) and new_ok:
            counts["fixed"] += 1
        elif old_ok and (not new_ok):
            counts["broken"] += 1
        elif (not old_ok) and (not new_ok):
            counts["still_wrong"] += 1
        else:
            counts["both_right_changed"] += 1
    return dict(sorted(counts.items()))


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def quantile_bins(train_col: pd.Series, test_col: pd.Series, bins: int) -> tuple[pd.Series, pd.Series]:
    _, edges = pd.qcut(train_col, q=bins, retbins=True, duplicates="drop")
    edges = np.unique(edges)
    if len(edges) <= 2:
        return (
            pd.Series(np.zeros(len(train_col), dtype=np.int16), index=train_col.index),
            pd.Series(np.zeros(len(test_col), dtype=np.int16), index=test_col.index),
        )
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = range(len(edges) - 1)
    train_bins = pd.cut(train_col, bins=edges, labels=labels, include_lowest=True).astype("int16")
    test_bins = pd.cut(test_col, bins=edges, labels=labels, include_lowest=True).astype("int16")
    return train_bins, test_bins


def build_feature_bins(train: pd.DataFrame, test: pd.DataFrame, bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_fe = add_advanced_features(train).copy()
    test_fe = add_advanced_features(test).copy()
    for col in ["redshift", "g-i", "u-r", "mag_range"]:
        train_fe[f"{col}_bin"], test_fe[f"{col}_bin"] = quantile_bins(train_fe[col], test_fe[col], bins)
    for frame in (train_fe, test_fe):
        frame["spectral_population"] = (
            frame["spectral_type"].astype(str) + "_" + frame["galaxy_population"].astype(str)
        )
    return train_fe, test_fe


def make_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    spectral = frame["spectral_type"].astype(str)
    spectral_pop = frame["spectral_population"].astype(str)
    rz = frame["redshift_bin"]
    gi = frame["g-i_bin"]
    ur = frame["u-r_bin"]
    mr = frame["mag_range_bin"]

    masks = {
        "weak_ur6": ur.eq(6),
        "weak_ob": spectral.eq("O/B") | spectral_pop.eq("O/B_Blue_Cloud"),
        "weak_gk_red": spectral_pop.eq("G/K_Red_Sequence"),
        "weak_mag2": mr.eq(2),
        "weak_gi2": gi.eq(2),
        "good_rz2": rz.eq(2),
        "good_af_blue": spectral_pop.eq("A/F_Blue_Cloud"),
        "good_ur1": ur.eq(1),
        "good_mag0": mr.eq(0),
        "good_high_gi": gi.ge(7),
        "good_m": spectral.eq("M") | spectral_pop.eq("M_Blue_Cloud"),
    }
    masks["weak_core"] = (
        masks["weak_ur6"]
        | masks["weak_ob"]
        | masks["weak_gk_red"]
        | masks["weak_mag2"]
        | masks["weak_gi2"]
    )
    masks["good_core"] = (
        masks["good_rz2"]
        | masks["good_af_blue"]
        | masks["good_ur1"]
        | masks["good_mag0"]
        | masks["good_high_gi"]
        | masks["good_m"]
    )
    masks["good_not_weak"] = masks["good_core"] & ~masks["weak_core"]
    return {name: mask.to_numpy() for name, mask in masks.items()}


def source_from_arrays(name: str, oof: np.ndarray, test: np.ndarray, y: np.ndarray) -> Source:
    pred = oof.argmax(axis=1)
    test_pred = test.argmax(axis=1)
    return Source(
        name=name,
        oof=oof,
        test=test,
        score=balanced_accuracy(y, pred),
        pred=pred,
        test_pred=test_pred,
        correct=(pred == y).astype(np.float32),
    )


def load_explicit_sources(y: np.ndarray, n_train: int, n_test: int) -> list[Source]:
    sources = []
    seen = set()
    for name, oof_path, test_path in EXPLICIT_SOURCES:
        if name in seen or not oof_path.exists() or not test_path.exists():
            continue
        oof = load_proba(oof_path, n_train)
        test = load_proba(test_path, n_test)
        sources.append(source_from_arrays(name, oof, test, y))
        seen.add(name)
    return sources


def load_available_sources(y: np.ndarray, n_train: int, n_test: int) -> list[Source]:
    records = []
    records.extend(archive4_pairs(n_train, n_test))
    records.extend(local_file_pairs(n_train, n_test))
    records.extend(own_model_pairs(n_train, n_test))
    out = []
    seen = set()
    for record in records:
        name = str(record["name"])
        if name in seen:
            continue
        try:
            out.append(
                source_from_arrays(
                    name,
                    normalize_probs(record["oof"]),
                    normalize_probs(record["test"]),
                    y,
                )
            )
            seen.add(name)
        except Exception as exc:
            progress(f"skip source {name}: {exc}")
    return out


def unique_sources(sources: list[Source]) -> list[Source]:
    out = []
    seen = set()
    for source in sources:
        if source.name in seen:
            continue
        seen.add(source.name)
        out.append(source)
    return out


def class_column_blend(current: np.ndarray, source: np.ndarray, class_idx: int, alpha: float) -> np.ndarray:
    out = current.copy()
    out[:, class_idx] = (1.0 - alpha) * out[:, class_idx] + alpha * source[:, class_idx]
    return normalize_probs(out)


def replace_where(base: np.ndarray, fill: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = base.copy()
    out[mask] = fill[mask]
    return normalize_probs(out)


def subset_rows(
    y: np.ndarray,
    reference_pred: np.ndarray,
    pred: np.ndarray,
    masks: dict[str, np.ndarray],
    candidate_name: str,
) -> list[dict]:
    rows = []
    for name, mask in masks.items():
        idx = np.flatnonzero(mask)
        if len(idx) < 300:
            continue
        ref_score = balanced_accuracy(y[idx], reference_pred[idx])
        score = balanced_accuracy(y[idx], pred[idx])
        rows.append(
            {
                "candidate": candidate_name,
                "subset": name,
                "count": int(len(idx)),
                "reference_bac": ref_score,
                "candidate_bac": score,
                "delta_vs_reference": score - ref_score,
                "changed_rows_vs_reference": int((pred[idx] != reference_pred[idx]).sum()),
            }
        )
    return rows


def meta_fold_stability(
    y: np.ndarray,
    reference_pred: np.ndarray,
    start_pred: np.ndarray,
    pred: np.ndarray,
    folds: int,
    seeds: int,
) -> dict[str, float]:
    deltas_ref = []
    deltas_start = []
    for seed in range(20260623, 20260623 + seeds):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for _, valid_idx in splitter.split(np.zeros(len(y)), y):
            score = balanced_accuracy(y[valid_idx], pred[valid_idx])
            ref_score = balanced_accuracy(y[valid_idx], reference_pred[valid_idx])
            start_score = balanced_accuracy(y[valid_idx], start_pred[valid_idx])
            deltas_ref.append(score - ref_score)
            deltas_start.append(score - start_score)
    ref_arr = np.asarray(deltas_ref, dtype=np.float64)
    start_arr = np.asarray(deltas_start, dtype=np.float64)
    return {
        "meta_fold_mean_delta_vs_reference": float(ref_arr.mean()),
        "meta_fold_min_delta_vs_reference": float(ref_arr.min()),
        "meta_fold_positive_rate_vs_reference": float((ref_arr > 0).mean()),
        "meta_fold_mean_delta_vs_start": float(start_arr.mean()),
        "meta_fold_min_delta_vs_start": float(start_arr.min()),
        "meta_fold_positive_rate_vs_start": float((start_arr > 0).mean()),
    }


def evaluate_scan_candidate(
    name: str,
    y: np.ndarray,
    reference: Source,
    start: Source,
    oof: np.ndarray,
    test: np.ndarray,
    masks: dict[str, np.ndarray],
    operation: dict,
) -> dict:
    pred = oof.argmax(axis=1)
    test_pred = test.argmax(axis=1)
    score = balanced_accuracy(y, pred)
    recalls = class_recalls(y, pred)
    ref_recalls = class_recalls(y, reference.pred)
    start_recalls = class_recalls(y, start.pred)
    recall_delta_ref = {label: recalls[label] - ref_recalls[label] for label in CLASSES}
    recall_delta_start = {label: recalls[label] - start_recalls[label] for label in CLASSES}
    subset = subset_rows(y, reference.pred, pred, masks, name)
    worst_subset_delta = min((row["delta_vs_reference"] for row in subset), default=0.0)
    quick_score = (
        (score - reference.score)
        + 0.15 * min(0.0, min(recall_delta_ref.values()))
        + 0.10 * min(0.0, worst_subset_delta)
        - 2e-8 * int((test_pred != start.test_pred).sum())
    )
    row = {
        "name": name,
        "operation": operation["operation"],
        "source": operation.get("source", ""),
        "class": operation.get("class", ""),
        "alpha": operation.get("alpha", 0.0),
        "mask": operation.get("mask", ""),
        "fallback": operation.get("fallback", ""),
        "oof_balanced_accuracy": score,
        "delta_vs_reference": score - reference.score,
        "delta_vs_start": score - start.score,
        "changed_rows_vs_reference": int((pred != reference.pred).sum()),
        "changed_rows_vs_start": int((pred != start.pred).sum()),
        "test_changed_rows_vs_reference": int((test_pred != reference.test_pred).sum()),
        "test_changed_rows_vs_start": int((test_pred != start.test_pred).sum()),
        "worst_subset_delta_vs_reference": float(worst_subset_delta),
        "worst_class_recall_delta_vs_reference": float(min(recall_delta_ref.values())),
        "worst_class_recall_delta_vs_start": float(min(recall_delta_start.values())),
        "quick_robust_score": float(quick_score),
        "class_recalls": recalls,
        "class_recall_delta_vs_reference": recall_delta_ref,
        "transition_counts_vs_start": transition_counts(start.pred, pred),
        "changed_outcomes_vs_start": changed_outcomes(y, start.pred, pred),
    }
    return row


def final_robust_score(row: dict) -> float:
    return float(
        row["delta_vs_reference"]
        + 0.35 * row["meta_fold_min_delta_vs_reference"]
        + 0.20 * min(0.0, row["worst_subset_delta_vs_reference"])
        + 0.15 * min(0.0, row["worst_class_recall_delta_vs_reference"])
        - 2e-8 * row["test_changed_rows_vs_start"]
    )


def make_submission(path: Path, sample: pd.DataFrame, proba: np.ndarray) -> None:
    submission = sample.copy()
    submission["class"] = np.array(CLASSES)[proba.argmax(axis=1)]
    submission.to_csv(path, index=False)


def source_summary_rows(sources: list[Source], start: Source) -> list[dict]:
    rows = []
    for source in sources:
        rows.append(
            {
                "name": source.name,
                "oof_balanced_accuracy": source.score,
                "pred_agreement_vs_start": float((source.pred == start.pred).mean()),
                "correct_corr_vs_start": safe_corr(source.correct, start.correct),
                "prob_corr_mean_vs_start": float(
                    np.mean([safe_corr(source.oof[:, idx], start.oof[:, idx]) for idx in range(len(CLASSES))])
                ),
                "diversity_vs_start": float(
                    (1.0 - max(0.0, safe_corr(source.correct, start.correct)))
                    * (1.0 - float((source.pred == start.pred).mean()))
                ),
            }
        )
    return rows


def save_graphs(summary: pd.DataFrame, final_summary: pd.DataFrame, output_dir: Path) -> None:
    if not summary.empty:
        top = summary.sort_values("oof_balanced_accuracy", ascending=False).head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(top) + 1.5)), dpi=160)
        ax.barh(top["name"], top["oof_balanced_accuracy"], color="#2563eb")
        ax.set_title("Available OOF Source Score")
        ax.set_xlabel("OOF balanced accuracy")
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        for y_idx, value in enumerate(top["oof_balanced_accuracy"]):
            ax.text(value + 0.00004, y_idx, f"{value:.6f}", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "source_oof_score_rank.png")
        fig.savefig(output_dir / "source_oof_score_rank.svg")
        plt.close(fig)

    if not final_summary.empty:
        top = final_summary.sort_values("final_robust_score", ascending=False).head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(13, max(6, 0.42 * len(top) + 1.5)), dpi=160)
        ax.barh(top["name"], top["final_robust_score"], color="#16a34a")
        ax.axvline(0, color="#dc2626", linestyle="--", linewidth=1)
        ax.set_title("Robust Candidate Ranking")
        ax.set_xlabel("robust score")
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(output_dir / "robust_candidate_rank.png")
        fig.savefig(output_dir / "robust_candidate_rank.svg")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)

    progress("Loading train/test/sample")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["class"].map(TARGET_MAP).to_numpy()

    progress("Loading reference and start candidates")
    reference = source_from_arrays(
        args.reference_name,
        load_proba(args.reference_oof, len(train)),
        load_proba(args.reference_test, len(sample)),
        y,
    )
    start = source_from_arrays(
        args.start_name,
        load_proba(args.start_oof, len(train)),
        load_proba(args.start_test, len(sample)),
        y,
    )
    progress(f"reference={reference.name} OOF={reference.score:.9f}")
    progress(f"start={start.name} OOF={start.score:.9f}")

    progress("Building subset masks")
    train_fe, test_fe = build_feature_bins(train, test, args.bins)
    train_masks = make_masks(train_fe)
    test_masks = make_masks(test_fe)

    progress("Loading available OOF/test sources")
    sources = unique_sources(load_explicit_sources(y, len(train), len(sample)) + load_available_sources(y, len(train), len(sample)))
    sources = [source for source in sources if source.score >= args.min_source_oof]
    sources = sorted(
        sources,
        key=lambda source: (
            source.score,
            (1.0 - max(0.0, safe_corr(source.correct, start.correct))) * (1.0 - float((source.pred == start.pred).mean())),
        ),
        reverse=True,
    )[: args.max_sources]
    source_summary = pd.DataFrame(source_summary_rows(sources, start)).sort_values(
        ["oof_balanced_accuracy", "diversity_vs_start"], ascending=[False, False]
    )
    source_summary.to_csv(output_dir / "source_summary.csv", index=False)
    progress(f"using {len(sources)} sources after OOF/source cap")

    scan_rows = []
    scan_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    alphas = np.linspace(args.alpha_max / args.alpha_steps, args.alpha_max, args.alpha_steps)

    progress("Scanning class-wise source blends")
    for source in sources:
        if source.name == start.name:
            continue
        for class_idx, class_label in enumerate(CLASSES):
            for alpha in alphas:
                name = f"classblend_{source.name}_{class_label}_a{alpha:.4f}".replace(".", "p")
                oof = class_column_blend(start.oof, source.oof, class_idx, float(alpha))
                test_proba = class_column_blend(start.test, source.test, class_idx, float(alpha))
                row = evaluate_scan_candidate(
                    name=name,
                    y=y,
                    reference=reference,
                    start=start,
                    oof=oof,
                    test=test_proba,
                    masks=train_masks,
                    operation={
                        "operation": "classblend",
                        "source": source.name,
                        "class": class_label,
                        "alpha": float(alpha),
                    },
                )
                scan_rows.append(row)
                scan_arrays[name] = (oof, test_proba)

    progress("Scanning weak-subset rollbacks")
    fallback_by_name = {source.name: source for source in sources}
    for fallback_name in ["84_classwise_research_blend", "68_research_material_stack", "56_te_disagreement", "69_guarded_research_stack"]:
        fallback = fallback_by_name.get(fallback_name)
        if fallback is None:
            continue
        changed_train = start.pred != fallback.pred
        changed_test = start.test_pred != fallback.test_pred
        for mask_name in ["weak_core", "weak_ur6", "weak_ob", "weak_gk_red", "weak_mag2", "weak_gi2"]:
            train_mask = changed_train & train_masks[mask_name]
            test_mask = changed_test & test_masks[mask_name]
            if not train_mask.any() and not test_mask.any():
                continue
            name = f"rollback_{mask_name}_to_{fallback_name}"
            oof = replace_where(start.oof, fallback.oof, train_mask)
            test_proba = replace_where(start.test, fallback.test, test_mask)
            row = evaluate_scan_candidate(
                name=name,
                y=y,
                reference=reference,
                start=start,
                oof=oof,
                test=test_proba,
                masks=train_masks,
                operation={
                    "operation": "rollback",
                    "fallback": fallback.name,
                    "mask": mask_name,
                },
            )
            scan_rows.append(row)
            scan_arrays[name] = (oof, test_proba)

    progress("Scanning good-subset transfers")
    for base_name in ["84_classwise_research_blend", "68_research_material_stack", "56_te_disagreement", "69_guarded_research_stack"]:
        base = fallback_by_name.get(base_name)
        if base is None or base.name == start.name:
            continue
        changed_train = start.pred != base.pred
        changed_test = start.test_pred != base.test_pred
        for mask_name in ["good_core", "good_not_weak", "good_rz2", "good_af_blue", "good_ur1", "good_m"]:
            train_mask = changed_train & train_masks[mask_name]
            test_mask = changed_test & test_masks[mask_name]
            if not train_mask.any() and not test_mask.any():
                continue
            name = f"{base_name}_plus90_{mask_name}"
            oof = replace_where(base.oof, start.oof, train_mask)
            test_proba = replace_where(base.test, start.test, test_mask)
            row = evaluate_scan_candidate(
                name=name,
                y=y,
                reference=reference,
                start=start,
                oof=oof,
                test=test_proba,
                masks=train_masks,
                operation={
                    "operation": "good_transfer",
                    "fallback": base.name,
                    "mask": mask_name,
                },
            )
            scan_rows.append(row)
            scan_arrays[name] = (oof, test_proba)

    scan_df = pd.DataFrame(
        [
            {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for row in scan_rows
        ]
    ).sort_values(["oof_balanced_accuracy", "quick_robust_score"], ascending=[False, False])
    scan_df.to_csv(output_dir / "scan_candidates.csv", index=False)
    if scan_df.empty:
        raise RuntimeError("No scan candidates were generated.")

    progress("Auditing top scanned candidates with repeated meta-folds")
    final_rows = []
    subset_detail_rows = []
    class_detail_rows = []
    detailed_reports = []
    top_names = scan_df.head(args.scan_top)["name"].tolist()
    for idx, name in enumerate(top_names, start=1):
        oof, test_proba = scan_arrays[name]
        row = next(row for row in scan_rows if row["name"] == name)
        pred = oof.argmax(axis=1)
        row.update(meta_fold_stability(y, reference.pred, start.pred, pred, args.folds, args.seeds))
        row["final_robust_score"] = final_robust_score(row)
        final_rows.append(row)
        subset_detail_rows.extend(subset_rows(y, reference.pred, pred, train_masks, name))
        class_detail_rows.extend(class_report_rows(name, y, pred))
        detailed_reports.append(row)
        progress(
            f"audit {idx}/{len(top_names)} {name}: "
            f"OOF={row['oof_balanced_accuracy']:.9f}, "
            f"delta_start={row['delta_vs_start']:+.9f}, "
            f"robust={row['final_robust_score']:+.9f}"
        )

    final_df = pd.DataFrame(
        [
            {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for row in final_rows
        ]
    ).sort_values(["final_robust_score", "oof_balanced_accuracy"], ascending=[False, False])
    final_df.to_csv(output_dir / "audited_candidates.csv", index=False)
    pd.DataFrame(subset_detail_rows).to_csv(output_dir / "audited_subset_deltas.csv", index=False)
    pd.DataFrame(class_detail_rows).to_csv(output_dir / "audited_class_report.csv", index=False)

    write_df = final_df
    if not args.include_lower_than_start:
        write_df = write_df[write_df["oof_balanced_accuracy"] >= start.score]
    output_rows = []
    for offset, row in enumerate(write_df.head(args.top_k).to_dict(orient="records")):
        name = row["name"]
        oof, test_proba = scan_arrays[name]
        score_tag = f"{row['oof_balanced_accuracy']:.6f}".replace(".", "")
        safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(name))[:96]
        output_name = f"{args.output_rank_start + offset:02d}_{args.output_prefix}_{safe_name}_oof{score_tag}"
        np.save(output_dir / f"{output_name}_oof.npy", oof.astype(np.float32))
        np.save(output_dir / f"{output_name}_test.npy", test_proba.astype(np.float32))
        make_submission(output_dir / f"{output_name}.csv", sample, test_proba)
        make_submission(OUTPUTS / f"{output_name}.csv", sample, test_proba)
        output_rows.append(
            {
                **row,
                "artifact_csv": str((output_dir / f"{output_name}.csv").relative_to(ROOT)),
                "output_csv": str((OUTPUTS / f"{output_name}.csv").relative_to(ROOT)),
            }
        )
        progress(f"wrote {OUTPUTS / f'{output_name}.csv'}")

    pd.DataFrame(output_rows).to_csv(output_dir / "output_candidates.csv", index=False)
    save_graphs(source_summary, final_df, output_dir)
    report = {
        "purpose": "Next private/CV robust candidates after 90. Public LB is not used.",
        "reference": {
            "name": reference.name,
            "oof_balanced_accuracy": reference.score,
        },
        "start": {
            "name": start.name,
            "oof_balanced_accuracy": start.score,
        },
        "source_count": len(sources),
        "scan_candidate_count": len(scan_rows),
        "audited_candidate_count": len(final_rows),
        "written_candidate_count": len(output_rows),
        "top_audited": final_df.head(20).to_dict(orient="records"),
        "outputs": [
            "source_summary.csv",
            "scan_candidates.csv",
            "audited_candidates.csv",
            "audited_subset_deltas.csv",
            "audited_class_report.csv",
            "output_candidates.csv",
            "source_oof_score_rank.png",
            "robust_candidate_rank.png",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
