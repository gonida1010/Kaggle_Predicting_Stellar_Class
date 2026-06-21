from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "artifacts" / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from build_available_prediction_stacker import CLASSES, DATA, TARGET_MAP

import sys

sys.path.append(str(ROOT))
from src.stellar_features import add_advanced_features  # noqa: E402


ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"
OUT_DIR = ARTIFACTS / "subset_guarded_candidates"


@dataclass(frozen=True)
class GuardSpec:
    name: str
    mask_name: str
    transition_rule: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build subset-guarded variants of a high-OOF candidate. "
            "The guard removes candidate changes in OOF-detected weak feature regions."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--base-oof", type=Path, default=ROOT / "external_sources/oof_test_predictions/oof_lr_stacker_v9.npy")
    parser.add_argument("--base-test", type=Path, default=ROOT / "external_preds/pred_lr_stacker_v9.npy")
    parser.add_argument(
        "--candidate-oof",
        type=Path,
        default=ARTIFACTS / "oof_generalization_stack_with_classwise_blender_fast/generalization_stack_oof.npy",
    )
    parser.add_argument(
        "--candidate-test",
        type=Path,
        default=ARTIFACTS / "oof_generalization_stack_with_classwise_blender_fast/generalization_stack_test.npy",
    )
    parser.add_argument("--candidate-name", default="37_classwise_greedy")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output-rank-start", type=int, default=44)
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
    if arr.shape != (expected_rows, len(CLASSES)):
        raise ValueError(f"{path} shape {arr.shape}, expected {(expected_rows, len(CLASSES))}")
    return normalize_probs(arr)


def quantile_bins(train_col: pd.Series, test_col: pd.Series, bins: int) -> tuple[pd.Series, pd.Series]:
    _, edges = pd.qcut(train_col, q=bins, retbins=True, duplicates="drop")
    edges = np.unique(edges)
    if len(edges) <= 2:
        return (
            pd.Series(np.zeros(len(train_col), dtype=int), index=train_col.index),
            pd.Series(np.zeros(len(test_col), dtype=int), index=test_col.index),
        )
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = range(len(edges) - 1)
    return (
        pd.cut(train_col, bins=edges, labels=labels, include_lowest=True).astype(int),
        pd.cut(test_col, bins=edges, labels=labels, include_lowest=True).astype(int),
    )


def build_features(train: pd.DataFrame, test: pd.DataFrame, bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_fe = add_advanced_features(train).copy()
    test_fe = add_advanced_features(test).copy()
    for col in ["redshift", "g-i", "u-r", "mag_range"]:
        train_fe[f"{col}_bin"], test_fe[f"{col}_bin"] = quantile_bins(train_fe[col], test_fe[col], bins)
    train_fe["spectral_population"] = train_fe["spectral_type"].astype(str) + "_" + train_fe["galaxy_population"].astype(str)
    test_fe["spectral_population"] = test_fe["spectral_type"].astype(str) + "_" + test_fe["galaxy_population"].astype(str)
    return train_fe, test_fe


def build_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    masks = {
        "none": np.zeros(len(frame), dtype=bool),
        "g_i_bin_0": frame["g-i_bin"].eq(0).to_numpy(),
        "g_i_bin_2": frame["g-i_bin"].eq(2).to_numpy(),
        "g_i_bin_0_2": frame["g-i_bin"].isin([0, 2]).to_numpy(),
        "g_i_bin_0_1_2": frame["g-i_bin"].isin([0, 1, 2]).to_numpy(),
        "mag_range_bin_0": frame["mag_range_bin"].eq(0).to_numpy(),
        "mag_range_bin_0_1_2": frame["mag_range_bin"].isin([0, 1, 2]).to_numpy(),
        "spectral_O_B": frame["spectral_type"].astype(str).eq("O/B").to_numpy(),
        "spectral_A_F": frame["spectral_type"].astype(str).eq("A/F").to_numpy(),
        "O_B_Blue_Cloud": frame["spectral_population"].eq("O/B_Blue_Cloud").to_numpy(),
        "A_F_Blue_Cloud": frame["spectral_population"].eq("A/F_Blue_Cloud").to_numpy(),
        "low_gi_or_low_mag": (
            frame["g-i_bin"].isin([0, 2]) | frame["mag_range_bin"].isin([0, 1, 2])
        ).to_numpy(),
        "weak_core": (
            frame["g-i_bin"].isin([0, 2])
            | frame["mag_range_bin"].eq(0)
            | frame["spectral_population"].isin(["O/B_Blue_Cloud", "A/F_Blue_Cloud"])
        ).to_numpy(),
        "weak_core_plus": (
            frame["g-i_bin"].isin([0, 1, 2])
            | frame["mag_range_bin"].isin([0, 1, 2])
            | frame["spectral_population"].isin(["O/B_Blue_Cloud", "A/F_Blue_Cloud"])
        ).to_numpy(),
    }
    masks["high_gain_region"] = (
        frame["redshift_bin"].isin([0, 1, 2, 3])
        | frame["g-i_bin"].isin([7, 8, 9])
        | frame["spectral_population"].isin(["O/B_Red_Sequence", "M_Blue_Cloud"])
    ).to_numpy()
    return masks


def transition_mask(base_pred: np.ndarray, cand_pred: np.ndarray, rule: str) -> np.ndarray:
    changed = base_pred != cand_pred
    base_galaxy = base_pred == 0
    base_qso = base_pred == 1
    base_star = base_pred == 2
    cand_galaxy = cand_pred == 0
    cand_qso = cand_pred == 1
    cand_star = cand_pred == 2
    if rule == "all_changed":
        return changed
    if rule == "base_galaxy_to_non_galaxy":
        return changed & base_galaxy & ~cand_galaxy
    if rule == "base_star_to_galaxy":
        return changed & base_star & cand_galaxy
    if rule == "base_star_to_qso":
        return changed & base_star & cand_qso
    if rule == "base_nonstar_to_star":
        return changed & ~base_star & cand_star
    if rule == "base_galaxy_to_star":
        return changed & base_galaxy & cand_star
    if rule == "base_galaxy_to_qso":
        return changed & base_galaxy & cand_qso
    if rule == "base_qso_to_non_qso":
        return changed & base_qso & ~cand_qso
    raise ValueError(f"Unknown transition rule: {rule}")


def apply_guard(
    base_proba: np.ndarray,
    cand_proba: np.ndarray,
    feature_mask: np.ndarray,
    trans_mask: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    base_pred = base_proba.argmax(axis=1)
    cand_pred = cand_proba.argmax(axis=1)
    changed = base_pred != cand_pred
    if mode == "drop_weak":
        keep_candidate = changed & ~(feature_mask & trans_mask)
    elif mode == "keep_gain":
        keep_candidate = changed & feature_mask & trans_mask
    else:
        raise ValueError(mode)
    out = base_proba.copy()
    out[keep_candidate] = cand_proba[keep_candidate]
    return normalize_probs(out), keep_candidate


def class_recalls(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        label: float((pred[y == idx] == idx).mean())
        for idx, label in enumerate(CLASSES)
    }


def transition_counts(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    changed = before != after
    counts = Counter(f"{CLASSES[int(b)]}->{CLASSES[int(a)]}" for b, a in zip(before[changed], after[changed]))
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
    return dict(counts)


def meta_fold_stats(y: np.ndarray, base_pred: np.ndarray, pred: np.ndarray, folds: int, seeds: int) -> dict[str, float]:
    deltas = []
    for seed in range(20260620, 20260620 + seeds):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for _, valid_idx in splitter.split(np.zeros(len(y)), y):
            base_score = balanced_accuracy_score(y[valid_idx], base_pred[valid_idx])
            score = balanced_accuracy_score(y[valid_idx], pred[valid_idx])
            deltas.append(float(score - base_score))
    arr = np.array(deltas)
    return {
        "meta_fold_mean_delta": float(arr.mean()),
        "meta_fold_min_delta": float(arr.min()),
        "meta_fold_max_delta": float(arr.max()),
        "meta_fold_positive_rate": float((arr > 0).mean()),
    }


def make_submission(path: Path, sample: pd.DataFrame, proba: np.ndarray) -> None:
    submission = sample.copy()
    submission["class"] = np.array(CLASSES)[proba.argmax(axis=1)]
    submission.to_csv(path, index=False)


def save_bar(df: pd.DataFrame, path: Path) -> None:
    view = df.sort_values("oof_balanced_accuracy", ascending=True).tail(16)
    labels = view["name"].str.replace("__AND__", "\n+ ", regex=False).str.replace("drop_", "", regex=False)
    fig, ax = plt.subplots(figsize=(13, max(6, len(view) * 0.55)), dpi=180)
    ax.barh(labels, view["oof_balanced_accuracy"], color="#2563eb")
    for y, v in enumerate(view["oof_balanced_accuracy"]):
        ax.text(v + 0.000001, y, f"{v:.6f}", va="center", fontsize=8)
    ax.set_xlabel("OOF balanced accuracy")
    ax.set_title("Subset-Guarded Candidate Search", weight="bold")
    xmin = max(0.0, float(view["oof_balanced_accuracy"].min()) - 0.00003)
    xmax = float(view["oof_balanced_accuracy"].max()) + 0.000015
    ax.set_xlim(xmin, xmax)
    ax.grid(True, axis="x", color="#e5e7eb")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["class"].map(TARGET_MAP).to_numpy()
    train_fe, test_fe = build_features(train, test, args.bins)

    base_oof = load_proba(args.base_oof, len(train))
    base_test = load_proba(args.base_test, len(sample))
    cand_oof = load_proba(args.candidate_oof, len(train))
    cand_test = load_proba(args.candidate_test, len(sample))
    base_pred = base_oof.argmax(axis=1)
    cand_pred = cand_oof.argmax(axis=1)
    base_test_pred = base_test.argmax(axis=1)
    cand_test_pred = cand_test.argmax(axis=1)
    base_score = float(balanced_accuracy_score(y, base_pred))
    cand_score = float(balanced_accuracy_score(y, cand_pred))
    progress(f"base OOF={base_score:.9f}; candidate OOF={cand_score:.9f}; changed={int((base_pred != cand_pred).sum())}")

    train_masks = build_masks(train_fe)
    test_masks = build_masks(test_fe)
    specs: list[GuardSpec] = []
    for mask_name in [
        "g_i_bin_0",
        "g_i_bin_2",
        "g_i_bin_0_2",
        "mag_range_bin_0",
        "mag_range_bin_0_1_2",
        "O_B_Blue_Cloud",
        "A_F_Blue_Cloud",
        "low_gi_or_low_mag",
        "weak_core",
        "weak_core_plus",
    ]:
        for rule in [
            "all_changed",
            "base_galaxy_to_non_galaxy",
            "base_galaxy_to_star",
            "base_nonstar_to_star",
            "base_star_to_galaxy",
            "base_qso_to_non_qso",
        ]:
            specs.append(GuardSpec(f"drop_{mask_name}_{rule}", mask_name, rule))
    for rule in ["all_changed", "base_star_to_galaxy", "base_nonstar_to_star", "base_star_to_qso"]:
        specs.append(GuardSpec(f"keep_high_gain_region_{rule}", "high_gain_region", rule))

    rows = []
    candidates = []
    # Single guards plus selected pairwise unions.
    spec_groups: list[tuple[str, list[GuardSpec], str]] = [(spec.name, [spec], "drop_weak") for spec in specs if spec.name.startswith("drop_")]
    spec_groups.extend((spec.name, [spec], "keep_gain") for spec in specs if spec.name.startswith("keep_"))
    important = [spec for spec in specs if spec.name.startswith("drop_weak_core") or spec.name.startswith("drop_g_i_bin_0") or spec.name.startswith("drop_mag_range_bin_0")]
    for a, b in combinations(important, 2):
        spec_groups.append((f"{a.name}__AND__{b.name}", [a, b], "drop_weak"))

    for idx_group, (name, group, mode) in enumerate(spec_groups, start=1):
        if idx_group == 1 or idx_group % 25 == 0 or idx_group == len(spec_groups):
            progress(f"evaluating guard {idx_group}/{len(spec_groups)}")
        train_feature_mask = np.zeros(len(train), dtype=bool)
        test_feature_mask = np.zeros(len(test), dtype=bool)
        train_trans_mask = np.zeros(len(train), dtype=bool)
        test_trans_mask = np.zeros(len(test), dtype=bool)
        for spec in group:
            train_feature_mask |= train_masks[spec.mask_name]
            test_feature_mask |= test_masks[spec.mask_name]
            train_trans_mask |= transition_mask(base_pred, cand_pred, spec.transition_rule)
            test_trans_mask |= transition_mask(base_test_pred, cand_test_pred, spec.transition_rule)

        guarded_oof, keep_train = apply_guard(base_oof, cand_oof, train_feature_mask, train_trans_mask, mode)
        guarded_test, keep_test = apply_guard(base_test, cand_test, test_feature_mask, test_trans_mask, mode)
        pred = guarded_oof.argmax(axis=1)
        test_pred = guarded_test.argmax(axis=1)
        score = float(balanced_accuracy_score(y, pred))
        stats = meta_fold_stats(y, base_pred, pred, args.folds, args.seeds)
        row = {
            "name": name[:180],
            "mode": mode,
            "oof_balanced_accuracy": score,
            "delta_vs_base": score - base_score,
            "delta_vs_candidate": score - cand_score,
            "changed_rows_vs_base_oof": int((base_pred != pred).sum()),
            "changed_rows_vs_base_test": int((base_test_pred != test_pred).sum()),
            "kept_candidate_rows_oof": int(keep_train.sum()),
            "kept_candidate_rows_test": int(keep_test.sum()),
            "class_recalls": class_recalls(y, pred),
            "transition_counts": transition_counts(base_pred, pred),
            "changed_outcomes": changed_outcomes(y, base_pred, pred),
            **stats,
        }
        rows.append(row)
        candidates.append((row, guarded_oof, guarded_test))

    summary = pd.DataFrame(rows)
    summary["robust_rank_score"] = (
        summary["delta_vs_base"]
        + summary["meta_fold_min_delta"].clip(upper=0.0)
        + 0.00000002 * summary["changed_rows_vs_base_oof"].clip(upper=500)
    )
    summary = summary.sort_values(
        ["oof_balanced_accuracy", "meta_fold_min_delta", "changed_rows_vs_base_test"],
        ascending=[False, False, True],
    )
    summary.to_csv(args.output_dir / "candidate_summary.csv", index=False)
    save_bar(summary.head(24), args.output_dir / "subset_guarded_candidate_scores.png")
    save_bar(summary.head(24), args.output_dir / "subset_guarded_candidate_scores.svg")

    by_name = {row["name"]: (row, oof, test_proba) for row, oof, test_proba in candidates}
    selected = []
    for rank_offset, row in enumerate(summary.head(args.top_k).to_dict(orient="records")):
        rank = args.output_rank_start + rank_offset
        _, _, test_proba = by_name[row["name"]]
        short = row["name"].replace("__AND__", "_and_")
        short = short.replace("drop_", "guard_").replace("keep_", "keep_")
        short = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in short)[:90]
        score_tag = f"{row['oof_balanced_accuracy']:.6f}".replace(".", "")
        out_path = OUTPUTS / f"{rank:02d}_PRIVATE_CV_subset_guard_{short}_oof{score_tag}.csv"
        make_submission(out_path, sample, test_proba)
        selected.append({**row, "rank": rank, "path": str(out_path.relative_to(ROOT))})
        progress(f"selected rank={rank} OOF={row['oof_balanced_accuracy']:.9f} path={out_path.relative_to(ROOT)}")

    report = {
        "purpose": "Subset guard search for high-OOF classwise candidate. Public LB is not used.",
        "base_oof_balanced_accuracy": base_score,
        "candidate_name": args.candidate_name,
        "candidate_oof_balanced_accuracy": cand_score,
        "candidate_changed_rows_oof": int((base_pred != cand_pred).sum()),
        "candidate_changed_rows_test": int((base_test_pred != cand_test_pred).sum()),
        "selected_outputs": selected,
        "candidate_summary": str((args.output_dir / "candidate_summary.csv").relative_to(ROOT)),
        "score_plot": str((args.output_dir / "subset_guarded_candidate_scores.png").relative_to(ROOT)),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
