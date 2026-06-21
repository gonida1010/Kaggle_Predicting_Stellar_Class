from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
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


@dataclass(frozen=True)
class Candidate:
    name: str
    score: float
    delta: float
    selected_rows: int
    test_selected_rows: int
    feature_mask: str
    transition_rule: str
    challenger_conf_min: float
    challenger_margin_min: float
    base_conf_max: float
    class_recalls: dict[str, float]
    transition_counts: dict[str, int]
    outcome_counts: dict[str, int]
    meta_fold_mean_delta: float
    meta_fold_min_delta: float
    meta_fold_positive_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search OOF-safe patches where fold-safe target-encoding LightGBM disagrees with "
            "the current high-CV classwise greedy stack. No public labels or public LB feedback are used."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS / "te_disagreement_patch_classwise37",
    )
    parser.add_argument(
        "--base-oof",
        type=Path,
        default=ARTIFACTS / "oof_generalization_stack_with_classwise_blender_fast" / "generalization_stack_oof.npy",
    )
    parser.add_argument(
        "--base-test",
        type=Path,
        default=ARTIFACTS / "oof_generalization_stack_with_classwise_blender_fast" / "generalization_stack_test.npy",
    )
    parser.add_argument(
        "--challenger-oof",
        type=Path,
        default=ARTIFACTS / "lgbm_foldsafe_te_realmlp" / "lgbm_te_oof_proba.npy",
    )
    parser.add_argument(
        "--challenger-test",
        type=Path,
        default=ARTIFACTS / "lgbm_foldsafe_te_realmlp" / "lgbm_te_test_proba.npy",
    )
    parser.add_argument("--challenger-name", default="lgbm_foldsafe_te")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-rank-start", type=int, default=56)
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_probs(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return proba / row_sum


def load_proba(path: Path, expected_rows: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    arr = np.asarray(arr, dtype=np.float64)
    expected = (expected_rows, len(CLASSES))
    if arr.shape != expected:
        raise ValueError(f"{path} shape {arr.shape}, expected {expected}")
    return normalize_probs(arr)


def qbins(train_col: pd.Series, test_col: pd.Series, bins: int) -> tuple[np.ndarray, np.ndarray]:
    _, edges = pd.qcut(train_col, q=bins, retbins=True, duplicates="drop")
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros(len(train_col), dtype=int), np.zeros(len(test_col), dtype=int)
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = range(len(edges) - 1)
    train_bins = pd.cut(train_col, bins=edges, labels=labels, include_lowest=True).astype(int).to_numpy()
    test_bins = pd.cut(test_col, bins=edges, labels=labels, include_lowest=True).astype(int).to_numpy()
    return train_bins, test_bins


def build_feature_masks(train: pd.DataFrame, test: pd.DataFrame, bins: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    progress("Building diagnostic feature masks")
    train_fe = add_advanced_features(train).copy()
    test_fe = add_advanced_features(test).copy()
    for col in ["redshift", "g-i", "u-r", "mag_range"]:
        train_fe[f"{col}_bin"], test_fe[f"{col}_bin"] = qbins(train_fe[col], test_fe[col], bins)

    for frame in (train_fe, test_fe):
        frame["spectral_population"] = (
            frame["spectral_type"].astype(str) + "_" + frame["galaxy_population"].astype(str)
        )

    def masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        spectral = frame["spectral_type"].astype(str)
        population = frame["galaxy_population"].astype(str)
        combo = frame["spectral_population"].astype(str)
        redshift = frame["redshift_bin"]
        gi = frame["g-i_bin"]
        ur = frame["u-r_bin"]
        mag_range = frame["mag_range_bin"]
        out = {
            "all": np.ones(len(frame), dtype=bool),
            "rz_0_2": redshift.isin([0, 1, 2]).to_numpy(),
            "rz_0_3": redshift.isin([0, 1, 2, 3]).to_numpy(),
            "rz_0_4": redshift.isin([0, 1, 2, 3, 4]).to_numpy(),
            "rz_5_9": redshift.isin([5, 6, 7, 8, 9]).to_numpy(),
            "gi_0_2": gi.isin([0, 1, 2]).to_numpy(),
            "gi_7_9": gi.isin([7, 8, 9]).to_numpy(),
            "ur_0_2": ur.isin([0, 1, 2]).to_numpy(),
            "mag_range_0_2": mag_range.isin([0, 1, 2]).to_numpy(),
            "spectral_ob": spectral.eq("O/B").to_numpy(),
            "spectral_af": spectral.eq("A/F").to_numpy(),
            "spectral_m": spectral.eq("M").to_numpy(),
            "blue_cloud": population.eq("Blue_Cloud").to_numpy(),
            "red_sequence": population.eq("Red_Sequence").to_numpy(),
            "m_red_sequence": combo.eq("M_Red_Sequence").to_numpy(),
            "m_blue_cloud": combo.eq("M_Blue_Cloud").to_numpy(),
            "ob_blue_cloud": combo.eq("O/B_Blue_Cloud").to_numpy(),
            "af_blue_cloud": combo.eq("A/F_Blue_Cloud").to_numpy(),
            "weak_low_color": (
                gi.isin([0, 1, 2])
                | mag_range.isin([0, 1, 2])
                | combo.isin(["O/B_Blue_Cloud", "A/F_Blue_Cloud"])
            ).to_numpy(),
            "high_gi_low_rz": (gi.isin([7, 8, 9]) & redshift.isin([0, 1, 2, 3])).to_numpy(),
            "low_rz_redseq": (redshift.isin([0, 1, 2, 3]) & population.eq("Red_Sequence")).to_numpy(),
        }
        out["not_weak_low_color"] = ~out["weak_low_color"]
        return out

    return masks(train_fe), masks(test_fe)


def transition_masks(base_pred: np.ndarray, challenger_pred: np.ndarray) -> dict[str, np.ndarray]:
    changed = base_pred != challenger_pred
    return {
        "all_changed": changed,
        "base_star_to_galaxy": changed & (base_pred == 2) & (challenger_pred == 0),
        "base_qso_to_galaxy": changed & (base_pred == 1) & (challenger_pred == 0),
        "base_galaxy_to_star": changed & (base_pred == 0) & (challenger_pred == 2),
        "base_galaxy_to_qso": changed & (base_pred == 0) & (challenger_pred == 1),
        "base_star_to_qso": changed & (base_pred == 2) & (challenger_pred == 1),
        "base_qso_to_star": changed & (base_pred == 1) & (challenger_pred == 2),
        "base_non_galaxy_to_galaxy": changed & (base_pred != 0) & (challenger_pred == 0),
        "base_galaxy_to_non_galaxy": changed & (base_pred == 0) & (challenger_pred != 0),
    }


def class_recalls(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {label: float((pred[y == idx] == idx).mean()) for idx, label in enumerate(CLASSES)}


def transition_counts(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    changed = before != after
    counts = Counter(f"{CLASSES[int(old)]}->{CLASSES[int(new)]}" for old, new in zip(before[changed], after[changed]))
    return dict(sorted(counts.items()))


def outcome_counts(y: np.ndarray, before: np.ndarray, after: np.ndarray) -> dict[str, int]:
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
    arr = np.array(deltas, dtype=np.float64)
    return {
        "meta_fold_mean_delta": float(arr.mean()),
        "meta_fold_min_delta": float(arr.min()),
        "meta_fold_positive_rate": float((arr > 0).mean()),
    }


def selected_score(
    y: np.ndarray,
    base_pred: np.ndarray,
    challenger_pred: np.ndarray,
    selected: np.ndarray,
    class_counts: np.ndarray,
    base_correct_by_class: np.ndarray,
) -> tuple[float, dict[str, float]]:
    base_ok = base_pred == y
    challenger_ok = challenger_pred == y
    delta = np.bincount(
        y[selected],
        weights=challenger_ok[selected].astype(np.int16) - base_ok[selected].astype(np.int16),
        minlength=len(CLASSES),
    )
    correct = base_correct_by_class + delta
    recalls = correct / class_counts
    return float(recalls.mean()), {label: float(recalls[idx]) for idx, label in enumerate(CLASSES)}


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def oof_token(score: float) -> str:
    return f"oof{score:.6f}".replace(".", "")


def write_submission(path: Path, sample: pd.DataFrame, proba: np.ndarray) -> None:
    out = sample.copy()
    out["class"] = np.array(CLASSES)[proba.argmax(axis=1)]
    out.to_csv(path, index=False)


def save_summary_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    view = summary.sort_values("oof_balanced_accuracy", ascending=True).tail(18)
    labels = [
        f"{row.feature_mask}\\n{row.transition_rule}\\nrows={int(row.selected_rows)}"
        for row in view.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(13, max(7, 0.48 * len(view))))
    colors = np.where(view["meta_fold_min_delta"].to_numpy() >= 0, "#2563eb", "#dc2626")
    ax.barh(np.arange(len(view)), view["oof_delta_vs_base"].to_numpy(), color=colors, alpha=0.86)
    ax.axvline(0, color="#111827", linewidth=1.2)
    ax.set_yticks(np.arange(len(view)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("OOF balanced accuracy delta vs base")
    ax.set_title("TE disagreement patch candidates")
    ax.grid(axis="x", alpha=0.25)
    for idx, row in enumerate(view.itertuples(index=False)):
        ax.text(
            row.oof_delta_vs_base,
            idx,
            f" {row.oof_balanced_accuracy:.6f}",
            va="center",
            ha="left" if row.oof_delta_vs_base >= 0 else "right",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "te_disagreement_candidate_scores.svg", format="svg")
    fig.savefig(output_dir / "te_disagreement_candidate_scores.png", format="png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    progress("Loading train/test/sample")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["class"].map(TARGET_MAP).to_numpy(dtype=np.int64)

    progress("Loading base and challenger probabilities")
    base_oof = load_proba(args.base_oof, len(train))
    base_test = load_proba(args.base_test, len(test))
    challenger_oof = load_proba(args.challenger_oof, len(train))
    challenger_test = load_proba(args.challenger_test, len(test))

    base_pred = base_oof.argmax(axis=1)
    challenger_pred = challenger_oof.argmax(axis=1)
    base_test_pred = base_test.argmax(axis=1)
    challenger_test_pred = challenger_test.argmax(axis=1)
    base_score = float(balanced_accuracy_score(y, base_pred))
    challenger_score = float(balanced_accuracy_score(y, challenger_pred))

    progress(
        f"Base OOF={base_score:.6f}, challenger OOF={challenger_score:.6f}, "
        f"OOF disagreements={(base_pred != challenger_pred).sum()}, "
        f"test disagreements={(base_test_pred != challenger_test_pred).sum()}"
    )

    train_masks, test_masks = build_feature_masks(train, test, args.bins)
    train_transitions = transition_masks(base_pred, challenger_pred)
    test_transitions = transition_masks(base_test_pred, challenger_test_pred)

    challenger_conf = challenger_oof.max(axis=1)
    challenger_test_conf = challenger_test.max(axis=1)
    challenger_margin = np.sort(challenger_oof, axis=1)[:, -1] - np.sort(challenger_oof, axis=1)[:, -2]
    challenger_test_margin = np.sort(challenger_test, axis=1)[:, -1] - np.sort(challenger_test, axis=1)[:, -2]
    base_conf = base_oof.max(axis=1)
    base_test_conf = base_test.max(axis=1)

    class_counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    base_correct_by_class = np.bincount(y[base_pred == y], minlength=len(CLASSES)).astype(np.float64)

    conf_grid = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    margin_grid = [0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    base_conf_grid = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50]

    progress("Searching disagreement gates")
    records: list[dict] = []
    for feature_name, feature_mask in train_masks.items():
        test_feature_mask = test_masks[feature_name]
        for transition_name, transition_mask in train_transitions.items():
            raw = feature_mask & transition_mask
            if int(raw.sum()) == 0:
                continue
            test_raw = test_feature_mask & test_transitions[transition_name]
            for conf_min in conf_grid:
                conf_mask = challenger_conf >= conf_min
                test_conf_mask = challenger_test_conf >= conf_min
                for margin_min in margin_grid:
                    margin_mask = challenger_margin >= margin_min
                    test_margin_mask = challenger_test_margin >= margin_min
                    for base_conf_max in base_conf_grid:
                        selected = raw & conf_mask & margin_mask & (base_conf <= base_conf_max)
                        selected_rows = int(selected.sum())
                        if selected_rows == 0:
                            continue
                        test_selected = test_raw & test_conf_mask & test_margin_mask & (base_test_conf <= base_conf_max)
                        score, recalls = selected_score(
                            y,
                            base_pred,
                            challenger_pred,
                            selected,
                            class_counts,
                            base_correct_by_class,
                        )
                        delta = score - base_score
                        if delta <= 0:
                            continue
                        records.append(
                            {
                                "name": safe_name(
                                    f"{feature_name}_{transition_name}_c{conf_min:.2f}_m{margin_min:.2f}_b{base_conf_max:.2f}"
                                ),
                                "oof_balanced_accuracy": score,
                                "oof_delta_vs_base": delta,
                                "selected_rows": selected_rows,
                                "test_selected_rows": int(test_selected.sum()),
                                "feature_mask": feature_name,
                                "transition_rule": transition_name,
                                "challenger_conf_min": conf_min,
                                "challenger_margin_min": margin_min,
                                "base_conf_max": base_conf_max,
                                "class_recalls": recalls,
                                "transition_counts": {},
                                "outcome_counts": {},
                                "meta_fold_mean_delta": np.nan,
                                "meta_fold_min_delta": np.nan,
                                "meta_fold_positive_rate": np.nan,
                            }
                        )
    if not records:
        raise RuntimeError("No OOF-improving TE disagreement gates found.")

    summary = pd.DataFrame(records).sort_values("oof_balanced_accuracy", ascending=False)
    progress("Computing meta-fold stability for top OOF gates")
    meta_limit = min(len(summary), max(args.top_k * 6, 50))
    for idx in summary.head(meta_limit).index:
        row = summary.loc[idx]
        train_selected = (
            train_masks[row["feature_mask"]]
            & train_transitions[row["transition_rule"]]
            & (challenger_conf >= row["challenger_conf_min"])
            & (challenger_margin >= row["challenger_margin_min"])
            & (base_conf <= row["base_conf_max"])
        )
        after_pred = base_pred.copy()
        after_pred[train_selected] = challenger_pred[train_selected]
        stats = meta_fold_stats(y, base_pred, after_pred, args.folds, args.seeds)
        summary.at[idx, "meta_fold_mean_delta"] = stats["meta_fold_mean_delta"]
        summary.at[idx, "meta_fold_min_delta"] = stats["meta_fold_min_delta"]
        summary.at[idx, "meta_fold_positive_rate"] = stats["meta_fold_positive_rate"]
        summary.at[idx, "transition_counts"] = transition_counts(base_pred, after_pred)
        summary.at[idx, "outcome_counts"] = outcome_counts(y, base_pred, after_pred)

    summary = summary.sort_values(
        ["oof_balanced_accuracy", "meta_fold_min_delta", "test_selected_rows"],
        ascending=[False, False, True],
    )
    summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    save_summary_plot(summary, output_dir)
    progress(f"Found {len(summary)} OOF-improving gates")

    selected_outputs = []
    for offset, row in enumerate(summary.head(args.top_k).itertuples(index=False), start=0):
        train_selected = (
            train_masks[row.feature_mask]
            & train_transitions[row.transition_rule]
            & (challenger_conf >= row.challenger_conf_min)
            & (challenger_margin >= row.challenger_margin_min)
            & (base_conf <= row.base_conf_max)
        )
        test_selected = (
            test_masks[row.feature_mask]
            & test_transitions[row.transition_rule]
            & (challenger_test_conf >= row.challenger_conf_min)
            & (challenger_test_margin >= row.challenger_margin_min)
            & (base_test_conf <= row.base_conf_max)
        )
        oof = base_oof.copy()
        test_proba = base_test.copy()
        oof[train_selected] = challenger_oof[train_selected]
        test_proba[test_selected] = challenger_test[test_selected]
        score = float(row.oof_balanced_accuracy)
        rank = args.output_rank_start + offset
        file_name = f"{rank:02d}_PRIVATE_CV_te_disagree_{row.name}_{oof_token(score)}.csv"
        output_path = OUTPUTS / file_name
        write_submission(output_path, sample, test_proba)
        np.save(output_dir / f"{rank:02d}_{row.name}_oof.npy", oof.astype(np.float32))
        np.save(output_dir / f"{rank:02d}_{row.name}_test.npy", test_proba.astype(np.float32))
        selected_outputs.append(
            {
                "rank": rank,
                "path": str(output_path.relative_to(ROOT)),
                "name": row.name,
                "oof_balanced_accuracy": score,
                "oof_delta_vs_base": float(row.oof_delta_vs_base),
                "selected_rows": int(row.selected_rows),
                "test_selected_rows": int(row.test_selected_rows),
                "meta_fold_min_delta": float(row.meta_fold_min_delta),
                "meta_fold_positive_rate": float(row.meta_fold_positive_rate),
            }
        )
        progress(
            f"wrote {output_path.relative_to(ROOT)} OOF={score:.6f} "
            f"train_rows={int(row.selected_rows)} test_rows={int(row.test_selected_rows)}"
        )

    report = {
        "purpose": "Patch only OOF-safe disagreement rows where fold-safe TE LightGBM challenges the classwise greedy base.",
        "base_oof": str(args.base_oof.relative_to(ROOT)),
        "base_test": str(args.base_test.relative_to(ROOT)),
        "challenger_oof": str(args.challenger_oof.relative_to(ROOT)),
        "challenger_test": str(args.challenger_test.relative_to(ROOT)),
        "challenger_name": args.challenger_name,
        "base_oof_balanced_accuracy": base_score,
        "challenger_oof_balanced_accuracy": challenger_score,
        "oof_disagreements": int((base_pred != challenger_pred).sum()),
        "test_disagreements": int((base_test_pred != challenger_test_pred).sum()),
        "candidate_count": int(len(summary)),
        "best": json.loads(summary.head(1).to_json(orient="records"))[0],
        "selected_outputs": selected_outputs,
        "candidate_summary": str((output_dir / "candidate_summary.csv").relative_to(ROOT)),
        "score_plot": str((output_dir / "te_disagreement_candidate_scores.png").relative_to(ROOT)),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
