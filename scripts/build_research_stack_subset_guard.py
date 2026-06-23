from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.build_available_prediction_stacker import CLASSES, DATA, TARGET_MAP  # noqa: E402
from src.stellar_features import add_advanced_features  # noqa: E402


ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"
DEFAULT_BASE56_OOF = (
    ARTIFACTS
    / "te_disagreement_patch_classwise37"
    / "56_high_gi_low_rz_base_galaxy_to_star_c0_55_m0_15_b0_60_oof.npy"
)
DEFAULT_BASE56_TEST = (
    ARTIFACTS
    / "te_disagreement_patch_classwise37"
    / "56_high_gi_low_rz_base_galaxy_to_star_c0_55_m0_15_b0_60_test.npy"
)
DEFAULT_STACK68_OOF = ARTIFACTS / "research_material_stack_20260623" / "research_material_stack_oof.npy"
DEFAULT_STACK68_TEST = ARTIFACTS / "research_material_stack_20260623" / "research_material_stack_test.npy"
DEFAULT_STABLE69_OOF = (
    ARTIFACTS
    / "private_cv_stable_research_material_stack_20260623"
    / "private_cv_guarded_01_all_changed_rz_0_2_allconf_oof.npy"
)
DEFAULT_STABLE69_TEST = (
    ARTIFACTS
    / "private_cv_stable_research_material_stack_20260623"
    / "private_cv_guarded_01_all_changed_rz_0_2_allconf_test.npy"
)


@dataclass
class SourceSet:
    name: str
    oof: np.ndarray
    test: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build private-CV subset guards from the 56/68/69 OOF probability sources. "
            "No public leaderboard labels or public-only CSV banks are used."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "research_stack_subset_guard_20260623")
    parser.add_argument("--base56-oof", type=Path, default=DEFAULT_BASE56_OOF)
    parser.add_argument("--base56-test", type=Path, default=DEFAULT_BASE56_TEST)
    parser.add_argument("--stack68-oof", type=Path, default=DEFAULT_STACK68_OOF)
    parser.add_argument("--stack68-test", type=Path, default=DEFAULT_STACK68_TEST)
    parser.add_argument("--stable69-oof", type=Path, default=DEFAULT_STABLE69_OOF)
    parser.add_argument("--stable69-test", type=Path, default=DEFAULT_STABLE69_TEST)
    parser.add_argument("--candidate-tag", default="68")
    parser.add_argument("--stable-tag", default="69")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output-rank-start", type=int, default=76)
    parser.add_argument("--top-k", type=int, default=8)
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


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float | int]]:
    rows: dict[str, dict[str, float | int]] = {}
    for idx, label in enumerate(CLASSES):
        tp = int(((y_true == idx) & (y_pred == idx)).sum())
        fp = int(((y_true != idx) & (y_pred == idx)).sum())
        fn = int(((y_true == idx) & (y_pred != idx)).sum())
        support = int((y_true == idx).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
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


def add_research_bins(train: pd.DataFrame, test: pd.DataFrame, bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_fe = add_advanced_features(train).copy()
    test_fe = add_advanced_features(test).copy()
    for col in ["redshift", "g-i", "u-r", "mag_range"]:
        train_fe[f"{col}_bin"], test_fe[f"{col}_bin"] = quantile_bins(train_fe[col], test_fe[col], bins)
    for frame in (train_fe, test_fe):
        frame["spectral_population"] = (
            frame["spectral_type"].astype(str) + "_" + frame["galaxy_population"].astype(str)
        )
    return train_fe, test_fe


def source_pred(source: SourceSet) -> tuple[np.ndarray, np.ndarray]:
    return source.oof.argmax(axis=1), source.test.argmax(axis=1)


def replace_where(base: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = base.copy()
    out[mask] = candidate[mask]
    return normalize_probs(out)


def fold_stability(y: np.ndarray, reference_pred: np.ndarray, pred: np.ndarray, folds: int, seeds: int) -> dict[str, float]:
    deltas = []
    scores = []
    for seed in range(20260623, 20260623 + seeds):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for _, valid_idx in splitter.split(np.zeros(len(y)), y):
            ref_score = balanced_accuracy(y[valid_idx], reference_pred[valid_idx])
            score = balanced_accuracy(y[valid_idx], pred[valid_idx])
            scores.append(score)
            deltas.append(score - ref_score)
    deltas_arr = np.array(deltas, dtype=np.float64)
    return {
        "meta_fold_mean_score": float(np.mean(scores)),
        "meta_fold_mean_delta_vs_56": float(np.mean(deltas_arr)),
        "meta_fold_min_delta_vs_56": float(np.min(deltas_arr)),
        "meta_fold_max_delta_vs_56": float(np.max(deltas_arr)),
        "meta_fold_positive_rate_vs_56": float((deltas_arr > 0).mean()),
    }


def subset_delta(
    name: str,
    mask: np.ndarray,
    y: np.ndarray,
    reference_pred: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float | int | str]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return {
            "subset": name,
            "count": 0,
            "ref_bac": 0.0,
            "candidate_bac": 0.0,
            "delta_bac_vs_56": 0.0,
            "changed_rows_vs_56": 0,
        }
    ref_score = balanced_accuracy(y[idx], reference_pred[idx])
    score = balanced_accuracy(y[idx], pred[idx])
    return {
        "subset": name,
        "count": int(len(idx)),
        "ref_bac": float(ref_score),
        "candidate_bac": float(score),
        "delta_bac_vs_56": float(score - ref_score),
        "changed_rows_vs_56": int((reference_pred[idx] != pred[idx]).sum()),
    }


def make_submission(path: Path, sample: pd.DataFrame, test_proba: np.ndarray) -> None:
    submission = sample.copy()
    submission["class"] = np.array(CLASSES)[test_proba.argmax(axis=1)]
    submission.to_csv(path, index=False)


def make_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    spectral = frame["spectral_type"].astype(str)
    spectral_pop = frame["spectral_population"].astype(str)
    gi = frame["g-i_bin"]
    ur = frame["u-r_bin"]
    mr = frame["mag_range_bin"]
    rz = frame["redshift_bin"]

    bad_color = gi.isin([0, 2, 4, 5]) | mr.isin([0, 2])
    bad_spectral = spectral.isin(["O/B", "A/F"]) | spectral_pop.isin(["O/B_Blue_Cloud", "A/F_Blue_Cloud"])
    bad_union = bad_color | bad_spectral
    bad_low_color = gi.le(5) & mr.le(2)

    good_high_gi = gi.isin([7, 8, 9])
    good_ur0 = ur.eq(0)
    good_m = spectral.eq("M") | spectral_pop.eq("M_Blue_Cloud")
    good_high_mag = mr.isin([7, 8, 9])
    good_rz2_high_gi = rz.eq(2) & gi.ge(6)
    good_union = good_high_gi | good_ur0 | good_m | good_high_mag | good_rz2_high_gi

    return {
        "bad_color": bad_color.to_numpy(),
        "bad_spectral": bad_spectral.to_numpy(),
        "bad_union": bad_union.to_numpy(),
        "bad_low_color": bad_low_color.to_numpy(),
        "good_high_gi": good_high_gi.to_numpy(),
        "good_ur0": good_ur0.to_numpy(),
        "good_m": good_m.to_numpy(),
        "good_high_mag": good_high_mag.to_numpy(),
        "good_rz2_high_gi": good_rz2_high_gi.to_numpy(),
        "good_union": good_union.to_numpy(),
        "safe_good": (good_union & ~bad_union).to_numpy(),
        "rz_0_2": rz.isin([0, 1, 2]).to_numpy(),
        "rz_1_2": rz.isin([1, 2]).to_numpy(),
        "not_ob_af": (~spectral.isin(["O/B", "A/F"])).to_numpy(),
    }


def build_candidates(
    base56: SourceSet,
    stack68: SourceSet,
    stable69: SourceSet,
    train_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    candidate_tag: str,
    stable_tag: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    base56_pred, base56_test_pred = source_pred(base56)
    stack68_pred, stack68_test_pred = source_pred(stack68)
    stable69_pred, stable69_test_pred = source_pred(stable69)
    changed_56_68 = base56_pred != stack68_pred
    changed_56_68_test = base56_test_pred != stack68_test_pred
    changed_69_68 = stable69_pred != stack68_pred
    changed_69_68_test = stable69_test_pred != stack68_test_pred

    specs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
        f"{candidate_tag}_rollback_bad_union": (
            stack68.oof,
            stack68.test,
            base56.oof,
            base56.test,
            changed_56_68 & train_masks["bad_union"],
            changed_56_68_test & test_masks["bad_union"],
        ),
        f"{candidate_tag}_rollback_bad_color": (
            stack68.oof,
            stack68.test,
            base56.oof,
            base56.test,
            changed_56_68 & train_masks["bad_color"],
            changed_56_68_test & test_masks["bad_color"],
        ),
        f"{candidate_tag}_rollback_bad_spectral": (
            stack68.oof,
            stack68.test,
            base56.oof,
            base56.test,
            changed_56_68 & train_masks["bad_spectral"],
            changed_56_68_test & test_masks["bad_spectral"],
        ),
        f"{candidate_tag}_rollback_low_color": (
            stack68.oof,
            stack68.test,
            base56.oof,
            base56.test,
            changed_56_68 & train_masks["bad_low_color"],
            changed_56_68_test & test_masks["bad_low_color"],
        ),
        f"56_plus_{candidate_tag}_safe_good": (
            base56.oof,
            base56.test,
            stack68.oof,
            stack68.test,
            changed_56_68 & train_masks["safe_good"],
            changed_56_68_test & test_masks["safe_good"],
        ),
        f"56_plus_{candidate_tag}_good_union": (
            base56.oof,
            base56.test,
            stack68.oof,
            stack68.test,
            changed_56_68 & train_masks["good_union"],
            changed_56_68_test & test_masks["good_union"],
        ),
        f"56_plus_{candidate_tag}_high_gi": (
            base56.oof,
            base56.test,
            stack68.oof,
            stack68.test,
            changed_56_68 & train_masks["good_high_gi"],
            changed_56_68_test & test_masks["good_high_gi"],
        ),
        f"56_plus_{candidate_tag}_m_or_highmag": (
            base56.oof,
            base56.test,
            stack68.oof,
            stack68.test,
            changed_56_68 & (train_masks["good_m"] | train_masks["good_high_mag"]) & ~train_masks["bad_spectral"],
            changed_56_68_test & (test_masks["good_m"] | test_masks["good_high_mag"]) & ~test_masks["bad_spectral"],
        ),
        f"{stable_tag}_plus_{candidate_tag}_safe_good": (
            stable69.oof,
            stable69.test,
            stack68.oof,
            stack68.test,
            changed_69_68 & train_masks["safe_good"],
            changed_69_68_test & test_masks["safe_good"],
        ),
        f"{stable_tag}_plus_{candidate_tag}_good_union": (
            stable69.oof,
            stable69.test,
            stack68.oof,
            stack68.test,
            changed_69_68 & train_masks["good_union"],
            changed_69_68_test & test_masks["good_union"],
        ),
        f"{stable_tag}_rollback_bad_union_to56": (
            stable69.oof,
            stable69.test,
            base56.oof,
            base56.test,
            (base56_pred != stable69_pred) & train_masks["bad_union"],
            (base56_test_pred != stable69_test_pred) & test_masks["bad_union"],
        ),
        f"{stable_tag}_rollback_bad_color_to56": (
            stable69.oof,
            stable69.test,
            base56.oof,
            base56.test,
            (base56_pred != stable69_pred) & train_masks["bad_color"],
            (base56_test_pred != stable69_test_pred) & test_masks["bad_color"],
        ),
    }

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (base_oof, base_test, fill_oof, fill_test, train_mask, test_mask) in specs.items():
        candidate_oof = replace_where(base_oof, fill_oof, train_mask)
        candidate_test = replace_where(base_test, fill_test, test_mask)
        out[name] = (candidate_oof, candidate_test)
    return out


def evaluate_candidate(
    name: str,
    y: np.ndarray,
    base56: SourceSet,
    stack68: SourceSet,
    stable69: SourceSet,
    oof: np.ndarray,
    test: np.ndarray,
    train_masks: dict[str, np.ndarray],
    folds: int,
    seeds: int,
) -> dict:
    pred = oof.argmax(axis=1)
    test_pred = test.argmax(axis=1)
    base56_pred, base56_test_pred = source_pred(base56)
    stack68_pred, _ = source_pred(stack68)
    stable69_pred, stable69_test_pred = source_pred(stable69)

    score = balanced_accuracy(y, pred)
    base56_score = balanced_accuracy(y, base56_pred)
    stack68_score = balanced_accuracy(y, stack68_pred)
    stable69_score = balanced_accuracy(y, stable69_pred)
    candidate_metrics = class_metrics(y, pred)
    base56_metrics = class_metrics(y, base56_pred)
    class_delta = {
        label: float(candidate_metrics[label]["recall"] - base56_metrics[label]["recall"])
        for label in CLASSES
    }
    critical_masks = {
        "bad_union": train_masks["bad_union"],
        "bad_color": train_masks["bad_color"],
        "bad_spectral": train_masks["bad_spectral"],
        "safe_good": train_masks["safe_good"],
        "good_union": train_masks["good_union"],
        "good_high_gi": train_masks["good_high_gi"],
        "good_m": train_masks["good_m"],
        "rz_0_2": train_masks["rz_0_2"],
    }
    subset_rows = [
        subset_delta(subset_name, mask, y, base56_pred, pred)
        for subset_name, mask in critical_masks.items()
    ]
    worst_subset = min(subset_rows, key=lambda row: float(row["delta_bac_vs_56"]))
    best_subset = max(subset_rows, key=lambda row: float(row["delta_bac_vs_56"]))
    stability = fold_stability(y, base56_pred, pred, folds, seeds)
    robust_rank_score = (
        (score - base56_score)
        + 0.35 * stability["meta_fold_min_delta_vs_56"]
        + 0.15 * min(0.0, min(class_delta.values()))
        + 0.15 * min(0.0, float(worst_subset["delta_bac_vs_56"]))
    )
    return {
        "name": name,
        "oof_balanced_accuracy": float(score),
        "delta_vs_56": float(score - base56_score),
        "delta_vs_68": float(score - stack68_score),
        "delta_vs_69": float(score - stable69_score),
        "changed_rows_vs_56": int((pred != base56_pred).sum()),
        "changed_rows_vs_68": int((pred != stack68_pred).sum()),
        "changed_rows_vs_69": int((pred != stable69_pred).sum()),
        "test_changed_rows_vs_56": int((test_pred != base56_test_pred).sum()),
        "test_changed_rows_vs_69": int((test_pred != stable69_test_pred).sum()),
        "class_metrics": candidate_metrics,
        "class_recall_delta_vs_56": class_delta,
        "worst_class_recall_delta_vs_56": float(min(class_delta.values())),
        "transition_counts_vs_56": transition_counts(base56_pred, pred),
        "changed_outcomes_vs_56": changed_outcomes(y, base56_pred, pred),
        "worst_subset": worst_subset["subset"],
        "worst_subset_delta_vs_56": float(worst_subset["delta_bac_vs_56"]),
        "best_subset": best_subset["subset"],
        "best_subset_delta_vs_56": float(best_subset["delta_bac_vs_56"]),
        "robust_rank_score": float(robust_rank_score),
        **stability,
    }


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

    progress("Loading OOF/test probability sources")
    base56 = SourceSet(
        name="56_te_disagreement",
        oof=load_proba(args.base56_oof, len(train)),
        test=load_proba(args.base56_test, len(sample)),
    )
    stack68 = SourceSet(
        name="68_research_material_stack",
        oof=load_proba(args.stack68_oof, len(train)),
        test=load_proba(args.stack68_test, len(sample)),
    )
    stable69 = SourceSet(
        name="69_guarded_research_material_stack",
        oof=load_proba(args.stable69_oof, len(train)),
        test=load_proba(args.stable69_test, len(sample)),
    )

    progress("Building feature bins and subset masks")
    train_fe, test_fe = add_research_bins(train, test, args.bins)
    train_masks = make_masks(train_fe)
    test_masks = make_masks(test_fe)

    base56_score = balanced_accuracy(y, base56.oof.argmax(axis=1))
    stack68_score = balanced_accuracy(y, stack68.oof.argmax(axis=1))
    stable69_score = balanced_accuracy(y, stable69.oof.argmax(axis=1))
    progress(
        "source OOF BAC: "
        f"56={base56_score:.9f}, 68={stack68_score:.9f}, 69={stable69_score:.9f}"
    )

    progress("Building subset-guard candidates")
    candidate_probs = build_candidates(
        base56,
        stack68,
        stable69,
        train_masks,
        test_masks,
        args.candidate_tag,
        args.stable_tag,
    )
    reports = []
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (oof, test_proba) in candidate_probs.items():
        report = evaluate_candidate(
            name=name,
            y=y,
            base56=base56,
            stack68=stack68,
            stable69=stable69,
            oof=oof,
            test=test_proba,
            train_masks=train_masks,
            folds=args.folds,
            seeds=args.seeds,
        )
        reports.append(report)
        arrays[name] = (oof, test_proba)
        progress(
            f"{name}: OOF={report['oof_balanced_accuracy']:.9f}, "
            f"delta56={report['delta_vs_56']:+.9f}, "
            f"test_changes56={report['test_changed_rows_vs_56']}"
        )

    summary_rows = []
    for report in reports:
        row = {
            key: value
            for key, value in report.items()
            if not isinstance(value, (dict, list))
        }
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["robust_rank_score", "oof_balanced_accuracy", "meta_fold_positive_rate_vs_56"],
        ascending=[False, False, False],
    )
    summary_df.to_csv(output_dir / "candidate_summary.csv", index=False)

    detailed_path = output_dir / "report.json"
    detailed_path.write_text(
        json.dumps(
            {
                "purpose": "Subset guard candidates derived from 56/68/69 OOF probability sources.",
                "source_scores": {
                    "56_te_disagreement": base56_score,
                    "68_research_material_stack": stack68_score,
                    "69_guarded_research_material_stack": stable69_score,
                },
                "candidates": reports,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_rows = []
    for offset, row in enumerate(summary_df.head(args.top_k).to_dict(orient="records")):
        name = row["name"]
        oof, test_proba = arrays[name]
        score_tag = f"{row['oof_balanced_accuracy']:.6f}".replace(".", "")
        output_name = f"{args.output_rank_start + offset:02d}_PRIVATE_CV_subset_guard_{name}_oof{score_tag}"
        np.save(output_dir / f"{output_name}_oof.npy", oof.astype(np.float32))
        np.save(output_dir / f"{output_name}_test.npy", test_proba.astype(np.float32))
        artifact_csv = output_dir / f"{output_name}.csv"
        output_csv = OUTPUTS / f"{output_name}.csv"
        make_submission(artifact_csv, sample, test_proba)
        make_submission(output_csv, sample, test_proba)
        output_rows.append({**row, "artifact_csv": str(artifact_csv.relative_to(ROOT)), "output_csv": str(output_csv.relative_to(ROOT))})
        progress(f"wrote {output_csv.relative_to(ROOT)}")

    pd.DataFrame(output_rows).to_csv(output_dir / "output_candidates.csv", index=False)
    print(json.dumps({"summary": output_rows}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
