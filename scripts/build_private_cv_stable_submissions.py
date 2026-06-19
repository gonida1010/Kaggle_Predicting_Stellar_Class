from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from build_available_prediction_stacker import CLASSES, DATA, ROOT, TARGET_MAP

sys.path.append(str(ROOT))
from src.stellar_features import add_advanced_features


ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"
OUT_DIR = ARTIFACTS / "private_cv_stable_submissions"


@dataclass
class CandidateResult:
    name: str
    oof_proba: np.ndarray
    test_proba: np.ndarray
    report: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build private-focused OOF/CV-stable submissions. Public LB scores and public submission banks are not used."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--base-oof", type=Path, default=ROOT / "external_sources/oof_test_predictions/oof_lr_stacker_v9.npy")
    parser.add_argument("--base-test", type=Path, default=ROOT / "external_preds/pred_lr_stacker_v9.npy")
    parser.add_argument("--candidate-oof", type=Path, default=ARTIFACTS / "oof_generalization_stack/generalization_stack_oof.npy")
    parser.add_argument("--candidate-test", type=Path, default=ARTIFACTS / "oof_generalization_stack/generalization_stack_test.npy")
    parser.add_argument("--stack-report", type=Path, default=ARTIFACTS / "oof_generalization_stack/report.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output-rank-start", type=int, default=19)
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_probs(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return proba / row_sum


def load_proba(path: Path, expected_rows: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        arr = df.iloc[:, -3:].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Unsupported probability file: {path}")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != (expected_rows, len(CLASSES)):
        raise ValueError(f"{path} shape {arr.shape}, expected {(expected_rows, len(CLASSES))}")
    return normalize_probs(arr)


def apply_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return normalize_probs(proba * bias.reshape(1, -1))


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


def confidence_margin(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(proba, axis=1)
    return ordered[:, -1], ordered[:, -1] - ordered[:, -2]


def quantile_bins(train_col: pd.Series, test_col: pd.Series, bins: int) -> tuple[pd.Series, pd.Series]:
    _, edges = pd.qcut(train_col, q=bins, retbins=True, duplicates="drop")
    edges = np.unique(edges)
    if len(edges) <= 2:
        train_bins = pd.Series(np.zeros(len(train_col), dtype=int), index=train_col.index)
        test_bins = pd.Series(np.zeros(len(test_col), dtype=int), index=test_col.index)
        return train_bins, test_bins
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = range(len(edges) - 1)
    train_bins = pd.cut(train_col, bins=edges, labels=labels, include_lowest=True).astype(int)
    test_bins = pd.cut(test_col, bins=edges, labels=labels, include_lowest=True).astype(int)
    return train_bins, test_bins


def add_research_bins(train: pd.DataFrame, test: pd.DataFrame, bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_fe = add_advanced_features(train).copy()
    test_fe = add_advanced_features(test).copy()
    for col in ["redshift", "g-i", "u-r", "mag_range"]:
        train_fe[f"{col}_bin"], test_fe[f"{col}_bin"] = quantile_bins(train_fe[col], test_fe[col], bins)
    train_fe["spectral_population"] = train_fe["spectral_type"].astype(str) + "_" + train_fe["galaxy_population"].astype(str)
    test_fe["spectral_population"] = test_fe["spectral_type"].astype(str) + "_" + test_fe["galaxy_population"].astype(str)
    return train_fe, test_fe


def meta_fold_deltas(
    y: np.ndarray,
    base_pred: np.ndarray,
    pred: np.ndarray,
    folds: int,
    seeds: int,
) -> dict[str, float]:
    deltas = []
    scores = []
    base_scores = []
    for seed in range(20260618, 20260618 + seeds):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for _, valid_idx in splitter.split(np.zeros(len(y)), y):
            base_score = balanced_accuracy(y[valid_idx], base_pred[valid_idx])
            score = balanced_accuracy(y[valid_idx], pred[valid_idx])
            base_scores.append(base_score)
            scores.append(score)
            deltas.append(score - base_score)
    return {
        "meta_fold_mean_score": float(np.mean(scores)),
        "meta_fold_mean_base_score": float(np.mean(base_scores)),
        "meta_fold_mean_delta": float(np.mean(deltas)),
        "meta_fold_min_delta": float(np.min(deltas)),
        "meta_fold_max_delta": float(np.max(deltas)),
        "meta_fold_positive_rate": float((np.array(deltas) > 0).mean()),
    }


def critical_subset_deltas(frame: pd.DataFrame, y: np.ndarray, base_pred: np.ndarray, pred: np.ndarray) -> dict[str, float | str]:
    subset_masks = {
        "u_r_bin_0": frame["u-r_bin"].eq(0),
        "u_r_bin_0_1": frame["u-r_bin"].isin([0, 1]),
        "mag_range_bin_0": frame["mag_range_bin"].eq(0),
        "mag_range_bin_0_1": frame["mag_range_bin"].isin([0, 1]),
        "g_i_bin_0": frame["g-i_bin"].eq(0),
        "g_i_bin_0_2": frame["g-i_bin"].isin([0, 1, 2]),
        "spectral_type_O_B": frame["spectral_type"].astype(str).eq("O/B"),
        "O_B_Blue_Cloud": frame["spectral_population"].eq("O/B_Blue_Cloud"),
        "redshift_bin_0_2": frame["redshift_bin"].isin([0, 1, 2]),
        "G_K_Red_Sequence": frame["spectral_population"].eq("G/K_Red_Sequence"),
    }
    rows = []
    for name, mask in subset_masks.items():
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) < 300:
            continue
        base_score = balanced_accuracy(y[idx], base_pred[idx])
        score = balanced_accuracy(y[idx], pred[idx])
        rows.append((name, len(idx), score - base_score))
    if not rows:
        return {
            "worst_subset": "",
            "worst_subset_delta": 0.0,
            "best_subset": "",
            "best_subset_delta": 0.0,
        }
    worst = min(rows, key=lambda row: row[2])
    best = max(rows, key=lambda row: row[2])
    return {
        "worst_subset": worst[0],
        "worst_subset_count": int(worst[1]),
        "worst_subset_delta": float(worst[2]),
        "best_subset": best[0],
        "best_subset_count": int(best[1]),
        "best_subset_delta": float(best[2]),
    }


def make_submission(path: Path, sample: pd.DataFrame, proba: np.ndarray) -> None:
    submission = sample.copy()
    submission["class"] = np.array(CLASSES)[proba.argmax(axis=1)]
    submission.to_csv(path, index=False)


def score_candidate(
    name: str,
    y: np.ndarray,
    sample: pd.DataFrame,
    train_fe: pd.DataFrame,
    base_raw_pred: np.ndarray,
    base_ref_pred: np.ndarray,
    oof_proba: np.ndarray,
    test_proba: np.ndarray,
    output_dir: Path,
    folds: int,
    seeds: int,
) -> CandidateResult:
    pred = oof_proba.argmax(axis=1)
    test_pred = test_proba.argmax(axis=1)
    base_raw_score = balanced_accuracy(y, base_raw_pred)
    base_ref_score = balanced_accuracy(y, base_ref_pred)
    score = balanced_accuracy(y, pred)
    recalls = class_recalls(y, pred)
    base_recalls = class_recalls(y, base_ref_pred)
    class_delta = {label: recalls[label] - base_recalls[label] for label in CLASSES}
    report = {
        "name": name,
        "oof_balanced_accuracy": float(score),
        "delta_vs_raw_lr_v9": float(score - base_raw_score),
        "delta_vs_reference": float(score - base_ref_score),
        "changed_rows_vs_reference": int((pred != base_ref_pred).sum()),
        "test_changed_rows_vs_reference": int((test_pred != base_ref_pred[: len(test_pred)]).sum()) if len(test_pred) == len(base_ref_pred) else None,
        "class_recalls": recalls,
        "class_recall_delta_vs_reference": class_delta,
        "worst_class_recall_delta": float(min(class_delta.values())),
        "transition_counts_vs_reference": transition_counts(base_ref_pred, pred),
        "changed_outcomes_vs_reference": changed_outcomes(y, base_ref_pred, pred),
        **meta_fold_deltas(y, base_ref_pred, pred, folds, seeds),
        **critical_subset_deltas(train_fe, y, base_ref_pred, pred),
    }
    report["robust_rank_score"] = float(
        report["delta_vs_raw_lr_v9"]
        + 0.35 * report["meta_fold_min_delta"]
        + 0.20 * min(0.0, report["worst_subset_delta"])
        + 0.20 * min(0.0, report["worst_class_recall_delta"])
    )
    np.save(output_dir / f"{name}_oof.npy", oof_proba.astype(np.float32))
    np.save(output_dir / f"{name}_test.npy", test_proba.astype(np.float32))
    make_submission(output_dir / f"{name}.csv", sample, test_proba)
    return CandidateResult(name=name, oof_proba=oof_proba, test_proba=test_proba, report=report)


def build_gate_catalog(
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    cand_oof: np.ndarray,
    cand_test: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    base_pred = base_oof.argmax(axis=1)
    cand_pred = cand_oof.argmax(axis=1)
    base_test_pred = base_test.argmax(axis=1)
    cand_test_pred = cand_test.argmax(axis=1)
    _, base_margin = confidence_margin(base_oof)
    _, cand_margin = confidence_margin(cand_oof)
    _, base_test_margin = confidence_margin(base_test)
    _, cand_test_margin = confidence_margin(cand_test)

    train_changed = base_pred != cand_pred
    test_changed = base_test_pred != cand_test_pred

    transition_specs = {
        "star_to_galaxy": (base_pred == TARGET_MAP["STAR"]) & (cand_pred == TARGET_MAP["GALAXY"]),
        "star_to_qso": (base_pred == TARGET_MAP["STAR"]) & (cand_pred == TARGET_MAP["QSO"]),
        "star_to_any": (base_pred == TARGET_MAP["STAR"]) & train_changed,
        "non_qso_to_qso": (base_pred != TARGET_MAP["QSO"]) & (cand_pred == TARGET_MAP["QSO"]),
        "all_changed": train_changed,
    }
    test_transition_specs = {
        "star_to_galaxy": (base_test_pred == TARGET_MAP["STAR"]) & (cand_test_pred == TARGET_MAP["GALAXY"]),
        "star_to_qso": (base_test_pred == TARGET_MAP["STAR"]) & (cand_test_pred == TARGET_MAP["QSO"]),
        "star_to_any": (base_test_pred == TARGET_MAP["STAR"]) & test_changed,
        "non_qso_to_qso": (base_test_pred != TARGET_MAP["QSO"]) & (cand_test_pred == TARGET_MAP["QSO"]),
        "all_changed": test_changed,
    }
    region_specs = {
        "all": (np.ones(len(train_fe), dtype=bool), np.ones(len(test_fe), dtype=bool)),
        "rz_0_2": (train_fe["redshift_bin"].isin([0, 1, 2]).to_numpy(), test_fe["redshift_bin"].isin([0, 1, 2]).to_numpy()),
        "rz_1_2": (train_fe["redshift_bin"].isin([1, 2]).to_numpy(), test_fe["redshift_bin"].isin([1, 2]).to_numpy()),
        "rz_0": (train_fe["redshift_bin"].eq(0).to_numpy(), test_fe["redshift_bin"].eq(0).to_numpy()),
        "rz_1": (train_fe["redshift_bin"].eq(1).to_numpy(), test_fe["redshift_bin"].eq(1).to_numpy()),
        "rz_2": (train_fe["redshift_bin"].eq(2).to_numpy(), test_fe["redshift_bin"].eq(2).to_numpy()),
        "rz_0_2_gi_ge3": (
            (train_fe["redshift_bin"].isin([0, 1, 2]) & train_fe["g-i_bin"].ge(3)).to_numpy(),
            (test_fe["redshift_bin"].isin([0, 1, 2]) & test_fe["g-i_bin"].ge(3)).to_numpy(),
        ),
        "rz_0_2_gi_ge3_not_ob": (
            (train_fe["redshift_bin"].isin([0, 1, 2]) & train_fe["g-i_bin"].ge(3) & ~train_fe["spectral_type"].astype(str).eq("O/B")).to_numpy(),
            (test_fe["redshift_bin"].isin([0, 1, 2]) & test_fe["g-i_bin"].ge(3) & ~test_fe["spectral_type"].astype(str).eq("O/B")).to_numpy(),
        ),
        "rz_0_2_safe_color": (
            (
                train_fe["redshift_bin"].isin([0, 1, 2])
                & train_fe["g-i_bin"].ge(3)
                & train_fe["u-r_bin"].ge(2)
                & train_fe["mag_range_bin"].ge(2)
            ).to_numpy(),
            (
                test_fe["redshift_bin"].isin([0, 1, 2])
                & test_fe["g-i_bin"].ge(3)
                & test_fe["u-r_bin"].ge(2)
                & test_fe["mag_range_bin"].ge(2)
            ).to_numpy(),
        ),
        "gk_redseq": (
            train_fe["spectral_population"].eq("G/K_Red_Sequence").to_numpy(),
            test_fe["spectral_population"].eq("G/K_Red_Sequence").to_numpy(),
        ),
        "not_bad_ob_lowcolor": (
            (
                ~train_fe["spectral_type"].astype(str).eq("O/B")
                & ~train_fe["u-r_bin"].isin([0, 1])
                & ~train_fe["mag_range_bin"].isin([0, 1])
                & ~train_fe["g-i_bin"].isin([0, 1, 2])
            ).to_numpy(),
            (
                ~test_fe["spectral_type"].astype(str).eq("O/B")
                & ~test_fe["u-r_bin"].isin([0, 1])
                & ~test_fe["mag_range_bin"].isin([0, 1])
                & ~test_fe["g-i_bin"].isin([0, 1, 2])
            ).to_numpy(),
        ),
    }
    confidence_specs = {
        "allconf": (
            np.ones(len(train_fe), dtype=bool),
            np.ones(len(test_fe), dtype=bool),
        ),
        "uncertain_base": (
            base_margin <= 0.18,
            base_test_margin <= 0.18,
        ),
        "cand_margin_005": (
            cand_margin >= 0.05,
            cand_test_margin >= 0.05,
        ),
        "cand_margin_010": (
            cand_margin >= 0.10,
            cand_test_margin >= 0.10,
        ),
        "uncertain_base_cand_margin_005": (
            (base_margin <= 0.18) & (cand_margin >= 0.05),
            (base_test_margin <= 0.18) & (cand_test_margin >= 0.05),
        ),
    }

    gates = []
    for transition_name, transition_mask in transition_specs.items():
        test_transition_mask = test_transition_specs[transition_name]
        for region_name, (region_mask, test_region_mask) in region_specs.items():
            for confidence_name, (confidence_mask, test_confidence_mask) in confidence_specs.items():
                name = f"gate_{transition_name}_{region_name}_{confidence_name}"
                train_mask = transition_mask & region_mask & confidence_mask
                test_mask = test_transition_mask & test_region_mask & test_confidence_mask
                if train_mask.sum() == 0:
                    continue
                gates.append((name, train_mask, test_mask))
    return gates


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["class"].map(TARGET_MAP).to_numpy()

    progress("Loading OOF/test probabilities")
    base_raw_oof = load_proba(args.base_oof, len(train))
    base_raw_test = load_proba(args.base_test, len(sample))
    candidate_oof = load_proba(args.candidate_oof, len(train))
    candidate_test = load_proba(args.candidate_test, len(sample))

    stack_report = json.loads(args.stack_report.read_text(encoding="utf-8"))
    base_bias_stage = next(stage for stage in stack_report["accepted_stages"] if stage["stage"] == "base_bias")
    base_bias = np.array([base_bias_stage["bias"][label] for label in CLASSES], dtype=np.float64)
    base_bias_oof = apply_bias(base_raw_oof, base_bias)
    base_bias_test = apply_bias(base_raw_test, base_bias)

    progress("Building research feature bins")
    train_fe, test_fe = add_research_bins(train, test, args.bins)

    base_raw_pred = base_raw_oof.argmax(axis=1)
    base_bias_pred = base_bias_oof.argmax(axis=1)
    base_raw_score = balanced_accuracy(y, base_raw_pred)
    base_bias_score = balanced_accuracy(y, base_bias_pred)
    candidate_score = balanced_accuracy(y, candidate_oof.argmax(axis=1))
    progress(f"base raw OOF={base_raw_score:.9f}; base bias OOF={base_bias_score:.9f}; aggressive candidate OOF={candidate_score:.9f}")

    results: list[CandidateResult] = []
    results.append(
        score_candidate(
            "private_cv_base_bias_only",
            y,
            sample,
            train_fe,
            base_raw_pred,
            base_raw_pred,
            base_bias_oof,
            base_bias_test,
            output_dir,
            args.folds,
            args.seeds,
        )
    )

    progress("Searching guarded override gates")
    gates = build_gate_catalog(train_fe, test_fe, base_bias_oof, base_bias_test, candidate_oof, candidate_test)
    search_rows = []
    for gate_name, train_mask, test_mask in gates:
        oof = base_bias_oof.copy()
        test_proba = base_bias_test.copy()
        oof[train_mask] = candidate_oof[train_mask]
        test_proba[test_mask] = candidate_test[test_mask]
        pred = oof.argmax(axis=1)
        score = balanced_accuracy(y, pred)
        report = {
            "gate": gate_name,
            "score": float(score),
            "delta_vs_raw_lr_v9": float(score - base_raw_score),
            "delta_vs_base_bias": float(score - base_bias_score),
            "changed_rows_vs_base_bias": int((pred != base_bias_pred).sum()),
            "selected_train_rows": int(train_mask.sum()),
            "selected_test_rows": int(test_mask.sum()),
            "transition_counts_vs_base_bias": transition_counts(base_bias_pred, pred),
            "changed_outcomes_vs_base_bias": changed_outcomes(y, base_bias_pred, pred),
        }
        report.update(meta_fold_deltas(y, base_bias_pred, pred, args.folds, args.seeds))
        report.update(critical_subset_deltas(train_fe, y, base_bias_pred, pred))
        recalls = class_recalls(y, pred)
        base_recalls = class_recalls(y, base_bias_pred)
        report["worst_class_recall_delta"] = float(min(recalls[label] - base_recalls[label] for label in CLASSES))
        report["robust_rank_score"] = float(
            report["delta_vs_raw_lr_v9"]
            + 0.35 * report["meta_fold_min_delta"]
            + 0.20 * min(0.0, report["worst_subset_delta"])
            + 0.20 * min(0.0, report["worst_class_recall_delta"])
        )
        search_rows.append(report)

    search_df = pd.DataFrame(search_rows).sort_values(
        ["robust_rank_score", "score", "meta_fold_positive_rate", "worst_subset_delta"],
        ascending=[False, False, False, False],
    )
    search_df.to_csv(output_dir / "guarded_gate_search.csv", index=False)

    accepted_gates = []
    for row in search_df.to_dict(orient="records"):
        if row["delta_vs_raw_lr_v9"] <= 0:
            continue
        if row["meta_fold_positive_rate"] < 0.52:
            continue
        if row["worst_class_recall_delta"] < -0.0012:
            continue
        if row["worst_subset_delta"] < -0.0040:
            continue
        accepted_gates.append(row)
        if len(accepted_gates) >= max(8, args.top_k):
            break

    for idx, row in enumerate(accepted_gates, start=1):
        gate_name = row["gate"]
        train_mask, test_mask = next((tr, te) for name, tr, te in gates if name == gate_name)
        oof = base_bias_oof.copy()
        test_proba = base_bias_test.copy()
        oof[train_mask] = candidate_oof[train_mask]
        test_proba[test_mask] = candidate_test[test_mask]
        safe_name = gate_name.replace("gate_", f"private_cv_guarded_{idx:02d}_")
        results.append(
            score_candidate(
                safe_name,
                y,
                sample,
                train_fe,
                base_raw_pred,
                base_bias_pred,
                oof,
                test_proba,
                output_dir,
                args.folds,
                args.seeds,
            )
        )

    reports = [result.report for result in results]
    report_df = pd.DataFrame(reports).sort_values(
        ["robust_rank_score", "oof_balanced_accuracy", "meta_fold_positive_rate", "worst_subset_delta"],
        ascending=[False, False, False, False],
    )
    report_df.to_csv(output_dir / "candidate_summary.csv", index=False)

    top_results = []
    seen_test_prediction_hashes: set[str] = set()
    for row in report_df.to_dict(orient="records"):
        result = next(item for item in results if item.name == row["name"])
        test_pred = result.test_proba.argmax(axis=1).astype(np.int8)
        digest = hashlib.sha1(test_pred.tobytes()).hexdigest()
        if digest in seen_test_prediction_hashes:
            continue
        seen_test_prediction_hashes.add(digest)
        top_results.append(result)
        if len(top_results) >= args.top_k:
            break

    copied = []
    for rank, result in enumerate(top_results, start=args.output_rank_start):
        score_code = int(round(result.report["oof_balanced_accuracy"] * 1_000_000))
        short = result.name.replace("private_cv_", "")
        output_name = f"{rank:02d}_PRIVATE_CV_{short}_oof{score_code}.csv"
        output_path = OUTPUTS / output_name
        make_submission(output_path, sample, result.test_proba)
        copied.append(
            {
                "rank": rank,
                "path": str(output_path.relative_to(ROOT)),
                "name": result.name,
                "oof_balanced_accuracy": result.report["oof_balanced_accuracy"],
                "delta_vs_raw_lr_v9": result.report["delta_vs_raw_lr_v9"],
                "robust_rank_score": result.report["robust_rank_score"],
                "changed_rows_vs_reference": result.report["changed_rows_vs_reference"],
                "test_changed_rows_vs_reference": int((result.test_proba.argmax(axis=1) != base_bias_test.argmax(axis=1)).sum()),
            }
        )

    final_report = {
        "purpose": "Private-focused CV-stable submissions generated without public LB scores or public submission-bank CSVs.",
        "base_raw_oof": base_raw_score,
        "base_bias_oof": base_bias_score,
        "aggressive_candidate_oof": candidate_score,
        "base_bias": dict(zip(CLASSES, base_bias.tolist())),
        "selected_outputs": copied,
        "candidate_summary": str((output_dir / "candidate_summary.csv").relative_to(ROOT)),
        "guarded_gate_search": str((output_dir / "guarded_gate_search.csv").relative_to(ROOT)),
    }
    (output_dir / "report.json").write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
