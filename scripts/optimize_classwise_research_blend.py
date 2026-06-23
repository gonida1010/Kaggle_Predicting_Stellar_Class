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


ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"
SOURCE_CANDIDATES = [
    (
        "56_te_disagreement",
        ARTIFACTS / "te_disagreement_patch_classwise37" / "56_high_gi_low_rz_base_galaxy_to_star_c0_55_m0_15_b0_60_oof.npy",
        ARTIFACTS / "te_disagreement_patch_classwise37" / "56_high_gi_low_rz_base_galaxy_to_star_c0_55_m0_15_b0_60_test.npy",
    ),
    (
        "68_research_material_stack",
        ARTIFACTS / "research_material_stack_20260623" / "research_material_stack_oof.npy",
        ARTIFACTS / "research_material_stack_20260623" / "research_material_stack_test.npy",
    ),
    (
        "69_guarded_research_stack",
        ARTIFACTS / "private_cv_stable_research_material_stack_20260623" / "private_cv_guarded_01_all_changed_rz_0_2_allconf_oof.npy",
        ARTIFACTS / "private_cv_stable_research_material_stack_20260623" / "private_cv_guarded_01_all_changed_rz_0_2_allconf_test.npy",
    ),
    (
        "classwise_blender_c010",
        ARTIFACTS / "classwise_logistic_blender_c010" / "classwise_blender_oof.npy",
        ARTIFACTS / "classwise_logistic_blender_c010" / "classwise_blender_test.npy",
    ),
    (
        "realmlp_feature_bank_stack",
        ARTIFACTS / "oof_generalization_stack_realmlp_feature_bank_fast" / "generalization_stack_oof.npy",
        ARTIFACTS / "oof_generalization_stack_realmlp_feature_bank_fast" / "generalization_stack_test.npy",
    ),
    (
        "realmlp_feature_guard_03",
        ARTIFACTS
        / "private_cv_stable_submissions_realmlp_feature_bank_after_public"
        / "private_cv_guarded_03_all_changed_rz_0_2_safe_color_cand_margin_005_oof.npy",
        ARTIFACTS
        / "private_cv_stable_submissions_realmlp_feature_bank_after_public"
        / "private_cv_guarded_03_all_changed_rz_0_2_safe_color_cand_margin_005_test.npy",
    ),
    (
        "available_ovr_xgb_realmlp",
        ARTIFACTS / "available_prediction_stacker_with_ovr_xgb_realmlp_c010" / "available_prediction_stacker_oof.npy",
        ARTIFACTS / "available_prediction_stacker_with_ovr_xgb_realmlp_c010" / "available_prediction_stacker_test.npy",
    ),
    (
        "available_realmlp_feature_bank",
        ARTIFACTS / "available_prediction_stacker_realmlp_feature_bank_c010" / "available_prediction_stacker_oof.npy",
        ARTIFACTS / "available_prediction_stacker_realmlp_feature_bank_c010" / "available_prediction_stacker_test.npy",
    ),
    (
        "available_foldsafe_te",
        ARTIFACTS / "available_prediction_stacker_with_foldsafe_te_c010" / "available_prediction_stacker_oof.npy",
        ARTIFACTS / "available_prediction_stacker_with_foldsafe_te_c010" / "available_prediction_stacker_test.npy",
    ),
    (
        "available_catv3",
        ARTIFACTS / "available_prediction_stacker_with_catv3_c010" / "available_prediction_stacker_oof.npy",
        ARTIFACTS / "available_prediction_stacker_with_catv3_c010" / "available_prediction_stacker_test.npy",
    ),
    (
        "oof_stack_with_catv3",
        ARTIFACTS / "oof_generalization_stack_with_catv3_fast" / "generalization_stack_oof.npy",
        ARTIFACTS / "oof_generalization_stack_with_catv3_fast" / "generalization_stack_test.npy",
    ),
    (
        "oof_stack_with_foldsafe_te",
        ARTIFACTS / "oof_generalization_stack_with_foldsafe_te_fast" / "generalization_stack_oof.npy",
        ARTIFACTS / "oof_generalization_stack_with_foldsafe_te_fast" / "generalization_stack_test.npy",
    ),
    (
        "oof_stack_with_classwise_blender",
        ARTIFACTS / "oof_generalization_stack_with_classwise_blender_fast" / "generalization_stack_oof.npy",
        ARTIFACTS / "oof_generalization_stack_with_classwise_blender_fast" / "generalization_stack_test.npy",
    ),
]


@dataclass
class Source:
    name: str
    oof: np.ndarray
    test: np.ndarray
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy class-column blend over private/CV research probability sources."
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "classwise_research_blend_20260623")
    parser.add_argument("--start-source", default="68_research_material_stack")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--alpha-max", type=float, default=0.08)
    parser.add_argument("--alpha-steps", type=int, default=16)
    parser.add_argument("--bias-low", type=float, default=0.985)
    parser.add_argument("--bias-high", type=float, default=1.015)
    parser.add_argument("--bias-steps", type=int, default=13)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output-rank", type=int, default=84)
    parser.add_argument("--output-prefix", default="PRIVATE_CV_classwise_research_blend")
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


def transition_counts(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    changed = before != after
    counts = Counter(
        f"{CLASSES[int(old)]}->{CLASSES[int(new)]}"
        for old, new in zip(before[changed], after[changed])
    )
    return dict(sorted(counts.items()))


def load_sources(y: np.ndarray, n_train: int, n_test: int) -> list[Source]:
    sources = []
    for name, oof_path, test_path in SOURCE_CANDIDATES:
        if not oof_path.exists() or not test_path.exists():
            progress(f"skip missing source {name}")
            continue
        oof = load_proba(oof_path, n_train)
        test = load_proba(test_path, n_test)
        score = balanced_accuracy(y, oof.argmax(axis=1))
        sources.append(Source(name=name, oof=oof, test=test, score=score))
        progress(f"loaded {name}: OOF={score:.9f}")
    return sources


def apply_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return normalize_probs(proba * bias.reshape(1, -1))


def search_bias(y: np.ndarray, proba: np.ndarray, low: float, high: float, steps: int) -> tuple[np.ndarray, float]:
    values = np.linspace(low, high, steps)
    best_bias = np.ones(len(CLASSES), dtype=np.float64)
    best_score = balanced_accuracy(y, proba.argmax(axis=1))
    for galaxy in values:
        for qso in values:
            star = 3.0 - galaxy - qso
            if star < low or star > high:
                continue
            bias = np.array([galaxy, qso, star], dtype=np.float64)
            score = balanced_accuracy(y, apply_bias(proba, bias).argmax(axis=1))
            if score > best_score:
                best_score = score
                best_bias = bias
    return best_bias, best_score


def class_column_blend(current: np.ndarray, source: np.ndarray, class_idx: int, alpha: float) -> np.ndarray:
    out = current.copy()
    out[:, class_idx] = (1.0 - alpha) * out[:, class_idx] + alpha * source[:, class_idx]
    return normalize_probs(out)


def fold_stability(y: np.ndarray, base_pred: np.ndarray, pred: np.ndarray, folds: int, seeds: int) -> dict[str, float]:
    deltas = []
    for seed in range(20260623, 20260623 + seeds):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for _, valid_idx in splitter.split(np.zeros(len(y)), y):
            base_score = balanced_accuracy(y[valid_idx], base_pred[valid_idx])
            score = balanced_accuracy(y[valid_idx], pred[valid_idx])
            deltas.append(score - base_score)
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "meta_fold_mean_delta_vs_start": float(arr.mean()),
        "meta_fold_min_delta_vs_start": float(arr.min()),
        "meta_fold_max_delta_vs_start": float(arr.max()),
        "meta_fold_positive_rate_vs_start": float((arr > 0).mean()),
    }


def make_submission(path: Path, sample: pd.DataFrame, proba: np.ndarray) -> None:
    submission = sample.copy()
    submission["class"] = np.array(CLASSES)[proba.argmax(axis=1)]
    submission.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)

    progress("Loading labels and sample")
    train = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["class"].map(TARGET_MAP).to_numpy()

    progress("Loading source probabilities")
    sources = load_sources(y, len(train), len(sample))
    by_name = {source.name: source for source in sources}
    if args.start_source not in by_name:
        raise ValueError(f"start source {args.start_source!r} not found. Available: {sorted(by_name)}")
    start = by_name[args.start_source]

    current_oof = start.oof.copy()
    current_test = start.test.copy()
    start_pred = start.oof.argmax(axis=1)
    start_score = balanced_accuracy(y, start_pred)
    progress(f"start={start.name}; OOF={start_score:.9f}")

    stage_rows = [
        {
            "stage": "start",
            "source": start.name,
            "class": "",
            "alpha": 0.0,
            "score": start_score,
            "delta_vs_start": 0.0,
        }
    ]
    alphas = np.linspace(args.alpha_max / args.alpha_steps, args.alpha_max, args.alpha_steps)
    current_score = start_score
    for round_idx in range(1, args.rounds + 1):
        best = None
        for source in sources:
            if source.name == start.name:
                continue
            for class_idx, class_label in enumerate(CLASSES):
                for alpha in alphas:
                    trial = class_column_blend(current_oof, source.oof, class_idx, float(alpha))
                    score = balanced_accuracy(y, trial.argmax(axis=1))
                    if best is None or score > best["score"]:
                        best = {
                            "source": source,
                            "class_idx": class_idx,
                            "class": class_label,
                            "alpha": float(alpha),
                            "score": float(score),
                            "trial_oof": trial,
                        }
        if best is None or best["score"] <= current_score:
            progress(f"round {round_idx}: no improvement; stop")
            break
        current_oof = best["trial_oof"]
        current_test = class_column_blend(current_test, best["source"].test, best["class_idx"], best["alpha"])
        current_score = best["score"]
        stage_rows.append(
            {
                "stage": f"class_blend_round_{round_idx}",
                "source": best["source"].name,
                "class": best["class"],
                "alpha": best["alpha"],
                "score": current_score,
                "delta_vs_start": current_score - start_score,
            }
        )
        progress(
            f"round {round_idx}: source={best['source'].name}, class={best['class']}, "
            f"alpha={best['alpha']:.4f}, OOF={current_score:.9f}"
        )

    progress("Searching small final class bias")
    best_bias, bias_score = search_bias(y, current_oof, args.bias_low, args.bias_high, args.bias_steps)
    if bias_score > current_score:
        current_oof = apply_bias(current_oof, best_bias)
        current_test = apply_bias(current_test, best_bias)
        current_score = bias_score
        stage_rows.append(
            {
                "stage": "final_bias",
                "source": "",
                "class": "",
                "alpha": 0.0,
                "score": current_score,
                "delta_vs_start": current_score - start_score,
                "bias": {label: float(value) for label, value in zip(CLASSES, best_bias)},
            }
        )
        progress(f"accepted final bias: OOF={current_score:.9f}, bias={best_bias.tolist()}")
    else:
        progress(f"final bias rejected: best={bias_score:.9f}, current={current_score:.9f}")

    pred = current_oof.argmax(axis=1)
    test_pred = current_test.argmax(axis=1)
    stability = fold_stability(y, start_pred, pred, args.folds, args.seeds)
    score_tag = f"{current_score:.6f}".replace(".", "")
    output_name = f"{args.output_rank:02d}_{args.output_prefix}_oof{score_tag}"
    np.save(output_dir / f"{output_name}_oof.npy", current_oof.astype(np.float32))
    np.save(output_dir / f"{output_name}_test.npy", current_test.astype(np.float32))
    make_submission(output_dir / f"{output_name}.csv", sample, current_test)
    make_submission(OUTPUTS / f"{output_name}.csv", sample, current_test)

    report = {
        "purpose": "Class-column OOF blend over private/generalization research materials.",
        "start_source": start.name,
        "start_oof_balanced_accuracy": start_score,
        "final_oof_balanced_accuracy": current_score,
        "delta_vs_start": current_score - start_score,
        "class_recalls": class_recalls(y, pred),
        "start_class_recalls": class_recalls(y, start_pred),
        "transition_counts_vs_start": transition_counts(start_pred, pred),
        "changed_rows_vs_start": int((pred != start_pred).sum()),
        "test_changed_rows_vs_start": int((test_pred != start.test.argmax(axis=1)).sum()),
        "stability": stability,
        "output_csv": str((OUTPUTS / f"{output_name}.csv").relative_to(ROOT)),
        "stage_rows": stage_rows,
        "source_scores": {source.name: source.score for source in sources},
    }
    pd.DataFrame(stage_rows).to_csv(output_dir / "accepted_stages.csv", index=False)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
