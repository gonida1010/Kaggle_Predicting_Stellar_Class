from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

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


OUT_DIR = ARTIFACTS / "oof_generalization_stack"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "OOF-only generalization optimizer. It never reads public leaderboard scores or submission-bank CSVs. "
            "It starts from lr-stacker-v9 and keeps only candidates that improve OOF balanced accuracy."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--base-model", type=str, default="lr-stacker-v9-public-oof")
    parser.add_argument("--max-added-weight", type=float, default=0.35)
    parser.add_argument("--weight-steps", type=int, default=36)
    parser.add_argument("--bias-low", type=float, default=0.82)
    parser.add_argument("--bias-high", type=float, default=1.22)
    parser.add_argument("--bias-steps", type=int, default=33)
    parser.add_argument("--rounds", type=int, default=3)
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
        out[label] = float((y_pred[mask] == idx).mean()) if mask.any() else float("nan")
    return out


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix.tolist()


def apply_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return normalize_probs(proba * bias.reshape(1, -1))


def score_proba(y: np.ndarray, proba: np.ndarray, bias: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    adjusted = apply_bias(proba, bias) if bias is not None else normalize_probs(proba)
    pred = adjusted.argmax(axis=1)
    return balanced_accuracy(y, pred), pred


def optimize_bias(
    y: np.ndarray,
    proba: np.ndarray,
    start_bias: np.ndarray,
    low: float,
    high: float,
    steps: int,
    rounds: int,
) -> tuple[np.ndarray, float, np.ndarray, list[dict]]:
    bias = start_bias.astype(np.float64).copy()
    best_score, best_pred = score_proba(y, proba, bias)
    rows = []
    for round_idx in range(rounds):
        improved = False
        for class_idx, label in enumerate(CLASSES):
            for multiplier in np.linspace(low, high, steps):
                trial = bias.copy()
                trial[class_idx] *= float(multiplier)
                trial = trial / trial.mean()
                score, pred = score_proba(y, proba, trial)
                row = {
                    "round": round_idx + 1,
                    "class": label,
                    "multiplier": float(multiplier),
                    "score": float(score),
                    "bias": dict(zip(CLASSES, trial.tolist())),
                }
                rows.append(row)
                if score > best_score:
                    best_score = score
                    best_pred = pred
                    bias = trial
                    improved = True
        if not improved:
            break
    return bias, best_score, best_pred, rows


def blend(base: np.ndarray, candidate: np.ndarray, weight: float) -> np.ndarray:
    return normalize_probs((1.0 - weight) * base + weight * candidate)


def load_records(include_own_models: bool) -> list[dict]:
    train = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    records = []
    records.extend(archive4_pairs(len(train), len(sample)))
    records.extend(local_file_pairs(len(train), len(sample)))
    if include_own_models:
        records.extend(own_model_pairs(len(train), len(sample)))
    names = set()
    unique = []
    for record in records:
        if record["name"] in names:
            continue
        names.add(record["name"])
        unique.append(record)
    return unique


def write_submission(path: Path, sample: pd.DataFrame, test_proba: np.ndarray, bias: np.ndarray) -> None:
    adjusted = apply_bias(test_proba, bias)
    submission = sample.copy()
    submission["class"] = np.array(CLASSES)[adjusted.argmax(axis=1)]
    submission.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["class"].map(TARGET_MAP).to_numpy()
    records = load_records(args.include_own_models)
    record_by_name = {record["name"]: record for record in records}
    if args.base_model not in record_by_name:
        raise FileNotFoundError(f"Base model {args.base_model!r} not found. Available: {sorted(record_by_name)}")

    base_record = record_by_name[args.base_model]
    base_oof = normalize_probs(base_record["oof"])
    base_test = normalize_probs(base_record["test"])
    raw_base_score, raw_base_pred = score_proba(y, base_oof)
    progress(f"base={args.base_model} raw OOF BAC={raw_base_score:.9f}")

    unit_bias = np.ones(len(CLASSES), dtype=np.float64)
    best_bias, best_score, best_pred, bias_rows = optimize_bias(
        y,
        base_oof,
        unit_bias,
        args.bias_low,
        args.bias_high,
        args.bias_steps,
        args.rounds,
    )
    best_oof = base_oof.copy()
    best_test = base_test.copy()
    stage_rows = [
        {
            "stage": "base_raw",
            "score": float(raw_base_score),
            "delta_vs_raw_base": 0.0,
            "model": args.base_model,
            "weight": 0.0,
            "bias": dict(zip(CLASSES, unit_bias.tolist())),
        },
        {
            "stage": "base_bias",
            "score": float(best_score),
            "delta_vs_raw_base": float(best_score - raw_base_score),
            "model": args.base_model,
            "weight": 0.0,
            "bias": dict(zip(CLASSES, best_bias.tolist())),
        },
    ]
    progress(f"base class-bias OOF BAC={best_score:.9f}; delta={best_score - raw_base_score:+.9f}")

    candidate_rows = []
    candidate_names = [name for name in record_by_name if name != args.base_model]
    for round_idx in range(args.rounds):
        progress(f"blend search round {round_idx + 1}/{args.rounds}")
        improved = False
        round_best = None
        for name in candidate_names:
            candidate = record_by_name[name]
            candidate_oof = normalize_probs(candidate["oof"])
            candidate_test = normalize_probs(candidate["test"])
            single_score, _ = score_proba(y, candidate_oof)
            for weight in np.linspace(0.0, args.max_added_weight, args.weight_steps + 1)[1:]:
                blended_oof = blend(best_oof, candidate_oof, float(weight))
                bias, score, pred, _ = optimize_bias(
                    y,
                    blended_oof,
                    best_bias,
                    args.bias_low,
                    args.bias_high,
                    max(9, args.bias_steps // 2),
                    1,
                )
                row = {
                    "round": round_idx + 1,
                    "candidate": name,
                    "candidate_single_oof": float(single_score),
                    "weight": float(weight),
                    "score": float(score),
                    "delta_vs_current": float(score - best_score),
                    "bias": dict(zip(CLASSES, bias.tolist())),
                }
                candidate_rows.append(row)
                if score > best_score and (round_best is None or score > round_best["score"]):
                    round_best = {
                        **row,
                        "oof": blended_oof,
                        "test": blend(best_test, candidate_test, float(weight)),
                        "bias_array": bias,
                        "pred": pred,
                    }
        if round_best is not None:
            best_score = float(round_best["score"])
            best_oof = round_best["oof"]
            best_test = round_best["test"]
            best_bias = round_best["bias_array"]
            best_pred = round_best["pred"]
            improved = True
            stage_rows.append(
                {
                    "stage": f"blend_round_{round_idx + 1}",
                    "score": best_score,
                    "delta_vs_raw_base": float(best_score - raw_base_score),
                    "model": round_best["candidate"],
                    "weight": round_best["weight"],
                    "bias": dict(zip(CLASSES, best_bias.tolist())),
                }
            )
            progress(
                f"accepted {round_best['candidate']} weight={round_best['weight']:.4f} "
                f"OOF BAC={best_score:.9f}"
            )
        if not improved:
            progress("no OOF-improving blend found; stopping")
            break

    pd.DataFrame(bias_rows).sort_values("score", ascending=False).to_csv(
        args.output_dir / "base_bias_search.csv",
        index=False,
    )
    pd.DataFrame(candidate_rows).sort_values("score", ascending=False).to_csv(
        args.output_dir / "blend_search.csv",
        index=False,
    )
    pd.DataFrame(stage_rows).to_csv(args.output_dir / "accepted_stages.csv", index=False)
    np.save(args.output_dir / "generalization_stack_oof.npy", apply_bias(best_oof, best_bias).astype(np.float32))
    np.save(args.output_dir / "generalization_stack_test.npy", apply_bias(best_test, best_bias).astype(np.float32))
    submission_path = args.output_dir / "generalization_stack_submission.csv"
    write_submission(submission_path, sample, best_test, best_bias)

    report = {
        "purpose": "OOF/CV-only generalization stack optimizer. No public LB scores or submission-bank CSVs are used.",
        "base_model": args.base_model,
        "available_models": sorted(record_by_name),
        "raw_base_oof_balanced_accuracy": raw_base_score,
        "best_oof_balanced_accuracy": best_score,
        "delta_vs_raw_base": float(best_score - raw_base_score),
        "best_bias": dict(zip(CLASSES, best_bias.tolist())),
        "accepted_stages": stage_rows,
        "class_recalls": class_recalls(y, best_pred),
        "confusion_matrix": confusion_matrix(y, best_pred),
        "submission_path": str(submission_path.relative_to(ROOT)),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
