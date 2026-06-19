from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "oof_fallback_submissions"
CLASSES = ["GALAXY", "QSO", "STAR"]
TARGET_MAP = {label: idx for idx, label in enumerate(CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a conservative fallback submission. The base prediction is kept by default; "
            "candidate predictions override only where OOF threshold search improves balanced accuracy."
        )
    )
    parser.add_argument("--base-oof", type=Path, required=True)
    parser.add_argument("--base-test", type=Path, required=True)
    parser.add_argument("--candidate-oof", type=Path, required=True)
    parser.add_argument("--candidate-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--name", type=str, default="fallback")
    parser.add_argument("--max-changed-frac", type=float, default=0.02)
    parser.add_argument("--min-candidate-conf", type=float, default=0.42)
    parser.add_argument("--max-candidate-conf", type=float, default=0.94)
    parser.add_argument("--candidate-conf-steps", type=int, default=27)
    parser.add_argument("--min-candidate-margin", type=float, default=0.00)
    parser.add_argument("--max-candidate-margin", type=float, default=0.45)
    parser.add_argument("--candidate-margin-steps", type=int, default=16)
    parser.add_argument(
        "--base-margin-max-values",
        nargs="+",
        type=float,
        default=[1.0, 0.45, 0.35, 0.25, 0.18, 0.12, 0.08],
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
    if arr.shape != (expected_rows, 3):
        raise ValueError(f"{path} shape {arr.shape}, expected {(expected_rows, 3)}")
    return normalize_probs(arr)


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


def confidence_and_margin(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(proba, axis=1)
    conf = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    return conf, margin


def transition_counts(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    changed = before != after
    counts = Counter(f"{CLASSES[b]}->{CLASSES[a]}" for b, a in zip(before[changed], after[changed]))
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["class"].map(TARGET_MAP).to_numpy()

    progress("Loading OOF/test probabilities")
    base_oof = load_proba(args.base_oof, len(train))
    base_test = load_proba(args.base_test, len(sample))
    cand_oof = load_proba(args.candidate_oof, len(train))
    cand_test = load_proba(args.candidate_test, len(sample))

    base_pred = base_oof.argmax(axis=1)
    cand_pred = cand_oof.argmax(axis=1)
    base_score = balanced_accuracy(y, base_pred)
    cand_score = balanced_accuracy(y, cand_pred)
    progress(f"base OOF BAC={base_score:.6f}; candidate OOF BAC={cand_score:.6f}")

    base_conf, base_margin = confidence_and_margin(base_oof)
    cand_conf, cand_margin = confidence_and_margin(cand_oof)
    disagree = base_pred != cand_pred
    max_changed = int(len(y) * float(args.max_changed_frac))

    rows = []
    best = {
        "score": base_score,
        "delta": 0.0,
        "candidate_conf_threshold": None,
        "candidate_margin_threshold": None,
        "base_margin_max": None,
        "changed_rows": 0,
        "pred": base_pred,
        "mask": np.zeros(len(y), dtype=bool),
    }
    conf_values = np.linspace(args.min_candidate_conf, args.max_candidate_conf, args.candidate_conf_steps)
    margin_values = np.linspace(args.min_candidate_margin, args.max_candidate_margin, args.candidate_margin_steps)

    progress("Searching fallback thresholds on OOF")
    for conf_thr in conf_values:
        for margin_thr in margin_values:
            for base_margin_max in args.base_margin_max_values:
                mask = (
                    disagree
                    & (cand_conf >= conf_thr)
                    & (cand_margin >= margin_thr)
                    & (base_margin <= base_margin_max)
                )
                changed_rows = int(mask.sum())
                if changed_rows == 0 or changed_rows > max_changed:
                    continue
                pred = base_pred.copy()
                pred[mask] = cand_pred[mask]
                score = balanced_accuracy(y, pred)
                row = {
                    "score": float(score),
                    "delta": float(score - base_score),
                    "candidate_conf_threshold": float(conf_thr),
                    "candidate_margin_threshold": float(margin_thr),
                    "base_margin_max": float(base_margin_max),
                    "changed_rows": changed_rows,
                    "transition_counts": transition_counts(base_pred, pred),
                    "class_recalls": class_recalls(y, pred),
                }
                rows.append(row)
                if score > best["score"]:
                    best = {**row, "pred": pred, "mask": mask}

    top_rows = sorted(rows, key=lambda row: row["score"], reverse=True)[:50]
    pd.DataFrame(top_rows).to_csv(args.output_dir / f"{args.name}_top_thresholds.csv", index=False)

    test_base_pred = base_test.argmax(axis=1)
    test_cand_pred = cand_test.argmax(axis=1)
    test_base_conf, test_base_margin = confidence_and_margin(base_test)
    test_cand_conf, test_cand_margin = confidence_and_margin(cand_test)
    test_disagree = test_base_pred != test_cand_pred

    if best["candidate_conf_threshold"] is None:
        test_pred = test_base_pred.copy()
        test_mask = np.zeros(len(sample), dtype=bool)
    else:
        test_mask = (
            test_disagree
            & (test_cand_conf >= best["candidate_conf_threshold"])
            & (test_cand_margin >= best["candidate_margin_threshold"])
            & (test_base_margin <= best["base_margin_max"])
        )
        test_pred = test_base_pred.copy()
        test_pred[test_mask] = test_cand_pred[test_mask]

    submission = sample.copy()
    submission["class"] = np.array(CLASSES)[test_pred]
    submission_path = args.output_dir / f"{args.name}_submission.csv"
    submission.to_csv(submission_path, index=False)

    best_report = {
        key: value
        for key, value in best.items()
        if key not in {"pred", "mask"}
    }
    report = {
        "purpose": "OOF-selected fallback override. Base is kept unless candidate clears confidence/margin thresholds.",
        "name": args.name,
        "base_oof": str(args.base_oof),
        "base_test": str(args.base_test),
        "candidate_oof": str(args.candidate_oof),
        "candidate_test": str(args.candidate_test),
        "base_oof_balanced_accuracy": base_score,
        "candidate_oof_balanced_accuracy": cand_score,
        "best": best_report,
        "test_changed_rows": int(test_mask.sum()),
        "test_transition_counts": transition_counts(test_base_pred, test_pred),
        "submission_path": str(submission_path.relative_to(ROOT)),
        "submission_class_share": submission["class"].value_counts(normalize=True).sort_index().to_dict(),
    }
    (args.output_dir / f"{args.name}_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
