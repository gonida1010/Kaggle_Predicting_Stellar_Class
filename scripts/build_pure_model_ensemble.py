from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "pure_model_ensemble"
DEFAULT_MODELS = ["lgbm", "catboost"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an anchor-free model ensemble optimized for balanced accuracy on OOF predictions."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory where reports and pure-model submissions will be written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model artifact prefixes to ensemble, e.g. lgbm catboost lgbm_te.",
    )
    return parser.parse_args()


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for class_idx in range(n_classes):
        mask = y_true == class_idx
        if not mask.any():
            continue
        recalls.append(float((y_pred[mask] == class_idx).mean()))
    return float(np.mean(recalls))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[list[int]]:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix.tolist()


def load_model_artifacts(model: str) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    report_path = ARTIFACTS / f"{model}_baseline_report.json"
    oof_path = ARTIFACTS / f"{model}_oof_proba.npy"
    test_path = ARTIFACTS / f"{model}_test_proba.npy"
    if not report_path.exists() or not oof_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing artifacts for {model}. Run its CV training script first."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report["classes"], np.load(oof_path), np.load(test_path), report


def normalize_probs(proba: np.ndarray) -> np.ndarray:
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return proba / row_sum


def predict_with_bias(proba: np.ndarray, class_bias: np.ndarray) -> np.ndarray:
    adjusted = proba * class_bias.reshape(1, -1)
    return adjusted.argmax(axis=1)


def coordinate_search(
    y: np.ndarray,
    oof_by_model: dict[str, np.ndarray],
    classes: list[str],
    models: list[str],
    n_rounds: int = 4,
) -> tuple[dict, list[dict]]:
    n_classes = len(classes)
    records: list[dict] = []

    model_weight_values = np.ones(len(models), dtype=np.float64) / len(models)
    class_bias = np.ones(n_classes, dtype=np.float64)

    def evaluate(weight_values: np.ndarray, bias: np.ndarray) -> dict:
        weight_values = weight_values / weight_values.sum()
        model_weights = {
            model: float(weight_values[idx])
            for idx, model in enumerate(models)
        }
        base = normalize_probs(sum(model_weights[model] * oof_by_model[model] for model in models))
        pred = predict_with_bias(base, bias)
        return {
            "score": balanced_accuracy(y, pred, n_classes),
            "model_weights": model_weights,
            "class_bias": dict(zip(classes, bias.tolist())),
        }

    best = evaluate(model_weight_values, class_bias)
    records.append(best)

    weight_steps = [0.50, 0.20, 0.08, 0.03] if len(models) == 2 else [0.0, 0.0, 0.0, 0.0]
    if len(models) > 2:
        weight_steps = [0.22, 0.10, 0.04, 0.015]
    bias_steps = [0.18, 0.08, 0.035, 0.015]

    for round_idx in range(n_rounds):
        if len(models) == 2:
            weight_grid = np.clip(
                np.linspace(
                    model_weight_values[0] - weight_steps[round_idx],
                    model_weight_values[0] + weight_steps[round_idx],
                    17,
                ),
                0.0,
                1.0,
            )
            for candidate_weight in np.unique(weight_grid):
                candidate_weights = np.array([candidate_weight, 1.0 - candidate_weight], dtype=np.float64)
                record = evaluate(candidate_weights, class_bias)
                records.append(record)
                if record["score"] > best["score"]:
                    best = record
                    model_weight_values = candidate_weights
        else:
            for model_idx in range(len(models)):
                for delta in np.linspace(-weight_steps[round_idx], weight_steps[round_idx], 17):
                    candidate_weights = model_weight_values.copy()
                    candidate_weights[model_idx] = max(0.001, candidate_weights[model_idx] + float(delta))
                    candidate_weights = candidate_weights / candidate_weights.sum()
                    record = evaluate(candidate_weights, class_bias)
                    records.append(record)
                    if record["score"] > best["score"]:
                        best = record
                        model_weight_values = candidate_weights

        for class_idx in range(n_classes):
            multipliers = np.linspace(1.0 - bias_steps[round_idx], 1.0 + bias_steps[round_idx], 17)
            for multiplier in multipliers:
                candidate_bias = class_bias.copy()
                candidate_bias[class_idx] *= float(multiplier)
                # Remove irrelevant global scale so only class boundaries move.
                candidate_bias = candidate_bias / candidate_bias.mean()
                record = evaluate(model_weight_values, candidate_bias)
                records.append(record)
                if record["score"] > best["score"]:
                    best = record
                    class_bias = candidate_bias

    records.sort(key=lambda row: row["score"], reverse=True)
    return best, records[:50]


def apply_ensemble(
    proba_by_model: dict[str, np.ndarray],
    model_weights: dict[str, float],
    class_bias: dict[str, float],
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_proba = sum(model_weights[model] * proba_by_model[model] for model in model_weights)
    raw_proba = normalize_probs(raw_proba)
    bias = np.array([class_bias[cls] for cls in classes], dtype=np.float64)
    adjusted_proba = normalize_probs(raw_proba * bias.reshape(1, -1))
    pred = adjusted_proba.argmax(axis=1)
    return pred, adjusted_proba, raw_proba


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = list(dict.fromkeys(args.models))
    if not models:
        raise ValueError("At least one model prefix is required.")

    train = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y_raw = train["class"].astype(str)
    classes = sorted(y_raw.unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    y = y_raw.map(class_to_idx).to_numpy()

    oof_by_model: dict[str, np.ndarray] = {}
    test_by_model: dict[str, np.ndarray] = {}
    source_reports: dict[str, dict] = {}
    for model in models:
        model_classes, oof, test_pred, report = load_model_artifacts(model)
        if model_classes != classes:
            raise ValueError(f"{model} class order differs: {model_classes} != {classes}")
        oof_by_model[model] = oof.astype(np.float64)
        test_by_model[model] = test_pred.astype(np.float64)
        source_reports[model] = report

    baseline_records = []
    for model in models:
        pred = oof_by_model[model].argmax(axis=1)
        baseline_records.append(
            {
                "name": model,
                "oof_balanced_accuracy": balanced_accuracy(y, pred, len(classes)),
                "confusion_matrix": confusion_matrix(y, pred, len(classes)),
            }
        )

    unweighted_oof = normalize_probs(sum(oof_by_model.values()) / len(models))
    unweighted_pred = unweighted_oof.argmax(axis=1)
    baseline_records.append(
        {
            "name": "unweighted_" + "_".join(models),
            "oof_balanced_accuracy": balanced_accuracy(y, unweighted_pred, len(classes)),
            "confusion_matrix": confusion_matrix(y, unweighted_pred, len(classes)),
        }
    )

    best, search_top = coordinate_search(y, oof_by_model, classes, models)

    oof_pred, oof_proba, raw_oof_proba = apply_ensemble(
        oof_by_model,
        best["model_weights"],
        best["class_bias"],
        classes,
    )
    test_pred, test_proba, raw_test_proba = apply_ensemble(
        test_by_model,
        best["model_weights"],
        best["class_bias"],
        classes,
    )

    submission = sample.copy()
    submission["class"] = np.array(classes)[test_pred]
    submission_path = args.output_dir / "pure_model_ensemble_submission.csv"
    submission.to_csv(submission_path, index=False)
    np.save(args.output_dir / "pure_model_ensemble_oof_proba.npy", oof_proba.astype(np.float32))
    np.save(args.output_dir / "pure_model_ensemble_test_proba.npy", test_proba.astype(np.float32))
    np.save(args.output_dir / "pure_model_ensemble_raw_oof_proba.npy", raw_oof_proba.astype(np.float32))
    np.save(args.output_dir / "pure_model_ensemble_raw_test_proba.npy", raw_test_proba.astype(np.float32))

    final_score = balanced_accuracy(y, oof_pred, len(classes))
    report = {
        "purpose": "Anchor-free train/test-only model ensemble. No public submission CSV is used.",
        "classes": classes,
        "baseline_records": baseline_records,
        "best_oof_balanced_accuracy": final_score,
        "best_config": best,
        "final_confusion_matrix": confusion_matrix(y, oof_pred, len(classes)),
        "search_top": search_top[:20],
        "submission_path": str(submission_path.relative_to(ROOT)),
        "submission_class_share": submission["class"].value_counts(normalize=True).sort_index().to_dict(),
        "source_model_reports": {
            model: {
                "oof_balanced_accuracy": source_reports[model].get("oof_balanced_accuracy"),
                "seed": source_reports[model].get("seed"),
                "features": source_reports[model].get("features"),
            }
            for model in models
        },
    }
    (args.output_dir / "pure_model_ensemble_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
