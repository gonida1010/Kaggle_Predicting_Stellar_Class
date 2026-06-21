from __future__ import annotations

import argparse
import json
import os
from collections import Counter
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from build_available_prediction_stacker import (
    ARTIFACTS,
    CLASSES,
    DATA,
    TARGET_MAP,
    archive4_pairs,
    local_file_pairs,
    own_model_pairs,
    prob_to_logit,
)


OUT_DIR = ARTIFACTS / "classwise_logistic_blender"
EPS = 1e-15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one binary logistic meta-model per target class from available OOF/test prediction pairs. "
            "This is designed to calibrate OVR model signals instead of raw multiclass normalization."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=700)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--boost-star", type=float, default=1.0)
    parser.add_argument("--bias-low", type=float, default=0.88)
    parser.add_argument("--bias-high", type=float, default=1.14)
    parser.add_argument("--bias-steps", type=int, default=27)
    parser.add_argument("--include-own-models", action="store_true", default=True)
    parser.add_argument("--no-own-models", dest="include_own_models", action="store_false")
    parser.add_argument("--only-models", nargs="*", default=None)
    parser.add_argument("--exclude-models", nargs="*", default=[])
    parser.add_argument("--log-period", type=int, default=1)
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_probs(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    proba = np.clip(proba, EPS, None)
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return proba / row_sum


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        label: float((y_pred[y_true == idx] == idx).mean())
        for idx, label in enumerate(CLASSES)
    }


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def transition_counts(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    changed = before != after
    counts = Counter(
        f"{CLASSES[int(b)]}->{CLASSES[int(a)]}"
        for b, a in zip(before[changed], after[changed])
    )
    return dict(sorted(counts.items()))


def apply_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return normalize_probs(proba * bias.reshape(1, -1))


def load_records(include_own_models: bool) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, np.ndarray]:
    train = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["class"].map(TARGET_MAP).to_numpy()

    records = []
    records.extend(archive4_pairs(len(train), len(sample)))
    records.extend(local_file_pairs(len(train), len(sample)))
    if include_own_models:
        records.extend(own_model_pairs(len(train), len(sample)))

    unique = []
    seen = set()
    for record in records:
        if record["name"] in seen:
            continue
        seen.add(record["name"])
        unique.append(record)
    return unique, train, sample, y


def filter_records(records: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.only_models:
        allowed = set(args.only_models)
        records = [record for record in records if record["name"] in allowed]
    if args.exclude_models:
        blocked = set(args.exclude_models)
        records = [record for record in records if record["name"] not in blocked]
    if len(records) < 2:
        raise RuntimeError(f"Need at least two prediction pairs, found {len(records)}")
    return records


def make_feature_matrix(records: list[dict], key: str) -> np.ndarray:
    blocks = [prob_to_logit(record[key]) for record in records]
    return np.concatenate(blocks, axis=1).astype(np.float32)


def sample_weight_for_binary(y_binary: np.ndarray, class_idx: int, args: argparse.Namespace) -> np.ndarray:
    weights = compute_sample_weight("balanced", y_binary).astype(np.float64)
    weights[y_binary == 1] *= float(args.positive_weight)
    if CLASSES[class_idx] == "STAR":
        weights[y_binary == 1] *= float(args.boost_star)
    return weights.astype(np.float32)


def fit_predict_classwise(
    x_all: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    n_train = len(y)
    n_test = len(x_test)
    oof_sum = np.zeros((n_train, len(CLASSES)), dtype=np.float64)
    test_sum = np.zeros((n_test, len(CLASSES)), dtype=np.float64)
    fold_rows = []
    binary_rows = []

    seeds = list(range(20260620, 20260620 + args.seeds))
    for seed_no, seed in enumerate(seeds, start=1):
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        seed_oof_scores = np.zeros((n_train, len(CLASSES)), dtype=np.float64)
        seed_test_scores = np.zeros((n_test, len(CLASSES)), dtype=np.float64)
        progress(f"seed {seed_no}/{len(seeds)} random_state={seed}")

        for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(n_train), y), start=1):
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x_all[tr_idx])
            x_va = scaler.transform(x_all[va_idx])
            x_te = scaler.transform(x_test)

            fold_valid_scores = np.zeros((len(va_idx), len(CLASSES)), dtype=np.float64)
            fold_test_scores = np.zeros((n_test, len(CLASSES)), dtype=np.float64)

            for class_idx, label in enumerate(CLASSES):
                y_tr_bin = (y[tr_idx] == class_idx).astype(int)
                y_va_bin = (y[va_idx] == class_idx).astype(int)
                weights = sample_weight_for_binary(y_tr_bin, class_idx, args)
                model = LogisticRegression(
                    C=float(args.c),
                    solver="lbfgs",
                    max_iter=int(args.max_iter),
                    random_state=seed + class_idx,
                )
                model.fit(x_tr, y_tr_bin, sample_weight=weights)
                valid_score = model.predict_proba(x_va)[:, 1]
                test_score = model.predict_proba(x_te)[:, 1]
                fold_valid_scores[:, class_idx] = valid_score
                fold_test_scores[:, class_idx] = test_score

                bin_pred = (valid_score >= 0.5).astype(int)
                binary_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "class": label,
                        "binary_balanced_accuracy": float(balanced_accuracy_score(y_va_bin, bin_pred)),
                        "binary_logloss": float(log_loss(y_va_bin, np.clip(valid_score, EPS, 1.0 - EPS))),
                        "positive_rate_valid": float(y_va_bin.mean()),
                        "mean_pred_positive": float(valid_score.mean()),
                    }
                )

            fold_valid_proba = normalize_probs(fold_valid_scores)
            fold_test_proba = normalize_probs(fold_test_scores)
            seed_oof_scores[va_idx] = fold_valid_scores
            seed_test_scores += fold_test_scores / args.folds

            fold_pred = fold_valid_proba.argmax(axis=1)
            fold_bac = balanced_accuracy(y[va_idx], fold_pred)
            fold_rows.append({"seed": seed, "fold": fold, "balanced_accuracy": fold_bac})
            if fold % max(1, args.log_period) == 0:
                progress(f"seed={seed} fold={fold}/{args.folds} classwise multiclass BAC={fold_bac:.6f}")

        seed_oof_proba = normalize_probs(seed_oof_scores)
        seed_test_proba = normalize_probs(seed_test_scores)
        seed_score = balanced_accuracy(y, seed_oof_proba.argmax(axis=1))
        progress(f"seed={seed} OOF BAC={seed_score:.9f}")
        oof_sum += seed_oof_proba
        test_sum += seed_test_proba

    return (
        normalize_probs(oof_sum / len(seeds)),
        normalize_probs(test_sum / len(seeds)),
        pd.DataFrame(fold_rows),
        pd.DataFrame(binary_rows),
    )


def bias_search(y: np.ndarray, proba: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float, pd.DataFrame]:
    best_bias = np.ones(len(CLASSES), dtype=np.float64)
    best_score = balanced_accuracy(y, proba.argmax(axis=1))
    rows = []
    grid = np.linspace(args.bias_low, args.bias_high, args.bias_steps)
    for g in grid:
        for q in grid:
            trial = np.array([g, q, 1.0], dtype=np.float64)
            trial = trial / trial.mean()
            adjusted = apply_bias(proba, trial)
            score = balanced_accuracy(y, adjusted.argmax(axis=1))
            row = {
                "GALAXY_bias": float(trial[0]),
                "QSO_bias": float(trial[1]),
                "STAR_bias": float(trial[2]),
                "balanced_accuracy": float(score),
            }
            rows.append(row)
            if score > best_score:
                best_score = score
                best_bias = trial
    return best_bias, best_score, pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)


def final_importance(
    x_all: np.ndarray,
    y: np.ndarray,
    model_names: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_all)
    rows = []
    for class_idx, label in enumerate(CLASSES):
        y_bin = (y == class_idx).astype(int)
        weights = sample_weight_for_binary(y_bin, class_idx, args)
        model = LogisticRegression(C=float(args.c), solver="lbfgs", max_iter=int(args.max_iter))
        model.fit(x_scaled, y_bin, sample_weight=weights)
        coef = np.abs(model.coef_[0])
        for idx, name in enumerate(model_names):
            block = coef[idx * len(CLASSES) : (idx + 1) * len(CLASSES)]
            rows.append(
                {
                    "target_class": label,
                    "model": name,
                    "importance": float(block.sum()),
                    "coef_GALAXY_feature": float(block[0]),
                    "coef_QSO_feature": float(block[1]),
                    "coef_STAR_feature": float(block[2]),
                }
            )
    return pd.DataFrame(rows).sort_values(["target_class", "importance"], ascending=[True, False])


def save_submission(path: Path, sample: pd.DataFrame, proba: np.ndarray) -> None:
    submission = sample.copy()
    submission["class"] = np.array(CLASSES)[proba.argmax(axis=1)]
    submission.to_csv(path, index=False)


def style_axes(ax) -> None:
    ax.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#d0d5dd")


def save_fold_plot(fold_scores: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=170)
    grouped = fold_scores.groupby("fold")["balanced_accuracy"]
    mean = grouped.mean()
    std = grouped.std().fillna(0.0)
    x = mean.index.to_numpy()
    ax.plot(x, mean.to_numpy(), marker="o", color="#2563eb", linewidth=2.0, label="mean")
    ax.fill_between(x, mean - std, mean + std, color="#93c5fd", alpha=0.28, label="seed std")
    for xi, yi in zip(x, mean):
        ax.text(xi, yi + 0.00004, f"{yi:.6f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xlabel("Meta fold")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Class-wise Logistic Blender: OOF Fold Balanced Accuracy", weight="bold")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_binary_plot(binary_scores: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.3), dpi=170)
    for label, group in binary_scores.groupby("class"):
        view = group.groupby("fold")["binary_balanced_accuracy"].mean()
        ax.plot(view.index, view.values, marker="o", linewidth=2.0, label=label)
    ax.set_xticks(sorted(binary_scores["fold"].unique()))
    ax.set_xlabel("Meta fold")
    ax.set_ylabel("Binary balanced accuracy")
    ax.set_title("Class-wise Binary Calibrator Quality", weight="bold")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_recall_plot(base_recalls: dict[str, float], cand_recalls: dict[str, float], path: Path) -> None:
    x = np.arange(len(CLASSES))
    w = 0.34
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=170)
    base_vals = [base_recalls[label] for label in CLASSES]
    cand_vals = [cand_recalls[label] for label in CLASSES]
    ax.bar(x - w / 2, base_vals, width=w, color="#475467", label="reference")
    ax.bar(x + w / 2, cand_vals, width=w, color="#12b76a", label="classwise blender")
    for xi, b, c in zip(x, base_vals, cand_vals):
        ax.text(xi - w / 2, b + 0.00045, f"{b:.5f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, c + 0.00045, f"{c:.5f}", ha="center", fontsize=8)
        ax.text(xi, min(b, c) - 0.0013, f"{c-b:+.5f}", ha="center", fontsize=8, color="#344054")
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Recall")
    ax.set_title("OOF Class Recall vs Reference", weight="bold")
    ax.legend(frameon=False)
    ax.set_ylim(min(base_vals + cand_vals) - 0.003, max(base_vals + cand_vals) + 0.003)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_confusion_delta_plot(y: np.ndarray, ref_pred: np.ndarray, cand_pred: np.ndarray, path: Path) -> None:
    base_cm = confusion_matrix(y, ref_pred, labels=[0, 1, 2])
    cand_cm = confusion_matrix(y, cand_pred, labels=[0, 1, 2])
    delta = cand_cm - base_cm
    vmax = max(1, int(np.abs(delta).max()))
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=170)
    im = ax.imshow(delta, cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(CLASSES)), CLASSES)
    ax.set_yticks(np.arange(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("OOF Confusion Delta: Classwise - Reference", weight="bold")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{delta[i, j]:+d}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.82, label="count delta")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_importance_plot(importance: pd.DataFrame, path: Path, top_n: int = 14) -> None:
    pivot = (
        importance.groupby(["model", "target_class"])["importance"]
        .sum()
        .unstack("target_class")
        .reindex(columns=CLASSES)
        .fillna(0.0)
    )
    pivot["total"] = pivot.sum(axis=1)
    view = pivot.sort_values("total", ascending=False).head(top_n).drop(columns=["total"])
    fig, ax = plt.subplots(figsize=(10.5, max(5.0, len(view) * 0.34)), dpi=170)
    y = np.arange(len(view))
    left = np.zeros(len(view))
    colors = ["#2563eb", "#f97316", "#16a34a"]
    for label, color in zip(CLASSES, colors):
        vals = view[label].to_numpy()
        ax.barh(y, vals, left=left, label=label, color=color)
        left += vals
    ax.set_yticks(y, view.index)
    ax.invert_yaxis()
    ax.set_xlabel("Sum of absolute standardized coefficients")
    ax.set_title("Class-wise Blender Model Importance", weight="bold")
    ax.legend(frameon=False, ncols=3)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, train, sample, y = load_records(args.include_own_models)
    records = filter_records(records, args)
    model_names = [record["name"] for record in records]
    progress(f"loaded {len(records)} prediction pairs")
    progress(", ".join(model_names))

    x_all = make_feature_matrix(records, "oof")
    x_test = make_feature_matrix(records, "test")
    progress(f"feature matrix train={x_all.shape} test={x_test.shape}")

    reference = next((record for record in records if record["name"] == "lr-stacker-v9-public-oof"), None)
    if reference is None:
        reference = max(records, key=lambda record: balanced_accuracy(y, record["oof"].argmax(axis=1)))
    ref_oof = normalize_probs(reference["oof"])
    ref_test = normalize_probs(reference["test"])
    ref_pred = ref_oof.argmax(axis=1)
    ref_score = balanced_accuracy(y, ref_pred)
    progress(f"reference={reference['name']} OOF BAC={ref_score:.9f}")

    oof_raw, test_raw, fold_scores, binary_scores = fit_predict_classwise(x_all, x_test, y, args)
    raw_score = balanced_accuracy(y, oof_raw.argmax(axis=1))
    progress(f"classwise raw OOF BAC={raw_score:.9f}")

    best_bias, best_score, bias_rows = bias_search(y, oof_raw, args)
    oof = apply_bias(oof_raw, best_bias)
    test = apply_bias(test_raw, best_bias)
    pred = oof.argmax(axis=1)
    progress(f"classwise biased OOF BAC={best_score:.9f}; bias={dict(zip(CLASSES, best_bias.tolist()))}")

    importance = final_importance(x_all, y, model_names, args)

    np.save(args.output_dir / "classwise_blender_oof_raw.npy", oof_raw.astype(np.float32))
    np.save(args.output_dir / "classwise_blender_test_raw.npy", test_raw.astype(np.float32))
    np.save(args.output_dir / "classwise_blender_oof.npy", oof.astype(np.float32))
    np.save(args.output_dir / "classwise_blender_test.npy", test.astype(np.float32))
    save_submission(args.output_dir / "classwise_blender_submission.csv", sample, test)
    fold_scores.to_csv(args.output_dir / "fold_scores.csv", index=False)
    binary_scores.to_csv(args.output_dir / "binary_fold_scores.csv", index=False)
    bias_rows.to_csv(args.output_dir / "bias_search.csv", index=False)
    importance.to_csv(args.output_dir / "model_class_importance.csv", index=False)

    save_fold_plot(fold_scores, args.output_dir / "classwise_fold_bac.png")
    save_fold_plot(fold_scores, args.output_dir / "classwise_fold_bac.svg")
    save_binary_plot(binary_scores, args.output_dir / "classwise_binary_bac.png")
    save_binary_plot(binary_scores, args.output_dir / "classwise_binary_bac.svg")
    save_recall_plot(
        class_recalls(y, ref_pred),
        class_recalls(y, pred),
        args.output_dir / "classwise_recall_vs_reference.png",
    )
    save_recall_plot(
        class_recalls(y, ref_pred),
        class_recalls(y, pred),
        args.output_dir / "classwise_recall_vs_reference.svg",
    )
    save_confusion_delta_plot(y, ref_pred, pred, args.output_dir / "classwise_confusion_delta.png")
    save_confusion_delta_plot(y, ref_pred, pred, args.output_dir / "classwise_confusion_delta.svg")
    save_importance_plot(importance, args.output_dir / "classwise_model_importance.png")
    save_importance_plot(importance, args.output_dir / "classwise_model_importance.svg")

    test_ref_pred = ref_test.argmax(axis=1)
    test_pred = test.argmax(axis=1)
    report = {
        "purpose": "Class-wise logistic blending of available OOF/test prediction pairs. No public LB score is used.",
        "models": model_names,
        "n_models": len(model_names),
        "folds": args.folds,
        "seeds": args.seeds,
        "C": args.c,
        "max_iter": args.max_iter,
        "positive_weight": args.positive_weight,
        "boost_star": args.boost_star,
        "reference_model": reference["name"],
        "reference_oof_balanced_accuracy": ref_score,
        "raw_oof_balanced_accuracy": raw_score,
        "biased_oof_balanced_accuracy": best_score,
        "delta_vs_reference": float(best_score - ref_score),
        "best_bias": dict(zip(CLASSES, best_bias.tolist())),
        "class_recalls": class_recalls(y, pred),
        "reference_class_recalls": class_recalls(y, ref_pred),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "transition_counts_vs_reference_oof": transition_counts(ref_pred, pred),
        "changed_rows_vs_reference_oof": int((ref_pred != pred).sum()),
        "transition_counts_vs_reference_test": transition_counts(test_ref_pred, test_pred),
        "changed_rows_vs_reference_test": int((test_ref_pred != test_pred).sum()),
        "submission_class_share": pd.Series(np.array(CLASSES)[test_pred]).value_counts(normalize=True).sort_index().to_dict(),
        "submission_path": str((args.output_dir / "classwise_blender_submission.csv").relative_to(ROOT)),
        "outputs": [
            "classwise_blender_submission.csv",
            "classwise_blender_oof.npy",
            "classwise_blender_test.npy",
            "classwise_blender_oof_raw.npy",
            "classwise_blender_test_raw.npy",
            "fold_scores.csv",
            "binary_fold_scores.csv",
            "bias_search.csv",
            "model_class_importance.csv",
            "classwise_fold_bac.png",
            "classwise_binary_bac.png",
            "classwise_recall_vs_reference.png",
            "classwise_confusion_delta.png",
            "classwise_model_importance.png",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
