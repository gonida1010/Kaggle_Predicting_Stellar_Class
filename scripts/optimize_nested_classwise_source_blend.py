from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


CLASSES = ["GALAXY", "QSO", "STAR"]
TARGET_MAP = {label: idx for idx, label in enumerate(CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested outer-fold selection of one class-column blend between two OOF probability sources."
    )
    parser.add_argument("--base-oof", type=Path, required=True)
    parser.add_argument("--base-test", type=Path, required=True)
    parser.add_argument("--candidate-oof", type=Path, required=True)
    parser.add_argument("--candidate-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memo-path", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--alpha-steps", type=int, default=101)
    parser.add_argument("--seed-start", type=int, default=20260630)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def normalize(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    return proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-15, None)


def prediction_grid(base: np.ndarray, candidate: np.ndarray, class_idx: int, alphas: np.ndarray) -> np.ndarray:
    other_classes = [idx for idx in range(3) if idx != class_idx]
    other_values = base[:, other_classes]
    other_choice = np.asarray(other_classes, dtype=np.int8)[other_values.argmax(axis=1)]
    other_max = other_values.max(axis=1)
    blended = base[:, class_idx, None] + (
        candidate[:, class_idx, None] - base[:, class_idx, None]
    ) * alphas[None, :]
    return np.where(blended > other_max[:, None], class_idx, other_choice[:, None]).astype(np.int8)


def classwise_score_from_counts(correct_counts: np.ndarray, supports: np.ndarray) -> np.ndarray:
    return (correct_counts / np.clip(supports[:, None], 1, None)).mean(axis=0)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    memo_path = resolve(args.memo_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    memo_path.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(ROOT / "data" / "train.csv")
    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    y = train["class"].map(TARGET_MAP).to_numpy()
    base_oof = normalize(np.load(resolve(args.base_oof)))
    base_test = normalize(np.load(resolve(args.base_test)))
    candidate_oof = normalize(np.load(resolve(args.candidate_oof)))
    candidate_test = normalize(np.load(resolve(args.candidate_test)))
    alphas = np.linspace(0.0, 1.0, int(args.alpha_steps))

    print(
        f"[nested] building prediction grids: rows={len(y)}, alpha_steps={len(alphas)}, "
        f"outer_splits={args.seeds}x{args.folds}",
        flush=True,
    )
    prediction_grids = [prediction_grid(base_oof, candidate_oof, class_idx, alphas) for class_idx in range(3)]
    correct_grids = []
    total_correct = []
    total_support = np.bincount(y, minlength=3)
    for grid in prediction_grids:
        per_class = np.stack([(grid[y == class_idx] == class_idx).sum(axis=0) for class_idx in range(3)])
        correct_grids.append(grid == y[:, None])
        total_correct.append(per_class)

    rows = []
    nested_delta = []
    selected_configs: list[tuple[int, int]] = []
    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.seeds)):
        splitter = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=seed)
        for fold, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
            valid_support = np.bincount(y[valid_idx], minlength=3)
            train_support = total_support - valid_support
            best_train = (-np.inf, 0, 0)
            valid_scores_by_class = []
            train_scores_by_class = []
            for class_idx in range(3):
                valid_correct = np.stack(
                    [
                        correct_grids[class_idx][valid_idx[y[valid_idx] == true_class]].sum(axis=0)
                        for true_class in range(3)
                    ]
                )
                train_correct = total_correct[class_idx] - valid_correct
                train_scores = classwise_score_from_counts(train_correct, train_support)
                valid_scores = classwise_score_from_counts(valid_correct, valid_support)
                train_scores_by_class.append(train_scores)
                valid_scores_by_class.append(valid_scores)
                alpha_idx = int(np.argmax(train_scores))
                if float(train_scores[alpha_idx]) > best_train[0]:
                    best_train = (float(train_scores[alpha_idx]), class_idx, alpha_idx)

            _, selected_class, selected_alpha_idx = best_train
            selected_alpha = float(alphas[selected_alpha_idx])
            selected_valid_score = float(valid_scores_by_class[selected_class][selected_alpha_idx])
            baseline_valid_score = float(valid_scores_by_class[0][0])
            delta = selected_valid_score - baseline_valid_score
            nested_delta.append(delta)
            selected_configs.append((selected_class, selected_alpha_idx))
            rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "selected_class": CLASSES[selected_class],
                    "selected_alpha": selected_alpha,
                    "train_balanced_accuracy": best_train[0],
                    "valid_balanced_accuracy": selected_valid_score,
                    "baseline_valid_balanced_accuracy": baseline_valid_score,
                    "valid_delta": delta,
                }
            )
            print(
                f"[nested] seed={seed} fold={fold} class={CLASSES[selected_class]} "
                f"alpha={selected_alpha:.3f} valid_delta={delta:+.9f}",
                flush=True,
            )

    nested_df = pd.DataFrame(rows)
    nested_df.to_csv(output_dir / "nested_fold_results.csv", index=False)
    nested_delta_arr = np.asarray(nested_delta, dtype=np.float64)
    config_counts = Counter((CLASSES[class_idx], float(alphas[alpha_idx])) for class_idx, alpha_idx in selected_configs)
    selected_class = max(
        range(3),
        key=lambda class_idx: sum(count for (label, _), count in config_counts.items() if label == CLASSES[class_idx]),
    )

    full_scores = []
    for class_idx in range(3):
        counts = total_correct[class_idx]
        full_scores.append(classwise_score_from_counts(counts, total_support))
    selected_alpha_idx = int(np.argmax(full_scores[selected_class]))
    selected_alpha = float(alphas[selected_alpha_idx])
    final_oof = base_oof.copy()
    final_test = base_test.copy()
    final_oof[:, selected_class] = (
        (1.0 - selected_alpha) * base_oof[:, selected_class]
        + selected_alpha * candidate_oof[:, selected_class]
    )
    final_test[:, selected_class] = (
        (1.0 - selected_alpha) * base_test[:, selected_class]
        + selected_alpha * candidate_test[:, selected_class]
    )
    final_oof = normalize(final_oof)
    final_test = normalize(final_test)
    base_score = float(balanced_accuracy_score(y, base_oof.argmax(axis=1)))
    final_score = float(balanced_accuracy_score(y, final_oof.argmax(axis=1)))

    rng = np.random.default_rng(int(args.seed_start))
    base_pred = base_oof.argmax(axis=1)
    final_pred = final_oof.argmax(axis=1)
    class_rows = [np.flatnonzero(y == class_idx) for class_idx in range(3)]
    bootstrap = np.empty(int(args.bootstrap_repeats), dtype=np.float64)
    for repeat in range(len(bootstrap)):
        delta_recalls = []
        for class_idx, indices in enumerate(class_rows):
            sampled = rng.choice(indices, size=len(indices), replace=True)
            delta_recalls.append(
                np.mean(final_pred[sampled] == class_idx) - np.mean(base_pred[sampled] == class_idx)
            )
        bootstrap[repeat] = np.mean(delta_recalls)
    np.save(output_dir / "bootstrap_final_deltas.npy", bootstrap)

    nested_positive_rate = float(np.mean(nested_delta_arr > 0))
    nested_mean = float(nested_delta_arr.mean())
    nested_min = float(nested_delta_arr.min())
    bootstrap_ci = np.quantile(bootstrap, [0.025, 0.5, 0.975])
    bootstrap_positive = float(np.mean(bootstrap > 0))
    if nested_mean > 0 and nested_positive_rate >= 0.60 and final_score > base_score:
        decision = "promote_as_research_source"
    elif final_score > base_score:
        decision = "research_only"
    else:
        decision = "reject"

    np.save(output_dir / "nested_classwise_oof.npy", final_oof.astype(np.float32))
    np.save(output_dir / "nested_classwise_test.npy", final_test.astype(np.float32))
    submission = sample.copy()
    submission["class"] = np.asarray(CLASSES)[final_test.argmax(axis=1)]
    submission.to_csv(output_dir / "nested_classwise_submission.csv", index=False)

    result = {
        "decision": decision,
        "selected_class": CLASSES[selected_class],
        "selected_alpha_full_oof": selected_alpha,
        "base_oof_balanced_accuracy": base_score,
        "final_oof_balanced_accuracy": final_score,
        "final_oof_delta": final_score - base_score,
        "nested_outer_validation": {
            "mean_delta": nested_mean,
            "min_delta": nested_min,
            "max_delta": float(nested_delta_arr.max()),
            "positive_count": int(np.sum(nested_delta_arr > 0)),
            "total": int(len(nested_delta_arr)),
            "positive_rate": nested_positive_rate,
        },
        "selection_counts": {
            f"{label}@{alpha:.3f}": count
            for (label, alpha), count in sorted(config_counts.items(), key=lambda item: (-item[1], item[0]))
        },
        "bootstrap_final": {
            "ci_2_5": float(bootstrap_ci[0]),
            "median": float(bootstrap_ci[1]),
            "ci_97_5": float(bootstrap_ci[2]),
            "positive_probability": bootstrap_positive,
        },
    }
    (output_dir / "decision.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    colors = ["#167D5A" if value >= 0 else "#C53A3A" for value in nested_delta_arr]
    axes[0].bar(np.arange(1, len(nested_delta_arr) + 1), nested_delta_arr, color=colors)
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].set_title("Nested outer-fold BAC delta")
    axes[0].set_xlabel("Outer fold index")
    axes[0].set_ylabel("Selected blend - base")
    axes[0].xaxis.set_major_locator(MaxNLocator(6, integer=True))
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].hist(bootstrap, bins=45, color="#3C6E9E", alpha=0.9)
    axes[1].axvline(0, color="#222222", linewidth=0.8)
    axes[1].set_title("Final fixed-config bootstrap delta")
    axes[1].set_xlabel("OOF BAC delta")
    axes[1].set_ylabel("Bootstrap count")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].xaxis.set_major_locator(MaxNLocator(5))
    axes[1].grid(axis="y", alpha=0.25)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"nested_classwise_blend.{suffix}", bbox_inches="tight")
    plt.close(fig)

    memo = f"""# Nested class-wise target-stat blend

확장 target-stat 모델은 전체 OOF에서는 basic 모델보다 낮았지만 GALAXY/QSO 경계에 다른 신호가 있었습니다. 전체 OOF에서 가중치를 고르는 낙관 편향을 피하기 위해 10시드 x 5폴드 바깥 검증을 사용했습니다.

- 선택 클래스: {CLASSES[selected_class]}
- 최종 고정 alpha: {selected_alpha:.3f}
- basic OOF: {base_score:.9f}
- 최종 OOF: {final_score:.9f}
- 최종 OOF 변화: {final_score - base_score:+.9f}
- nested valid 평균 변화: {nested_mean:+.9f}
- nested valid 최악 변화: {nested_min:+.9f}
- nested 개선 비율: {int(np.sum(nested_delta_arr > 0))}/{len(nested_delta_arr)}
- 최종 설정 부트스트랩 95% 구간: [{bootstrap_ci[0]:+.9f}, {bootstrap_ci[2]:+.9f}]
- 판정: `{decision}`
"""
    memo_path.write_text(memo, encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"memo: {memo_path}")
    print(f"graph: {output_dir / 'nested_classwise_blend.png'}")


if __name__ == "__main__":
    main()
