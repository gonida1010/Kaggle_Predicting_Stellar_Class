from __future__ import annotations

import argparse
import json
import os
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score


CLASSES = ["GALAXY", "QSO", "STAR"]
TARGET_MAP = {label: idx for idx, label in enumerate(CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired analysis of basic and extended fold-safe target statistics.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memo-path", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260630)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    recalls = recall_score(y, pred, labels=[0, 1, 2], average=None)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "class_recalls": dict(zip(CLASSES, map(float, recalls))),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def bootstrap_delta(
    y: np.ndarray,
    baseline_pred: np.ndarray,
    candidate_pred: np.ndarray,
    repeats: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    class_rows = [np.flatnonzero(y == class_idx) for class_idx in range(3)]
    result = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        baseline_recalls = []
        candidate_recalls = []
        for class_idx, rows in enumerate(class_rows):
            sampled = rng.choice(rows, size=len(rows), replace=True)
            baseline_recalls.append(np.mean(baseline_pred[sampled] == class_idx))
            candidate_recalls.append(np.mean(candidate_pred[sampled] == class_idx))
        result[repeat] = np.mean(candidate_recalls) - np.mean(baseline_recalls)
    return result


def render_graph(
    output_dir: Path,
    fold_compare: pd.DataFrame,
    recall_delta: dict[str, float],
    bootstrap: np.ndarray,
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 140})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)

    colors = ["#167D5A" if value >= 0 else "#C53A3A" for value in fold_compare["delta"]]
    axes[0].bar(fold_compare["fold"], fold_compare["delta"], color=colors)
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].set_title("Paired fold BAC delta")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Extended - basic")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].grid(axis="y", alpha=0.25)

    recall_values = [recall_delta[label] for label in CLASSES]
    recall_colors = ["#167D5A" if value >= 0 else "#C53A3A" for value in recall_values]
    axes[1].bar(CLASSES, recall_values, color=recall_colors)
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].set_title("Class recall delta")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Extended - basic")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].hist(bootstrap, bins=45, color="#3C6E9E", alpha=0.9)
    axes[2].axvline(0, color="#222222", linewidth=0.9)
    axes[2].axvline(np.median(bootstrap), color="#B24A1B", linewidth=1.3)
    axes[2].set_title("Stratified paired bootstrap")
    axes[2].set_xlabel("OOF BAC delta")
    axes[2].set_ylabel("Bootstrap count")
    axes[2].xaxis.set_major_locator(MaxNLocator(5))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    axes[2].xaxis.set_major_formatter(formatter)
    axes[2].grid(axis="y", alpha=0.25)

    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"extended_target_stats_ablation.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline_dir = resolve(args.baseline_dir)
    candidate_dir = resolve(args.candidate_dir)
    output_dir = resolve(args.output_dir)
    memo_path = resolve(args.memo_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    memo_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_report = json.loads((baseline_dir / "lgbm_te_report.json").read_text(encoding="utf-8"))
    candidate_report = json.loads((candidate_dir / "lgbm_te_report.json").read_text(encoding="utf-8"))
    baseline_fold = pd.read_csv(baseline_dir / "lgbm_te_fold_scores.csv")
    candidate_fold = pd.read_csv(candidate_dir / "lgbm_te_fold_scores.csv")
    fold_compare = baseline_fold[["fold", "balanced_accuracy"]].merge(
        candidate_fold[["fold", "balanced_accuracy"]],
        on="fold",
        suffixes=("_basic", "_extended"),
        validate="one_to_one",
    )
    fold_compare["delta"] = (
        fold_compare["balanced_accuracy_extended"] - fold_compare["balanced_accuracy_basic"]
    )
    fold_compare.to_csv(output_dir / "paired_fold_deltas.csv", index=False)

    train = pd.read_csv(ROOT / "data" / "train.csv")
    y = train["class"].map(TARGET_MAP).to_numpy()
    baseline_oof = np.load(baseline_dir / "lgbm_te_oof_proba.npy")
    candidate_oof = np.load(candidate_dir / "lgbm_te_oof_proba.npy")
    baseline_metrics = metrics(y, baseline_oof)
    candidate_metrics = metrics(y, candidate_oof)
    baseline_pred = baseline_oof.argmax(axis=1)
    candidate_pred = candidate_oof.argmax(axis=1)
    bootstrap = bootstrap_delta(
        y,
        baseline_pred,
        candidate_pred,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    np.save(output_dir / "bootstrap_oof_deltas.npy", bootstrap)

    recall_delta = {
        label: candidate_metrics["class_recalls"][label] - baseline_metrics["class_recalls"][label]
        for label in CLASSES
    }
    overall_delta = candidate_metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
    ci = np.quantile(bootstrap, [0.025, 0.5, 0.975])
    changed = baseline_pred != candidate_pred
    baseline_correct = baseline_pred == y
    candidate_correct = candidate_pred == y
    newly_correct = int(np.sum(changed & ~baseline_correct & candidate_correct))
    newly_wrong = int(np.sum(changed & baseline_correct & ~candidate_correct))
    positive_folds = int(np.sum(fold_compare["delta"] > 0))
    positive_probability = float(np.mean(bootstrap > 0))

    if overall_delta > 0 and positive_folds >= 4 and ci[0] > 0:
        decision = "strong_promote_as_source"
    elif overall_delta > 0 and positive_folds >= 3 and positive_probability >= 0.80:
        decision = "promote_as_research_source"
    elif overall_delta > 0:
        decision = "research_only"
    else:
        decision = "reject"

    result = {
        "decision": decision,
        "baseline_config": {
            "target_stat_mode": baseline_report.get("te_stat_mode", "basic"),
            "interaction_mode": baseline_report.get("te_interaction_mode", "base"),
            "feature_count": baseline_report["target_encoding_feature_count"],
        },
        "candidate_config": {
            "target_stat_mode": candidate_report.get("te_stat_mode"),
            "interaction_mode": candidate_report.get("te_interaction_mode"),
            "feature_count": candidate_report["target_encoding_feature_count"],
            "te_min_count": candidate_report.get("te_min_count"),
            "te_smoothing": candidate_report["te_smoothing"],
        },
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "overall_oof_delta": float(overall_delta),
        "paired_folds": {
            "positive_count": positive_folds,
            "total": int(len(fold_compare)),
            "mean_delta": float(fold_compare["delta"].mean()),
            "min_delta": float(fold_compare["delta"].min()),
            "max_delta": float(fold_compare["delta"].max()),
        },
        "bootstrap": {
            "repeats": int(args.bootstrap_repeats),
            "mean": float(np.mean(bootstrap)),
            "ci_2_5": float(ci[0]),
            "median": float(ci[1]),
            "ci_97_5": float(ci[2]),
            "positive_probability": positive_probability,
        },
        "class_recall_delta": recall_delta,
        "prediction_change": {
            "changed_rows": int(np.sum(changed)),
            "newly_correct": newly_correct,
            "newly_wrong": newly_wrong,
            "net_correct": newly_correct - newly_wrong,
        },
    }
    (output_dir / "decision.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    render_graph(output_dir, fold_compare, recall_delta, bootstrap)

    memo = f"""# 확장 fold-safe target-stat 검증

기존 basic target encoding과 동일한 LightGBM 설정, 동일한 5개 fold를 사용했습니다. 차이는 fold train에서 만드는 통계 피처뿐입니다.

- basic target-stat 피처: {baseline_report["target_encoding_feature_count"]}개
- extended target-stat 피처: {candidate_report["target_encoding_feature_count"]}개
- basic OOF: {baseline_metrics["balanced_accuracy"]:.9f}
- extended OOF: {candidate_metrics["balanced_accuracy"]:.9f}
- OOF 변화: {overall_delta:+.9f}
- 개선 fold: {positive_folds}/{len(fold_compare)}
- 최악 fold 변화: {fold_compare["delta"].min():+.9f}
- 부트스트랩 95% 구간: [{ci[0]:+.9f}, {ci[2]:+.9f}]
- 부트스트랩 개선 확률: {positive_probability:.3%}
- 클래스 재현율 변화: GALAXY {recall_delta["GALAXY"]:+.9f}, QSO {recall_delta["QSO"]:+.9f}, STAR {recall_delta["STAR"]:+.9f}
- 변경 예측: {int(np.sum(changed)):,}행
- 새로 정답 / 새로 오답: {newly_correct:,} / {newly_wrong:,}
- 판정: `{decision}`

누수 방지 조건:

- 모든 통계는 각 fold의 train 행과 label만 사용했습니다.
- validation과 test는 해당 fold train에서 만든 mapping으로 transform했습니다.
- unseen 또는 최소 빈도 미만 범주는 fold train의 class prior로 대체했습니다.
"""
    memo_path.write_text(memo, encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"memo: {memo_path}")
    print(f"graph: {output_dir / 'extended_target_stats_ablation.png'}")


if __name__ == "__main__":
    main()
