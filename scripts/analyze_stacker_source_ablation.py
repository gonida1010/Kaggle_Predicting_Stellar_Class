from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score


CLASSES = ["GALAXY", "QSO", "STAR"]
TARGET_MAP = {label: idx for idx, label in enumerate(CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare otherwise identical stacker runs with and without one OOF source."
    )
    parser.add_argument("--without-dir", type=Path, required=True)
    parser.add_argument("--with-dir", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memo-path", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260630)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_report(run_dir: Path) -> dict:
    return json.loads((run_dir / "report.json").read_text(encoding="utf-8"))


def class_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    recalls = recall_score(y, pred, labels=[0, 1, 2], average=None)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "class_recalls": {name: float(value) for name, value in zip(CLASSES, recalls)},
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def stratified_bootstrap_deltas(
    y: np.ndarray,
    without_pred: np.ndarray,
    with_pred: np.ndarray,
    repeats: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y == class_id) for class_id in range(3)]
    deltas = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        recalls_without = []
        recalls_with = []
        for class_id, indices in enumerate(class_indices):
            sample = rng.choice(indices, size=len(indices), replace=True)
            recalls_without.append(np.mean(without_pred[sample] == class_id))
            recalls_with.append(np.mean(with_pred[sample] == class_id))
        deltas[repeat] = np.mean(recalls_with) - np.mean(recalls_without)
    return deltas


def render_graphs(
    output_dir: Path,
    fold_compare: pd.DataFrame,
    bootstrap_deltas: np.ndarray,
    recall_delta: dict[str, float],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 140,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)

    colors = np.where(fold_compare["delta"] >= 0, "#167D5A", "#C53A3A")
    axes[0].bar(np.arange(1, len(fold_compare) + 1), fold_compare["delta"], color=colors, width=0.78)
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].set_title("Paired 5-seed x 5-fold BAC delta")
    axes[0].set_xlabel("Paired fold index")
    axes[0].set_ylabel("With source - without source")
    axes[0].xaxis.set_major_locator(MaxNLocator(6, integer=True))
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].hist(bootstrap_deltas, bins=45, color="#3C6E9E", alpha=0.9)
    axes[1].axvline(0, color="#222222", linewidth=0.9)
    axes[1].axvline(np.median(bootstrap_deltas), color="#B24A1B", linewidth=1.3)
    axes[1].set_title("Stratified paired bootstrap delta")
    axes[1].set_xlabel("OOF balanced accuracy delta")
    axes[1].set_ylabel("Bootstrap count")
    axes[1].xaxis.set_major_locator(MaxNLocator(5))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].grid(axis="y", alpha=0.25)

    names = list(recall_delta)
    values = [recall_delta[name] for name in names]
    recall_colors = ["#167D5A" if value >= 0 else "#C53A3A" for value in values]
    axes[2].bar(names, values, color=recall_colors)
    axes[2].axhline(0, color="#222222", linewidth=0.8)
    axes[2].set_title("Class recall delta")
    axes[2].set_xlabel("Class")
    axes[2].set_ylabel("With source - without source")
    axes[2].grid(axis="y", alpha=0.25)

    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"stacker_source_ablation.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    without_dir = resolve(args.without_dir)
    with_dir = resolve(args.with_dir)
    output_dir = resolve(args.output_dir)
    memo_path = resolve(args.memo_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    memo_path.parent.mkdir(parents=True, exist_ok=True)

    without_report = load_report(without_dir)
    with_report = load_report(with_dir)
    without_models = without_report["models"]
    with_models = with_report["models"]
    expected_models = without_models + [args.source_name]
    if set(with_models) != set(expected_models) or len(with_models) != len(expected_models):
        raise RuntimeError(
            "Ablation input mismatch. Runs must differ by exactly the requested source.\n"
            f"without={without_models}\nwith={with_models}\nsource={args.source_name}"
        )

    without_fold = pd.read_csv(without_dir / "fold_scores.csv")
    with_fold = pd.read_csv(with_dir / "fold_scores.csv")
    fold_compare = without_fold.merge(
        with_fold,
        on=["seed", "fold"],
        how="inner",
        suffixes=("_without", "_with"),
        validate="one_to_one",
    )
    if len(fold_compare) != len(without_fold) or len(fold_compare) != len(with_fold):
        raise RuntimeError("Fold rows do not align one-to-one between ablation runs.")
    fold_compare["delta"] = (
        fold_compare["balanced_accuracy_with"] - fold_compare["balanced_accuracy_without"]
    )
    fold_compare.to_csv(output_dir / "paired_fold_deltas.csv", index=False)

    train = pd.read_csv(ROOT / "data" / "train.csv")
    y = train["class"].map(TARGET_MAP).to_numpy()
    without_oof = np.load(without_dir / "available_prediction_stacker_oof.npy")
    with_oof = np.load(with_dir / "available_prediction_stacker_oof.npy")
    without_metrics = class_metrics(y, without_oof)
    with_metrics = class_metrics(y, with_oof)
    without_pred = without_oof.argmax(axis=1)
    with_pred = with_oof.argmax(axis=1)
    bootstrap_deltas = stratified_bootstrap_deltas(
        y,
        without_pred,
        with_pred,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    np.save(output_dir / "bootstrap_oof_deltas.npy", bootstrap_deltas)

    recall_delta = {
        name: with_metrics["class_recalls"][name] - without_metrics["class_recalls"][name]
        for name in CLASSES
    }
    overall_delta = with_metrics["balanced_accuracy"] - without_metrics["balanced_accuracy"]
    positive_fold_rate = float(np.mean(fold_compare["delta"] > 0))
    bootstrap_positive_probability = float(np.mean(bootstrap_deltas > 0))
    bootstrap_ci = np.quantile(bootstrap_deltas, [0.025, 0.5, 0.975])
    changed = without_pred != with_pred
    without_correct = without_pred == y
    with_correct = with_pred == y
    newly_correct = int(np.sum(changed & ~without_correct & with_correct))
    newly_wrong = int(np.sum(changed & without_correct & ~with_correct))

    importance = pd.read_csv(with_dir / "model_importance.csv")
    source_rows = importance.loc[importance["model"] == args.source_name]
    source_importance = source_rows.iloc[0].to_dict() if len(source_rows) else None
    source_rank = (
        int(importance.index[importance["model"] == args.source_name][0]) + 1
        if len(source_rows)
        else None
    )

    if bootstrap_ci[0] > 0 and positive_fold_rate >= 0.60:
        decision = "strong_promote_as_stacker_source"
        reason = "OOF improved and the paired bootstrap 95% interval is entirely positive."
    elif overall_delta > 0 and bootstrap_positive_probability >= 0.90 and positive_fold_rate >= 0.60:
        decision = "promote_as_stacker_source"
        reason = (
            "OOF and most paired folds improved, but the bootstrap 95% interval slightly crosses zero. "
            "Keep it as an ensemble source rather than a standalone final candidate."
        )
    elif overall_delta > 0:
        decision = "research_only"
        reason = "OOF increased, but paired stability evidence is not yet strong enough for final promotion."
    else:
        decision = "reject_from_stacker"
        reason = "Adding the source did not improve the identical-condition OOF stacker."

    result = {
        "source_name": args.source_name,
        "decision": decision,
        "decision_reason": reason,
        "model_count_without": len(without_models),
        "model_count_with": len(with_models),
        "models_without": without_models,
        "models_with": with_models,
        "without": without_metrics,
        "with": with_metrics,
        "overall_oof_delta": float(overall_delta),
        "paired_fold_delta": {
            "mean": float(fold_compare["delta"].mean()),
            "median": float(fold_compare["delta"].median()),
            "min": float(fold_compare["delta"].min()),
            "max": float(fold_compare["delta"].max()),
            "positive_count": int(np.sum(fold_compare["delta"] > 0)),
            "total": int(len(fold_compare)),
            "positive_rate": positive_fold_rate,
        },
        "bootstrap_oof_delta": {
            "repeats": int(args.bootstrap_repeats),
            "mean": float(np.mean(bootstrap_deltas)),
            "ci_2_5": float(bootstrap_ci[0]),
            "median": float(bootstrap_ci[1]),
            "ci_97_5": float(bootstrap_ci[2]),
            "positive_probability": bootstrap_positive_probability,
        },
        "class_recall_delta": recall_delta,
        "prediction_change": {
            "changed_rows": int(np.sum(changed)),
            "newly_correct": newly_correct,
            "newly_wrong": newly_wrong,
            "net_correct": newly_correct - newly_wrong,
        },
        "source_importance_rank": source_rank,
        "source_importance": source_importance,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    render_graphs(output_dir, fold_compare, bootstrap_deltas, recall_delta)

    memo = f"""# CatBoost SDSS 0.03 stacker source ablation

검증 목적은 새 `CatBoost + SDSS17 weight 0.03` 모델이 단순 단일 모델 개선을 넘어 기존 OOF 스태커에 독립적인 신호를 추가하는지 확인하는 것이었습니다.

- 제외 스태커 OOF balanced accuracy: {without_metrics["balanced_accuracy"]:.9f}
- 포함 스태커 OOF balanced accuracy: {with_metrics["balanced_accuracy"]:.9f}
- OOF 변화: {overall_delta:+.9f}
- 동일 seed/fold 개선: {int(np.sum(fold_compare["delta"] > 0))}/{len(fold_compare)}
- 폴드 변화 평균: {fold_compare["delta"].mean():+.9f}
- 폴드 변화 최솟값: {fold_compare["delta"].min():+.9f}
- 부트스트랩 95% 구간: [{bootstrap_ci[0]:+.9f}, {bootstrap_ci[2]:+.9f}]
- 부트스트랩 개선 확률: {bootstrap_positive_probability:.3%}
- 예측 변경 행: {int(np.sum(changed)):,}
- 새로 정답이 된 행 / 오답이 된 행: {newly_correct:,} / {newly_wrong:,}
- 클래스 재현율 변화: GALAXY {recall_delta["GALAXY"]:+.9f}, QSO {recall_delta["QSO"]:+.9f}, STAR {recall_delta["STAR"]:+.9f}
- 새 소스 중요도 순위: {source_rank if source_rank is not None else "확인 불가"} / {len(with_models)}
- 판정: `{decision}`

판정 이유: {reason}

그래프는 `stacker_source_ablation.png`에 저장했습니다. 이 비교는 두 실행의 입력 모델, 폴드, 시드를 고정하고 새 소스 하나만 추가한 인과적 ablation입니다.
"""
    memo_path.write_text(memo, encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"memo: {memo_path}")
    print(f"graph: {output_dir / 'stacker_source_ablation.png'}")


if __name__ == "__main__":
    main()
