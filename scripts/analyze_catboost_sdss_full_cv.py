from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare full-CV CatBoost SDSS augmentation against its local baseline.")
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memo-path", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    baseline_path = resolve(args.baseline_report)
    candidate_path = resolve(args.candidate_report)
    output_dir = resolve(args.output_dir)
    memo_path = resolve(args.memo_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    memo_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline_folds = {int(row["fold"]): row for row in baseline["fold_scores"]}
    candidate_folds = {int(row["fold"]): row for row in candidate["fold_scores"]}
    if set(baseline_folds) != set(candidate_folds) or len(candidate_folds) != 5:
        raise ValueError("Baseline and candidate must contain the same five folds.")

    rows = []
    for fold in sorted(candidate_folds):
        base_row = baseline_folds[fold]
        cand_row = candidate_folds[fold]
        row = {
            "fold": fold,
            "baseline_bac": float(base_row["balanced_accuracy"]),
            "candidate_bac": float(cand_row["balanced_accuracy"]),
            "delta": float(cand_row["balanced_accuracy"]) - float(base_row["balanced_accuracy"]),
            "baseline_iteration": int(base_row["prediction_iteration"]),
            "candidate_iteration": int(cand_row["prediction_iteration"]),
        }
        for label in ("GALAXY", "QSO", "STAR"):
            row[f"baseline_recall_{label}"] = float(base_row["class_recalls"][label])
            row[f"candidate_recall_{label}"] = float(cand_row["class_recalls"][label])
            row[f"recall_delta_{label}"] = (
                float(cand_row["class_recalls"][label]) - float(base_row["class_recalls"][label])
            )
        rows.append(row)

    folds = pd.DataFrame(rows)
    folds.to_csv(output_dir / "catboost_sdss_full_fold_comparison.csv", index=False)
    baseline_score = float(baseline["oof_balanced_accuracy"])
    candidate_score = float(candidate["oof_balanced_accuracy"])
    positive_folds = int((folds["delta"] > 0).sum())
    worst_fold_delta = float(folds["delta"].min())
    worst_recall_delta = float(
        folds[[f"recall_delta_{label}" for label in ("GALAXY", "QSO", "STAR")]].mean().min()
    )
    accepted = bool(
        candidate_score > baseline_score
        and positive_folds >= 4
        and worst_fold_delta >= -0.00015
        and worst_recall_delta >= -0.00020
    )
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "baseline_report": str(args.baseline_report),
        "candidate_report": str(args.candidate_report),
        "baseline_oof_bac": baseline_score,
        "candidate_oof_bac": candidate_score,
        "oof_delta": candidate_score - baseline_score,
        "positive_folds": positive_folds,
        "worst_fold_delta": worst_fold_delta,
        "mean_fold_delta": float(folds["delta"].mean()),
        "worst_mean_class_recall_delta": worst_recall_delta,
        "accepted_for_stacker_screen": accepted,
        "decision": "promote_to_stacker_screen" if accepted else "reject_or_research",
        "acceptance_rule": {
            "candidate_oof_above_baseline": True,
            "minimum_positive_folds": 4,
            "minimum_worst_fold_delta": -0.00015,
            "minimum_worst_mean_class_recall_delta": -0.00020,
        },
    }
    (output_dir / "catboost_sdss_full_decision.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
        colors = ["#039855" if value >= 0 else "#d92d20" for value in folds["delta"]]
        ax.bar(folds["fold"].astype(str), folds["delta"], color=colors)
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_title("CatBoost + SDSS 0.03: Paired Full-CV Delta")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Balanced-accuracy delta vs local CatBoost baseline")
        ax.grid(True, axis="y", color="#e5e7eb")
        fig.tight_layout()
        fig.savefig(output_dir / "catboost_sdss_full_fold_delta.png")
        fig.savefig(output_dir / "catboost_sdss_full_fold_delta.svg")
        plt.close(fig)
    except Exception:
        pass

    memo = [
        "# CatBoost + SDSS 0.03 전체 5-fold 결과",
        "",
        f"- 생성 시각: {report['generated_at']}",
        f"- 기존 CatBoost OOF BAC: {baseline_score:.9f}",
        f"- SDSS 0.03 CatBoost OOF BAC: {candidate_score:.9f}",
        f"- OOF 변화: {candidate_score - baseline_score:+.9f}",
        f"- 개선 fold: {positive_folds}/5",
        f"- 최악 fold 변화: {worst_fold_delta:+.9f}",
        f"- 최악 class 평균 recall 변화: {worst_recall_delta:+.9f}",
        f"- 판단: `{report['decision']}`",
        "",
        "## fold별 변화",
        "",
    ]
    for row in rows:
        memo.append(
            f"- fold {row['fold']}: {row['baseline_bac']:.6f} -> "
            f"{row['candidate_bac']:.6f} ({row['delta']:+.6f})"
        )
    memo.extend(
        [
            "",
            "## 다음 단계",
            "",
            "- promote_to_stacker_screen이면 이 모델의 OOF/test probability를 기존 private stacker source bank에 추가한다.",
            "- 추가 후 전체 OOF뿐 아니라 반복 meta-fold, class recall, 취약 subset의 최악 변화까지 다시 검사한다.",
            "- reject_or_research이면 외부 데이터 혼합을 최종 후보에 사용하지 않는다.",
            "",
        ]
    )
    memo_path.write_text("\n".join(memo), encoding="utf-8")
    print(folds.to_string(index=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"memo: {memo_path}")


if __name__ == "__main__":
    main()
