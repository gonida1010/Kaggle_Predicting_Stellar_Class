from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired CatBoost folds across SDSS sample weights and write research artifacts."
    )
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--memo-path", type=Path, required=True)
    parser.add_argument("--max-worst-fold-drop", type=float, default=0.00015)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_runs(screen_dir: Path) -> list[dict]:
    runs = []
    for report_path in sorted(screen_dir.glob("w*/catboost_baseline_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        external = report.get("external_augmentation", {})
        weight = float(external.get("sample_weight", 0.0))
        fold_scores = report.get("fold_scores", [])
        if not fold_scores:
            continue
        runs.append(
            {
                "report_path": report_path,
                "report": report,
                "weight": weight,
                "fold_scores": fold_scores,
            }
        )
    if not runs:
        raise FileNotFoundError(f"No CatBoost reports found under {screen_dir}")
    return runs


def summarize(
    runs: list[dict],
    max_worst_fold_drop: float,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    baseline_runs = [run for run in runs if np.isclose(run["weight"], 0.0)]
    if len(baseline_runs) != 1:
        raise ValueError("Exactly one weight=0 control run is required.")
    baseline = baseline_runs[0]
    baseline_by_fold = {
        int(row["fold"]): float(row["balanced_accuracy"]) for row in baseline["fold_scores"]
    }

    rows = []
    paired_rows = []
    for run in sorted(runs, key=lambda item: item["weight"]):
        scores = np.asarray([float(row["balanced_accuracy"]) for row in run["fold_scores"]])
        recalls = {
            label: np.asarray([float(row["class_recalls"][label]) for row in run["fold_scores"]])
            for label in ("GALAXY", "QSO", "STAR")
        }
        deltas = []
        for fold_row in run["fold_scores"]:
            fold = int(fold_row["fold"])
            if fold not in baseline_by_fold:
                raise ValueError(f"Control run is missing fold {fold}")
            delta = float(fold_row["balanced_accuracy"]) - baseline_by_fold[fold]
            deltas.append(delta)
            paired_rows.append(
                {
                    "weight": run["weight"],
                    "fold": fold,
                    "balanced_accuracy": float(fold_row["balanced_accuracy"]),
                    "delta_vs_weight0": delta,
                }
            )
        delta_array = np.asarray(deltas)
        required_positive = max(2, math.ceil(0.6 * len(delta_array)))
        robust_score = float(delta_array.mean() - 0.5 * delta_array.std(ddof=0))
        eligible = bool(
            run["weight"] > 0
            and delta_array.mean() > 0
            and int((delta_array > 0).sum()) >= required_positive
            and delta_array.min() >= -max_worst_fold_drop
        )
        rows.append(
            {
                "weight": run["weight"],
                "mean_bac": float(scores.mean()),
                "std_bac": float(scores.std(ddof=0)),
                "worst_fold_bac": float(scores.min()),
                "mean_delta_vs_weight0": float(delta_array.mean()),
                "worst_fold_delta_vs_weight0": float(delta_array.min()),
                "positive_folds": int((delta_array > 0).sum()),
                "fold_count": int(len(delta_array)),
                "robust_score": robust_score,
                "eligible_for_full_cv": eligible,
                "mean_recall_GALAXY": float(recalls["GALAXY"].mean()),
                "mean_recall_QSO": float(recalls["QSO"].mean()),
                "mean_recall_STAR": float(recalls["STAR"].mean()),
                "report_path": display_path(run["report_path"]),
            }
        )

    summary = pd.DataFrame(rows).sort_values("weight").reset_index(drop=True)
    eligible = summary[summary["eligible_for_full_cv"]]
    if eligible.empty:
        selected = summary.loc[summary["weight"].idxmin()]
        decision = "reject_external_weights"
    else:
        selected = eligible.sort_values(
            ["robust_score", "mean_bac"],
            ascending=False,
        ).iloc[0]
        decision = "promote_to_full_cv"

    decision_report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "purpose": "Fold-paired SDSS low-weight screening for the strongest local CatBoost recipe.",
        "decision": decision,
        "selected_weight": float(selected["weight"]),
        "selected_mean_bac": float(selected["mean_bac"]),
        "selected_mean_delta_vs_weight0": float(selected["mean_delta_vs_weight0"]),
        "selected_worst_fold_delta_vs_weight0": float(selected["worst_fold_delta_vs_weight0"]),
        "selection_rule": {
            "mean_delta_positive": True,
            "minimum_positive_folds": "max(2, ceil(60% of folds))",
            "maximum_allowed_worst_fold_drop": float(max_worst_fold_drop),
            "ranking": "mean paired delta - 0.5 * std paired delta",
        },
        "important_limit": (
            "This screen is not final generalization evidence. A promoted weight must pass full 5-fold OOF "
            "and external/generalization diagnostics before entering the stacker."
        ),
        "rows": summary.drop(columns=["report_path"]).to_dict(orient="records"),
    }
    return summary, decision_report, pd.DataFrame(paired_rows)


def write_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=160)
    axes[0].errorbar(
        summary["weight"],
        summary["mean_bac"],
        yerr=summary["std_bac"],
        marker="o",
        linewidth=2,
        capsize=5,
        color="#155eef",
    )
    axes[0].set_title("CatBoost SDSS Weight Screen")
    axes[0].set_xlabel("SDSS sample weight")
    axes[0].set_ylabel("Mean validation balanced accuracy")
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=8))
    axes[0].grid(True, color="#e5e7eb")

    colors = ["#667085" if np.isclose(weight, 0.0) else "#039855" for weight in summary["weight"]]
    axes[1].bar(
        summary["weight"].astype(str),
        summary["mean_delta_vs_weight0"],
        color=colors,
    )
    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[1].set_title("Paired Fold Delta vs Weight 0")
    axes[1].set_xlabel("SDSS sample weight")
    axes[1].set_ylabel("Mean balanced-accuracy delta")
    axes[1].grid(True, axis="y", color="#e5e7eb")

    fig.tight_layout()
    fig.savefig(output_dir / "catboost_sdss_weight_screen.png")
    fig.savefig(output_dir / "catboost_sdss_weight_screen.svg")
    plt.close(fig)


def write_memo(path: Path, summary: pd.DataFrame, decision: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CatBoost + SDSS 저가중치 선별 기록",
        "",
        f"- 생성 시각: {decision['generated_at']}",
        "- 기준 모델: RealMLP식 비누수 피처를 사용한 CatBoost",
        "- 검증 원칙: SDSS는 각 학습 fold에만 추가하고 validation은 competition row만 사용",
        "- 비교 원칙: 같은 seed와 같은 fold의 weight=0 대조군과 paired 비교",
        "",
        "## 결과",
        "",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            "- weight={weight:.3f}: mean BAC={mean_bac:.6f}, delta={mean_delta_vs_weight0:+.6f}, "
            "worst fold delta={worst_fold_delta_vs_weight0:+.6f}, positive folds={positive_folds}/{fold_count}, "
            "full-CV 후보={eligible_for_full_cv}".format(**row)
        )
    lines.extend(
        [
            "",
            "## 판단",
            "",
            f"- 결정: `{decision['decision']}`",
            f"- 선택 weight: `{decision['selected_weight']:.3f}`",
            f"- 대조군 대비 평균 변화: `{decision['selected_mean_delta_vs_weight0']:+.6f}`",
            f"- 최악 fold 변화: `{decision['selected_worst_fold_delta_vs_weight0']:+.6f}`",
            f"- 제한: {decision['important_limit']}",
            "",
            "## 다음 단계",
            "",
            "- promote_to_full_cv이면 같은 설정으로 5-fold 전체 OOF를 재학습한다.",
            "- 5-fold OOF, class recall, SDSS 외부 검증, 기존 stacker 추가 이득을 모두 통과해야 채택한다.",
            "- reject_external_weights이면 이 외부 데이터 혼합은 폐기하고 다음 미실험 항목으로 이동한다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    screen_dir = resolve(args.screen_dir)
    memo_path = resolve(args.memo_path)
    runs = load_runs(screen_dir)
    summary, decision, paired = summarize(runs, float(args.max_worst_fold_drop))

    summary.to_csv(screen_dir / "catboost_sdss_weight_summary.csv", index=False)
    paired.to_csv(screen_dir / "catboost_sdss_paired_fold_deltas.csv", index=False)
    (screen_dir / "catboost_sdss_weight_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (screen_dir / "selected_weight.txt").write_text(
        f"{decision['selected_weight']:.6f}\n",
        encoding="utf-8",
    )
    write_plot(summary, screen_dir)
    write_memo(memo_path, summary, decision)

    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"memo: {memo_path}")


if __name__ == "__main__":
    main()
