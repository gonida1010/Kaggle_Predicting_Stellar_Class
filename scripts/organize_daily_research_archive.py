from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research_private_generalization_20260619"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"


@dataclass(frozen=True)
class CopySpec:
    date: str
    category: str
    src: Path
    dst_name: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a date-based blog/research archive without moving source artifacts."
    )
    parser.add_argument("--archive-dir", type=Path, default=RESEARCH / "daily")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def dated_name(date: str, src: Path, dst_name: str | None = None) -> str:
    name = dst_name or src.name
    if name.startswith(f"{date}_"):
        return name
    return f"{date}_{name}"


def copy_file(spec: CopySpec, archive_dir: Path, dry_run: bool) -> dict[str, str]:
    src = spec.src if spec.src.is_absolute() else ROOT / spec.src
    dst_dir = archive_dir / spec.date / spec.category
    dst = dst_dir / dated_name(spec.date, src, spec.dst_name)
    row = {
        "date": spec.date,
        "category": spec.category,
        "source": str(src.relative_to(ROOT)) if src.exists() else str(src),
        "archive": str(dst.relative_to(ROOT)),
        "status": "missing",
    }
    if not src.exists():
        return row
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    row["status"] = "copied"
    return row


def specs() -> list[CopySpec]:
    out: list[CopySpec] = [
        CopySpec(
            "2026-06-19",
            "notes",
            RESEARCH / "2026-06-19_single_model_import_notes.md",
        ),
        CopySpec(
            "2026-06-20",
            "notes",
            RESEARCH / "2026-06-20_realmlp_feature_bank_feedback.md",
        ),
        CopySpec(
            "2026-06-21",
            "notes",
            RESEARCH / "2026-06-21_shared_notebook_method_audit_and_next_plan.md",
        ),
        CopySpec(
            "2026-06-22",
            "notes",
            RESEARCH / "current_best_methods_and_hyperparams_20260622.md",
            "current_best_methods_and_hyperparams.md",
        ),
        CopySpec(
            "2026-06-22",
            "notes",
            RESEARCH / "algorithm_inventory.md",
            "algorithm_inventory.md",
        ),
        CopySpec(
            "2026-06-23",
            "notes",
            RESEARCH / "current_best_methods_and_hyperparams_20260622.md",
            "current_best_methods_snapshot_after_90.md",
        ),
    ]

    out.extend(
        [
            CopySpec("2026-06-20", "submissions", OUTPUTS / "35_PRIVATE_CV_greedy_cat_realmlp_plus_realmlp0_oof970479.csv"),
            CopySpec("2026-06-20", "submissions", OUTPUTS / "37_PRIVATE_CV_greedy_with_classwise_blender_oof0970528.csv"),
            CopySpec("2026-06-20", "submissions", OUTPUTS / "56_PRIVATE_CV_te_disagree_high_gi_low_rz_base_galaxy_to_star_c0_55_m0_15_b0_60_oof0970573.csv"),
            CopySpec("2026-06-20", "metrics", ARTIFACTS / "te_disagreement_patch_classwise37" / "candidate_summary.csv"),
            CopySpec("2026-06-20", "metrics", ARTIFACTS / "te_disagreement_patch_classwise37" / "report.json"),
            CopySpec("2026-06-20", "figures", ARTIFACTS / "classwise_logistic_blender_c010" / "classwise_model_importance.png"),
            CopySpec("2026-06-20", "figures", ARTIFACTS / "classwise_logistic_blender_c010" / "classwise_recall_vs_reference.png"),
            CopySpec("2026-06-20", "figures", ARTIFACTS / "classwise_logistic_blender_c010" / "classwise_confusion_delta.png"),
        ]
    )

    out.extend(
        [
            CopySpec("2026-06-21", "submissions", OUTPUTS / "64_PRIVATE_CV_catboost_catv3_bac_chunked_direct_oof0968334.csv"),
            CopySpec("2026-06-21", "metrics", ARTIFACTS / "catboost_cv_catv3_bac_chunked" / "catboost_fold_scores.csv"),
            CopySpec("2026-06-21", "metrics", ARTIFACTS / "catboost_cv_catv3_bac_chunked" / "catboost_training_diagnostics.csv"),
            CopySpec("2026-06-21", "figures", ARTIFACTS / "catboost_cv_catv3_bac_chunked" / "catboost_balanced_accuracy_curve.png"),
            CopySpec("2026-06-21", "figures", ARTIFACTS / "catboost_cv_catv3_bac_chunked" / "catboost_logloss_curve.png"),
            CopySpec("2026-06-21", "figures", ARTIFACTS / "catboost_cv_catv3_bac_chunked" / "catboost_logloss_curve_zoom.png"),
            CopySpec("2026-06-21", "metrics", ARTIFACTS / "private_candidate_audit_20260623" / "candidate_audit_summary.csv"),
            CopySpec("2026-06-21", "figures", ARTIFACTS / "private_candidate_audit_20260623" / "candidate_oof_balanced_accuracy.svg"),
            CopySpec("2026-06-21", "figures", ARTIFACTS / "private_candidate_audit_20260623" / "candidate_meta_fold_delta_box.svg"),
        ]
    )

    out.extend(
        [
            CopySpec("2026-06-23", "submissions", OUTPUTS / "68_PRIVATE_CV_research_material_stack_oof0970603.csv"),
            CopySpec("2026-06-23", "submissions", OUTPUTS / "69_PRIVATE_CV_guarded_01_all_changed_rz_0_2_allconf_oof970595.csv"),
            CopySpec("2026-06-23", "submissions", OUTPUTS / "84_PRIVATE_CV_classwise_research_blend_oof0970621.csv"),
            CopySpec("2026-06-23", "submissions", OUTPUTS / "90_PRIVATE_CV_subset_guard_68_plus_84_good_union_oof0970627.csv"),
            CopySpec(
                "2026-06-23",
                "metrics",
                ARTIFACTS / "research_material_stack_20260623" / "accepted_stages.csv",
                "research_material_stack_accepted_stages.csv",
            ),
            CopySpec("2026-06-23", "metrics", ARTIFACTS / "research_material_stack_20260623" / "source_summary_top.csv"),
            CopySpec(
                "2026-06-23",
                "metrics",
                ARTIFACTS / "research_material_stack_20260623" / "report.json",
                "research_material_stack_report.json",
            ),
            CopySpec(
                "2026-06-23",
                "metrics",
                ARTIFACTS / "classwise_research_blend_20260623" / "accepted_stages.csv",
                "classwise_research_blend_accepted_stages.csv",
            ),
            CopySpec(
                "2026-06-23",
                "metrics",
                ARTIFACTS / "classwise_research_blend_20260623" / "report.json",
                "classwise_research_blend_report.json",
            ),
            CopySpec("2026-06-23", "metrics", ARTIFACTS / "classwise_research_blend_84_guard_20260623" / "candidate_summary.csv"),
            CopySpec("2026-06-23", "metrics", ARTIFACTS / "classwise_research_blend_84_guard_20260623" / "output_candidates.csv"),
            CopySpec(
                "2026-06-23",
                "metrics",
                ARTIFACTS / "classwise_research_blend_84_guard_20260623" / "report.json",
                "classwise_research_blend_84_guard_report.json",
            ),
            CopySpec("2026-06-23", "metrics", ARTIFACTS / "private_candidate_audit_20260623" / "candidate_audit_summary.csv"),
            CopySpec("2026-06-23", "metrics", ARTIFACTS / "private_candidate_audit_20260623" / "candidate_class_report.csv"),
            CopySpec("2026-06-23", "metrics", ARTIFACTS / "private_candidate_audit_20260623" / "candidate_meta_fold_deltas.csv"),
            CopySpec("2026-06-23", "metrics", ARTIFACTS / "private_candidate_audit_20260623" / "candidate_subset_deltas.csv"),
            CopySpec("2026-06-23", "figures", ARTIFACTS / "private_candidate_audit_20260623" / "candidate_oof_balanced_accuracy.svg"),
            CopySpec("2026-06-23", "figures", ARTIFACTS / "private_candidate_audit_20260623" / "candidate_meta_fold_delta_box.svg"),
            CopySpec(
                "2026-06-23",
                "figures",
                ARTIFACTS / "oof_diagnostics_classwise_research_blend_84_vs68_20260623" / "class_recall_comparison.svg",
                "84_vs_68_class_recall_comparison.svg",
            ),
            CopySpec(
                "2026-06-23",
                "figures",
                ARTIFACTS / "oof_diagnostics_subset_guard_90_vs84_20260623" / "class_recall_comparison.svg",
                "90_vs_84_class_recall_comparison.svg",
            ),
            CopySpec(
                "2026-06-23",
                "figures",
                ARTIFACTS / "oof_diagnostics_subset_guard_90_vs84_20260623" / "redshift_g_i_oof_accuracy_delta_map.svg",
                "90_vs_84_redshift_g_i_oof_accuracy_delta_map.svg",
            ),
            CopySpec(
                "2026-06-23",
                "figures",
                ARTIFACTS / "oof_diagnostics_subset_guard_90_vs84_20260623" / "subset_delta_bac.svg",
                "90_vs_84_subset_delta_bac.svg",
            ),
        ]
    )
    return out


def write_readme(archive_dir: Path, rows: list[dict[str, str]], dry_run: bool) -> None:
    copied_rows = [row for row in rows if row["status"] == "copied"]
    dates = sorted({row["date"] for row in copied_rows})
    lines = [
        "# Daily Research Archive",
        "",
        "이 폴더는 블로그 정리와 연구 복기를 위해 날짜별로 문서, 수치 CSV, 제출 CSV, 그래프 이미지를 모은 보관본이다.",
        "",
        "원본 artifacts/outputs 파일은 실행 경로 보존을 위해 이동하지 않는다.",
        "",
        "## Dates",
        "",
    ]
    for date in dates:
        count = sum(1 for row in copied_rows if row["date"] == date)
        lines.append(f"- `{date}`: {count} files")
    lines.extend(["", "## Categories", "", "- `notes`: 블로그/연구용 Markdown", "- `metrics`: report, score, candidate summary", "- `submissions`: 제출 CSV", "- `figures`: 그래프와 이미지"])
    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)
        (archive_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for date in dates:
        date_rows = [row for row in copied_rows if row["date"] == date]
        date_lines = [
            f"# {date}",
            "",
            "| category | file | source |",
            "|---|---|---|",
        ]
        for row in sorted(date_rows, key=lambda item: (item["category"], item["archive"])):
            archive_name = Path(row["archive"]).name
            date_lines.append(f"| `{row['category']}` | `{archive_name}` | `{row['source']}` |")
        if not dry_run:
            (archive_dir / date).mkdir(parents=True, exist_ok=True)
            (archive_dir / date / "README.md").write_text("\n".join(date_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    archive_dir = args.archive_dir if args.archive_dir.is_absolute() else ROOT / args.archive_dir
    progress(f"archive_dir={archive_dir}")
    rows = []
    for spec in specs():
        row = copy_file(spec, archive_dir, args.dry_run)
        rows.append(row)
        progress(f"{row['status']}: {row['archive']}")
    write_readme(archive_dir, rows, args.dry_run)
    copied = sum(1 for row in rows if row["status"] == "copied")
    missing = sum(1 for row in rows if row["status"] == "missing")
    progress(f"done copied={copied}, missing={missing}")


if __name__ == "__main__":
    main()
