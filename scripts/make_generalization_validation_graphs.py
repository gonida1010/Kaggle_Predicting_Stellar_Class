from __future__ import annotations

import argparse
import ast
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
EXTERNAL_EXP = ARTIFACTS / "external_feature_experiment"
BOUNDARY_EXP = ARTIFACTS / "oof_proba_boundary_analysis"
OUT_DIR = ARTIFACTS / "generalization_validation_graphs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create validation-focused SVG graphs from external shift and OOF boundary diagnostics."
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--archive-zip",
        type=Path,
        default=Path("/Users/parkyeonggon/Downloads/archive (3).zip"),
        help="Optional public-feedback archive for flip-history diagnostics.",
    )
    parser.add_argument("--top-n", type=int, default=18)
    return parser.parse_args()


def esc(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def short_label(value: object, max_len: int = 34) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def write_bar_svg(
    df: pd.DataFrame,
    path: Path,
    title: str,
    label_col: str,
    value_col: str,
    subtitle: str = "",
    color: str = "#2f80ed",
    value_fmt: str = "{:.3f}",
) -> None:
    view = df.copy()
    width = 1080
    row_h = 28
    top = 74
    left = 300
    bar_w = 620
    height = top + row_h * len(view) + 54
    max_value = max(float(view[value_col].max()), 1e-12)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" font-family="Arial" font-size="20" font-weight="700" fill="#111827">{esc(title)}</text>',
    ]
    if subtitle:
        lines.append(
            f'<text x="24" y="54" font-family="Arial" font-size="12" fill="#667085">{esc(subtitle)}</text>'
        )
    for idx, row in view.reset_index(drop=True).iterrows():
        y = top + idx * row_h
        value = float(row[value_col])
        width_px = max(1.0, value / max_value * bar_w)
        lines.extend(
            [
                f'<text x="24" y="{y + 18}" font-family="Arial" font-size="12" fill="#344054">{esc(short_label(row[label_col]))}</text>',
                f'<rect x="{left}" y="{y + 4}" width="{bar_w}" height="16" rx="2" fill="#edf2f7"/>',
                f'<rect x="{left}" y="{y + 4}" width="{width_px:.1f}" height="16" rx="2" fill="{color}"/>',
                f'<text x="{left + bar_w + 14}" y="{y + 18}" font-family="Arial" font-size="12" fill="#111827">{esc(value_fmt.format(value))}</text>',
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_grouped_class_svg(df: pd.DataFrame, path: Path) -> None:
    classes = ["GALAXY", "QSO", "STAR"]
    datasets = df["dataset"].drop_duplicates().tolist()
    width = 860
    height = 360
    left = 120
    top = 70
    group_w = 210
    bar_w = 44
    scale_h = 210
    colors = {"GALAXY": "#2f80ed", "QSO": "#27ae60", "STAR": "#eb5757"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-family="Arial" font-size="20" font-weight="700" fill="#111827">Class Distribution Check</text>',
        '<text x="24" y="54" font-family="Arial" font-size="12" fill="#667085">External class prior differs from competition train; direct concat can bias STAR/GALAXY boundary.</text>',
    ]
    for d_idx, dataset in enumerate(datasets):
        x0 = left + d_idx * group_w
        sub = df[df["dataset"] == dataset].set_index("class")
        lines.append(
            f'<text x="{x0 + 58}" y="{top + scale_h + 34}" text-anchor="middle" font-family="Arial" font-size="12" fill="#344054">{esc(dataset)}</text>'
        )
        for c_idx, cls in enumerate(classes):
            share = float(sub.loc[cls, "share"]) if cls in sub.index else 0.0
            h = share * scale_h
            x = x0 + c_idx * (bar_w + 12)
            y = top + scale_h - h
            lines.extend(
                [
                    f'<rect x="{x}" y="{top}" width="{bar_w}" height="{scale_h}" rx="2" fill="#f2f4f7"/>',
                    f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="2" fill="{colors[cls]}"/>',
                    f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#111827">{share:.1%}</text>',
                    f'<text x="{x + bar_w / 2:.1f}" y="{top + scale_h + 16}" text-anchor="middle" font-family="Arial" font-size="10" fill="#667085">{cls}</text>',
                ]
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_transition_segment_svg(df: pd.DataFrame, path: Path, top_n: int) -> None:
    view = df[df["comparison"].eq("current_vs_pure_pred")].copy()
    view["label"] = view["group_col"] + "=" + view["group_value"] + " / " + view["transition"]
    view = view.sort_values("count", ascending=False).head(top_n)
    write_bar_svg(
        view,
        path,
        "Current Public Candidate vs Pure Model: Disagreement Segments",
        "label",
        "count",
        "Counts where current public candidate and pure generalization model disagree.",
        color="#9b51e0",
        value_fmt="{:.0f}",
    )


def load_archive_history(archive_zip: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not archive_zip.exists():
        return None, None
    with zipfile.ZipFile(archive_zip) as zf:
        history = pd.read_csv(zf.open("submission_history.csv"))
        summary = json.loads(zf.read("score_summary.json").decode("utf-8"))
    rows = []
    for row in history.itertuples(index=False):
        try:
            actions = ast.literal_eval(row.actions)
        except Exception:
            actions = []
        for action in actions:
            rows.append(
                {
                    "score": float(row.score),
                    "keep": bool(row.keep),
                    "batch": row.batch,
                    "transition": f"{action.get('from')}->{action.get('to')}",
                    "id": int(action.get("id")),
                }
            )
    actions_df = pd.DataFrame(rows)
    history.attrs["summary"] = summary
    return history, actions_df


def write_public_history_svg(history: pd.DataFrame, path: Path) -> None:
    view = history.reset_index(drop=True).copy()
    width = 1080
    height = 380
    left = 70
    right = 30
    top = 58
    bottom = 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    scores = view["score"].astype(float).to_numpy()
    min_s = min(scores.min(), 0.9719)
    max_s = max(scores.max(), 0.9721)
    denom = max(max_s - min_s, 1e-6)
    points = []
    for idx, score in enumerate(scores):
        x = left + idx / max(1, len(scores) - 1) * plot_w
        y = top + (max_s - score) / denom * plot_h
        points.append((x, y, score, bool(view.at[idx, "keep"])))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-family="Arial" font-size="20" font-weight="700" fill="#111827">Archive Public-Feedback Walk-Forward Scores</text>',
        '<text x="24" y="50" font-family="Arial" font-size="12" fill="#667085">This is public-feedback behavior, not a private/generalization validation curve.</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#d0d5dd"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#d0d5dd"/>',
    ]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in points)
    lines.append(f'<polyline points="{poly}" fill="none" stroke="#2f80ed" stroke-width="2"/>')
    for x, y, score, keep in points:
        color = "#27ae60" if keep else "#d92d20"
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    for tick in np.linspace(min_s, max_s, 5):
        y = top + (max_s - tick) / denom * plot_h
        lines.extend(
            [
                f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#98a2b3"/>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#667085">{tick:.5f}</text>',
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class_dist = pd.read_csv(EXTERNAL_EXP / "class_distribution.csv")
    feature_shift = pd.read_csv(EXTERNAL_EXP / "feature_shift.csv")
    subset = pd.read_csv(BOUNDARY_EXP / "oof_subset_metrics.csv")
    segments = pd.read_csv(BOUNDARY_EXP / "test_disagreement_segments.csv")
    external_report = json.loads((EXTERNAL_EXP / "report.json").read_text(encoding="utf-8"))
    boundary_report = json.loads((BOUNDARY_EXP / "report.json").read_text(encoding="utf-8"))

    write_grouped_class_svg(class_dist, args.output_dir / "class_distribution_validation.svg")

    ext_shift = feature_shift[
        feature_shift["left"].eq("competition_train") & feature_shift["right"].eq("external")
    ].sort_values("ks_stat", ascending=False).head(args.top_n)
    write_bar_svg(
        ext_shift,
        args.output_dir / "external_feature_shift_top_ks.svg",
        "External vs Competition Train: Top KS Shift",
        "feature",
        "ks_stat",
        f"Domain AUC={external_report['domain_screen']['domain_auc']:.3f}; high KS means external concat risk.",
        color="#d92d20",
    )

    test_shift = feature_shift[
        feature_shift["left"].eq("competition_train") & feature_shift["right"].eq("competition_test")
    ].sort_values("ks_stat", ascending=False).head(args.top_n)
    write_bar_svg(
        test_shift,
        args.output_dir / "train_test_feature_shift_top_ks.svg",
        "Competition Train vs Test: Top KS Shift",
        "feature",
        "ks_stat",
        "Very low KS supports using OOF validation as a reliable generalization proxy.",
        color="#27ae60",
    )

    subset_view = subset.sort_values("error_rate", ascending=False).head(args.top_n).copy()
    subset_view["label"] = subset_view["group_col"] + "=" + subset_view["group_value"] + " / " + subset_view["top_error_pair"]
    write_bar_svg(
        subset_view,
        args.output_dir / "oof_worst_subset_error_rate.svg",
        "OOF Worst Subsets by Error Rate",
        "label",
        "error_rate",
        "Validation graph from OOF predictions; not public leaderboard feedback.",
        color="#f2994a",
        value_fmt="{:.2%}",
    )

    write_transition_segment_svg(
        segments,
        args.output_dir / "current_vs_pure_disagreement_segments.svg",
        args.top_n,
    )

    history, actions = load_archive_history(args.archive_zip)
    archive_outputs = []
    if history is not None and actions is not None:
        write_public_history_svg(history, args.output_dir / "archive3_public_feedback_history.svg")
        action_counts = actions.groupby("transition", observed=True).size().reset_index(name="count").sort_values("count", ascending=False)
        write_bar_svg(
            action_counts.head(args.top_n),
            args.output_dir / "archive3_flip_transition_counts.svg",
            "Archive (3) Flip Actions by Transition",
            "transition",
            "count",
            "Public-feedback archive: useful as boundary signal, not generalization proof.",
            color="#2f80ed",
            value_fmt="{:.0f}",
        )
        archive_outputs = [
            "archive3_public_feedback_history.svg",
            "archive3_flip_transition_counts.svg",
        ]

    report = {
        "purpose": "Validation-focused graphs from OOF/subset/external-shift diagnostics.",
        "source_files": {
            "class_distribution": str(EXTERNAL_EXP / "class_distribution.csv"),
            "feature_shift": str(EXTERNAL_EXP / "feature_shift.csv"),
            "external_report": str(EXTERNAL_EXP / "report.json"),
            "oof_subset_metrics": str(BOUNDARY_EXP / "oof_subset_metrics.csv"),
            "test_disagreement_segments": str(BOUNDARY_EXP / "test_disagreement_segments.csv"),
            "boundary_report": str(BOUNDARY_EXP / "report.json"),
            "archive_zip": str(args.archive_zip) if args.archive_zip.exists() else None,
        },
        "validated_facts": {
            "external_domain_auc": external_report["domain_screen"]["domain_auc"],
            "current_vs_pure_rows": boundary_report["rows"]["current_vs_pure"],
            "current_vs_reference_rows": boundary_report["rows"]["current_vs_reference"],
            "current_vs_bank_consensus_rows": boundary_report["rows"]["current_vs_bank_consensus"],
        },
        "outputs": [
            "class_distribution_validation.svg",
            "external_feature_shift_top_ks.svg",
            "train_test_feature_shift_top_ks.svg",
            "oof_worst_subset_error_rate.svg",
            "current_vs_pure_disagreement_segments.svg",
            *archive_outputs,
            "graph_report.json",
        ],
    }
    (args.output_dir / "graph_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
