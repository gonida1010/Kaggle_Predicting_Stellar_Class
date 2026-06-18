from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "artifacts" / "boundary_pair_experiments"
DEFAULT_OUTPUT = DEFAULT_ROOT / "publication_graphs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create publication-quality SVG graphs for boundary experiments.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nice_ticks(min_value: float, max_value: float, count: int = 6) -> list[float]:
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return []
    if min_value == max_value:
        return [min_value]
    raw_step = (max_value - min_value) / max(1, count - 1)
    exponent = np.floor(np.log10(abs(raw_step)))
    fraction = raw_step / (10**exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    step = nice_fraction * (10**exponent)
    start = np.floor(min_value / step) * step
    end = np.ceil(max_value / step) * step
    ticks = []
    value = start
    while value <= end + step * 0.5:
        ticks.append(float(value))
        value += step
    return ticks


def linear_ticks(min_value: float, max_value: float, count: int = 7) -> list[float]:
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return []
    if min_value == max_value:
        return [min_value]
    return [float(value) for value in np.linspace(min_value, max_value, count)]


def format_short(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def read_reports(root: Path) -> pd.DataFrame:
    rows = []
    for report_path in sorted(root.glob("*/boundary_pair_calibrator_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        row = {
            "run": report_path.parent.name,
            "accepted": bool(report.get("accepted_as_candidate")),
            "combined_delta": float(report.get("combined_delta", np.nan)),
            "combined_oof": float(report.get("combined_eval_oof_balanced_accuracy", np.nan)),
            "base_oof": float(report.get("base_eval_oof_balanced_accuracy", np.nan)),
            "num_leaves": report.get("model_params", {}).get("num_leaves"),
            "max_depth": report.get("model_params", {}).get("max_depth"),
            "min_child_samples": report.get("model_params", {}).get("min_child_samples"),
            "reg_alpha": report.get("model_params", {}).get("reg_alpha"),
            "reg_lambda": report.get("model_params", {}).get("reg_lambda"),
            "fixed_iteration": report.get("fixed_iteration"),
        }
        pair_payload = report.get("pair_outputs", {}).get("GALAXY:STAR", {})
        best = pair_payload.get("best", {})
        row.update(
            {
                "gs_delta": best.get("delta"),
                "gs_changed_rows": best.get("changed_rows"),
                "gs_mask": best.get("mask"),
                "gs_to_right": best.get("to_right_threshold"),
                "gs_to_left": best.get("to_left_threshold"),
                "gs_transitions": json.dumps(best.get("transition_counts", {}), ensure_ascii=False),
            }
        )
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No reports found under {root}")
    return pd.DataFrame(rows).sort_values("combined_delta", ascending=False)


def line_chart_svg(
    output: Path,
    title: str,
    subtitle: str,
    df: pd.DataFrame,
    x_col: str,
    series: list[tuple[str, str, str]],
    y_label: str,
    best_mode: str = "max",
) -> None:
    width = 1500
    height = 760
    left = 116
    right = 240
    top = 92
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_values = sorted(df[x_col].dropna().astype(float).unique().tolist())
    if not x_values:
        return
    all_y = []
    for col, _, _ in series:
        all_y.extend(df[col].dropna().astype(float).tolist())
    y_min, y_max = min(all_y), max(all_y)
    y_pad = max((y_max - y_min) * 0.16, 1e-6)
    y_min -= y_pad
    y_max += y_pad
    y_ticks = linear_ticks(y_min, y_max, 7)
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    def x_pos(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="32" y="38" font-family="Arial" font-size="22" font-weight="700" fill="#111827">{esc(title)}</text>',
        f'<text x="32" y="66" font-family="Arial" font-size="13" fill="#4b5563">{esc(subtitle)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>',
    ]

    for tick in y_ticks:
        y = y_pos(tick)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>',
                f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#374151">{tick:.6f}</text>',
            ]
        )
    for tick in x_values:
        x = x_pos(tick)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}" stroke="#f3f4f6" stroke-width="1"/>',
                f'<text x="{x:.2f}" y="{top+plot_h+28}" text-anchor="middle" font-family="Arial" font-size="12" fill="#374151">{int(tick) if tick.is_integer() else tick:g}</text>',
            ]
        )

    best_points = []
    for col, color, label in series:
        grouped = df.groupby(x_col, as_index=False)[col].mean().sort_values(x_col)
        points = " ".join(
            f'{x_pos(float(row[x_col])):.2f},{y_pos(float(row[col])):.2f}'
            for _, row in grouped.iterrows()
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3.2"/>')
        for _, row in grouped.iterrows():
            x = x_pos(float(row[x_col]))
            y = y_pos(float(row[col]))
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" stroke="#ffffff" stroke-width="1.4"/>')
        idx = grouped[col].idxmax() if best_mode == "max" else grouped[col].idxmin()
        best_points.append((label, color, grouped.loc[idx]))

    for idx, (label, color, row) in enumerate(best_points):
        x = x_pos(float(row[x_col]))
        y = y_pos(float(row[series[idx][0]]))
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8" fill="#f59e0b" stroke="#111827" stroke-width="1.4"/>')

    parts.extend(
        [
            f'<text x="{left+plot_w/2}" y="{height-28}" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827">{esc(x_col)}</text>',
            f'<text x="28" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 28 {top+plot_h/2})" font-family="Arial" font-size="14" fill="#111827">{esc(y_label)}</text>',
        ]
    )

    legend_x = left + plot_w + 28
    legend_y = top + 16
    for idx, (_, color, label) in enumerate(series):
        y = legend_y + idx * 30
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+28}" y2="{y}" stroke="{color}" stroke-width="3.2"/>',
                f'<text x="{legend_x+38}" y="{y+5}" font-family="Arial" font-size="13" fill="#111827">{esc(label)}</text>',
            ]
        )
    parts.append(f'<circle cx="{legend_x+14}" cy="{legend_y+len(series)*30}" r="7" fill="#f59e0b" stroke="#111827" stroke-width="1.2"/>')
    parts.append(f'<text x="{legend_x+38}" y="{legend_y+len(series)*30+5}" font-family="Arial" font-size="13" fill="#111827">best point</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def experiment_delta_svg(output: Path, summary: pd.DataFrame) -> None:
    df = summary.copy().sort_values("combined_delta", ascending=False)
    width = 1500
    row_h = 62
    top = 92
    bottom = 70
    left = 390
    right = 140
    height = top + len(df) * row_h + bottom
    plot_w = width - left - right
    max_delta = max(float(df["combined_delta"].max()), 0.0001)

    def x_pos(value: float) -> float:
        return left + value / max_delta * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#111827">Boundary Experiment OOF Delta</text>',
        '<text x="32" y="66" font-family="Arial" font-size="14" fill="#4b5563">Full 5-fold pure-model delta; dashed line is candidate threshold 0.0001</text>',
    ]
    threshold_x = x_pos(0.0001)
    parts.append(f'<line x1="{threshold_x:.2f}" y1="{top-26}" x2="{threshold_x:.2f}" y2="{height-bottom+16}" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 5"/>')
    parts.append(f'<text x="{threshold_x+8:.2f}" y="{top-34}" font-family="Arial" font-size="13" fill="#dc2626">candidate threshold</text>')

    ticks = [0.0, max_delta * 0.25, max_delta * 0.5, max_delta * 0.75, max_delta]
    for tick in ticks:
        x = x_pos(tick)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top-18}" x2="{x:.2f}" y2="{height-bottom+16}" stroke="#e5e7eb" stroke-width="1"/>',
                f'<text x="{x:.2f}" y="{height-bottom+42}" text-anchor="middle" font-family="Arial" font-size="12" fill="#374151">{tick:.6f}</text>',
            ]
        )

    for idx, (_, row) in enumerate(df.iterrows()):
        y = top + idx * row_h
        delta = float(row["combined_delta"])
        color = "#2563eb" if delta < 0.0001 else "#16a34a"
        label = esc(row["run"])
        threshold = row.get("GALAXY_STAR_to_right", row.get("gs_to_right", ""))
        sub = (
            f"changed={int(row.get('GALAXY_STAR_changed_rows', row.get('gs_changed_rows', 0)) or 0)}  "
            f"threshold={format_short(threshold, 3)}"
        )
        parts.extend(
            [
                f'<text x="32" y="{y+18}" font-family="Arial" font-size="13" font-weight="700" fill="#111827">{label}</text>',
                f'<text x="32" y="{y+40}" font-family="Arial" font-size="12" fill="#4b5563">{esc(sub)}</text>',
                f'<rect x="{left}" y="{y}" width="{x_pos(delta)-left:.2f}" height="28" fill="{color}" opacity="0.88"/>',
                f'<text x="{x_pos(delta)+10:.2f}" y="{y+20}" font-family="Arial" font-size="13" fill="#111827">{delta:.8f}</text>',
            ]
        )
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = read_reports(args.root)
    summary.to_csv(args.output_dir / "boundary_experiment_summary.csv", index=False)
    experiment_delta_svg(args.output_dir / "boundary_experiment_delta.svg", summary)

    best_run = summary.iloc[0]["run"]
    diag_path = args.root / best_run / "boundary_pair_training_diagnostics.csv"
    diag = pd.read_csv(diag_path)
    line_chart_svg(
        args.output_dir / "best_run_balanced_accuracy_curve.svg",
        "Boundary Run Balanced Accuracy",
        f"best run: {best_run}; fold mean with actual iteration ticks; orange marks the best mean validation point.",
        diag,
        "iteration",
        [
            ("train_binary_balanced_accuracy", "#dc2626", "train mean"),
            ("valid_binary_balanced_accuracy", "#2563eb", "valid mean"),
        ],
        "binary balanced accuracy",
        "max",
    )
    line_chart_svg(
        args.output_dir / "best_run_logloss_curve.svg",
        "Boundary Run Logloss",
        f"best run: {best_run}; logloss may keep improving after balanced accuracy has plateaued.",
        diag,
        "iteration",
        [
            ("train_binary_logloss", "#dc2626", "train mean"),
            ("valid_binary_logloss", "#2563eb", "valid mean"),
        ],
        "binary logloss",
        "min",
    )
    print("wrote publication boundary graphs:")
    for path in sorted(args.output_dir.glob("*.svg")):
        print(f"- {path}")


if __name__ == "__main__":
    main()
