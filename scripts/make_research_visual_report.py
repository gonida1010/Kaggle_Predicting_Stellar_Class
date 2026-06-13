from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "research_report"

BASE_ANCHOR_CANDIDATES = [
    ARTIFACTS / "probe_queue" / "group_minconf_095_star_to_galaxy_except_rank_01.csv",
    ARTIFACTS / "probe_queue" / "group_minconf_095_all.csv",
    DATA / "submission.csv",
]

CLASS_COLORS = {
    "GALAXY": "#4C78A8",
    "QSO": "#F58518",
    "STAR": "#54A24B",
}
TYPE_COLORS = {
    "M": "#4C78A8",
    "G/K": "#F58518",
    "A/F": "#54A24B",
    "O/B": "#B279A2",
}


def load_base_anchor() -> tuple[pd.DataFrame, Path]:
    for path in BASE_ANCHOR_CANDIDATES:
        if path.exists():
            return pd.read_csv(path), path
    raise FileNotFoundError("No usable base anchor submission was found.")


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222} .axis{stroke:#555;stroke-width:1} .grid{stroke:#ddd;stroke-width:1}</style>',
    ]


def write_svg(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines + ["</svg>\n"]), encoding="utf-8")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def dataframe_to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    if index:
        display = df.reset_index()
    else:
        display = df.reset_index(drop=True)

    columns = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    widths = [
        max(len(columns[col_idx]), *(len(row[col_idx]) for row in rows)) if rows else len(columns[col_idx])
        for col_idx in range(len(columns))
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [
        fmt_row(columns),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def scale(values: pd.Series | np.ndarray, lo: float, hi: float, reverse: bool = False):
    arr = np.asarray(values, dtype=float)
    vmin = np.nanquantile(arr, 0.01)
    vmax = np.nanquantile(arr, 0.99)
    if vmax <= vmin:
        vmax = vmin + 1.0

    def mapper(v: float) -> float:
        clipped = min(max(float(v), vmin), vmax)
        ratio = (clipped - vmin) / (vmax - vmin)
        if reverse:
            ratio = 1.0 - ratio
        return lo + ratio * (hi - lo)

    return mapper, vmin, vmax


def draw_class_mix(train: pd.DataFrame, original_anchor: pd.DataFrame, base_anchor: pd.DataFrame, base_name: str) -> None:
    width, height = 860, 440
    margin_l, margin_t, margin_b = 80, 58, 80
    plot_w, plot_h = width - 130, height - margin_t - margin_b

    frames = {
        "train": train["class"],
        "original anchor": original_anchor["class"],
        "base anchor": base_anchor["class"],
    }
    classes = ["GALAXY", "QSO", "STAR"]
    shares = pd.DataFrame({name: series.value_counts(normalize=True).reindex(classes, fill_value=0) for name, series in frames.items()})

    lines = svg_header(width, height)
    lines.append('<text x="24" y="32" font-size="22" font-weight="700">Class Mix: Train vs Anchors</text>')
    lines.append(f'<text x="24" y="54" font-size="12">base anchor: {esc(base_name)}</text>')

    for tick in np.linspace(0, 0.7, 8):
        y = margin_t + plot_h - tick / 0.7 * plot_h
        lines.append(f'<line class="grid" x1="{margin_l}" y1="{y:.1f}" x2="{margin_l + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text x="42" y="{y + 4:.1f}" font-size="11">{tick:.0%}</text>')

    group_w = plot_w / len(frames)
    bar_w = 44
    for group_idx, name in enumerate(frames):
        center = margin_l + group_w * group_idx + group_w / 2
        for class_idx, cls in enumerate(classes):
            value = shares.loc[cls, name]
            x = center + (class_idx - 1) * (bar_w + 6)
            y = margin_t + plot_h - value / 0.7 * plot_h
            h = value / 0.7 * plot_h
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{CLASS_COLORS[cls]}"/>'
            )
            lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" font-size="10" text-anchor="middle">{value:.1%}</text>')
        lines.append(f'<text x="{center:.1f}" y="{height - 38}" font-size="12" text-anchor="middle">{esc(name)}</text>')

    for idx, cls in enumerate(classes):
        x = 610 + idx * 78
        lines.append(f'<rect x="{x}" y="24" width="12" height="12" fill="{CLASS_COLORS[cls]}"/>')
        lines.append(f'<text x="{x + 18}" y="35" font-size="12">{cls}</text>')

    lines.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}"/>')
    write_svg(OUT_DIR / "class_mix.svg", lines)


def draw_candidate_score(candidates: pd.DataFrame) -> None:
    width, height = 820, 560
    margin_l, margin_r, margin_t, margin_b = 72, 28, 58, 72
    plot_w, plot_h = width - margin_l - margin_r, height - margin_t - margin_b

    xmap, xmin, xmax = scale(candidates["local_p_GALAXY"], margin_l, margin_l + plot_w)
    ymap, ymin, ymax = scale(candidates["min_p_GALAXY"], margin_t, margin_t + plot_h, reverse=True)

    lines = svg_header(width, height)
    lines.append('<text x="24" y="34" font-size="22" font-weight="700">STAR -> GALAXY Candidate Score Map</text>')
    lines.append('<text x="24" y="54" font-size="12">x = local train evidence, y = minimum model GALAXY probability</text>')

    for ratio in np.linspace(0, 1, 6):
        x = margin_l + ratio * plot_w
        y = margin_t + plot_h - ratio * plot_h
        xv = xmin + ratio * (xmax - xmin)
        yv = ymin + ratio * (ymax - ymin)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_t}" x2="{x:.1f}" y2="{margin_t + plot_h}"/>')
        lines.append(f'<line class="grid" x1="{margin_l}" y1="{y:.1f}" x2="{margin_l + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text x="{x:.1f}" y="{height - 44}" font-size="10" text-anchor="middle">{xv:.2f}</text>')
        lines.append(f'<text x="42" y="{y + 4:.1f}" font-size="10">{yv:.2f}</text>')

    for _, row in candidates.sort_values("patchability_score").iterrows():
        x = xmap(row["local_p_GALAXY"])
        y = ymap(row["min_p_GALAXY"])
        color = TYPE_COLORS.get(row["spectral_type"], "#777")
        radius = 3.5 if row.get("next_probe_rank", 999) > 10 else 6.0
        opacity = 0.55 if row.get("next_probe_rank", 999) > 10 else 0.95
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" fill-opacity="{opacity}">'
            f'<title>id={int(row["id"])} rank={int(row.get("next_probe_rank", 0))} score={row["patchability_score"]:.2f}</title></circle>'
        )

    top = candidates.head(10)
    for _, row in top.iterrows():
        x = xmap(row["local_p_GALAXY"])
        y = ymap(row["min_p_GALAXY"])
        lines.append(f'<text x="{x + 8:.1f}" y="{y - 7:.1f}" font-size="10">{int(row["next_probe_rank"])}</text>')

    lines.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}"/>')
    lines.append(f'<text x="{width / 2:.1f}" y="{height - 16}" font-size="13" text-anchor="middle">local_p_GALAXY</text>')
    lines.append('<text x="18" y="300" font-size="13" transform="rotate(-90 18 300)" text-anchor="middle">min_p_GALAXY</text>')

    legend_y = 86
    for idx, (label, color) in enumerate(TYPE_COLORS.items()):
        y = legend_y + idx * 20
        lines.append(f'<circle cx="700" cy="{y}" r="5" fill="{color}"/>')
        lines.append(f'<text x="712" y="{y + 4}" font-size="12">{esc(label)}</text>')

    write_svg(OUT_DIR / "candidate_score_map.svg", lines)


def draw_color_redshift(train: pd.DataFrame, candidates: pd.DataFrame) -> None:
    width, height = 860, 560
    margin_l, margin_r, margin_t, margin_b = 72, 30, 58, 72
    plot_w, plot_h = width - margin_l - margin_r, height - margin_t - margin_b

    sample = train.sample(min(9000, len(train)), random_state=20260611).copy()
    sample["g-r"] = sample["g"] - sample["r"]
    candidates = candidates.copy()

    all_x = pd.concat([sample["g-r"], candidates["g-r"]], ignore_index=True)
    all_y = pd.concat([sample["redshift"], candidates["redshift"]], ignore_index=True)
    xmap, xmin, xmax = scale(all_x, margin_l, margin_l + plot_w)
    ymap, ymin, ymax = scale(all_y, margin_t, margin_t + plot_h, reverse=True)

    lines = svg_header(width, height)
    lines.append('<text x="24" y="34" font-size="22" font-weight="700">Color-Redshift View</text>')
    lines.append('<text x="24" y="54" font-size="12">background = train sample, outlined points = next STAR->GALAXY probe pool</text>')

    for ratio in np.linspace(0, 1, 6):
        x = margin_l + ratio * plot_w
        y = margin_t + plot_h - ratio * plot_h
        xv = xmin + ratio * (xmax - xmin)
        yv = ymin + ratio * (ymax - ymin)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_t}" x2="{x:.1f}" y2="{margin_t + plot_h}"/>')
        lines.append(f'<line class="grid" x1="{margin_l}" y1="{y:.1f}" x2="{margin_l + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text x="{x:.1f}" y="{height - 44}" font-size="10" text-anchor="middle">{xv:.2f}</text>')
        lines.append(f'<text x="38" y="{y + 4:.1f}" font-size="10">{yv:.2f}</text>')

    for _, row in sample.iterrows():
        color = CLASS_COLORS.get(row["class"], "#999")
        lines.append(
            f'<circle cx="{xmap(row["g-r"]):.1f}" cy="{ymap(row["redshift"]):.1f}" r="1.6" fill="{color}" fill-opacity="0.16"/>'
        )

    for _, row in candidates.iterrows():
        color = TYPE_COLORS.get(row["spectral_type"], "#222")
        lines.append(
            f'<circle cx="{xmap(row["g-r"]):.1f}" cy="{ymap(row["redshift"]):.1f}" r="5.8" fill="white" stroke="{color}" stroke-width="2">'
            f'<title>id={int(row["id"])} rank={int(row["next_probe_rank"])}</title></circle>'
        )

    for _, row in candidates.head(10).iterrows():
        lines.append(
            f'<text x="{xmap(row["g-r"]) + 8:.1f}" y="{ymap(row["redshift"]) - 5:.1f}" font-size="10">{int(row["next_probe_rank"])}</text>'
        )

    lines.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}"/>')
    lines.append(f'<text x="{width / 2:.1f}" y="{height - 16}" font-size="13" text-anchor="middle">g-r color index</text>')
    lines.append('<text x="18" y="300" font-size="13" transform="rotate(-90 18 300)" text-anchor="middle">redshift</text>')
    write_svg(OUT_DIR / "color_redshift_candidates.svg", lines)


def write_markdown_report(train: pd.DataFrame, original_anchor: pd.DataFrame, base_anchor: pd.DataFrame, base_path: Path, candidates: pd.DataFrame) -> None:
    class_mix = pd.DataFrame(
        {
            "train": train["class"].value_counts(normalize=True),
            "original_anchor": original_anchor["class"].value_counts(normalize=True),
            "base_anchor": base_anchor["class"].value_counts(normalize=True),
        }
    ).reindex(["GALAXY", "QSO", "STAR"]).fillna(0)

    top_cols = [
        "next_probe_rank",
        "id",
        "patchability_score",
        "min_p_GALAXY",
        "local_p_GALAXY",
        "local_p_STAR",
        "spectral_type",
        "galaxy_population",
        "redshift",
    ]
    top_table = candidates[top_cols].head(20).copy()
    for col in ["patchability_score", "min_p_GALAXY", "local_p_GALAXY", "local_p_STAR", "redshift"]:
        top_table[col] = top_table[col].map(lambda value: f"{value:.6f}")

    lines = [
        "# Research Visual Report",
        "",
        "This report separates three things: the original shared anchor, the current base anchor, and the next model-driven STAR->GALAXY probe pool.",
        "",
        f"- base anchor: `{base_path.relative_to(ROOT)}`",
        f"- train rows: `{len(train):,}`",
        f"- test rows: `{len(original_anchor):,}`",
        f"- next probe candidates: `{len(candidates):,}`",
        "",
        "## Figures",
        "",
        "- `class_mix.svg`: class share difference between train, original anchor, and current base anchor.",
        "- `candidate_score_map.svg`: candidate map using model probability and local train evidence.",
        "- `color_redshift_candidates.svg`: train color-redshift background with candidate overlay.",
        "",
        "## Class Mix",
        "",
        dataframe_to_markdown(class_mix.map(lambda value: f"{value:.4f}")),
        "",
        "## Top Candidates",
        "",
        dataframe_to_markdown(top_table, index=False),
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")

    metrics = {
        "base_anchor": str(base_path.relative_to(ROOT)),
        "class_mix": class_mix.to_dict(),
        "next_probe_candidates": int(len(candidates)),
        "top_candidate_ids": candidates["id"].head(20).astype(int).tolist(),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    original_anchor = pd.read_csv(DATA / "submission.csv")
    base_anchor, base_path = load_base_anchor()
    candidates_path = ARTIFACTS / "star_to_galaxy_research" / "next_probe_pool.csv"
    if not candidates_path.exists():
        raise FileNotFoundError("Run scripts/score_star_to_galaxy_candidates.py first.")
    candidates = pd.read_csv(candidates_path)

    draw_class_mix(train, original_anchor, base_anchor, str(base_path.relative_to(ROOT)))
    draw_candidate_score(candidates)
    draw_color_redshift(train, candidates)
    write_markdown_report(train, original_anchor, base_anchor, base_path, candidates)

    print(f"wrote visual report to {OUT_DIR}")
    print(f"- {OUT_DIR / 'README.md'}")
    print(f"- {OUT_DIR / 'class_mix.svg'}")
    print(f"- {OUT_DIR / 'candidate_score_map.svg'}")
    print(f"- {OUT_DIR / 'color_redshift_candidates.svg'}")


if __name__ == "__main__":
    main()
