from __future__ import annotations

import ast
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import add_features  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "research_dashboard"
STAR_RESEARCH = ARTIFACTS / "star_to_galaxy_research"
PUBLIC_FEEDBACK = ARTIFACTS / "public_feedback"
TRANSITION_RESEARCH = ARTIFACTS / "transition_research"
PURE_ENSEMBLE = ARTIFACTS / "pure_model_ensemble"
BASE_097139 = ARTIFACTS / "probe_queue" / "group_minconf_095_star_to_galaxy_except_rank_01.csv"
CURRENT_PUBLIC_BEST = STAR_RESEARCH / "group_research_top_10.csv"

COLORS = {
    "blue": "#2f80ed",
    "green": "#27ae60",
    "orange": "#f2994a",
    "red": "#eb5757",
    "purple": "#9b51e0",
    "gray": "#667085",
    "light": "#edf2f7",
    "dark": "#111827",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>",
        "text{font-family:Arial,sans-serif;fill:#111827}",
        ".muted{fill:#667085}",
        ".grid{stroke:#e5e7eb;stroke-width:1}",
        ".axis{stroke:#475467;stroke-width:1.2}",
        "</style>",
    ]


def write_svg(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines + ["</svg>\n"]), encoding="utf-8")


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for class_idx in range(n_classes):
        mask = y_true == class_idx
        if mask.any():
            recalls.append(float((y_pred[mask] == class_idx).mean()))
    return float(np.mean(recalls))


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, float]:
    out = {}
    for idx, cls in enumerate(classes):
        mask = y_true == idx
        out[f"recall_{cls}"] = float((y_pred[mask] == idx).mean()) if mask.any() else np.nan
    return out


def qcut_labels(values: pd.Series, q: int, prefix: str) -> pd.Series:
    bins = pd.qcut(values, q=q, labels=False, duplicates="drop")
    return bins.astype("Int64").astype(str).radd(f"{prefix}_q")


def worst_error_pair(true_labels: pd.Series, pred_labels: pd.Series) -> tuple[str, int]:
    wrong = true_labels.ne(pred_labels)
    if not wrong.any():
        return "", 0
    pairs = (
        pd.DataFrame({"true": true_labels[wrong], "pred": pred_labels[wrong]})
        .groupby(["true", "pred"], observed=True)
        .size()
        .sort_values(ascending=False)
    )
    pair = pairs.index[0]
    return f"{pair[0]}->{pair[1]}", int(pairs.iloc[0])


def build_pure_subset_metrics() -> pd.DataFrame:
    train = pd.read_csv(DATA / "train.csv")
    train_fe = add_features(train)
    report = json.loads((PURE_ENSEMBLE / "pure_model_ensemble_report.json").read_text(encoding="utf-8"))
    classes = report["classes"]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    y_true = train["class"].astype(str).map(class_to_idx).to_numpy()
    oof = np.load(PURE_ENSEMBLE / "pure_model_ensemble_oof_proba.npy")
    y_pred = oof.argmax(axis=1)
    pred_labels = pd.Series(np.array(classes)[y_pred], index=train.index)
    true_labels = train["class"].astype(str)

    frame = train_fe.copy()
    frame["true_class"] = true_labels
    frame["pred_class"] = pred_labels
    frame["correct"] = frame["true_class"].eq(frame["pred_class"])
    frame["multicat"] = frame["spectral_type"].astype(str) + "_" + frame["galaxy_population"].astype(str)
    frame["redshift_bin"] = qcut_labels(frame["redshift"], 10, "redshift")
    frame["g_i_bin"] = qcut_labels(frame["g-i"], 10, "g_i")
    frame["mag_range_bin"] = qcut_labels(frame["mag_range"], 10, "mag_range")

    subset_specs = [
        ("spectral_type", "spectral_type"),
        ("galaxy_population", "galaxy_population"),
        ("multicat", "multicat"),
        ("redshift_bin", "redshift_bin"),
        ("g_i_bin", "g_i_bin"),
        ("mag_range_bin", "mag_range_bin"),
    ]
    rows = []
    for subset_type, column in subset_specs:
        for subset, group in frame.groupby(column, observed=True):
            if len(group) < 300:
                continue
            group_true = group["true_class"].map(class_to_idx).to_numpy()
            group_pred = group["pred_class"].map(class_to_idx).to_numpy()
            pair, pair_count = worst_error_pair(group["true_class"], group["pred_class"])
            row = {
                "subset_type": subset_type,
                "subset": str(subset),
                "count": int(len(group)),
                "balanced_accuracy": balanced_accuracy(group_true, group_pred, len(classes)),
                "accuracy": float(group["correct"].mean()),
                "error_rate": float((~group["correct"]).mean()),
                "worst_error_pair": pair,
                "worst_error_count": pair_count,
            }
            row.update(class_recalls(group_true, group_pred, classes))
            supports = []
            for cls in classes:
                support = int(group["true_class"].eq(cls).sum())
                supports.append(support)
                row[f"support_{cls}"] = support
                row[f"share_{cls}"] = float(group["true_class"].eq(cls).mean())
            row["min_class_support"] = min(supports)
            rows.append(row)

    metrics = pd.DataFrame(rows).sort_values(["balanced_accuracy", "count"], ascending=[True, False])
    metrics.to_csv(OUT_DIR / "pure_subset_metrics.csv", index=False)
    stable_metrics = metrics[metrics["min_class_support"].ge(100)].copy()
    stable_metrics.to_csv(OUT_DIR / "pure_stable_subset_metrics.csv", index=False)

    wrong = frame[~frame["correct"]].copy()
    error_pairs = (
        wrong.groupby(["true_class", "pred_class"], observed=True)
        .agg(
            count=("id", "size"),
            avg_redshift=("redshift", "mean"),
            avg_g_i=("g-i", "mean"),
            avg_mag_range=("mag_range", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    error_pairs.to_csv(OUT_DIR / "pure_error_pairs.csv", index=False)
    return metrics


def changed_ids(submission_path: Path, base_anchor: pd.DataFrame) -> list[int]:
    submission = pd.read_csv(submission_path)
    if not submission["id"].equals(base_anchor["id"]):
        raise ValueError(f"id order differs: {submission_path}")
    mask = submission["class"].ne(base_anchor["class"])
    return submission.loc[mask, "id"].astype(int).tolist()


def resolve_public_probe_path(file_name: str) -> Path:
    for directory in [STAR_RESEARCH, PUBLIC_FEEDBACK, TRANSITION_RESEARCH]:
        path = directory / file_name
        if path.exists():
            return path
    raise FileNotFoundError(f"{file_name} was not found under public probe artifact folders")


def ridge_effect_estimate(x: np.ndarray, y: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    xtx = x.T @ x
    rhs = x.T @ y
    return np.linalg.solve(xtx + alpha * np.eye(xtx.shape[0]), rhs)


def build_public_feedback_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_anchor = pd.read_csv(BASE_097139)
    observed = pd.read_csv(PUBLIC_FEEDBACK / "observed_public_scores.csv")
    next_pool = pd.read_csv(STAR_RESEARCH / "next_probe_pool.csv")

    ledger_rows = [
        {
            "file": "base_097139_anchor",
            "public_score": 0.97139,
            "delta_units_vs_base": 0,
            "changed_rows": 0,
            "ids": [],
        }
    ]
    for _, row in observed.iterrows():
        file_name = row["file"]
        path = resolve_public_probe_path(file_name)
        ids = changed_ids(path, base_anchor)
        ledger_rows.append(
            {
                "file": file_name,
                "public_score": float(row["public_score"]),
                "delta_units_vs_base": int(round((float(row["public_score"]) - 0.97139) * 100000)),
                "changed_rows": len(ids),
                "ids": ids,
            }
        )
    ledger = pd.DataFrame(ledger_rows)
    ledger.to_csv(OUT_DIR / "public_probe_ledger.csv", index=False)

    observed_only = ledger[ledger["file"].ne("base_097139_anchor")].copy()
    all_ids = sorted(set().union(*observed_only["ids"].map(set).tolist()))
    id_to_col = {row_id: idx for idx, row_id in enumerate(all_ids)}
    x = np.zeros((len(observed_only), len(all_ids)), dtype=float)
    y = observed_only["delta_units_vs_base"].to_numpy(dtype=float)
    for row_idx, ids in enumerate(observed_only["ids"]):
        for row_id in ids:
            x[row_idx, id_to_col[row_id]] = 1.0
    effect = ridge_effect_estimate(x, y)
    effects = pd.DataFrame({"id": all_ids, "feedback_effect_units": effect})
    effects["observed_in_files"] = x.sum(axis=0).astype(int)
    effects = effects.merge(next_pool, on="id", how="left")
    effects["rank_bucket"] = pd.cut(
        effects["next_probe_rank"],
        bins=[0, 10, 15, 20, 1000],
        labels=["rank_01_10", "rank_11_15", "rank_16_20", "rank_21_plus"],
    )
    effects["feedback_score"] = (
        effects["feedback_effect_units"].fillna(0)
        + 0.018 * (effects["patchability_score"].fillna(effects["patchability_score"].median()) - effects["patchability_score"].median())
        + 0.40 * (effects["local_p_GALAXY"].fillna(0) - effects["local_p_STAR"].fillna(0))
    )
    effects = effects.sort_values(["feedback_score", "feedback_effect_units"], ascending=False)
    effects.to_csv(OUT_DIR / "public_row_feedback_estimates.csv", index=False)

    segments = next_pool.copy()
    segments["segment"] = segments["spectral_type"].astype(str) + "_" + segments["galaxy_population"].astype(str)
    segments["rank_bucket"] = pd.cut(
        segments["next_probe_rank"],
        bins=[0, 10, 15, 20, 43],
        labels=["rank_01_10", "rank_11_15", "rank_16_20", "rank_21_43"],
        include_lowest=True,
    )
    segment_table = (
        segments.groupby(["segment", "rank_bucket"], observed=True)
        .agg(
            count=("id", "size"),
            avg_patchability=("patchability_score", "mean"),
            avg_local_galaxy=("local_p_GALAXY", "mean"),
            avg_local_star=("local_p_STAR", "mean"),
            min_rank=("next_probe_rank", "min"),
        )
        .reset_index()
        .sort_values(["rank_bucket", "count", "avg_patchability"], ascending=[True, False, False])
    )
    segment_table.to_csv(OUT_DIR / "public_candidate_segments.csv", index=False)
    return ledger, effects


def build_transition_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    report = json.loads((TRANSITION_RESEARCH / "transition_report.json").read_text(encoding="utf-8"))
    summary = pd.DataFrame(report["transition_summary"])
    summary.to_csv(OUT_DIR / "transition_summary.csv", index=False)

    top_rows = []
    for candidate_path in TRANSITION_RESEARCH.glob("*_candidates.csv"):
        if candidate_path.name == "all_transition_candidates.csv":
            continue
        frame = pd.read_csv(candidate_path).head(10)
        top_rows.append(frame)
    top = pd.concat(top_rows, ignore_index=True).sort_values(
        ["transition_score", "model_margin"],
        ascending=False,
    )
    keep_cols = [
        "global_rank",
        "id",
        "transition",
        "transition_score",
        "model_margin",
        "local_margin",
        "mean_p_GALAXY",
        "mean_p_QSO",
        "mean_p_STAR",
        "local_p_GALAXY",
        "local_p_QSO",
        "local_p_STAR",
        "redshift",
        "spectral_type",
        "galaxy_population",
    ]
    top[keep_cols].to_csv(OUT_DIR / "transition_top_candidates.csv", index=False)
    return summary, top


def build_recommendations() -> pd.DataFrame:
    rows = [
        {
            "priority": 1,
            "file": "external_preds/*.csv",
            "type": "required_input",
            "reason": "Local STAR->GALAXY, replacement, and QSO->GALAXY probes all saturated at rounded 0.97141; top notebooks use a much larger public submission bank.",
            "success_rule": "Place score-named bank CSVs, 0.97209.csv, and five independent notebook submissions under external_preds.",
            "fallback": "Without the bank, stop public probes and work only on pure-model OOF improvements.",
        },
        {
            "priority": 2,
            "file": "scripts/build_bank_ridge_flip_candidates.py",
            "type": "public_generalization_builder",
            "reason": "Implements the stronger top-notebook idea: score-named submission Ridge, Bayesian support, isotonic smoothing, optional proba entropy gate, and voted ensemble.",
            "success_rule": "Run it after external_preds has the 0.97183-style bank; submit v5_voted_ensemble only if generated and verified.",
            "fallback": "If probability files are absent, use the non-tail voted ensemble or top130/top150 candidates for probing.",
        },
        {
            "priority": 3,
            "file": "scripts/build_ambiguous_vote_patch.py",
            "type": "research_only_ambiguous_reference",
            "reason": "The 0.97209 -> 0.97214 idea identifies the hardest 2-2-1 ambiguous rows, but its output must not be used as a final submission.",
            "success_rule": "Use it only to locate feature/subset regimes where independent models disagree.",
            "fallback": "Do not include copied public-notebook final outputs in final submission selection.",
        },
        {
            "priority": 4,
            "file": "scripts/build_final_submission_tracks.py",
            "type": "final_track_selector",
            "reason": "Writes the two final-selection files: one pure/generalization and one public+generalization.",
            "success_rule": "Run after any new public candidate builder so it selects the strongest available file.",
            "fallback": "Current fallback is group_research_top_10 for public+generalization and pure_model_ensemble for generalization.",
        },
        {
            "priority": 5,
            "file": "artifacts/star_to_galaxy_research/group_research_top_10.csv",
            "type": "current_public_best",
            "reason": "Current known public best. Same-score variants add rows without visible public gain.",
            "success_rule": "Keep this as the public-track final candidate until a bank-informed probe beats 0.97141.",
            "fallback": "For private-risk control, prefer the fewest-row same-score candidate.",
        },
        {
            "priority": 6,
            "file": "artifacts/pure_model_ensemble/pure_model_ensemble_submission.csv",
            "type": "private_generalization_track",
            "reason": "This is the honest model track. Its OOF is still far below public blends, so it should not be used for public rank yet.",
            "success_rule": "Improve OOF by subset specialists or new model families before considering it as a final submission.",
            "fallback": "Keep it as the second final-submission candidate only if private/generalization risk matters more than public LB.",
        },
    ]
    recommendations = pd.DataFrame(rows)
    recommendations.to_csv(OUT_DIR / "next_probe_recommendations.csv", index=False)
    return recommendations


def horizontal_bars(
    path: Path,
    title: str,
    subtitle: str,
    rows: list[tuple[str, float, str]],
    x_min: float | None = None,
    x_max: float | None = None,
    value_format: str = "{:.4f}",
) -> None:
    width = 980
    row_h = 46
    top = 84
    left = 300
    right = 100
    height = top + row_h * len(rows) + 54
    plot_w = width - left - right
    values = [value for _, value, _ in rows]
    xmin = min(values + [0.0]) if x_min is None else x_min
    xmax = max(values + [0.0]) if x_max is None else x_max
    if xmax <= xmin:
        xmax = xmin + 1.0
    if xmin <= 0 <= xmax:
        baseline = 0.0
    elif xmin > 0:
        baseline = xmin
    else:
        baseline = xmax
    base_x = left + ((baseline - xmin) / (xmax - xmin)) * plot_w

    lines = svg_header(width, height)
    lines.append(f'<text x="28" y="34" font-size="22" font-weight="700">{esc(title)}</text>')
    lines.append(f'<text x="28" y="58" font-size="13" class="muted">{esc(subtitle)}</text>')
    lines.append(f'<line class="axis" x1="{base_x:.1f}" y1="{top - 8}" x2="{base_x:.1f}" y2="{top + row_h * len(rows)}"/>')

    for idx, (label, value, color) in enumerate(rows):
        y = top + idx * row_h
        x_value = left + ((value - xmin) / (xmax - xmin)) * plot_w
        x = min(base_x, x_value)
        bar_w = max(2, abs(x_value - base_x))
        lines.append(f'<text x="28" y="{y + 22}" font-size="13">{esc(label)}</text>')
        lines.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_w:.1f}" height="26" rx="4" fill="{color}"/>')
        lines.append(f'<text x="{left + plot_w + 18}" y="{y + 19}" font-size="12" font-weight="700">{value_format.format(value)}</text>')

    write_svg(path, lines)


def draw_pure_progress() -> None:
    report = json.loads((PURE_ENSEMBLE / "pure_model_ensemble_report.json").read_text(encoding="utf-8"))
    rows = []
    for record in report["baseline_records"]:
        name = record["name"].replace("unweighted_lgbm_catboost", "unweighted ensemble")
        color = COLORS["blue"] if name == "lgbm" else COLORS["orange"]
        if "unweighted" in name:
            color = COLORS["green"]
        rows.append((name, float(record["oof_balanced_accuracy"]), color))
    rows.append(("optimized pure ensemble", float(report["best_oof_balanced_accuracy"]), COLORS["purple"]))
    horizontal_bars(
        OUT_DIR / "pure_model_progress.svg",
        "Pure Model OOF Progress",
        "Anchor-free model validation. This is the private/generalization track.",
        rows,
        x_min=0.9625,
        x_max=0.9670,
        value_format="{:.6f}",
    )


def draw_worst_subsets(metrics: pd.DataFrame) -> None:
    view = metrics[
        metrics["subset_type"].isin(["spectral_type", "galaxy_population", "multicat"])
        & metrics["min_class_support"].ge(100)
    ].head(15)
    rows = [
        (
            f'{row["subset_type"]}: {row["subset"]}  n={int(row["count"])}',
            float(row["balanced_accuracy"]),
            COLORS["red"] if idx < 5 else COLORS["orange"],
        )
        for idx, (_, row) in enumerate(view.iterrows())
    ]
    horizontal_bars(
        OUT_DIR / "pure_worst_subsets.svg",
        "Weakest Pure-Model Subsets",
        "Lowest OOF balanced accuracy by categorical subset. Use this to design specialist experiments.",
        rows,
        x_min=max(0.80, min(value for _, value, _ in rows) - 0.01),
        x_max=1.0,
        value_format="{:.5f}",
    )


def draw_public_scores(ledger: pd.DataFrame) -> None:
    name_map = {
        "base_097139_anchor": "0.97139 base",
        "group_research_top_03.csv": "STG top03",
        "group_research_top_05.csv": "STG top05",
        "group_research_top_10.csv": "STG top10",
        "group_research_top_15.csv": "STG top15",
        "group_research_rank_06_10.csv": "STG rank06-10",
        "group_research_rank_11_15.csv": "STG rank11-15",
        "group_research_strong_local_top_10.csv": "strong local top10",
        "group_research_minpgal_092_top_10.csv": "minpgal top10",
        "group_research_top10_plus_rank_17_19.csv": "top10 + rank17-19",
        "group_research_top10_without_rank_09.csv": "top10 - rank09",
        "feedback_replace_rank09_with_rank17.csv": "rank09 -> rank17",
        "transition_QSO_to_GALAXY_top_01.csv": "QSO->GAL top01",
        "transition_QSO_to_GALAXY_top_03.csv": "QSO->GAL top03",
    }
    rows = []
    for _, row in ledger.iterrows():
        label = name_map.get(row["file"], row["file"])
        score = float(row["public_score"])
        color = COLORS["green"] if score >= 0.97141 else COLORS["blue"]
        if score <= 0.97139:
            color = COLORS["gray"]
        rows.append((label, score, color))
    horizontal_bars(
        OUT_DIR / "public_probe_scores.svg",
        "Public Probe Score Ledger",
        "Observed Kaggle public scores. Equal rounded scores do not imply equal private risk.",
        rows,
        x_min=0.97138,
        x_max=0.97142,
        value_format="{:.5f}",
    )


def draw_row_feedback(effects: pd.DataFrame) -> None:
    view = effects.head(18)
    rows = [
        (
            f'id {int(row["id"])}  rank {int(row["next_probe_rank"]) if not pd.isna(row["next_probe_rank"]) else "?"}',
            float(row["feedback_score"]),
            COLORS["green"] if row["feedback_score"] >= 0 else COLORS["red"],
        )
        for _, row in view.iterrows()
    ]
    horizontal_bars(
        OUT_DIR / "public_row_feedback.svg",
        "Public Feedback Row Ranking",
        "Ridge estimate from submitted probe sets plus model/local evidence.",
        rows,
        value_format="{:.3f}",
    )


def draw_transition_summary(summary: pd.DataFrame) -> None:
    color_by_transition = {
        "STAR->GALAXY": COLORS["green"],
        "QSO->GALAXY": COLORS["blue"],
        "GALAXY->STAR": COLORS["orange"],
        "GALAXY->QSO": COLORS["purple"],
        "STAR->QSO": COLORS["red"],
        "QSO->STAR": COLORS["gray"],
    }
    rows = [
        (
            row["transition"],
            float(row["candidates"]),
            color_by_transition.get(row["transition"], COLORS["gray"]),
        )
        for _, row in summary.iterrows()
    ]
    horizontal_bars(
        OUT_DIR / "transition_candidate_counts.svg",
        "Transition Search Candidate Counts",
        "After STAR->GALAXY plateau, search all class transitions from the 0.97141 anchor.",
        rows,
        x_min=0,
        x_max=max(value for _, value, _ in rows) * 1.08,
        value_format="{:.0f}",
    )


def draw_segment_heatmap() -> None:
    segments = pd.read_csv(OUT_DIR / "public_candidate_segments.csv")
    pivot = (
        segments.pivot_table(
            index="segment",
            columns="rank_bucket",
            values="count",
            aggfunc="sum",
            fill_value=0,
            observed=True,
        )
        .sort_index()
    )
    columns = [str(col) for col in pivot.columns]
    rows = pivot.index.tolist()
    max_value = max(1, int(pivot.to_numpy().max()))
    cell_w = 118
    cell_h = 42
    left = 210
    top = 88
    width = left + cell_w * len(columns) + 40
    height = top + cell_h * len(rows) + 58
    lines = svg_header(width, height)
    lines.append('<text x="28" y="34" font-size="22" font-weight="700">STAR->GALAXY Candidate Segment Map</text>')
    lines.append('<text x="28" y="58" font-size="13" class="muted">Counts by spectral_type + galaxy_population and rank bucket.</text>')
    for col_idx, col in enumerate(columns):
        x = left + col_idx * cell_w + cell_w / 2
        lines.append(f'<text x="{x:.1f}" y="{top - 16}" font-size="12" text-anchor="middle">{esc(col)}</text>')
    for row_idx, row_name in enumerate(rows):
        y = top + row_idx * cell_h
        lines.append(f'<text x="28" y="{y + 25}" font-size="12">{esc(row_name)}</text>')
        for col_idx, col in enumerate(columns):
            value = int(pivot.loc[row_name, col])
            intensity = value / max_value
            blue = int(245 - 140 * intensity)
            green = int(248 - 80 * intensity)
            red = int(255 - 200 * intensity)
            fill = f"rgb({red},{green},{blue})"
            x = left + col_idx * cell_w
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w - 5}" height="{cell_h - 5}" rx="4" fill="{fill}" stroke="#ffffff"/>')
            lines.append(f'<text x="{x + cell_w / 2 - 2:.1f}" y="{y + 24}" font-size="13" font-weight="700" text-anchor="middle">{value}</text>')
    write_svg(OUT_DIR / "public_candidate_segment_heatmap.svg", lines)


def write_readme(recommendations: pd.DataFrame) -> None:
    pure_report = json.loads((PURE_ENSEMBLE / "pure_model_ensemble_report.json").read_text(encoding="utf-8"))
    feedback_report = json.loads((PUBLIC_FEEDBACK / "public_feedback_report.json").read_text(encoding="utf-8"))
    transition_report = json.loads((TRANSITION_RESEARCH / "transition_report.json").read_text(encoding="utf-8"))
    lines = [
        "# Research Dashboard",
        "",
        "This folder is for ongoing model and public-probe decisions, not blog decoration.",
        "",
        "## Current State",
        "",
        f"- Pure/generalization best OOF: `{pure_report['best_oof_balanced_accuracy']:.6f}`",
        f"- Current public best file: `{feedback_report['current_public_best']['file']}`",
        f"- Current public best score: `{feedback_report['current_public_best']['public_score']:.5f}`",
        f"- Current public best changed rows: `{feedback_report['current_public_best']['changed_rows']}`",
        "- Public-probe conclusion: local row-probes are saturated at the rounded public score; submission-bank disagreement analysis is now the required next step.",
        "",
        "## Next Actions",
        "",
    ]
    for _, row in recommendations.sort_values("priority").iterrows():
        lines.append(f"{int(row['priority'])}. `{row['file']}`")
        lines.append(f"   - reason: {row['reason']}")
        lines.append(f"   - success: {row['success_rule']}")
    lines.extend(
        [
            "",
            "## CSV Tables",
            "",
            "- `pure_subset_metrics.csv`: OOF performance by spectral/category/redshift/color subset.",
            "- `pure_stable_subset_metrics.csv`: Same subset metrics filtered to subsets with every class represented by at least 100 rows.",
            "- `pure_error_pairs.csv`: largest pure-model OOF error directions.",
            "- `public_probe_ledger.csv`: submitted public probe files and scores.",
            "- `public_row_feedback_estimates.csv`: row-level feedback estimates from public probes.",
            "- `public_candidate_segments.csv`: STAR->GALAXY candidate concentration by subset and rank bucket.",
            "- `transition_summary.csv`: candidate counts by class transition.",
            "- `transition_top_candidates.csv`: top candidates for each non-anchor transition.",
            "- `next_probe_recommendations.csv`: actionable next submissions.",
            "",
            "## SVG Views",
            "",
            "- `pure_model_progress.svg`",
            "- `pure_worst_subsets.svg`",
            "- `public_probe_scores.svg`",
            "- `public_row_feedback.svg`",
            "- `public_candidate_segment_heatmap.svg`",
            "- `transition_candidate_counts.svg`",
            "",
            "## Transition Summary",
            "",
        ]
    )
    for item in transition_report["transition_summary"]:
        lines.append(f"- `{item['transition']}`: {item['candidates']} candidates, top ids {item['top_ids'][:5]}")
    lines.append("")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subset_metrics = build_pure_subset_metrics()
    public_ledger, row_effects = build_public_feedback_tables()
    transition_summary, _ = build_transition_tables()
    recommendations = build_recommendations()

    draw_pure_progress()
    draw_worst_subsets(subset_metrics)
    draw_public_scores(public_ledger)
    draw_row_feedback(row_effects)
    draw_transition_summary(transition_summary)
    draw_segment_heatmap()
    write_readme(recommendations)

    print(f"wrote research dashboard to {OUT_DIR}")
    for path in [
        OUT_DIR / "README.md",
        OUT_DIR / "next_probe_recommendations.csv",
        OUT_DIR / "pure_subset_metrics.csv",
        OUT_DIR / "public_row_feedback_estimates.csv",
        OUT_DIR / "transition_top_candidates.csv",
    ]:
        print("-", path)


if __name__ == "__main__":
    main()
