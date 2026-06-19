from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "artifacts" / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.append(str(ROOT))

from src.stellar_features import add_advanced_features  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "oof_generalization_diagnostics"
CLASSES = ["GALAXY", "QSO", "STAR"]
CLASS_TO_IDX = {label: idx for idx, label in enumerate(CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visual diagnostics for an OOF/CV generalization candidate versus a base stacker."
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--base-name", default="07_lr_v9")
    parser.add_argument("--candidate-name", default="18_oof_bias_realmlp0")
    parser.add_argument("--base-oof", type=Path, default=ROOT / "external_sources/oof_test_predictions/oof_lr_stacker_v9.npy")
    parser.add_argument("--base-test", type=Path, default=ROOT / "external_preds/pred_lr_stacker_v9.npy")
    parser.add_argument("--candidate-oof", type=Path, default=ARTIFACTS / "oof_generalization_stack/generalization_stack_oof.npy")
    parser.add_argument("--candidate-test", type=Path, default=ARTIFACTS / "oof_generalization_stack/generalization_stack_test.npy")
    parser.add_argument("--base-public-score", type=float, default=None)
    parser.add_argument("--candidate-public-score", type=float, default=None)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def load_proba(path: Path, expected_rows: int) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != (expected_rows, len(CLASSES)):
        raise ValueError(f"{path} shape {arr.shape}, expected {(expected_rows, len(CLASSES))}")
    row_sum = arr.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return arr / row_sum


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {}
    for idx, label in enumerate(CLASSES):
        mask = y_true == idx
        out[label] = float((y_pred[mask] == idx).mean()) if mask.any() else float("nan")
    return out


def confidence_margin(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(proba, axis=1)
    return ordered[:, -1], ordered[:, -1] - ordered[:, -2]


def transition_frame(before: np.ndarray, after: np.ndarray, prefix: str) -> pd.DataFrame:
    changed = before != after
    counts = Counter(
        f"{CLASSES[int(b)]}->{CLASSES[int(a)]}"
        for b, a in zip(before[changed], after[changed])
    )
    return pd.DataFrame(
        [{"scope": prefix, "transition": key, "count": value} for key, value in sorted(counts.items())]
    )


def changed_outcome_frame(y: np.ndarray, base_pred: np.ndarray, cand_pred: np.ndarray) -> pd.DataFrame:
    changed = base_pred != cand_pred
    rows = []
    for true, base, cand in zip(y[changed], base_pred[changed], cand_pred[changed]):
        base_ok = base == true
        cand_ok = cand == true
        if not base_ok and cand_ok:
            outcome = "fixed"
        elif base_ok and not cand_ok:
            outcome = "broken"
        elif not base_ok and not cand_ok:
            outcome = "still_wrong"
        else:
            outcome = "both_right_changed"
        rows.append(
            {
                "true": CLASSES[int(true)],
                "base": CLASSES[int(base)],
                "candidate": CLASSES[int(cand)],
                "transition": f"{CLASSES[int(base)]}->{CLASSES[int(cand)]}",
                "outcome": outcome,
            }
        )
    return pd.DataFrame(rows)


def add_bins(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    out = frame.copy()
    for col in ["redshift", "g-i", "mag_range", "u-r"]:
        out[f"{col}_bin"] = pd.qcut(out[col], q=bins, labels=False, duplicates="drop")
    out["spectral_population"] = out["spectral_type"].astype(str) + "_" + out["galaxy_population"].astype(str)
    return out


def subset_metrics(frame: pd.DataFrame, y: np.ndarray, base_pred: np.ndarray, cand_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    specs = [
        ("spectral_type", "spectral_type"),
        ("galaxy_population", "galaxy_population"),
        ("spectral_population", "spectral_population"),
        ("redshift_bin", "redshift_bin"),
        ("g_i_bin", "g-i_bin"),
        ("mag_range_bin", "mag_range_bin"),
        ("u_r_bin", "u-r_bin"),
    ]
    frame = frame.copy()
    frame["y"] = y
    frame["base_pred"] = base_pred
    frame["candidate_pred"] = cand_pred
    frame["base_correct"] = frame["base_pred"].eq(frame["y"])
    frame["candidate_correct"] = frame["candidate_pred"].eq(frame["y"])
    for group_name, col in specs:
        if col not in frame.columns:
            continue
        for value, group in frame.groupby(col, observed=True):
            if len(group) < 300:
                continue
            idx = group.index.to_numpy()
            y_g = y[idx]
            base_g = base_pred[idx]
            cand_g = cand_pred[idx]
            support = {label: int((y_g == class_idx).sum()) for class_idx, label in enumerate(CLASSES)}
            min_support = min(support.values())
            base_bac = balanced_accuracy_score(y_g, base_g)
            cand_bac = balanced_accuracy_score(y_g, cand_g)
            rows.append(
                {
                    "group": group_name,
                    "value": str(value),
                    "count": int(len(group)),
                    "min_class_support": int(min_support),
                    "base_bac": float(base_bac),
                    "candidate_bac": float(cand_bac),
                    "delta_bac": float(cand_bac - base_bac),
                    "base_acc": float((base_g == y_g).mean()),
                    "candidate_acc": float((cand_g == y_g).mean()),
                    "delta_acc": float((cand_g == y_g).mean() - (base_g == y_g).mean()),
                    "changed_rows": int((base_g != cand_g).sum()),
                    **{f"support_{k}": v for k, v in support.items()},
                }
            )
    return pd.DataFrame(rows).sort_values(["delta_bac", "count"], ascending=[True, False])


def style_axes(ax) -> None:
    ax.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#d0d5dd")


def save_class_recall_plot(metrics: dict, path: Path) -> None:
    x = np.arange(len(CLASSES))
    w = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    base_vals = [metrics["base_recalls"][cls] for cls in CLASSES]
    cand_vals = [metrics["candidate_recalls"][cls] for cls in CLASSES]
    ax.bar(x - w / 2, base_vals, width=w, label=metrics["base_name"], color="#2f80ed")
    ax.bar(x + w / 2, cand_vals, width=w, label=metrics["candidate_name"], color="#eb5757")
    for xi, b, c in zip(x, base_vals, cand_vals):
        ax.text(xi - w / 2, b + 0.001, f"{b:.5f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + w / 2, c + 0.001, f"{c:.5f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi, min(b, c) - 0.004, f"{c-b:+.5f}", ha="center", va="top", fontsize=9, color="#344054")
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(min(min(base_vals), min(cand_vals)) - 0.012, max(max(base_vals), max(cand_vals)) + 0.012)
    ax.set_title("OOF Class Recall: Base vs Candidate", fontsize=14, weight="bold")
    ax.set_ylabel("Recall")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_confusion_delta(cm_base: np.ndarray, cm_cand: np.ndarray, path: Path) -> None:
    delta = cm_cand - cm_base
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=160)
    vmax = max(1, np.abs(delta).max())
    im = ax.imshow(delta, cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(CLASSES)), CLASSES)
    ax.set_yticks(np.arange(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("OOF Confusion Matrix Delta: Candidate - Base", fontsize=13, weight="bold")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{delta[i, j]:+d}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.82, label="count delta")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_transition_plot(transitions: pd.DataFrame, path: Path, title: str) -> None:
    if transitions.empty:
        return
    view = transitions.sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, len(view) * 0.42)), dpi=160)
    colors = np.where(view["scope"].eq("oof"), "#2f80ed", "#f2994a")
    ax.barh(view["scope"] + " / " + view["transition"], view["count"], color=colors)
    for y, v in enumerate(view["count"]):
        ax.text(v + max(view["count"].max() * 0.01, 1), y, f"{int(v)}", va="center", fontsize=9)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Changed rows")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_outcome_plot(outcomes: pd.DataFrame, path: Path) -> None:
    if outcomes.empty:
        return
    counts = outcomes["outcome"].value_counts().reindex(["fixed", "broken", "still_wrong", "both_right_changed"]).fillna(0)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    colors = ["#27ae60", "#d92d20", "#f2994a", "#667085"]
    ax.bar(counts.index, counts.values, color=colors)
    for x, v in enumerate(counts.values):
        ax.text(x, v + max(counts.max() * 0.015, 1), f"{int(v)}", ha="center", fontsize=10)
    ax.set_title("OOF Changed Rows Outcome", fontsize=13, weight="bold")
    ax.set_ylabel("Rows")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_score_plot(metrics: dict, path: Path) -> None:
    labels = ["OOF BAC"]
    base_vals = [metrics["base_oof_bac"]]
    cand_vals = [metrics["candidate_oof_bac"]]
    if metrics.get("base_public_score") is not None and metrics.get("candidate_public_score") is not None:
        labels.append("Public LB")
        base_vals.append(metrics["base_public_score"])
        cand_vals.append(metrics["candidate_public_score"])
    x = np.arange(len(labels))
    w = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=160)
    ax.bar(x - w / 2, base_vals, width=w, color="#2f80ed", label=metrics["base_name"])
    ax.bar(x + w / 2, cand_vals, width=w, color="#eb5757", label=metrics["candidate_name"])
    for xi, b, c in zip(x, base_vals, cand_vals):
        ax.text(xi - w / 2, b + 0.00005, f"{b:.5f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, c + 0.00005, f"{c:.5f}", ha="center", fontsize=9)
        ax.text(xi, min(b, c) - 0.00009, f"{c-b:+.5f}", ha="center", fontsize=10, color="#344054")
    ax.set_xticks(x, labels)
    ax.set_title("OOF Improvement vs Public Response", fontsize=13, weight="bold")
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    lower = min(min(base_vals), min(cand_vals)) - 0.00035
    upper = max(max(base_vals), max(cand_vals)) + 0.00035
    ax.set_ylim(lower, upper)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_feature_map(frame: pd.DataFrame, value_col: str, title: str, path: Path) -> None:
    pivot = frame.pivot_table(index="redshift_bin", columns="g-i_bin", values=value_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9.2, 6.8), dpi=160)
    vmax = max(abs(float(np.nanmin(pivot.to_numpy()))), abs(float(np.nanmax(pivot.to_numpy()))), 1e-9)
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu", vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("g-i quantile bin")
    ax.set_ylabel("redshift quantile bin")
    ax.set_xticks(np.arange(len(pivot.columns)), [str(int(c)) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), [str(int(i)) for i in pivot.index])
    fig.colorbar(im, ax=ax, shrink=0.84, label=value_col)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_subset_delta_plot(subsets: pd.DataFrame, path: Path) -> None:
    view = subsets[subsets["min_class_support"].ge(100)].copy()
    if view.empty:
        view = subsets.copy()
    worst = view.nsmallest(12, "delta_bac")
    best = view.nlargest(12, "delta_bac")
    plot_df = pd.concat([worst, best]).drop_duplicates()
    plot_df["label"] = plot_df["group"] + "=" + plot_df["value"]
    plot_df = plot_df.sort_values("delta_bac")
    fig, ax = plt.subplots(figsize=(11.2, max(5.5, len(plot_df) * 0.32)), dpi=160)
    colors = np.where(plot_df["delta_bac"] >= 0, "#27ae60", "#d92d20")
    ax.barh(plot_df["label"], plot_df["delta_bac"], color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    for y, v in enumerate(plot_df["delta_bac"]):
        ax.text(v + (0.00003 if v >= 0 else -0.00003), y, f"{v:+.5f}", ha="left" if v >= 0 else "right", va="center", fontsize=8)
    ax.set_title("OOF Subset BAC Delta: Candidate - Base", fontsize=13, weight="bold")
    ax.set_xlabel("Balanced accuracy delta")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_interactive_dashboard(
    metrics: dict,
    cm_base: np.ndarray,
    cm_cand: np.ndarray,
    transitions: pd.DataFrame,
    outcomes: pd.DataFrame,
    subsets: pd.DataFrame,
    grid: pd.DataFrame,
    test_grid: pd.DataFrame,
    path: Path,
) -> None:
    confusion_delta = cm_cand - cm_base
    subset_view = subsets[subsets["min_class_support"].ge(100)].copy()
    if subset_view.empty:
        subset_view = subsets.copy()
    subset_view = pd.concat(
        [subset_view.nsmallest(14, "delta_bac"), subset_view.nlargest(14, "delta_bac")]
    ).drop_duplicates()
    subset_view["label"] = subset_view["group"] + "=" + subset_view["value"]
    subset_view = subset_view.sort_values("delta_bac")

    transition_view = transitions.copy()
    transition_view["label"] = transition_view["scope"] + " / " + transition_view["transition"]
    transition_view = transition_view.sort_values("count")

    outcome_counts = outcomes["outcome"].value_counts().reindex(["fixed", "broken", "still_wrong", "both_right_changed"]).fillna(0)

    oof_accuracy_pivot = grid.pivot_table(index="redshift_bin", columns="g-i_bin", values="accuracy_delta", aggfunc="mean")
    oof_change_pivot = grid.pivot_table(index="redshift_bin", columns="g-i_bin", values="changed_rate", aggfunc="mean")
    test_change_pivot = test_grid.pivot_table(index="redshift_bin", columns="g-i_bin", values="changed_rate", aggfunc="mean")

    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "table"}],
        ],
        subplot_titles=[
            "OOF/Public Score",
            "OOF Class Recall",
            "OOF Confusion Delta",
            "Prediction Transitions",
            "Changed OOF Row Outcomes",
            "OOF Accuracy Delta by redshift and g-i",
            "Test Changed-Row Density by redshift and g-i",
            "Worst OOF Subsets",
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.12,
    )

    score_labels = ["OOF BAC"]
    base_scores = [metrics["base_oof_bac"]]
    cand_scores = [metrics["candidate_oof_bac"]]
    if metrics.get("base_public_score") is not None and metrics.get("candidate_public_score") is not None:
        score_labels.append("Public LB")
        base_scores.append(metrics["base_public_score"])
        cand_scores.append(metrics["candidate_public_score"])
    fig.add_trace(go.Bar(name=metrics["base_name"], x=score_labels, y=base_scores, marker_color="#2f80ed", text=[f"{v:.6f}" for v in base_scores], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(name=metrics["candidate_name"], x=score_labels, y=cand_scores, marker_color="#eb5757", text=[f"{v:.6f}" for v in cand_scores], textposition="outside"), row=1, col=1)

    base_recalls = [metrics["base_recalls"][cls] for cls in CLASSES]
    cand_recalls = [metrics["candidate_recalls"][cls] for cls in CLASSES]
    fig.add_trace(go.Bar(name=f"{metrics['base_name']} recall", x=CLASSES, y=base_recalls, marker_color="#56ccf2", text=[f"{v:.5f}" for v in base_recalls], textposition="outside", showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(name=f"{metrics['candidate_name']} recall", x=CLASSES, y=cand_recalls, marker_color="#f2994a", text=[f"{v:.5f}" for v in cand_recalls], textposition="outside", showlegend=False), row=1, col=2)

    vmax = max(1, int(np.abs(confusion_delta).max()))
    fig.add_trace(
        go.Heatmap(
            z=confusion_delta,
            x=CLASSES,
            y=CLASSES,
            colorscale="RdBu",
            zmin=-vmax,
            zmax=vmax,
            text=confusion_delta,
            texttemplate="%{text:+d}",
            colorbar=dict(title="count", len=0.22, y=0.70),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(go.Bar(x=transition_view["count"], y=transition_view["label"], orientation="h", marker_color=np.where(transition_view["scope"].eq("oof"), "#2f80ed", "#f2994a"), text=transition_view["count"], textposition="outside", showlegend=False), row=2, col=2)

    fig.add_trace(go.Bar(x=outcome_counts.index, y=outcome_counts.values, marker_color=["#27ae60", "#d92d20", "#f2994a", "#667085"], text=[int(v) for v in outcome_counts.values], textposition="outside", showlegend=False), row=3, col=1)

    acc_vmax = max(abs(float(np.nanmin(oof_accuracy_pivot.to_numpy()))), abs(float(np.nanmax(oof_accuracy_pivot.to_numpy()))), 1e-9)
    fig.add_trace(
        go.Heatmap(
            z=oof_accuracy_pivot.to_numpy(),
            x=[str(int(c)) for c in oof_accuracy_pivot.columns],
            y=[str(int(i)) for i in oof_accuracy_pivot.index],
            colorscale="RdBu",
            zmin=-acc_vmax,
            zmax=acc_vmax,
            colorbar=dict(title="delta", len=0.22, y=0.42),
        ),
        row=3,
        col=2,
    )

    change_vmax = max(float(np.nanmax(oof_change_pivot.to_numpy())), float(np.nanmax(test_change_pivot.to_numpy())), 1e-9)
    fig.add_trace(
        go.Heatmap(
            z=test_change_pivot.to_numpy(),
            x=[str(int(c)) for c in test_change_pivot.columns],
            y=[str(int(i)) for i in test_change_pivot.index],
            colorscale="Viridis",
            zmin=0,
            zmax=change_vmax,
            colorbar=dict(title="changed", len=0.22, y=0.15),
        ),
        row=4,
        col=1,
    )

    worst_table = subsets[subsets["min_class_support"].ge(100)].nsmallest(12, "delta_bac").copy()
    fig.add_trace(
        go.Table(
            header=dict(values=["group", "value", "count", "delta_bac", "changed"], fill_color="#111827", font=dict(color="white"), align="left"),
            cells=dict(
                values=[
                    worst_table["group"],
                    worst_table["value"],
                    worst_table["count"],
                    worst_table["delta_bac"].map(lambda x: f"{x:+.5f}"),
                    worst_table["changed_rows"],
                ],
                fill_color="#f9fafb",
                align="left",
                height=24,
            ),
        ),
        row=4,
        col=2,
    )

    fig.update_layout(
        title=f"OOF Generalization Diagnostics: {metrics['base_name']} vs {metrics['candidate_name']}",
        height=1800,
        width=1450,
        template="plotly_white",
        barmode="group",
        margin=dict(l=70, r=50, t=90, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["class"].map(CLASS_TO_IDX).to_numpy()
    train_fe = add_bins(add_advanced_features(train), args.bins)
    test_fe = add_bins(add_advanced_features(test), args.bins)

    base_oof = load_proba(args.base_oof, len(train))
    cand_oof = load_proba(args.candidate_oof, len(train))
    base_test = load_proba(args.base_test, len(test))
    cand_test = load_proba(args.candidate_test, len(test))
    base_pred = base_oof.argmax(axis=1)
    cand_pred = cand_oof.argmax(axis=1)
    base_test_pred = base_test.argmax(axis=1)
    cand_test_pred = cand_test.argmax(axis=1)

    metrics = {
        "base_name": args.base_name,
        "candidate_name": args.candidate_name,
        "base_oof_bac": float(balanced_accuracy_score(y, base_pred)),
        "candidate_oof_bac": float(balanced_accuracy_score(y, cand_pred)),
        "base_public_score": args.base_public_score,
        "candidate_public_score": args.candidate_public_score,
        "base_recalls": class_recalls(y, base_pred),
        "candidate_recalls": class_recalls(y, cand_pred),
        "oof_changed_rows": int((base_pred != cand_pred).sum()),
        "test_changed_rows": int((base_test_pred != cand_test_pred).sum()),
        "base_oof_pred_share": {cls: float((base_pred == idx).mean()) for idx, cls in enumerate(CLASSES)},
        "candidate_oof_pred_share": {cls: float((cand_pred == idx).mean()) for idx, cls in enumerate(CLASSES)},
        "base_test_pred_share": {cls: float((base_test_pred == idx).mean()) for idx, cls in enumerate(CLASSES)},
        "candidate_test_pred_share": {cls: float((cand_test_pred == idx).mean()) for idx, cls in enumerate(CLASSES)},
    }
    metrics["oof_delta"] = metrics["candidate_oof_bac"] - metrics["base_oof_bac"]
    if args.base_public_score is not None and args.candidate_public_score is not None:
        metrics["public_delta"] = args.candidate_public_score - args.base_public_score

    cm_base = confusion_matrix(y, base_pred, labels=[0, 1, 2])
    cm_cand = confusion_matrix(y, cand_pred, labels=[0, 1, 2])
    pd.DataFrame(cm_base, index=CLASSES, columns=CLASSES).to_csv(args.output_dir / "base_oof_confusion_matrix.csv")
    pd.DataFrame(cm_cand, index=CLASSES, columns=CLASSES).to_csv(args.output_dir / "candidate_oof_confusion_matrix.csv")
    pd.DataFrame(cm_cand - cm_base, index=CLASSES, columns=CLASSES).to_csv(args.output_dir / "oof_confusion_delta.csv")

    transitions = pd.concat(
        [
            transition_frame(base_pred, cand_pred, "oof"),
            transition_frame(base_test_pred, cand_test_pred, "test"),
        ],
        ignore_index=True,
    )
    transitions.to_csv(args.output_dir / "transition_counts.csv", index=False)
    outcomes = changed_outcome_frame(y, base_pred, cand_pred)
    outcomes.to_csv(args.output_dir / "oof_changed_row_outcomes.csv", index=False)

    subsets = subset_metrics(train_fe, y, base_pred, cand_pred)
    subsets.to_csv(args.output_dir / "subset_delta_metrics.csv", index=False)

    train_map = train_fe[["redshift_bin", "g-i_bin"]].copy()
    train_map["base_correct"] = base_pred == y
    train_map["candidate_correct"] = cand_pred == y
    train_map["accuracy_delta"] = train_map["candidate_correct"].astype(int) - train_map["base_correct"].astype(int)
    train_map["changed"] = base_pred != cand_pred
    grid = (
        train_map.groupby(["redshift_bin", "g-i_bin"], observed=True)
        .agg(
            count=("changed", "size"),
            changed_rate=("changed", "mean"),
            accuracy_delta=("accuracy_delta", "mean"),
        )
        .reset_index()
    )
    grid.to_csv(args.output_dir / "redshift_g_i_oof_grid.csv", index=False)

    test_map = test_fe[["redshift_bin", "g-i_bin"]].copy()
    test_map["changed"] = base_test_pred != cand_test_pred
    test_grid = (
        test_map.groupby(["redshift_bin", "g-i_bin"], observed=True)
        .agg(count=("changed", "size"), changed_rate=("changed", "mean"))
        .reset_index()
    )
    test_grid.to_csv(args.output_dir / "redshift_g_i_test_change_grid.csv", index=False)

    save_score_plot(metrics, args.output_dir / "score_oof_vs_public.svg")
    save_class_recall_plot(metrics, args.output_dir / "class_recall_comparison.svg")
    save_confusion_delta(cm_base, cm_cand, args.output_dir / "confusion_delta_heatmap.svg")
    save_transition_plot(transitions, args.output_dir / "transition_counts.svg", "Prediction Transitions: Base -> Candidate")
    save_outcome_plot(outcomes, args.output_dir / "changed_row_outcomes.svg")
    save_subset_delta_plot(subsets, args.output_dir / "subset_delta_bac.svg")
    save_feature_map(grid, "accuracy_delta", "OOF Accuracy Delta Map by redshift and g-i", args.output_dir / "redshift_g_i_oof_accuracy_delta_map.svg")
    save_feature_map(grid, "changed_rate", "OOF Changed-Row Density by redshift and g-i", args.output_dir / "redshift_g_i_oof_changed_density_map.svg")
    save_feature_map(test_grid, "changed_rate", "Test Changed-Row Density by redshift and g-i", args.output_dir / "redshift_g_i_test_changed_density_map.svg")
    save_interactive_dashboard(
        metrics,
        cm_base,
        cm_cand,
        transitions,
        outcomes,
        subsets,
        grid,
        test_grid,
        args.output_dir / "interactive_dashboard.html",
    )

    report = {
        "purpose": "Diagnostics for why an OOF-improved candidate may diverge from public LB behavior.",
        "metrics": metrics,
        "top_bad_subsets": subsets[subsets["min_class_support"].ge(100)].nsmallest(10, "delta_bac").to_dict(orient="records"),
        "top_good_subsets": subsets[subsets["min_class_support"].ge(100)].nlargest(10, "delta_bac").to_dict(orient="records"),
        "outputs": [
            "score_oof_vs_public.svg",
            "class_recall_comparison.svg",
            "confusion_delta_heatmap.svg",
            "transition_counts.svg",
            "changed_row_outcomes.svg",
            "subset_delta_bac.svg",
            "redshift_g_i_oof_accuracy_delta_map.svg",
            "redshift_g_i_oof_changed_density_map.svg",
            "redshift_g_i_test_changed_density_map.svg",
            "interactive_dashboard.html",
            "subset_delta_metrics.csv",
            "transition_counts.csv",
            "oof_changed_row_outcomes.csv",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
