from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts" / "oof_generalization_stack"
CLASSES = ["GALAXY", "QSO", "STAR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an interactive dashboard for OOF stack optimizer traces.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def parse_bias(text: object) -> dict[str, float]:
    if isinstance(text, dict):
        return {k: float(v) for k, v in text.items()}
    try:
        value = ast.literal_eval(str(text))
    except (SyntaxError, ValueError):
        return {label: float("nan") for label in CLASSES}
    return {label: float(value.get(label, float("nan"))) for label in CLASSES}


def add_bias_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    parsed = out["bias"].map(parse_bias)
    for label in CLASSES:
        out[f"bias_{label}"] = parsed.map(lambda row: row[label])
    out["bias_text"] = parsed.map(lambda row: "<br>".join(f"{label}: {row[label]:.6f}" for label in CLASSES))
    return out


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    output = args.output or input_dir / "optimizer_dashboard.html"
    output = output if output.is_absolute() else ROOT / output

    bias = add_bias_columns(pd.read_csv(input_dir / "base_bias_search.csv"))
    blend = add_bias_columns(pd.read_csv(input_dir / "blend_search.csv"))
    stages = add_bias_columns(pd.read_csv(input_dir / "accepted_stages.csv"))
    report = json.loads((input_dir / "report.json").read_text(encoding="utf-8"))

    best_bias = bias.loc[bias["score"].idxmax()]
    best_blend = blend.loc[blend["score"].idxmax()]
    best_by_candidate = (
        blend.sort_values("score", ascending=False)
        .groupby("candidate", as_index=False)
        .first()
        .sort_values("score", ascending=True)
    )

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "table"}],
        ],
        subplot_titles=[
            "Base Class-Bias Search",
            "Accepted OOF Stages",
            "Blend Weight Search by Candidate",
            "Best Blend per Candidate",
            "Final Bias Values",
            "Best Settings",
        ],
        vertical_spacing=0.11,
        horizontal_spacing=0.13,
    )

    for label in CLASSES:
        sub = bias[bias["class"].eq(label)].sort_values(["round", "multiplier"])
        fig.add_trace(
            go.Scatter(
                x=sub["multiplier"],
                y=sub["score"],
                mode="markers",
                name=f"bias {label}",
                customdata=sub[["round", "bias_text"]],
                hovertemplate="class=%{fullData.name}<br>round=%{customdata[0]}<br>multiplier=%{x:.4f}<br>OOF=%{y:.9f}<br>%{customdata[1]}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=[best_bias["multiplier"]],
            y=[best_bias["score"]],
            mode="markers+text",
            marker=dict(size=14, color="#d92d20", symbol="star"),
            text=["best"],
            textposition="top center",
            name="best base bias",
            hovertemplate="best base bias<br>OOF=%{y:.9f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=stages["stage"],
            y=stages["score"],
            marker_color=["#667085", "#2f80ed", "#27ae60"][: len(stages)],
            text=stages["score"].map(lambda x: f"{x:.9f}"),
            textposition="outside",
            hovertext=stages["bias_text"],
            hovertemplate="stage=%{x}<br>OOF=%{y:.9f}<br>%{hovertext}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    for candidate, sub in blend.groupby("candidate"):
        sub = sub.sort_values("weight")
        fig.add_trace(
            go.Scatter(
                x=sub["weight"],
                y=sub["score"],
                mode="lines+markers",
                name=f"blend {candidate}",
                customdata=sub[["round", "delta_vs_current", "bias_text"]],
                hovertemplate="candidate=%{fullData.name}<br>round=%{customdata[0]}<br>weight=%{x:.4f}<br>OOF=%{y:.9f}<br>delta_vs_current=%{customdata[1]:+.9f}<br>%{customdata[2]}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=[best_blend["weight"]],
            y=[best_blend["score"]],
            mode="markers+text",
            marker=dict(size=14, color="#d92d20", symbol="star"),
            text=["best"],
            textposition="top center",
            name="best blend",
            hovertemplate="best blend<br>candidate=%{text}<br>weight=%{x:.4f}<br>OOF=%{y:.9f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=best_by_candidate["score"],
            y=best_by_candidate["candidate"],
            orientation="h",
            marker_color="#56ccf2",
            text=best_by_candidate["score"].map(lambda x: f"{x:.9f}"),
            textposition="outside",
            customdata=best_by_candidate[["weight", "delta_vs_current", "bias_text"]],
            hovertemplate="candidate=%{y}<br>best OOF=%{x:.9f}<br>weight=%{customdata[0]:.4f}<br>delta_vs_current=%{customdata[1]:+.9f}<br>%{customdata[2]}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    final_stage = stages.iloc[-1]
    fig.add_trace(
        go.Bar(
            x=CLASSES,
            y=[final_stage[f"bias_{label}"] for label in CLASSES],
            marker_color=["#2f80ed", "#9b51e0", "#f2994a"],
            text=[f"{final_stage[f'bias_{label}']:.6f}" for label in CLASSES],
            textposition="outside",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=1.0, line_width=1, line_dash="dash", line_color="#667085", row=3, col=1)

    settings = [
        ("base_model", report["base_model"]),
        ("raw_base_oof", f"{report['raw_base_oof_balanced_accuracy']:.9f}"),
        ("best_oof", f"{report['best_oof_balanced_accuracy']:.9f}"),
        ("delta_vs_raw", f"{report['delta_vs_raw_base']:+.9f}"),
        ("accepted_model", str(final_stage["model"])),
        ("accepted_weight", f"{float(final_stage['weight']):.4f}"),
        ("bias_GALAXY", f"{final_stage['bias_GALAXY']:.6f}"),
        ("bias_QSO", f"{final_stage['bias_QSO']:.6f}"),
        ("bias_STAR", f"{final_stage['bias_STAR']:.6f}"),
    ]
    fig.add_trace(
        go.Table(
            header=dict(values=["setting", "value"], fill_color="#111827", font=dict(color="white"), align="left"),
            cells=dict(values=[[k for k, _ in settings], [v for _, v in settings]], fill_color="#f9fafb", align="left", height=28),
        ),
        row=3,
        col=2,
    )

    fig.update_layout(
        title="OOF Generalization Stack Optimizer Trace",
        height=1450,
        width=1450,
        template="plotly_white",
        margin=dict(l=70, r=70, t=90, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.write_html(output, include_plotlyjs="cdn")

    summary = {
        "dashboard": str(output.relative_to(ROOT)),
        "best_base_bias_score": float(best_bias["score"]),
        "best_blend_candidate": str(best_blend["candidate"]),
        "best_blend_weight": float(best_blend["weight"]),
        "best_blend_score": float(best_blend["score"]),
        "final_stage": final_stage.to_dict(),
    }
    (input_dir / "optimizer_dashboard_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
