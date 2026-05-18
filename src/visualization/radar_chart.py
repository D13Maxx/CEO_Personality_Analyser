"""Plotly radar chart for Big Five personality scores."""

from typing import Dict, Optional
import plotly.graph_objects as go

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TRAITS, TRAIT_LABELS, SCORE_MIN, SCORE_MAX, SCORE_MIDPOINT


def create_radar_chart(
    scores: Dict[str, float],
    comparison_scores: Optional[Dict[str, float]] = None,
    ceo_name: str = "CEO",
    comparison_name: str = "Sample Average",
) -> go.Figure:
    labels = [TRAIT_LABELS[t] for t in TRAITS]
    values = [scores.get(t, SCORE_MIDPOINT) for t in TRAITS]
    labels.append(labels[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labels, fill="toself", name=ceo_name,
        line_color="#3B82F6", fillcolor="rgba(59,130,246,0.4)",
        marker=dict(size=8, symbol="circle"),
    ))

    if comparison_scores:
        comp = [comparison_scores.get(t, SCORE_MIDPOINT) for t in TRAITS]
        comp.append(comp[0])
        fig.add_trace(go.Scatterpolar(
            r=comp, theta=labels, fill="none", name=comparison_name,
            line=dict(color="#64748B", dash="dash", width=2),
            marker=dict(size=6, symbol="square"),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[SCORE_MIN, SCORE_MAX],
                tickvals=[1, 2, 3, 4, 5, 6, 7],
                ticktext=["1 (Low)", "2", "3", "4 (Avg)", "5", "6", "7 (High)"],
                gridcolor="#E2E8F0", linecolor="#CBD5E1",
                tickangle=0, tickfont=dict(size=10, color="#64748B"),
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#1E293B", family="Inter, Helvetica, sans-serif"),
                rotation=90, direction="clockwise",
                gridcolor="#E2E8F0", linecolor="#CBD5E1",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=bool(comparison_scores),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
    )
    return fig
