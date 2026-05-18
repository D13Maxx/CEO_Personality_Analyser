"""Percentile band helpers for trait score display."""

from typing import Dict, Tuple
import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TRAITS, TRAIT_LABELS


def get_band_category(percentile: float) -> Tuple[str, str]:
    if percentile < 33:
        return "Low", "#EF4444"
    if percentile < 67:
        return "Average", "#F59E0B"
    return "High", "#10B981"


def create_percentile_table(scores: Dict[str, float], percentiles: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for t in TRAITS:
        band, color = get_band_category(percentiles.get(t, 0))
        rows.append({
            "Trait": TRAIT_LABELS[t],
            "Score (1-7)": round(scores.get(t, 0), 2),
            "Percentile": f"{int(round(percentiles.get(t, 0)))}th",
            "Category": band,
            "_color": color,
        })
    return pd.DataFrame(rows)


def style_percentile_dataframe(df: pd.DataFrame):
    def color_row(row):
        styles = [""] * len(row)
        idx = df.columns.get_loc("Category")
        styles[idx] = (f"color: white; background-color: {row['_color']}; "
                       "font-weight: bold; text-align: center;")
        return styles

    styler = df.style.apply(color_row, axis=1)
    styler.hide(axis="columns", subset=["_color"])
    styler.format({"Score (1-7)": "{:.2f}"})
    return styler
