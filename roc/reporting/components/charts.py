"""Chart factories built on Panel's Vega pane."""

from __future__ import annotations

from typing import Any

import panel as pn

from roc.reporting.components.tokens import ACCENT, FONT, TEXT_DIM, no_data_html


def event_bar_chart(
    event_data: dict[str, Any] | None,
) -> pn.pane.Vega | pn.pane.HTML:
    """Create a compact horizontal bar chart of event bus counts.

    Args:
        event_data: Dict mapping bus names to event counts. None returns placeholder.

    Returns:
        A Vega-Lite bar chart pane or an HTML placeholder.
    """
    if event_data is None:
        return pn.pane.HTML(no_data_html("event"), sizing_mode="stretch_width")

    values = [
        {"bus": str(k), "count": int(v)}
        for k, v in event_data.items()
        if k not in ("step", "game_number") and isinstance(v, (int, float))
    ]

    if not values:
        return pn.pane.HTML(no_data_html("event"), sizing_mode="stretch_width")

    spec: dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": "container",
        "height": max(len(values) * 18, 60),
        "autosize": {"type": "fit", "contains": "padding"},
        "data": {"values": values},
        "mark": {"type": "bar", "cornerRadiusEnd": 2},
        "encoding": {
            "y": {
                "field": "bus",
                "type": "nominal",
                "sort": "-x",
                "axis": {
                    "labelColor": TEXT_DIM,
                    "labelFont": FONT,
                    "labelFontSize": 10,
                    "title": None,
                    "domain": False,
                    "ticks": False,
                },
            },
            "x": {
                "field": "count",
                "type": "quantitative",
                "axis": {
                    "labelColor": TEXT_DIM,
                    "labelFont": FONT,
                    "labelFontSize": 9,
                    "title": None,
                    "grid": False,
                    "domain": False,
                    "ticks": False,
                },
            },
            "color": {"value": ACCENT},
        },
        "config": {
            "view": {"stroke": "transparent"},
            "background": "transparent",
        },
    }

    return pn.pane.Vega(spec, sizing_mode="stretch_width")
