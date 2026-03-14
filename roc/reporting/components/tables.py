"""Compact table factories built on Panel's Tabulator widget."""

from __future__ import annotations

from typing import Any

import pandas as pd
import panel as pn

from roc.reporting.components.tokens import (
    COMPACT_TABLE_CSS,
    TEXT_MUTED,
    no_data_html,
)

#: Severity level numbers for log filtering.
_LEVEL_NUMBERS: dict[str, int] = {
    "DEBUG": 5,
    "INFO": 9,
    "WARN": 13,
    "WARNING": 13,
    "ERROR": 17,
}

#: Severity level colors for log display.
_LEVEL_COLORS: dict[str, str] = {
    "DEBUG": "#484f58",
    "INFO": "#c9d1d9",
    "WARN": "#d29922",
    "WARNING": "#d29922",
    "ERROR": "#f85149",
}


def compact_kv_table(
    data: dict[str, Any] | None,
    title: str = "",
    max_width: int = 320,
) -> pn.widgets.Tabulator | pn.pane.HTML:
    """Create a compact key-value Tabulator from a dict.

    Args:
        data: Key-value dict to display. None returns a placeholder.
        title: Label for the "no data" placeholder.
        max_width: Maximum table width in pixels.

    Returns:
        A styled Tabulator widget or an HTML placeholder.
    """
    if data is None:
        return pn.pane.HTML(no_data_html(title), sizing_mode="stretch_width")

    # Filter out 'raw' key and truncate long values
    rows = []
    for k, v in data.items():
        if k == "raw":
            continue
        val_str = str(v)
        if len(val_str) > 80:
            val_str = val_str[:77] + "..."
        rows.append({"key": str(k), "value": val_str})

    if not rows:
        return pn.pane.HTML(no_data_html(title), sizing_mode="stretch_width")

    df = pd.DataFrame(rows)
    row_height = 20
    table_height = min(len(rows) * row_height + 4, 400)
    return pn.widgets.Tabulator(
        df,
        theme="simple",
        theme_classes=["table-sm"],
        show_index=False,
        header_filters=False,
        stylesheets=[COMPACT_TABLE_CSS],
        sizing_mode="fixed",
        width=max_width,
        height=table_height,
        disabled=True,
        pagination=None,
    )


def compact_log_table(
    logs: list[dict[str, Any]] | None,
    min_level: str = "DEBUG",
) -> pn.widgets.Tabulator | pn.pane.HTML:
    """Create a compact log table with severity coloring.

    Args:
        logs: List of log record dicts with severity_text, severity_number, body.
        min_level: Minimum severity level to include.

    Returns:
        A styled Tabulator widget or an HTML placeholder.
    """
    if logs is None:
        return pn.pane.HTML(no_data_html("log"), sizing_mode="stretch_width")

    min_num = _LEVEL_NUMBERS.get(min_level, 0)
    rows = []
    for log in logs:
        severity = log.get("severity_number")
        if severity is not None and severity < min_num:
            continue
        level = log.get("severity_text", "?")
        body = log.get("body", "")
        color = _LEVEL_COLORS.get(level, "#c9d1d9")
        rows.append({"level": level, "message": str(body), "_color": color})

    if not rows:
        return pn.pane.HTML(no_data_html("log messages at this level"), sizing_mode="stretch_width")

    df = pd.DataFrame(rows)

    # Tabulator formatters for severity coloring
    from bokeh.models.widgets.tables import HTMLTemplateFormatter

    level_fmt = HTMLTemplateFormatter(
        template='<span style="color:<%= _color %>;font-weight:500;">[<%= value %>]</span>'
    )
    msg_fmt = HTMLTemplateFormatter(
        template='<span style="color:<%= _color %>;"><%= value %></span>'
    )

    log_css = (
        COMPACT_TABLE_CSS
        + f"""
:host .tabulator-row {{
    border-bottom: none;
}}
:host .tabulator-cell {{
    padding: 1px 4px;
    white-space: normal;
    color: {TEXT_MUTED};
}}
"""
    )

    return pn.widgets.Tabulator(
        df,
        theme="simple",
        theme_classes=["table-sm"],
        show_index=False,
        header_filters=False,
        stylesheets=[log_css],
        sizing_mode="stretch_width",
        height=200,
        disabled=True,
        pagination=None,
        formatters={"level": level_fmt, "message": msg_fmt},
        hidden_columns=["_color"],
    )
