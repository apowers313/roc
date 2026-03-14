"""Status bar factory using pn.indicators.Number or HTML fallback."""

from __future__ import annotations

import html as html_mod
from typing import Any

import panel as pn

from roc.reporting.components.tokens import (
    BORDER,
    ERROR,
    FONT,
    SUCCESS,
    SURFACE,
    TEXT,
    TEXT_DIM,
    WARNING,
)


def _stat_badge(label: str, value: Any, color: str = TEXT) -> str:
    """Render a compact inline stat badge."""
    return (
        f'<span style="margin-right:12px;font-size:11px;font-family:{FONT};">'
        f'<span style="color:{TEXT_DIM};">{html_mod.escape(str(label))}</span> '
        f'<span style="color:{color};font-weight:500;">'
        f"{html_mod.escape(str(value))}</span></span>"
    )


def compact_status_bar(
    metrics: dict[str, Any] | None,
    step: int = 0,
    game_number: int = 0,
) -> pn.pane.HTML:
    """Create a compact status bar showing key game metrics.

    Args:
        metrics: Game metrics dict (hp, hp_max, score, depth, etc.).
        step: Current step number (fallback if no metrics).
        game_number: Current game number (fallback if no metrics).

    Returns:
        An HTML pane with inline metric badges.
    """
    badges: list[str] = []
    if metrics:
        hp = metrics.get("hp", "?")
        hp_max = metrics.get("hp_max", "?")
        hp_color = SUCCESS
        if isinstance(hp, (int, float)) and isinstance(hp_max, (int, float)) and hp_max > 0:
            ratio = hp / hp_max
            if ratio < 0.25:
                hp_color = ERROR
            elif ratio < 0.5:
                hp_color = WARNING
        badges.append(_stat_badge("HP", f"{hp}/{hp_max}", hp_color))
        badges.append(_stat_badge("Score", metrics.get("score", "?")))
        badges.append(_stat_badge("Depth", metrics.get("depth", "?")))
        badges.append(_stat_badge("Gold", metrics.get("gold", "?")))
        badges.append(
            _stat_badge(
                "Energy",
                f"{metrics.get('energy', '?')}/{metrics.get('energy_max', '?')}",
            )
        )
        hunger = metrics.get("hunger", "?")
        if isinstance(hunger, (int, float)):
            hunger = "OK" if hunger == 0 else str(hunger)
        badges.append(_stat_badge("Hunger", hunger))
    else:
        badges.append(_stat_badge("Step", step))
        badges.append(_stat_badge("Game", game_number))

    html = (
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;'
        f"padding:4px 8px;background:{SURFACE};border:1px solid {BORDER};"
        f'border-radius:4px;font-family:{FONT};">'
        f"{''.join(badges)}</div>"
    )
    return pn.pane.HTML(html, sizing_mode="stretch_width")
