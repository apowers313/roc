"""GridViewer -- reactive Viewer for colored monospace character grids."""

from __future__ import annotations

from typing import Any

import param
import panel as pn
from panel.viewable import Viewer

from roc.reporting.components.tokens import TEXT_MUTED, FONT, title_html
from roc.reporting.screen_renderer import render_grid_pane


class GridViewer(Viewer):
    """Reactive viewer for {chars, fg, bg} character grids.

    Wraps ``render_grid_pane()`` with param-based data updates so the grid
    re-renders automatically when ``grid_data`` changes.
    """

    grid_data = param.Dict(
        default=None,
        allow_None=True,
        doc="Grid data: {chars: int[][], fg: hex[][], bg: hex[][]}",
    )
    title = param.String(default="", doc="Optional title above the grid")

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._html_pane = pn.pane.HTML(
            self._render(),
            sizing_mode="stretch_width",
        )

    @param.depends("grid_data", watch=True)
    def _update(self) -> None:
        self._html_pane.object = self._render()

    def _render(self) -> str:
        if self.grid_data is None:
            return self._placeholder("No data")
        if "chars" not in self.grid_data:
            return self._placeholder("Invalid grid data")
        return render_grid_pane(self.grid_data)

    @staticmethod
    def _placeholder(message: str) -> str:
        return (
            f'<div style="font-family:{FONT};color:{TEXT_MUTED};'
            f'font-size:11px;padding:4px 0;">{message}</div>'
        )

    def __panel__(self) -> pn.Column:
        children: list[Any] = []
        if self.title:
            children.append(pn.pane.HTML(title_html(self.title)))
        children.append(self._html_pane)
        return pn.Column(
            *children,
            sizing_mode="stretch_width",
            styles={"gap": "0px"},
        )
