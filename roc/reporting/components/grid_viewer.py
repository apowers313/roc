"""GridViewer -- reactive Viewer for colored monospace character grids."""

from __future__ import annotations

from typing import Any

import param
import panel as pn
from panel.viewable import Viewer

from roc.reporting.screen_renderer import render_grid_pane


class GridViewer(Viewer):
    """Reactive viewer for {chars, fg, bg} character grids.

    Wraps ``render_grid_pane()`` with param-based data updates so the grid
    re-renders automatically when ``grid_data`` changes.

    Raw HTML is justified here because there is no Panel widget that renders
    a per-character-colored monospace grid.
    """

    grid_data = param.Dict(
        default=None,
        allow_None=True,
        doc="Grid data: {chars: int[][], fg: hex[][], bg: hex[][]}",
    )

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
        """Render grid data to HTML, or return empty string for no data."""
        if self.grid_data is None:
            return ""
        if "chars" not in self.grid_data:
            return ""
        return render_grid_pane(self.grid_data)

    def __panel__(self) -> pn.pane.HTML:
        return self._html_pane
