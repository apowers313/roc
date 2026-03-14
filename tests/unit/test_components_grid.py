# mypy: disable-error-code="no-untyped-def"

"""Tests for roc/reporting/components/grid_viewer.py."""

import panel as pn

from roc.reporting.components.grid_viewer import GridViewer


class TestGridViewer:
    def test_renders_valid_grid(self):
        grid = {
            "chars": [[65, 66]],
            "fg": [["ffffff", "ff0000"]],
            "bg": [["000000", "000000"]],
        }
        viewer = GridViewer(grid_data=grid)
        html = viewer._render()
        assert "<span" in html
        assert "monospace" in html

    def test_none_returns_empty(self):
        viewer = GridViewer(grid_data=None)
        html = viewer._render()
        assert html == ""

    def test_missing_chars_returns_empty(self):
        viewer = GridViewer(grid_data={"fg": [], "bg": []})
        html = viewer._render()
        assert html == ""

    def test_panel_returns_html_pane(self):
        grid = {
            "chars": [[65]],
            "fg": [["ffffff"]],
            "bg": [["000000"]],
        }
        viewer = GridViewer(grid_data=grid)
        panel = viewer.__panel__()
        assert isinstance(panel, pn.pane.HTML)

    def test_reactive_update(self):
        viewer = GridViewer(grid_data=None)
        assert viewer._html_pane.object == ""

        viewer.grid_data = {
            "chars": [[65]],
            "fg": [["ffffff"]],
            "bg": [["000000"]],
        }
        assert "<span" in viewer._html_pane.object
