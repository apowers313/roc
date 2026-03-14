# mypy: disable-error-code="no-untyped-def"

"""Tests for roc/reporting/components/grid_viewer.py."""

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

    def test_none_shows_placeholder(self):
        viewer = GridViewer(grid_data=None)
        html = viewer._render()
        assert "No data" in html

    def test_missing_chars_shows_placeholder(self):
        viewer = GridViewer(grid_data={"fg": [], "bg": []})
        html = viewer._render()
        assert "Invalid" in html

    def test_title_rendered(self):
        grid = {
            "chars": [[65]],
            "fg": [["ffffff"]],
            "bg": [["000000"]],
        }
        viewer = GridViewer(grid_data=grid, title="Screen")
        panel = viewer.__panel__()
        # Title should be in one of the children
        html_panes = [c for c in panel if hasattr(c, "object") and isinstance(c.object, str)]
        title_found = any("Screen" in p.object for p in html_panes if p.object)
        assert title_found

    def test_reactive_update(self):
        viewer = GridViewer(grid_data=None)
        assert "No data" in viewer._html_pane.object

        viewer.grid_data = {
            "chars": [[65]],
            "fg": [["ffffff"]],
            "bg": [["000000"]],
        }
        assert "<span" in viewer._html_pane.object
