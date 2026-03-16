# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/screen_renderer.py."""

from roc.reporting.screen_renderer import (
    CURSES_COLORS_HEX,
    render_grid_html,
    screen_to_html_vals,
)


class TestScreenToHtmlVals:
    def test_basic_conversion(self):
        """Converts chars and curses color indices to {chars, fg, bg} format."""
        screen = {
            "chars": [[65, 66]],
            "colors": [[1, 2]],
        }
        result = screen_to_html_vals(screen)
        assert result["chars"] == [[65, 66]]
        assert result["fg"] == [[CURSES_COLORS_HEX[1], CURSES_COLORS_HEX[2]]]
        assert result["bg"] == [["000000", "000000"]]

    def test_nonprintable_chars_become_space(self):
        """Characters < 32 are replaced with 32 (space)."""
        screen = {
            "chars": [[0, 10, 31]],
            "colors": [[7, 7, 7]],
        }
        result = screen_to_html_vals(screen)
        assert result["chars"] == [[32, 32, 32]]

    def test_unknown_color_falls_back_to_white(self):
        """Unknown curses color index falls back to color 7 (white/light gray)."""
        screen = {
            "chars": [[65]],
            "colors": [[99]],
        }
        result = screen_to_html_vals(screen)
        assert result["fg"] == [[CURSES_COLORS_HEX[7]]]

    def test_bg_is_always_black(self):
        """All background values are black."""
        screen = {
            "chars": [[65, 66], [67, 68]],
            "colors": [[1, 2], [3, 4]],
        }
        result = screen_to_html_vals(screen)
        for row in result["bg"]:
            for val in row:
                assert val == "000000"

    def test_dimensions_match_input(self):
        """Output grid dimensions match input."""
        screen = {
            "chars": [[65, 66, 67], [68, 69, 70]],
            "colors": [[1, 2, 3], [4, 5, 6]],
        }
        result = screen_to_html_vals(screen)
        assert len(result["chars"]) == 2
        assert len(result["chars"][0]) == 3
        assert len(result["fg"]) == 2
        assert len(result["fg"][0]) == 3

    def test_known_screen(self):
        """Convert a real NetHack screen and verify shape."""
        from helpers.nethack_screens import screens

        screen = screens[0]
        result = screen_to_html_vals(screen)
        assert len(result["chars"]) == len(screen["chars"])
        assert len(result["chars"][0]) == len(screen["chars"][0])


class TestRenderGridHtml:
    def test_html_structure(self):
        """Output is a self-contained HTML document with JS renderer."""
        grid: dict[str, list[list[str | int]]] = {
            "chars": [[65, 66]],
            "fg": [["ffffff", "ff0000"]],
            "bg": [["000000", "000000"]],
        }
        result = render_grid_html(grid)
        assert "<!DOCTYPE html>" in result
        assert "<script>" in result
        assert "monospace" in result
        assert 'id="grid"' in result

    def test_data_embedded(self):
        """The JSON data is embedded in the HTML."""
        grid: dict[str, list[list[str | int]]] = {
            "chars": [[65]],
            "fg": [["abcdef"]],
            "bg": [["123456"]],
        }
        result = render_grid_html(grid)
        assert '"abcdef"' in result
        assert '"123456"' in result
        assert "65" in result

    def test_saliency_data(self):
        """Saliency-style data with non-black backgrounds works."""
        grid: dict[str, list[list[str | int]]] = {
            "chars": [[46, 64]],
            "fg": [["ffffff", "ffff55"]],
            "bg": [["0000ff", "ff0000"]],
        }
        result = render_grid_html(grid)
        assert '"0000ff"' in result
        assert '"ff0000"' in result

    def test_roundtrip_screen(self):
        """screen_to_html_vals -> render_grid_html produces valid HTML."""
        screen = {
            "chars": [[65, 60, 62, 38]],  # A, <, >, &
            "colors": [[7, 1, 2, 3]],
        }
        vals = screen_to_html_vals(screen)
        result = render_grid_html(vals)
        assert "<!DOCTYPE html>" in result
        assert "</html>" in result
        assert len(result) > 100
