"""Shared rendering for game screens and saliency heatmaps.

Both W&B and Grafana render from the same JSON data format:
``{chars: int[][], fg: hex_str[][], bg: hex_str[][]}``.

- ``screen_to_html_vals()`` converts a raw screen dict (chars + curses color
  indices) into this shared format.
- ``SaliencyMap.to_html_vals()`` already produces this format.
- ``render_grid_html()`` wraps the JSON in a self-contained HTML document with
  embedded JS that renders colored ``<span>`` elements -- the same logic used
  by the Grafana ``parseColorScreen`` Handlebars helper.
"""

from __future__ import annotations

import json
from typing import Any

# Standard curses 16-color palette mapped to hex strings.
# Index 0 (black) is rendered as dark gray so text is visible on a black bg.
CURSES_COLORS_HEX: dict[int, str] = {
    0: "646464",  # black -> dark gray
    1: "bb0000",  # red
    2: "00bb00",  # green
    3: "bb7f00",  # brown/dark yellow
    4: "0000bb",  # blue
    5: "bb00bb",  # magenta
    6: "00bbbb",  # cyan
    7: "bbbbbb",  # white/light gray
    8: "555555",  # bright black (dark gray)
    9: "ff5555",  # bright red
    10: "55ff55",  # bright green
    11: "ffff55",  # bright yellow
    12: "5555ff",  # bright blue
    13: "ff55ff",  # bright magenta
    14: "55ffff",  # bright cyan
    15: "ffffff",  # bright white
}

# JS renderer -- identical logic to the Grafana parseColorScreen Handlebars
# helper so both systems produce the same visual output.
_RENDER_JS = """\
const data = __DATA__;
const esc = s => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
let html = "";
data.chars.forEach((row, y) => {
  row.forEach((col, x) => {
    const ch = String.fromCharCode(data.chars[y][x]);
    const fg = data.fg[y][x];
    const bg = data.bg[y][x];
    html += '<span style="color:#' + fg + ';background-color:#' + bg + '">'
          + esc(ch) + '</span>';
  });
  html += "<br>";
});
document.getElementById("grid").innerHTML = html;
"""

_HTML_TEMPLATE = """\
<!DOCTYPE html><html><head><style>\
body {{ background: #000; margin: 0; padding: 2px; }}\
#grid {{ font-family: 'DejaVu Sans Mono', 'Courier New', monospace;\
 font-size: 9px; line-height: 1.15; letter-spacing: 0; margin: 0; }}\
</style></head><body>\
<div id="grid"></div>\
<script>{script}</script>\
</body></html>"""


def screen_to_html_vals(screen: dict[str, Any]) -> dict[str, list[list[str | int]]]:
    """Convert a raw screen dict to the shared ``{chars, fg, bg}`` format.

    Args:
        screen: Dict with ``chars`` (2D list/array of ints) and ``colors``
            (2D list/array of curses color indices).

    Returns:
        Dict with ``chars`` (2D ints), ``fg`` (2D hex strings),
        ``bg`` (2D hex strings).
    """
    raw_chars = screen["chars"]
    raw_colors = screen["colors"]
    rows = len(raw_chars)
    cols = len(raw_chars[0]) if rows > 0 else 0

    chars: list[list[str | int]] = []
    fg: list[list[str | int]] = []
    bg: list[list[str | int]] = []

    for y in range(rows):
        char_row: list[str | int] = []
        fg_row: list[str | int] = []
        bg_row: list[str | int] = []
        for x in range(cols):
            ch = int(raw_chars[y][x])
            char_row.append(ch if ch >= 32 else 32)
            fg_row.append(CURSES_COLORS_HEX.get(int(raw_colors[y][x]), CURSES_COLORS_HEX[7]))
            bg_row.append("000000")
        chars.append(char_row)
        fg.append(fg_row)
        bg.append(bg_row)

    return {"chars": chars, "fg": fg, "bg": bg}


def render_grid_pane(grid_data: dict[str, list[list[str | int]]]) -> str:
    """Render a ``{chars, fg, bg}`` dict as inline HTML suitable for Panel embedding.

    Unlike ``render_grid_html()`` which produces a full HTML document with JS,
    this produces a static HTML string of ``<span>`` elements that can be
    placed directly into a ``pn.pane.HTML`` widget.

    Args:
        grid_data: Dict with ``chars`` (2D ints), ``fg`` (2D hex strings),
            ``bg`` (2D hex strings).

    Returns:
        HTML fragment string with colored ``<span>`` elements.
    """
    import html as html_mod

    chars = grid_data["chars"]
    fg = grid_data["fg"]
    bg = grid_data["bg"]
    rows = len(chars)
    parts: list[str] = [
        "<div style=\"font-family: 'DejaVu Sans Mono', 'Courier New', monospace;"
        " font-size: 9px; line-height: 1.15; letter-spacing: 0;"
        ' background: #000; padding: 2px;">'
    ]
    for y in range(rows):
        # Each row in its own div with white-space:pre to preserve spaces
        parts.append('<div style="white-space:pre;">')
        cols = len(chars[y])
        for x in range(cols):
            ch = chr(int(chars[y][x])) if int(chars[y][x]) >= 32 else " "
            fg_hex = fg[y][x]
            bg_hex = bg[y][x]
            parts.append(
                f'<span style="color:#{fg_hex};background-color:#{bg_hex}">'
                f"{html_mod.escape(ch)}</span>"
            )
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def render_grid_html(grid_data: dict[str, list[list[str | int]]]) -> str:
    """Wrap a ``{chars, fg, bg}`` dict in a self-contained HTML document.

    The embedded JS renderer is identical to the Grafana ``parseColorScreen``
    Handlebars helper, ensuring both systems produce the same visual output.

    Args:
        grid_data: Dict with ``chars`` (2D ints), ``fg`` (2D hex strings),
            ``bg`` (2D hex strings).  Produced by ``screen_to_html_vals()``
            or ``SaliencyMap.to_html_vals()``.

    Returns:
        Self-contained HTML string.
    """
    data_json = json.dumps(grid_data, separators=(",", ":"))
    script = _RENDER_JS.replace("__DATA__", data_json)
    return _HTML_TEMPLATE.format(script=script)
