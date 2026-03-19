/** Colored character grid renderer for game screen and saliency map. */

import type { GridData } from "../../types/step-data";
import { useHighlight } from "../../state/highlight";

interface CharGridProps {
    data: GridData;
    /** Row offset applied to highlight coordinates.
     *  The game screen TTY has a 1-row message header, so dungeon y=N
     *  maps to screen row N+1. Set highlightRowOffset={1} for game screen,
     *  leave as 0 (default) for saliency map which matches dungeon coords.
     */
    highlightRowOffset?: number;
}

/** HTML entity escapes for characters that would break the markup. */
function escapeChar(ch: string): string {
    switch (ch) {
        case "<":
            return "&lt;";
        case ">":
            return "&gt;";
        case "&":
            return "&amp;";
        default:
            return ch;
    }
}

/** Build the HTML string for a character grid.
 *  If highlights is provided, matching cells get a bright outline.
 */
function buildGridHtml(data: GridData, highlights?: Set<string>): string {
    const { chars, fg, bg } = data;
    const rows: string[] = [];
    for (let r = 0; r < chars.length; r++) {
        const row = chars[r];
        const fgRow = fg[r];
        const bgRow = bg[r];
        if (!row || !fgRow || !bgRow) continue;

        const spans: string[] = [];
        for (let c = 0; c < row.length; c++) {
            const charCode = row[c];
            if (charCode === undefined) continue;
            const ch = escapeChar(String.fromCharCode(charCode));
            const rawFg = fgRow[c] ?? "ffffff";
            const rawBg = bgRow[c] ?? "000000";
            const fgColor = rawFg.startsWith("#") ? rawFg : `#${rawFg}`;
            const bgColor = rawBg.startsWith("#") ? rawBg : `#${rawBg}`;

            // Highlight coordinates use (x=col, y=row)
            const isHighlighted = highlights?.has(`${c},${r}`);
            if (isHighlighted) {
                spans.push(
                    `<span style="color:${fgColor};background:${bgColor};outline:2px solid #ff0;outline-offset:-1px;position:relative;z-index:1">${ch}</span>`,
                );
            } else {
                spans.push(
                    `<span style="color:${fgColor};background:${bgColor}">${ch}</span>`,
                );
            }
        }
        rows.push(spans.join(""));
    }
    return rows.join("\n");
}

/**
 * Renders a 2D character grid with per-cell foreground and background colors.
 * Uses dangerouslySetInnerHTML for performance -- building 1659 React elements
 * per render is slower than a single HTML string.
 *
 * Reads from HighlightContext to render outlined markers on highlighted cells.
 *
 * No React.memo -- the parent re-renders when data changes, and we always
 * want to reflect the new data immediately.
 */
export function CharGrid({ data, highlightRowOffset = 0 }: CharGridProps) {
    const { points } = useHighlight();

    const highlightSet = points.length > 0
        ? new Set(points.map((p) => `${p.x},${p.y + highlightRowOffset}`))
        : undefined;

    const html = buildGridHtml(data, highlightSet);

    return (
        <pre
            style={{
                fontFamily: "'DejaVu Sans Mono', monospace",
                fontSize: "9px",
                lineHeight: 1.15,
                background: "#000",
                padding: "4px",
                margin: 0,
                overflow: "auto",
            }}
            dangerouslySetInnerHTML={{ __html: html }}
        />
    );
}
