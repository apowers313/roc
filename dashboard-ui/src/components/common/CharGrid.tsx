/** Colored character grid renderer for game screen and saliency map. */

import type { GridData } from "../../types/step-data";

interface CharGridProps {
    data: GridData;
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

/** Build the HTML string for a character grid. */
function buildGridHtml(data: GridData): string {
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
            spans.push(
                `<span style="color:${fgColor};background:${bgColor}">${ch}</span>`,
            );
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
 * No React.memo -- the parent re-renders when data changes, and we always
 * want to reflect the new data immediately.
 */
export function CharGrid({ data }: CharGridProps) {
    const html = buildGridHtml(data);

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
