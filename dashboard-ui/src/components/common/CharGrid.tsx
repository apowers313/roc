/** Colored character grid renderer for game screen and saliency map. */

import { useEffect, useRef } from "react";

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

function normalizeColor(raw: string): string {
    return raw.startsWith("#") ? raw : `#${raw}`;
}

/** Build an HTML span for a single cell. */
function buildCellSpan(ch: string, fgColor: string, bgColor: string, highlightColor?: string): string {
    if (highlightColor) {
        return `<span style="color:${fgColor};background:${bgColor};outline:2px solid ${highlightColor};outline-offset:-1px;position:relative;z-index:1">${ch}</span>`;
    }
    return `<span style="color:${fgColor};background:${bgColor}">${ch}</span>`;
}

/** Build the HTML string for a character grid.
 *  If highlights is provided, matching cells get a colored outline.
 */
function buildGridHtml(data: GridData, highlights?: Map<string, string>): string {
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
            const ch = escapeChar(String.fromCodePoint(charCode));
            const fgColor = normalizeColor(fgRow[c] ?? "ffffff");
            const bgColor = normalizeColor(bgRow[c] ?? "000000");
            const highlightColor = highlights?.get(`${c},${r}`);
            spans.push(buildCellSpan(ch, fgColor, bgColor, highlightColor));
        }
        rows.push(spans.join(""));
    }
    return rows.join("\n");
}

/**
 * Renders a 2D character grid with per-cell foreground and background colors.
 * Uses innerHTML for performance -- building 1659 React elements per render
 * is slower than a single HTML string.
 *
 * Sets innerHTML via a ref + useEffect instead of dangerouslySetInnerHTML
 * so we can force a repaint after each update. Safari/WebKit skips repainting
 * fixed-dimension elements when only innerHTML changes; reading offsetHeight
 * after the write forces the repaint.
 *
 * Reads from HighlightContext to render outlined markers on highlighted cells.
 *
 * No React.memo -- the parent re-renders when data changes, and we always
 * want to reflect the new data immediately.
 */
export function CharGrid({ data, highlightRowOffset = 0 }: Readonly<CharGridProps>) {
    const { points } = useHighlight();
    const preRef = useRef<HTMLPreElement>(null);

    const highlightMap = points.length > 0
        ? new Map(points.map((p) => [`${p.x},${p.y + highlightRowOffset}`, p.color]))
        : undefined;

    const html = buildGridHtml(data, highlightMap);
    const htmlLen = html.length;
    const snippet = html.slice(-80);

    // eslint-disable-next-line no-console
    console.log("[CharGrid] render", JSON.stringify({ htmlLen, snippet }));

    useEffect(() => {
        if (preRef.current) {
            preRef.current.innerHTML = html;
            // Force Safari/WebKit to repaint after innerHTML change.
            void preRef.current.offsetHeight;
            // eslint-disable-next-line no-console
            console.log("[CharGrid] innerHTML set", JSON.stringify({ htmlLen }));
        }
    }, [html, htmlLen]);

    return (
        <pre
            ref={preRef}
            style={{
                fontFamily: "'DejaVu Sans Mono', monospace",
                fontSize: "9px",
                lineHeight: 1.15,
                background: "#000",
                padding: "4px",
                margin: 0,
                overflow: "auto",
                width: "fit-content",
            }}
        />
    );
}
