/** Cross-component point highlighting context.
 *
 * Components that display spatial data (focus points, resolved objects,
 * attenuated locations) can set highlighted coordinates. Map components
 * (game screen, saliency map) render circle overlays at those coordinates.
 *
 * Each highlighted point gets a distinct color from a rotating palette,
 * used consistently everywhere the point appears across all panels.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import { DashboardContext } from "./context";

/** Rotating palette chosen for distinctness and color-blind accessibility. */
const HIGHLIGHT_PALETTE = [
    "#ff4444", // red
    "#4488ff", // blue
    "#44cc44", // green
    "#ff8800", // orange
    "#cc44ff", // purple
    "#44cccc", // cyan
] as const;

export interface HighlightPoint {
    x: number;
    y: number;
    label?: string;
    /** Assigned automatically from the rotating palette. */
    color: string;
    /** Which panel originated this highlight. */
    source?: string;
}

interface HighlightContextValue {
    /** Currently highlighted points. */
    points: HighlightPoint[];
    /** Replace highlighted points. Pass [] to clear. */
    setPoints: (pts: HighlightPoint[]) => void;
    /** Toggle a single point: add if missing (with auto-assigned color), remove if present. */
    togglePoint: (pt: Omit<HighlightPoint, "color">) => void;
    /** Clear all highlights. */
    clear: () => void;
}

const HighlightContext = createContext<HighlightContextValue>({
    points: [],
    setPoints: () => {},
    togglePoint: () => {},
    clear: () => {},
});

export function HighlightProvider({ children }: Readonly<{ children: ReactNode }>) {
    const [points, setPoints] = useState<HighlightPoint[]>([]);

    // Auto-clear highlights when the dashboard step changes.
    // DashboardContext is optional -- standalone tests pass null.
    const dashCtx = useContext(DashboardContext);
    const prevStep = useRef(dashCtx?.step);
    useEffect(() => {
        if (prevStep.current !== undefined && dashCtx != null && prevStep.current !== dashCtx.step) {
            setPoints([]);
        }
        prevStep.current = dashCtx?.step;
    }, [dashCtx?.step]);

    const togglePoint = useCallback((pt: Omit<HighlightPoint, "color">) => {
        setPoints((prev) => {
            const exists = prev.some((p) => p.x === pt.x && p.y === pt.y);
            if (exists) {
                return prev.filter((p) => !(p.x === pt.x && p.y === pt.y));
            }
            // Assign the next color from the rotating palette
            const colorIndex = prev.length % HIGHLIGHT_PALETTE.length;
            return [...prev, { ...pt, color: HIGHLIGHT_PALETTE[colorIndex]! }];
        });
    }, []);

    const clear = useCallback(() => setPoints([]), []);

    const contextValue = useMemo(
        () => ({ points, setPoints, togglePoint, clear }),
        [points, setPoints, togglePoint, clear],
    );

    return (
        <HighlightContext.Provider value={contextValue}>
            {children}
        </HighlightContext.Provider>
    );
}

export function useHighlight() {
    return useContext(HighlightContext);
}

/** Find the highlight color for a point, or undefined if not highlighted. */
export function findHighlightColor(
    points: HighlightPoint[],
    x: number,
    y: number,
): string | undefined {
    const pt = points.find((p) => p.x === x && p.y === y);
    return pt?.color;
}

/** Convert a hex highlight color to a semi-transparent background for table rows. */
export function highlightBg(color: string | undefined): string | undefined {
    if (!color) return undefined;
    // Parse hex color and return rgba with 0.15 alpha
    const r = Number.parseInt(color.slice(1, 3), 16);
    const g = Number.parseInt(color.slice(3, 5), 16);
    const b = Number.parseInt(color.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, 0.15)`;
}
