/** Cross-component point highlighting context.
 *
 * Components that display spatial data (focus points, resolved objects,
 * attenuated locations) can set highlighted coordinates. Map components
 * (game screen, saliency map) render circle overlays at those coordinates.
 */

import { createContext, useCallback, useContext, useState } from "react";
import type { ReactNode } from "react";

export interface HighlightPoint {
    x: number;
    y: number;
    label?: string;
}

interface HighlightContextValue {
    /** Currently highlighted points. */
    points: HighlightPoint[];
    /** Replace highlighted points. Pass [] to clear. */
    setPoints: (pts: HighlightPoint[]) => void;
    /** Toggle a single point: add if missing, remove if present. */
    togglePoint: (pt: HighlightPoint) => void;
    /** Clear all highlights. */
    clear: () => void;
}

const HighlightContext = createContext<HighlightContextValue>({
    points: [],
    setPoints: () => {},
    togglePoint: () => {},
    clear: () => {},
});

export function HighlightProvider({ children }: { children: ReactNode }) {
    const [points, setPoints] = useState<HighlightPoint[]>([]);

    const togglePoint = useCallback((pt: HighlightPoint) => {
        setPoints((prev) => {
            const exists = prev.some((p) => p.x === pt.x && p.y === pt.y);
            if (exists) {
                return prev.filter((p) => !(p.x === pt.x && p.y === pt.y));
            }
            return [...prev, pt];
        });
    }, []);

    const clear = useCallback(() => setPoints([]), []);

    return (
        <HighlightContext.Provider value={{ points, setPoints, togglePoint, clear }}>
            {children}
        </HighlightContext.Provider>
    );
}

export function useHighlight() {
    return useContext(HighlightContext);
}
