/**
 * useRatchetHeight -- prevents accordion section height from shrinking between steps.
 *
 * Tracks the maximum observed content height via ResizeObserver and applies it
 * as minHeight. Sections grow but never shrink. Resets when:
 * - The accordion panel collapses (content height drops below 1px)
 * - The current run or game changes
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { useDashboard } from "../state/context";

export function useRatchetHeight(): {
    contentRef: (el: HTMLDivElement | null) => void;
    minHeight: number;
} {
    const { run, game } = useDashboard();
    const [element, setElement] = useState<HTMLDivElement | null>(null);
    const maxHeightRef = useRef(0);
    const [minHeight, setMinHeight] = useState(0);

    // Callback ref -- triggers observer setup when the DOM element mounts
    const contentRef = useCallback((el: HTMLDivElement | null) => {
        setElement(el);
    }, []);

    // Reset when run or game changes
    useEffect(() => {
        maxHeightRef.current = 0;
        setMinHeight(0);
    }, [run, game]);

    // Observe content size and ratchet upward
    useEffect(() => {
        if (!element) return;

        const observer = new ResizeObserver((entries) => {
            const entry = entries[0];
            if (!entry) return;
            const height = entry.contentRect.height;

            if (height < 1) {
                // Panel collapsed -- reset so next open starts fresh
                maxHeightRef.current = 0;
                setMinHeight(0);
                return;
            }

            if (height > maxHeightRef.current) {
                maxHeightRef.current = height;
                setMinHeight(height);
            }
        });

        observer.observe(element);
        return () => observer.disconnect();
    }, [element]);

    return { contentRef, minHeight };
}
