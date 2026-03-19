/** Wrapper that adds click-to-navigate-to-step on recharts charts.
 *
 * Usage:
 *   <ClickableChart onStepClick={setStep}>
 *     <ResponsiveContainer>
 *       <LineChart data={data} ...>
 *         ...
 *       </LineChart>
 *     </ResponsiveContainer>
 *   </ClickableChart>
 *
 * When the user clicks inside the chart plot area, this component finds the
 * nearest data point's `step` value and calls `onStepClick(step)`.
 * The chart data must have a `step` field in each data point.
 */

import type { ReactNode } from "react";
import { useCallback, useRef } from "react";

interface ClickableChartProps {
    /** Called with the step number when the user clicks on the chart. */
    onStepClick: (step: number) => void;
    /** The chart data array -- each entry must have a `step` field. */
    data: readonly { step: number }[];
    children: ReactNode;
}

export function ClickableChart({ onStepClick, data, children }: ClickableChartProps) {
    const containerRef = useRef<HTMLDivElement>(null);

    const handleClick = useCallback(
        (e: React.MouseEvent<HTMLDivElement>) => {
            if (!containerRef.current || !data || data.length === 0) return;

            // Find the recharts CartesianGrid rect element -- it defines
            // the exact plot area (excluding axes, legends, margins).
            const gridRect = containerRef.current.querySelector(
                ".recharts-cartesian-grid",
            );
            if (gridRect) {
                const plotBounds = gridRect.getBoundingClientRect();
                const relativeX = e.clientX - plotBounds.left;
                const fraction = Math.max(0, Math.min(1, relativeX / plotBounds.width));
                const index = Math.round(fraction * (data.length - 1));
                const point = data[index];
                if (point && typeof point.step === "number") {
                    onStepClick(point.step);
                }
                return;
            }

            // Fallback: use the container bounds with estimated margins
            const rect = containerRef.current.getBoundingClientRect();
            const leftMargin = 48;
            const rightMargin = 8;
            const chartWidth = rect.width - leftMargin - rightMargin;
            const relativeX = e.clientX - rect.left - leftMargin;
            const fraction = Math.max(0, Math.min(1, relativeX / chartWidth));
            const index = Math.round(fraction * (data.length - 1));
            const point = data[index];
            if (point && typeof point.step === "number") {
                onStepClick(point.step);
            }
        },
        [onStepClick, data],
    );

    return (
        <div
            ref={containerRef}
            onClick={handleClick}
            style={{ cursor: "crosshair" }}
        >
            {children}
        </div>
    );
}
