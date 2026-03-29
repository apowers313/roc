import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { AttentionCycleSummary, type CycleSummaryEntry } from "./AttentionCycleSummary";

function makeCycle(x: number, y: number, strength: number): CycleSummaryEntry {
    return {
        preIorPeak: { x, y, strength },
        postIorPeak: { x, y, strength },
        focusedPoint: { x, y, strength },
    };
}

describe("AttentionCycleSummary", () => {
    it("renders summary row per cycle with three columns", () => {
        const cycles: CycleSummaryEntry[] = [
            {
                preIorPeak: { x: 10, y: 5, strength: 0.95 },
                postIorPeak: { x: 10, y: 5, strength: 0.95 },
                focusedPoint: { x: 10, y: 5, strength: 0.95 },
            },
            {
                preIorPeak: { x: 10, y: 5, strength: 0.95 },
                postIorPeak: { x: 15, y: 8, strength: 0.82 },
                focusedPoint: { x: 15, y: 8, strength: 0.82 },
            },
            {
                preIorPeak: { x: 10, y: 5, strength: 0.95 },
                postIorPeak: { x: 22, y: 1, strength: 0.71 },
                focusedPoint: { x: 22, y: 1, strength: 0.71 },
            },
        ];
        renderWithProviders(<AttentionCycleSummary cycles={cycles} />);
        expect(screen.getAllByRole("row")).toHaveLength(4); // header + 3
        expect(screen.getByText("Pre-IOR Peak")).toBeInTheDocument();
        expect(screen.getByText("Post-IOR Peak")).toBeInTheDocument();
        expect(screen.getByText("Focused Point")).toBeInTheDocument();
    });

    it("stepper switches cycle selection", () => {
        const onChange = vi.fn();
        const cycles = [
            makeCycle(10, 5, 0.95),
            makeCycle(15, 8, 0.82),
        ];
        renderWithProviders(
            <AttentionCycleSummary
                cycles={cycles}
                selectedCycle={0}
                onCycleChange={onChange}
            />,
        );
        const cycle2Label = screen.getByText("Cycle 2");
        fireEvent.click(cycle2Label);
        expect(onChange).toHaveBeenCalledWith(1);
    });

    it("single cycle shows no stepper (backward compat)", () => {
        renderWithProviders(
            <AttentionCycleSummary cycles={[makeCycle(10, 5, 0.95)]} />,
        );
        // With only 1 cycle and no onCycleChange, no SegmentedControl is rendered
        expect(screen.queryByText("Cycle 1")).not.toBeInTheDocument();
        // But the table should still have 1 data row
        expect(screen.getAllByRole("row")).toHaveLength(2); // header + 1
    });

    it("shows empty state with no cycles", () => {
        renderWithProviders(<AttentionCycleSummary cycles={[]} />);
        expect(screen.getByText("No cycle data")).toBeInTheDocument();
    });
});
