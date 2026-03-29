import { screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { TransitionPanel } from "./TransitionPanel";

// Mock useObjectHistory for ObjectLink's ObjectModal
vi.mock("../../api/queries", () => ({
    useObjectHistory: vi.fn().mockReturnValue({
        data: { info: { uuid: 100, resolve_count: 1 }, states: [], transforms: [] },
        isLoading: false,
    }),
}));

describe("TransitionPanel three-column", () => {
    it("shows 'No transform data' when undefined", () => {
        renderWithProviders(<TransitionPanel data={undefined} />);
        expect(screen.getByText("No transform data")).toBeInTheDocument();
    });

    it("shows 'No changes this step' when empty", () => {
        const data = makeStepData({
            transform_summary: { count: 0, changes: [] },
        });
        renderWithProviders(<TransitionPanel data={data} />);
        expect(screen.getByText("No changes this step")).toBeInTheDocument();
    });

    it("aligns same object across prev/current columns", () => {
        const data = makeStepData({
            transform_summary: {
                count: 1,
                changes: [],
                object_transforms: [{
                    uuid: 100,
                    glyph: "@",
                    color: "WHITE",
                    node_id: -10,
                    changes: [
                        { property: "x", type: "continuous", delta: 2, old_value: 10, new_value: 12 },
                    ],
                }],
            },
            sequence_summary: {
                tick: 5,
                object_count: 1,
                intrinsic_count: 0,
                objects: [{ id: "-10", x: 12, y: 5, glyph: "@", matched_previous: true }],
                intrinsics: {},
            },
        });
        renderWithProviders(<TransitionPanel data={data} />);
        // Column headers
        expect(screen.getByText("Previous", { exact: false })).toBeInTheDocument();
        expect(screen.getByText("Current", { exact: false })).toBeInTheDocument();
        expect(screen.getByText("Delta")).toBeInTheDocument();
        // Object glyph
        expect(screen.getByText("@")).toBeInTheDocument();
        // Previous position (old_value x=10, y not in changes so just x)
        expect(screen.getByText("(10, ?)")).toBeInTheDocument();
        // Current position
        expect(screen.getByText("(12, ?)")).toBeInTheDocument();
    });

    it("shows new object with -- in previous column", () => {
        const data = makeStepData({
            transform_summary: { count: 0, changes: [] },
            sequence_summary: {
                tick: 5,
                object_count: 1,
                intrinsic_count: 0,
                objects: [{ id: "-20", x: 15, y: 8, glyph: "d", matched_previous: false }],
                intrinsics: {},
            },
        });
        renderWithProviders(<TransitionPanel data={data} />);
        // New object shows "--" in previous column and "(new)" in delta
        const dashes = screen.getAllByText("--");
        expect(dashes.length).toBeGreaterThanOrEqual(1);
        expect(screen.getByText("(new)")).toBeInTheDocument();
        expect(screen.getByText("(15, 8)")).toBeInTheDocument();
    });

    it("shows gone object with -- in current column", () => {
        const data = makeStepData({
            transform_summary: {
                count: 1,
                changes: [],
                object_transforms: [{
                    uuid: 200,
                    glyph: ".",
                    status: "gone",
                    changes: [
                        { property: "x", type: "continuous", delta: null, old_value: 22, new_value: null },
                        { property: "y", type: "continuous", delta: null, old_value: 1, new_value: null },
                    ],
                }],
            },
        });
        renderWithProviders(<TransitionPanel data={data} />);
        expect(screen.getByText("(gone)")).toBeInTheDocument();
        // Previous should show position
        expect(screen.getByText("(22, 1)")).toBeInTheDocument();
        // Current should show "--"
        const dashes = screen.getAllByText("--");
        expect(dashes.length).toBeGreaterThanOrEqual(1);
    });

    it("shows intrinsic deltas in same three-column layout", () => {
        const data = makeStepData({
            transform_summary: {
                count: 2,
                changes: [
                    { description: "hp changed", type: "intrinsic", name: "hp", normalized_change: -0.05 },
                    { description: "energy unchanged", type: "intrinsic", name: "energy", normalized_change: 0 },
                ],
            },
            sequence_summary: {
                tick: 7,
                object_count: 0,
                intrinsic_count: 2,
                objects: [],
                intrinsics: { hp: 0.8, energy: 0.5 },
            },
        });
        renderWithProviders(<TransitionPanel data={data} />);
        // Intrinsic section
        expect(screen.getByText("Intrinsic Changes")).toBeInTheDocument();
        expect(screen.getByText("hp")).toBeInTheDocument();
        expect(screen.getByText("energy")).toBeInTheDocument();
        // Delta values
        expect(screen.getByText("-0.0500")).toBeInTheDocument();
        // hp: current 0.8000, previous 0.8500 (0.8 - (-0.05))
        expect(screen.getByText("0.8000")).toBeInTheDocument();
        expect(screen.getByText("0.8500")).toBeInTheDocument();
        // energy: current 0.5000, previous 0.5000 (unchanged)
        expect(screen.getAllByText("0.5000").length).toBe(2);
    });

    it("object cells use ObjectLink", () => {
        const data = makeStepData({
            transform_summary: {
                count: 1,
                changes: [],
                object_transforms: [{
                    uuid: 100,
                    glyph: "@",
                    color: "WHITE",
                    node_id: -10,
                    changes: [
                        { property: "x", type: "continuous", delta: 2, old_value: 10, new_value: 12 },
                    ],
                }],
            },
        });
        renderWithProviders(<TransitionPanel data={data} />);
        const glyph = screen.getByText("@");
        expect(glyph).toHaveStyle({ cursor: "pointer" });
        // Click opens modal
        fireEvent.click(glyph);
        expect(screen.getByText(/Object:/)).toBeInTheDocument();
    });

    it("step references in header are clickable", () => {
        const onStepClick = vi.fn();
        const data = makeStepData({
            transform_summary: {
                count: 1,
                changes: [],
                object_transforms: [{
                    uuid: 100,
                    glyph: "@",
                    node_id: -10,
                    changes: [{ property: "x", type: "continuous", delta: 1, old_value: 5, new_value: 6 }],
                }],
            },
            sequence_summary: {
                tick: 7,
                object_count: 1,
                intrinsic_count: 0,
                objects: [],
                intrinsics: {},
            },
        });
        renderWithProviders(<TransitionPanel data={data} onStepClick={onStepClick} />);
        // Previous header shows tick=6 (tick - 1)
        const prevTickBtn = screen.getByText("tick=6");
        fireEvent.click(prevTickBtn);
        expect(onStepClick).toHaveBeenCalledWith(6);
        // Current header shows tick=7
        const currTickBtn = screen.getByText("tick=7");
        fireEvent.click(currTickBtn);
        expect(onStepClick).toHaveBeenCalledWith(7);
    });
});
