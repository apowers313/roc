import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

import type { ObjectHistoryData } from "../../api/client";
import { renderWithProviders } from "../../test-utils";
import { ObjectModal } from "./ObjectModal";

vi.mock("../../api/queries", () => ({
    useObjectHistory: vi.fn(),
}));

import { useObjectHistory } from "../../api/queries";

const MOCK_HISTORY: ObjectHistoryData = {
    info: { uuid: "12345", resolve_count: 42 },
    states: [
        { tick: 1, x: 10, y: 5, glyph_type: 333, color_type: 7, shape_type: 64, distance: null, flood_size: null, line_size: null, motion_direction: null, delta_old: null, delta_new: null },
        { tick: 2, x: 12, y: 5, glyph_type: 333, color_type: 3, shape_type: 64, distance: 2.5, flood_size: null, line_size: null, motion_direction: null, delta_old: null, delta_new: null },
        { tick: 3, x: 12, y: 6, glyph_type: 333, color_type: 3, shape_type: 64, distance: 1, flood_size: null, line_size: null, motion_direction: null, delta_old: null, delta_new: null },
    ],
    transforms: [
        { num_discrete_changes: 0, num_continuous_changes: 2, changes: [
            { property: "x", type: "continuous", delta: 2, old_value: 10, new_value: 12 },
            { property: "color_type", type: "discrete", delta: null, old_value: 7, new_value: 3 },
        ]},
        { num_discrete_changes: 0, num_continuous_changes: 1, changes: [
            { property: "y", type: "continuous", delta: 1, old_value: 5, new_value: 6 },
        ]},
    ],
};

function mockHistory(data?: ObjectHistoryData) {
    vi.mocked(useObjectHistory).mockReturnValue({
        data,
        isLoading: false,
    } as ReturnType<typeof useObjectHistory>);
}

describe("ObjectModal", () => {
    beforeEach(() => {
        mockHistory(MOCK_HISTORY);
    });

    it("renders object info header", () => {
        renderWithProviders(
            <ObjectModal objectId={42} opened={true} onClose={() => {}} glyph="@" color="#fff" />,
        );
        expect(screen.getByText(/Object:/)).toBeInTheDocument();
        expect(screen.getByText("@")).toBeInTheDocument();
        expect(screen.getByText(/Matches: 42/)).toBeInTheDocument();
    });

    it("renders state history table with all observations", () => {
        renderWithProviders(
            <ObjectModal objectId={42} opened={true} onClose={() => {}} glyph="@" />,
        );
        expect(screen.getByText("Observations (3)")).toBeInTheDocument();
        // 3 observation ticks (rendered as buttons)
        const tickButtons = screen.getAllByRole("button");
        const tickLabels = tickButtons.map((b) => b.textContent);
        expect(tickLabels).toContain("1");
        expect(tickLabels).toContain("2");
        expect(tickLabels).toContain("3");
        // State data is rendered: x values 10, 12 (may appear in both state and transform tables)
        expect(screen.getAllByText("10").length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText("12").length).toBeGreaterThanOrEqual(1);
        // Column headers
        expect(screen.getByText("tick")).toBeInTheDocument();
        expect(screen.getAllByText("x").length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText("y").length).toBeGreaterThanOrEqual(1);
    });

    it("renders transform history table", () => {
        renderWithProviders(
            <ObjectModal objectId={42} opened={true} onClose={() => {}} glyph="@" />,
        );
        expect(screen.getByText("Transforms (2)")).toBeInTheDocument();
        // Transform tick labels
        expect(screen.getByText("1 -> 2")).toBeInTheDocument();
        expect(screen.getByText("2 -> 3")).toBeInTheDocument();
    });

    it("tick references are clickable for navigation", () => {
        const onClose = vi.fn();
        renderWithProviders(
            <ObjectModal objectId={42} opened={true} onClose={onClose} glyph="@" />,
        );
        // Click tick 2 in observations table
        const tick2 = screen.getByRole("button", { name: "2" });
        fireEvent.click(tick2);
        // Modal should close after navigation
        expect(onClose).toHaveBeenCalled();
    });

    it("shows node_id and first_seen in header", () => {
        renderWithProviders(
            <ObjectModal objectId={-42} opened={true} onClose={() => {}} glyph="@" color="#fff" />,
        );
        expect(screen.getByText("Node: -42")).toBeInTheDocument();
        // First seen is tick 1 from mock data
        expect(screen.getByText(/First seen:/)).toBeInTheDocument();
        expect(screen.getByText("step 1")).toBeInTheDocument();
    });

    it("first_seen click navigates to that step", () => {
        const onClose = vi.fn();
        renderWithProviders(
            <ObjectModal objectId={-42} opened={true} onClose={onClose} glyph="@" />,
        );
        fireEvent.click(screen.getByText("step 1"));
        expect(onClose).toHaveBeenCalled();
    });

    it("shows 'No history data' when data is empty", () => {
        mockHistory({ info: { uuid: "1", resolve_count: 0 }, states: [], transforms: [] });
        renderWithProviders(
            <ObjectModal objectId={1} opened={true} onClose={() => {}} />,
        );
        expect(screen.getByText("No history data")).toBeInTheDocument();
    });

    it("does not render content when closed", () => {
        renderWithProviders(
            <ObjectModal objectId={42} opened={false} onClose={() => {}} glyph="@" />,
        );
        // Modal content should not be in the document when closed
        expect(screen.queryByText("Observations")).not.toBeInTheDocument();
    });
});
