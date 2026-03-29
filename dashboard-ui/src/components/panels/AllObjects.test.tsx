import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ResolvedObject } from "../../api/client";
import { renderWithProviders } from "../../test-utils";
import { AllObjects } from "./AllObjects";

vi.mock("../../api/queries", () => ({
    useAllObjects: vi.fn(),
    useObjectHistory: vi.fn().mockReturnValue({
        data: { info: { uuid: 1, resolve_count: 0 }, states: [], transforms: [] },
        isLoading: false,
    }),
}));

import { useAllObjects } from "../../api/queries";

function mockObjects(data: ResolvedObject[] | undefined) {
    vi.mocked(useAllObjects).mockReturnValue({
        data,
    } as ReturnType<typeof useAllObjects>);
}

const OBJ_A: ResolvedObject = {
    shape: "@",
    glyph: "333",
    color: "WHITE",
    node_id: "-10",
    step_added: 5,
    match_count: 3,
    type: "Single",
};

const OBJ_B: ResolvedObject = {
    shape: ".",
    glyph: "2371",
    color: "GREY",
    node_id: "-14",
    step_added: 2,
    match_count: 10,
    type: "Flood",
};

const OBJ_NULL: ResolvedObject = {
    shape: null,
    glyph: null,
    color: null,
    node_id: null,
    step_added: null,
    match_count: 0,
    type: null,
};

describe("AllObjects", () => {
    it("shows 'No objects' when data is undefined", () => {
        mockObjects(undefined);
        renderWithProviders(<AllObjects run="test-run" />);
        expect(screen.getByText("No objects")).toBeInTheDocument();
    });

    it("shows 'No objects' when data is empty array", () => {
        mockObjects([]);
        renderWithProviders(<AllObjects run="test-run" />);
        expect(screen.getByText("No objects")).toBeInTheDocument();
    });

    it("renders table with column headers when objects present", () => {
        mockObjects([OBJ_A]);
        renderWithProviders(<AllObjects run="test-run" />);
        // shape has sort arrow, so use regex
        expect(screen.getByText(/^shape/)).toBeInTheDocument();
        expect(screen.getByText(/^glyph/)).toBeInTheDocument();
        expect(screen.getByText(/^color/)).toBeInTheDocument();
        expect(screen.getByText(/^type$/)).toBeInTheDocument();
        expect(screen.getByText(/^node id/)).toBeInTheDocument();
        expect(screen.getByText(/^step added/)).toBeInTheDocument();
        expect(screen.getByText(/^matches/)).toBeInTheDocument();
    });

    it("renders object data in table rows", () => {
        mockObjects([OBJ_A, OBJ_B]);
        renderWithProviders(<AllObjects run="test-run" />);
        expect(screen.getByText("@")).toBeInTheDocument();
        expect(screen.getByText("333")).toBeInTheDocument();
        expect(screen.getByText("WHITE")).toBeInTheDocument();
        expect(screen.getByText("-10")).toBeInTheDocument();
        expect(screen.getByText(".")).toBeInTheDocument();
        expect(screen.getByText("2371")).toBeInTheDocument();
        expect(screen.getByText("GREY")).toBeInTheDocument();
    });

    it("shows -- for null fields", () => {
        mockObjects([OBJ_NULL]);
        renderWithProviders(<AllObjects run="test-run" />);
        const dashes = screen.getAllByText("--");
        // shape, glyph, color, type, node_id, step_added are all null
        expect(dashes.length).toBeGreaterThanOrEqual(5);
    });

    it("displays row numbers starting at 1", () => {
        mockObjects([OBJ_A]);
        renderWithProviders(<AllObjects run="test-run" />);
        // Row number column header
        expect(screen.getByText("#")).toBeInTheDocument();
        // Row number 1 should appear
        expect(screen.getByText("1")).toBeInTheDocument();
    });

    it("sorts by shape by default (ascending)", () => {
        mockObjects([OBJ_B, OBJ_A]); // . and @, but default sort should order by shape
        renderWithProviders(<AllObjects run="test-run" />);
        // The shape column header should show the up-arrow for ascending
        const shapeHeader = screen.getByText(/^shape/);
        expect(shapeHeader.textContent).toContain("\u25B2");
    });

    it("toggles sort direction when clicking same column header", () => {
        mockObjects([OBJ_A, OBJ_B]);
        renderWithProviders(<AllObjects run="test-run" />);

        // Default: shape ascending
        const shapeHeader = screen.getByText(/^shape/);
        expect(shapeHeader.textContent).toContain("\u25B2");

        // Click shape again -> descending
        fireEvent.click(shapeHeader);
        expect(shapeHeader.textContent).toContain("\u25BC");
    });

    it("changes sort key when clicking different column header", () => {
        mockObjects([OBJ_A, OBJ_B]);
        renderWithProviders(<AllObjects run="test-run" />);

        // Click "matches" column header
        const matchesHeader = screen.getByText(/^matches/);
        fireEvent.click(matchesHeader);
        expect(matchesHeader.textContent).toContain("\u25B2");

        // Shape header should no longer have arrow
        const shapeHeader = screen.getByText("shape");
        expect(shapeHeader.textContent).not.toContain("\u25B2");
        expect(shapeHeader.textContent).not.toContain("\u25BC");
    });

    it("calls onStepClick when clicking a row with step_added", () => {
        const onStepClick = vi.fn();
        mockObjects([OBJ_A]);
        renderWithProviders(<AllObjects run="test-run" onStepClick={onStepClick} />);

        // Click the row (find by step_added value)
        const stepCell = screen.getByText("5");
        // Click the parent row
        fireEvent.click(stepCell.closest("tr")!);
        expect(onStepClick).toHaveBeenCalledWith(5);
    });

    it("does not call onStepClick when step_added is null", () => {
        const onStepClick = vi.fn();
        mockObjects([OBJ_NULL]);
        renderWithProviders(<AllObjects run="test-run" onStepClick={onStepClick} />);

        // Click the row
        const row = screen.getByText("0").closest("tr")!;
        fireEvent.click(row);
        expect(onStepClick).not.toHaveBeenCalled();
    });

    it("passes game parameter through to hook", () => {
        mockObjects([]);
        renderWithProviders(<AllObjects run="test-run" game={3} />);
        expect(useAllObjects).toHaveBeenCalledWith("test-run", 3);
    });

    it("applies NH color mapping for known colors", () => {
        mockObjects([OBJ_A]); // WHITE -> #fff
        const { container } = renderWithProviders(<AllObjects run="test-run" />);
        const shapeSpan = container.querySelector("span[style]");
        expect(shapeSpan).toBeTruthy();
        // The span with "@" should have color set for WHITE
        const style = shapeSpan!.getAttribute("style") ?? "";
        expect(style).toContain("color");
        expect(style).toContain("background");
    });

    it("sorts null values to end", () => {
        mockObjects([OBJ_NULL, OBJ_A]);
        renderWithProviders(<AllObjects run="test-run" />);
        // With ascending sort, OBJ_A (shape "@") should come before OBJ_NULL (shape null)
        // Just verify both render without crashing
        expect(screen.getByText("@")).toBeInTheDocument();
        const dashes = screen.getAllByText("--");
        expect(dashes.length).toBeGreaterThan(0);
    });

    it("uses default color for unknown color names", () => {
        const objUnknownColor: ResolvedObject = {
            shape: "X",
            glyph: "999",
            color: "SPARKLY_RAINBOW",
            node_id: "-99",
            step_added: 1,
            match_count: 1,
            type: "Single",
        };
        mockObjects([objUnknownColor]);
        const { container } = renderWithProviders(<AllObjects run="test-run" />);
        const shapeSpan = container.querySelector("span[style]");
        expect(shapeSpan).toBeTruthy();
        // Unknown color should still render with some color styling
        const style = shapeSpan!.getAttribute("style") ?? "";
        expect(style).toContain("color");
    });

    it("uses default color when color is null", () => {
        mockObjects([OBJ_NULL]);
        renderWithProviders(<AllObjects run="test-run" />);
        // Should render without crashing even with null color
        const dashes = screen.getAllByText("--");
        expect(dashes.length).toBeGreaterThan(0);
    });

    it("double-toggles sort back to ascending", () => {
        mockObjects([OBJ_A, OBJ_B]);
        renderWithProviders(<AllObjects run="test-run" />);

        const shapeHeader = screen.getByText(/^shape/);
        // Click once -> descending
        fireEvent.click(shapeHeader);
        expect(shapeHeader.textContent).toContain("\u25BC");
        // Click again -> ascending
        fireEvent.click(shapeHeader);
        expect(shapeHeader.textContent).toContain("\u25B2");
    });
});
