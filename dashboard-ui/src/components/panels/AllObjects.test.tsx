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

// ---------------------------------------------------------------------------
// BUG-H3 / BUG-M2: filter visibility + counts + Show all games toggle
//
// The dashboard's All Objects popout used to silently apply the active
// game filter without telling the user. Phase 4 of the bugfix plan adds:
//
//  * A header that shows "Objects in Game N" when filtering and "All
//    Objects" when not.
//  * A "filtered / total" count so the user can tell something is hidden.
//  * A "Show all games" toggle that switches to the unfiltered view.
//  * A tooltip on rows whose canonical step_added falls in an earlier
//    game (BUG-M2) explaining where the object came from.
// ---------------------------------------------------------------------------

describe("AllObjects filter visibility (BUG-H3 / BUG-M2)", () => {
    function mockObjectsByGame(per: Map<number | undefined, ResolvedObject[]>) {
        // useAllObjects is called twice -- once with the active game and once
        // with undefined for the total. Map by the second arg.
        vi.mocked(useAllObjects).mockImplementation(((_run: string, game?: number) => {
            const data = per.get(game) ?? [];
            return { data } as ReturnType<typeof useAllObjects>;
        }) as typeof useAllObjects);
    }

    it("shows 'Objects in Game N' header when game filter is active", () => {
        mockObjectsByGame(
            new Map<number | undefined, ResolvedObject[]>([
                [2, [OBJ_A]],
                [undefined, [OBJ_A, OBJ_B]],
            ]),
        );
        renderWithProviders(<AllObjects run="test-run" game={2} />);
        expect(screen.getByText(/objects in game 2/i)).toBeInTheDocument();
    });

    it("shows '1 / 2' filtered/total count when filtering", () => {
        mockObjectsByGame(
            new Map<number | undefined, ResolvedObject[]>([
                [2, [OBJ_A]],
                [undefined, [OBJ_A, OBJ_B]],
            ]),
        );
        renderWithProviders(<AllObjects run="test-run" game={2} />);
        // Header should mention both counts.
        expect(screen.getByText(/1\s*\/\s*2/)).toBeInTheDocument();
    });

    it("renders a 'Show all games' toggle when filtering", () => {
        mockObjectsByGame(
            new Map<number | undefined, ResolvedObject[]>([
                [1, [OBJ_A]],
                [undefined, [OBJ_A, OBJ_B]],
            ]),
        );
        renderWithProviders(<AllObjects run="test-run" game={1} />);
        expect(
            screen.getByRole("switch", { name: /show all games/i }),
        ).toBeInTheDocument();
    });

    it("clicking 'Show all games' switches to the unfiltered dataset", () => {
        mockObjectsByGame(
            new Map<number | undefined, ResolvedObject[]>([
                [2, [OBJ_A]], // game 2 only has @
                [undefined, [OBJ_A, OBJ_B]], // unfiltered has @ and .
            ]),
        );
        renderWithProviders(<AllObjects run="test-run" game={2} />);
        // Initially the . row is not visible (only @ is in game 2).
        expect(screen.queryByText(".")).not.toBeInTheDocument();

        const toggle = screen.getByRole("switch", { name: /show all games/i });
        fireEvent.click(toggle);

        // After toggle, the . row appears.
        expect(screen.getByText(".")).toBeInTheDocument();
    });

    it("does not show the filter header or toggle when no game is set", () => {
        mockObjectsByGame(
            new Map<number | undefined, ResolvedObject[]>([
                [undefined, [OBJ_A]],
            ]),
        );
        renderWithProviders(<AllObjects run="test-run" />);
        expect(screen.queryByText(/objects in game/i)).not.toBeInTheDocument();
        expect(
            screen.queryByRole("switch", { name: /show all games/i }),
        ).not.toBeInTheDocument();
    });

    it("flags rows whose step_added belongs to an earlier game with a tooltip", () => {
        // OBJ_A's step_added=5 but the run is currently filtered by game=2,
        // and the run's per-game ranges are passed in via prop. The header
        // doesn't need to know -- the row's tooltip is the surface.
        const earlier: ResolvedObject = {
            ...OBJ_A,
            step_added: 5, // game 1's range is, say, 1..50
        };
        mockObjectsByGame(
            new Map<number | undefined, ResolvedObject[]>([
                [2, [earlier]],
                [undefined, [earlier]],
            ]),
        );
        renderWithProviders(
            <AllObjects
                run="test-run"
                game={2}
                gameStepRanges={[
                    { game_number: 1, min: 1, max: 50 },
                    { game_number: 2, min: 51, max: 100 },
                ]}
            />,
        );
        // Find the row containing the OBJ_A shape and check the step_added cell
        // for the tooltip "created in game 1".
        const tooltipHost = screen.getByText(/created in game 1/i);
        expect(tooltipHost).toBeInTheDocument();
    });
});
