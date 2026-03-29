import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { ObjectLink } from "./ObjectLink";

vi.mock("../../api/queries", () => ({
    useObjectHistory: vi.fn().mockReturnValue({
        data: {
            info: { uuid: 42, resolve_count: 5 },
            states: [{ tick: 1, x: 10, y: 5, glyph_type: 333, color_type: 7, shape_type: 64, distance: null, flood_size: null, line_size: null, motion_direction: null, delta_old: null, delta_new: null }],
            transforms: [],
        },
        isLoading: false,
    }),
}));

describe("ObjectLink", () => {
    it("renders glyph badge with color", () => {
        renderWithProviders(<ObjectLink objectId={42} glyph="@" color="WHITE" />);
        expect(screen.getByText("@")).toBeInTheDocument();
    });

    it("applies cursor pointer style", () => {
        renderWithProviders(<ObjectLink objectId={42} glyph="@" color="WHITE" />);
        const badge = screen.getByText("@");
        expect(badge).toHaveStyle({ cursor: "pointer" });
    });

    it("opens Object Modal on click", () => {
        renderWithProviders(<ObjectLink objectId={42} glyph="@" color="WHITE" />);
        fireEvent.click(screen.getByText("@"));
        expect(screen.getByText(/Object:/)).toBeInTheDocument();
    });

    it("sets title from label prop", () => {
        renderWithProviders(<ObjectLink objectId={42} glyph="@" color="WHITE" label="player" />);
        expect(screen.getByTitle("player")).toBeInTheDocument();
    });

    it("uses default color when color is not provided", () => {
        renderWithProviders(<ObjectLink objectId={42} glyph="." />);
        expect(screen.getByText(".")).toBeInTheDocument();
    });
});
