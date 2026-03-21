import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { InventoryPanel } from "./InventoryPanel";

describe("InventoryPanel", () => {
    it("shows 'No inventory data' when data is undefined", () => {
        renderWithProviders(<InventoryPanel data={undefined} />);
        expect(screen.getByText("No inventory data")).toBeInTheDocument();
    });

    it("shows 'No inventory data' when inventory is null", () => {
        renderWithProviders(<InventoryPanel data={makeStepData()} />);
        expect(screen.getByText("No inventory data")).toBeInTheDocument();
    });

    it("shows 'No inventory data' when inventory is empty", () => {
        renderWithProviders(<InventoryPanel data={makeStepData({ inventory: [] })} />);
        expect(screen.getByText("No inventory data")).toBeInTheDocument();
    });

    it("renders table with column headers when inventory has items", () => {
        const data = makeStepData({
            inventory: [
                { letter: "a", item: "a +0 dagger (weapon in hand)", glyph: 1234 },
            ],
        });
        renderWithProviders(<InventoryPanel data={data} />);
        expect(screen.getByText("Slot")).toBeInTheDocument();
        expect(screen.getByText("Item")).toBeInTheDocument();
    });

    it("renders inventory items with slot letter and item name", () => {
        const data = makeStepData({
            inventory: [
                { letter: "a", item: "a +0 dagger (weapon in hand)", glyph: 1234 },
                { letter: "b", item: "a blessed +0 leather armor (being worn)", glyph: 5678 },
            ],
        });
        renderWithProviders(<InventoryPanel data={data} />);
        expect(screen.getByText("a")).toBeInTheDocument();
        expect(screen.getByText("a +0 dagger (weapon in hand)")).toBeInTheDocument();
        expect(screen.getByText("b")).toBeInTheDocument();
        expect(screen.getByText("a blessed +0 leather armor (being worn)")).toBeInTheDocument();
    });

    it("renders multiple inventory items as table rows", () => {
        const data = makeStepData({
            inventory: [
                { letter: "a", item: "dagger", glyph: 100 },
                { letter: "b", item: "armor", glyph: 200 },
                { letter: "c", item: "ring", glyph: 300 },
            ],
        });
        const { container } = renderWithProviders(<InventoryPanel data={data} />);
        // Should have 3 body rows plus 1 header row
        const rows = container.querySelectorAll("tr");
        expect(rows.length).toBe(4); // 1 header + 3 data rows
    });
});
