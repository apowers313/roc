import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { GraphHistory } from "./GraphHistory";

vi.mock("../../api/queries", () => ({
    useGraphHistory: vi.fn(),
}));

import { useGraphHistory } from "../../api/queries";

describe("GraphHistory", () => {
    it("shows empty message when no data", () => {
        vi.mocked(useGraphHistory).mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useGraphHistory>);

        renderWithProviders(
            <GraphHistory run="test-run" currentStep={1} />,
        );
        expect(screen.getByText("No graph history")).toBeInTheDocument();
    });

    it("shows empty message when data is empty array", () => {
        vi.mocked(useGraphHistory).mockReturnValue({
            data: [],
        } as unknown as ReturnType<typeof useGraphHistory>);

        renderWithProviders(
            <GraphHistory run="test-run" currentStep={1} />,
        );
        expect(screen.getByText("No graph history")).toBeInTheDocument();
    });

    it("renders chart container when data is present", () => {
        vi.mocked(useGraphHistory).mockReturnValue({
            data: [
                { step: 1, node_count: 10, node_max: 100, edge_count: 20, edge_max: 200 },
                { step: 2, node_count: 15, node_max: 100, edge_count: 30, edge_max: 200 },
            ],
        } as unknown as ReturnType<typeof useGraphHistory>);

        const { container } = renderWithProviders(
            <GraphHistory run="test-run" currentStep={1} />,
        );
        expect(screen.getByText("Graph Cache")).toBeInTheDocument();
        expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
    });
});
