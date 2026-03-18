import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { GraphSummary } from "./GraphSummary";

describe("GraphSummary", () => {
    it("shows 'No graph data' when data is undefined", () => {
        renderWithProviders(<GraphSummary data={undefined} />);
        expect(screen.getByText("No graph data")).toBeInTheDocument();
    });

    it("renders cache gauge bars when data is present", () => {
        const data = makeStepData({
            graph_summary: { node_count: 42, node_max: 100, edge_count: 75, edge_max: 200 },
        });
        renderWithProviders(<GraphSummary data={data} />);
        expect(screen.getByText("Graph DB")).toBeInTheDocument();
        expect(screen.getByText("Nodes")).toBeInTheDocument();
        expect(screen.getByText("42 / 100")).toBeInTheDocument();
        expect(screen.getByText("Edges")).toBeInTheDocument();
        expect(screen.getByText("75 / 200")).toBeInTheDocument();
    });

    it("handles zero max values gracefully", () => {
        const data = makeStepData({
            graph_summary: { node_count: 0, node_max: 0, edge_count: 0, edge_max: 0 },
        });
        renderWithProviders(<GraphSummary data={data} />);
        expect(screen.getAllByText("0 / 0")).toHaveLength(2);
    });
});
