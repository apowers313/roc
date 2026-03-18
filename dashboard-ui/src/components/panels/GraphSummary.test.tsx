import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { GraphSummary } from "./GraphSummary";

describe("GraphSummary", () => {
    it("shows 'No graph data' when data is undefined", () => {
        renderWithProviders(<GraphSummary data={undefined} />);
        expect(screen.getByText("No graph data")).toBeInTheDocument();
    });

    it("renders graph summary via KVTable", () => {
        const data = makeStepData({
            graph_summary: { nodes: 42, edges: 100 },
        });
        renderWithProviders(<GraphSummary data={data} />);
        expect(screen.getByText("Graph DB")).toBeInTheDocument();
        expect(screen.getByText("nodes")).toBeInTheDocument();
        expect(screen.getByText("42")).toBeInTheDocument();
    });
});
