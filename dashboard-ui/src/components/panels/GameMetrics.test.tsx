import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { GameMetrics } from "./GameMetrics";

describe("GameMetrics", () => {
    it("shows 'No metrics data' when data is undefined", () => {
        renderWithProviders(<GameMetrics data={undefined} />);
        expect(screen.getByText("No metrics data")).toBeInTheDocument();
    });

    it("renders metrics via KVTable", () => {
        const data = makeStepData({
            game_metrics: { hp: 20, score: 100 },
        });
        renderWithProviders(<GameMetrics data={data} />);
        expect(screen.getByText("Metrics")).toBeInTheDocument();
        expect(screen.getByText("hp")).toBeInTheDocument();
        expect(screen.getByText("20")).toBeInTheDocument();
    });
});
