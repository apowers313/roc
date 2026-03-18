import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { FeatureTable } from "./FeatureTable";

describe("FeatureTable", () => {
    it("shows 'No feature data' when data is undefined", () => {
        renderWithProviders(<FeatureTable data={undefined} />);
        expect(screen.getByText("No feature data")).toBeInTheDocument();
    });

    it("shows 'No feature data' when features is empty array", () => {
        renderWithProviders(
            <FeatureTable data={makeStepData({ features: [] })} />,
        );
        expect(screen.getByText("No feature data")).toBeInTheDocument();
    });

    it("renders feature counts from dict entry", () => {
        const data = makeStepData({
            features: [{ Flood: 4, Line: 114, Single: 10 }],
        });
        renderWithProviders(<FeatureTable data={data} />);
        expect(screen.getByText("Feature Counts")).toBeInTheDocument();
        expect(screen.getByText("Flood")).toBeInTheDocument();
        expect(screen.getByText("4")).toBeInTheDocument();
    });

    it("parses raw string feature data", () => {
        const data = makeStepData({
            features: [
                { raw: "\t\tFlood: 4\n\t\tLine: 114\n\t\tSingle: 10" },
            ],
        });
        renderWithProviders(<FeatureTable data={data} />);
        expect(screen.getByText("4")).toBeInTheDocument();
        expect(screen.getByText("114")).toBeInTheDocument();
    });

    it("shows -- for missing feature types", () => {
        const data = makeStepData({
            features: [{ Flood: 5 }],
        });
        renderWithProviders(<FeatureTable data={data} />);
        // Line, Single, etc. should show -- since they're not present
        const dashes = screen.getAllByText("--");
        expect(dashes.length).toBeGreaterThan(0);
    });
});
