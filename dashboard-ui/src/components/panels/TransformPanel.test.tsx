import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { TransformPanel } from "./TransformPanel";

describe("TransformPanel", () => {
    it("shows 'No transform data' when data is undefined", () => {
        renderWithProviders(<TransformPanel data={undefined} />);
        expect(screen.getByText("No transform data")).toBeInTheDocument();
    });

    it("shows prediction '--' when no prediction data", () => {
        renderWithProviders(<TransformPanel data={makeStepData()} />);
        expect(screen.getByText("Prediction")).toBeInTheDocument();
        expect(screen.getByText("--")).toBeInTheDocument();
    });

    it("shows PREDICTED badge when prediction.made is true", () => {
        const data = makeStepData({
            prediction: { made: true },
        });
        renderWithProviders(<TransformPanel data={data} />);
        expect(screen.getByText("PREDICTED")).toBeInTheDocument();
    });

    it("shows NO PREDICTION badge when prediction.made is false", () => {
        const data = makeStepData({
            prediction: { made: false },
        });
        renderWithProviders(<TransformPanel data={data} />);
        expect(screen.getByText("NO PREDICTION")).toBeInTheDocument();
    });

    it("shows candidate count when provided", () => {
        const data = makeStepData({
            prediction: { made: true, candidates: 5 },
        });
        renderWithProviders(<TransformPanel data={data} />);
        expect(screen.getByText("5 candidates")).toBeInTheDocument();
    });

    it("shows confidence when provided", () => {
        const data = makeStepData({
            prediction: { made: true, confidence: 0.87654 },
        });
        renderWithProviders(<TransformPanel data={data} />);
        expect(screen.getByText("conf: 0.877")).toBeInTheDocument();
    });

    it("shows candidate_expmod badge when provided", () => {
        const data = makeStepData({
            prediction: { made: true, candidate_expmod: "my-candidate-mod" },
        });
        renderWithProviders(<TransformPanel data={data} />);
        expect(screen.getByText("my-candidate-mod")).toBeInTheDocument();
    });

    it("shows confidence_expmod badge when provided", () => {
        const data = makeStepData({
            prediction: { made: true, confidence_expmod: "my-conf-mod" },
        });
        renderWithProviders(<TransformPanel data={data} />);
        expect(screen.getByText("my-conf-mod")).toBeInTheDocument();
    });

    it("shows 'No changes this step' when transform count is 0", () => {
        const data = makeStepData({
            transform_summary: { count: 0, changes: [] },
        });
        renderWithProviders(<TransformPanel data={data} />);
        expect(screen.getByText("No changes this step")).toBeInTheDocument();
    });

    it("renders transform changes table when changes present", () => {
        const data = makeStepData({
            transform_summary: {
                count: 2,
                changes: ["Object -10 moved from (3,5) to (4,5)", "Object -14 color changed"],
            },
        });
        renderWithProviders(<TransformPanel data={data} />);
        expect(screen.getByText("#")).toBeInTheDocument();
        expect(screen.getByText("Change")).toBeInTheDocument();
        expect(screen.getByText("Object -10 moved from (3,5) to (4,5)")).toBeInTheDocument();
        expect(screen.getByText("Object -14 color changed")).toBeInTheDocument();
    });

    it("shows row numbers in changes table", () => {
        const data = makeStepData({
            transform_summary: {
                count: 2,
                changes: ["change one", "change two"],
            },
        });
        renderWithProviders(<TransformPanel data={data} />);
        expect(screen.getByText("1")).toBeInTheDocument();
        expect(screen.getByText("2")).toBeInTheDocument();
    });

    it("renders prediction and transform together", () => {
        const data = makeStepData({
            prediction: { made: true, candidates: 3, confidence: 0.95 },
            transform_summary: { count: 1, changes: ["something changed"] },
        });
        renderWithProviders(<TransformPanel data={data} />);
        expect(screen.getByText("PREDICTED")).toBeInTheDocument();
        expect(screen.getByText("3 candidates")).toBeInTheDocument();
        expect(screen.getByText("conf: 0.950")).toBeInTheDocument();
        expect(screen.getByText("something changed")).toBeInTheDocument();
    });
});
