import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { TransitionPanel } from "./TransitionPanel";
import { PredictionPanel } from "./PredictionPanel";
import { SequencePanel } from "./SequencePanel";

describe("TransitionPanel", () => {
    it("shows 'No transform data' when data is undefined", () => {
        renderWithProviders(<TransitionPanel data={undefined} />);
        expect(screen.getByText("No transform data")).toBeInTheDocument();
    });

    it("shows 'No changes this step' when transform count is 0", () => {
        const data = makeStepData({
            transform_summary: { count: 0, changes: [] },
        });
        renderWithProviders(<TransitionPanel data={data} />);
        expect(screen.getByText("No changes this step")).toBeInTheDocument();
    });

    it("renders structured changes table with type/name/delta", () => {
        const data = makeStepData({
            transform_summary: {
                count: 2,
                changes: [
                    { description: "IntrinsicTransform('hp', -0.1)", type: "IntrinsicTransform", name: "hp", normalized_change: -0.1 },
                    { description: "IntrinsicTransform('hunger', 0.05)", type: "IntrinsicTransform", name: "hunger", normalized_change: 0.05 },
                ],
            },
        });
        renderWithProviders(<TransitionPanel data={data} />);
        expect(screen.getByText("2")).toBeInTheDocument();
        expect(screen.getByText("hp")).toBeInTheDocument();
        expect(screen.getByText("hunger")).toBeInTheDocument();
        expect(screen.getByText("-0.1000")).toBeInTheDocument();
        expect(screen.getByText("+0.0500")).toBeInTheDocument();
    });

    it("falls back to description-only when no type info", () => {
        const data = makeStepData({
            transform_summary: {
                count: 1,
                changes: [{ description: "something changed" }],
            },
        });
        renderWithProviders(<TransitionPanel data={data} />);
        expect(screen.getByText("something changed")).toBeInTheDocument();
    });

    it("handles old-format plain string changes", () => {
        const data = makeStepData({
            transform_summary: {
                count: 1,
                changes: ["Node(-755, labels={'Frame'})"],
            },
        });
        renderWithProviders(<TransitionPanel data={data} />);
        expect(screen.getByText("Node(-755, labels={'Frame'})")).toBeInTheDocument();
    });
});

describe("PredictionPanel", () => {
    it("shows 'No prediction data' when data is undefined", () => {
        renderWithProviders(<PredictionPanel data={undefined} />);
        expect(screen.getByText("No prediction data")).toBeInTheDocument();
    });

    it("shows PREDICTED badge when prediction.made is true", () => {
        const data = makeStepData({
            prediction: { made: true },
        });
        renderWithProviders(<PredictionPanel data={data} />);
        expect(screen.getByText("PREDICTED")).toBeInTheDocument();
    });

    it("shows NO PREDICTION badge when prediction.made is false", () => {
        const data = makeStepData({
            prediction: { made: false },
        });
        renderWithProviders(<PredictionPanel data={data} />);
        expect(screen.getByText("NO PREDICTION")).toBeInTheDocument();
    });

    it("shows candidate count and confidence", () => {
        const data = makeStepData({
            prediction: { made: true, candidate_count: 5, confidence: 0.87654 },
        });
        renderWithProviders(<PredictionPanel data={data} />);
        expect(screen.getByText("5")).toBeInTheDocument();
        expect(screen.getByText("0.877")).toBeInTheDocument();
    });

    it("does not render expmod inline (expmod is shown by Section)", () => {
        const data = makeStepData({
            prediction: { made: true, candidate_expmod: "my-candidate-mod", confidence_expmod: "my-conf-mod" },
        });
        renderWithProviders(<PredictionPanel data={data} />);
        expect(screen.queryByText("my-candidate-mod")).not.toBeInTheDocument();
        expect(screen.queryByText("my-conf-mod")).not.toBeInTheDocument();
    });
});

describe("SequencePanel", () => {
    it("shows 'No sequence data' when data is undefined", () => {
        renderWithProviders(<SequencePanel data={undefined} />);
        expect(screen.getByText("No sequence data")).toBeInTheDocument();
    });

    it("renders frame composition", () => {
        const data = makeStepData({
            sequence_summary: {
                tick: 42,
                object_count: 3,
                objects: [
                    { id: "abc12345" },
                    { id: "def67890", x: 5, y: 10, resolve_count: 7 },
                ],
                intrinsic_count: 2,
                intrinsics: { hp: 0.8, hunger: 0.3 },
                significance: 0.1234,
            },
        });
        renderWithProviders(<SequencePanel data={data} />);
        expect(screen.getByText("42")).toBeInTheDocument();
        expect(screen.getByText("3")).toBeInTheDocument();
        expect(screen.getByText("abc12345")).toBeInTheDocument();
        expect(screen.getByText("(5, 10)")).toBeInTheDocument();
        expect(screen.getByText("0.1234")).toBeInTheDocument();
    });
});
