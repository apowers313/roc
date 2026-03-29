import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { TransitionPanel } from "./TransitionPanel";
import { PredictionPanel } from "./PredictionPanel";
import { SequencePanel } from "./SequencePanel";

// ObjectLink (used by TransitionPanel and SequencePanel) imports ObjectModal
// which calls useObjectHistory
vi.mock("../../api/queries", () => ({
    useObjectHistory: vi.fn().mockReturnValue({
        data: { info: { uuid: 1, resolve_count: 0 }, states: [], transforms: [] },
        isLoading: false,
    }),
}));

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

    it("renders intrinsic changes in three-column layout", () => {
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
        expect(screen.getByText("Intrinsic Changes")).toBeInTheDocument();
        expect(screen.getByText("hp")).toBeInTheDocument();
        expect(screen.getByText("hunger")).toBeInTheDocument();
        expect(screen.getByText("-0.1000")).toBeInTheDocument();
        expect(screen.getByText("+0.0500")).toBeInTheDocument();
    });

    it("falls back to description when no name info", () => {
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

    it("shows object changes as summary rows", () => {
        const data = makeStepData({
            transform_summary: {
                count: 0,
                changes: [],
                object_transforms: [
                    { uuid: 42, glyph: "@", changes: [
                        { property: "x", type: "continuous", delta: 2, old_value: 10, new_value: 12 },
                    ]},
                ],
            },
        });
        renderWithProviders(<TransitionPanel data={data} />);
        expect(screen.getByText("Object Changes")).toBeInTheDocument();
        expect(screen.getByText("@")).toBeInTheDocument();
        // Three-column: Previous, Current, Delta headers
        expect(screen.getByText("Previous", { exact: false })).toBeInTheDocument();
        expect(screen.getByText("Current", { exact: false })).toBeInTheDocument();
        expect(screen.getByText("Delta")).toBeInTheDocument();
    });

    it("handles empty object transforms gracefully", () => {
        const data = makeStepData({
            transform_summary: {
                count: 1,
                changes: [{ description: "foo" }],
            },
        });
        renderWithProviders(<TransitionPanel data={data} />);
        expect(screen.queryByText("Object Changes")).not.toBeInTheDocument();
    });

    it("shows no changes when count is 0 and no object transforms", () => {
        const data = makeStepData({
            transform_summary: { count: 0, changes: [], object_transforms: [] },
        });
        renderWithProviders(<TransitionPanel data={data} />);
        expect(screen.getByText("No changes this step")).toBeInTheDocument();
    });

    it("shows object delta summary for multiple changes", () => {
        const data = makeStepData({
            transform_summary: {
                count: 0,
                changes: [],
                object_transforms: [{
                    uuid: 42,
                    glyph: "@",
                    changes: [
                        { property: "x", type: "continuous", delta: 2 },
                        { property: "color_type", type: "discrete", delta: null, old_value: 7, new_value: 3 },
                    ],
                }],
            },
        });
        renderWithProviders(<TransitionPanel data={data} />);
        // Delta column shows summary of all changes
        expect(screen.getByText(/x: \+2/)).toBeInTheDocument();
        expect(screen.getByText(/color_type: 7 -> 3/)).toBeInTheDocument();
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
        // Single-line header: "Frame: tick=42 | 3 objects | 2 intrinsics | significance=0.1234"
        expect(screen.getByText(/tick=42/)).toBeInTheDocument();
        expect(screen.getByText(/3 objects/)).toBeInTheDocument();
        expect(screen.getByText(/significance=0.1234/)).toBeInTheDocument();
        expect(screen.getByText("(5, 10)")).toBeInTheDocument();
    });

    it("shows glyph column for objects", () => {
        const data = makeStepData({
            sequence_summary: {
                tick: 1,
                object_count: 1,
                objects: [
                    { id: "abc12345", x: 5, y: 3, glyph: "@", resolve_count: 2 },
                ],
                intrinsic_count: 0,
                intrinsics: {},
            },
        });
        renderWithProviders(<SequencePanel data={data} />);
        expect(screen.getByText("Glyph")).toBeInTheDocument();
        expect(screen.getByText("@")).toBeInTheDocument();
    });

    it("shows -- for missing glyph", () => {
        const data = makeStepData({
            sequence_summary: {
                tick: 1,
                object_count: 1,
                objects: [{ id: "abc12345" }],
                intrinsic_count: 0,
                intrinsics: {},
            },
        });
        renderWithProviders(<SequencePanel data={data} />);
        // The glyph column shows "--" for missing glyph
        const cells = screen.getAllByText("--");
        expect(cells.length).toBeGreaterThan(0);
    });

    it("shows color and shape columns", () => {
        const data = makeStepData({
            sequence_summary: {
                tick: 1,
                object_count: 1,
                objects: [{ id: "abc12345", x: 5, y: 3, glyph: "@", color: 7, shape: 64 }],
                intrinsic_count: 0,
                intrinsics: {},
            },
        });
        renderWithProviders(<SequencePanel data={data} />);
        expect(screen.getByText("Color")).toBeInTheDocument();
        expect(screen.getByText("Shape")).toBeInTheDocument();
        expect(screen.getByText("7")).toBeInTheDocument();
        expect(screen.getByText("64")).toBeInTheDocument();
    });

    it("shows matched_previous column", () => {
        const data = makeStepData({
            sequence_summary: {
                tick: 1,
                object_count: 1,
                objects: [{ id: "abc12345", matched_previous: true }],
                intrinsic_count: 0,
                intrinsics: {},
            },
        });
        renderWithProviders(<SequencePanel data={data} />);
        expect(screen.getByText("Matched")).toBeInTheDocument();
        expect(screen.getByText("Yes")).toBeInTheDocument();
    });
});
