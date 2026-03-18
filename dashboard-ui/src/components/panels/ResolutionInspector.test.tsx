import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { ResolutionInspector } from "./ResolutionInspector";
import type { StepData } from "../../types/step-data";

function makeData(resolution: Record<string, unknown> | null): StepData {
    return {
        step: 1,
        game_number: 1,
        timestamp: null,
        screen: null,
        saliency: null,
        features: null,
        object_info: null,
        focus_points: null,
        attenuation: null,
        resolution_metrics: resolution,
        graph_summary: null,
        event_summary: null,
        game_metrics: null,
        logs: null,
    };
}

describe("ResolutionInspector", () => {
    it("shows empty state when no data", () => {
        renderWithProviders(<ResolutionInspector data={undefined} />);
        expect(screen.getByText("No resolution data")).toBeInTheDocument();
    });

    it("shows empty state when resolution_metrics is null", () => {
        renderWithProviders(<ResolutionInspector data={makeData(null)} />);
        expect(screen.getByText("No resolution data")).toBeInTheDocument();
    });

    it("shows MATCHED badge for match outcome", () => {
        renderWithProviders(
            <ResolutionInspector
                data={makeData({
                    outcome: "match",
                    algorithm: "dirichlet",
                    matched_object_id: 42,
                    num_candidates: 3,
                })}
            />,
        );
        expect(screen.getByText("MATCHED")).toBeInTheDocument();
        expect(screen.getByText("dirichlet")).toBeInTheDocument();
        expect(screen.getByText("42")).toBeInTheDocument();
        expect(screen.getByText("3")).toBeInTheDocument();
    });

    it("shows NEW OBJECT badge", () => {
        renderWithProviders(
            <ResolutionInspector data={makeData({ outcome: "new_object" })} />,
        );
        expect(screen.getByText("NEW OBJECT")).toBeInTheDocument();
    });

    it("shows LOW CONFIDENCE badge", () => {
        renderWithProviders(
            <ResolutionInspector data={makeData({ outcome: "low_confidence" })} />,
        );
        expect(screen.getByText("LOW CONFIDENCE")).toBeInTheDocument();
    });

    it("shows location when x and y present", () => {
        renderWithProviders(
            <ResolutionInspector
                data={makeData({ outcome: "match", x: 5, y: 10 })}
            />,
        );
        expect(screen.getByText("(5, 10)")).toBeInTheDocument();
    });

    it("renders candidates table with posteriors", () => {
        renderWithProviders(
            <ResolutionInspector
                data={makeData({
                    outcome: "match",
                    posteriors: [
                        ["obj-1", 0.85],
                        ["obj-2", 0.15],
                    ],
                })}
            />,
        );
        expect(screen.getByText("Candidates")).toBeInTheDocument();
        expect(screen.getByText("probability")).toBeInTheDocument();
        expect(screen.getByText("obj-1")).toBeInTheDocument();
        expect(screen.getByText("0.8500")).toBeInTheDocument();
    });

    it("renders candidates table with distances", () => {
        renderWithProviders(
            <ResolutionInspector
                data={makeData({
                    outcome: "match",
                    candidate_distances: [
                        ["obj-a", 1.234],
                        ["obj-b", 5.678],
                    ],
                })}
            />,
        );
        expect(screen.getByText("distance")).toBeInTheDocument();
        expect(screen.getByText("obj-a")).toBeInTheDocument();
        expect(screen.getByText("1.2340")).toBeInTheDocument();
    });

    it("renders feature list", () => {
        renderWithProviders(
            <ResolutionInspector
                data={makeData({
                    outcome: "match",
                    features: ["color", "shape", "size"],
                })}
            />,
        );
        expect(screen.getByText("Features")).toBeInTheDocument();
        expect(screen.getByText("color, shape, size")).toBeInTheDocument();
    });

    it("truncates long feature lists", () => {
        const features = Array.from({ length: 25 }, (_, i) => `f${i}`);
        renderWithProviders(
            <ResolutionInspector data={makeData({ outcome: "match", features })} />,
        );
        expect(screen.getByText(/\+5\)/)).toBeInTheDocument();
    });
});
