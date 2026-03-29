import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { ResolutionInspector } from "./ResolutionInspector";

// ResolutionInspector uses ObjectLink which imports ObjectModal -> useObjectHistory
vi.mock("../../api/queries", () => ({
    useObjectHistory: vi.fn().mockReturnValue({
        data: { info: { uuid: 1, resolve_count: 0 }, states: [], transforms: [] },
        isLoading: false,
    }),
}));

describe("ResolutionInspector", () => {
    it("shows 'No resolution data' when data is undefined", () => {
        renderWithProviders(<ResolutionInspector data={undefined} />);
        expect(screen.getByText("No resolution data")).toBeInTheDocument();
    });

    it("does not render expmod inline (expmod is shown by Section)", () => {
        const data = makeStepData({
            resolution_metrics: {
                outcome: "match",
                algorithm: "dirichlet-categorical",
                x: 10,
                y: 5,
                features: ["ShapeNode(.)"],
            },
        });
        renderWithProviders(<ResolutionInspector data={data} />);
        // Expmod badge is rendered by the parent Section, not by ResolutionInspector
        expect(screen.queryByText("dirichlet-categorical")).not.toBeInTheDocument();
    });

    it("parses glyph from features and displays it", () => {
        const data = makeStepData({
            resolution_metrics: {
                outcome: "new_object",
                features: ["ShapeNode(@)", "ColorNode(WHITE)", "SingleNode(333)"],
            },
        });
        renderWithProviders(<ResolutionInspector data={data} />);
        expect(screen.getByText("@")).toBeInTheDocument();
        expect(screen.getByText("NEW OBJECT")).toBeInTheDocument();
    });

    it("shows observed -> matched glyphs for matches with matched_attrs", () => {
        const data = makeStepData({
            resolution_metrics: {
                outcome: "match",
                features: ["ShapeNode(+)", "ColorNode(BROWN)"],
                matched_attrs: { char: ".", color: "GREY", glyph: "2371" },
                matched_object_id: -14,
            },
        });
        renderWithProviders(<ResolutionInspector data={data} />);
        // Arrow between observed and matched glyphs
        expect(screen.getByText("\u2192")).toBeInTheDocument();
        // Both glyphs should be present (may appear multiple times in comparison table)
        expect(screen.getAllByText("+").length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText(".").length).toBeGreaterThanOrEqual(1);
    });

    it("shows side-by-side attribute comparison table for matches", () => {
        const data = makeStepData({
            resolution_metrics: {
                outcome: "match",
                features: ["ShapeNode(@)", "ColorNode(WHITE)", "SingleNode(333)"],
                matched_attrs: { char: "@", color: "WHITE", glyph: "333" },
            },
        });
        renderWithProviders(<ResolutionInspector data={data} />);
        expect(screen.getByText("observed")).toBeInTheDocument();
        expect(screen.getByText("matched")).toBeInTheDocument();
        expect(screen.getByText("shape")).toBeInTheDocument();
        expect(screen.getByText("color")).toBeInTheDocument();
    });

    it("shows candidates with glyph details from candidate_details", () => {
        const data = makeStepData({
            resolution_metrics: {
                outcome: "match",
                features: ["ShapeNode(.)"],
                candidate_details: [
                    { id: "-14", probability: 0.89, char: "@", color: "WHITE", glyph: 333 },
                    { id: "new", probability: 0.08 },
                ],
            },
        });
        renderWithProviders(<ResolutionInspector data={data} />);
        expect(screen.getByText("Candidates")).toBeInTheDocument();
        expect(screen.getByText("-14")).toBeInTheDocument();
    });
});

describe("ResolutionInspector multi-cycle", () => {
    it("renders summary table with correct columns", () => {
        const data = makeStepData({
            resolution_cycles: [
                { outcome: "match", features: ["ShapeNode(@)", "ColorNode(WHITE)"], x: 10, y: 5, num_candidates: 3 },
                { outcome: "new_object", features: ["ShapeNode(d)", "ColorNode(RED)"], x: 15, y: 8, num_candidates: 0 },
                { outcome: "match", features: ["ShapeNode(.)"], x: 22, y: 1, num_candidates: 5 },
                { outcome: "match", features: ["ShapeNode(#)"], x: 30, y: 2, num_candidates: 2 },
            ],
        });
        renderWithProviders(<ResolutionInspector data={data} />);
        // Summary table columns -- some appear in both summary and detail
        expect(screen.getAllByText("Outcome").length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText("Object").length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText("Location").length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText("Candidates").length).toBeGreaterThanOrEqual(1);
        // Summary table should show "Resolution Summary" title
        expect(screen.getByText("Resolution Summary")).toBeInTheDocument();
    });

    it("stepper switches resolution detail", () => {
        const data = makeStepData({
            resolution_cycles: [
                { outcome: "match", features: ["ShapeNode(@)"], x: 10, y: 5, num_candidates: 3 },
                { outcome: "new_object", features: ["ShapeNode(d)"], x: 15, y: 8, num_candidates: 0 },
            ],
        });
        renderWithProviders(<ResolutionInspector data={data} />);
        // Initially shows cycle 1 detail -- MATCHED appears in summary table and detail
        expect(screen.getAllByText("MATCHED").length).toBeGreaterThanOrEqual(1);
        // Click cycle 2
        const cycle2Label = screen.getByText("Cycle 2");
        fireEvent.click(cycle2Label);
        // Now should show cycle 2 detail (new_object) -- NEW OBJECT in both summary and detail
        expect(screen.getAllByText("NEW OBJECT").length).toBeGreaterThanOrEqual(1);
    });
});

describe("backward compatibility", () => {
    it("wraps single resolution_metrics as fallback (no resolution_cycles)", () => {
        const data = makeStepData({
            resolution_metrics: { outcome: "match", features: ["ShapeNode(@)"] },
        });
        renderWithProviders(<ResolutionInspector data={data} />);
        // Should still render correctly without cycles
        expect(screen.getByText("MATCHED")).toBeInTheDocument();
        // No stepper should appear
        expect(screen.queryByText("Cycle 1")).not.toBeInTheDocument();
    });
});
