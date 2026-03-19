import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeStepData, renderWithProviders } from "../../test-utils";
import { ResolutionInspector } from "./ResolutionInspector";

describe("ResolutionInspector", () => {
    it("shows 'No resolution data' when data is undefined", () => {
        renderWithProviders(<ResolutionInspector data={undefined} />);
        expect(screen.getByText("No resolution data")).toBeInTheDocument();
    });

    it("displays expmod badge above Resolution line", () => {
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
        const badge = screen.getByText("dirichlet-categorical");
        const resolution = screen.getByText("Resolution");
        // Badge should appear before Resolution text in DOM order
        expect(
            badge.compareDocumentPosition(resolution) & Node.DOCUMENT_POSITION_FOLLOWING,
        ).toBeTruthy();
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
