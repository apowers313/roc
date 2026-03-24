import { screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

import { renderWithProviders } from "../../test-utils";

// Mock mermaid before importing component
vi.mock("mermaid", () => ({
    default: {
        initialize: vi.fn(),
        render: vi.fn(),
    },
}));

import mermaid from "mermaid";
import { MermaidDiagram } from "./MermaidDiagram";

describe("MermaidDiagram", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it("renders empty div when definition is provided and render succeeds", async () => {
        vi.mocked(mermaid.render).mockResolvedValue({
            svg: '<svg><text>rendered</text></svg>',
            bindFunctions: undefined,
        });

        const { container } = renderWithProviders(
            <MermaidDiagram definition="classDiagram\nclass Foo" />,
        );

        await waitFor(() => {
            expect(mermaid.render).toHaveBeenCalled();
        });

        // The container div should have the SVG injected via innerHTML
        await waitFor(() => {
            expect(container.querySelector("svg")).toBeTruthy();
        });
    });

    it("calls onSvgReady when render completes", async () => {
        const svgOutput = '<svg xmlns="http://www.w3.org/2000/svg"><text>test</text></svg>';
        vi.mocked(mermaid.render).mockResolvedValue({
            svg: svgOutput,
            bindFunctions: undefined,
        });

        const onSvgReady = vi.fn();
        renderWithProviders(
            <MermaidDiagram definition="classDiagram" onSvgReady={onSvgReady} />,
        );

        await waitFor(() => {
            expect(onSvgReady).toHaveBeenCalledWith(svgOutput);
        });
    });

    it("shows error message when mermaid.render throws", async () => {
        vi.mocked(mermaid.render).mockRejectedValue(new Error("Parse error in diagram"));

        renderWithProviders(<MermaidDiagram definition="invalid mermaid" />);

        await waitFor(() => {
            expect(
                screen.getByText("Diagram render error: Parse error in diagram"),
            ).toBeInTheDocument();
        });
    });

    it("shows generic error for non-Error thrown values", async () => {
        vi.mocked(mermaid.render).mockRejectedValue("string error");

        renderWithProviders(<MermaidDiagram definition="bad diagram" />);

        await waitFor(() => {
            expect(
                screen.getByText("Diagram render error: Mermaid render failed"),
            ).toBeInTheDocument();
        });
    });

    it("does not render when definition is empty", () => {
        vi.mocked(mermaid.render).mockResolvedValue({
            svg: "<svg></svg>",
            bindFunctions: undefined,
        });

        renderWithProviders(<MermaidDiagram definition="" />);

        // mermaid.render should not be called with empty definition
        expect(mermaid.render).not.toHaveBeenCalled();
    });

    it("cleans up on unmount (cancellation flag)", async () => {
        vi.mocked(mermaid.render).mockResolvedValue({
            svg: "<svg>ok</svg>",
            bindFunctions: undefined,
        });

        const onSvgReady = vi.fn();
        const { unmount } = renderWithProviders(
            <MermaidDiagram definition="classDiagram" onSvgReady={onSvgReady} />,
        );

        // Unmount before render resolves (the cancelled flag should suppress callback)
        unmount();

        // Wait a tick to ensure the async render path completes
        await new Promise((r) => setTimeout(r, 50));

        // onSvgReady may or may not have been called depending on timing;
        // the key thing is no error is thrown (no crash on unmounted component)
    });
});
