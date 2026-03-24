import { fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { DiagramViewer } from "./DiagramViewer";

// Mock MermaidDiagram to avoid mermaid rendering in jsdom
vi.mock("./MermaidDiagram", () => ({
    MermaidDiagram: ({
        definition,
        onSvgReady,
    }: {
        definition: string;
        onSvgReady?: (svg: string) => void;
    }) => (
        <div data-testid="mermaid-diagram" data-definition={definition}>
            <button
                data-testid="trigger-svg-ready"
                onClick={() => onSvgReady?.('<svg xmlns="http://www.w3.org/2000/svg"><text>Test</text></svg>')}
            >
                Trigger SVG
            </button>
        </div>
    ),
}));

// Mock react-zoom-pan-pinch -- provide basic passthrough wrappers
vi.mock("react-zoom-pan-pinch", () => ({
    TransformWrapper: ({
        children,
    }: {
        children: React.ReactNode;
    }) => <div data-testid="transform-wrapper">{children}</div>,
    TransformComponent: ({
        children,
    }: {
        children: React.ReactNode;
    }) => <div data-testid="transform-component">{children}</div>,
}));

// jsdom does not have URL.createObjectURL / revokeObjectURL -- stub them globally
beforeEach(() => {
    if (!URL.createObjectURL) {
        URL.createObjectURL = vi.fn().mockReturnValue("blob:mock-url");
    }
    if (!URL.revokeObjectURL) {
        URL.revokeObjectURL = vi.fn();
    }
});

/** Helper: get all ActionIcon buttons (Mantine renders them as plain <button>). */
function getActionButtons() {
    // All ActionIcon buttons have the mantine-ActionIcon-root class
    return document.querySelectorAll<HTMLButtonElement>("button.mantine-ActionIcon-root");
}

describe("DiagramViewer", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        // Re-stub after restoreAllMocks
        URL.createObjectURL = vi.fn().mockReturnValue("blob:mock-url");
        URL.revokeObjectURL = vi.fn();
    });

    it("renders Diagram label and usage hint", () => {
        renderWithProviders(<DiagramViewer definition="classDiagram" />);
        expect(screen.getByText("Diagram")).toBeInTheDocument();
        expect(screen.getByText("Scroll to zoom, drag to pan")).toBeInTheDocument();
    });

    it("renders the MermaidDiagram with the provided definition", () => {
        renderWithProviders(<DiagramViewer definition="classDiagram" />);
        const diagram = screen.getByTestId("mermaid-diagram");
        expect(diagram.dataset.definition).toBe("classDiagram");
    });

    it("renders five toolbar action buttons (zoom-in, zoom-out, reset, download, open)", () => {
        renderWithProviders(<DiagramViewer definition="classDiagram" />);
        const buttons = getActionButtons();
        expect(buttons.length).toBe(5);
    });

    it("disables download and open buttons when no SVG is ready", () => {
        renderWithProviders(<DiagramViewer definition="classDiagram" />);
        const buttons = getActionButtons();
        // buttons[3] = download, buttons[4] = open-in-new-window
        expect(buttons[3]).toBeDisabled();
        expect(buttons[4]).toBeDisabled();
    });

    it("enables download and open buttons after SVG is ready", async () => {
        renderWithProviders(<DiagramViewer definition="classDiagram" />);

        // Trigger the onSvgReady callback
        fireEvent.click(screen.getByTestId("trigger-svg-ready"));

        await waitFor(() => {
            const buttons = getActionButtons();
            expect(buttons[3]).not.toBeDisabled();
            expect(buttons[4]).not.toBeDisabled();
        });
    });

    it("triggers download when download button is clicked", async () => {
        const clickSpy = vi.fn();
        const createElementOrig = document.createElement.bind(document);
        vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
            const el = createElementOrig(tag);
            if (tag === "a") {
                Object.defineProperty(el, "click", { value: clickSpy });
            }
            return el;
        });

        renderWithProviders(<DiagramViewer definition="classDiagram" filename="test-schema" />);

        fireEvent.click(screen.getByTestId("trigger-svg-ready"));

        await waitFor(() => {
            const buttons = getActionButtons();
            expect(buttons[3]).not.toBeDisabled();
        });

        const buttons = getActionButtons();
        fireEvent.click(buttons[3]);

        expect(URL.createObjectURL).toHaveBeenCalled();
        expect(clickSpy).toHaveBeenCalled();
        expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:mock-url");
    });

    it("opens SVG in new window when open button is clicked", async () => {
        const openSpy = vi.spyOn(globalThis, "open").mockImplementation(() => null);

        renderWithProviders(<DiagramViewer definition="classDiagram" />);

        fireEvent.click(screen.getByTestId("trigger-svg-ready"));

        await waitFor(() => {
            const buttons = getActionButtons();
            expect(buttons[4]).not.toBeDisabled();
        });

        const buttons = getActionButtons();
        fireEvent.click(buttons[4]);

        expect(openSpy).toHaveBeenCalledWith("blob:mock-url", "_blank");
    });

    it("uses default filename 'schema' when none provided", async () => {
        const clickSpy = vi.fn();
        const createElementOrig = document.createElement.bind(document);
        vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
            const el = createElementOrig(tag);
            if (tag === "a") {
                Object.defineProperty(el, "click", { value: clickSpy });
            }
            return el;
        });

        renderWithProviders(<DiagramViewer definition="classDiagram" />);

        fireEvent.click(screen.getByTestId("trigger-svg-ready"));
        await waitFor(() => {
            const buttons = getActionButtons();
            expect(buttons[3]).not.toBeDisabled();
        });

        const buttons = getActionButtons();
        fireEvent.click(buttons[3]);

        expect(clickSpy).toHaveBeenCalled();
    });

    it("renders the transform wrapper and component", () => {
        renderWithProviders(<DiagramViewer definition="classDiagram" />);
        expect(screen.getByTestId("transform-wrapper")).toBeInTheDocument();
        expect(screen.getByTestId("transform-component")).toBeInTheDocument();
    });

    it("does nothing on download click when svgString is null", () => {
        renderWithProviders(<DiagramViewer definition="classDiagram" />);
        // Don't trigger SVG ready -- svgString stays null
        const buttons = getActionButtons();
        // Download button is disabled; clicking it should not call createObjectURL
        fireEvent.click(buttons[3]);
        expect(URL.createObjectURL).not.toHaveBeenCalled();
    });

    it("does nothing on open-new-window click when svgString is null", () => {
        const openSpy = vi.spyOn(globalThis, "open").mockImplementation(() => null);
        renderWithProviders(<DiagramViewer definition="classDiagram" />);
        const buttons = getActionButtons();
        fireEvent.click(buttons[4]);
        expect(openSpy).not.toHaveBeenCalled();
    });
});
