import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { KeyboardHelp } from "./KeyboardHelp";

describe("KeyboardHelp", () => {
    it("renders nothing when closed", () => {
        const { container } = renderWithProviders(
            <KeyboardHelp opened={false} onClose={() => {}} />,
        );
        // Modal should not render content when closed
        expect(container.querySelector("[role='dialog']")).toBeNull();
    });

    it("renders modal when opened", () => {
        renderWithProviders(
            <KeyboardHelp opened={true} onClose={() => {}} />,
        );
        expect(screen.getByText("Keyboard Shortcuts")).toBeInTheDocument();
    });

    it("shows navigation shortcuts", () => {
        renderWithProviders(
            <KeyboardHelp opened={true} onClose={() => {}} />,
        );
        // Check for key bindings displayed in the modal
        expect(screen.getByText("Next step")).toBeInTheDocument();
        expect(screen.getByText("Previous step")).toBeInTheDocument();
        expect(screen.getByText("Play / Pause")).toBeInTheDocument();
        expect(screen.getByText("First step")).toBeInTheDocument();
        expect(screen.getByText("Last step")).toBeInTheDocument();
    });

    it("shows skip shortcuts", () => {
        renderWithProviders(
            <KeyboardHelp opened={true} onClose={() => {}} />,
        );
        expect(screen.getByText("+10 steps")).toBeInTheDocument();
        expect(screen.getByText("-10 steps")).toBeInTheDocument();
    });

    it("shows bookmark shortcuts", () => {
        renderWithProviders(
            <KeyboardHelp opened={true} onClose={() => {}} />,
        );
        expect(screen.getByText("Toggle bookmark")).toBeInTheDocument();
        expect(screen.getByText("Next bookmark")).toBeInTheDocument();
        expect(screen.getByText("Previous bookmark")).toBeInTheDocument();
    });

    it("shows the go live shortcut", () => {
        renderWithProviders(
            <KeyboardHelp opened={true} onClose={() => {}} />,
        );
        expect(screen.getByText("Go live")).toBeInTheDocument();
    });

    it("shows the help shortcut itself", () => {
        renderWithProviders(
            <KeyboardHelp opened={true} onClose={() => {}} />,
        );
        expect(screen.getByText("Show this help")).toBeInTheDocument();
    });
});
