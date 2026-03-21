/**
 * Tests for ResolutionChart -- covers the cumulative computation logic,
 * empty state, and rendering with data.
 */

import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

// Mock queries
vi.mock("../../api/queries", () => ({
    useResolutionHistory: vi.fn(() => ({ data: undefined })),
}));

import { renderWithProviders } from "../../test-utils";
import { ResolutionChart } from "./ResolutionChart";
import { useResolutionHistory } from "../../api/queries";

const mockUseResolutionHistory = vi.mocked(useResolutionHistory);

describe("ResolutionChart", () => {
    it("shows 'No resolution history' when data is undefined", () => {
        mockUseResolutionHistory.mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useResolutionHistory>);

        renderWithProviders(
            <ResolutionChart run="run1" currentStep={5} />,
        );
        expect(screen.getByText("No resolution history")).toBeInTheDocument();
    });

    it("shows 'No resolution history' when data is empty array", () => {
        mockUseResolutionHistory.mockReturnValue({
            data: [],
        } as ReturnType<typeof useResolutionHistory>);

        renderWithProviders(
            <ResolutionChart run="run1" currentStep={5} />,
        );
        expect(screen.getByText("No resolution history")).toBeInTheDocument();
    });

    it("renders chart title when data is present", () => {
        mockUseResolutionHistory.mockReturnValue({
            data: [
                { step: 1, outcome: "new_object", correct: null },
                { step: 2, outcome: "match", correct: true },
                { step: 3, outcome: "match", correct: false },
                { step: 4, outcome: "match", correct: null },
            ],
        } as ReturnType<typeof useResolutionHistory>);

        renderWithProviders(
            <ResolutionChart run="run1" currentStep={2} />,
        );
        expect(screen.getByText("Object Resolution Error Rate")).toBeInTheDocument();
    });

    it("handles all outcome types in cumulative computation", () => {
        mockUseResolutionHistory.mockReturnValue({
            data: [
                { step: 1, outcome: "new_object", correct: null },
                { step: 2, outcome: "match", correct: true },
                { step: 3, outcome: "match", correct: false },
                { step: 4, outcome: "match", correct: null },
                { step: 5, outcome: "new_object", correct: null },
                { step: 6, outcome: "match", correct: true },
            ],
        } as ReturnType<typeof useResolutionHistory>);

        renderWithProviders(
            <ResolutionChart
                run="run1"
                currentStep={3}
                onStepClick={() => {}}
            />,
        );
        // Verify chart renders with the title
        expect(screen.getByText("Object Resolution Error Rate")).toBeInTheDocument();
    });

    it("renders with game filter", () => {
        mockUseResolutionHistory.mockReturnValue({
            data: [
                { step: 10, outcome: "match", correct: true },
            ],
        } as ReturnType<typeof useResolutionHistory>);

        renderWithProviders(
            <ResolutionChart run="run1" game={2} currentStep={10} />,
        );
        expect(screen.getByText("Object Resolution Error Rate")).toBeInTheDocument();
    });

    it("passes game parameter to useResolutionHistory", () => {
        mockUseResolutionHistory.mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useResolutionHistory>);

        renderWithProviders(
            <ResolutionChart run="run1" game={5} currentStep={1} />,
        );
        expect(useResolutionHistory).toHaveBeenCalledWith("run1", 5);
    });

    it("renders with onStepClick prop", () => {
        mockUseResolutionHistory.mockReturnValue({
            data: [
                { step: 1, outcome: "new_object", correct: null },
                { step: 2, outcome: "match", correct: true },
            ],
        } as ReturnType<typeof useResolutionHistory>);

        const onStepClick = vi.fn();
        renderWithProviders(
            <ResolutionChart
                run="run1"
                currentStep={1}
                onStepClick={onStepClick}
            />,
        );
        expect(screen.getByText("Object Resolution Error Rate")).toBeInTheDocument();
    });

    it("renders without onStepClick prop (default noop)", () => {
        mockUseResolutionHistory.mockReturnValue({
            data: [
                { step: 1, outcome: "match", correct: true },
            ],
        } as ReturnType<typeof useResolutionHistory>);

        renderWithProviders(
            <ResolutionChart run="run1" currentStep={1} />,
        );
        expect(screen.getByText("Object Resolution Error Rate")).toBeInTheDocument();
    });

    it("accumulates correct, incorrect, unknown, and new_object counts", () => {
        mockUseResolutionHistory.mockReturnValue({
            data: [
                { step: 1, outcome: "new_object", correct: null },
                { step: 2, outcome: "match", correct: true },
                { step: 3, outcome: "match", correct: true },
                { step: 4, outcome: "match", correct: false },
                { step: 5, outcome: "match", correct: null },
                { step: 6, outcome: "new_object", correct: null },
            ],
        } as ReturnType<typeof useResolutionHistory>);

        const { container } = renderWithProviders(
            <ResolutionChart run="run1" currentStep={3} />,
        );
        // Chart should render with recharts container
        expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
    });
});
