import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { IntrinsicsChart } from "./IntrinsicsChart";

vi.mock("../../api/queries", () => ({
    useIntrinsicsHistory: vi.fn(),
}));

import { useIntrinsicsHistory } from "../../api/queries";

describe("IntrinsicsChart", () => {
    it("shows empty message when no data", () => {
        vi.mocked(useIntrinsicsHistory).mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useIntrinsicsHistory>);

        renderWithProviders(
            <IntrinsicsChart run="test-run" currentStep={1} />,
        );
        expect(screen.getByText("No intrinsics history")).toBeInTheDocument();
    });

    it("shows empty message when data is empty array", () => {
        vi.mocked(useIntrinsicsHistory).mockReturnValue({
            data: [],
        } as unknown as ReturnType<typeof useIntrinsicsHistory>);

        renderWithProviders(
            <IntrinsicsChart run="test-run" currentStep={1} />,
        );
        expect(screen.getByText("No intrinsics history")).toBeInTheDocument();
    });

    it("renders chart container when data is present", () => {
        vi.mocked(useIntrinsicsHistory).mockReturnValue({
            data: [
                { step: 1, normalized: { hp: 0.5, energy: 0.8 } },
                { step: 2, normalized: { hp: 0.4, energy: 0.7 } },
            ],
        } as unknown as ReturnType<typeof useIntrinsicsHistory>);

        const { container } = renderWithProviders(
            <IntrinsicsChart run="test-run" currentStep={1} />,
        );
        expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
    });

    it("renders chart without onStepClick (no ClickableChart wrapper)", () => {
        vi.mocked(useIntrinsicsHistory).mockReturnValue({
            data: [
                { step: 1, normalized: { hp: 0.5 } },
            ],
        } as unknown as ReturnType<typeof useIntrinsicsHistory>);

        const { container } = renderWithProviders(
            <IntrinsicsChart run="test-run" currentStep={1} />,
        );
        expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
    });

    it("renders chart with onStepClick (ClickableChart wrapper)", () => {
        vi.mocked(useIntrinsicsHistory).mockReturnValue({
            data: [
                { step: 1, normalized: { hp: 0.5 } },
                { step: 2, normalized: { hp: 0.6 } },
            ],
        } as unknown as ReturnType<typeof useIntrinsicsHistory>);

        const onStepClick = vi.fn();
        const { container } = renderWithProviders(
            <IntrinsicsChart run="test-run" currentStep={1} onStepClick={onStepClick} />,
        );
        expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
    });

    it("handles entries without normalized data", () => {
        vi.mocked(useIntrinsicsHistory).mockReturnValue({
            data: [
                { step: 1 },
                { step: 2, normalized: { hp: 0.5 } },
            ],
        } as unknown as ReturnType<typeof useIntrinsicsHistory>);

        const { container } = renderWithProviders(
            <IntrinsicsChart run="test-run" currentStep={1} />,
        );
        expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
    });

    it("passes game parameter to hook", () => {
        vi.mocked(useIntrinsicsHistory).mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useIntrinsicsHistory>);

        renderWithProviders(
            <IntrinsicsChart run="test-run" game={2} currentStep={1} />,
        );
        expect(useIntrinsicsHistory).toHaveBeenCalledWith("test-run", 2);
    });

    it("discovers keys from first entry with multiple normalized fields", () => {
        vi.mocked(useIntrinsicsHistory).mockReturnValue({
            data: [
                { step: 1, normalized: { hp: 0.5, energy: 0.8, hunger: 0.3 } },
                { step: 2, normalized: { hp: 0.4, energy: 0.7, hunger: 0.2 } },
            ],
        } as unknown as ReturnType<typeof useIntrinsicsHistory>);

        const { container } = renderWithProviders(
            <IntrinsicsChart run="test-run" currentStep={1} />,
        );
        expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
    });

    it("handles first entry with empty normalized object", () => {
        vi.mocked(useIntrinsicsHistory).mockReturnValue({
            data: [
                { step: 1, normalized: {} },
                { step: 2, normalized: { hp: 0.5 } },
            ],
        } as unknown as ReturnType<typeof useIntrinsicsHistory>);

        const { container } = renderWithProviders(
            <IntrinsicsChart run="test-run" currentStep={1} />,
        );
        // Should still render a chart container even with no keys from first entry
        expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
    });
});
