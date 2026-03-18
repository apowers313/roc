import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { MetricsChart } from "./MetricsChart";

vi.mock("../../api/queries", () => ({
    useMetricsHistory: vi.fn(),
}));

import { useMetricsHistory } from "../../api/queries";

describe("MetricsChart", () => {
    it("shows empty message when no data", () => {
        vi.mocked(useMetricsHistory).mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useMetricsHistory>);

        renderWithProviders(
            <MetricsChart run="test-run" currentStep={1} />,
        );
        expect(screen.getByText("No metrics history")).toBeInTheDocument();
    });

    it("shows empty message when data is empty array", () => {
        vi.mocked(useMetricsHistory).mockReturnValue({
            data: [],
        } as unknown as ReturnType<typeof useMetricsHistory>);

        renderWithProviders(
            <MetricsChart run="test-run" currentStep={1} />,
        );
        expect(screen.getByText("No metrics history")).toBeInTheDocument();
    });

    it("renders chart container when data is present", () => {
        vi.mocked(useMetricsHistory).mockReturnValue({
            data: [
                { step: 1, hp: 10, hp_max: 15, score: 0, energy: 5, energy_max: 10 },
                { step: 2, hp: 9, hp_max: 15, score: 10, energy: 4, energy_max: 10 },
            ],
        } as unknown as ReturnType<typeof useMetricsHistory>);

        const { container } = renderWithProviders(
            <MetricsChart run="test-run" currentStep={1} />,
        );
        // Recharts renders inside a ResponsiveContainer div
        expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
    });
});
