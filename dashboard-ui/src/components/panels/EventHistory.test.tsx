import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { EventHistory } from "./EventHistory";

vi.mock("../../api/queries", () => ({
    useEventHistory: vi.fn(),
}));

import { useEventHistory } from "../../api/queries";

describe("EventHistory", () => {
    it("shows empty message when no data", () => {
        vi.mocked(useEventHistory).mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useEventHistory>);

        renderWithProviders(
            <EventHistory run="test-run" currentStep={1} />,
        );
        expect(screen.getByText("No event history")).toBeInTheDocument();
    });

    it("shows empty message when data is empty array", () => {
        vi.mocked(useEventHistory).mockReturnValue({
            data: [],
        } as unknown as ReturnType<typeof useEventHistory>);

        renderWithProviders(
            <EventHistory run="test-run" currentStep={1} />,
        );
        expect(screen.getByText("No event history")).toBeInTheDocument();
    });

    it("renders chart container when data is present", () => {
        vi.mocked(useEventHistory).mockReturnValue({
            data: [
                { step: 1, "roc.perception": 5, "roc.attention": 3 },
                { step: 2, "roc.perception": 4, "roc.attention": 6 },
            ],
        } as unknown as ReturnType<typeof useEventHistory>);

        const { container } = renderWithProviders(
            <EventHistory run="test-run" currentStep={1} />,
        );
        expect(screen.getByText("Event Activity")).toBeInTheDocument();
        expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
    });
});
