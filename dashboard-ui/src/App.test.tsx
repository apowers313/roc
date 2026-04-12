import { screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

vi.mock("socket.io-client");
vi.mock("./api/queries");
vi.mock("./hooks/usePrefetchWindow");
vi.mock("./hooks/useRunSubscription", () => ({
    useRunSubscription: vi.fn(),
    useGameState: vi.fn(() => null),
}));

import { renderWithProviders, makeStepData, makeGridData } from "./test-utils";
import { App } from "./App";
import { useStepData } from "./api/queries";

const mockUseStepData = vi.mocked(useStepData);

describe("App", () => {
    beforeEach(() => {
        vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
            ok: true,
            json: () =>
                Promise.resolve({
                    active: false,
                    run_name: null,
                    step: 0,
                    game_number: 0,
                    step_min: 0,
                    step_max: 0,
                    game_numbers: [],
                }),
        }));
        mockUseStepData.mockReturnValue({
            data: undefined,
            isLoading: false,
            isPlaceholderData: false,
        } as unknown as ReturnType<typeof useStepData>);
    });

    it("renders the main layout with accordion sections", () => {
        renderWithProviders(<App />);

        expect(screen.getByText("Game State")).toBeInTheDocument();
        expect(screen.getByText("Visual Perception")).toBeInTheDocument();
        expect(screen.getByText("Visual Attention")).toBeInTheDocument();
        expect(screen.getByText("Object Resolution")).toBeInTheDocument();
        expect(screen.getByText("Log Messages")).toBeInTheDocument();
    });

    it("shows loading text when loading with no data", () => {
        mockUseStepData.mockReturnValue({
            data: undefined,
            isLoading: true,
            isPlaceholderData: false,
        } as unknown as ReturnType<typeof useStepData>);

        renderWithProviders(<App />);
        expect(screen.getByText("Loading...")).toBeInTheDocument();
    });

    it("does not show loading when data is present", () => {
        const data = makeStepData({
            screen: makeGridData(),
            game_metrics: { hp: 10, hp_max: 20, score: 50 },
        });
        mockUseStepData.mockReturnValue({
            data,
            isLoading: false,
            isPlaceholderData: false,
        } as unknown as ReturnType<typeof useStepData>);

        renderWithProviders(<App />);
        expect(screen.queryByText("Loading...")).not.toBeInTheDocument();
    });

    it("renders transport bar and status bar", () => {
        renderWithProviders(<App />);

        // Transport bar buttons
        expect(screen.getByLabelText("Play")).toBeInTheDocument();
        expect(screen.getByLabelText("First step")).toBeInTheDocument();
    });
});
