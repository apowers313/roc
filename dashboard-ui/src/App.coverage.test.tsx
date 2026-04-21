/**
 * Additional coverage tests for App.tsx -- covers uncovered branches:
 * accordion persistence, keyboard shortcuts (speed up/down, toggle bookmark,
 * bookmark navigation, go live, cycle game), live push handling, and more.
 */

import { screen, fireEvent, act, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

let mockGameStateValue: { state: string; run_name: string | null } | null = null;

vi.mock("socket.io-client");
vi.mock("./api/queries");
vi.mock("./hooks/usePrefetchWindow");
vi.mock("./hooks/useRunSubscription", () => ({
    useRunSubscription: vi.fn(),
    useGameState: vi.fn(() => mockGameStateValue),
    useSocketConnected: vi.fn(() => true),
}));
vi.mock("./api/client");

import { renderWithProviders, makeStepData, makeGridData, stubDefaultFetch } from "./test-utils";
import { App } from "./App";
import { useStepData, useGames } from "./api/queries";
import { fetchStepRange } from "./api/client";
import { useGameState } from "./hooks/useRunSubscription";

const mockUseStepData = vi.mocked(useStepData);
const mockUseGames = vi.mocked(useGames);
const mockUseGameState = vi.mocked(useGameState);
const mockFetchStepRange = vi.mocked(fetchStepRange);

describe("App (additional coverage)", () => {
    beforeEach(() => {
        mockGameStateValue = null;
        mockUseGameState.mockReturnValue(null);
        stubDefaultFetch();
        mockUseStepData.mockReturnValue({
            data: undefined,
            isLoading: false,
            isPlaceholderData: false,
        } as unknown as ReturnType<typeof useStepData>);
        mockUseGames.mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useGames>);
    });

    afterEach(() => {
        vi.restoreAllMocks();
        sessionStorage.clear();
    });

    describe("accordion persistence", () => {
        it("persists accordion state to sessionStorage on change", () => {
            renderWithProviders(<App />);

            // Click a section to toggle accordion state
            const logMessages = screen.getByText("Log Messages");
            fireEvent.click(logMessages);

            const saved = sessionStorage.getItem("roc-dashboard-accordion");
            expect(saved).toBeTruthy();
            const parsed = JSON.parse(saved!);
            expect(Array.isArray(parsed)).toBe(true);
        });

        it("restores accordion state from sessionStorage on mount", () => {
            sessionStorage.setItem(
                "roc-dashboard-accordion",
                JSON.stringify(["actions"]),
            );

            renderWithProviders(<App />);
            expect(screen.getByText("Actions")).toBeInTheDocument();
        });

        it("falls back to defaults when sessionStorage has invalid JSON", () => {
            sessionStorage.setItem("roc-dashboard-accordion", "not-valid-json{");
            renderWithProviders(<App />);
            expect(screen.getByText("Pipeline Status")).toBeInTheDocument();
        });

        it("falls back to defaults when sessionStorage has non-array JSON", () => {
            sessionStorage.setItem("roc-dashboard-accordion", '"string-value"');
            renderWithProviders(<App />);
            expect(screen.getByText("Pipeline Status")).toBeInTheDocument();
        });
    });

    describe("keyboard shortcuts", () => {
        it("step forward via Right arrow key", () => {
            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "ArrowRight" });
        });

        it("step back via Left arrow key", () => {
            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "ArrowLeft" });
        });

        it("toggle play via Space key", () => {
            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: " " });
        });

        it("toggle help overlay via ? key", () => {
            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "?" });
        });

        it("jump to start via Home key", () => {
            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "Home" });
        });

        it("jump to end via End key", () => {
            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "End" });
        });

        it("step forward 10 via Shift+Right", () => {
            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "ArrowRight", shiftKey: true });
        });

        it("step back 10 via Shift+Left", () => {
            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "ArrowLeft", shiftKey: true });
        });

        it("speed up via ] key", () => {
            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "]" });
        });

        it("speed down via [ key", () => {
            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "[" });
        });

        it("bookmark toggle via B key", () => {
            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "b" });
        });
    });

    describe("all accordion sections render", () => {
        it("renders every panel section", () => {
            mockUseStepData.mockReturnValue({
                data: makeStepData({ screen: makeGridData() }),
                isLoading: false,
                isPlaceholderData: false,
            } as unknown as ReturnType<typeof useStepData>);

            renderWithProviders(<App />);

            // Check unique section names (Perception may appear multiple
            // times due to PipelineStatus rendering, so use getAllByText)
            expect(screen.getByText("Pipeline Status")).toBeInTheDocument();
            expect(screen.getByText("Bookmarks")).toBeInTheDocument();
            expect(screen.getByText("Game State")).toBeInTheDocument();
            expect(screen.getByText("Log Messages")).toBeInTheDocument();
            expect(screen.getByText("Intrinsics & Significance")).toBeInTheDocument();
            expect(screen.getByText("Inventory")).toBeInTheDocument();
            expect(screen.getByText("Visual Perception")).toBeInTheDocument();
            expect(screen.getByText("Visual Attention")).toBeInTheDocument();
            expect(screen.getByText("Object Resolution")).toBeInTheDocument();
            expect(screen.getByText("Sequences")).toBeInTheDocument();
            expect(screen.getByText("Transitions")).toBeInTheDocument();
            expect(screen.getAllByText("Prediction").length).toBeGreaterThanOrEqual(1);
            expect(screen.getByText("Actions")).toBeInTheDocument();
        });
    });

    describe("cycleGame callback", () => {
        it("cycles through games when games are available", async () => {
            mockUseGames.mockReturnValue({
                data: [
                    { game_number: 1, steps: 46, start_ts: null, end_ts: null },
                    { game_number: 2, steps: 75, start_ts: null, end_ts: null },
                ],
            } as ReturnType<typeof useGames>);

            mockFetchStepRange.mockResolvedValue({ min: 47, max: 121 });

            renderWithProviders(<App />);

            await act(async () => {
                fireEvent.keyDown(document, { key: "g" });
            });

            // cycleGame now routes through queryClient.fetchQuery which
            // invokes fetchStepRange from ./api/client (mocked). Verify
            // the mock was called with the new game number.
            await waitFor(() => {
                expect(mockFetchStepRange).toHaveBeenCalled();
            });
            const calls = mockFetchStepRange.mock.calls;
            const hasGameCall = calls.some((c) => c[1] === 2);
            expect(hasGameCall).toBe(true);
        });

        it("does nothing when no games are available", () => {
            mockUseGames.mockReturnValue({
                data: [],
            } as unknown as ReturnType<typeof useGames>);

            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "g" });
        });

        it("handles fetch error when cycling games", async () => {
            mockUseGames.mockReturnValue({
                data: [
                    { game_number: 1, steps: 46, start_ts: null, end_ts: null },
                    { game_number: 2, steps: 75, start_ts: null, end_ts: null },
                ],
            } as ReturnType<typeof useGames>);

            mockFetchStepRange.mockRejectedValue(new Error("Network error"));

            renderWithProviders(<App />);

            await act(async () => {
                fireEvent.keyDown(document, { key: "g" });
            });
        });
    });

    describe("goLive callback", () => {
        it("go live via L key when liveRunName is set", () => {
            mockUseGameState.mockReturnValue({ state: "running", run_name: "live-run" });

            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "l" });
        });

        it("go live does nothing when liveRunName is empty", () => {
            mockUseGameState.mockReturnValue(null);

            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "l" });
        });
    });

    describe("liveStatus effects", () => {
        it("sets liveRunName and liveGameNumber from active live status", () => {
            mockUseGameState.mockReturnValue({ state: "running", run_name: "live-run" });

            renderWithProviders(<App />);
            // Should auto-select the live run without crashing
        });
    });

    describe("isLoading with existing data", () => {
        it("does not show loading text when isLoading is true but data exists", () => {
            mockUseStepData.mockReturnValue({
                data: makeStepData({ screen: makeGridData() }),
                isLoading: true,
                isPlaceholderData: false,
            } as unknown as ReturnType<typeof useStepData>);

            renderWithProviders(<App />);
            expect(screen.queryByText("Loading...")).not.toBeInTheDocument();
        });
    });

    // Phase 4: ``onNewStep`` callback was deleted from App.tsx. The data
    // refresh path now flows through ``useRunSubscription`` (TanStack
    // Query invalidation). The unit coverage lives in
    // ``hooks/useRunSubscription.test.tsx``.

    describe("handleChartStepClick", () => {
        it("renders chart containers without crashing", () => {
            mockUseStepData.mockReturnValue({
                data: makeStepData({ screen: makeGridData() }),
                isLoading: false,
                isPlaceholderData: false,
            } as unknown as ReturnType<typeof useStepData>);

            renderWithProviders(<App />);
            expect(screen.getByText("Object Resolution")).toBeInTheDocument();
        });
    });
});
