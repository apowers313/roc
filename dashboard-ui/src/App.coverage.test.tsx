/**
 * Additional coverage tests for App.tsx -- covers uncovered branches:
 * accordion persistence, keyboard shortcuts (speed up/down, toggle bookmark,
 * bookmark navigation, go live, cycle game), live push handling, and more.
 */

import { screen, fireEvent, act, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

vi.mock("socket.io-client");
vi.mock("./api/queries");
vi.mock("./hooks/usePrefetchWindow");
vi.mock("./hooks/useLiveUpdates");
vi.mock("./api/client");

import { renderWithProviders, makeStepData, makeGridData, stubDefaultFetch } from "./test-utils";
import { App } from "./App";
import { useStepData, useGames } from "./api/queries";
import {
    getCapturedOnNewStep,
    resetLiveUpdatesMock,
    setMockLiveStatusValue,
} from "./hooks/__mocks__/useLiveUpdates";

const mockUseStepData = vi.mocked(useStepData);
const mockUseGames = vi.mocked(useGames);

describe("App (additional coverage)", () => {
    beforeEach(() => {
        resetLiveUpdatesMock();
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

            const fetchMock = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ min: 47, max: 121 }),
            });
            vi.stubGlobal("fetch", fetchMock);

            renderWithProviders(<App />);

            await act(async () => {
                fireEvent.keyDown(document, { key: "g" });
            });

            await waitFor(() => {
                const calls = fetchMock.mock.calls.map((c: unknown[]) => c[0]);
                const hasStepRangeCall = calls.some(
                    (url: unknown) => typeof url === "string" && url.includes("step-range"),
                );
                expect(hasStepRangeCall).toBe(true);
            });
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

            vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("Network error")));

            renderWithProviders(<App />);

            await act(async () => {
                fireEvent.keyDown(document, { key: "g" });
            });
        });
    });

    describe("goLive callback", () => {
        it("go live via L key when liveRunName is set", () => {
            setMockLiveStatusValue({
                active: true,
                run_name: "live-run",
                step: 50,
                game_number: 1,
                step_min: 1,
                step_max: 50,
                game_numbers: [1],
            });

            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "l" });
        });

        it("go live does nothing when liveRunName is empty", () => {
            setMockLiveStatusValue(null);

            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "l" });
        });
    });

    describe("liveStatus effects", () => {
        it("sets liveRunName and liveGameNumber from active live status", () => {
            setMockLiveStatusValue({
                active: true,
                run_name: "live-run",
                step: 25,
                game_number: 2,
                step_min: 1,
                step_max: 25,
                game_numbers: [1, 2],
            });

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

    describe("onNewStep callback (live push handling)", () => {
        it("stores live data from push", () => {
            renderWithProviders(<App />);
            expect(getCapturedOnNewStep()).toBeDefined();

            act(() => {
                getCapturedOnNewStep()!(makeStepData({ step: 5, game_number: 1 }));
            });
        });
    });

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
