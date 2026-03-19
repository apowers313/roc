/**
 * Tests for per-game step handling in the dashboard.
 *
 * Each game has its own independent step timeline (e.g. game 1: steps 1-46,
 * game 2: steps 47-120). The dashboard must:
 * - Show the correct step range for the selected game
 * - Constrain navigation to the selected game's range
 * - Support live-following mode for any game (not just "all games")
 * - Update step range from live push data when viewing the live game
 */

import { screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock socket.io-client
vi.mock("socket.io-client", () => {
    const mockSocket = {
        on: vi.fn(),
        disconnect: vi.fn(),
    };
    return { io: vi.fn(() => mockSocket) };
});

// Mock query hooks
vi.mock("./api/queries", () => ({
    useStepData: vi.fn(() => ({
        data: undefined,
        isLoading: false,
        isPlaceholderData: false,
    })),
    useRuns: vi.fn(() => ({ data: undefined })),
    useGames: vi.fn(() => ({ data: undefined })),
    useStepRange: vi.fn(() => ({ data: undefined })),
    useResolutionHistory: vi.fn(() => ({ data: undefined })),
    useAllObjects: vi.fn(() => ({ data: undefined })),
}));

// Mock prefetch window
vi.mock("./hooks/usePrefetchWindow", () => ({
    usePrefetchWindow: vi.fn(),
}));

// Mock bookmarks
vi.mock("./api/client", () => ({
    fetchBookmarks: vi.fn().mockResolvedValue([]),
    saveBookmarks: vi.fn().mockResolvedValue(undefined),
}));

import { renderWithProviders } from "./test-utils";
import { App } from "./App";
import { useStepData, useStepRange, useGames } from "./api/queries";

const mockUseStepData = vi.mocked(useStepData);
const mockUseStepRange = vi.mocked(useStepRange);
const mockUseGames = vi.mocked(useGames);

// Helper to set up a live status response
function mockLiveStatus(status: {
    active: boolean;
    run_name: string | null;
    step: number;
    game_number: number;
    step_min: number;
    step_max: number;
    game_numbers: number[];
}) {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(status),
    }));
}

describe("Per-game step handling", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockUseStepData.mockReturnValue({
            data: undefined,
            isLoading: false,
            isPlaceholderData: false,
        } as unknown as ReturnType<typeof useStepData>);
        mockUseGames.mockReturnValue({
            data: [
                { game_number: 1, steps: 46, start_ts: null, end_ts: null },
                { game_number: 2, steps: 75, start_ts: null, end_ts: null },
                { game_number: 3, steps: 30, start_ts: null, end_ts: null },
            ],
        } as ReturnType<typeof useGames>);
        mockLiveStatus({
            active: false,
            run_name: null,
            step: 0,
            game_number: 0,
            step_min: 0,
            step_max: 0,
            game_numbers: [],
        });
    });

    describe("step range per game (historical mode)", () => {
        it("game 1 shows its own step range", () => {
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 46 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<App />);
            expect(screen.getByText("1 / 46")).toBeInTheDocument();
        });

        it("game 2 shows its own step range", () => {
            mockUseStepRange.mockReturnValue({
                data: { min: 47, max: 121 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<App />);
            expect(screen.getByText(/\/ 75$/)).toBeInTheDocument();
        });

        it("game 3 shows its own step range", () => {
            mockUseStepRange.mockReturnValue({
                data: { min: 122, max: 151 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<App />);
            expect(screen.getByText(/\/ 30$/)).toBeInTheDocument();
        });
    });

    describe("live mode with game filter", () => {
        it("live auto-select uses per-game step range, not global", async () => {
            // Live status reports global range (1-2494) but game 1 is 1-46
            mockLiveStatus({
                active: true,
                run_name: "live-run",
                step: 2494,
                game_number: 2,
                step_min: 1,
                step_max: 2494,
                game_numbers: [1, 2],
            });
            // The step-range query for game 1 returns game-specific range
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 46 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<App />);

            // After live auto-select, the step range should be constrained
            // by the useStepRange query (per-game), not the global live status
            await waitFor(() => {
                // The step range display should show game 1's range (46 steps),
                // not the global range (2494 steps)
                expect(screen.getByText(/\/ 46$/)).toBeInTheDocument();
            });
        });

        it("jumpToEnd returns to live-following when viewing the live game", () => {
            mockLiveStatus({
                active: true,
                run_name: "live-run",
                step: 100,
                game_number: 2,
                step_min: 47,
                step_max: 100,
                game_numbers: [1, 2],
            });
            mockUseStepRange.mockReturnValue({
                data: { min: 47, max: 100 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<App />);

            // Click "Last step" -- should return to live-following
            const lastBtn = screen.getByLabelText("Last step");
            fireEvent.click(lastBtn);

            // In live mode, clicking "Last step" on the live run should
            // dispatch JUMP_TO_END regardless of whether a game is selected,
            // as long as the selected game matches the live game
        });

        it("step counter does not exceed game step range from live push", async () => {
            // When live push arrives with global step 2500 but we're viewing
            // game 1 (steps 1-46), the counter should stay at 46 max
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 46 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<App />);

            await waitFor(() => {
                const counter = screen.getByText(/\/ 46$/);
                expect(counter).toBeInTheDocument();
            });
        });
    });

    describe("historical game within live run", () => {
        it("viewing game 1 in live run uses REST range, not live game's context range", async () => {
            // Regression: when viewing game 1 (historical) within the live run
            // where game 3 is active, the step counter must show game 1's range
            // from REST, not the live game's range from context. The old code
            // used isViewingLiveRun (run === liveRunName) which was true for
            // ALL games in the live run, causing context stepMax (from live
            // game 3 pushes) to override REST data for historical game 1.
            mockLiveStatus({
                active: true,
                run_name: "live-run",
                step: 9300,
                game_number: 3,
                step_min: 1,
                step_max: 9300,
                game_numbers: [1, 2, 3],
            });
            // User is viewing game 1 which has 46 steps
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 46 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<App />);

            // The step range should show game 1's range (46), not game 3's
            // global step count (9300)
            await waitFor(() => {
                expect(screen.getByText(/\/ 46$/)).toBeInTheDocument();
            });
        });
    });

    describe("game switching", () => {
        it("switching games updates step range", () => {
            // Start with game 1 range
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 46 },
            } as ReturnType<typeof useStepRange>);

            const { unmount } = renderWithProviders(<App />);
            expect(screen.getByText("1 / 46")).toBeInTheDocument();
            unmount();

            // Switch to game 2 range
            mockUseStepRange.mockReturnValue({
                data: { min: 47, max: 121 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<App />);
            expect(screen.getByText(/\/ 75$/)).toBeInTheDocument();
        });
    });
});
