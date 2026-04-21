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

import { renderWithProviders } from "./test-utils";
import { App } from "./App";
import { useStepData, useStepRange, useGames } from "./api/queries";
import { useGameState } from "./hooks/useRunSubscription";

const mockUseStepData = vi.mocked(useStepData);
const mockUseStepRange = vi.mocked(useStepRange);
const mockUseGames = vi.mocked(useGames);
const mockUseGameState = vi.mocked(useGameState);

describe("Per-game step handling", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockGameStateValue = null;
        mockUseGameState.mockReturnValue(null);
        vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve({ min: 0, max: 0 }),
        }));
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
            // Live status reports a running game; game 1 has a specific step range
            mockUseGameState.mockReturnValue({ state: "running", run_name: "live-run" });
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
            mockUseGameState.mockReturnValue({ state: "running", run_name: "live-run" });
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
            mockUseGameState.mockReturnValue({ state: "running", run_name: "live-run" });
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

    // TC-GAME-004 regression: full-app integration test for the
    // "break auto-follow on a live run, then GO LIVE again" flow.
    // Pre-fix, TransportBar.tsx had an ``isViewingLiveGame`` guard
    // that stopped syncing ``stepRangeData`` to context when the run
    // was tail-growing, so the slider max froze as soon as the user
    // broke auto-follow. The goLive callback then read that frozen
    // max when clicked. This test drives the full flow through the
    // public component surface so a future regression in either
    // TransportBar's sync or the goLive handler fails here rather
    // than silently escaping to production.
    describe("TC-GAME-004: GO LIVE flow on a tail-growing run", () => {
        it("slider max tracks live REST range after breaking auto-follow", async () => {
            mockUseGameState.mockReturnValue({
                state: "running",
                run_name: "live-run",
            });
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 100, tail_growing: true },
            } as ReturnType<typeof useStepRange>);

            const { rerender } = renderWithProviders(<App />);

            // Initial range renders.
            await waitFor(() => {
                expect(screen.getByText(/\/ 100$/)).toBeInTheDocument();
            });

            // User breaks auto-follow with an explicit navigation.
            fireEvent.keyDown(document, { key: "ArrowLeft" });

            // Live range grows to 150 (simulated Socket.io invalidation
            // + query refetch delivering a fresh payload). The slider
            // must re-render against the new max even though autoFollow
            // is now false.
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 150, tail_growing: true },
            } as ReturnType<typeof useStepRange>);
            rerender(<App />);

            await waitFor(() => {
                expect(screen.getByText(/\/ 150$/)).toBeInTheDocument();
            });
        });

        it("L keyboard shortcut snaps to fresh max via goLive", async () => {
            mockUseGameState.mockReturnValue({
                state: "running",
                run_name: "live-run",
            });
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 100, tail_growing: true },
            } as ReturnType<typeof useStepRange>);

            const { rerender } = renderWithProviders(<App />);

            await waitFor(() => {
                expect(screen.getByText(/\/ 100$/)).toBeInTheDocument();
            });

            // Break follow, then let the range grow.
            fireEvent.keyDown(document, { key: "ArrowLeft" });

            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 200, tail_growing: true },
            } as ReturnType<typeof useStepRange>);
            rerender(<App />);

            await waitFor(() => {
                expect(screen.getByText(/\/ 200$/)).toBeInTheDocument();
            });

            // L = goLive. Must snap step to 200 (not the stale 100 from
            // before the break). We verify via the slider counter text:
            // after the shortcut, current == max, which displays as
            // "200 / 200".
            fireEvent.keyDown(document, { key: "l" });

            await waitFor(() => {
                expect(screen.getByText("200 / 200")).toBeInTheDocument();
            });
        });
    });

    // Regression: a URL like ?run=behindhand-danny-colson&step=2500 against
    // a 314-step game would set step=2500 in state. The slider would clamp
    // to 314/314, but the underlying state stayed at 2500 and the StatusBar
    // would show "Step 2500 | Game 0". Once the per-game step-range query
    // resolves, setStepRange must clamp the step into the actual range so
    // the slider and StatusBar agree.
    describe("out-of-range step navigation", () => {
        beforeEach(() => {
            // Pretend the URL had ?step=2500 by seeding the test browser
            // history. The DashboardProvider reads this on mount.
            globalThis.history.replaceState(null, "", "/?step=2500");
        });

        it("clamps step state to data range when REST step-range arrives", async () => {
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 314 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<App />);

            // After the REST step-range query resolves, the slider readout
            // should show 314 / 314 (clamped from 2500), not 2500 / 314.
            await waitFor(() => {
                expect(screen.getByText("314 / 314")).toBeInTheDocument();
            });
        });
    });
});
