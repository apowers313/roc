/**
 * Default-step selection regression tests (BUG-H1).
 *
 * The auto-select-first-run effect in ``TransportBar.tsx`` was hardcoding
 * ``?game=1`` when fetching the step range for the newly-selected run.
 * This caused two failures when the URL specified a non-1 game:
 *
 * 1. The slider/range was populated with game 1's range, even though the
 *    user wanted game N. The user landed on game 1's range until the
 *    React Query for ``useStepRange(run, game)`` resolved and the sync
 *    effect overwrote it.
 *
 * 2. The default step ended up at game 1's start, not game N's start.
 *    For a run with 256 total steps but only 6 in game 2, the slider
 *    would briefly show "1/250" (game 1's range) instead of "1/6" (game
 *    2's range), and the data fetcher would request step 1 of game 2
 *    -- which doesn't exist -- producing a 404/500.
 *
 * The fix is to pass the current ``game`` from DashboardContext to the
 * fetch URL in the auto-select-first-run effect, so the eager step-range
 * fetch matches the URL's game param.
 *
 * This test mocks the four query hooks and ``fetch``, then asserts that
 * (a) the fetch is called with the right game param and (b) the slider
 * lands at the per-game range, not the run-total range.
 */

import { screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

vi.mock("../../api/queries", () => ({
    useRuns: vi.fn(() => ({ data: undefined })),
    useGames: vi.fn(() => ({ data: undefined })),
    useStepRange: vi.fn(() => ({ data: undefined })),
}));

import { renderWithProviders } from "../../test-utils";
import { TransportBar } from "./TransportBar";
import { useRuns, useGames, useStepRange } from "../../api/queries";

const mockUseRuns = vi.mocked(useRuns);
const mockUseGames = vi.mocked(useGames);
const mockUseStepRange = vi.mocked(useStepRange);

describe("TransportBar default-step selection (BUG-H1)", () => {
    let fetchSpy: ReturnType<typeof vi.fn>;

    beforeEach(() => {
        fetchSpy = vi.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve({ min: 251, max: 256 }),
        });
        vi.stubGlobal("fetch", fetchSpy);
        mockUseRuns.mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useRuns>);
        mockUseGames.mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useGames>);
        mockUseStepRange.mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useStepRange>);
    });

    afterEach(() => {
        vi.restoreAllMocks();
        // Reset URL between tests so seeded URLs do not leak.
        globalThis.history.replaceState(null, "", "/");
    });

    describe("auto-select-first-run effect", () => {
        it("uses the current game from context when fetching step-range, not hardcoded game=1", async () => {
            // URL has ``?game=2`` but no run -- the auto-select-first-run
            // effect should kick in and fetch the range for game 2.
            globalThis.history.replaceState(null, "", "/?game=2");

            mockUseRuns.mockReturnValue({
                data: [
                    {
                        name: "20260408120000-some-run",
                        games: 2,
                        steps: 256,
                        status: "ok",
                    },
                ],
            } as unknown as ReturnType<typeof useRuns>);

            renderWithProviders(<TransportBar />);

            // The effect should fetch step-range with game=2, NOT game=1.
            await waitFor(() => {
                const calls = fetchSpy.mock.calls;
                // Find any call to step-range
                const stepRangeCalls = calls.filter((c) =>
                    typeof c[0] === "string" && c[0].includes("step-range"),
                );
                expect(stepRangeCalls.length).toBeGreaterThan(0);
            });

            // Inspect every step-range fetch -- none should use ?game=1
            // (since the URL says ?game=2). All should use ?game=2.
            const stepRangeCalls = fetchSpy.mock.calls.filter((c) =>
                typeof c[0] === "string" && c[0].includes("step-range"),
            );
            const usesGame2 = stepRangeCalls.some(
                (c) => typeof c[0] === "string" && c[0].includes("game=2"),
            );
            const usesGame1 = stepRangeCalls.some(
                (c) => typeof c[0] === "string" && c[0].includes("game=1"),
            );
            expect(usesGame2).toBe(true);
            expect(usesGame1).toBe(false);
        });

        it("falls back to game=1 when URL has no game param", async () => {
            globalThis.history.replaceState(null, "", "/");

            mockUseRuns.mockReturnValue({
                data: [
                    {
                        name: "20260408120000-some-run",
                        games: 1,
                        steps: 100,
                        status: "ok",
                    },
                ],
            } as unknown as ReturnType<typeof useRuns>);

            renderWithProviders(<TransportBar />);

            // Without an explicit game param, the effect should default
            // to game 1 (the context default).
            await waitFor(() => {
                const calls = fetchSpy.mock.calls;
                const stepRangeCalls = calls.filter((c) =>
                    typeof c[0] === "string" && c[0].includes("step-range"),
                );
                expect(stepRangeCalls.length).toBeGreaterThan(0);
                expect(
                    stepRangeCalls.some(
                        (c) =>
                            typeof c[0] === "string" && c[0].includes("game=1"),
                    ),
                ).toBe(true);
            });
        });
    });

    describe("slider counter for non-game-1 default landing", () => {
        it("slider lands at per-game range (1/6), not run total (256/256), when URL says game=2", async () => {
            // The scenario: a run with 256 total steps, 250 in game 1
            // and 6 in game 2 (numbered 251-256). The user navigates to
            // ?game=2 directly. The slider should show "1/6" once the
            // step-range query resolves.
            globalThis.history.replaceState(null, "", "/?run=20260408120000-some-run&game=2");

            mockUseRuns.mockReturnValue({
                data: [
                    {
                        name: "20260408120000-some-run",
                        games: 2,
                        steps: 256,
                        status: "ok",
                    },
                ],
            } as unknown as ReturnType<typeof useRuns>);
            mockUseGames.mockReturnValue({
                data: [
                    { game_number: 1, steps: 250, start_ts: null, end_ts: null },
                    { game_number: 2, steps: 6, start_ts: null, end_ts: null },
                ],
            } as ReturnType<typeof useGames>);
            // useStepRange(run, 2) returns the per-game range.
            mockUseStepRange.mockReturnValue({
                data: { min: 251, max: 256 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<TransportBar />);

            // The slider counter should show "1 / 6" -- step=251 (clamped
            // from default 1 to per-game min) and total=6.
            await waitFor(() => {
                expect(screen.getByText("1 / 6")).toBeInTheDocument();
            });
            // Crucially, it should NOT show "256 / 256" or "1 / 256"
            // (run-total range).
            expect(screen.queryByText("256 / 256")).not.toBeInTheDocument();
            expect(screen.queryByText("1 / 256")).not.toBeInTheDocument();
        });
    });
});
