/**
 * URL parameter sovereignty regression tests (BUG-C2).
 *
 * The dashboard's documented "URL parameter sovereignty" invariant says:
 * when ``?run=X`` is in the URL, the dashboard MUST NEVER auto-navigate
 * to a different run, even if X's endpoints fail or a different run
 * becomes live in the background. The user explicitly asked to look at
 * X; silently teleporting them away violates that contract.
 *
 * Two effects in the codebase can violate the invariant:
 *
 * 1. ``App.tsx`` -- the auto-select-live-run effect (the one that calls
 *    ``setRun(liveStatus.run_name)`` when a new game starts) was using a
 *    condition that allowed navigation away from the URL run when a NEW
 *    live game appeared with a different name. The fix is to bail
 *    entirely when ``initialUrlRun.current`` is set and differs from
 *    ``liveStatus.run_name``.
 *
 * 2. ``TransportBar.tsx`` -- the auto-select-first-run effect was guarded
 *    on ``!run``. As a defensive measure, it must also check the URL: if
 *    the URL specified an explicit run, the effect must not clobber it
 *    even if context ``run`` is somehow empty.
 *
 * These tests use ``vi.mock`` for queries + ``useGameState`` and seed
 * the URL via ``history.replaceState`` before each test, mirroring the
 * pattern in ``App.game-handling.test.tsx``.
 */

import { screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

let mockGameStateValue: { state: string; run_name: string | null } | null = null;

vi.mock("socket.io-client");
vi.mock("./api/queries");
vi.mock("./hooks/usePrefetchWindow");
vi.mock("./hooks/useRunSubscription", () => ({
    useRunSubscription: vi.fn(),
    useGameState: vi.fn(() => mockGameStateValue),
}));

vi.mock("./api/client");

import { renderWithProviders } from "./test-utils";
import { App } from "./App";
import { useStepData, useGames, useRuns, useStepRange } from "./api/queries";

const mockUseStepData = vi.mocked(useStepData);
const mockUseGames = vi.mocked(useGames);
const mockUseRuns = vi.mocked(useRuns);
const mockUseStepRange = vi.mocked(useStepRange);

describe("URL parameter sovereignty (BUG-C2)", () => {
    beforeEach(() => {
        vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve({ min: 1, max: 100 }),
        }));
        mockUseStepData.mockReturnValue({
            data: undefined,
            isLoading: false,
            isPlaceholderData: false,
            isError: false,
            error: null,
        } as unknown as ReturnType<typeof useStepData>);
        mockUseGames.mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useGames>);
        mockUseRuns.mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useRuns>);
        mockUseStepRange.mockReturnValue({
            data: { min: 1, max: 100 },
        } as ReturnType<typeof useStepRange>);
        mockGameStateValue = null;
    });

    afterEach(() => {
        vi.restoreAllMocks();
        sessionStorage.clear();
        // Reset URL between tests so seeded URLs do not leak.
        globalThis.history.replaceState(null, "", "/");
    });

    describe("auto-select-live-run effect (App.tsx)", () => {
        it("does not navigate to liveStatus run when URL has explicit run", async () => {
            // URL says: I want to look at "specific-run".
            globalThis.history.replaceState(null, "", "/?run=specific-run&game=1&step=10");

            // Live game is running on a DIFFERENT run.
            mockGameStateValue = { state: "running", run_name: "different-live-run" };

            renderWithProviders(<App />);

            // Wait for rAF-debounced URL writes to flush. The URL update
            // is queued via requestAnimationFrame in DashboardProvider.
            await waitFor(() => {
                expect(globalThis.location.search).toContain("run=specific-run");
            });

            // The bug: a follow-up rAF might overwrite the URL with the
            // live run. Wait long enough for any subsequent setRun to flush
            // and re-assert the URL is still preserved.
            await new Promise((resolve) => setTimeout(resolve, 100));
            expect(globalThis.location.search).toContain("run=specific-run");
            expect(globalThis.location.search).not.toContain("run=different-live-run");
        });

        it("does not navigate when /games for the URL run returns no data", async () => {
            // URL says: I want to look at "broken-run".
            globalThis.history.replaceState(null, "", "/?run=broken-run&game=1&step=10");

            // /games for broken-run returns nothing (simulating a 500 turning
            // into ``data: undefined`` after the React Query failure).
            mockUseGames.mockReturnValue({
                data: undefined,
            } as ReturnType<typeof useGames>);

            // Meanwhile, a different live run is active.
            mockGameStateValue = { state: "running", run_name: "other-live-run" };

            renderWithProviders(<App />);

            // Wait for the URL to be observable, then make sure no later
            // effect overwrites it.
            await waitFor(() => {
                expect(globalThis.location.search).toContain("run=broken-run");
            });
            await new Promise((resolve) => setTimeout(resolve, 100));
            expect(globalThis.location.search).toContain("run=broken-run");
            expect(globalThis.location.search).not.toContain("run=other-live-run");
        });

        it("still auto-selects the live run when the URL has no explicit run", async () => {
            // No URL params -- the user landed on the bare page.
            globalThis.history.replaceState(null, "", "/");

            mockGameStateValue = { state: "running", run_name: "live-run" };

            renderWithProviders(<App />);

            // The auto-navigate effect should fire and the URL should
            // gain ``run=live-run``. This is the documented "no explicit
            // URL run" path -- it must KEEP working after the fix.
            await waitFor(() => {
                expect(globalThis.location.search).toContain("run=live-run");
            });
        });
    });

    describe("auto-select-first-run effect (TransportBar.tsx)", () => {
        it("does not clobber explicit URL run with runs[0] when URL has explicit run", async () => {
            // URL says: ?run=user-run
            globalThis.history.replaceState(null, "", "/?run=user-run");

            // The /runs list has a different first run.
            mockUseRuns.mockReturnValue({
                data: [
                    { name: "first-in-list", games: 1, steps: 100 },
                    { name: "user-run", games: 1, steps: 50 },
                ],
            } as ReturnType<typeof useRuns>);
            // No live game.
            mockGameStateValue = null;

            renderWithProviders(<App />);

            // Wait for any effect-driven URL writes to flush.
            await waitFor(() => {
                expect(globalThis.location.search).toContain("run=user-run");
            });
            await new Promise((resolve) => setTimeout(resolve, 100));
            expect(globalThis.location.search).toContain("run=user-run");
            expect(globalThis.location.search).not.toContain("run=first-in-list");
        });

        it("auto-selects runs[0] when URL has no explicit run", async () => {
            // No URL params.
            globalThis.history.replaceState(null, "", "/");

            mockUseRuns.mockReturnValue({
                data: [
                    { name: "newest-run", games: 1, steps: 100 },
                    { name: "older-run", games: 1, steps: 50 },
                ],
            } as ReturnType<typeof useRuns>);
            mockGameStateValue = null;

            renderWithProviders(<App />);

            // Without an explicit URL run, the fallback should kick in
            // and pick newest-run.
            await waitFor(() => {
                expect(globalThis.location.search).toContain("run=newest-run");
            });
        });
    });

    describe("Section sanity check", () => {
        it("renders without crashing when URL has an unknown run", () => {
            globalThis.history.replaceState(null, "", "/?run=ghost-run&game=1&step=1");
            mockGameStateValue = null;
            renderWithProviders(<App />);
            // The accordion sections should render even if the run does
            // not exist server-side.
            expect(screen.getByText("Pipeline Status")).toBeInTheDocument();
        });
    });

    // ----------------------------------------------------------------
    // BUG-C2 follow-up: an error banner with a "Browse runs" escape
    // hatch when an explicit URL run cannot be loaded. The plan asks
    // for this in T2.6 because URL sovereignty alone is not enough --
    // if the user navigated to ?run=X and X is broken, they need to
    // see WHY the dashboard is empty and have a way out.
    // ----------------------------------------------------------------
    describe("error banner for failed-to-load explicit URL run", () => {
        it("shows a banner with the run name when /games returns an error", async () => {
            globalThis.history.replaceState(null, "", "/?run=broken-run&game=1&step=1");

            // /games for broken-run errored out -- React Query reports
            // ``isError=true`` for the games query.
            mockUseGames.mockReturnValue({
                data: undefined,
                isError: true,
                error: new Error("API error: 500 Internal Server Error"),
            } as unknown as ReturnType<typeof useGames>);
            mockGameStateValue = null;

            renderWithProviders(<App />);

            // The banner should mention the run name and the error.
            await waitFor(() => {
                expect(
                    screen.getByText(/broken-run/i),
                ).toBeInTheDocument();
            });
            // It should also offer a "Browse runs" escape hatch.
            expect(
                screen.getByRole("button", { name: /browse runs/i }),
            ).toBeInTheDocument();
        });

        it("clears the URL run when the Browse runs button is clicked", async () => {
            globalThis.history.replaceState(null, "", "/?run=broken-run&game=1&step=1");

            mockUseGames.mockReturnValue({
                data: undefined,
                isError: true,
                error: new Error("API error: 500 Internal Server Error"),
            } as unknown as ReturnType<typeof useGames>);
            mockGameStateValue = null;

            renderWithProviders(<App />);

            const browseBtn = await waitFor(() =>
                screen.getByRole("button", { name: /browse runs/i }),
            );
            browseBtn.click();

            // After click, the URL run should be cleared.
            await waitFor(() => {
                expect(globalThis.location.search).not.toContain("run=broken-run");
            });
        });

        it("does not show the banner when /games succeeds", async () => {
            globalThis.history.replaceState(null, "", "/?run=good-run&game=1&step=1");

            mockUseGames.mockReturnValue({
                data: [
                    { game_number: 1, steps: 50, start_ts: null, end_ts: null },
                ],
                isError: false,
                error: null,
            } as unknown as ReturnType<typeof useGames>);
            mockGameStateValue = null;

            renderWithProviders(<App />);

            // No "could not be loaded" text should appear.
            await new Promise((resolve) => setTimeout(resolve, 100));
            expect(
                screen.queryByText(/could not be loaded/i),
            ).not.toBeInTheDocument();
        });
    });
});
