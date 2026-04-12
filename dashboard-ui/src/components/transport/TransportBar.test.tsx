import { screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock the query hooks to avoid actual API calls
vi.mock("../../api/queries", () => ({
    useRuns: vi.fn(() => ({ data: undefined })),
    useGames: vi.fn(() => ({ data: undefined })),
    useStepRange: vi.fn(() => ({ data: undefined })),
}));

import { renderWithProviders } from "../../test-utils";
import { TransportBar, buildRunOption } from "./TransportBar";
import { useRuns, useGames, useStepRange } from "../../api/queries";
import type { RunSummary } from "../../types/api";

const mockUseRuns = vi.mocked(useRuns);
const mockUseGames = vi.mocked(useGames);
const mockUseStepRange = vi.mocked(useStepRange);

describe("TransportBar", () => {
    beforeEach(() => {
        vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve({ min: 1, max: 100 }),
        }));
        // Reset to default returns
        mockUseRuns.mockReturnValue({ data: undefined } as ReturnType<typeof useRuns>);
        mockUseGames.mockReturnValue({ data: undefined } as ReturnType<typeof useGames>);
        mockUseStepRange.mockReturnValue({ data: undefined } as ReturnType<typeof useStepRange>);
    });

    it("renders transport controls", () => {
        renderWithProviders(<TransportBar />);

        expect(screen.getByLabelText("First step")).toBeInTheDocument();
        expect(screen.getByLabelText("Previous step")).toBeInTheDocument();
        expect(screen.getByLabelText("Play")).toBeInTheDocument();
        expect(screen.getByLabelText("Next step")).toBeInTheDocument();
        expect(screen.getByLabelText("Last step")).toBeInTheDocument();
    });

    it("shows connected indicator when connected=true", () => {
        const { container } = renderWithProviders(
            <TransportBar connected={true} />,
        );
        const dot = container.querySelector('[title="Connected"]');
        expect(dot).toBeTruthy();
    });

    it("shows disconnected indicator when connected=false", () => {
        const { container } = renderWithProviders(
            <TransportBar connected={false} />,
        );
        const dot = container.querySelector('[title="Disconnected"]');
        expect(dot).toBeTruthy();
    });

    it("does not show connection indicator when connected is undefined", () => {
        const { container } = renderWithProviders(<TransportBar />);
        expect(container.querySelector('[title="Connected"]')).toBeNull();
        expect(container.querySelector('[title="Disconnected"]')).toBeNull();
    });

    it("renders run selector with run options", () => {
        mockUseRuns.mockReturnValue({
            data: [
                { name: "run-1", games: 2, steps: 100 },
                { name: "run-2", games: 1, steps: 50 },
            ],
        } as ReturnType<typeof useRuns>);

        renderWithProviders(<TransportBar />);
        expect(screen.getByPlaceholderText("Run")).toBeInTheDocument();
    });

    it("shows step counter text", () => {
        mockUseStepRange.mockReturnValue({
            data: { min: 1, max: 50 },
        } as ReturnType<typeof useStepRange>);

        renderWithProviders(<TransportBar />);
        // Default step=1, range 1-50 -> "1 / 50"
        expect(screen.getByText("1 / 50")).toBeInTheDocument();
    });

    it("toggles play/pause label on click", () => {
        renderWithProviders(<TransportBar />);

        const playBtn = screen.getByLabelText("Play");
        expect(playBtn).toBeInTheDocument();

        fireEvent.click(playBtn);
        expect(screen.getByLabelText("Pause")).toBeInTheDocument();
    });

    // ---- Step range per game ----
    //
    // The slider counter displays: (step - effectiveMin + 1) / (effectiveMax - effectiveMin + 1)
    // where effectiveMin/Max come from useStepRange(run, game). When a different
    // game is selected, useStepRange returns that game's range and the counter
    // should reflect the new total.

    describe("slider step range per game", () => {
        it("shows correct range for game 1 (200 steps)", () => {
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 200 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<TransportBar />);
            // step=1 (context default), min=1, max=200 -> "1 / 200"
            expect(screen.getByText("1 / 200")).toBeInTheDocument();
        });

        it("shows correct total for game 2 with offset range", () => {
            // Game 2 might have global step numbers 201-275
            mockUseStepRange.mockReturnValue({
                data: { min: 201, max: 275 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<TransportBar />);
            // Total steps in the range = 275 - 201 + 1 = 75
            // The counter text includes "/ 75" as the denominator.
            // (The current position may be negative if step hasn't been
            // set to min yet, but the total is always correct.)
            expect(screen.getByText(/\/ 75$/)).toBeInTheDocument();
        });

        it("shows correct total for game 3 with offset range", () => {
            // Game 3: steps 276-305 (30 steps)
            mockUseStepRange.mockReturnValue({
                data: { min: 276, max: 305 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<TransportBar />);
            expect(screen.getByText(/\/ 30$/)).toBeInTheDocument();
        });

        it("falls back to context stepMin/stepMax when useStepRange returns no data", () => {
            mockUseStepRange.mockReturnValue({
                data: undefined,
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<TransportBar />);
            // Context defaults: stepMin=1, stepMax=1, step=1 -> "1 / 1"
            expect(screen.getByText("1 / 1")).toBeInTheDocument();
        });

        it("renders game selector with per-game step counts in labels", () => {
            mockUseGames.mockReturnValue({
                data: [
                    { game_number: 1, steps: 200, start_ts: null, end_ts: null },
                    { game_number: 2, steps: 75, start_ts: null, end_ts: null },
                    { game_number: 3, steps: 30, start_ts: null, end_ts: null },
                ],
            } as ReturnType<typeof useGames>);

            renderWithProviders(<TransportBar />);
            expect(screen.getByPlaceholderText("Game")).toBeInTheDocument();
        });

        it("different games produce different slider totals", () => {
            // Render with game 1 range
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 100 },
            } as ReturnType<typeof useStepRange>);
            const { unmount } = renderWithProviders(<TransportBar />);
            expect(screen.getByText("1 / 100")).toBeInTheDocument();
            unmount();

            // Render with game 2 range (different total)
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 50 },
            } as ReturnType<typeof useStepRange>);
            renderWithProviders(<TransportBar />);
            expect(screen.getByText("1 / 50")).toBeInTheDocument();
        });
    });

    // Phase 4: TanStack Query is the only data path. Socket.io is
    // invalidation-only -- the ``step_added`` handler calls
    // ``invalidateQueries`` and the ``useStepRange`` refetch delivers
    // fresh data. So REST is always authoritative, whether the run is
    // historical or tail-growing. The earlier "Socket.io pushes beat
    // stale REST" guard belonged to Phase 3 (liveData/onNewStep) and
    // caused TC-GAME-004: with the guard, a live run with autoFollow
    // off showed a frozen slider max, so clicking GO LIVE snapped to
    // a stale head instead of the real one.
    describe("step range data source", () => {
        it("falls back to context range when REST data is undefined", () => {
            mockUseStepRange.mockReturnValue({
                data: undefined,
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<TransportBar />);
            // Context defaults: stepMin=1, stepMax=1 -> "1 / 1"
            expect(screen.getByText("1 / 1")).toBeInTheDocument();
        });

        it("shows REST range for historical mode", () => {
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 500 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<TransportBar />);
            expect(screen.getByText("1 / 500")).toBeInTheDocument();
        });

        // TC-GAME-004 regression: when viewing a tail-growing live run,
        // the slider max must reflect the fresh REST data as the range
        // grows, NOT a frozen context snapshot. Otherwise clicking the
        // GO LIVE badge (after the user broke auto-follow) snaps to a
        // stale head that is hundreds of steps behind the true one.
        it("shows REST range for tail-growing live run (TC-GAME-004)", () => {
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 500, tail_growing: true },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<TransportBar />);
            expect(screen.getByText("1 / 500")).toBeInTheDocument();
        });
    });

    // Regression: clicking the slider then pressing arrow keys moved 2 steps
    // because the Slider's native arrow key behavior fired alongside our
    // global keyboard shortcuts. The fix: capture-phase onKeyDown on the
    // slider wrapper that stops propagation + prevents default for nav keys.
    describe("slider keyboard isolation", () => {
        it("arrow keys on focused slider are stopped by capture handler", () => {
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 100 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<TransportBar />);
            const slider = screen.getByRole("slider");

            // Focus the slider thumb
            slider.focus();

            // Fire an ArrowRight on the slider -- should be prevented
            const event = new KeyboardEvent("keydown", {
                key: "ArrowRight",
                bubbles: true,
                cancelable: true,
            });
            const wasPrevented = !slider.dispatchEvent(event);
            expect(wasPrevented).toBe(true);
        });
    });

    // ---------------------------------------------------------------
    // "Show all runs" toggle (regression for missing-runs-in-dropdown)
    //
    // Three protections must hold simultaneously:
    //   1. The toggle starts off and reads "Show all" so the user can find it.
    //   2. The dropdown labels mark non-ok runs ("[short]"/"[empty]"/etc).
    //   3. If the active run is not in the default list (e.g. URL param
    //      points at a corrupt run), the toggle auto-flips to true so
    //      the dropdown is not silently blank.
    // ---------------------------------------------------------------
    describe("show-all-runs toggle and status badges", () => {
        beforeEach(() => {
            // Reset localStorage between tests so persistence does
            // not leak across cases.
            window.localStorage.clear();
        });

        it("renders the Show all checkbox", () => {
            renderWithProviders(<TransportBar />);
            expect(screen.getByLabelText("Show all")).toBeInTheDocument();
        });

        it("buildRunOption tags non-ok runs", () => {
            const ok: RunSummary = {
                name: "20260408120000-rancorous-fey-devy",
                games: 1,
                steps: 314,
                status: "ok",
            };
            const short: RunSummary = {
                name: "20260408120100-tiny-foo-bar",
                games: 1,
                steps: 5,
                status: "short",
            };
            const empty: RunSummary = {
                name: "20260408120200-zero-foo-bar",
                games: 0,
                steps: 0,
                status: "empty",
            };
            const corrupt: RunSummary = {
                name: "20260408120300-bad-foo-bar",
                games: 0,
                steps: 0,
                status: "corrupt",
                error: "DuckLake catalog open failed",
            };
            const okLabel = buildRunOption(ok).label;
            expect(okLabel).toContain("rancorous-fey-devy");
            // ok labels carry only the date prefix [MM/DD HH:MM] -- no status tag
            expect(okLabel).not.toMatch(/\[(short|empty|corrupt|missing)\]/);
            expect(buildRunOption(short).label).toContain("[short]");
            expect(buildRunOption(empty).label).toContain("[empty]");
            expect(buildRunOption(corrupt).label).toContain("[corrupt]");
        });

        it("buildRunOption falls back to ok when status is omitted", () => {
            const r: RunSummary = {
                name: "20260101000000-foo-bar-baz",
                games: 1,
                steps: 50,
            };
            const opt = buildRunOption(r);
            expect(opt.label).not.toMatch(/\[(short|empty|corrupt|missing)\]/);
            expect(opt.label).toContain("[01/01 00:00]");
        });

        it("calls useRuns with includeAll=false by default", () => {
            mockUseRuns.mockClear();
            renderWithProviders(<TransportBar />);
            // First render after stubGlobal
            expect(mockUseRuns).toHaveBeenCalledWith(false);
        });

        it("calls useRuns with includeAll=true after toggling Show all", async () => {
            mockUseRuns.mockReturnValue({
                data: [],
            } as unknown as ReturnType<typeof useRuns>);
            renderWithProviders(<TransportBar />);
            const checkbox = screen.getByLabelText("Show all") as HTMLInputElement;
            expect(checkbox.checked).toBe(false);
            fireEvent.click(checkbox);
            await waitFor(() => {
                expect(checkbox.checked).toBe(true);
            });
            // After toggling, the most recent useRuns call must be true.
            const calls = mockUseRuns.mock.calls;
            const lastCall = calls[calls.length - 1];
            expect(lastCall?.[0]).toBe(true);
        });

        it(
            "remembers the show-all setting via localStorage across remounts",
            () => {
                window.localStorage.setItem("roc.dashboard.showAllRuns", "1");
                mockUseRuns.mockClear();
                renderWithProviders(<TransportBar />);
                // First call must already pass true because the saved setting wins.
                expect(mockUseRuns.mock.calls[0]?.[0]).toBe(true);
                const checkbox = screen.getByLabelText("Show all") as HTMLInputElement;
                expect(checkbox.checked).toBe(true);
            },
        );
    });
});
