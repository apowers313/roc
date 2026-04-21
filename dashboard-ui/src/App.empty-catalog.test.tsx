/**
 * Empty-catalog run regression tests.
 *
 * Some runs end up on disk with status="ok" in /api/runs but their
 * DuckLake catalog has zero tables -- e.g., the writer was started
 * but the ParquetExporter background thread never wrote any rows
 * before the game was killed. For those runs:
 *
 *   - GET /api/runs/<run>/games  -> 200 with []
 *   - GET /api/runs/<run>/step-range -> 200 with {min:0,max:0}
 *   - GET /api/runs/<run>/step/N -> 404 out_of_range
 *
 * The Mantine ``Select`` component used for the Game dropdown is
 * non-interactive when its ``data`` prop is an empty array (clicks
 * register on the input but the popup never opens). The user is
 * stranded -- the Run dropdown still works but it is far from
 * obvious that the Game dropdown is "empty" rather than "broken".
 *
 * The Phase 7 fix surfaces the same red Alert that ``BUG-C2`` /
 * ``BUG-M3`` use for /games failures, so the user sees a clear
 * "could not be loaded" message with a "Browse runs" escape hatch.
 *
 * These tests pin the new contract:
 *   1. When games is [] and step-range is {min:0,max:0}, the Alert
 *      is visible.
 *   2. When games has entries OR step-range is non-zero, the Alert
 *      is NOT shown -- we do not regress on the working case.
 *   3. When the run is empty string, the Alert is NOT shown -- the
 *      auto-select-first-run effect handles that case separately.
 */

import { screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

vi.mock("socket.io-client");
vi.mock("./api/queries");
vi.mock("./hooks/usePrefetchWindow");
vi.mock("./hooks/useRunSubscription", () => ({
    useRunSubscription: vi.fn(),
    useGameState: vi.fn(() => null),
    useSocketConnected: vi.fn(() => true),
}));
vi.mock("./api/client");

import { renderWithProviders } from "./test-utils";
import { App } from "./App";
import { useStepData, useGames, useRuns, useStepRange } from "./api/queries";

const mockUseStepData = vi.mocked(useStepData);
const mockUseGames = vi.mocked(useGames);
const mockUseRuns = vi.mocked(useRuns);
const mockUseStepRange = vi.mocked(useStepRange);

describe("Empty-catalog run banner", () => {
    beforeEach(() => {
        vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve({ min: 0, max: 0 }),
        }));
        mockUseStepData.mockReturnValue({
            data: undefined,
            isLoading: false,
            isPlaceholderData: false,
            isError: true,
            error: new Error("API error: 404 Not Found"),
        } as unknown as ReturnType<typeof useStepData>);
        mockUseRuns.mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useRuns>);
    });

    afterEach(() => {
        vi.restoreAllMocks();
        sessionStorage.clear();
        globalThis.history.replaceState(null, "", "/");
    });

    it("shows the load-failure banner when games=[] and step-range=0/0", async () => {
        globalThis.history.replaceState(null, "", "/?run=empty-catalog-run&game=1&step=1");

        // /games returns 200 with []  -- no error, just no games
        mockUseGames.mockReturnValue({
            data: [],
            isError: false,
            error: null,
        } as unknown as ReturnType<typeof useGames>);

        // /step-range returns {min:0,max:0}
        mockUseStepRange.mockReturnValue({
            data: { min: 0, max: 0 },
        } as ReturnType<typeof useStepRange>);

        renderWithProviders(<App />);

        await waitFor(() => {
            expect(
                screen.getByText(/Run "empty-catalog-run" could not be loaded/i),
            ).toBeInTheDocument();
        });

        // The user must have an escape hatch.
        expect(
            screen.getByRole("button", { name: /browse runs/i }),
        ).toBeInTheDocument();

        // The body of the alert should mention the empty-catalog reason.
        expect(
            screen.getByText(/empty catalog/i),
        ).toBeInTheDocument();
    });

    it("does NOT show the banner when games has entries (working run)", async () => {
        globalThis.history.replaceState(null, "", "/?run=working-run&game=1&step=10");

        mockUseGames.mockReturnValue({
            data: [{ game_number: 1, steps: 100, start_ts: null, end_ts: null }],
            isError: false,
            error: null,
        } as unknown as ReturnType<typeof useGames>);

        mockUseStepRange.mockReturnValue({
            data: { min: 1, max: 100 },
        } as ReturnType<typeof useStepRange>);

        renderWithProviders(<App />);

        // Give effects a chance to run.
        await new Promise((resolve) => setTimeout(resolve, 100));

        expect(
            screen.queryByText(/could not be loaded/i),
        ).not.toBeInTheDocument();
    });

    it("does NOT show the banner when run is empty (no URL navigation)", async () => {
        globalThis.history.replaceState(null, "", "/");

        mockUseGames.mockReturnValue({
            data: [],
            isError: false,
            error: null,
        } as unknown as ReturnType<typeof useGames>);

        mockUseStepRange.mockReturnValue({
            data: { min: 0, max: 0 },
        } as ReturnType<typeof useStepRange>);

        renderWithProviders(<App />);

        await new Promise((resolve) => setTimeout(resolve, 100));

        expect(
            screen.queryByText(/could not be loaded/i),
        ).not.toBeInTheDocument();
    });

    it("does NOT show the banner while step-range is still loading (undefined)", async () => {
        globalThis.history.replaceState(null, "", "/?run=loading-run&game=1&step=1");

        mockUseGames.mockReturnValue({
            data: [],
            isError: false,
            error: null,
        } as unknown as ReturnType<typeof useGames>);

        // step-range still loading
        mockUseStepRange.mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useStepRange>);

        renderWithProviders(<App />);

        await new Promise((resolve) => setTimeout(resolve, 100));

        // We must NOT flash the empty-catalog banner before the
        // step-range query has had a chance to resolve, otherwise the
        // banner appears for one frame on every navigation. The two
        // conditions (games=[] AND step-range=0/0) must both be
        // observed before deciding the catalog is empty.
        expect(
            screen.queryByText(/empty catalog/i),
        ).not.toBeInTheDocument();
    });
});
