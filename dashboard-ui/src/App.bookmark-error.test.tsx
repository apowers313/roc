/**
 * Bookmark error-state regression test (BUG-M4).
 *
 * Clicking the "Add bookmark" button when no step data is loaded (e.g.,
 * the dashboard is in ERROR state because /games returned 500) used to
 * silently add a bookmark anyway -- creating a phantom bookmark for an
 * empty step that the user couldn't navigate back to. The fix: when
 * there's no step data, the App-level toggleBookmark wrapper must NOT
 * call into useBookmarks and must surface a notification so the user
 * understands why the click did nothing.
 *
 * The guard lives in App.tsx (not the hook) because "is there a step
 * loaded" is App-level state -- the hook deliberately stays pure.
 */

import { fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

vi.mock("socket.io-client");
vi.mock("./api/queries");
vi.mock("./hooks/usePrefetchWindow");
vi.mock("./hooks/useRunSubscription", () => ({
    useRunSubscription: vi.fn(),
    useGameState: vi.fn(() => null),
}));
vi.mock("./api/client");

const mockToggleBookmark = vi.fn();
vi.mock("./hooks/useBookmarks", () => ({
    useBookmarks: vi.fn(() => ({
        bookmarks: [],
        bookmarkSteps: [],
        isBookmarked: () => false,
        toggleBookmark: mockToggleBookmark,
        nextBookmark: () => null,
        prevBookmark: () => null,
        updateAnnotation: vi.fn(),
        updateBookmark: vi.fn(),
    })),
}));

import { renderWithProviders } from "./test-utils";
import { App } from "./App";
import { useStepData, useGames, useRuns, useStepRange } from "./api/queries";

const mockUseStepData = vi.mocked(useStepData);
const mockUseGames = vi.mocked(useGames);
const mockUseRuns = vi.mocked(useRuns);
const mockUseStepRange = vi.mocked(useStepRange);

describe("Bookmark guard when no step data (BUG-M4)", () => {
    beforeEach(() => {
        mockToggleBookmark.mockClear();
        vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve({ min: 1, max: 100 }),
        }));
        mockUseGames.mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useGames>);
        mockUseRuns.mockReturnValue({
            data: undefined,
        } as ReturnType<typeof useRuns>);
        mockUseStepRange.mockReturnValue({
            data: { min: 1, max: 100 },
        } as ReturnType<typeof useStepRange>);
    });

    afterEach(() => {
        vi.restoreAllMocks();
        sessionStorage.clear();
        globalThis.history.replaceState(null, "", "/");
    });

    it("does not call toggleBookmark when step data is missing", async () => {
        // Seed an explicit run so the bookmark button renders.
        globalThis.history.replaceState(null, "", "/?run=broken-run&game=1&step=10");

        // ERROR state: useStepData reports no data and an error.
        mockUseStepData.mockReturnValue({
            data: undefined,
            isLoading: false,
            isError: true,
            error: new Error("API error: 500 Internal Server Error"),
        } as unknown as ReturnType<typeof useStepData>);

        // Also flag the run as broken so the load-failure banner does not
        // shadow the bookmark button.
        mockUseGames.mockReturnValue({
            data: undefined,
            isError: false,
            error: null,
        } as unknown as ReturnType<typeof useGames>);

        renderWithProviders(<App />);

        // The Add bookmark button is in the StatusBar/transport. Click it.
        const btn = await waitFor(() =>
            screen.getByRole("button", { name: /add bookmark/i }),
        );
        fireEvent.click(btn);

        // The hook's toggleBookmark must NOT have been called.
        expect(mockToggleBookmark).not.toHaveBeenCalled();

        // A notification must be visible.
        await waitFor(() => {
            expect(
                screen.getByText(/cannot bookmark.*no step loaded/i),
            ).toBeInTheDocument();
        });
    });

    it("calls toggleBookmark normally when step data is present", async () => {
        globalThis.history.replaceState(null, "", "/?run=good-run&game=1&step=10");

        mockUseStepData.mockReturnValue({
            data: { step: 10, game_number: 1 } as unknown,
            isLoading: false,
            isError: false,
            error: null,
        } as unknown as ReturnType<typeof useStepData>);

        mockUseGames.mockReturnValue({
            data: [{ game_number: 1, steps: 100, start_ts: null, end_ts: null }],
            isError: false,
            error: null,
        } as unknown as ReturnType<typeof useGames>);

        renderWithProviders(<App />);

        const btn = await waitFor(() =>
            screen.getByRole("button", { name: /add bookmark/i }),
        );
        fireEvent.click(btn);

        expect(mockToggleBookmark).toHaveBeenCalledTimes(1);
        expect(mockToggleBookmark).toHaveBeenCalledWith(10, 1);

        // No "cannot bookmark" notification when the click succeeds.
        expect(
            screen.queryByText(/cannot bookmark/i),
        ).not.toBeInTheDocument();
    });
});
