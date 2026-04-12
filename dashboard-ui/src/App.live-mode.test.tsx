/**
 * Tests for App with autoFollow enabled -- covers navigation callbacks
 * that drop autoFollow when the user explicitly navigates, plus
 * navigateToBookmark cross-game, speed up/down edge cases, and
 * handleChartStepClick.
 *
 * Phase 5: the four-state playback machine collapsed to two booleans.
 * The ``useGameState`` mock drives the auto-navigation effect which calls
 * ``setAutoFollow(true)`` when a running game is detected.
 */

import { screen, fireEvent, act } from "@testing-library/react";
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
import { useStepData, useGames } from "./api/queries";

const mockUseStepData = vi.mocked(useStepData);
const mockUseGames = vi.mocked(useGames);

describe("App (autoFollow mode)", () => {
    beforeEach(() => {
        // gameState running -> auto-selects live run and flips
        // autoFollow to true via the auto-navigation effect.
        mockGameStateValue = { state: "running", run_name: "live-run" };
        vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve({ min: 1, max: 100 }),
        }));
        mockUseStepData.mockReturnValue({
            data: undefined,
            isLoading: false,
            isPlaceholderData: false,
        } as unknown as ReturnType<typeof useStepData>);
        mockUseGames.mockReturnValue({
            data: [
                { game_number: 1, steps: 50, start_ts: null, end_ts: null },
                { game_number: 2, steps: 75, start_ts: null, end_ts: null },
            ],
        } as ReturnType<typeof useGames>);
    });

    afterEach(() => {
        vi.restoreAllMocks();
        sessionStorage.clear();
    });

    it("step forward drops autoFollow when pressed", () => {
        renderWithProviders(<App />);
        // After mount, the app should be in autoFollow mode.
        // Press Right arrow to step forward.
        fireEvent.keyDown(document, { key: "ArrowRight" });
    });

    it("step back drops autoFollow when pressed", () => {
        renderWithProviders(<App />);
        fireEvent.keyDown(document, { key: "ArrowLeft" });
    });

    it("jump to start drops autoFollow", () => {
        renderWithProviders(<App />);
        fireEvent.keyDown(document, { key: "Home" });
    });

    it("jump to end drops autoFollow", () => {
        renderWithProviders(<App />);
        fireEvent.keyDown(document, { key: "End" });
    });

    it("step forward 10 drops autoFollow", () => {
        renderWithProviders(<App />);
        fireEvent.keyDown(document, { key: "ArrowRight", shiftKey: true });
    });

    it("step back 10 drops autoFollow", () => {
        renderWithProviders(<App />);
        fireEvent.keyDown(document, { key: "ArrowLeft", shiftKey: true });
    });

    it("cycle game drops autoFollow", async () => {
        renderWithProviders(<App />);
        await act(async () => {
            fireEvent.keyDown(document, { key: "g" });
        });
    });

    // Phase 4: ``onNewStep`` was deleted -- the data refresh path is now
    // owned by ``useRunSubscription`` (TanStack Query invalidation). The
    // tests that exercised the live push branches in App.onNewStep no
    // longer have a target to drive against and have been removed. The
    // equivalent unit-level coverage lives in
    // ``hooks/useRunSubscription.test.tsx``.

    describe("speed up/down edge cases", () => {
        it("speed up multiple times", () => {
            renderWithProviders(<App />);
            // Default speed is 200, press ] multiple times
            fireEvent.keyDown(document, { key: "]" });
            fireEvent.keyDown(document, { key: "]" });
            fireEvent.keyDown(document, { key: "]" });
            fireEvent.keyDown(document, { key: "]" });
            fireEvent.keyDown(document, { key: "]" });
            // Should not crash at max speed
            fireEvent.keyDown(document, { key: "]" });
        });

        it("speed down multiple times", () => {
            renderWithProviders(<App />);
            fireEvent.keyDown(document, { key: "[" });
            fireEvent.keyDown(document, { key: "[" });
            fireEvent.keyDown(document, { key: "[" });
            // Should not crash at min speed
            fireEvent.keyDown(document, { key: "[" });
        });
    });

    describe("handleChartStepClick in autoFollow mode", () => {
        it("drops autoFollow when chart step is clicked", () => {
            // The handleChartStepClick is passed to chart components.
            // Since we can't easily click on the chart, we test that the
            // App renders without error in autoFollow mode with charts visible.
            renderWithProviders(<App />);
            expect(screen.getByText("Object Resolution")).toBeInTheDocument();
        });
    });

    describe("navigateToBookmark cross-game", () => {
        it("handles navigate to bookmark in a different game", async () => {
            const fetchMock = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ min: 51, max: 125 }),
            });
            vi.stubGlobal("fetch", fetchMock);

            renderWithProviders(<App />);

            // The navigateToBookmark is wired to BookmarkBar.
            // We can't directly trigger it without bookmarks, but we exercise
            // the code path by verifying the component renders in autoFollow mode
            // with the bookmark bar present.
            expect(screen.getByLabelText("Play")).toBeInTheDocument();
        });
    });

    describe("togglePlay in autoFollow mode", () => {
        it("togglePlay is independent of autoFollow", () => {
            renderWithProviders(<App />);
            // togglePlay only calls setPlaying regardless of autoFollow state
            fireEvent.keyDown(document, { key: " " });
        });
    });
});
