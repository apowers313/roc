import { screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock the query hooks to avoid actual API calls
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
});
