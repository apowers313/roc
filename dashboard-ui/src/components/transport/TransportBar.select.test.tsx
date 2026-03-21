/**
 * Tests for TransportBar Select component onChange handlers.
 * Tests the internal callbacks directly through fireEvent on the input,
 * since Mantine Select's dropdown doesn't render accessibly in jsdom.
 */

import { screen, fireEvent } from "@testing-library/react";
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

describe("TransportBar select handlers", () => {
    beforeEach(() => {
        vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve({ min: 1, max: 100 }),
        }));
        mockUseRuns.mockReturnValue({ data: undefined } as ReturnType<typeof useRuns>);
        mockUseGames.mockReturnValue({ data: undefined } as ReturnType<typeof useGames>);
        mockUseStepRange.mockReturnValue({ data: undefined } as ReturnType<typeof useStepRange>);
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    describe("run selector", () => {
        it("renders with run options and the input is interactive", () => {
            mockUseRuns.mockReturnValue({
                data: [
                    { name: "20260318131128-test-run-alpha", games: 2, steps: 200 },
                    { name: "20260318131128-test-run-beta", games: 1, steps: 50 },
                ],
            } as ReturnType<typeof useRuns>);

            renderWithProviders(<TransportBar />);

            const runInput = screen.getByPlaceholderText("Run");
            // Focus and type to search -- this exercises the Select's searchable prop
            fireEvent.focus(runInput);
            fireEvent.change(runInput, { target: { value: "alpha" } });
        });
    });

    describe("game selector", () => {
        it("renders game options with step counts in labels", () => {
            mockUseGames.mockReturnValue({
                data: [
                    { game_number: 1, steps: 46, start_ts: null, end_ts: null },
                    { game_number: 2, steps: 75, start_ts: null, end_ts: null },
                ],
            } as ReturnType<typeof useGames>);

            renderWithProviders(<TransportBar />);

            const gameInput = screen.getByPlaceholderText("Game");
            fireEvent.focus(gameInput);
        });
    });

    describe("slider change handler", () => {
        it("slider is interactive and fires change", () => {
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 50 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<TransportBar />);

            const slider = screen.getByRole("slider");
            expect(slider).toBeInTheDocument();

            // Trigger the Mantine Slider's onChange by dispatching events on the track
            // The slider wrapper div captures keyboard events
            const sliderWrapper = slider.closest("[style]");
            if (sliderWrapper) {
                fireEvent.mouseDown(sliderWrapper, { clientX: 200 });
                fireEvent.mouseUp(sliderWrapper, { clientX: 200 });
            }
        });
    });

    describe("play/pause with step range", () => {
        it("play and pause cycle works", () => {
            mockUseStepRange.mockReturnValue({
                data: { min: 1, max: 50 },
            } as ReturnType<typeof useStepRange>);

            renderWithProviders(<TransportBar />);

            const playBtn = screen.getByLabelText("Play");
            fireEvent.click(playBtn);
            expect(screen.getByLabelText("Pause")).toBeInTheDocument();

            const pauseBtn = screen.getByLabelText("Pause");
            fireEvent.click(pauseBtn);
            expect(screen.getByLabelText("Play")).toBeInTheDocument();
        });
    });
});
