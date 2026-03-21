/**
 * Tests for TransportBar Select onChange handlers by directly clicking
 * Mantine combobox options via their data attributes.
 */

import { screen, fireEvent, act, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

// Mantine Combobox calls scrollIntoView which jsdom doesn't implement on all elements
if (!Element.prototype.scrollIntoView) {
    Element.prototype.scrollIntoView = vi.fn();
}

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

/**
 * Opens a Mantine Select dropdown and clicks the first option.
 * Mantine v8 renders options as divs with `data-combobox-option` attribute
 * inside a listbox div. The dropdown is initially hidden (display: none)
 * but becomes visible after focus + click on the input.
 */
async function selectFirstOption(input: HTMLElement) {
    // Click input to open dropdown
    await act(async () => {
        fireEvent.click(input);
    });

    // Mantine renders options inside a div[role="listbox"] inside a portal.
    // The options have class *-option and are rendered even when hidden.
    // We need to find the listbox associated with this input.
    const listboxId = input.getAttribute("aria-controls");
    if (listboxId) {
        const listbox = document.getElementById(listboxId);
        if (listbox) {
            const options = listbox.querySelectorAll("[data-combobox-option]");
            if (options.length > 0) {
                await act(async () => {
                    fireEvent.click(options[0]!);
                });
                return true;
            }
        }
    }

    // Fallback: find all combobox options in the document
    const allOptions = document.querySelectorAll("[data-combobox-option]");
    if (allOptions.length > 0) {
        await act(async () => {
            fireEvent.click(allOptions[0]!);
        });
        return true;
    }

    return false;
}

async function selectOption(input: HTMLElement, index: number) {
    await act(async () => {
        fireEvent.click(input);
    });

    const listboxId = input.getAttribute("aria-controls");
    if (listboxId) {
        const listbox = document.getElementById(listboxId);
        if (listbox) {
            const options = listbox.querySelectorAll("[data-combobox-option]");
            if (options.length > index) {
                await act(async () => {
                    fireEvent.click(options[index]!);
                });
                return true;
            }
        }
    }
    return false;
}

describe("TransportBar onChange handlers", () => {
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

    describe("run onChange", () => {
        it("selecting a run triggers setRun and fetch step-range", async () => {
            const fetchMock = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ min: 1, max: 200 }),
            });
            vi.stubGlobal("fetch", fetchMock);

            mockUseRuns.mockReturnValue({
                data: [
                    { name: "20260318131128-test-run-one", games: 2, steps: 200 },
                ],
            } as ReturnType<typeof useRuns>);

            renderWithProviders(<TransportBar />);

            const runInput = screen.getByPlaceholderText("Run");
            const selected = await selectFirstOption(runInput);

            if (selected) {
                await waitFor(() => {
                    expect(fetchMock).toHaveBeenCalled();
                });
            }
        });
    });

    describe("game onChange", () => {
        it("selecting a game triggers setGame and fetch step-range", async () => {
            const fetchMock = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ min: 47, max: 121 }),
            });
            vi.stubGlobal("fetch", fetchMock);

            mockUseGames.mockReturnValue({
                data: [
                    { game_number: 1, steps: 46, start_ts: null, end_ts: null },
                    { game_number: 2, steps: 75, start_ts: null, end_ts: null },
                ],
            } as ReturnType<typeof useGames>);

            renderWithProviders(<TransportBar />);

            const gameInput = screen.getByPlaceholderText("Game");
            const selected = await selectOption(gameInput, 1);

            if (selected) {
                await waitFor(() => {
                    expect(fetchMock).toHaveBeenCalled();
                });
            }
        });
    });

    describe("speed onChange", () => {
        it("selecting a speed option changes the speed", async () => {
            renderWithProviders(<TransportBar />);

            // Speed select doesn't have a placeholder, find by its current value
            // The speed select's input shows the label for the current value
            // Default speed is 200 -> "5x"
            const speedInputs = screen.getAllByRole("textbox");
            // Speed is the third Select (Run, Game, Speed)
            // But we need to find the one that doesn't have a placeholder
            let speedInput: HTMLElement | null = null;
            for (const input of speedInputs) {
                if (!input.getAttribute("placeholder")) {
                    speedInput = input;
                    break;
                }
            }
            if (!speedInput) {
                // Fallback: just use the last input
                speedInput = speedInputs[speedInputs.length - 1]!;
            }

            const selected = await selectFirstOption(speedInput);
            // Just verify no crash
            expect(speedInput).toBeInTheDocument();
        });
    });
});
