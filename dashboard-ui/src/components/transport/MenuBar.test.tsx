/**
 * MenuBar tests -- post-consolidation.
 *
 * MenuBar now reads game state from ``useGameState`` (the dashboard's
 * single source of truth) instead of maintaining its own fetch +
 * useState copy. These tests mock the hook directly and drive
 * different game states through it.
 *
 * The previous shared helpers (``mockFetchIdle``, ``mockGameState``)
 * were built around the old per-component fetch pattern and are no
 * longer meaningful here -- the whole point of the consolidation is
 * that MenuBar no longer talks to the network at all for state.
 */

import { fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

vi.mock("../../hooks/useRunSubscription", () => ({
    useGameState: vi.fn(() => ({ state: "idle", run_name: null })),
    useRunSubscription: vi.fn(),
    useSocketConnected: vi.fn(() => true),
}));

import { renderWithProviders } from "../../test-utils";
import { MenuBar } from "./MenuBar";
import { useGameState, type GameState } from "../../hooks/useRunSubscription";

const mockUseGameState = vi.mocked(useGameState);

function setGameState(state: Partial<GameState> & { state: string }) {
    mockUseGameState.mockReturnValue({
        run_name: null,
        exit_code: null,
        error: null,
        ...state,
    } as GameState);
}

function openMenu() {
    fireEvent.click(screen.getByText("Game"));
}

beforeEach(() => {
    vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(new Response("{}", { status: 200 })),
    );
    setGameState({ state: "idle" });
});

afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
});

describe("MenuBar", () => {
    it("renders the Game button", () => {
        renderWithProviders(<MenuBar />);
        expect(screen.getByText("Game")).toBeInTheDocument();
    });

    it("renders the Copy Link button", () => {
        renderWithProviders(<MenuBar />);
        expect(screen.getByText("Copy Link")).toBeInTheDocument();
    });

    it("copies URL to clipboard when Copy Link is clicked", async () => {
        const writeTextSpy = vi.fn().mockResolvedValue(undefined);
        Object.defineProperty(navigator, "clipboard", {
            value: { writeText: writeTextSpy },
            configurable: true,
            writable: true,
        });

        renderWithProviders(<MenuBar />);
        fireEvent.click(screen.getByText("Copy Link"));

        await waitFor(() => {
            expect(writeTextSpy).toHaveBeenCalled();
        });
        await waitFor(() => {
            expect(screen.getByText("Copied!")).toBeInTheDocument();
        });
    });

    it("opens Game menu dropdown when Game button is clicked", async () => {
        renderWithProviders(<MenuBar />);
        openMenu();
        expect(await screen.findByText("No game running")).toBeInTheDocument();
    });

    it("shows Start Game in dropdown when idle", async () => {
        renderWithProviders(<MenuBar />);
        openMenu();
        expect(await screen.findByText("Start Game")).toBeInTheDocument();
    });

    it("shows Stop Game when game is running", async () => {
        setGameState({ state: "running", run_name: "live-run" });
        renderWithProviders(<MenuBar />);
        openMenu();
        expect(await screen.findByText("Stop Game")).toBeInTheDocument();
    });

    // TC-GAME-004 consolidation: the ``error`` field must survive
    // state transitions because useGameState now carries it from both
    // the Socket.io event and the initial REST fetch. Previously
    // MenuBar held its own fetch copy so an error could disappear
    // whenever Socket.io emitted a state update without the field.
    it("shows error message when idle state carries an error", async () => {
        setGameState({ state: "idle", error: "Process crashed" });
        renderWithProviders(<MenuBar />);
        openMenu();
        expect(await screen.findByText("Process crashed")).toBeInTheDocument();
    });

    it("shows Stopping badge when game is stopping", async () => {
        setGameState({ state: "stopping", run_name: "live-run" });
        renderWithProviders(<MenuBar />);
        openMenu();
        expect(await screen.findByText("Stopping...")).toBeInTheDocument();
    });

    it("shows initializing state as running", async () => {
        setGameState({ state: "initializing", run_name: "live-run" });
        renderWithProviders(<MenuBar />);
        openMenu();
        expect(await screen.findByText("Game Running")).toBeInTheDocument();
    });

    it("calls /api/game/start when Start Game is clicked", async () => {
        renderWithProviders(<MenuBar />);
        openMenu();
        fireEvent.click(await screen.findByText("Start Game"));
        await waitFor(() => {
            expect(globalThis.fetch).toHaveBeenCalledWith(
                "/api/game/start?num_games=5",
                { method: "POST" },
            );
        });
    });

    it("calls /api/game/stop when Stop Game is clicked", async () => {
        setGameState({ state: "running", run_name: "live-run" });
        renderWithProviders(<MenuBar />);
        openMenu();
        fireEvent.click(await screen.findByText("Stop Game"));
        await waitFor(() => {
            expect(globalThis.fetch).toHaveBeenCalledWith(
                "/api/game/stop",
                { method: "POST" },
            );
        });
    });

    it("shows error message when start game fetch is rejected", async () => {
        vi.mocked(globalThis.fetch).mockRejectedValue(new Error("Network down"));
        renderWithProviders(<MenuBar />);
        openMenu();
        fireEvent.click(await screen.findByText("Start Game"));
        await waitFor(() => {
            expect(screen.getByText("Start game failed: Network down")).toBeInTheDocument();
        });
    });

    it("shows error message when start game returns non-ok", async () => {
        vi.mocked(globalThis.fetch).mockResolvedValue(
            new Response(null, { status: 500, statusText: "Internal Server Error" }),
        );
        renderWithProviders(<MenuBar />);
        openMenu();
        fireEvent.click(await screen.findByText("Start Game"));
        await waitFor(() => {
            expect(
                screen.getByText("Start game failed: 500 Internal Server Error"),
            ).toBeInTheDocument();
        });
    });

    it("does not fetch /api/game/status directly", async () => {
        // MenuBar must not hold its own game-status fetch -- the hook
        // owns it. Drives home the TC-GAME-004 consolidation invariant.
        renderWithProviders(<MenuBar />);
        openMenu();
        await screen.findByText("Start Game");
        const calls = vi.mocked(globalThis.fetch).mock.calls;
        for (const call of calls) {
            expect(call[0]).not.toBe("/api/game/status");
        }
    });
});
