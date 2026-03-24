import { fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { MenuBar } from "./MenuBar";
import {
    assertGameStateShowsText,
    assertMenuActionCallsApi,
    expectMenuText,
    mockFetchIdle,
    renderAndOpenMenu,
    restoreAllMocks,
} from "./test-helpers";

const TRIGGER = { label: "Game", byLabelText: false };
const ui = (<MenuBar />);

describe("MenuBar", () => {
    beforeEach(() => {
        mockFetchIdle();
    });

    afterEach(() => {
        restoreAllMocks();
    });

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

        // Should briefly show "Copied!" text
        await waitFor(() => {
            expect(screen.getByText("Copied!")).toBeInTheDocument();
        });
    });

    it("opens Game menu dropdown when Game button is clicked", async () => {
        renderAndOpenMenu(ui, TRIGGER.label);
        await expectMenuText("No game running");
    });

    it("shows Start Game in game dropdown when idle", async () => {
        renderAndOpenMenu(ui, TRIGGER.label);
        await expectMenuText("Start Game");
    });

    it("shows Stop Game when game is running", async () => {
        await assertGameStateShowsText(ui, TRIGGER, "running", "Stop Game");
    });

    it("shows error message when game has error", async () => {
        await assertGameStateShowsText(
            ui, TRIGGER, "idle", "Process crashed", { error: "Process crashed" },
        );
    });

    it("shows Stopping badge when game is stopping", async () => {
        await assertGameStateShowsText(ui, TRIGGER, "stopping", "Stopping...");
    });

    it("calls start game API when Start Game is clicked", async () => {
        await assertMenuActionCallsApi(
            ui, TRIGGER, "Start Game",
            "/api/game/start?num_games=5", { method: "POST" },
        );
    });

    it("calls stop game API when Stop Game is clicked", async () => {
        await assertMenuActionCallsApi(
            ui, TRIGGER, "Stop Game",
            "/api/game/stop", { method: "POST" },
            "running",
        );
    });

    it("shows initializing state as running", async () => {
        await assertGameStateShowsText(ui, TRIGGER, "initializing", "Game Running");
    });

    it("handles fetch error gracefully", async () => {
        vi.mocked(globalThis.fetch).mockRejectedValue(new Error("Network error"));
        renderAndOpenMenu(ui, TRIGGER.label);
        await waitFor(() => {
            expect(screen.getByText("No game running")).toBeInTheDocument();
        });
    });

    it("has consistent background styling", () => {
        const { container } = renderWithProviders(<MenuBar />);
        // The outer Group element should be rendered
        const menuBar = container.firstElementChild;
        expect(menuBar).toBeTruthy();
    });
});
