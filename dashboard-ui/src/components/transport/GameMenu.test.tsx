import { screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

import { renderWithProviders } from "../../test-utils";
import { GameMenu } from "./GameMenu";
import {
    assertGameStateShowsText,
    assertMenuActionCallsApi,
    expectApiCall,
    expectMenuText,
    mockFetchIdle,
    mockGameState,
    renderAndOpenMenu,
    restoreAllMocks,
} from "./test-helpers";

const TRIGGER = { label: "Game menu", byLabelText: true };
const ui = (<GameMenu />);

describe("GameMenu", () => {
    beforeEach(() => {
        mockFetchIdle();
    });

    afterEach(() => {
        restoreAllMocks();
    });

    it("renders the game menu button with aria label", () => {
        renderWithProviders(<GameMenu />);
        expect(screen.getByLabelText("Game menu")).toBeInTheDocument();
    });

    it("shows 'No game running' in dropdown when idle", async () => {
        renderAndOpenMenu(ui, TRIGGER.label, { byLabelText: true });
        await expectMenuText("No game running");
    });

    it("shows Start Game menu item when idle", async () => {
        renderAndOpenMenu(ui, TRIGGER.label, { byLabelText: true });
        await expectMenuText("Start Game");
    });

    it("shows Number of games input when idle", async () => {
        renderAndOpenMenu(ui, TRIGGER.label, { byLabelText: true });
        await expectMenuText("Number of games");
    });

    it("calls /api/game/status on menu open", async () => {
        renderAndOpenMenu(ui, TRIGGER.label, { byLabelText: true });
        await expectApiCall("/api/game/status");
    });

    it("shows 'Game Running' badge when state is running", async () => {
        await assertGameStateShowsText(ui, TRIGGER, "running", "Game Running");
    });

    it("shows Stop Game button when game is running", async () => {
        await assertGameStateShowsText(ui, TRIGGER, "running", "Stop Game");
    });

    it("shows 'Stopping...' badge when state is stopping", async () => {
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

    it("shows error message when game has error and is not running", async () => {
        await assertGameStateShowsText(ui, TRIGGER, "idle", "Game crashed", { error: "Game crashed" });
    });

    it("handles fetch error gracefully (does not crash)", async () => {
        vi.mocked(globalThis.fetch).mockRejectedValue(new Error("Network error"));
        renderAndOpenMenu(ui, TRIGGER.label, { byLabelText: true });
        await waitFor(() => {
            expect(screen.getByText("No game running")).toBeInTheDocument();
        });
    });

    it("shows initializing state as running", async () => {
        mockGameState("initializing");
        renderAndOpenMenu(ui, TRIGGER.label, { byLabelText: true });
        await expectMenuText("Game Running");
    });
});
