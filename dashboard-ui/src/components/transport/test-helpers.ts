/** Shared test utilities for transport component tests. */

import { fireEvent, screen, waitFor } from "@testing-library/react";
import { type ReactElement } from "react";
import { expect, vi } from "vitest";

import { renderWithProviders } from "../../test-utils";

/** Mock fetch to return an idle game state response. Install via beforeEach. */
export function mockFetchIdle(): void {
    vi.spyOn(globalThis, "fetch").mockImplementation(() =>
        Promise.resolve(new Response(JSON.stringify({ state: "idle" }), { status: 200 })),
    );
}

/** Restore all mocks. Install via afterEach. */
export function restoreAllMocks(): void {
    vi.restoreAllMocks();
}

/** Override the mocked fetch to return a specific game state. */
export function mockGameState(state: string, extra?: Record<string, unknown>): void {
    vi.mocked(globalThis.fetch).mockResolvedValue(
        new Response(JSON.stringify({ state, ...extra }), { status: 200 }),
    );
}

/**
 * Render a component and open its game menu by clicking the specified trigger.
 * Returns the render result.
 */
export function renderAndOpenMenu(
    ui: ReactElement,
    triggerLabel: string,
    options?: { byLabelText?: boolean },
) {
    const result = renderWithProviders(ui);
    const trigger = options?.byLabelText
        ? screen.getByLabelText(triggerLabel)
        : screen.getByText(triggerLabel);
    fireEvent.click(trigger);
    return result;
}

/** Assert that a text element appears after menu is opened (waits for async render). */
export async function expectMenuText(text: string): Promise<void> {
    await waitFor(() => {
        expect(screen.getByText(text)).toBeInTheDocument();
    });
}

/** Assert that a specific API call was made with optional method. */
export async function expectApiCall(
    url: string,
    options?: { method: string },
): Promise<void> {
    await waitFor(() => {
        if (options) {
            expect(globalThis.fetch).toHaveBeenCalledWith(url, options);
        } else {
            expect(globalThis.fetch).toHaveBeenCalledWith(url);
        }
    });
}

/**
 * Set game state, render component, open menu via trigger, and assert expected text.
 * Combines the most common test pattern into a single call.
 */
export async function assertGameStateShowsText(
    ui: ReactElement,
    trigger: { label: string; byLabelText?: boolean },
    gameState: string,
    expectedText: string,
    gameStateExtra?: Record<string, unknown>,
): Promise<void> {
    if (gameState !== "idle" || gameStateExtra) {
        mockGameState(gameState, gameStateExtra);
    }
    renderAndOpenMenu(ui, trigger.label, { byLabelText: trigger.byLabelText });
    await expectMenuText(expectedText);
}

/**
 * Render component, open menu, click a menu item, then assert an API call was made.
 */
export async function assertMenuActionCallsApi(
    ui: ReactElement,
    trigger: { label: string; byLabelText?: boolean },
    menuItemText: string,
    apiUrl: string,
    apiOptions: { method: string },
    gameState?: string,
): Promise<void> {
    if (gameState) {
        mockGameState(gameState);
    }
    renderAndOpenMenu(ui, trigger.label, { byLabelText: trigger.byLabelText });
    await expectMenuText(menuItemText);
    fireEvent.click(screen.getByText(menuItemText));
    await expectApiCall(apiUrl, apiOptions);
}
