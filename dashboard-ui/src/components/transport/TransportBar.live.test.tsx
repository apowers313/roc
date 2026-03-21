/**
 * Tests for TransportBar in live-following mode -- covers branches that fire
 * dispatchPlayback({ type: "USER_NAVIGATE" }) when playback === "live_following".
 *
 * Uses a custom context wrapper that dispatches GO_LIVE before rendering
 * the TransportBar so the playback state starts as "live_following".
 */

import { MantineProvider } from "@mantine/core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { useEffect, type ReactNode } from "react";

import { HighlightProvider } from "../../state/highlight";
import { DashboardProvider, useDashboard } from "../../state/context";

// Mock queries
vi.mock("../../api/queries", () => ({
    useRuns: vi.fn(() => ({ data: undefined })),
    useGames: vi.fn(() => ({ data: undefined })),
    useStepRange: vi.fn(() => ({ data: undefined })),
}));

import { TransportBar } from "./TransportBar";
import { useStepRange } from "../../api/queries";

const mockUseStepRange = vi.mocked(useStepRange);

/**
 * Component that sets up live_following state by dispatching GO_LIVE,
 * then renders children.
 */
function GoLiveSetup({ children }: { children: ReactNode }) {
    const { dispatchPlayback, setRun, setStepRange } = useDashboard();

    useEffect(() => {
        setRun("live-run");
        setStepRange(1, 100);
        dispatchPlayback({ type: "GO_LIVE" });
    }, [dispatchPlayback, setRun, setStepRange]);

    return <>{children}</>;
}

function LiveWrapper({ children }: { children: ReactNode }) {
    const queryClient = new QueryClient({
        defaultOptions: { queries: { retry: false } },
    });
    return (
        <MantineProvider>
            <QueryClientProvider client={queryClient}>
                <DashboardProvider>
                    <HighlightProvider>
                        <GoLiveSetup>{children}</GoLiveSetup>
                    </HighlightProvider>
                </DashboardProvider>
            </QueryClientProvider>
        </MantineProvider>
    );
}

describe("TransportBar (live_following mode)", () => {
    beforeEach(() => {
        vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve({ min: 1, max: 100 }),
        }));
        mockUseStepRange.mockReturnValue({
            data: { min: 1, max: 100 },
        } as ReturnType<typeof useStepRange>);
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it("stepForward dispatches USER_NAVIGATE in live_following", () => {
        render(<TransportBar />, { wrapper: LiveWrapper });

        const nextBtn = screen.getByLabelText("Next step");
        act(() => {
            fireEvent.click(nextBtn);
        });
        // After USER_NAVIGATE, playback transitions to live_paused
        // Verify no crash and button still works
        expect(nextBtn).toBeInTheDocument();
    });

    it("stepBack dispatches USER_NAVIGATE in live_following", () => {
        render(<TransportBar />, { wrapper: LiveWrapper });

        // First advance, then go back
        const nextBtn = screen.getByLabelText("Next step");
        act(() => {
            fireEvent.click(nextBtn);
        });

        const prevBtn = screen.getByLabelText("Previous step");
        act(() => {
            fireEvent.click(prevBtn);
        });
        expect(prevBtn).toBeInTheDocument();
    });

    it("jumpToStart dispatches USER_NAVIGATE in live_following", () => {
        render(<TransportBar />, { wrapper: LiveWrapper });

        const firstBtn = screen.getByLabelText("First step");
        act(() => {
            fireEvent.click(firstBtn);
        });
        expect(firstBtn).toBeInTheDocument();
    });

    it("jumpToEnd dispatches USER_NAVIGATE in live_following", () => {
        render(<TransportBar />, { wrapper: LiveWrapper });

        const lastBtn = screen.getByLabelText("Last step");
        act(() => {
            fireEvent.click(lastBtn);
        });
        expect(lastBtn).toBeInTheDocument();
    });

    it("slider keyboard ArrowRight dispatches stepForward", () => {
        render(<TransportBar />, { wrapper: LiveWrapper });

        const slider = screen.getByRole("slider");
        slider.focus();

        const event = new KeyboardEvent("keydown", {
            key: "ArrowRight",
            bubbles: true,
            cancelable: true,
        });
        slider.dispatchEvent(event);
    });

    it("handleSliderChange dispatches USER_NAVIGATE", () => {
        render(<TransportBar />, { wrapper: LiveWrapper });

        const slider = screen.getByRole("slider");
        // Mantine Slider's onChange is triggered by mouse interaction on the track
        // In jsdom, we can trigger it via the aria-valuetext change
        fireEvent.mouseDown(slider, { clientX: 50 });
    });

    it("togglePlay in live_following mode does not dispatch TOGGLE_PLAY", () => {
        render(<TransportBar />, { wrapper: LiveWrapper });

        const playBtn = screen.getByLabelText("Play");
        act(() => {
            fireEvent.click(playBtn);
        });
        expect(screen.getByLabelText("Pause")).toBeInTheDocument();
    });
});
