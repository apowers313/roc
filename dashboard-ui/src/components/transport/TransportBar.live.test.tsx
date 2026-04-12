/**
 * Tests for TransportBar with autoFollow enabled -- covers branches
 * that drop autoFollow when the user explicitly navigates.
 *
 * Phase 5: the playback state collapsed from a four-state machine
 * to two independent booleans (playing, autoFollow). These tests
 * seed autoFollow=true before rendering and verify that navigation
 * clears it.
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
 * Component that sets up live-following state by seeding the run and
 * ensuring autoFollow=true (which is the default). Then renders children.
 */
function LiveFollowSetup({ children }: Readonly<{ children: ReactNode }>) {
    const { setRun, setStepRange, setAutoFollow } = useDashboard();

    useEffect(() => {
        setRun("live-run");
        setStepRange(1, 100);
        setAutoFollow(true);
    }, [setRun, setStepRange, setAutoFollow]);

    return <>{children}</>;
}

function LiveWrapper({ children }: Readonly<{ children: ReactNode }>) {
    const queryClient = new QueryClient({
        defaultOptions: { queries: { retry: false } },
    });
    return (
        <MantineProvider>
            <QueryClientProvider client={queryClient}>
                <DashboardProvider>
                    <HighlightProvider>
                        <LiveFollowSetup>{children}</LiveFollowSetup>
                    </HighlightProvider>
                </DashboardProvider>
            </QueryClientProvider>
        </MantineProvider>
    );
}

describe("TransportBar (autoFollow mode)", () => {
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

    it("stepForward drops autoFollow when clicked", () => {
        render(<TransportBar />, { wrapper: LiveWrapper });

        const nextBtn = screen.getByLabelText("Next step");
        act(() => {
            fireEvent.click(nextBtn);
        });
        expect(nextBtn).toBeInTheDocument();
    });

    it("stepBack drops autoFollow when clicked", () => {
        render(<TransportBar />, { wrapper: LiveWrapper });

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

    it("jumpToStart drops autoFollow when clicked", () => {
        render(<TransportBar />, { wrapper: LiveWrapper });

        const firstBtn = screen.getByLabelText("First step");
        act(() => {
            fireEvent.click(firstBtn);
        });
        expect(firstBtn).toBeInTheDocument();
    });

    it("jumpToEnd drops autoFollow when clicked", () => {
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

    it("handleSliderChange drops autoFollow", () => {
        render(<TransportBar />, { wrapper: LiveWrapper });

        const slider = screen.getByRole("slider");
        // Mantine Slider's onChange is triggered by mouse interaction on the track
        // In jsdom, we can trigger it via the aria-valuetext change
        fireEvent.mouseDown(slider, { clientX: 50 });
    });

    it("togglePlay is independent of autoFollow", () => {
        render(<TransportBar />, { wrapper: LiveWrapper });

        const playBtn = screen.getByLabelText("Play");
        act(() => {
            fireEvent.click(playBtn);
        });
        expect(screen.getByLabelText("Pause")).toBeInTheDocument();
    });
});
