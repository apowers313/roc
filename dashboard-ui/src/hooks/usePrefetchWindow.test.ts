import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { createElement } from "react";

vi.mock("../api/client", () => ({
    fetchStepsBatch: vi.fn(),
}));

import { usePrefetchWindow } from "./usePrefetchWindow";
import { fetchStepsBatch } from "../api/client";

const mockFetchStepsBatch = vi.mocked(fetchStepsBatch);

function createWrapper() {
    const queryClient = new QueryClient({
        defaultOptions: { queries: { retry: false } },
    });
    return {
        queryClient,
        Wrapper({ children }: { children: ReactNode }) {
            return createElement(
                QueryClientProvider,
                { client: queryClient },
                children,
            );
        },
    };
}

describe("usePrefetchWindow", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        vi.useFakeTimers();
        mockFetchStepsBatch.mockResolvedValue({});
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it("does not prefetch when run is empty", () => {
        const { Wrapper } = createWrapper();
        renderHook(
            () => usePrefetchWindow("", 50, 1, 100),
            { wrapper: Wrapper },
        );
        act(() => { vi.advanceTimersByTime(500); });
        expect(mockFetchStepsBatch).not.toHaveBeenCalled();
    });

    it("does not prefetch when step is 0", () => {
        const { Wrapper } = createWrapper();
        renderHook(
            () => usePrefetchWindow("run1", 0, 1, 100),
            { wrapper: Wrapper },
        );
        act(() => { vi.advanceTimersByTime(500); });
        expect(mockFetchStepsBatch).not.toHaveBeenCalled();
    });

    it("fetches batch of steps after debounce", () => {
        const { Wrapper } = createWrapper();
        renderHook(
            () => usePrefetchWindow("run1", 50, 1, 100),
            { wrapper: Wrapper },
        );

        expect(mockFetchStepsBatch).not.toHaveBeenCalled();

        act(() => { vi.advanceTimersByTime(400); });
        expect(mockFetchStepsBatch).toHaveBeenCalled();

        const firstBatch = mockFetchStepsBatch.mock.calls[0]![1];
        expect(firstBatch).toContain(49);
        expect(firstBatch).toContain(51);
    });

    it("uses batch size of 50 for efficiency", () => {
        const { Wrapper } = createWrapper();
        renderHook(
            () => usePrefetchWindow("run1", 50, 1, 100),
            { wrapper: Wrapper },
        );

        act(() => { vi.advanceTimersByTime(400); });

        // First batch should have at most 50 steps
        const firstBatch = mockFetchStepsBatch.mock.calls[0]![1];
        expect(firstBatch.length).toBeLessThanOrEqual(50);
    });

    it("respects stepMin and stepMax bounds", () => {
        const { Wrapper } = createWrapper();
        renderHook(
            () => usePrefetchWindow("run1", 3, 1, 10),
            { wrapper: Wrapper },
        );

        act(() => { vi.advanceTimersByTime(400); });

        for (const call of mockFetchStepsBatch.mock.calls) {
            for (const s of call[1]) {
                expect(s).toBeGreaterThanOrEqual(1);
                expect(s).toBeLessThanOrEqual(10);
            }
        }

        const allSteps = mockFetchStepsBatch.mock.calls.flatMap((c) => c[1]);
        expect(allSteps).not.toContain(3);
    });

    it("passes AbortSignal to fetchStepsBatch", () => {
        const { Wrapper } = createWrapper();
        renderHook(
            () => usePrefetchWindow("run1", 50, 1, 100),
            { wrapper: Wrapper },
        );

        act(() => { vi.advanceTimersByTime(400); });

        // The 4th argument should be an AbortSignal
        const signal = mockFetchStepsBatch.mock.calls[0]![3];
        expect(signal).toBeInstanceOf(AbortSignal);
        expect(signal!.aborted).toBe(false);
    });

    it("aborts in-flight requests on step change", () => {
        const { Wrapper } = createWrapper();
        const { rerender } = renderHook(
            ({ step }) => usePrefetchWindow("run1", step, 1, 200),
            { wrapper: Wrapper, initialProps: { step: 50 } },
        );

        act(() => { vi.advanceTimersByTime(400); });

        // Capture the signal from the first sweep
        const firstSignal = mockFetchStepsBatch.mock.calls[0]![3] as AbortSignal;
        expect(firstSignal.aborted).toBe(false);

        // Change step -- should abort the old signal
        rerender({ step: 100 });
        expect(firstSignal.aborted).toBe(true);

        // New sweep should use a fresh signal
        mockFetchStepsBatch.mockClear();
        act(() => { vi.advanceTimersByTime(400); });

        const newSignal = mockFetchStepsBatch.mock.calls[0]![3] as AbortSignal;
        expect(newSignal.aborted).toBe(false);
        expect(newSignal).not.toBe(firstSignal);
    });

    it("skips steps that are already cached", () => {
        const { Wrapper, queryClient } = createWrapper();

        queryClient.setQueryData(["step", "run1", 49, undefined], { step: 49 });
        queryClient.setQueryData(["step", "run1", 51, undefined], { step: 51 });

        renderHook(
            () => usePrefetchWindow("run1", 50, 1, 100),
            { wrapper: Wrapper },
        );

        act(() => { vi.advanceTimersByTime(400); });

        const allSteps = mockFetchStepsBatch.mock.calls.flatMap((c) => c[1]);
        expect(allSteps).not.toContain(49);
        expect(allSteps).not.toContain(51);
        expect(allSteps).toContain(48);
        expect(allSteps).toContain(52);
    });

    it("populates cache from batch response", async () => {
        vi.useRealTimers();
        const { Wrapper, queryClient } = createWrapper();

        let resolveRequest!: (v: Record<string, never>) => void;
        mockFetchStepsBatch.mockReturnValue(
            new Promise((r) => { resolveRequest = r; }),
        );

        renderHook(
            () => usePrefetchWindow("run1", 50, 48, 52),
            { wrapper: Wrapper },
        );

        await new Promise((r) => setTimeout(r, 400));

        await act(async () => {
            resolveRequest({
                "49": { step: 49, game_number: 1 },
                "51": { step: 51, game_number: 1 },
            } as unknown as Record<string, never>);
            await new Promise((r) => setTimeout(r, 10));
        });

        expect(queryClient.getQueryData(["step", "run1", 49, undefined])).toEqual({
            step: 49,
            game_number: 1,
        });
        expect(queryClient.getQueryData(["step", "run1", 51, undefined])).toEqual({
            step: 51,
            game_number: 1,
        });
    });
});
