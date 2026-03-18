import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";

import { useGames, useRuns, useStepData, useStepRange } from "./queries";

// Mock the client module
vi.mock("./client", () => ({
    fetchRuns: vi.fn(),
    fetchGames: vi.fn(),
    fetchStep: vi.fn(),
    fetchStepRange: vi.fn(),
}));

import { fetchRuns, fetchGames, fetchStep, fetchStepRange } from "./client";

const mockFetchRuns = vi.mocked(fetchRuns);
const mockFetchGames = vi.mocked(fetchGames);
const mockFetchStep = vi.mocked(fetchStep);
const mockFetchStepRange = vi.mocked(fetchStepRange);

function createWrapper() {
    const queryClient = new QueryClient({
        defaultOptions: { queries: { retry: false } },
    });
    return function Wrapper({ children }: { children: ReactNode }) {
        return (
            <QueryClientProvider client={queryClient}>
                {children}
            </QueryClientProvider>
        );
    };
}

describe("TanStack Query hooks", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    describe("useRuns", () => {
        it("fetches runs", async () => {
            const runs = [{ name: "run1", games: 1, steps: 10 }];
            mockFetchRuns.mockResolvedValue(runs);

            const { result } = renderHook(() => useRuns(), {
                wrapper: createWrapper(),
            });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));
            expect(result.current.data).toEqual(runs);
        });
    });

    describe("useGames", () => {
        it("fetches games when run is provided", async () => {
            const games = [
                { game_number: 1, steps: 50, start_ts: null, end_ts: null },
            ];
            mockFetchGames.mockResolvedValue(games);

            const { result } = renderHook(() => useGames("run1"), {
                wrapper: createWrapper(),
            });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));
            expect(result.current.data).toEqual(games);
        });

        it("does not fetch when run is empty", () => {
            const { result } = renderHook(() => useGames(""), {
                wrapper: createWrapper(),
            });

            expect(result.current.fetchStatus).toBe("idle");
            expect(mockFetchGames).not.toHaveBeenCalled();
        });
    });

    describe("useStepData", () => {
        it("fetches step data", async () => {
            const stepData = { step: 5, game_number: 1 };
            mockFetchStep.mockResolvedValue(stepData as ReturnType<typeof mockFetchStep> extends Promise<infer T> ? T : never);

            const { result } = renderHook(() => useStepData("run1", 5), {
                wrapper: createWrapper(),
            });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));
            expect(result.current.data).toEqual(stepData);
        });

        it("does not fetch when step is 0", () => {
            const { result } = renderHook(() => useStepData("run1", 0), {
                wrapper: createWrapper(),
            });

            expect(result.current.fetchStatus).toBe("idle");
        });

        it("does not fetch when run is empty", () => {
            const { result } = renderHook(() => useStepData("", 5), {
                wrapper: createWrapper(),
            });

            expect(result.current.fetchStatus).toBe("idle");
        });
    });

    describe("useStepRange", () => {
        it("fetches step range", async () => {
            const range = { min: 1, max: 100 };
            mockFetchStepRange.mockResolvedValue(range);

            const { result } = renderHook(() => useStepRange("run1"), {
                wrapper: createWrapper(),
            });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));
            expect(result.current.data).toEqual(range);
        });

        it("does not fetch when run is empty", () => {
            const { result } = renderHook(() => useStepRange(""), {
                wrapper: createWrapper(),
            });

            expect(result.current.fetchStatus).toBe("idle");
        });
    });
});
