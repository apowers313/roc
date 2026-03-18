import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";

import { useEventHistory, useGames, useGraphHistory, useMetricsHistory, useRuns, useStepData, useStepRange } from "./queries";

// Mock the client module
vi.mock("./client", () => ({
    fetchRuns: vi.fn(),
    fetchGames: vi.fn(),
    fetchStep: vi.fn(),
    fetchStepRange: vi.fn(),
    fetchMetricsHistory: vi.fn(),
    fetchGraphHistory: vi.fn(),
    fetchEventHistory: vi.fn(),
}));

import { fetchRuns, fetchGames, fetchStep, fetchStepRange, fetchMetricsHistory, fetchGraphHistory, fetchEventHistory } from "./client";

const mockFetchRuns = vi.mocked(fetchRuns);
const mockFetchGames = vi.mocked(fetchGames);
const mockFetchStep = vi.mocked(fetchStep);
const mockFetchStepRange = vi.mocked(fetchStepRange);
const mockFetchMetricsHistory = vi.mocked(fetchMetricsHistory);
const mockFetchGraphHistory = vi.mocked(fetchGraphHistory);
const mockFetchEventHistory = vi.mocked(fetchEventHistory);

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

    describe("useMetricsHistory", () => {
        it("fetches metrics history", async () => {
            const data = [{ step: 1, hp: 10 }];
            mockFetchMetricsHistory.mockResolvedValue(data);

            const { result } = renderHook(() => useMetricsHistory("run1"), {
                wrapper: createWrapper(),
            });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));
            expect(result.current.data).toEqual(data);
        });

        it("does not fetch when run is empty", () => {
            const { result } = renderHook(() => useMetricsHistory(""), {
                wrapper: createWrapper(),
            });

            expect(result.current.fetchStatus).toBe("idle");
        });
    });

    describe("useGraphHistory", () => {
        it("fetches graph history", async () => {
            const data = [{ step: 1, node_count: 10, node_max: 100, edge_count: 20, edge_max: 200 }];
            mockFetchGraphHistory.mockResolvedValue(data);

            const { result } = renderHook(() => useGraphHistory("run1"), {
                wrapper: createWrapper(),
            });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));
            expect(result.current.data).toEqual(data);
        });

        it("does not fetch when run is empty", () => {
            const { result } = renderHook(() => useGraphHistory(""), {
                wrapper: createWrapper(),
            });

            expect(result.current.fetchStatus).toBe("idle");
        });
    });

    describe("useEventHistory", () => {
        it("fetches event history", async () => {
            const data = [{ step: 1, "roc.perception": 5 }];
            mockFetchEventHistory.mockResolvedValue(data);

            const { result } = renderHook(() => useEventHistory("run1"), {
                wrapper: createWrapper(),
            });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));
            expect(result.current.data).toEqual(data);
        });

        it("does not fetch when run is empty", () => {
            const { result } = renderHook(() => useEventHistory(""), {
                wrapper: createWrapper(),
            });

            expect(result.current.fetchStatus).toBe("idle");
        });
    });
});
