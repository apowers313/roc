/**
 * Additional coverage tests for TanStack Query hooks -- covers hooks that were
 * previously untested: useIntrinsicsHistory, useActionHistory,
 * useResolutionHistory, useAllObjects.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";

import {
    useActionHistory,
    useAllObjects,
    useIntrinsicsHistory,
    useResolutionHistory,
} from "./queries";

// Mock the client module
vi.mock("./client", () => ({
    fetchRuns: vi.fn(),
    fetchGames: vi.fn(),
    fetchStep: vi.fn(),
    fetchStepRange: vi.fn(),
    fetchMetricsHistory: vi.fn(),
    fetchGraphHistory: vi.fn(),
    fetchEventHistory: vi.fn(),
    fetchIntrinsicsHistory: vi.fn(),
    fetchActionHistory: vi.fn(),
    fetchResolutionHistory: vi.fn(),
    fetchAllObjects: vi.fn(),
}));

import {
    fetchIntrinsicsHistory,
    fetchActionHistory,
    fetchResolutionHistory,
    fetchAllObjects,
} from "./client";

const mockFetchIntrinsicsHistory = vi.mocked(fetchIntrinsicsHistory);
const mockFetchActionHistory = vi.mocked(fetchActionHistory);
const mockFetchResolutionHistory = vi.mocked(fetchResolutionHistory);
const mockFetchAllObjects = vi.mocked(fetchAllObjects);

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

describe("TanStack Query hooks (additional coverage)", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    describe("useIntrinsicsHistory", () => {
        it("fetches intrinsics history when run is provided", async () => {
            const data = [{ step: 1, raw: { hp: 10 }, normalized: { hp: 0.5 } }];
            mockFetchIntrinsicsHistory.mockResolvedValue(data);

            const { result } = renderHook(() => useIntrinsicsHistory("run1"), {
                wrapper: createWrapper(),
            });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));
            expect(result.current.data).toEqual(data);
        });

        it("does not fetch when run is empty", () => {
            const { result } = renderHook(() => useIntrinsicsHistory(""), {
                wrapper: createWrapper(),
            });

            expect(result.current.fetchStatus).toBe("idle");
            expect(mockFetchIntrinsicsHistory).not.toHaveBeenCalled();
        });

        it("passes game parameter to client", async () => {
            mockFetchIntrinsicsHistory.mockResolvedValue([]);

            const { result } = renderHook(() => useIntrinsicsHistory("run1", 2), {
                wrapper: createWrapper(),
            });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));
            expect(mockFetchIntrinsicsHistory).toHaveBeenCalledWith("run1", 2);
        });
    });

    describe("useActionHistory", () => {
        it("fetches action history when run is provided", async () => {
            const data = [{ step: 1, action_id: 5, action_name: "move_north" }];
            mockFetchActionHistory.mockResolvedValue(data);

            const { result } = renderHook(() => useActionHistory("run1"), {
                wrapper: createWrapper(),
            });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));
            expect(result.current.data).toEqual(data);
        });

        it("does not fetch when run is empty", () => {
            const { result } = renderHook(() => useActionHistory(""), {
                wrapper: createWrapper(),
            });

            expect(result.current.fetchStatus).toBe("idle");
            expect(mockFetchActionHistory).not.toHaveBeenCalled();
        });
    });

    describe("useResolutionHistory", () => {
        it("fetches resolution history when run is provided", async () => {
            const data = [{ step: 1, outcome: "match", correct: true }];
            mockFetchResolutionHistory.mockResolvedValue(data);

            const { result } = renderHook(() => useResolutionHistory("run1"), {
                wrapper: createWrapper(),
            });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));
            expect(result.current.data).toEqual(data);
        });

        it("does not fetch when run is empty", () => {
            const { result } = renderHook(() => useResolutionHistory(""), {
                wrapper: createWrapper(),
            });

            expect(result.current.fetchStatus).toBe("idle");
            expect(mockFetchResolutionHistory).not.toHaveBeenCalled();
        });
    });

    describe("useAllObjects", () => {
        it("fetches all objects when run is provided", async () => {
            const data = [
                {
                    shape: "circle",
                    glyph: "@",
                    color: "white",
                    node_id: "n1",
                    step_added: 1,
                    match_count: 5,
                    feature_type: "glyph",
                },
            ];
            mockFetchAllObjects.mockResolvedValue(data);

            const { result } = renderHook(() => useAllObjects("run1"), {
                wrapper: createWrapper(),
            });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));
            expect(result.current.data).toEqual(data);
        });

        it("does not fetch when run is empty", () => {
            const { result } = renderHook(() => useAllObjects(""), {
                wrapper: createWrapper(),
            });

            expect(result.current.fetchStatus).toBe("idle");
            expect(mockFetchAllObjects).not.toHaveBeenCalled();
        });

        it("passes game parameter to client", async () => {
            mockFetchAllObjects.mockResolvedValue([]);

            const { result } = renderHook(() => useAllObjects("run1", 3), {
                wrapper: createWrapper(),
            });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));
            expect(mockFetchAllObjects).toHaveBeenCalledWith("run1", 3);
        });
    });
});
