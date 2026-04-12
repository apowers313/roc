/**
 * Tests for useCacheInvalidation.
 *
 * These lock in the query-key shapes that the centralized hook emits,
 * so that if somebody changes them the tests fail loudly instead of
 * the dashboard silently going stale after a game state transition.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import { useCacheInvalidation } from "./useCacheInvalidation";

function makeWrapper(queryClient: QueryClient) {
    return function Wrapper({ children }: Readonly<{ children: ReactNode }>) {
        return (
            <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
        );
    };
}

describe("useCacheInvalidation", () => {
    it("invalidateStepRange targets exactly [step-range, run]", () => {
        const client = new QueryClient();
        const spy = vi.spyOn(client, "invalidateQueries");
        const { result } = renderHook(() => useCacheInvalidation(), {
            wrapper: makeWrapper(client),
        });

        result.current.invalidateStepRange("run-42");

        expect(spy).toHaveBeenCalledWith({ queryKey: ["step-range", "run-42"] });
    });

    it("invalidateAllStepRanges targets the step-range prefix", () => {
        const client = new QueryClient();
        const spy = vi.spyOn(client, "invalidateQueries");
        const { result } = renderHook(() => useCacheInvalidation(), {
            wrapper: makeWrapper(client),
        });

        result.current.invalidateAllStepRanges();

        expect(spy).toHaveBeenCalledWith({ queryKey: ["step-range"] });
    });

    it("invalidateRunList targets exactly [runs]", () => {
        const client = new QueryClient();
        const spy = vi.spyOn(client, "invalidateQueries");
        const { result } = renderHook(() => useCacheInvalidation(), {
            wrapper: makeWrapper(client),
        });

        result.current.invalidateRunList();

        expect(spy).toHaveBeenCalledWith({ queryKey: ["runs"] });
    });

    it("returns a stable object identity across rerenders", () => {
        const client = new QueryClient();
        const { result, rerender } = renderHook(() => useCacheInvalidation(), {
            wrapper: makeWrapper(client),
        });

        const first = result.current;
        rerender();
        expect(result.current).toBe(first);
    });
});
