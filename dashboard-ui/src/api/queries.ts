/** TanStack Query hooks for data fetching with caching. */

import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { fetchGames, fetchRuns, fetchStep, fetchStepRange } from "./client";

export function useRuns() {
    return useQuery({
        queryKey: ["runs"],
        queryFn: fetchRuns,
        refetchInterval: 10_000,
    });
}

export function useGames(run: string) {
    return useQuery({
        queryKey: ["games", run],
        queryFn: () => fetchGames(run),
        enabled: run !== "",
    });
}

export function useStepData(run: string, step: number, game?: number) {
    return useQuery({
        queryKey: ["step", run, step, game],
        queryFn: () => fetchStep(run, step, game),
        enabled: run !== "" && step > 0,
        staleTime: Infinity, // step data is immutable
        retry: false, // don't retry on rapid navigation cancellations
        // Keep previous step's data visible while the next step loads.
        // This prevents flicker (alternating "No data" / data) during
        // playback.  Safe because the DuckLake catalog always returns
        // correct data for each query key.
        placeholderData: keepPreviousData,
    });
}

export function useStepRange(run: string, game?: number) {
    return useQuery({
        queryKey: ["step-range", run, game],
        queryFn: () => fetchStepRange(run, game),
        enabled: run !== "",
    });
}

/** Debounced prefetch of adjacent steps for smooth scrubbing. */
export function usePrefetchAdjacentSteps(
    run: string,
    step: number,
    game?: number,
) {
    const queryClient = useQueryClient();
    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
        if (!run || step <= 0) return;

        // Debounce: only prefetch after user stops clicking for 300ms
        if (timerRef.current) clearTimeout(timerRef.current);
        timerRef.current = setTimeout(() => {
            for (const offset of [-1, 1]) {
                const target = step + offset;
                if (target > 0) {
                    void queryClient.prefetchQuery({
                        queryKey: ["step", run, target, game],
                        queryFn: () => fetchStep(run, target, game),
                        staleTime: Infinity,
                    });
                }
            }
        }, 300);

        return () => {
            if (timerRef.current) clearTimeout(timerRef.current);
        };
    }, [queryClient, run, step, game]);
}
