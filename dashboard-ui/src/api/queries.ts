/** TanStack Query hooks for data fetching with caching. */

import { keepPreviousData, useQuery } from "@tanstack/react-query";

import {
    fetchEventHistory,
    fetchGames,
    fetchGraphHistory,
    fetchMetricsHistory,
    fetchRuns,
    fetchStep,
    fetchStepRange,
} from "./client";

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

// History queries cache for 5 minutes. Data is keyed by [run, game] so
// switching runs/games triggers a fresh fetch. Within the same game the
// data only grows as new steps arrive during live play -- a 5-minute
// window avoids redundant refetches while ensuring the chart eventually
// picks up new data. TanStack Query's gcTime (default 5 min) evicts
// entries once they lose all active observers, preventing memory buildup
// when browsing many runs.
const HISTORY_STALE_MS = 5 * 60_000;

export function useMetricsHistory(run: string, game?: number, fields?: string[]) {
    return useQuery({
        queryKey: ["metrics-history", run, game, fields],
        queryFn: () => fetchMetricsHistory(run, game, fields),
        enabled: run !== "",
        staleTime: HISTORY_STALE_MS,
    });
}

export function useGraphHistory(run: string, game?: number) {
    return useQuery({
        queryKey: ["graph-history", run, game],
        queryFn: () => fetchGraphHistory(run, game),
        enabled: run !== "",
        staleTime: HISTORY_STALE_MS,
    });
}

export function useEventHistory(run: string, game?: number) {
    return useQuery({
        queryKey: ["event-history", run, game],
        queryFn: () => fetchEventHistory(run, game),
        enabled: run !== "",
        staleTime: HISTORY_STALE_MS,
    });
}

