/** TanStack Query hooks for data fetching with caching. */

import { keepPreviousData, useQuery } from "@tanstack/react-query";

import {
    fetchActionHistory,
    fetchActionMap,
    fetchEventHistory,
    fetchFrameGraph,
    fetchGames,
    fetchGraphHistory,
    fetchIntrinsicsHistory,
    fetchMetricsHistory,
    fetchAllObjects,
    fetchObjectHistory,
    fetchObjectHistoryGraph,
    fetchResolutionHistory,
    fetchRuns,
    fetchSchema,
    fetchStep,
    fetchStepRange,
} from "./client";

export function useRuns(includeAll = false) {
    return useQuery({
        queryKey: ["runs", includeAll],
        queryFn: () => fetchRuns(includeAll),
        refetchInterval: 10_000,
    });
}

export function useGames(run: string) {
    return useQuery({
        queryKey: ["games", run],
        queryFn: () => fetchGames(run),
        enabled: run !== "",
        // Refetch periodically so step counts and new games appear during live play
        refetchInterval: 10_000,
    });
}

export function useStepData(
    run: string,
    step: number,
    game?: number,
) {
    return useQuery({
        queryKey: ["step", run, step, game],
        queryFn: ({ signal }) => fetchStep(run, step, game, signal),
        enabled: run !== "" && step > 0,
        staleTime: Infinity, // step data is immutable
        retry: false, // don't retry on rapid navigation cancellations
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

export function useIntrinsicsHistory(run: string, game?: number) {
    return useQuery({
        queryKey: ["intrinsics-history", run, game],
        queryFn: () => fetchIntrinsicsHistory(run, game),
        enabled: run !== "",
        staleTime: HISTORY_STALE_MS,
    });
}

export function useActionHistory(run: string, game?: number) {
    return useQuery({
        queryKey: ["action-history", run, game],
        queryFn: () => fetchActionHistory(run, game),
        enabled: run !== "",
        staleTime: HISTORY_STALE_MS,
    });
}

export function useResolutionHistory(run: string, game?: number) {
    return useQuery({
        queryKey: ["resolution-history", run, game],
        queryFn: () => fetchResolutionHistory(run, game),
        enabled: run !== "",
        staleTime: HISTORY_STALE_MS,
    });
}

export function useAllObjects(run: string, game?: number) {
    return useQuery({
        queryKey: ["all-objects", run, game],
        queryFn: () => fetchAllObjects(run, game),
        enabled: run !== "",
        staleTime: HISTORY_STALE_MS,
    });
}

export function useActionMap(run: string) {
    return useQuery({
        queryKey: ["action-map", run],
        queryFn: async () => {
            try {
                return await fetchActionMap(run);
            } catch {
                // Return empty array on 404 / network error so the query
                // doesn't enter a permanent error state.
                return [] as Awaited<ReturnType<typeof fetchActionMap>>;
            }
        },
        enabled: run !== "",
        staleTime: Infinity, // immutable once we have data
        // Retry every 2s while the map is empty (race: game subprocess
        // may not have sent the map yet when the dashboard first queries).
        refetchInterval: (query) => {
            const data = query.state.data;
            return data && data.length > 0 ? false : 2_000;
        },
    });
}

export function useSchema(run: string) {
    return useQuery({
        queryKey: ["schema", run],
        queryFn: () => fetchSchema(run),
        enabled: run !== "",
        staleTime: Infinity, // schema is immutable per run
        retry: false,
    });
}

export function useObjectHistory(run: string, objectId: number | null) {
    return useQuery({
        queryKey: ["object-history", run, objectId],
        queryFn: () => fetchObjectHistory(run, objectId!),
        enabled: run !== "" && objectId != null,
        staleTime: HISTORY_STALE_MS,
    });
}

export function useFrameGraph(run: string, tick: number, game?: number, depth?: number) {
    return useQuery({
        queryKey: ["frame-graph", run, tick, game, depth],
        queryFn: ({ signal }) => fetchFrameGraph(run, tick, game, depth, signal),
        enabled: run !== "" && tick > 0,
        staleTime: Infinity, // graph data is immutable per run/tick
        retry: false,
        placeholderData: keepPreviousData,
    });
}

export function useObjectHistoryGraph(run: string, uuid: string | null) {
    // uuid is a string because ROC UUIDs are 63-bit ints (see ObjectHistoryInfo
    // for details). Passing a number here would already have lost precision.
    return useQuery({
        queryKey: ["object-history-graph", run, uuid],
        queryFn: () => fetchObjectHistoryGraph(run, uuid!),
        enabled: run !== "" && uuid != null && uuid !== "",
        staleTime: Infinity, // object history is immutable per run/uuid
        retry: false,
    });
}
