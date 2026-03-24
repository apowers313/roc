/** Prefetch a window of +/-RADIUS steps around the current position.
 *
 * Fetches outward from the current step (nearest first) using batch
 * requests to minimize HTTP round-trips. Each batch fetches BATCH_SIZE
 * steps in a single request. Skips steps already in the TanStack Query
 * cache. Uses AbortController to cancel in-flight prefetch requests
 * when the user navigates, preventing stale batch requests from
 * blocking the server's DuckDB lock.
 */

import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { fetchStepsBatch } from "../api/client";

const DEFAULT_RADIUS = 100;
const BATCH_SIZE = 50;
const DEBOUNCE_MS = 50;

export interface PrefetchOptions {
    radius?: number;
}

export function usePrefetchWindow(
    run: string,
    step: number,
    stepMin: number,
    stepMax: number,
    game?: number,
    options?: PrefetchOptions,
): void {
    const radius = options?.radius ?? DEFAULT_RADIUS;
    const queryClient = useQueryClient();
    const abortRef = useRef<AbortController | null>(null);
    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
        // Abort any in-flight prefetch requests and cancel pending sweep
        if (abortRef.current) abortRef.current.abort();
        if (timerRef.current) clearTimeout(timerRef.current);

        if (!run || step <= 0) return;

        const controller = new AbortController();
        abortRef.current = controller;

        timerRef.current = setTimeout(() => {
            void sweep({ run, center: step, stepMin, stepMax, game, radius, queryClient, signal: controller.signal });
        }, DEBOUNCE_MS);

        return () => {
            controller.abort();
            if (timerRef.current) clearTimeout(timerRef.current);
        };
    }, [queryClient, run, step, stepMin, stepMax, game, radius]);
}

interface SweepParams {
    run: string;
    center: number;
    stepMin: number;
    stepMax: number;
    game: number | undefined;
    radius: number;
    queryClient: ReturnType<typeof useQueryClient>;
    signal: AbortSignal;
}

/** Build the list of uncached steps to fetch, spreading outward from center. */
function buildTargets(params: SweepParams): number[] {
    const { run, center, stepMin, stepMax, game, radius, queryClient } = params;
    const targets: number[] = [];
    for (let offset = 1; offset <= radius; offset++) {
        const below = center - offset;
        if (below >= stepMin && !queryClient.getQueryData(["step", run, below, game])) {
            targets.push(below);
        }
        const above = center + offset;
        if (above <= stepMax && !queryClient.getQueryData(["step", run, above, game])) {
            targets.push(above);
        }
    }
    return targets;
}

/** Fetch one batch of steps and populate the query cache. Returns false on abort/error. */
async function fetchBatch(
    run: string,
    batch: number[],
    game: number | undefined,
    signal: AbortSignal,
    queryClient: SweepParams["queryClient"],
): Promise<boolean> {
    try {
        const results = await fetchStepsBatch(run, batch, game, signal);
        for (const [stepStr, data] of Object.entries(results)) {
            queryClient.setQueryData(["step", run, Number(stepStr), game], data);
        }
        return true;
    } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") return false;
        return false;
    }
}

async function sweep(params: SweepParams): Promise<void> {
    const { run, game, signal, queryClient } = params;
    const targets = buildTargets(params);

    for (let i = 0; i < targets.length; i += BATCH_SIZE) {
        if (signal.aborted) return;
        const batch = targets.slice(i, i + BATCH_SIZE);
        if (batch.length === 0) continue;
        const ok = await fetchBatch(run, batch, game, signal, queryClient);
        if (!ok) return;
    }
}
