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
            void sweep(run, step, stepMin, stepMax, game, radius, queryClient, controller.signal);
        }, DEBOUNCE_MS);

        return () => {
            controller.abort();
            if (timerRef.current) clearTimeout(timerRef.current);
        };
    }, [queryClient, run, step, stepMin, stepMax, game, radius]);
}

async function sweep(
    run: string,
    center: number,
    stepMin: number,
    stepMax: number,
    game: number | undefined,
    radius: number,
    queryClient: ReturnType<typeof useQueryClient>,
    signal: AbortSignal,
): Promise<void> {
    // Build the list of steps to fetch in outward-spreading order,
    // skipping any already in cache
    const targets: number[] = [];
    for (let offset = 1; offset <= radius; offset++) {
        const below = center - offset;
        const above = center + offset;
        if (below >= stepMin) {
            const cached = queryClient.getQueryData(["step", run, below, game]);
            if (!cached) targets.push(below);
        }
        if (above <= stepMax) {
            const cached = queryClient.getQueryData(["step", run, above, game]);
            if (!cached) targets.push(above);
        }
    }

    // Process in small batches, checking abort between each
    for (let i = 0; i < targets.length; i += BATCH_SIZE) {
        if (signal.aborted) return;

        const batch = targets.slice(i, i + BATCH_SIZE);
        if (batch.length === 0) continue;

        try {
            const results = await fetchStepsBatch(run, batch, game, signal);
            // Populate the TanStack Query cache with each step
            for (const [stepStr, data] of Object.entries(results)) {
                const stepNum = Number(stepStr);
                queryClient.setQueryData(
                    ["step", run, stepNum, game],
                    data,
                );
            }
        } catch (err: unknown) {
            // AbortError is expected when user navigates -- exit silently
            if (err instanceof DOMException && err.name === "AbortError") return;
            // Other network errors -- stop the sweep
            return;
        }
    }
}
