/**
 * Centralized TanStack Query cache invalidation for server-owned data.
 *
 * Every call site that invalidates a server-data query key should go
 * through this hook. Keeping the keys in one place means that if we
 * ever need to change how a key is shaped (e.g. add a ``gameKey``
 * dimension to step-range), we only have to update this file instead
 * of grepping for every ``invalidateQueries`` call in the codebase.
 *
 * Rule: production code does not call ``queryClient.invalidateQueries``
 * directly for any of the keys owned by ``api/queries.ts``. Use this
 * hook instead. Raw ``invalidateQueries`` is fine in tests and in
 * prefetch/snapshot helpers (``usePrefetchWindow``) that manipulate
 * cache contents rather than invalidating them.
 */

import { useQueryClient } from "@tanstack/react-query";
import { useMemo } from "react";

export interface CacheInvalidator {
    /**
     * Invalidate the step-range query for a specific run.
     *
     * Called on ``step_added`` Socket.io invalidations so TanStack
     * Query refetches the latest range for the currently-subscribed
     * run. Invalidates all game keys under the run -- the query key
     * shape is ``["step-range", run, game]`` but we rely on the
     * prefix match so any game's range is refreshed.
     */
    invalidateStepRange: (run: string) => void;

    /**
     * Invalidate all step-range queries (across all runs).
     *
     * Called on ``game_state_changed`` events because a game
     * start/stop transition flips ``tail_growing`` for whichever run
     * is involved, and we do not always know which run from the event
     * payload alone. Refreshing all step-range queries is cheap and
     * avoids stale GO LIVE badges after a game stop.
     */
    invalidateAllStepRanges: () => void;

    /**
     * Invalidate the run list query.
     *
     * Called when the set of runs on disk may have changed -- game
     * start (new run appears), game stop (run transitions from live
     * to complete). The dropdown picks up the new state on the next
     * render.
     */
    invalidateRunList: () => void;
}

export function useCacheInvalidation(): CacheInvalidator {
    const queryClient = useQueryClient();

    return useMemo<CacheInvalidator>(
        () => ({
            invalidateStepRange: (run: string) => {
                void queryClient.invalidateQueries({
                    queryKey: ["step-range", run],
                });
            },
            invalidateAllStepRanges: () => {
                void queryClient.invalidateQueries({
                    queryKey: ["step-range"],
                });
            },
            invalidateRunList: () => {
                void queryClient.invalidateQueries({ queryKey: ["runs"] });
            },
        }),
        [queryClient],
    );
}
