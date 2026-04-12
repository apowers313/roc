/**
 * Socket.io hooks for the unified-run architecture.
 *
 * ``useRunSubscription`` subscribes to ``step_added`` invalidation
 * events for the current run and refetches via TanStack Query.
 *
 * ``useGameState`` listens for ``game_state_changed`` events so the
 * dashboard can auto-navigate to a freshly started game.
 */

import { useEffect, useState } from "react";
import { io, type Socket } from "socket.io-client";

import { useCacheInvalidation } from "./useCacheInvalidation";

interface StepAddedPayload {
    run: string;
    step: number;
}

export interface GameState {
    state: string;
    run_name: string | null;
    /** Last exit code from the game subprocess, or null if still running
     *  / never started / exited cleanly. Only meaningful when ``state``
     *  is "idle" after a previously-running game. */
    exit_code?: number | null;
    /** Last error message from the game subprocess, or null. Only
     *  meaningful when ``state`` is "idle" and the previous run
     *  crashed. */
    error?: string | null;
}

// Singleton Socket.io connection shared across all hook instances. The
// dashboard only ever talks to one server, so we share one socket.
let _socket: Socket | null = null;

function getSocket(): Socket {
    if (_socket == null) {
        _socket = io({
            path: "/socket.io",
            transports: ["polling", "websocket"],
        });
    }
    return _socket;
}

/** Reset the singleton socket. Test-only. */
export function __resetSocketForTesting(): void {
    if (_socket != null) {
        _socket.disconnect();
    }
    _socket = null;
}

/**
 * Subscribe to step_added invalidations for the given run.
 *
 * Emits ``subscribe_run`` on mount, ``unsubscribe_run`` on unmount or
 * run change. On every ``step_added`` for the matching run, the hook
 * invalidates the ``step-range`` query so the slider and any
 * downstream consumers (StatusBar, TransportBar) refetch the latest
 * range.
 */
export function useRunSubscription(run: string): void {
    const cache = useCacheInvalidation();

    useEffect(() => {
        if (!run) return;
        const socket = getSocket();
        socket.emit("subscribe_run", run);

        const onStepAdded = (payload: StepAddedPayload) => {
            // Filter to the run we're currently subscribed to. This
            // guards against the brief window where two subscriptions
            // overlap during a run switch.
            if (!payload || payload.run !== run) return;
            cache.invalidateStepRange(run);
        };

        socket.on("step_added", onStepAdded);
        return () => {
            socket.off("step_added", onStepAdded);
            socket.emit("unsubscribe_run", run);
        };
    }, [run, cache]);
}

/**
 * Listen for ``game_state_changed`` Socket.io events.
 *
 * Returns the latest game state (``{state, run_name}``) or ``null``
 * before the first event arrives. This replaces the deleted
 * ``useLiveUpdates`` polling hook -- game state changes are now
 * event-driven via Socket.io rather than polled from /api/live/status.
 *
 * On mount the hook also issues a one-shot fetch to ``/api/game/status``
 * so consumers (``goLive``, the auto-navigation effect) have a valid
 * starting value. Without this fetch, a page loaded DURING a running
 * game sits on ``null`` until the next state transition, which makes
 * the GO LIVE badge click a no-op (TC-GAME-004).
 */
export function useGameState(): GameState | null {
    const [state, setState] = useState<GameState | null>(null);
    const cache = useCacheInvalidation();

    useEffect(() => {
        const socket = getSocket();
        const onGameStateChanged = (payload: GameState) => {
            setState(payload);
            // Invalidate runs so the dropdown picks up new/stopped runs,
            // and step-range so tail_growing flips (otherwise the GO LIVE
            // badge persists after game stop with stale cached data).
            cache.invalidateRunList();
            cache.invalidateAllStepRanges();
        };

        socket.on("game_state_changed", onGameStateChanged);

        // One-shot initial fetch so goLive works on a cold page load
        // while a game is already running. Socket.io only emits on
        // state transitions, so without this, the first interaction
        // (L shortcut or GO LIVE click) reads a stale null and bails.
        // The REST endpoint returns ``{state, run_name, exit_code,
        // error}`` -- we keep all four so MenuBar can display crash
        // errors without its own local fetch (single source of truth
        // for game state).
        let cancelled = false;
        void fetch("/api/game/status")
            .then((r) => (r.ok ? r.json() : null))
            .then((data: unknown) => {
                if (cancelled) return;
                if (
                    data != null &&
                    typeof data === "object" &&
                    "state" in data &&
                    typeof (data as { state: unknown }).state === "string"
                ) {
                    const parsed = data as {
                        state: string;
                        run_name?: string | null;
                        exit_code?: number | null;
                        error?: string | null;
                    };
                    setState({
                        state: parsed.state,
                        run_name: parsed.run_name ?? null,
                        exit_code: parsed.exit_code ?? null,
                        error: parsed.error ?? null,
                    });
                }
            })
            .catch(() => {
                // Swallow -- the Socket.io handler is the fallback path.
            });

        return () => {
            cancelled = true;
            socket.off("game_state_changed", onGameStateChanged);
        };
    }, [cache]);

    return state;
}
