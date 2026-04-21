import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Mock socket.io-client BEFORE importing the hook so the hook picks
// up the mocked io() factory. The mock returns a single shared socket
// across all io() calls so the test can drive _emit() on it.
vi.mock("socket.io-client", () => {
    const listeners: Record<string, ((...args: unknown[]) => void)[]> = {};
    const mockSocket = {
        on: vi.fn((event: string, cb: (...args: unknown[]) => void) => {
            listeners[event] = listeners[event] ?? [];
            listeners[event].push(cb);
        }),
        off: vi.fn((event: string, cb: (...args: unknown[]) => void) => {
            const arr = listeners[event];
            if (!arr) return;
            const idx = arr.indexOf(cb);
            if (idx >= 0) arr.splice(idx, 1);
        }),
        emit: vi.fn(),
        disconnect: vi.fn(),
        _emit: (event: string, ...args: unknown[]) => {
            for (const cb of [...(listeners[event] ?? [])]) {
                cb(...args);
            }
        },
        _listeners: listeners,
    };
    return {
        io: vi.fn(() => mockSocket),
        __mockSocket: mockSocket,
    };
});

import { __resetSocketForTesting, useGameState, useRunSubscription, useSocketConnected } from "./useRunSubscription";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const { __mockSocket: mockSocket } = (await import("socket.io-client")) as any;

function makeWrapper(queryClient: QueryClient) {
    return function Wrapper({ children }: Readonly<{ children: ReactNode }>) {
        return (
            <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
        );
    };
}

beforeEach(() => {
    __resetSocketForTesting();
    mockSocket.emit.mockClear();
    mockSocket.on.mockClear();
    mockSocket.off.mockClear();
    for (const key of Object.keys(mockSocket._listeners)) {
        delete mockSocket._listeners[key];
    }
    // Default: stub fetch so useGameState's initial /api/game/status
    // call never resolves during tests that focus on the Socket.io
    // event path. Tests that want to exercise the initial-fetch path
    // override this stub locally.
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
});

afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
});

describe("useRunSubscription", () => {
    it("emits subscribe_run on mount", () => {
        const queryClient = new QueryClient();
        renderHook(() => useRunSubscription("run-1"), {
            wrapper: makeWrapper(queryClient),
        });
        expect(mockSocket.emit).toHaveBeenCalledWith("subscribe_run", "run-1");
    });

    it("emits unsubscribe_run on unmount", () => {
        const queryClient = new QueryClient();
        const { unmount } = renderHook(() => useRunSubscription("run-1"), {
            wrapper: makeWrapper(queryClient),
        });
        mockSocket.emit.mockClear();
        unmount();
        expect(mockSocket.emit).toHaveBeenCalledWith("unsubscribe_run", "run-1");
    });

    it("does nothing for an empty run name", () => {
        const queryClient = new QueryClient();
        renderHook(() => useRunSubscription(""), {
            wrapper: makeWrapper(queryClient),
        });
        expect(mockSocket.emit).not.toHaveBeenCalled();
    });

    it("invalidates step-range query on step_added for the same run", () => {
        const queryClient = new QueryClient();
        const spy = vi.spyOn(queryClient, "invalidateQueries");
        renderHook(() => useRunSubscription("run-1"), {
            wrapper: makeWrapper(queryClient),
        });
        mockSocket._emit("step_added", { run: "run-1", step: 5 });
        expect(spy).toHaveBeenCalledWith({ queryKey: ["step-range", "run-1"] });
    });

    it("ignores step_added events for a different run", () => {
        const queryClient = new QueryClient();
        const spy = vi.spyOn(queryClient, "invalidateQueries");
        renderHook(() => useRunSubscription("run-1"), {
            wrapper: makeWrapper(queryClient),
        });
        mockSocket._emit("step_added", { run: "run-2", step: 5 });
        expect(spy).not.toHaveBeenCalled();
    });

    it("ignores malformed step_added payloads", () => {
        const queryClient = new QueryClient();
        const spy = vi.spyOn(queryClient, "invalidateQueries");
        renderHook(() => useRunSubscription("run-1"), {
            wrapper: makeWrapper(queryClient),
        });
        mockSocket._emit("step_added", null);
        expect(spy).not.toHaveBeenCalled();
    });

    it("re-subscribes when the run prop changes", () => {
        const queryClient = new QueryClient();
        const { rerender } = renderHook(({ run }) => useRunSubscription(run), {
            wrapper: makeWrapper(queryClient),
            initialProps: { run: "run-A" },
        });
        expect(mockSocket.emit).toHaveBeenCalledWith("subscribe_run", "run-A");
        mockSocket.emit.mockClear();
        rerender({ run: "run-B" });
        expect(mockSocket.emit).toHaveBeenCalledWith("unsubscribe_run", "run-A");
        expect(mockSocket.emit).toHaveBeenCalledWith("subscribe_run", "run-B");
    });
});

describe("useGameState", () => {
    it("returns null synchronously before the initial fetch resolves", () => {
        // beforeEach stubs fetch with a never-resolving promise.
        const queryClient = new QueryClient();
        const { result } = renderHook(() => useGameState(), {
            wrapper: makeWrapper(queryClient),
        });
        // Initial render returns null because the fetch is still pending.
        // Event-driven updates populate it later (see subsequent tests).
        expect(result.current).toBeNull();
    });

    // TC-GAME-004 regression: when the page loads during a running
    // game, Socket.io does NOT emit a fresh game_state_changed event,
    // so the hook must populate its initial value via a one-shot
    // fetch to /api/game/status. Without this, goLive reads null and
    // the GO LIVE badge click silently fails.
    it("fetches /api/game/status on mount and populates state", async () => {
        const fetchMock = vi.fn().mockResolvedValue({
            ok: true,
            json: () =>
                Promise.resolve({
                    state: "running",
                    run_name: "cold-load-run",
                }),
        });
        vi.stubGlobal("fetch", fetchMock);

        const queryClient = new QueryClient();
        const { result } = renderHook(() => useGameState(), {
            wrapper: makeWrapper(queryClient),
        });

        await vi.waitFor(() => {
            expect(result.current).toEqual({
                state: "running",
                run_name: "cold-load-run",
                exit_code: null,
                error: null,
            });
        });
        expect(fetchMock).toHaveBeenCalledWith("/api/game/status");
    });

    // Consolidation invariant: the hook must carry exit_code and error
    // so that MenuBar (the only UI consumer of crash details) does not
    // need to maintain its own parallel fetch state. If this test
    // regresses, expect the game menu to silently drop error messages
    // across state transitions.
    it("preserves exit_code and error from the initial fetch", async () => {
        const fetchMock = vi.fn().mockResolvedValue({
            ok: true,
            json: () =>
                Promise.resolve({
                    state: "idle",
                    run_name: "crashed-run",
                    exit_code: 1,
                    error: "Game subprocess crashed",
                }),
        });
        vi.stubGlobal("fetch", fetchMock);

        const queryClient = new QueryClient();
        const { result } = renderHook(() => useGameState(), {
            wrapper: makeWrapper(queryClient),
        });

        await vi.waitFor(() => {
            expect(result.current).toEqual({
                state: "idle",
                run_name: "crashed-run",
                exit_code: 1,
                error: "Game subprocess crashed",
            });
        });
    });

    it("retries /api/game/status and succeeds on second attempt", async () => {
        const fetchMock = vi.fn()
            .mockRejectedValueOnce(new Error("network down"))
            .mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ state: "running", run_name: "retry-run" }),
            });
        vi.stubGlobal("fetch", fetchMock);

        const queryClient = new QueryClient();
        const { result } = renderHook(() => useGameState(), {
            wrapper: makeWrapper(queryClient),
        });

        await vi.waitFor(() => {
            expect(result.current).toEqual({
                state: "running",
                run_name: "retry-run",
                exit_code: null,
                error: null,
            });
        });
        expect(fetchMock).toHaveBeenCalledTimes(2);
    });

    it("stays null after all retry attempts fail", async () => {
        vi.useFakeTimers();
        const fetchMock = vi.fn().mockRejectedValue(new Error("network down"));
        vi.stubGlobal("fetch", fetchMock);

        const queryClient = new QueryClient();
        const { result } = renderHook(() => useGameState(), {
            wrapper: makeWrapper(queryClient),
        });

        // Drain initial fetch + retries by advancing timers
        for (let i = 0; i < 5; i++) {
            await vi.advanceTimersByTimeAsync(1000);
        }
        expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(3);
        expect(result.current).toBeNull();
        vi.useRealTimers();
    });

    it("returns game state after game_state_changed event", () => {
        const queryClient = new QueryClient();
        const { result } = renderHook(() => useGameState(), {
            wrapper: makeWrapper(queryClient),
        });
        act(() => {
            mockSocket._emit("game_state_changed", {
                state: "running",
                run_name: "live-run",
            });
        });
        expect(result.current).toEqual({
            state: "running",
            run_name: "live-run",
        });
    });

    it("invalidates runs queries on game_state_changed", () => {
        const queryClient = new QueryClient();
        const spy = vi.spyOn(queryClient, "invalidateQueries");
        renderHook(() => useGameState(), {
            wrapper: makeWrapper(queryClient),
        });
        mockSocket._emit("game_state_changed", {
            state: "running",
            run_name: "live-run",
        });
        expect(spy).toHaveBeenCalledWith({ queryKey: ["runs"] });
    });

    it("invalidates step-range queries on game_state_changed so tail_growing updates", () => {
        const queryClient = new QueryClient();
        const spy = vi.spyOn(queryClient, "invalidateQueries");
        renderHook(() => useGameState(), {
            wrapper: makeWrapper(queryClient),
        });
        mockSocket._emit("game_state_changed", {
            state: "idle",
            run_name: null,
        });
        // Bug regression: game stop must invalidate step-range queries so
        // the cached tail_growing flips to false and the GO LIVE badge
        // disappears.
        expect(spy).toHaveBeenCalledWith({ queryKey: ["step-range"] });
    });

    it("cleans up listener on unmount", () => {
        const queryClient = new QueryClient();
        const { unmount } = renderHook(() => useGameState(), {
            wrapper: makeWrapper(queryClient),
        });
        unmount();
        expect(mockSocket.off).toHaveBeenCalledWith(
            "game_state_changed",
            expect.any(Function),
        );
    });
});

describe("useSocketConnected", () => {
    it("starts as false before socket connects", () => {
        const queryClient = new QueryClient();
        const { result } = renderHook(() => useSocketConnected(), {
            wrapper: makeWrapper(queryClient),
        });
        expect(result.current).toBe(false);
    });

    it("becomes true on connect event", () => {
        const queryClient = new QueryClient();
        const { result } = renderHook(() => useSocketConnected(), {
            wrapper: makeWrapper(queryClient),
        });
        act(() => {
            mockSocket._emit("connect");
        });
        expect(result.current).toBe(true);
    });

    it("becomes false on disconnect event", () => {
        const queryClient = new QueryClient();
        const { result } = renderHook(() => useSocketConnected(), {
            wrapper: makeWrapper(queryClient),
        });
        act(() => {
            mockSocket._emit("connect");
        });
        expect(result.current).toBe(true);
        act(() => {
            mockSocket._emit("disconnect", "transport close");
        });
        expect(result.current).toBe(false);
    });

    it("becomes false on connect_error event", () => {
        const queryClient = new QueryClient();
        const { result } = renderHook(() => useSocketConnected(), {
            wrapper: makeWrapper(queryClient),
        });
        act(() => {
            mockSocket._emit("connect_error", { message: "timeout" });
        });
        expect(result.current).toBe(false);
    });

    it("invalidates caches on reconnect", () => {
        const queryClient = new QueryClient();
        const spy = vi.spyOn(queryClient, "invalidateQueries");
        // useRunSubscription sets _reconnectCache
        renderHook(() => useRunSubscription("run-1"), {
            wrapper: makeWrapper(queryClient),
        });
        renderHook(() => useSocketConnected(), {
            wrapper: makeWrapper(queryClient),
        });
        spy.mockClear();
        act(() => {
            mockSocket._emit("connect");
        });
        expect(spy).toHaveBeenCalledWith({ queryKey: ["step-range"] });
        expect(spy).toHaveBeenCalledWith({ queryKey: ["runs"] });
    });
});
