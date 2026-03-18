import { renderHook, act, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Mock socket.io-client before importing the hook
vi.mock("socket.io-client", () => {
    const listeners: Record<string, ((...args: unknown[]) => void)[]> = {};
    const mockSocket = {
        on: vi.fn((event: string, cb: (...args: unknown[]) => void) => {
            listeners[event] = listeners[event] ?? [];
            listeners[event].push(cb);
        }),
        disconnect: vi.fn(),
        _emit: (event: string, ...args: unknown[]) => {
            for (const cb of listeners[event] ?? []) {
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

import { useLiveUpdates } from "./useLiveUpdates";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const { __mockSocket: mockSocket } = await import("socket.io-client") as any;

describe("useLiveUpdates", () => {
    const mockFetch = vi.fn();

    beforeEach(() => {
        vi.stubGlobal("fetch", mockFetch);
        mockFetch.mockResolvedValue({
            ok: true,
            json: () =>
                Promise.resolve({
                    active: false,
                    run_name: null,
                    step: 0,
                    game_number: 0,
                    step_min: 0,
                    step_max: 0,
                    game_numbers: [],
                }),
        });
    });

    afterEach(() => {
        vi.restoreAllMocks();
        for (const key of Object.keys(mockSocket._listeners)) {
            delete mockSocket._listeners[key];
        }
    });

    it("returns connected=false initially", () => {
        const { result } = renderHook(() => useLiveUpdates());
        expect(result.current.connected).toBe(false);
    });

    it("sets connected=true on socket connect", () => {
        const { result } = renderHook(() => useLiveUpdates());
        act(() => mockSocket._emit("connect"));
        expect(result.current.connected).toBe(true);
    });

    it("sets connected=false on socket disconnect", () => {
        const { result } = renderHook(() => useLiveUpdates());
        act(() => mockSocket._emit("connect"));
        expect(result.current.connected).toBe(true);

        act(() => mockSocket._emit("disconnect"));
        expect(result.current.connected).toBe(false);
    });

    it("calls onNewStep when new_step event arrives", () => {
        const onNewStep = vi.fn();
        renderHook(() => useLiveUpdates({ onNewStep }));

        const stepData = { step: 5, game_number: 1 };
        act(() => mockSocket._emit("new_step", stepData));
        expect(onNewStep).toHaveBeenCalledWith(stepData);
    });

    it("polls live status on mount and updates liveStatus", async () => {
        const status = {
            active: true,
            run_name: "test-run",
            step: 10,
            game_number: 1,
            step_min: 1,
            step_max: 10,
            game_numbers: [1],
        };
        mockFetch.mockResolvedValue({
            ok: true,
            json: () => Promise.resolve(status),
        });

        const { result } = renderHook(() => useLiveUpdates());

        await waitFor(() => {
            expect(mockFetch).toHaveBeenCalledWith("/api/live/status");
            expect(result.current.liveStatus).toEqual(status);
        });
    });

    it("handles fetch errors gracefully", async () => {
        mockFetch.mockRejectedValue(new Error("network error"));

        const { result } = renderHook(() => useLiveUpdates());

        // Should not throw -- liveStatus stays null
        await waitFor(() => {
            expect(mockFetch).toHaveBeenCalled();
        });
        expect(result.current.liveStatus).toBeNull();
    });
});
