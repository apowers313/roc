/**
 * Additional coverage tests for the API client -- covers functions that were
 * previously untested: fetchIntrinsicsHistory, fetchActionHistory,
 * fetchResolutionHistory, fetchAllObjects, and signal-passing in fetchStepsBatch.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
    fetchActionHistory,
    fetchAllObjects,
    fetchIntrinsicsHistory,
    fetchResolutionHistory,
    fetchStepsBatch,
} from "./client";

describe("API client (additional coverage)", () => {
    const mockFetch = vi.fn();

    beforeEach(() => {
        vi.stubGlobal("fetch", mockFetch);
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    function okJson(data: unknown) {
        return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(data),
        });
    }

    function errorResponse(status: number, statusText: string) {
        return Promise.resolve({ ok: false, status, statusText });
    }

    describe("fetchIntrinsicsHistory", () => {
        it("fetches intrinsics history without game filter", async () => {
            const data = [{ step: 1, raw: { hp: 10 }, normalized: { hp: 0.5 } }];
            mockFetch.mockReturnValue(okJson(data));

            const result = await fetchIntrinsicsHistory("run1");
            expect(result).toEqual(data);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/intrinsics-history",
            );
        });

        it("includes game parameter when provided", async () => {
            mockFetch.mockReturnValue(okJson([]));

            await fetchIntrinsicsHistory("run1", 3);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/intrinsics-history?game=3",
            );
        });

        it("throws on non-ok response", async () => {
            mockFetch.mockReturnValue(errorResponse(500, "Internal Server Error"));

            await expect(fetchIntrinsicsHistory("run1")).rejects.toThrow(
                "API error: 500 Internal Server Error",
            );
        });
    });

    describe("fetchActionHistory", () => {
        it("fetches action history without game filter", async () => {
            const data = [{ step: 1, action_id: 5, action_name: "move_north" }];
            mockFetch.mockReturnValue(okJson(data));

            const result = await fetchActionHistory("run1");
            expect(result).toEqual(data);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/action-history",
            );
        });

        it("includes game parameter when provided", async () => {
            mockFetch.mockReturnValue(okJson([]));

            await fetchActionHistory("run1", 2);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/action-history?game=2",
            );
        });
    });

    describe("fetchResolutionHistory", () => {
        it("fetches resolution history without game filter", async () => {
            const data = [{ step: 1, outcome: "match", correct: true }];
            mockFetch.mockReturnValue(okJson(data));

            const result = await fetchResolutionHistory("run1");
            expect(result).toEqual(data);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/resolution-history",
            );
        });

        it("includes game parameter when provided", async () => {
            mockFetch.mockReturnValue(okJson([]));

            await fetchResolutionHistory("run1", 4);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/resolution-history?game=4",
            );
        });
    });

    describe("fetchAllObjects", () => {
        it("fetches all objects without game filter", async () => {
            const data = [
                {
                    shape: "circle",
                    glyph: "@",
                    color: "white",
                    node_id: "n1",
                    step_added: 1,
                    match_count: 5,
                    feature_type: "glyph",
                },
            ];
            mockFetch.mockReturnValue(okJson(data));

            const result = await fetchAllObjects("run1");
            expect(result).toEqual(data);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/all-objects",
            );
        });

        it("includes game parameter when provided", async () => {
            mockFetch.mockReturnValue(okJson([]));

            await fetchAllObjects("run1", 2);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/all-objects?game=2",
            );
        });
    });

    describe("fetchStepsBatch with signal", () => {
        it("passes AbortSignal to fetch", async () => {
            const data = { "1": { step: 1 } };
            mockFetch.mockReturnValue(okJson(data));
            const controller = new AbortController();

            const result = await fetchStepsBatch("run1", [1], undefined, controller.signal);
            expect(result).toEqual(data);
            // When signal is provided, fetch should be called with the signal option
            expect(mockFetch).toHaveBeenCalledWith(
                expect.stringContaining("/api/runs/run1/steps?"),
                { signal: controller.signal },
            );
        });

        it("fetches without signal when not provided", async () => {
            const data = { "1": { step: 1 } };
            mockFetch.mockReturnValue(okJson(data));

            await fetchStepsBatch("run1", [1]);
            // Should be called with just the URL (no options)
            expect(mockFetch).toHaveBeenCalledWith(
                expect.stringContaining("/api/runs/run1/steps?"),
            );
        });
    });

    describe("fetchJson signal branch", () => {
        it("calls fetch with signal when provided via fetchStepsBatch", async () => {
            const data = {};
            mockFetch.mockReturnValue(okJson(data));
            const controller = new AbortController();

            await fetchStepsBatch("run1", [1, 2], 3, controller.signal);
            const callArgs = mockFetch.mock.calls[0];
            expect(callArgs).toHaveLength(2);
            expect(callArgs![1]).toEqual({ signal: controller.signal });
        });
    });
});
