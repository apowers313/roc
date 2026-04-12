import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
    fetchActionHistory,
    fetchAllObjects,
    fetchBookmarks,
    fetchEventHistory,
    fetchGames,
    fetchGraphHistory,
    fetchIntrinsicsHistory,
    fetchMetricsHistory,
    fetchResolutionHistory,
    fetchRuns,
    fetchStep,
    fetchStepRange,
    fetchStepsBatch,
    saveBookmarks,
} from "./client";

function okJson(data: unknown) {
    return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(data),
    });
}

function errorResponse(status: number, statusText: string) {
    return Promise.resolve({ ok: false, status, statusText });
}

describe("API client", () => {
    const mockFetch = vi.fn();

    beforeEach(() => {
        vi.stubGlobal("fetch", mockFetch);
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    describe("fetchRuns", () => {
        it("fetches runs from /api/runs without include_all by default", async () => {
            const runs = [{ name: "run1", games: 2, steps: 100, status: "ok" }];
            mockFetch.mockReturnValue(okJson(runs));

            const result = await fetchRuns();
            expect(result).toEqual(runs);
            expect(mockFetch).toHaveBeenCalledWith("/api/runs?min_steps=10");
        });

        it("passes include_all=true when requested", async () => {
            mockFetch.mockReturnValue(okJson([]));
            await fetchRuns(true);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs?min_steps=10&include_all=true",
            );
        });

        it("throws on non-ok response", async () => {
            mockFetch.mockReturnValue(errorResponse(500, "Internal Server Error"));

            await expect(fetchRuns()).rejects.toThrow(
                "API error: 500 Internal Server Error",
            );
        });
    });

    describe("fetchGames", () => {
        it("fetches games for a run", async () => {
            const games = [{ game_number: 1, steps: 50, start_ts: null, end_ts: null }];
            mockFetch.mockReturnValue(okJson(games));

            const result = await fetchGames("run1");
            expect(result).toEqual(games);
            expect(mockFetch).toHaveBeenCalledWith("/api/runs/run1/games");
        });

        it("encodes run name in URL", async () => {
            mockFetch.mockReturnValue(okJson([]));
            await fetchGames("run with spaces");
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run%20with%20spaces/games",
            );
        });
    });

    describe("fetchStep", () => {
        it("fetches step data without game filter", async () => {
            const step = { step: 1, game_number: 1 };
            mockFetch.mockReturnValue(okJson(step));

            const result = await fetchStep("run1", 1);
            expect(result).toEqual(step);
            expect(mockFetch).toHaveBeenCalledWith("/api/runs/run1/step/1");
        });

        it("includes game parameter when provided", async () => {
            mockFetch.mockReturnValue(okJson({ step: 1 }));

            await fetchStep("run1", 5, 2);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/step/5?game=2",
            );
        });
    });

    describe("fetchStepRange", () => {
        it("fetches step range without game filter", async () => {
            mockFetch.mockReturnValue(okJson({ min: 1, max: 100 }));

            const result = await fetchStepRange("run1");
            expect(result).toEqual({ min: 1, max: 100 });
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/step-range",
            );
        });

        it("includes game parameter when provided", async () => {
            mockFetch.mockReturnValue(okJson({ min: 1, max: 50 }));

            await fetchStepRange("run1", 3);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/step-range?game=3",
            );
        });
    });

    describe("fetchBookmarks", () => {
        it("fetches bookmarks for a run", async () => {
            const bookmarks = [{ step: 1, game: 1, annotation: "test", created: "2025-01-01" }];
            mockFetch.mockReturnValue(okJson(bookmarks));

            const result = await fetchBookmarks("run1");
            expect(result).toEqual(bookmarks);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/bookmarks",
            );
        });
    });

    describe("fetchStepsBatch", () => {
        it("fetches multiple steps", async () => {
            const data = { "1": { step: 1 }, "2": { step: 2 } };
            mockFetch.mockReturnValue(okJson(data));

            const result = await fetchStepsBatch("run1", [1, 2]);
            expect(result).toEqual(data);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/steps?steps=1%2C2",
            );
        });

        it("includes game parameter", async () => {
            mockFetch.mockReturnValue(okJson({}));

            await fetchStepsBatch("run1", [1], 3);
            const url = mockFetch.mock.calls[0]![0] as string;
            expect(url).toContain("game=3");
        });
    });

    describe("fetchMetricsHistory", () => {
        it("fetches metrics history without filters", async () => {
            const data = [{ step: 1, hp: 10 }];
            mockFetch.mockReturnValue(okJson(data));

            const result = await fetchMetricsHistory("run1");
            expect(result).toEqual(data);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/metrics-history",
            );
        });

        it("includes game and fields parameters", async () => {
            mockFetch.mockReturnValue(okJson([]));

            await fetchMetricsHistory("run1", 2, ["hp", "score"]);
            const url = mockFetch.mock.calls[0]![0] as string;
            expect(url).toContain("game=2");
            expect(url).toContain("fields=hp%2Cscore");
        });
    });

    describe("fetchGraphHistory", () => {
        it("fetches graph history", async () => {
            const data = [{ step: 1, node_count: 10, node_max: 100, edge_count: 20, edge_max: 200 }];
            mockFetch.mockReturnValue(okJson(data));

            const result = await fetchGraphHistory("run1");
            expect(result).toEqual(data);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/graph-history",
            );
        });

        it("includes game parameter", async () => {
            mockFetch.mockReturnValue(okJson([]));

            await fetchGraphHistory("run1", 2);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/graph-history?game=2",
            );
        });
    });

    describe("fetchEventHistory", () => {
        it("fetches event history", async () => {
            const data = [{ step: 1, "roc.perception": 5 }];
            mockFetch.mockReturnValue(okJson(data));

            const result = await fetchEventHistory("run1");
            expect(result).toEqual(data);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/event-history",
            );
        });

        it("includes game parameter", async () => {
            mockFetch.mockReturnValue(okJson([]));

            await fetchEventHistory("run1", 1);
            expect(mockFetch).toHaveBeenCalledWith(
                "/api/runs/run1/event-history?game=1",
            );
        });
    });

    describe("saveBookmarks", () => {
        it("posts bookmarks with correct headers", async () => {
            mockFetch.mockReturnValue(Promise.resolve({ ok: true }));

            const bookmarks = [{ step: 1, game: 1, annotation: "test", created: "2025-01-01" }];
            await saveBookmarks("run1", bookmarks);

            expect(mockFetch).toHaveBeenCalledWith("/api/runs/run1/bookmarks", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(bookmarks),
            });
        });

        it("throws on non-ok response", async () => {
            mockFetch.mockReturnValue(errorResponse(400, "Bad Request"));

            await expect(saveBookmarks("run1", [])).rejects.toThrow(
                "API error: 400 Bad Request",
            );
        });
    });

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
            expect(mockFetch).toHaveBeenCalledWith(
                expect.stringContaining("/api/runs/run1/steps?"),
                { signal: controller.signal },
            );
        });

        it("fetches without signal when not provided", async () => {
            const data = { "1": { step: 1 } };
            mockFetch.mockReturnValue(okJson(data));

            await fetchStepsBatch("run1", [1]);
            expect(mockFetch).toHaveBeenCalledWith(
                expect.stringContaining("/api/runs/run1/steps?"),
            );
        });

        it("calls fetch with signal when game is provided", async () => {
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
