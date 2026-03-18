import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
    fetchBookmarks,
    fetchGames,
    fetchRuns,
    fetchStep,
    fetchStepRange,
    saveBookmarks,
} from "./client";

describe("API client", () => {
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

    describe("fetchRuns", () => {
        it("fetches runs from /api/runs", async () => {
            const runs = [{ name: "run1", games: 2, steps: 100 }];
            mockFetch.mockReturnValue(okJson(runs));

            const result = await fetchRuns();
            expect(result).toEqual(runs);
            expect(mockFetch).toHaveBeenCalledWith("/api/runs");
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
});
