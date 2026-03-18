import { renderHook, act, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock the API client
vi.mock("../api/client", () => ({
    fetchBookmarks: vi.fn(),
    saveBookmarks: vi.fn(),
}));

import { useBookmarks } from "./useBookmarks";
import { fetchBookmarks, saveBookmarks } from "../api/client";
import type { Bookmark } from "../types/api";

const mockFetchBookmarks = vi.mocked(fetchBookmarks);
const mockSaveBookmarks = vi.mocked(saveBookmarks);

describe("useBookmarks", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockFetchBookmarks.mockResolvedValue([]);
        mockSaveBookmarks.mockResolvedValue(undefined);
    });

    it("starts with empty bookmarks", () => {
        const { result } = renderHook(() => useBookmarks("test-run"));
        expect(result.current.bookmarks).toEqual([]);
    });

    it("loads bookmarks from API on mount", async () => {
        const saved: Bookmark[] = [
            { step: 10, game: 1, annotation: "interesting", created: "2026-01-01" },
        ];
        mockFetchBookmarks.mockResolvedValue(saved);

        const { result } = renderHook(() => useBookmarks("test-run"));

        await waitFor(() => {
            expect(result.current.bookmarks).toEqual(saved);
        });
        expect(mockFetchBookmarks).toHaveBeenCalledWith("test-run");
    });

    it("toggleBookmark adds a bookmark if not present", async () => {
        const { result } = renderHook(() => useBookmarks("test-run"));

        act(() => {
            result.current.toggleBookmark(5, 1);
        });

        expect(result.current.bookmarks).toHaveLength(1);
        expect(result.current.bookmarks[0]!.step).toBe(5);
        expect(result.current.bookmarks[0]!.game).toBe(1);
    });

    it("toggleBookmark removes a bookmark if already present", async () => {
        const { result } = renderHook(() => useBookmarks("test-run"));

        act(() => {
            result.current.toggleBookmark(5, 1);
        });
        expect(result.current.bookmarks).toHaveLength(1);

        act(() => {
            result.current.toggleBookmark(5, 1);
        });
        expect(result.current.bookmarks).toHaveLength(0);
    });

    it("isBookmarked returns true for bookmarked steps", () => {
        const { result } = renderHook(() => useBookmarks("test-run"));

        act(() => {
            result.current.toggleBookmark(5, 1);
        });

        expect(result.current.isBookmarked(5)).toBe(true);
        expect(result.current.isBookmarked(6)).toBe(false);
    });

    it("nextBookmark returns the next bookmarked step", () => {
        const { result } = renderHook(() => useBookmarks("test-run"));

        act(() => {
            result.current.toggleBookmark(5, 1);
            result.current.toggleBookmark(15, 1);
            result.current.toggleBookmark(25, 1);
        });

        expect(result.current.nextBookmark(1)).toEqual(expect.objectContaining({ step: 5 }));
        expect(result.current.nextBookmark(5)).toEqual(expect.objectContaining({ step: 15 }));
        expect(result.current.nextBookmark(10)).toEqual(expect.objectContaining({ step: 15 }));
        expect(result.current.nextBookmark(25)).toBeNull();
    });

    it("prevBookmark returns the previous bookmarked step", () => {
        const { result } = renderHook(() => useBookmarks("test-run"));

        act(() => {
            result.current.toggleBookmark(5, 1);
            result.current.toggleBookmark(15, 1);
            result.current.toggleBookmark(25, 1);
        });

        expect(result.current.prevBookmark(30)).toEqual(expect.objectContaining({ step: 25 }));
        expect(result.current.prevBookmark(25)).toEqual(expect.objectContaining({ step: 15 }));
        expect(result.current.prevBookmark(15)).toEqual(expect.objectContaining({ step: 5 }));
        expect(result.current.prevBookmark(5)).toBeNull();
    });

    it("saves bookmarks to API on toggle", async () => {
        const { result } = renderHook(() => useBookmarks("test-run"));

        act(() => {
            result.current.toggleBookmark(5, 1);
        });

        await waitFor(() => {
            expect(mockSaveBookmarks).toHaveBeenCalledWith(
                "test-run",
                expect.arrayContaining([expect.objectContaining({ step: 5 })]),
            );
        });
    });

    it("updateAnnotation changes the annotation text", () => {
        const { result } = renderHook(() => useBookmarks("test-run"));

        act(() => {
            result.current.toggleBookmark(5, 1);
        });

        act(() => {
            result.current.updateAnnotation(5, "new note");
        });

        expect(result.current.bookmarks[0]!.annotation).toBe("new note");
    });

    it("reloads bookmarks when run changes", async () => {
        const bookmarksRun1: Bookmark[] = [
            { step: 1, game: 1, annotation: "a", created: "2026-01-01" },
        ];
        const bookmarksRun2: Bookmark[] = [
            { step: 2, game: 1, annotation: "b", created: "2026-01-02" },
        ];

        mockFetchBookmarks
            .mockResolvedValueOnce(bookmarksRun1)
            .mockResolvedValueOnce(bookmarksRun2);

        const { result, rerender } = renderHook(
            ({ run }) => useBookmarks(run),
            { initialProps: { run: "run-1" } },
        );

        await waitFor(() => {
            expect(result.current.bookmarks).toEqual(bookmarksRun1);
        });

        rerender({ run: "run-2" });

        await waitFor(() => {
            expect(result.current.bookmarks).toEqual(bookmarksRun2);
        });
    });

    it("bookmarkSteps returns sorted list of bookmarked step numbers", () => {
        const { result } = renderHook(() => useBookmarks("test-run"));

        act(() => {
            result.current.toggleBookmark(25, 1);
            result.current.toggleBookmark(5, 1);
            result.current.toggleBookmark(15, 1);
        });

        expect(result.current.bookmarkSteps).toEqual([5, 15, 25]);
    });
});
