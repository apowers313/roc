/** Bookmark management hook -- load, toggle, navigate, persist. */

import { useCallback, useEffect, useMemo, useState } from "react";

import { fetchBookmarks, saveBookmarks } from "../api/client";
import type { Bookmark } from "../types/api";

export interface UseBookmarksReturn {
    bookmarks: Bookmark[];
    bookmarkSteps: number[];
    isBookmarked: (step: number) => boolean;
    toggleBookmark: (step: number, game: number) => void;
    nextBookmark: (currentStep: number) => Bookmark | null;
    prevBookmark: (currentStep: number) => Bookmark | null;
    updateAnnotation: (step: number, annotation: string) => void;
    updateBookmark: (oldStep: number, updates: Partial<Pick<Bookmark, "step" | "game" | "annotation">>) => void;
}

export function useBookmarks(run: string): UseBookmarksReturn {
    const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);

    // Load bookmarks when run changes
    useEffect(() => {
        if (!run) return;
        let cancelled = false;
        fetchBookmarks(run)
            .then((data) => {
                if (!cancelled) setBookmarks(data);
            })
            .catch(() => {
                if (!cancelled) setBookmarks([]);
            });
        return () => { cancelled = true; };
    }, [run]);

    const persist = useCallback(
        (updated: Bookmark[]) => {
            if (run) {
                void saveBookmarks(run, updated);
            }
        },
        [run],
    );

    const toggleBookmark = useCallback(
        (step: number, game: number) => {
            setBookmarks((prev) => {
                const exists = prev.some((b) => b.step === step);
                const updated = exists
                    ? prev.filter((b) => b.step !== step)
                    : [...prev, {
                        step,
                        game,
                        annotation: "",
                        created: new Date().toISOString(),
                    }];
                persist(updated);
                return updated;
            });
        },
        [persist],
    );

    const isBookmarked = useCallback(
        (step: number) => bookmarks.some((b) => b.step === step),
        [bookmarks],
    );

    const bookmarkSteps = useMemo(
        () => bookmarks.map((b) => b.step).sort((a, b) => a - b),
        [bookmarks],
    );

    const sortedBookmarks = useMemo(
        () => [...bookmarks].sort((a, b) => a.step - b.step),
        [bookmarks],
    );

    const nextBookmark = useCallback(
        (currentStep: number): Bookmark | null => {
            return sortedBookmarks.find((b) => b.step > currentStep) ?? null;
        },
        [sortedBookmarks],
    );

    const prevBookmark = useCallback(
        (currentStep: number): Bookmark | null => {
            return [...sortedBookmarks].reverse().find((b) => b.step < currentStep) ?? null;
        },
        [sortedBookmarks],
    );

    const updateAnnotation = useCallback(
        (step: number, annotation: string) => {
            setBookmarks((prev) => {
                const updated = prev.map((b) =>
                    b.step === step ? { ...b, annotation } : b,
                );
                persist(updated);
                return updated;
            });
        },
        [persist],
    );

    const updateBookmark = useCallback(
        (oldStep: number, updates: Partial<Pick<Bookmark, "step" | "game" | "annotation">>) => {
            setBookmarks((prev) => {
                const updated = prev.map((b) =>
                    b.step === oldStep ? { ...b, ...updates } : b,
                );
                persist(updated);
                return updated;
            });
        },
        [persist],
    );

    return {
        bookmarks,
        bookmarkSteps,
        isBookmarked,
        toggleBookmark,
        nextBookmark,
        prevBookmark,
        updateAnnotation,
        updateBookmark,
    };
}
