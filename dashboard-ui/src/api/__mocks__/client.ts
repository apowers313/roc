import { vi } from "vitest";

export const fetchBookmarks = vi.fn(() => Promise.resolve([]));
export const saveBookmarks = vi.fn(() => Promise.resolve(undefined));
export const fetchStepRange = vi.fn(() =>
    Promise.resolve({ min: 1, max: 1, tail_growing: false }),
);
