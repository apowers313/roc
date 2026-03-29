import { vi } from "vitest";

export const fetchBookmarks = vi.fn(() => Promise.resolve([]));
export const saveBookmarks = vi.fn(() => Promise.resolve(undefined));
