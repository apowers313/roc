import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ReactNode } from "react";

import { HighlightProvider, useHighlight } from "./highlight";

function wrapper({ children }: { children: ReactNode }) {
    return <HighlightProvider>{children}</HighlightProvider>;
}

describe("useHighlight", () => {
    it("starts with empty points", () => {
        const { result } = renderHook(() => useHighlight(), { wrapper });
        expect(result.current.points).toEqual([]);
    });

    it("setPoints replaces all points", () => {
        const { result } = renderHook(() => useHighlight(), { wrapper });
        act(() => {
            result.current.setPoints([{ x: 1, y: 2 }, { x: 3, y: 4 }]);
        });
        expect(result.current.points).toHaveLength(2);
        expect(result.current.points[0]).toEqual({ x: 1, y: 2 });
    });

    it("togglePoint adds then removes", () => {
        const { result } = renderHook(() => useHighlight(), { wrapper });
        act(() => {
            result.current.togglePoint({ x: 5, y: 10 });
        });
        expect(result.current.points).toHaveLength(1);
        expect(result.current.points[0]).toEqual({ x: 5, y: 10 });

        // Toggle again removes it
        act(() => {
            result.current.togglePoint({ x: 5, y: 10 });
        });
        expect(result.current.points).toHaveLength(0);
    });

    it("clear removes all points", () => {
        const { result } = renderHook(() => useHighlight(), { wrapper });
        act(() => {
            result.current.setPoints([{ x: 1, y: 2 }, { x: 3, y: 4 }]);
        });
        expect(result.current.points).toHaveLength(2);
        act(() => {
            result.current.clear();
        });
        expect(result.current.points).toHaveLength(0);
    });

    it("togglePoint keeps other points", () => {
        const { result } = renderHook(() => useHighlight(), { wrapper });
        act(() => {
            result.current.setPoints([{ x: 1, y: 2 }, { x: 3, y: 4 }]);
        });
        act(() => {
            result.current.togglePoint({ x: 1, y: 2 });
        });
        expect(result.current.points).toHaveLength(1);
        expect(result.current.points[0]).toEqual({ x: 3, y: 4 });
    });
});
