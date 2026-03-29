import { act, renderHook, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ReactNode } from "react";

import { renderWithProviders } from "../test-utils";
import { useDashboard } from "./context";
import { HighlightProvider, useHighlight, findHighlightColor, highlightBg } from "./highlight";

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
            result.current.setPoints([
                { x: 1, y: 2, color: "#ff4444" },
                { x: 3, y: 4, color: "#4488ff" },
            ]);
        });
        expect(result.current.points).toHaveLength(2);
        expect(result.current.points[0]).toMatchObject({ x: 1, y: 2 });
    });

    it("togglePoint adds with auto-assigned color then removes", () => {
        const { result } = renderHook(() => useHighlight(), { wrapper });
        act(() => {
            result.current.togglePoint({ x: 5, y: 10 });
        });
        expect(result.current.points).toHaveLength(1);
        expect(result.current.points[0]).toMatchObject({ x: 5, y: 10 });
        // Color should be auto-assigned (first in palette)
        expect(result.current.points[0]!.color).toBe("#ff4444");

        // Toggle again removes it
        act(() => {
            result.current.togglePoint({ x: 5, y: 10 });
        });
        expect(result.current.points).toHaveLength(0);
    });

    it("assigns rotating colors from palette", () => {
        const { result } = renderHook(() => useHighlight(), { wrapper });
        act(() => {
            result.current.togglePoint({ x: 1, y: 1 });
        });
        act(() => {
            result.current.togglePoint({ x: 2, y: 2 });
        });
        act(() => {
            result.current.togglePoint({ x: 3, y: 3 });
        });
        expect(result.current.points[0]!.color).toBe("#ff4444"); // red
        expect(result.current.points[1]!.color).toBe("#4488ff"); // blue
        expect(result.current.points[2]!.color).toBe("#44cc44"); // green
    });

    it("clear removes all points", () => {
        const { result } = renderHook(() => useHighlight(), { wrapper });
        act(() => {
            result.current.setPoints([
                { x: 1, y: 2, color: "#ff4444" },
                { x: 3, y: 4, color: "#4488ff" },
            ]);
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
            result.current.setPoints([
                { x: 1, y: 2, color: "#ff4444" },
                { x: 3, y: 4, color: "#4488ff" },
            ]);
        });
        act(() => {
            result.current.togglePoint({ x: 1, y: 2 });
        });
        expect(result.current.points).toHaveLength(1);
        expect(result.current.points[0]).toMatchObject({ x: 3, y: 4 });
    });

    it("preserves source field on toggled points", () => {
        const { result } = renderHook(() => useHighlight(), { wrapper });
        act(() => {
            result.current.togglePoint({ x: 5, y: 10, source: "attention" });
        });
        expect(result.current.points[0]).toMatchObject({ x: 5, y: 10, source: "attention" });
    });
});

describe("HighlightContext multi-color", () => {
    it("assigns distinct colors from palette", () => {
        const { result } = renderHook(() => useHighlight(), { wrapper });
        act(() => {
            result.current.togglePoint({ x: 1, y: 1 });
        });
        act(() => {
            result.current.togglePoint({ x: 2, y: 2 });
        });
        act(() => {
            result.current.togglePoint({ x: 3, y: 3 });
        });
        const colors = result.current.points.map((p) => p.color);
        expect(new Set(colors).size).toBe(3);
    });

    it("toggles off on second click", () => {
        const { result } = renderHook(() => useHighlight(), { wrapper });
        act(() => {
            result.current.togglePoint({ x: 5, y: 5 });
        });
        expect(result.current.points).toHaveLength(1);
        act(() => {
            result.current.togglePoint({ x: 5, y: 5 });
        });
        expect(result.current.points).toHaveLength(0);
    });

    it("clears on step change", () => {
        function TestComponent() {
            const { setStep } = useDashboard();
            const { points, togglePoint } = useHighlight();
            return (
                <div>
                    <button onClick={() => togglePoint({ x: 1, y: 1 })}>add</button>
                    <button onClick={() => setStep(99)}>go</button>
                    <span data-testid="count">{points.length}</span>
                </div>
            );
        }
        renderWithProviders(<TestComponent />);
        fireEvent.click(screen.getByText("add"));
        expect(screen.getByTestId("count")).toHaveTextContent("1");
        fireEvent.click(screen.getByText("go"));
        expect(screen.getByTestId("count")).toHaveTextContent("0");
    });
});

describe("findHighlightColor", () => {
    it("returns color for matching point", () => {
        const pts = [
            { x: 1, y: 2, color: "#ff4444" },
            { x: 3, y: 4, color: "#4488ff" },
        ];
        expect(findHighlightColor(pts, 3, 4)).toBe("#4488ff");
    });

    it("returns undefined for non-matching point", () => {
        const pts = [{ x: 1, y: 2, color: "#ff4444" }];
        expect(findHighlightColor(pts, 9, 9)).toBeUndefined();
    });
});

describe("highlightBg", () => {
    it("converts hex to rgba with 0.15 alpha", () => {
        expect(highlightBg("#ff4444")).toBe("rgba(255, 68, 68, 0.15)");
    });

    it("returns undefined for undefined input", () => {
        expect(highlightBg(undefined)).toBeUndefined();
    });
});
