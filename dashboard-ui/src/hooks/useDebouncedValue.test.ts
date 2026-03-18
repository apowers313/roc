import { renderHook, act } from "@testing-library/react";
import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";

import { useDebouncedValue } from "./useDebouncedValue";

describe("useDebouncedValue", () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it("returns the initial value immediately", () => {
        const { result } = renderHook(() => useDebouncedValue("hello", 300));
        expect(result.current).toBe("hello");
    });

    it("does not update before the delay", () => {
        const { result, rerender } = renderHook(
            ({ value, delay }) => useDebouncedValue(value, delay),
            { initialProps: { value: "a", delay: 300 } },
        );

        rerender({ value: "b", delay: 300 });
        expect(result.current).toBe("a");

        act(() => vi.advanceTimersByTime(200));
        expect(result.current).toBe("a");
    });

    it("updates after the delay", () => {
        const { result, rerender } = renderHook(
            ({ value, delay }) => useDebouncedValue(value, delay),
            { initialProps: { value: "a", delay: 300 } },
        );

        rerender({ value: "b", delay: 300 });
        act(() => vi.advanceTimersByTime(300));
        expect(result.current).toBe("b");
    });

    it("resets the timer on rapid changes", () => {
        const { result, rerender } = renderHook(
            ({ value, delay }) => useDebouncedValue(value, delay),
            { initialProps: { value: "a", delay: 300 } },
        );

        rerender({ value: "b", delay: 300 });
        act(() => vi.advanceTimersByTime(200));
        rerender({ value: "c", delay: 300 });
        act(() => vi.advanceTimersByTime(200));

        // 200ms after "c" change, should still show "a"
        expect(result.current).toBe("a");

        act(() => vi.advanceTimersByTime(100));
        // Now 300ms after "c" change, should show "c" (not "b")
        expect(result.current).toBe("c");
    });
});
