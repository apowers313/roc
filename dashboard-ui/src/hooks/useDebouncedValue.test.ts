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

    it("immediately adopts value when resetKey changes", () => {
        // Regression: switching runs must not let the old debounced step
        // leak into a request against the new run's API endpoint.
        const { result, rerender } = renderHook(
            ({ value, delay, resetKey }) =>
                useDebouncedValue(value, delay, resetKey),
            { initialProps: { value: 400, delay: 300, resetKey: "run-A" } },
        );

        expect(result.current).toBe(400);

        // Simulate run switch: both value and resetKey change in the same render
        rerender({ value: 1, delay: 300, resetKey: "run-B" });

        // Value must adopt immediately -- no 300ms lag
        expect(result.current).toBe(1);
    });

    it("does not immediately adopt when resetKey is unchanged", () => {
        const { result, rerender } = renderHook(
            ({ value, delay, resetKey }) =>
                useDebouncedValue(value, delay, resetKey),
            { initialProps: { value: 1, delay: 300, resetKey: "run-A" } },
        );

        // Normal step scrub within the same run -- should still debounce
        rerender({ value: 50, delay: 300, resetKey: "run-A" });
        expect(result.current).toBe(1);

        act(() => vi.advanceTimersByTime(300));
        expect(result.current).toBe(50);
    });

    it("works without resetKey (backwards compatible)", () => {
        const { result, rerender } = renderHook(
            ({ value, delay }) => useDebouncedValue(value, delay),
            { initialProps: { value: "x", delay: 200 } },
        );

        rerender({ value: "y", delay: 200 });
        expect(result.current).toBe("x");

        act(() => vi.advanceTimersByTime(200));
        expect(result.current).toBe("y");
    });
});
