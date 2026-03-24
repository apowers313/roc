import { renderHook, act } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

import { useRatchetHeight } from "./useRatchetHeight";

// ---------------------------------------------------------------------------
// Mock useDashboard -- we only need run and game
// ---------------------------------------------------------------------------

let mockRun = "run-1";
let mockGame = 1;

vi.mock("../state/context", () => ({
    useDashboard: () => ({ run: mockRun, game: mockGame }),
}));

// ---------------------------------------------------------------------------
// Mock ResizeObserver -- capture callback so tests can fire it manually
// ---------------------------------------------------------------------------

type ROCallback = (entries: Array<{ contentRect: { height: number } }>) => void;
let roCallback: ROCallback | null = null;
const mockDisconnect = vi.fn();
const mockObserve = vi.fn();

class MockResizeObserver {
    constructor(cb: ROCallback) {
        roCallback = cb;
    }
    observe() { mockObserve(); }
    unobserve() {}
    disconnect() { mockDisconnect(); }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mockElement = document.createElement("div");

function fireResize(height: number) {
    act(() => {
        roCallback?.([{ contentRect: { height } }]);
    });
}

function mountAndAttach() {
    const hook = renderHook(() => useRatchetHeight());
    // Attach the callback ref to a mock element, triggering observer setup
    act(() => {
        hook.result.current.contentRef(mockElement);
    });
    return hook;
}

describe("useRatchetHeight", () => {
    beforeEach(() => {
        mockRun = "run-1";
        mockGame = 1;
        roCallback = null;
        mockDisconnect.mockClear();
        mockObserve.mockClear();
        vi.stubGlobal("ResizeObserver", MockResizeObserver);
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it("returns initial minHeight of 0", () => {
        const { result } = renderHook(() => useRatchetHeight());
        expect(result.current.minHeight).toBe(0);
        expect(result.current.contentRef).toBeDefined();
    });

    it("sets up ResizeObserver when ref is attached", () => {
        mountAndAttach();
        expect(mockObserve).toHaveBeenCalled();
    });

    it("ratchets minHeight upward", () => {
        const { result } = mountAndAttach();

        fireResize(100);
        expect(result.current.minHeight).toBe(100);

        fireResize(200);
        expect(result.current.minHeight).toBe(200);
    });

    it("does not shrink when content gets smaller", () => {
        const { result } = mountAndAttach();

        fireResize(200);
        expect(result.current.minHeight).toBe(200);

        fireResize(150);
        expect(result.current.minHeight).toBe(200);

        fireResize(50);
        expect(result.current.minHeight).toBe(200);
    });

    it("resets when panel collapses (height < 1)", () => {
        const { result } = mountAndAttach();

        fireResize(200);
        expect(result.current.minHeight).toBe(200);

        // Simulate accordion collapse
        fireResize(0);
        expect(result.current.minHeight).toBe(0);
    });

    it("re-ratchets after collapse and reopen", () => {
        const { result } = mountAndAttach();

        fireResize(200);
        expect(result.current.minHeight).toBe(200);

        // Collapse
        fireResize(0);
        expect(result.current.minHeight).toBe(0);

        // Reopen with smaller content -- ratchets to new height
        fireResize(120);
        expect(result.current.minHeight).toBe(120);
    });

    it("resets when run changes", () => {
        const { result, rerender } = mountAndAttach();

        fireResize(200);
        expect(result.current.minHeight).toBe(200);

        // Change run
        mockRun = "run-2";
        rerender();
        expect(result.current.minHeight).toBe(0);
    });

    it("resets when game changes", () => {
        const { result, rerender } = mountAndAttach();

        fireResize(200);
        expect(result.current.minHeight).toBe(200);

        // Change game
        mockGame = 2;
        rerender();
        expect(result.current.minHeight).toBe(0);
    });

    it("handles fractional heights near zero as collapse", () => {
        const { result } = mountAndAttach();

        fireResize(200);
        expect(result.current.minHeight).toBe(200);

        // Mantine animation may produce very small intermediate values
        fireResize(0.5);
        expect(result.current.minHeight).toBe(0);
    });

    it("disconnects ResizeObserver on unmount", () => {
        const { unmount } = mountAndAttach();
        unmount();
        expect(mockDisconnect).toHaveBeenCalled();
    });
});
