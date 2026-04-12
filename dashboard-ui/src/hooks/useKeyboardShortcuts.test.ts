import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock react-hotkeys-hook so we can capture registered handlers
type HotkeyCallback = (...args: unknown[]) => void;
const registeredHotkeys: Map<string, HotkeyCallback> = new Map();
vi.mock("react-hotkeys-hook", () => ({
    useHotkeys: vi.fn((keys: string, callback: HotkeyCallback) => {
        registeredHotkeys.set(keys, callback);
    }),
}));

import { useKeyboardShortcuts } from "./useKeyboardShortcuts";

function invoke(key: string) {
    const fn = registeredHotkeys.get(key);
    expect(fn).toBeDefined();
    fn!();
}

describe("useKeyboardShortcuts", () => {
    const handlers = {
        stepForward: vi.fn(),
        stepBack: vi.fn(),
        togglePlay: vi.fn(),
        jumpToStart: vi.fn(),
        jumpToEnd: vi.fn(),
        stepForward10: vi.fn(),
        stepBack10: vi.fn(),
        toggleHelp: vi.fn(),
        toggleBookmark: vi.fn(),
        nextBookmark: vi.fn(),
        prevBookmark: vi.fn(),
        goLive: vi.fn(),
        speedUp: vi.fn(),
        speedDown: vi.fn(),
        cycleGame: vi.fn(),
    };

    beforeEach(() => {
        vi.clearAllMocks();
        registeredHotkeys.clear();
    });

    it("registers Right arrow for step forward", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("right");
        expect(handlers.stepForward).toHaveBeenCalledOnce();
    });

    it("registers Left arrow for step back", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("left");
        expect(handlers.stepBack).toHaveBeenCalledOnce();
    });

    it("registers Space for play/pause toggle", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("space");
        expect(handlers.togglePlay).toHaveBeenCalledOnce();
    });

    it("registers Home for jump to start", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("home");
        expect(handlers.jumpToStart).toHaveBeenCalledOnce();
    });

    it("registers End for jump to end", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("end");
        expect(handlers.jumpToEnd).toHaveBeenCalledOnce();
    });

    it("registers Shift+Right for +10 steps", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("shift+right");
        expect(handlers.stepForward10).toHaveBeenCalledOnce();
    });

    it("registers Shift+Left for -10 steps", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("shift+left");
        expect(handlers.stepBack10).toHaveBeenCalledOnce();
    });

    it("registers ? for keyboard help toggle", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("shift+slash");
        expect(handlers.toggleHelp).toHaveBeenCalledOnce();
    });

    it("registers b for bookmark toggle", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("b");
        expect(handlers.toggleBookmark).toHaveBeenCalledOnce();
    });

    it("registers ] for next bookmark", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("]");
        expect(handlers.nextBookmark).toHaveBeenCalledOnce();
    });

    it("registers [ for previous bookmark", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("[");
        expect(handlers.prevBookmark).toHaveBeenCalledOnce();
    });

    it("registers Ctrl+Left as alternative for jump to start", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("ctrl+left");
        expect(handlers.jumpToStart).toHaveBeenCalledOnce();
    });

    it("registers Ctrl+Right as alternative for jump to end", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("ctrl+right");
        expect(handlers.jumpToEnd).toHaveBeenCalledOnce();
    });

    it("registers L for go live", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("l");
        expect(handlers.goLive).toHaveBeenCalledOnce();
    });

    it("registers Shift+= for speed up", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("shift+equal");
        expect(handlers.speedUp).toHaveBeenCalledOnce();
    });

    it("registers = for speed up", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("=");
        expect(handlers.speedUp).toHaveBeenCalledOnce();
    });

    it("registers - for speed down", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("minus");
        expect(handlers.speedDown).toHaveBeenCalledOnce();
    });

    it("registers G for cycle game", () => {
        renderHook(() => useKeyboardShortcuts(handlers));
        invoke("g");
        expect(handlers.cycleGame).toHaveBeenCalledOnce();
    });
});
