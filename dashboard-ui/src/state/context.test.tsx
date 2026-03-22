import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ReactNode } from "react";
import { act } from "react";

import { DashboardProvider, useDashboard } from "./context";

function wrapper({ children }: { children: ReactNode }) {
    return <DashboardProvider>{children}</DashboardProvider>;
}

describe("DashboardProvider / useDashboard", () => {
    it("provides default values", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });

        expect(result.current.run).toBe("");
        expect(result.current.game).toBe(1);
        expect(result.current.step).toBe(1);
        expect(result.current.stepMin).toBe(1);
        expect(result.current.stepMax).toBe(1);
        expect(result.current.playing).toBe(false);
        expect(result.current.speed).toBe(200);
        expect(result.current.liveRunName).toBe("");
        expect(result.current.liveGameActive).toBe(false);
        expect(result.current.playback).toBe("historical");
    });

    it("updates run", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });

        act(() => result.current.setRun("test-run"));
        expect(result.current.run).toBe("test-run");
    });

    it("updates step with a number", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });

        act(() => result.current.setStep(42));
        expect(result.current.step).toBe(42);
    });

    it("updates step with a function", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });

        act(() => result.current.setStep(10));
        act(() => result.current.setStep((prev) => prev + 5));
        expect(result.current.step).toBe(15);
    });

    it("updates step range", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });

        act(() => result.current.setStepRange(5, 100));
        expect(result.current.stepMin).toBe(5);
        expect(result.current.stepMax).toBe(100);
    });

    it("dispatches playback actions", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });

        act(() => result.current.dispatchPlayback({ type: "GO_LIVE" }));
        expect(result.current.playback).toBe("live_following");
    });

    it("throws when used outside provider", () => {
        expect(() => {
            renderHook(() => useDashboard());
        }).toThrow("useDashboard must be used within DashboardProvider");
    });
});
