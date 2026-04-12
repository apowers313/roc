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
        expect(result.current.autoFollow).toBe(true);
        expect(result.current.speed).toBe(200);
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

    // Regression: a URL like ?step=2500 against a 314-step game would set
    // step=2500 in state. The slider clamps display but the state stays
    // out of range, so the StatusBar shows "Step 2500 | Game 0" with no
    // data. setStepRange must clamp the current step into the new range.
    it("setStepRange clamps current step down to the new max", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });

        act(() => result.current.setStep(2500));
        expect(result.current.step).toBe(2500);

        act(() => result.current.setStepRange(1, 314));
        expect(result.current.step).toBe(314);
        expect(result.current.stepMax).toBe(314);
    });

    it("setStepRange clamps current step up to the new min", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });

        act(() => result.current.setStep(5));
        act(() => result.current.setStepRange(50, 100));
        expect(result.current.step).toBe(50);
    });

    it("setStepRange leaves an in-range step unchanged", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });

        act(() => result.current.setStep(75));
        act(() => result.current.setStepRange(1, 100));
        expect(result.current.step).toBe(75);
    });

    // Skip the clamp when max <= 0 (no data yet) so we don't clobber an
    // in-progress URL navigation while the range query is still loading.
    it("setStepRange does not clamp when max is 0", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });

        act(() => result.current.setStep(2500));
        act(() => result.current.setStepRange(0, 0));
        expect(result.current.step).toBe(2500);
    });

    it("toggles playing", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });

        act(() => result.current.setPlaying(true));
        expect(result.current.playing).toBe(true);
        act(() => result.current.setPlaying(false));
        expect(result.current.playing).toBe(false);
    });

    it("toggles autoFollow independently of playing", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });

        act(() => result.current.setAutoFollow(false));
        expect(result.current.autoFollow).toBe(false);
        expect(result.current.playing).toBe(false);

        act(() => result.current.setPlaying(true));
        expect(result.current.autoFollow).toBe(false);
        expect(result.current.playing).toBe(true);

        act(() => result.current.setAutoFollow(true));
        expect(result.current.autoFollow).toBe(true);
        expect(result.current.playing).toBe(true);
    });

    it("throws when used outside provider", () => {
        expect(() => {
            renderHook(() => useDashboard());
        }).toThrow("useDashboard must be used within DashboardProvider");
    });
});
