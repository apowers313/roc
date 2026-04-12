/**
 * Tests for the Phase 5 two-boolean playback model.
 *
 * The previous four-state machine (`historical`, `live_following`,
 * `live_paused`, `live_catchup`) collapsed to two independent booleans:
 *
 *   - `playing`:    is the auto-play timer advancing the cursor?
 *   - `autoFollow`: should the cursor chase the growing tail?
 *
 * Liveness is carried by `useStepRange().data.tail_growing`, NOT by
 * the playback state. Combined with `autoFollow`, the two enum cases
 * the UI actually distinguishes are:
 *
 *   LIVE     badge = tail_growing &&  autoFollow
 *   GO LIVE  badge = tail_growing && !autoFollow
 *
 * Behavior tests for the auto-follow effect itself (the App-level
 * effect that pulls `step` forward as `range.max` grows) live next
 * to the consumers -- App.live-mode.test.tsx for keyboard navigation
 * paths, StatusBar.test.tsx for badge derivation, and
 * TransportBar.live.test.tsx for transport controls. This file
 * covers the data-model invariants and the provider-level behavior.
 */

import { renderHook, act } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { describe, expect, it } from "vitest";

import { DashboardProvider, useDashboard } from "./context";
import { initialPlayback, type PlaybackState } from "./playback";

function wrapper({ children }: { children: ReactNode }) {
    return createElement(DashboardProvider, null, children);
}

describe("playback (boolean model)", () => {
    it("default state has playing=false and autoFollow=true", () => {
        expect(initialPlayback.playing).toBe(false);
        expect(initialPlayback.autoFollow).toBe(true);
    });

    it("PlaybackState shape has only playing and autoFollow", () => {
        const state: PlaybackState = {
            playing: false,
            autoFollow: true,
        };
        const keys = Object.keys(state).sort();
        expect(keys).toEqual(["autoFollow", "playing"]);
    });

    it("the four valid combinations are all representable", () => {
        const s1: PlaybackState = { playing: true, autoFollow: true };
        const s2: PlaybackState = { playing: true, autoFollow: false };
        const s3: PlaybackState = { playing: false, autoFollow: true };
        const s4: PlaybackState = { playing: false, autoFollow: false };
        expect(s1.playing).toBe(true);
        expect(s1.autoFollow).toBe(true);
        expect(s2.playing).toBe(true);
        expect(s2.autoFollow).toBe(false);
        expect(s3.playing).toBe(false);
        expect(s3.autoFollow).toBe(true);
        expect(s4.playing).toBe(false);
        expect(s4.autoFollow).toBe(false);
    });

    // -----------------------------------------------------------------
    // Provider-level behavior: exercise the actual state machine via
    // the DashboardContext (there is no separate `usePlayback` hook --
    // the two booleans live in the shared context).
    // -----------------------------------------------------------------

    it("autoFollow defaults to true on a new live run", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });
        expect(result.current.autoFollow).toBe(true);
        expect(result.current.playing).toBe(false);
    });

    it("explicit navigation drops autoFollow to false", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });
        expect(result.current.autoFollow).toBe(true);

        // Simulate the "user clicks the slider" / "user hits the arrow
        // key" path: transport controls call setAutoFollow(false) as
        // part of every explicit navigation action.
        act(() => {
            result.current.setStep(5);
            result.current.setAutoFollow(false);
        });
        expect(result.current.autoFollow).toBe(false);
        expect(result.current.step).toBe(5);
    });

    it("GO LIVE sets autoFollow=true and snaps to head", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });

        // Seed a live-style range and drop autoFollow (user navigated
        // away from the tail, producing the GO LIVE badge state).
        act(() => {
            result.current.setStepRange(1, 100);
            result.current.setStep(42);
            result.current.setAutoFollow(false);
        });
        expect(result.current.autoFollow).toBe(false);
        expect(result.current.step).toBe(42);

        // goLive: flip autoFollow on and snap to the current max.
        // This mirrors the goLive() callback in App.tsx.
        act(() => {
            result.current.setAutoFollow(true);
            result.current.setStep(result.current.stepMax);
        });
        expect(result.current.autoFollow).toBe(true);
        expect(result.current.step).toBe(100);
    });

    it("toggling playing does not touch autoFollow", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });
        expect(result.current.autoFollow).toBe(true);

        act(() => result.current.setPlaying(true));
        expect(result.current.playing).toBe(true);
        expect(result.current.autoFollow).toBe(true); // unchanged

        act(() => result.current.setPlaying(false));
        expect(result.current.playing).toBe(false);
        expect(result.current.autoFollow).toBe(true); // still unchanged
    });

    it("toggling autoFollow does not touch playing", () => {
        const { result } = renderHook(() => useDashboard(), { wrapper });

        act(() => result.current.setPlaying(true));
        expect(result.current.playing).toBe(true);

        act(() => result.current.setAutoFollow(false));
        expect(result.current.autoFollow).toBe(false);
        expect(result.current.playing).toBe(true); // unchanged

        act(() => result.current.setAutoFollow(true));
        expect(result.current.autoFollow).toBe(true);
        expect(result.current.playing).toBe(true); // still unchanged
    });

    it("reaches the 'live paused' equivalent (autoFollow=false while range grows)", () => {
        // Old state machine: `live_paused` = user-navigated away while
        // the game is still running. New model: autoFollow=false and
        // tail_growing=true (tail_growing lives on the step-range
        // query, not in this state).
        const { result } = renderHook(() => useDashboard(), { wrapper });

        act(() => {
            result.current.setStepRange(1, 50);
            result.current.setStep(20);
            result.current.setAutoFollow(false);
        });

        // Simulate the range growing as new steps arrive via Socket.io
        // invalidation; in the paused state, the cursor MUST NOT move.
        act(() => {
            result.current.setStepRange(1, 60);
        });
        expect(result.current.step).toBe(20);
        expect(result.current.autoFollow).toBe(false);
    });

    it("reaches the 'live following' equivalent (autoFollow=true while range grows)", () => {
        // Old state machine: `live_following` = auto-track the tail.
        // New model: autoFollow=true. The actual cursor advancement
        // happens in the App-level effect (exercised in App tests);
        // this test just proves the state model permits the pattern.
        const { result } = renderHook(() => useDashboard(), { wrapper });

        act(() => {
            result.current.setStepRange(1, 50);
            result.current.setStep(50);
            result.current.setAutoFollow(true);
        });

        // autoFollow stays on; the App effect advances the cursor.
        // Here we directly simulate the effect's action.
        act(() => {
            result.current.setStepRange(1, 60);
            result.current.setStep(60);
        });
        expect(result.current.step).toBe(60);
        expect(result.current.autoFollow).toBe(true);
    });
});
