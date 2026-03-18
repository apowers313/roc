import { describe, expect, it } from "vitest";

import { playbackReducer, type PlaybackState } from "./playback";

describe("playbackReducer", () => {
    describe("historical state", () => {
        const state: PlaybackState = "historical";

        it("transitions to live_following on GO_LIVE", () => {
            expect(playbackReducer(state, { type: "GO_LIVE" })).toBe(
                "live_following",
            );
        });

        it("stays historical on TOGGLE_PLAY", () => {
            expect(playbackReducer(state, { type: "TOGGLE_PLAY" })).toBe(
                "historical",
            );
        });

        it("stays historical on USER_NAVIGATE", () => {
            expect(playbackReducer(state, { type: "USER_NAVIGATE" })).toBe(
                "historical",
            );
        });

        it("stays historical on JUMP_TO_END", () => {
            expect(playbackReducer(state, { type: "JUMP_TO_END" })).toBe(
                "historical",
            );
        });

        it("stays historical on unknown actions", () => {
            expect(playbackReducer(state, { type: "PAUSE" })).toBe(
                "historical",
            );
            expect(playbackReducer(state, { type: "RESUME" })).toBe(
                "historical",
            );
        });
    });

    describe("live_following state", () => {
        const state: PlaybackState = "live_following";

        it("transitions to live_paused on PAUSE", () => {
            expect(playbackReducer(state, { type: "PAUSE" })).toBe(
                "live_paused",
            );
        });

        it("transitions to live_paused on USER_NAVIGATE", () => {
            expect(playbackReducer(state, { type: "USER_NAVIGATE" })).toBe(
                "live_paused",
            );
        });

        it("stays following on PUSH_ARRIVED", () => {
            expect(
                playbackReducer(state, {
                    type: "PUSH_ARRIVED",
                    atEdge: true,
                }),
            ).toBe("live_following");
        });

        it("transitions to live_following on GO_LIVE", () => {
            expect(playbackReducer(state, { type: "GO_LIVE" })).toBe(
                "live_following",
            );
        });
    });

    describe("live_paused state", () => {
        const state: PlaybackState = "live_paused";

        it("transitions to live_catchup on RESUME", () => {
            expect(playbackReducer(state, { type: "RESUME" })).toBe(
                "live_catchup",
            );
        });

        it("transitions to live_following on GO_LIVE", () => {
            expect(playbackReducer(state, { type: "GO_LIVE" })).toBe(
                "live_following",
            );
        });

        it("stays paused on PUSH_ARRIVED", () => {
            expect(
                playbackReducer(state, {
                    type: "PUSH_ARRIVED",
                    atEdge: false,
                }),
            ).toBe("live_paused");
        });

        it("stays paused on USER_NAVIGATE", () => {
            expect(playbackReducer(state, { type: "USER_NAVIGATE" })).toBe(
                "live_paused",
            );
        });
    });

    describe("live_catchup state", () => {
        const state: PlaybackState = "live_catchup";

        it("transitions to live_paused on PAUSE", () => {
            expect(playbackReducer(state, { type: "PAUSE" })).toBe(
                "live_paused",
            );
        });

        it("transitions to live_paused on USER_NAVIGATE", () => {
            expect(playbackReducer(state, { type: "USER_NAVIGATE" })).toBe(
                "live_paused",
            );
        });

        it("transitions to live_following on GO_LIVE", () => {
            expect(playbackReducer(state, { type: "GO_LIVE" })).toBe(
                "live_following",
            );
        });

        it("transitions to live_following when PUSH_ARRIVED at edge", () => {
            expect(
                playbackReducer(state, {
                    type: "PUSH_ARRIVED",
                    atEdge: true,
                }),
            ).toBe("live_following");
        });

        it("stays catchup when PUSH_ARRIVED not at edge", () => {
            expect(
                playbackReducer(state, {
                    type: "PUSH_ARRIVED",
                    atEdge: false,
                }),
            ).toBe("live_catchup");
        });
    });
});
