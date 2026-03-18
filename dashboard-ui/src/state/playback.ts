/** Client-side playback state machine implemented as a React useReducer. */

export type PlaybackState =
    | "historical"
    | "live_following"
    | "live_paused"
    | "live_catchup";

export type PlaybackAction =
    | { type: "GO_LIVE" }
    | { type: "PAUSE" }
    | { type: "RESUME" }
    | { type: "JUMP_TO_END" }
    | { type: "USER_NAVIGATE" }
    | { type: "PUSH_ARRIVED"; atEdge: boolean }
    | { type: "TOGGLE_PLAY" };

export function playbackReducer(
    state: PlaybackState,
    action: PlaybackAction,
): PlaybackState {
    // GO_LIVE transitions to live_following from any state
    if (action.type === "GO_LIVE") return "live_following";

    switch (state) {
        case "historical":
            switch (action.type) {
                case "TOGGLE_PLAY":
                case "USER_NAVIGATE":
                case "JUMP_TO_END":
                    return "historical";
                default:
                    return state;
            }
        case "live_following":
            switch (action.type) {
                case "PAUSE":
                case "USER_NAVIGATE":
                    return "live_paused";
                case "PUSH_ARRIVED":
                    return "live_following";
                default:
                    return state;
            }
        case "live_paused":
            switch (action.type) {
                case "RESUME":
                    return "live_catchup";
                case "PUSH_ARRIVED":
                case "USER_NAVIGATE":
                    return "live_paused";
                default:
                    return state;
            }
        case "live_catchup":
            switch (action.type) {
                case "PAUSE":
                case "USER_NAVIGATE":
                    return "live_paused";
                case "PUSH_ARRIVED":
                    return action.atEdge ? "live_following" : "live_catchup";
                default:
                    return state;
            }
    }
}
