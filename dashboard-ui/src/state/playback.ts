/**
 * Client-side playback state as two independent booleans.
 *
 * Phase 5: the previous four-state machine (`historical`, `live_following`,
 * `live_paused`, `live_catchup`) collapsed to two independent booleans:
 *
 *   - `playing`: is the auto-play timer advancing the step cursor?
 *   - `autoFollow`: should the step cursor jump to the new max as
 *     `useStepRange(run).data.tail_growing` pushes it forward?
 *
 * `autoFollow` is the liveness surface -- combined with
 * `useStepRange(run).data.tail_growing`, it covers every case the old
 * state machine enumerated:
 *
 *   - old "live_following"  ->  autoFollow=true,  tail_growing=true
 *   - old "live_paused"     ->  autoFollow=false, tail_growing=true
 *   - old "live_catchup"    ->  autoFollow=true,  tail_growing=true (step < max)
 *   - old "historical"      ->  tail_growing=false (autoFollow irrelevant)
 *
 * New live runs default to `autoFollow=true` so the dashboard tracks
 * the head by default. Explicit user navigation drops `autoFollow` to
 * `false`; clicking the GO LIVE badge sets it back to `true` and snaps
 * the step to the current max.
 */

export interface PlaybackState {
    playing: boolean;
    autoFollow: boolean;
}

export const initialPlayback: PlaybackState = {
    playing: false,
    autoFollow: true,
};
