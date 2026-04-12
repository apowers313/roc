/** API response types for the FastAPI backend. */

/**
 * Status values mirror the Python backend (RunSummary.status):
 * - "ok": catalog readable, has at least min_steps steps
 * - "short": has data but below min_steps -- shown only with "Show all runs"
 * - "empty": catalog readable, no games/steps -- crashed before first step
 * - "corrupt": catalog open or query raised an exception
 * - "missing": directory vanished between scan and query
 *
 * We deliberately keep this loose (string) so the backend can add
 * new values without breaking the build. Unknown values render as
 * "unknown" in the UI.
 */
export type RunStatus =
    | "ok"
    | "short"
    | "empty"
    | "corrupt"
    | "missing";

export interface RunSummary {
    name: string;
    games: number;
    steps: number;
    /** Default "ok" for backwards compat with older servers. */
    status?: RunStatus;
    /** Diagnostic message for status="corrupt". */
    error?: string | null;
}

export interface GameSummary {
    game_number: number;
    steps: number;
    start_ts: number | null;
    end_ts: number | null;
}

export interface StepRange {
    min: number;
    max: number;
    /**
     * The only liveness signal at the API boundary. ``true`` means a
     * RunWriter is currently attached to this run on the server (i.e.
     * the game is actively writing). ``false`` means the run is closed
     * (or has not been opened by any writer in this process).
     *
     * Defaulted to ``false`` for older servers that did not yet emit
     * the field.
     */
    tail_growing?: boolean;
}

export interface Bookmark {
    step: number;
    game: number;
    annotation: string;
    created: string;
}
