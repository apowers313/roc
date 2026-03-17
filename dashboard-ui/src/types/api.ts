/** API response types for the FastAPI backend. */

export interface RunSummary {
    name: string;
    games: number;
    steps: number;
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
}

export interface Bookmark {
    step: number;
    game: number;
    annotation: string;
    created: string;
}
