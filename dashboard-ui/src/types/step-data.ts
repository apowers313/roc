/** Mirrors the Python StepData dataclass from roc/reporting/run_store.py. */

export interface StepData {
    step: number;
    game_number: number;
    timestamp: number | null;
    screen: GridData | null;
    saliency: GridData | null;
    features: Record<string, unknown>[] | null;
    object_info: Record<string, unknown>[] | null;
    focus_points: Record<string, unknown>[] | null;
    attenuation: Record<string, unknown> | null;
    resolution_metrics: Record<string, unknown> | null;
    graph_summary: Record<string, unknown> | null;
    event_summary: Record<string, unknown>[] | null;
    game_metrics: Record<string, unknown> | null;
    logs: LogEntry[] | null;
}

export interface GridData {
    chars: number[][];
    fg: string[][];
    bg: string[][];
}

export interface LogEntry {
    body?: string;
    severity_text?: string;
    timestamp?: number;
    [key: string]: unknown;
}
