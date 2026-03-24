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
    intrinsics: IntrinsicsData | null;
    significance: number | null;
    action_taken: ActionData | null;
    sequence_summary: SequenceData | null;
    transform_summary: TransformData | null;
    prediction: PredictionData | null;
    message: string | null;
    phonemes: PhonemeEntry[] | null;
    inventory: InventoryItem[] | null;
}

export interface IntrinsicsData {
    raw: Record<string, number>;
    normalized: Record<string, number>;
}

export interface ActionData {
    action_id: number;
    action_name?: string;
    action_key?: string;
    expmod_name?: string;
}

export interface SequenceObjectData {
    id: string;
    x?: number;
    y?: number;
    resolve_count?: number;
}

export interface SequenceData {
    tick: number;
    object_count: number;
    objects: SequenceObjectData[];
    intrinsic_count: number;
    intrinsics: Record<string, number>;
    significance?: number;
}

export interface TransformChangeData {
    description: string;
    type?: string;
    name?: string;
    normalized_change?: number;
}

export interface TransformData {
    count: number;
    /** Structured objects (new format) or plain strings (old format). */
    changes: (TransformChangeData | string)[];
}

export interface PredictionData {
    made: boolean;
    candidate_count?: number;
    confidence?: number;
    all_scores?: number[];
    predicted_intrinsics?: Record<string, number>;
    candidate_expmod?: string;
    confidence_expmod?: string;
}

export interface InventoryItem {
    letter: string;
    item: string;
    glyph: number;
}

export interface PhonemeEntry {
    word: string;
    phonemes: string[];
    is_break: boolean;
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
