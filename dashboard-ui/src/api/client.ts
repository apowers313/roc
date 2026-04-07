/** REST API client for the FastAPI backend. */

import type { Bookmark, GameSummary, RunSummary, StepRange } from "../types/api";
import type { CytoscapeData } from "../types/step-data";
import type { StepData } from "../types/step-data";

const BASE = "/api";

function qsSuffix(qs: string): string {
    return qs ? `?${qs}` : "";
}

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
    const opts: RequestInit | undefined = signal ? { signal } : undefined;
    const res = opts ? await fetch(url, opts) : await fetch(url);
    if (!res.ok) {
        throw new Error(`API error: ${res.status} ${res.statusText}`);
    }
    return res.json() as Promise<T>;
}

export async function fetchRuns(): Promise<RunSummary[]> {
    return fetchJson<RunSummary[]>(`${BASE}/runs?min_steps=10`);
}

export async function fetchGames(run: string): Promise<GameSummary[]> {
    return fetchJson<GameSummary[]>(
        `${BASE}/runs/${encodeURIComponent(run)}/games`,
    );
}

export async function fetchStep(
    run: string,
    step: number,
    game?: number,
): Promise<StepData> {
    const params = game == null ? "" : `?game=${game}`;
    return fetchJson<StepData>(
        `${BASE}/runs/${encodeURIComponent(run)}/step/${step}${params}`,
    );
}

export async function fetchStepsBatch(
    run: string,
    steps: number[],
    game?: number,
    signal?: AbortSignal,
): Promise<Record<string, StepData>> {
    const params = new URLSearchParams({ steps: steps.join(",") });
    if (game != null) params.set("game", String(game));
    return fetchJson<Record<string, StepData>>(
        `${BASE}/runs/${encodeURIComponent(run)}/steps?${params}`,
        signal,
    );
}

export async function fetchStepRange(
    run: string,
    game?: number,
): Promise<StepRange> {
    const params = game == null ? "" : `?game=${game}`;
    return fetchJson<StepRange>(
        `${BASE}/runs/${encodeURIComponent(run)}/step-range${params}`,
    );
}

export interface MetricsPoint {
    step: number;
    [key: string]: unknown;
}

export async function fetchMetricsHistory(
    run: string,
    game?: number,
    fields?: string[],
): Promise<MetricsPoint[]> {
    const params = new URLSearchParams();
    if (game != null) params.set("game", String(game));
    if (fields && fields.length > 0) params.set("fields", fields.join(","));
    const qs = params.toString();
    return fetchJson<MetricsPoint[]>(
        `${BASE}/runs/${encodeURIComponent(run)}/metrics-history${qsSuffix(qs)}`,
    );
}

export interface GraphPoint {
    step: number;
    node_count: number;
    node_max: number;
    edge_count: number;
    edge_max: number;
}

export async function fetchGraphHistory(
    run: string,
    game?: number,
): Promise<GraphPoint[]> {
    const params = new URLSearchParams();
    if (game != null) params.set("game", String(game));
    const qs = params.toString();
    return fetchJson<GraphPoint[]>(
        `${BASE}/runs/${encodeURIComponent(run)}/graph-history${qsSuffix(qs)}`,
    );
}

export interface EventPoint {
    step: number;
    [key: string]: unknown;
}

export async function fetchEventHistory(
    run: string,
    game?: number,
): Promise<EventPoint[]> {
    const params = new URLSearchParams();
    if (game != null) params.set("game", String(game));
    const qs = params.toString();
    return fetchJson<EventPoint[]>(
        `${BASE}/runs/${encodeURIComponent(run)}/event-history${qsSuffix(qs)}`,
    );
}

export interface IntrinsicsPoint {
    step: number;
    raw?: Record<string, number>;
    normalized?: Record<string, number>;
}

export async function fetchIntrinsicsHistory(
    run: string,
    game?: number,
): Promise<IntrinsicsPoint[]> {
    const params = new URLSearchParams();
    if (game != null) params.set("game", String(game));
    const qs = params.toString();
    return fetchJson<IntrinsicsPoint[]>(
        `${BASE}/runs/${encodeURIComponent(run)}/intrinsics-history${qsSuffix(qs)}`,
    );
}

export interface ActionPoint {
    step: number;
    action_id: number;
    action_name?: string;
    action_key?: string;
}

export async function fetchActionHistory(
    run: string,
    game?: number,
): Promise<ActionPoint[]> {
    const params = new URLSearchParams();
    if (game != null) params.set("game", String(game));
    const qs = params.toString();
    return fetchJson<ActionPoint[]>(
        `${BASE}/runs/${encodeURIComponent(run)}/action-history${qsSuffix(qs)}`,
    );
}

export interface ResolutionPoint {
    step: number;
    outcome: string;
    correct?: boolean | null;
}

export async function fetchResolutionHistory(
    run: string,
    game?: number,
): Promise<ResolutionPoint[]> {
    const params = new URLSearchParams();
    if (game != null) params.set("game", String(game));
    const qs = params.toString();
    return fetchJson<ResolutionPoint[]>(
        `${BASE}/runs/${encodeURIComponent(run)}/resolution-history${qsSuffix(qs)}`,
    );
}

export interface ResolvedObject {
    shape: string | null;
    glyph: string | null;
    color: string | null;
    type: string | null;
    node_id: string | null;
    step_added: number | null;
    match_count: number;
}

export async function fetchAllObjects(
    run: string,
    game?: number,
): Promise<ResolvedObject[]> {
    const params = new URLSearchParams();
    if (game != null) params.set("game", String(game));
    const qs = params.toString();
    return fetchJson<ResolvedObject[]>(
        `${BASE}/runs/${encodeURIComponent(run)}/all-objects${qsSuffix(qs)}`,
    );
}

export interface SchemaField {
    name: string;
    type: string;
    default: string | null;
    local: boolean;
    exclude: boolean;
}

export interface SchemaMethod {
    name: string;
    params: string;
    return_type: string;
    local: boolean;
}

export interface SchemaNode {
    name: string;
    parents: string[];
    fields: SchemaField[];
    methods: SchemaMethod[];
}

export interface SchemaEdge {
    name: string;
    type: string;
    connections: [string, string][];
    fields: SchemaField[];
}

export interface GraphSchema {
    mermaid: string;
    nodes: SchemaNode[];
    edges: SchemaEdge[];
}

export interface ActionMapEntry {
    action_id: number;
    action_name: string;
    action_key?: string;
}

export async function fetchActionMap(run: string): Promise<ActionMapEntry[]> {
    return fetchJson<ActionMapEntry[]>(
        `${BASE}/runs/${encodeURIComponent(run)}/action-map`,
    );
}

export async function fetchSchema(run: string): Promise<GraphSchema> {
    return fetchJson<GraphSchema>(
        `${BASE}/runs/${encodeURIComponent(run)}/schema`,
    );
}

export interface ObjectHistoryState {
    tick: number;
    x: number;
    y: number;
    glyph_type: number | null;
    color_type: number | null;
    shape_type: number | null;
    flood_size: number | null;
    line_size: number | null;
    distance: number | null;
    motion_direction: string | null;
    delta_old: number | null;
    delta_new: number | null;
}

export interface ObjectHistoryChange {
    property: string;
    type: string | null;
    delta: number | null;
    old_value: unknown;
    new_value: unknown;
}

export interface ObjectHistoryTransform {
    num_discrete_changes: number;
    num_continuous_changes: number;
    changes: ObjectHistoryChange[];
}

export interface ObjectHistoryInfo {
    uuid: number;
    resolve_count: number;
}

export interface ObjectHistoryData {
    states: ObjectHistoryState[];
    transforms: ObjectHistoryTransform[];
    info: ObjectHistoryInfo;
}

export async function fetchObjectHistory(
    run: string,
    objectId: number,
): Promise<ObjectHistoryData> {
    return fetchJson<ObjectHistoryData>(
        `${BASE}/runs/${encodeURIComponent(run)}/object/${objectId}/history`,
    );
}

export async function fetchFrameGraph(
    run: string,
    tick: number,
    game?: number,
    depth?: number,
): Promise<CytoscapeData> {
    const params = new URLSearchParams();
    if (game != null) params.set("game", String(game));
    if (depth != null) params.set("depth", String(depth));
    const qs = params.toString();
    return fetchJson<CytoscapeData>(
        `${BASE}/runs/${encodeURIComponent(run)}/graph/frame/${tick}${qsSuffix(qs)}`,
    );
}

export async function fetchObjectHistoryGraph(
    run: string,
    uuid: number,
): Promise<CytoscapeData> {
    return fetchJson<CytoscapeData>(
        `${BASE}/runs/${encodeURIComponent(run)}/graph/object/${uuid}`,
    );
}

export async function fetchNodeGraph(
    run: string,
    nodeId: string,
    depth?: number,
): Promise<CytoscapeData> {
    const params = new URLSearchParams();
    if (depth != null) params.set("depth", String(depth));
    const qs = params.toString();
    return fetchJson<CytoscapeData>(
        `${BASE}/runs/${encodeURIComponent(run)}/graph/node/${encodeURIComponent(nodeId)}${qsSuffix(qs)}`,
    );
}

export async function fetchBookmarks(run: string): Promise<Bookmark[]> {
    return fetchJson<Bookmark[]>(
        `${BASE}/runs/${encodeURIComponent(run)}/bookmarks`,
    );
}

export async function saveBookmarks(
    run: string,
    bookmarks: Bookmark[],
): Promise<void> {
    const res = await fetch(
        `${BASE}/runs/${encodeURIComponent(run)}/bookmarks`,
        {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(bookmarks),
        },
    );
    if (!res.ok) {
        throw new Error(`API error: ${res.status} ${res.statusText}`);
    }
}
