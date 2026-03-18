/** REST API client for the FastAPI backend. */

import type { Bookmark, GameSummary, RunSummary, StepRange } from "../types/api";
import type { StepData } from "../types/step-data";

const BASE = "/api";

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
    const opts: RequestInit | undefined = signal ? { signal } : undefined;
    const res = opts ? await fetch(url, opts) : await fetch(url);
    if (!res.ok) {
        throw new Error(`API error: ${res.status} ${res.statusText}`);
    }
    return res.json() as Promise<T>;
}

export async function fetchRuns(): Promise<RunSummary[]> {
    return fetchJson<RunSummary[]>(`${BASE}/runs`);
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
    const params = game != null ? `?game=${game}` : "";
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
    const params = game != null ? `?game=${game}` : "";
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
        `${BASE}/runs/${encodeURIComponent(run)}/metrics-history${qs ? `?${qs}` : ""}`,
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
        `${BASE}/runs/${encodeURIComponent(run)}/graph-history${qs ? `?${qs}` : ""}`,
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
        `${BASE}/runs/${encodeURIComponent(run)}/event-history${qs ? `?${qs}` : ""}`,
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
