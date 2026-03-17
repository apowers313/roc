/** REST API client for the FastAPI backend. */

import type { Bookmark, GameSummary, RunSummary, StepRange } from "../types/api";
import type { StepData } from "../types/step-data";

const BASE = "/api";

async function fetchJson<T>(url: string): Promise<T> {
    const res = await fetch(url);
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

export async function fetchStepRange(
    run: string,
    game?: number,
): Promise<StepRange> {
    const params = game != null ? `?game=${game}` : "";
    return fetchJson<StepRange>(
        `${BASE}/runs/${encodeURIComponent(run)}/step-range${params}`,
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
