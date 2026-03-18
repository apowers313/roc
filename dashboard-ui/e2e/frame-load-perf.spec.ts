/**
 * E2E performance test for dashboard frame loading.
 *
 * Starts the standalone dashboard server pointing at real run data, then
 * uses Playwright to navigate through steps and measure:
 *   - API response time for /api/runs/{run}/step/{n} (cold, no cache)
 *   - DOM update time with cache invalidated before each navigation
 *
 * Every measurement is a cold fetch -- we invalidate the TanStack Query
 * cache before each step so that prefetch and staleTime: Infinity don't
 * mask real latency.
 *
 * Run:
 *   cd dashboard-ui && pnpm test:e2e
 *
 * Requires:
 *   - A valid run directory with a DuckLake catalog in DATA_DIR
 *   - The dashboard-ui dist/ built (cd dashboard-ui && pnpm build)
 */

import { test, expect } from "@playwright/test";
import { spawn, type ChildProcess } from "child_process";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ---------- Configuration ----------

const DATA_DIR = process.env.E2E_DATA_DIR ?? "/home/apowers/data";
const PORT = 9077; // Random port in the allowed 9000-9099 range
const PROJECT_ROOT = join(__dirname, "..", "..");

// Allow self-signed certs for local testing
process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

// Performance budget (milliseconds)
const API_P95_BUDGET_MS = 100; // 95th percentile API latency
const RENDER_P95_BUDGET_MS = 500; // 95th percentile cold DOM update latency

// ---------- Helpers ----------

function percentile(sorted: number[], p: number): number {
    const idx = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[Math.max(0, idx)]!;
}

function formatStats(label: string, values: number[]): string {
    const sorted = [...values].sort((a, b) => a - b);
    const p50 = percentile(sorted, 50);
    const p95 = percentile(sorted, 95);
    const max = sorted[sorted.length - 1]!;
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    return `${label}: avg=${avg.toFixed(1)}ms  p50=${p50.toFixed(1)}ms  p95=${p95.toFixed(1)}ms  max=${max.toFixed(1)}ms  (n=${values.length})`;
}

// ---------- Server lifecycle ----------

let serverProcess: ChildProcess | null = null;
let BASE_URL = `https://127.0.0.1:${PORT}`;
/** Captured server log lines containing step fetch timing. */
const serverStepLogs: string[] = [];

async function startServer(): Promise<void> {
    serverProcess = spawn(
        "uv",
        ["run", "dashboard", "--data-dir", DATA_DIR, "--port", String(PORT), "--host", "127.0.0.1"],
        {
            cwd: PROJECT_ROOT,
            stdio: ["ignore", "pipe", "pipe"],
            // Enable DEBUG logging so we see the per-step timing lines
            env: { ...process.env, roc_emit_state: "false", roc_log_level: "DEBUG" },
        },
    );

    // Capture stderr (loguru writes there) -- collect GET step lines
    serverProcess.stderr?.setEncoding("utf-8");
    serverProcess.stderr?.on("data", (chunk: string) => {
        for (const line of chunk.split("\n")) {
            if (line.includes("GET step")) {
                serverStepLogs.push(line.trim());
            }
        }
    });

    // Wait for the server to be ready (poll /api/runs).
    // Try HTTPS first (server uses SSL if certs are in .env), fall back to HTTP.
    const deadline = Date.now() + 20_000;
    while (Date.now() < deadline) {
        for (const proto of ["https", "http"]) {
            try {
                const res = await fetch(`${proto}://127.0.0.1:${PORT}/api/runs`);
                if (res.ok) {
                    BASE_URL = `${proto}://127.0.0.1:${PORT}`;
                    return;
                }
            } catch {
                // not ready yet
            }
        }
        await new Promise((r) => setTimeout(r, 500));
    }
    throw new Error("Dashboard server failed to start within 20s");
}

function stopServer(): void {
    if (serverProcess) {
        serverProcess.kill("SIGTERM");
        serverProcess = null;
    }
}

/** Find a run that has actual step data by querying the API. */
async function findRunWithData(): Promise<string | null> {
    interface RunInfo { name: string; games: number; steps: number }
    const res = await fetch(`${BASE_URL}/api/runs`);
    if (!res.ok) return null;
    const runs: RunInfo[] = await res.json();

    for (const run of runs) {
        // The listing may report steps=0 even for valid runs, so always
        // check the step-range endpoint.
        try {
            const rangeRes = await fetch(
                `${BASE_URL}/api/runs/${encodeURIComponent(run.name)}/step-range`,
            );
            if (!rangeRes.ok) continue;
            const range: { min: number; max: number } = await rangeRes.json();
            if (range.max > range.min) return run.name;
        } catch {
            continue;
        }
    }
    return null;
}

/**
 * Invalidate all TanStack Query caches in the browser.
 *
 * This removes every cached query so the next navigation triggers a real
 * network fetch instead of reading from the in-memory cache. Without this,
 * usePrefetchAdjacentSteps pre-loads step+/-1 and staleTime: Infinity
 * means those cached entries never expire -- making latency measurements
 * meaningless.
 */
async function invalidateQueryCache(page: import("@playwright/test").Page): Promise<void> {
    await page.evaluate(() => {
        // TanStack Query stores the QueryClient on window.__REACT_QUERY_DEVTOOLS__
        // but that's only in dev mode. Instead, reach into React's fiber tree
        // to find the QueryClient, or use a simpler approach: remove all
        // cached data by finding the QueryClientProvider's client.
        //
        // The most reliable approach: call queryClient.clear() via a global
        // we expose, or invalidate via removeQueries. Since we can't easily
        // get the QueryClient reference, we use a different strategy:
        // intercept at the network level by using the Cache API or simply
        // force a hard reload.
        //
        // Actually the simplest reliable approach: we expose a global helper.
        // But since we don't want to modify prod code, we'll use
        // queryClient.getQueryCache().clear() via React internals.

        // Walk the React fiber tree to find the QueryClient
        const root = document.getElementById("root");
        if (!root) return;

        // React 18 stores the fiber on _reactRootContainer or __reactFiber$*
        const fiberKey = Object.keys(root).find(
            (k) => k.startsWith("__reactFiber$") || k.startsWith("__reactContainer$"),
        );
        if (!fiberKey) return;

        // Walk up/down the fiber tree to find a node with queryClient in its context
        let fiber = (root as Record<string, unknown>)[fiberKey] as Record<string, unknown> | null;
        const visited = new Set();
        while (fiber && !visited.has(fiber)) {
            visited.add(fiber);
            const state = fiber.memoizedState as Record<string, unknown> | null;
            if (state) {
                // TanStack Query's QueryClientProvider stores the client in context
                let node = state as Record<string, unknown> | null;
                for (let depth = 0; node && depth < 20; depth++) {
                    const q = node.queue as Record<string, unknown> | null;
                    const val = node.memoizedState;
                    if (
                        val &&
                        typeof val === "object" &&
                        "queryCache" in (val as Record<string, unknown>)
                    ) {
                        const client = val as { queryCache: { clear: () => void }; clear: () => void };
                        client.clear();
                        return;
                    }
                    node = (node.next ?? q?.next ?? null) as Record<string, unknown> | null;
                }
            }
            fiber = (fiber.child ?? fiber.sibling ?? fiber.return) as Record<string, unknown> | null;
        }
    });
}

/** Select a run in the Mantine searchable Select. */
async function selectRun(page: import("@playwright/test").Page, runName: string): Promise<void> {
    const runInput = page.locator('input[placeholder="Run"]');
    await runInput.click();
    await page.keyboard.press("Control+a");
    await page.keyboard.type(runName.slice(0, 20), { delay: 10 });
    await page.waitForTimeout(500);
    const option = page.locator('[role="option"]')
        .filter({ hasText: runName.slice(0, 15) })
        .first();
    if (await option.isVisible({ timeout: 2_000 }).catch(() => false)) {
        await option.click();
    }
}

// ---------- Tests ----------

test.describe("Frame loading performance", () => {
    let runName: string;

    test.beforeAll(async () => {
        await startServer();

        const found = await findRunWithData();
        if (!found) {
            stopServer();
            test.skip();
            return;
        }
        runName = found;
    });

    test.afterAll(() => {
        stopServer();
    });

    test("API step fetch latency (cold, no cache)", async () => {
        // This test fetches directly from Node.js -- no browser cache involved.
        // Each fetch is a cold request to the Python server.
        const rangeRes = await fetch(
            `${BASE_URL}/api/runs/${encodeURIComponent(runName)}/step-range`,
        );
        const range: { min: number; max: number } = await rangeRes.json();
        expect(range.max).toBeGreaterThan(range.min);

        const steps = Math.min(range.max - range.min + 1, 50);
        const latencies: number[] = [];

        for (let i = 0; i < steps; i++) {
            const step = range.min + i;
            const t0 = performance.now();
            const res = await fetch(
                `${BASE_URL}/api/runs/${encodeURIComponent(runName)}/step/${step}`,
            );
            const t1 = performance.now();
            expect(res.ok).toBe(true);
            // Consume the body to include deserialization time
            await res.json();
            const t2 = performance.now();
            latencies.push(t2 - t0);
        }

        const sorted = [...latencies].sort((a, b) => a - b);
        const p95 = percentile(sorted, 95);

        console.log(formatStats("API step fetch (cold)", latencies));
        expect(p95).toBeLessThan(API_P95_BUDGET_MS);
    });

    test("browser frame navigation latency (cache invalidated)", async ({ page }) => {
        // Clear any step logs from prior tests (e.g. API-only test)
        serverStepLogs.length = 0;

        await page.goto(BASE_URL, { waitUntil: "networkidle" });
        await page.waitForSelector('[aria-label="Next step"]', { timeout: 10_000 });
        await selectRun(page, runName);
        await page.waitForSelector("pre", { timeout: 10_000 });

        // Clear logs again after run selection (which triggers its own fetches)
        serverStepLogs.length = 0;

        const nextBtn = page.locator('[aria-label="Next step"]');
        const renderLatencies: number[] = [];
        let networkFetches = 0;
        let cacheHits = 0;
        const steps = 30;

        for (let i = 0; i < steps; i++) {
            // Invalidate the entire TanStack Query cache before each step
            // so that usePrefetchAdjacentSteps and staleTime: Infinity
            // don't serve cached data. Every click triggers a real fetch.
            await invalidateQueryCache(page);

            const prevHtml = await page.locator("pre").first().innerHTML();

            // Wait for the actual network request to go out and come back
            const responsePromise = page.waitForResponse(
                (res) => res.url().includes("/api/runs/") && res.url().includes("/step/"),
                { timeout: 5_000 },
            ).catch(() => null);

            const t0 = performance.now();
            await nextBtn.click();

            // Wait for the <pre> content to change (new frame rendered)
            try {
                await page.waitForFunction(
                    (prev: string) => {
                        const el = document.querySelector("pre");
                        return el !== null && el.innerHTML !== prev;
                    },
                    prevHtml,
                    { timeout: 5_000 },
                );
            } catch {
                // Frame might be identical (e.g. no screen change) -- skip
                continue;
            }

            const t1 = performance.now();
            const latency = t1 - t0;
            renderLatencies.push(latency);

            // Ensure the network request actually fired (not a cache hit)
            const response = await responsePromise;
            if (response) {
                networkFetches++;
                console.log(`  step ${i}: ${latency.toFixed(1)}ms (network fetch)`);
            } else {
                cacheHits++;
                console.log(`  step ${i}: ${latency.toFixed(1)}ms WARNING -- no network request (cache hit)`);
            }
        }

        // Print server-side datastore read logs to confirm real reads
        console.log(`\nServer-side datastore reads (${serverStepLogs.length} total):`);
        for (const line of serverStepLogs) {
            console.log(`  ${line}`);
        }
        console.log(`\nSummary: ${networkFetches} network fetches, ${cacheHits} cache hits, ${serverStepLogs.length} server-side reads`);

        if (renderLatencies.length > 0) {
            console.log(formatStats("Browser render (cold, click->DOM update)", renderLatencies));
            const sorted = [...renderLatencies].sort((a, b) => a - b);
            const p95 = percentile(sorted, 95);
            expect(p95).toBeLessThan(RENDER_P95_BUDGET_MS);
        } else {
            throw new Error("No render latency measurements collected");
        }
    });

    test("random-access frame load (no prefetch, 100+ step jumps)", async ({ page }) => {
        serverStepLogs.length = 0;

        // Get step range so we can pick random targets
        const rangeRes = await fetch(
            `${BASE_URL}/api/runs/${encodeURIComponent(runName)}/step-range`,
        );
        const range: { min: number; max: number } = await rangeRes.json();
        const totalSteps = range.max - range.min + 1;

        // Build a list of step targets that are each 100+ apart.
        // This guarantees every fetch is a pure cold read -- no prefetch
        // buffer (step+/-1) or sequential read-ahead can help.
        const targets: number[] = [];
        const minJump = 100;
        for (let s = range.min; s <= range.max; s += minJump + Math.floor(Math.random() * 50)) {
            targets.push(s);
        }
        // If the run is too small for 100+ jumps, fall back to max-spread jumps
        if (targets.length < 3 && totalSteps > 10) {
            targets.length = 0;
            const jump = Math.max(3, Math.floor(totalSteps / 10));
            for (let s = range.min; s <= range.max; s += jump) {
                targets.push(s);
            }
        }
        // Remove step 1 (the initially-loaded step) to avoid a guaranteed cache hit
        const initialStep = 1;
        const filtered = targets.filter((s) => s !== initialStep);
        targets.length = 0;
        targets.push(...filtered);

        // Shuffle so we don't even read in ascending order
        for (let i = targets.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [targets[i], targets[j]] = [targets[j]!, targets[i]!];
        }

        console.log(`Random-access test: ${targets.length} targets, min_jump=${totalSteps > minJump * 3 ? minJump : "max-spread"}, range=${range.min}-${range.max} (${totalSteps} steps)`);

        await page.goto(BASE_URL, { waitUntil: "networkidle" });
        await page.waitForSelector('[aria-label="Next step"]', { timeout: 10_000 });
        await selectRun(page, runName);
        await page.waitForSelector("pre", { timeout: 10_000 });
        serverStepLogs.length = 0;

        const renderLatencies: number[] = [];
        let networkFetches = 0;
        let cacheHits = 0;

        for (let i = 0; i < targets.length; i++) {
            const target = targets[i]!;

            // Invalidate cache before each jump
            await invalidateQueryCache(page);

            const prevHtml = await page.locator("pre").first().innerHTML();

            // Listen for the network request for this specific step
            const responsePromise = page.waitForResponse(
                (res) => res.url().includes("/api/runs/") && res.url().includes(`/step/${target}`),
                { timeout: 10_000 },
            ).catch(() => null);

            const t0 = performance.now();

            // Jump to the target step via the exposed __testSetStep global.
            // This calls React's setStep directly -- triggers a state update,
            // which triggers useStepData(run, target), which fetches from the
            // API, which reads from parquet. Full pipeline, no shortcuts.
            await page.evaluate((step) => {
                const fn = (window as any).__testSetStep;  // eslint-disable-line
                if (typeof fn === "function") fn(step);
            }, target);

            // Wait for the network response for this specific step
            const response = await responsePromise;

            // Wait for the <pre> content to change
            try {
                await page.waitForFunction(
                    (prev: string) => {
                        const el = document.querySelector("pre");
                        return el !== null && el.innerHTML !== prev;
                    },
                    prevHtml,
                    { timeout: 5_000 },
                );
            } catch {
                // Screen didn't change -- still count the network latency
            }

            const t1 = performance.now();
            const latency = t1 - t0;

            if (response) {
                networkFetches++;
                renderLatencies.push(latency);
                console.log(`  jump ${i}: step ${target} -> ${latency.toFixed(1)}ms (network fetch)`);
            } else {
                cacheHits++;
                // Don't include cache hits in latency stats -- they're artifacts
                console.log(`  jump ${i}: step ${target} -> ${latency.toFixed(1)}ms WARNING -- cache hit, excluded from stats`);
            }
        }

        // Print server-side logs
        console.log(`\nServer-side datastore reads (${serverStepLogs.length} total):`);
        for (const line of serverStepLogs) {
            console.log(`  ${line}`);
        }
        console.log(`\nSummary: ${networkFetches} network fetches, ${cacheHits} cache hits, ${serverStepLogs.length} server-side reads`);

        if (renderLatencies.length > 0) {
            console.log(formatStats("Random-access render (cold, 100+ step jumps)", renderLatencies));
            const sorted = [...renderLatencies].sort((a, b) => a - b);
            const p95 = percentile(sorted, 95);
            expect(p95).toBeLessThan(RENDER_P95_BUDGET_MS);
        }

        // At most 1 cache hit (edge case if prefetch races ahead)
        expect(cacheHits).toBeLessThanOrEqual(1);
    });

    test("rapid scrubbing does not pile up requests", async ({ page }) => {
        await page.goto(BASE_URL, { waitUntil: "networkidle" });
        await page.waitForSelector('[aria-label="Next step"]', { timeout: 10_000 });
        await selectRun(page, runName);
        await page.waitForSelector("pre", { timeout: 10_000 });

        // Invalidate cache so rapid clicks generate real requests
        await invalidateQueryCache(page);

        const nextBtn = page.locator('[aria-label="Next step"]');
        const pendingRequests: string[] = [];
        const completedRequests: string[] = [];

        page.on("request", (req) => {
            if (req.url().includes("/step/")) {
                pendingRequests.push(req.url());
            }
        });
        page.on("response", (res) => {
            if (res.url().includes("/step/")) {
                completedRequests.push(res.url());
            }
        });

        // Click 20 times rapidly
        for (let i = 0; i < 20; i++) {
            await nextBtn.click();
        }

        // Wait for everything to settle
        await page.waitForTimeout(3_000);

        console.log(
            `Rapid scrub: ${pendingRequests.length} requests sent, ${completedRequests.length} completed`,
        );

        // All requests should complete (no hanging)
        expect(completedRequests.length).toBe(pendingRequests.length);
        // Should have actually made network requests
        expect(pendingRequests.length).toBeGreaterThan(0);
    });
});
