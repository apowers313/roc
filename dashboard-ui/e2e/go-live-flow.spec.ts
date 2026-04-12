/**
 * TC-GAME-004 regression: end-to-end "break auto-follow -> GO LIVE" flow.
 *
 * The bug: TransportBar stopped syncing step-range to context when a run
 * was tail-growing, under the obsolete Phase-3 assumption that Socket.io
 * pushed data directly into context. In Phase 4 Socket.io is invalidation
 * -only and TanStack Query is the sole data path, so the guard froze the
 * slider max at whatever value it held when the user broke auto-follow.
 * Clicking GO LIVE (or pressing the L shortcut) then read a stale closure
 * and snapped to the frozen max rather than the true live head.
 *
 * The unit-level regression lives in
 * ``src/App.game-handling.test.tsx > TC-GAME-004: GO LIVE flow on a
 * tail-growing run``. This spec exercises the same contract against the
 * real built SPA + real dashboard server, mocking only the REST endpoints
 * needed to simulate a tail-growing run (so the test does not depend on a
 * real NetHack subprocess). If either layer regresses -- the TransportBar
 * sync effect, the App auto-follow effect, the goLive callback, or the
 * useGameState initial fetch -- one of the two tiers catches it.
 *
 * Run:
 *   cd dashboard-ui && pnpm exec playwright test go-live-flow.spec.ts
 */

import { test, expect } from "@playwright/test";
import { spawn, spawnSync, type ChildProcess } from "child_process";
import { dirname, join } from "path";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ---------- Configuration ----------

const PORT = 9081; // url-sovereignty=9080, default-loads=9079, live-update=9078, frame-load-perf=9077
const PROJECT_ROOT = join(__dirname, "..", "..");
const RUN_NAME = "20260409000000-go-live-e2e";

process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

// ---------- Server lifecycle ----------

let serverProcess: ChildProcess | null = null;
let tempDataDir: string | null = null;
let BASE_URL = `http://127.0.0.1:${PORT}`;

async function startServer(dataDir: string): Promise<boolean> {
    serverProcess = spawn(
        "uv",
        [
            "run",
            "dashboard",
            "--data-dir",
            dataDir,
            "--port",
            String(PORT),
            "--host",
            "127.0.0.1",
        ],
        {
            cwd: PROJECT_ROOT,
            stdio: ["ignore", "pipe", "pipe"],
            env: { ...process.env, roc_emit_state: "false", roc_log_level: "INFO" },
        },
    );
    serverProcess.stderr?.setEncoding("utf-8");
    serverProcess.stderr?.on("data", () => {});

    const deadline = Date.now() + 20_000;
    while (Date.now() < deadline) {
        for (const proto of ["https", "http"]) {
            try {
                const res = await fetch(`${proto}://127.0.0.1:${PORT}/api/runs`);
                if (res.ok) {
                    BASE_URL = `${proto}://127.0.0.1:${PORT}`;
                    return true;
                }
            } catch {
                // not ready yet
            }
        }
        await new Promise((r) => setTimeout(r, 500));
    }
    return false;
}

function stopServer(): void {
    if (serverProcess) {
        serverProcess.kill("SIGTERM");
        serverProcess = null;
    }
    if (tempDataDir) {
        try {
            rmSync(tempDataDir, { recursive: true, force: true });
        } catch {}
        tempDataDir = null;
    }
}

/** Seed a real DuckLake run so /api/runs listing is non-empty. */
function seedRun(dataDir: string, runName: string, steps: number): void {
    const script = [
        "import sys",
        "from pathlib import Path",
        "from roc.reporting.ducklake_store import DuckLakeStore",
        "run_dir = Path(sys.argv[1]) / sys.argv[2]",
        "steps = int(sys.argv[3])",
        "store = DuckLakeStore(run_dir, read_only=False)",
        "try:",
        "    records = [",
        '        {"step": s, "game_number": 1, "timestamp": s * 1000, "body": "{}"}',
        "        for s in range(1, steps + 1)",
        "    ]",
        '    store.insert("screens", records)',
        "finally:",
        "    store.close()",
    ].join("\n");

    const result = spawnSync(
        "uv",
        ["run", "python", "-c", script, dataDir, runName, String(steps)],
        { cwd: PROJECT_ROOT, stdio: "pipe", encoding: "utf-8" },
    );
    if (result.status !== 0) {
        throw new Error(
            `seedRun failed (status=${result.status}): ${result.stderr || result.stdout}`,
        );
    }
}

// ---------- Tests ----------

test.describe("TC-GAME-004 GO LIVE flow", () => {
    test.beforeAll(async () => {
        tempDataDir = mkdtempSync(join(tmpdir(), "roc-e2e-go-live-"));
        seedRun(tempDataDir, RUN_NAME, 25);
        const ok = await startServer(tempDataDir);
        if (!ok) {
            stopServer();
            test.skip(true, "Dashboard server failed to start");
        }
    });

    test.afterAll(() => {
        stopServer();
    });

    test("cold load with running game: LIVE badge, break, then L restores follow", async ({
        page,
    }) => {
        // Simulate a tail-growing run. The step-range endpoint returns
        // ``tail_growing: true`` on every call. The client only
        // refetches when Socket.io sends a step_added event or when
        // the user triggers an invalidation, so we do NOT rely on the
        // ``max`` growing inside this test -- that behavior is covered
        // by the unit-level integration test in
        // ``src/App.game-handling.test.tsx``. Here we pin the
        // cross-layer contracts: cold load + badge visibility + L
        // shortcut + goLive handler wiring.
        await page.route(
            `**/api/runs/${RUN_NAME}/step-range**`,
            (route) => {
                void route.fulfill({
                    status: 200,
                    contentType: "application/json",
                    body: JSON.stringify({
                        min: 1,
                        max: 100,
                        tail_growing: true,
                    }),
                });
            },
        );

        // useGameState does a one-shot fetch to /api/game/status on
        // mount (added as part of the TC-GAME-004 consolidation to
        // unblock goLive on a cold page load while a game is already
        // running). Return ``state: running`` so the goLive callback
        // has a valid run_name.
        await page.route(
            "**/api/game/status",
            (route) => {
                void route.fulfill({
                    status: 200,
                    contentType: "application/json",
                    body: JSON.stringify({
                        state: "running",
                        run_name: RUN_NAME,
                        exit_code: null,
                        error: null,
                    }),
                });
            },
        );

        // Navigate directly to the run so auto-select does not race
        // with the mocks.
        await page.goto(`${BASE_URL}/?run=${encodeURIComponent(RUN_NAME)}`, {
            waitUntil: "networkidle",
        });
        // Let the initial fetches + auto-follow effect settle.
        await page.waitForTimeout(1500);

        // Contract 1: LIVE badge appears on cold load.
        // This exercises:
        //   - useGameState's /api/game/status initial fetch (returns
        //     state=running) -- the fix that unblocks goLive on cold load.
        //   - TransportBar's sync effect (TransportBar.tsx) writing
        //     stepRangeData into context regardless of tail_growing --
        //     the TC-GAME-004 primary fix.
        //   - App.tsx's auto-follow effect advancing step when
        //     autoFollow + tail_growing.
        await expect(page.getByText(/^LIVE$/).first()).toBeVisible({
            timeout: 4_000,
        });

        // Contract 2: breaking auto-follow surfaces the GO LIVE badge.
        await page.keyboard.press("ArrowLeft");
        await expect(page.getByText("GO LIVE")).toBeVisible({ timeout: 2_000 });
        await expect(page.getByText(/^LIVE$/)).toBeHidden({ timeout: 2_000 });

        // Contract 3: the L keyboard shortcut restores auto-follow.
        // Pre-consolidation bug: goLive read gameState from the
        // Socket.io-only useGameState hook, which started at null on
        // cold load because no state-change event had fired yet, so
        // the callback early-returned silently. The /api/game/status
        // initial fetch added in useRunSubscription.ts ensures
        // gameState is populated before the user can press L.
        await page.keyboard.press("l");
        await expect(page.getByText(/^LIVE$/).first()).toBeVisible({
            timeout: 2_000,
        });
        await expect(page.getByText("GO LIVE")).toBeHidden({ timeout: 2_000 });
    });

    test("MenuBar reads game state from useGameState (no direct /api/game/status fetch)", async ({
        page,
    }) => {
        // Consolidation invariant: MenuBar must not fetch
        // /api/game/status itself. The pre-consolidation code had two
        // independent copies (GameMenu.tsx + MenuBar.tsx), each with
        // its own useState + fetch, which drifted against useGameState
        // during latency windows. Post-consolidation, MenuBar reads
        // via useGameState and never hits the endpoint on its own.
        //
        // We count network calls via page.on("request") rather than
        // page.route so the listener sees ALL requests regardless of
        // whether another route handler handles them. This also
        // avoids any single-handler-per-url restriction.
        const statusCalls: string[] = [];
        page.on("request", (req) => {
            const url = req.url();
            if (url.includes("/api/game/status")) {
                statusCalls.push(url);
            }
        });

        await page.goto(`${BASE_URL}/?run=${encodeURIComponent(RUN_NAME)}`, {
            waitUntil: "networkidle",
        });
        await page.waitForTimeout(1500);

        const initialCalls = statusCalls.length;
        expect(
            initialCalls,
            "useGameState should make the initial /api/game/status fetch",
        ).toBeGreaterThanOrEqual(1);

        // Open the Game menu multiple times. Each open used to trigger
        // an onOpen={refreshStatus} fetch. Post-consolidation, there
        // is no such fetch.
        const gameButton = page.getByText("Game").first();
        await gameButton.click();
        await page.waitForTimeout(300);
        // Close by pressing Escape
        await page.keyboard.press("Escape");
        await page.waitForTimeout(200);
        await gameButton.click();
        await page.waitForTimeout(300);
        await page.keyboard.press("Escape");

        // The only additional /api/game/status calls allowed after
        // page load are zero. Pre-fix, each menu open added one.
        expect(
            statusCalls.length,
            `MenuBar re-fetched /api/game/status on menu open: ${statusCalls.length - initialCalls} extra call(s)`,
        ).toBe(initialCalls);
    });
});
