/**
 * Phase 7 regression: explicit ?run=X URL must be preserved.
 *
 * The dashboard documents an "URL parameter sovereignty" invariant in
 * ``dashboard-ui/CLAUDE.md``: when ``?run=X`` is in the URL, the SPA
 * MUST NOT silently teleport the user to a different run, even if X's
 * endpoints fail. Phase 2 (T2.3) added the guard in
 * ``App.tsx::initialUrlRun`` to enforce this; Phase 2 (T2.6) added a
 * red Alert banner so the failure is visible instead of rendering a
 * confusingly-empty dashboard.
 *
 * This spec pins both contracts end-to-end:
 *   1. Intercept ``/api/runs/<broken-run>/games`` with a 500 response.
 *   2. Navigate to ``?run=<broken-run>&game=1&step=10``.
 *   3. Assert the URL still contains ``run=<broken-run>`` after the
 *      auto-select effects have had a chance to run.
 *   4. Assert the load-failure Alert appears.
 *
 * The Phase 2 unit test ``App.url-sovereignty.test.tsx`` covers the
 * same logic with mocked queries; this e2e spec exercises the real
 * SPA + real server (modulo a single mocked endpoint), so a future
 * regression in either the auto-select effect ordering or the banner
 * wiring is caught by the same test.
 *
 * Run:
 *   cd dashboard-ui && pnpm exec playwright test url-sovereignty.spec.ts
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

const PORT = 9080; // Phase 7 url-sovereignty (default-loads=9079)
const PROJECT_ROOT = join(__dirname, "..", "..");
const REAL_RUN_NAME = "20260409000000-url-sovereignty-real";
const BROKEN_RUN_NAME = "20260409000000-url-sovereignty-broken";

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
    serverProcess.stderr?.on("data", () => {
        // ignore
    });

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
        } catch {
            // ignore
        }
        tempDataDir = null;
    }
}

/** Seed a real DuckLake run with ``steps`` rows in the screens table. */
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

test.describe("Phase 7 URL parameter sovereignty", () => {
    test.beforeAll(async () => {
        tempDataDir = mkdtempSync(join(tmpdir(), "roc-e2e-url-sovereignty-"));
        // Seed both runs so /api/runs returns a non-empty list. We need
        // a "real" sibling to make sure the auto-select-first-run effect
        // has a fallback target it COULD pick if the sovereignty guard
        // were broken -- otherwise the test would pass trivially even
        // with a regression.
        seedRun(tempDataDir, REAL_RUN_NAME, 25);
        seedRun(tempDataDir, BROKEN_RUN_NAME, 25);
        const ok = await startServer(tempDataDir);
        if (!ok) {
            stopServer();
            test.skip(true, "Dashboard server failed to start");
        }
    });

    test.afterAll(() => {
        stopServer();
    });

    test("?run=X is preserved when /games returns 500", async ({ page }) => {
        // Force /api/runs/<broken>/games to fail. The route handler
        // matches the BROKEN run only -- the REAL run still works,
        // proving the sovereignty guard rejects the auto-fallback.
        await page.route(
            (url) =>
                url.pathname.endsWith(`/api/runs/${BROKEN_RUN_NAME}/games`) ||
                url.pathname.endsWith(`/api/runs/${BROKEN_RUN_NAME}/games/`),
            (route) => {
                route.fulfill({
                    status: 500,
                    contentType: "application/json",
                    body: JSON.stringify({ detail: "simulated 500 for sovereignty test" }),
                });
            },
        );

        const targetUrl = `${BASE_URL}/?run=${encodeURIComponent(BROKEN_RUN_NAME)}&game=1&step=10`;
        await page.goto(targetUrl, { waitUntil: "networkidle" });

        // Give every auto-select effect (live-status polling, runs
        // listing, games fetch, step-range fetch) a chance to settle.
        // The sovereignty guard fires synchronously inside a useEffect
        // when liveStatus arrives -- 4s covers the 3s polling interval
        // plus margin.
        await page.waitForTimeout(4_000);

        // Contract 1: URL still names the broken run. Read window.location
        // directly so we see what the SPA actually committed, not just
        // what Playwright thinks.
        const finalUrl = await page.evaluate(() => window.location.href);
        expect(
            finalUrl,
            `URL was rewritten away from ${BROKEN_RUN_NAME}: ${finalUrl}`,
        ).toContain(`run=${BROKEN_RUN_NAME}`);

        // Contract 2: the load-failure banner is visible. The banner
        // title is `Run "<name>" could not be loaded` -- match on the
        // distinctive substring.
        const banner = page.getByText(/could not be loaded/i);
        await expect(banner).toBeVisible({ timeout: 2_000 });
    });

    test("?run=X is preserved across explicit URL refresh", async ({ page }) => {
        // Same route mock as above so reload also hits the 500.
        await page.route(
            (url) =>
                url.pathname.endsWith(`/api/runs/${BROKEN_RUN_NAME}/games`) ||
                url.pathname.endsWith(`/api/runs/${BROKEN_RUN_NAME}/games/`),
            (route) => {
                route.fulfill({
                    status: 500,
                    contentType: "application/json",
                    body: JSON.stringify({ detail: "simulated 500" }),
                });
            },
        );

        const targetUrl = `${BASE_URL}/?run=${encodeURIComponent(BROKEN_RUN_NAME)}&game=1&step=5`;
        await page.goto(targetUrl, { waitUntil: "networkidle" });
        await page.waitForTimeout(2_000);

        // Reload preserves URL params; verify the SPA does not strip
        // the run param on the second mount.
        await page.reload({ waitUntil: "networkidle" });
        await page.waitForTimeout(2_000);

        const finalUrl = await page.evaluate(() => window.location.href);
        expect(finalUrl).toContain(`run=${BROKEN_RUN_NAME}`);
    });
});
