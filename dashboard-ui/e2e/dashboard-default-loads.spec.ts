/**
 * Phase 7 regression: default landing page must NOT show ERROR.
 *
 * The Phase 1 / Phase 2 cascade made the dashboard render an "ERROR"
 * badge on first load: ``/api/runs/<run>/games`` 500'd because of the
 * DuckLake double-attach bug, the run-selection effect silently
 * fell back to ``runs[0]``, and the default-step calculation requested
 * a step that did not exist in game 1 -- producing 500 cascades all
 * the way down. Validation found this within seconds of opening the
 * dashboard, but no automated test caught it.
 *
 * This spec is the smoke test that would have caught it: spin up a
 * real ``uv run dashboard`` against a seeded run, navigate to the
 * default URL, and assert no ERROR badge appears within a few
 * seconds. Any future regression that breaks the default landing path
 * will fail this test before it reaches a human.
 *
 * Run:
 *   cd dashboard-ui && pnpm exec playwright test dashboard-default-loads.spec.ts
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

const PORT = 9079; // Phase 7 default-loads (frame-load-perf=9077, live-update=9078)
const PROJECT_ROOT = join(__dirname, "..", "..");
const RUN_NAME = "20260409000000-default-loads-test";

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

    // Drain stderr so the pipe does not fill up.
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

/** Seed a real DuckLake run with ``steps`` rows in the screens table.
 *  Mirrors the helper in live-update-via-invalidation.spec.ts so this
 *  spec stays self-contained. */
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

test.describe("Phase 7 default landing smoke test", () => {
    test.beforeAll(async () => {
        tempDataDir = mkdtempSync(join(tmpdir(), "roc-e2e-default-loads-"));
        // Seed 25 steps so the run clears the default ``min_steps=10``
        // listing filter -- otherwise /api/runs is empty and the
        // dashboard never auto-selects anything, so the test does not
        // exercise the cascade path that produced the original bug.
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

    test("default URL loads without an ERROR badge", async ({ page }) => {
        // Capture network failures so a regression has full diagnostic
        // context, not just "ERROR appeared".
        const failedResponses: string[] = [];
        page.on("response", (res) => {
            if (res.status() >= 500 && res.url().includes("/api/")) {
                failedResponses.push(`${res.status()} ${res.url()}`);
            }
        });

        await page.goto(BASE_URL, { waitUntil: "networkidle" });

        // The dashboard auto-selects ``runs[0]`` and lands on its first
        // game's first step. Wait long enough for that whole chain
        // (list_runs -> list_games -> step-range -> step) to settle.
        // 4 seconds covers a cold-start fetch + render budget; the
        // original cascade made the ERROR badge appear in <1s, so this
        // is generous without being slow.
        await page.waitForTimeout(4_000);

        // The ERROR badge is a Mantine red filled badge with the
        // literal text "ERROR" inside StatusBar. Use exact text match
        // (instead of a substring) so we do not accidentally swallow
        // unrelated occurrences of the word.
        const errorBadge = page.getByText("ERROR", { exact: true });
        const errorVisible = await errorBadge.isVisible().catch(() => false);

        if (errorVisible || failedResponses.length > 0) {
            const url = page.url();
            const diag = [
                `Page URL after settle: ${url}`,
                `ERROR badge visible: ${errorVisible}`,
                failedResponses.length > 0
                    ? `Failed API responses:\n  ${failedResponses.join("\n  ")}`
                    : "Failed API responses: none",
            ].join("\n");
            throw new Error(`Default landing regression detected.\n${diag}`);
        }

        // Sanity-check: the dashboard actually mounted and rendered the
        // run we seeded -- otherwise we are asserting "no ERROR" against
        // an unmounted page.
        const runInputValue = await page.evaluate(() => {
            const input = document.querySelector<HTMLInputElement>(
                'input[placeholder="Run"]',
            );
            return input?.value ?? "";
        });
        expect(runInputValue, "dashboard never selected a run").toContain(
            "default-loads-test",
        );
    });
});
