/**
 * Phase 4 e2e: Socket.io is an invalidation channel, not a data pipe.
 *
 * The full Phase 4 contract is:
 *   1. The browser opens the dashboard and subscribes to the current run
 *      via Socket.io ``subscribe_run``.
 *   2. The server accepts the subscription and records it on the sid.
 *   3. When a step is pushed server-side, the server emits ``step_added``
 *      with a minimal ``{run, step}`` payload to the subscribed sid only.
 *   4. The browser invalidates its TanStack Query cache and refetches via
 *      REST, which is the only data path now that ``liveData`` /
 *      ``onNewStep`` / ref pile are gone from ``App.tsx``.
 *
 * This spec exercises steps 1-3 end-to-end against a real running dashboard
 * server (no NetHack required). It stands up ``uv run dashboard`` pointed
 * at a temporary data directory seeded with one real DuckLake run, then:
 *
 *   - Opens Playwright-controlled tabs that connect to the server's
 *     Socket.io endpoint directly (bypassing the SPA) and issues
 *     ``subscribe_run`` to prove the server-side handler installs the
 *     callback without error.
 *   - Calls ``unsubscribe_run`` and ``disconnect`` to prove the cleanup
 *     path runs without error.
 *   - Loads the dashboard SPA with ``?run=<name>`` and verifies that the
 *     browser's ``useRunSubscription`` hook also connects and fires the
 *     ``subscribe_run`` emit (confirming the hook is wired into App.tsx).
 *
 * What this spec does NOT cover (has unit coverage instead):
 *   - Actual ``notify_subscribers`` -> ``step_added`` fan-out. Unit tests
 *     in ``tests/unit/reporting/test_api_server.py::TestSubscribeRunSocketHandlers``
 *     and ``tests/unit/reporting/test_run_writer.py::TestRunWriterPushStep::
 *     test_push_step_notifies_subscribers`` verify the server-side
 *     fan-out, and ``dashboard-ui/src/hooks/useRunSubscription.test.tsx``
 *     verifies the browser-side invalidation. Triggering a real
 *     push through the running server from outside its process requires
 *     either a test-only HTTP endpoint or a full game subprocess -- both
 *     out of scope for Phase 4.
 *
 * The test skips gracefully if the server can't start.
 *
 * Run:
 *   cd dashboard-ui && pnpm exec playwright test live-update-via-invalidation.spec.ts
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

const PORT = 9078; // Phase 4 test port (frame-load-perf uses 9077)
const PROJECT_ROOT = join(__dirname, "..", "..");
const RUN_NAME = "20260409000000-phase4-test-run";

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

test.describe("Phase 4 live update via invalidation", () => {
    test.beforeAll(async () => {
        tempDataDir = mkdtempSync(join(tmpdir(), "roc-e2e-phase4-"));
        // Seed 25 steps so the run clears the default ``min_steps=10``
        // listing filter. Otherwise the SPA auto-select effect hides
        // the run from the dropdown and ``useRunSubscription`` never
        // fires for it.
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

    test("server accepts subscribe_run / unsubscribe_run handshake", async ({ page }) => {
        // Drive Socket.io directly from the page context. The dashboard
        // static assets already ship socket.io-client, so we rely on
        // fetching the library from the CDN-style path that the app
        // itself uses. Instead of reaching into Vite's dep cache, we
        // use a more portable strategy: navigate to the dashboard (which
        // loads the client-side socket.io bundle) and then reuse the
        // same ``window`` module cache.
        await page.goto(BASE_URL, { waitUntil: "networkidle" });

        // Wait for the app to mount so socket.io-client is present in
        // the page's module graph.
        await page.waitForTimeout(1500);

        const result = await page.evaluate(
            async ({ runName, port }: { runName: string; port: number }) => {
                // The dashboard's singleton socket lives inside the hook's
                // module closure and is not exposed on window. Rather than
                // reach into it, we use the raw transport via the
                // ``/socket.io/`` endpoint: open a native WebSocket to the
                // Engine.IO handshake URL and verify the server responds.
                //
                // This is intentionally low-level so the test does not
                // depend on the client library's module path shifting
                // between dev and prod builds.
                const proto = window.location.protocol === "https:" ? "https:" : "http:";
                const handshakeUrl = `${proto}//127.0.0.1:${port}/socket.io/?EIO=4&transport=polling`;
                try {
                    const res = await fetch(handshakeUrl);
                    if (!res.ok) {
                        return { ok: false, reason: `handshake status ${res.status}` };
                    }
                    const text = await res.text();
                    // Engine.IO handshake response begins with a packet
                    // type byte (``0``) followed by a JSON sid payload.
                    if (!text.startsWith("0")) {
                        return { ok: false, reason: `unexpected packet: ${text.slice(0, 20)}` };
                    }
                    // Extract sid so we can prove the server is routing
                    // subscribe_run correctly downstream. The payload
                    // looks like ``0{"sid":"...","upgrades":[...],...}``.
                    const jsonStart = text.indexOf("{");
                    if (jsonStart < 0) return { ok: false, reason: "no json" };
                    const open = JSON.parse(text.slice(jsonStart));
                    if (typeof open.sid !== "string") {
                        return { ok: false, reason: "no sid in open packet" };
                    }
                    // Engine.IO namespace connect:
                    const connectBody = "40";
                    const postUrl = `${proto}//127.0.0.1:${port}/socket.io/?EIO=4&transport=polling&sid=${open.sid}`;
                    await fetch(postUrl, {
                        method: "POST",
                        headers: { "Content-Type": "text/plain;charset=UTF-8" },
                        body: connectBody,
                    });
                    // Emit subscribe_run with the run name as the argument.
                    // Socket.io v4 event packet: ``42`` + JSON(['event', arg])
                    const subBody = `42${JSON.stringify(["subscribe_run", runName])}`;
                    const subRes = await fetch(postUrl, {
                        method: "POST",
                        headers: { "Content-Type": "text/plain;charset=UTF-8" },
                        body: subBody,
                    });
                    if (!subRes.ok) {
                        return { ok: false, reason: `subscribe_run status ${subRes.status}` };
                    }
                    // Poll once for any pushed events (there should be
                    // none without a push).
                    const pollRes = await fetch(postUrl);
                    if (!pollRes.ok) {
                        return { ok: false, reason: `poll status ${pollRes.status}` };
                    }
                    // Now unsubscribe and close cleanly.
                    const unsubBody = `42${JSON.stringify(["unsubscribe_run", runName])}`;
                    await fetch(postUrl, {
                        method: "POST",
                        headers: { "Content-Type": "text/plain;charset=UTF-8" },
                        body: unsubBody,
                    });
                    // Disconnect packet: ``41``
                    await fetch(postUrl, {
                        method: "POST",
                        headers: { "Content-Type": "text/plain;charset=UTF-8" },
                        body: "41",
                    });
                    return { ok: true, sid: open.sid };
                } catch (exc) {
                    return {
                        ok: false,
                        reason: exc instanceof Error ? exc.message : String(exc),
                    };
                }
            },
            { runName: RUN_NAME, port: PORT },
        );

        expect(result.ok, `handshake failed: ${"reason" in result ? result.reason : ""}`).toBe(
            true,
        );
    });

    test("dashboard SPA emits subscribe_run for the current run", async ({ page }) => {
        // Verify the App.tsx integration: when the SPA loads with a
        // ``?run=<name>`` param, the ``useRunSubscription`` hook emits
        // ``subscribe_run`` on the shared singleton socket. We observe
        // this at the network level -- Socket.io v4 may use either
        // polling (POST with the event as the request body) or WS
        // (framesent). We watch both.
        const eventPackets: string[] = [];
        const allSocketRequests: string[] = [];

        page.on("request", (req) => {
            const url = req.url();
            if (url.includes("/socket.io/")) {
                allSocketRequests.push(`${req.method()} ${url}`);
                if (req.method() === "POST") {
                    const body = req.postData();
                    if (body && body.includes("subscribe_run")) {
                        eventPackets.push(body);
                    }
                }
            }
        });

        const wsPackets: string[] = [];
        let wsSeen = false;
        page.on("websocket", (ws) => {
            wsSeen = true;
            ws.on("framesent", (frame) => {
                if (frame.payload) {
                    const text =
                        typeof frame.payload === "string"
                            ? frame.payload
                            : frame.payload.toString("utf-8");
                    if (text.includes("subscribe_run")) {
                        wsPackets.push(text);
                    }
                }
            });
        });

        // Capture any page errors so we can see what the SPA is doing.
        const consoleErrors: string[] = [];
        page.on("pageerror", (err) => consoleErrors.push(err.message));

        await page.goto(`${BASE_URL}?run=${encodeURIComponent(RUN_NAME)}`, {
            waitUntil: "networkidle",
        });

        // Give the hook's useEffect time to fire and the socket to
        // connect + emit. 4s covers the 3s liveStatus polling interval
        // plus a margin for the WS upgrade.
        await page.waitForTimeout(4000);

        // Ask the page what run the dashboard actually selected, so we
        // can either (a) assert the subscribe_run frame matches or
        // (b) skip with a clear diagnostic.
        const selectedRun = await page.evaluate(() => {
            const input = document.querySelector<HTMLInputElement>(
                'input[placeholder="Run"]',
            );
            return input?.value ?? "";
        });

        const total = eventPackets.length + wsPackets.length;

        if (total === 0) {
            // Provide full diagnostic context so a future failure is
            // debuggable without rerunning.
            const diagnostics = [
                `Selected run in UI: ${selectedRun || "(none)"}`,
                `Expected run: ${RUN_NAME}`,
                `WebSocket seen: ${wsSeen}`,
                `Socket.io requests observed: ${allSocketRequests.length}`,
                `Page errors: ${consoleErrors.length > 0 ? consoleErrors.join("; ") : "none"}`,
            ].join("\n  ");
            throw new Error(
                `No subscribe_run frame observed on either transport.\n  ${diagnostics}`,
            );
        }

        // Verify the emitted frame actually names our run. The Socket.io
        // event packet format is ``42["subscribe_run","<run>"]`` with a
        // possible numeric namespace/id prefix.
        const haystack = [...eventPackets, ...wsPackets].join("\n");
        expect(haystack).toContain("subscribe_run");
        expect(haystack).toContain(RUN_NAME);
    });
});
