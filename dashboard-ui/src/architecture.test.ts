/**
 * Architectural invariant tests.
 *
 * These tests check structural properties of the source tree -- rules
 * that are hard to express in TypeScript alone but that keep the
 * dashboard from regressing toward the multi-source-of-truth patterns
 * that caused TC-GAME-004, TC-SEL-005, and the Phase-3-era data races
 * fixed by the unified-run architecture.
 *
 * The pattern: glob the production source tree (excluding tests and
 * node_modules), read each file as text, and assert that certain
 * sentinels appear in exactly the expected place (usually "once" or
 * "only in this specific file").
 *
 * When one of these tests fails, DO NOT just widen the allowlist. The
 * failure means someone reintroduced a duplicated data source. Fix the
 * duplication. If the test itself is wrong (the rule genuinely needs
 * to change), update both the rule and the CLAUDE.md invariant it
 * encodes, and write a one-paragraph comment here explaining why.
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const SRC_ROOT = resolve(__dirname);

/** Walk ``root`` recursively and yield paths to production source files. */
function walkProductionSources(root: string): string[] {
    const results: string[] = [];
    const stack = [root];
    while (stack.length > 0) {
        const dir = stack.pop();
        if (dir == null) continue;
        let entries: string[];
        try {
            entries = readdirSync(dir);
        } catch {
            continue;
        }
        for (const entry of entries) {
            if (entry === "node_modules" || entry === "__mocks__") continue;
            const full = join(dir, entry);
            const st = statSync(full);
            if (st.isDirectory()) {
                stack.push(full);
                continue;
            }
            if (!st.isFile()) continue;
            // Production files only: .ts / .tsx, excluding tests, mocks,
            // and this file itself.
            if (!/\.(ts|tsx)$/.test(entry)) continue;
            if (/\.test\.tsx?$/.test(entry)) continue;
            if (entry === "architecture.test.ts") continue;
            results.push(full);
        }
    }
    return results;
}

function grepFiles(files: string[], pattern: RegExp): string[] {
    const matches: string[] = [];
    for (const file of files) {
        const text = readFileSync(file, "utf-8");
        if (pattern.test(text)) {
            matches.push(relative(SRC_ROOT, file));
        }
    }
    return matches;
}

describe("architectural invariants", () => {
    const productionFiles = walkProductionSources(SRC_ROOT);

    /**
     * TC-GAME-004 consolidation: the Socket.io + REST game-status
     * source of truth lives in ``useRunSubscription.ts``. Any other
     * file that calls ``fetch("/api/game/status")`` is duplicating
     * state and will eventually drift. If you find yourself needing
     * game status elsewhere, call ``useGameState()`` instead.
     */
    it("game-status fetch has exactly one home", () => {
        const matches = grepFiles(
            productionFiles,
            /fetch\s*\(\s*["'`]\/api\/game\/status["'`]/,
        );
        expect(matches).toEqual(["hooks/useRunSubscription.ts"]);
    });

    /**
     * Enforces the invariant stated in dashboard-ui/CLAUDE.md:
     * "TanStack Query is the only client store for server data.
     * Components never hold parallel state for step data, step
     * ranges, run lists, or live status." The concrete rule: no
     * production file may combine ``useState`` with an inline GET
     * ``fetch`` of a dashboard API endpoint. Form inputs / UI-only
     * state are fine; POST mutations are fine; reading server data
     * into a local store bypasses TanStack Query's cache and is the
     * exact anti-pattern that caused TC-GAME-004 and the Phase-3
     * race conditions.
     *
     * The rule intentionally does not ban every ``useState``; local
     * UI state like ``numGames`` or ``copied`` is legitimate. It
     * also does not ban every fetch -- POST mutations and bootstrap
     * reads owned by specific single-source hooks are allowed.
     *
     * The allowlist is explicit and small on purpose. Growing it
     * without a written justification is a signal that the
     * codebase is drifting back toward the fragmented state the
     * unified-run architecture cleaned up. Every entry here should
     * have a one-line reason and, ideally, a TODO to migrate to
     * ``queryClient.fetchQuery`` through the existing query hook.
     */
    it("no component holds local useState + dashboard API GET fetch", () => {
        // Historical allowlist is now empty. All former entries have
        // been migrated to ``queryClient.fetchQuery(...)`` through the
        // TanStack Query cache, so every GET of dashboard data flows
        // through a single source of truth. Adding a new entry here
        // requires a written justification.
        const ALLOWLIST = new Set<string>();

        const offenders: string[] = [];
        for (const file of productionFiles) {
            const text = readFileSync(file, "utf-8");
            // Skip the one hook that legitimately owns the REST +
            // Socket.io bridge for game state.
            if (file.endsWith("useRunSubscription.ts")) continue;
            if (!/\buseState\s*[<(]/.test(text)) continue;

            // Scan each fetch call individually. A POST fetch (a
            // mutation) is fine -- it is not a data store. A GET
            // fetch against /api/* is the anti-pattern.
            const fetchCalls = [
                ...text.matchAll(
                    /fetch\s*\(\s*(?:`[^`]*`|"[^"]*"|'[^']*')(?:\s*,\s*(\{[^}]*\}))?/g,
                ),
            ];
            let hasGetApiFetch = false;
            for (const match of fetchCalls) {
                const callText = match[0];
                const options = match[1] ?? "";
                if (!/\/api\/(game|runs|live)\//.test(callText)) continue;
                if (/method\s*:\s*["'`]POST["'`]/i.test(options)) continue;
                hasGetApiFetch = true;
                break;
            }
            if (hasGetApiFetch) {
                offenders.push(relative(SRC_ROOT, file));
            }
        }

        const unexpected = offenders.filter((f) => !ALLOWLIST.has(f));
        expect(unexpected).toEqual([]);
    });

    /**
     * Dead-code invariant: the pre-consolidation ``GameMenu.tsx``
     * file must stay deleted. If someone recreates it, we either
     * duplicate game state again or ship a file that is not imported
     * anywhere.
     */
    it("GameMenu.tsx stays deleted", () => {
        const matches = grepFiles(
            productionFiles,
            /^\/\/ placeholder for impossible match/,
        );
        expect(matches).toEqual([]);
        // Explicitly check the file does not exist on disk by looking
        // at every candidate path under components/transport/.
        const exists = productionFiles.some((f) =>
            f.endsWith("/components/transport/GameMenu.tsx"),
        );
        expect(exists).toBe(false);
    });

    /**
     * Phase-4 invariant: Socket.io is an invalidation channel, not a
     * data pipe. The one place that connects the Socket.io client is
     * ``useRunSubscription.ts``. Any other ``io(`` call means a
     * component is opening its own socket -- almost certainly to
     * receive data directly, which is the pattern Phase 4 deleted.
     */
    it("Socket.io client is instantiated in exactly one hook", () => {
        const matches = grepFiles(
            productionFiles,
            /\bio\s*\(\s*\{/,
        );
        expect(matches).toEqual(["hooks/useRunSubscription.ts"]);
    });
});
