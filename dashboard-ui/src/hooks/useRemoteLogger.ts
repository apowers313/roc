/** Remote logger setup -- sends browser console output to the remote log server. */

import { RemoteLogClient } from "@graphty/remote-logger/client";
import { useEffect, useRef } from "react";

interface RemoteLoggerConfig {
    /** Remote log server URL (without /log path). */
    serverUrl: string;
    /** Project marker for filtering logs. */
    projectMarker: string;
    /** Whether remote logging is enabled. */
    enabled: boolean;
}

// Access Vite env vars via a typed helper to avoid ImportMeta issues in tests.
function envVar(name: string, fallback: string): string {
    try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const env = (import.meta as any).env;
        return (env?.[name] as string) ?? fallback;
    } catch {
        return fallback;
    }
}

const DEFAULT_CONFIG: RemoteLoggerConfig = {
    serverUrl: envVar("VITE_REMOTE_LOG_URL", "https://dev.ato.ms:9080"),
    projectMarker: envVar("VITE_REMOTE_LOG_MARKER", "roc-dashboard"),
    enabled: envVar("VITE_REMOTE_LOG_ENABLED", "true") !== "false",
};

type ConsoleFn = (...args: unknown[]) => void;
const METHODS = ["log", "info", "warn", "error", "debug"] as const;
const LEVEL_MAP: Record<string, string> = {
    log: "INFO",
    info: "INFO",
    warn: "WARN",
    error: "ERROR",
    debug: "DEBUG",
};

/**
 * Hook that initializes the remote logger client and intercepts console output.
 *
 * Configure via environment variables:
 *   VITE_REMOTE_LOG_URL      - Server URL (default: https://dev.ato.ms:9080)
 *   VITE_REMOTE_LOG_MARKER   - Project marker (default: roc-dashboard)
 *   VITE_REMOTE_LOG_ENABLED  - Enable/disable (default: true)
 */
export function useRemoteLogger(config: Partial<RemoteLoggerConfig> = {}): void {
    const clientRef = useRef<RemoteLogClient | null>(null);

    const merged = { ...DEFAULT_CONFIG, ...config };

    useEffect(() => {
        if (!merged.enabled) return;

        const client = new RemoteLogClient({
            serverUrl: merged.serverUrl,
            projectMarker: merged.projectMarker,
        });
        clientRef.current = client;

        // Intercept console methods and forward to remote logger
        const originals = new Map<string, ConsoleFn>();
        for (const method of METHODS) {
            const orig: ConsoleFn = console[method].bind(console);
            originals.set(method, orig);
            console[method] = (...args: unknown[]) => {
                orig(...args);
                const message = args
                    .map((a) => (typeof a === "string" ? a : JSON.stringify(a)))
                    .join(" ");
                client.log(LEVEL_MAP[method] ?? "INFO", message);
            };
        }

        return () => {
            for (const method of METHODS) {
                const orig = originals.get(method);
                if (orig) console[method] = orig as typeof console.log;
            }
            client.close();
            clientRef.current = null;
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [merged.enabled, merged.serverUrl, merged.projectMarker]);
}
