/**
 * DashboardContext -- global state for the dashboard.
 *
 * Holds the current run, game, step, playback state, and step range.
 * All components read from this context rather than prop-drilling.
 */

import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState,
    type ReactNode,
} from "react";

import { initialPlayback } from "./playback";

// ---------------------------------------------------------------------------
// URL param helpers -- persist navigation state across reloads
// ---------------------------------------------------------------------------

// When true, ?latest was present on load.  readUrlParams() returns empty
// state so auto-select picks the newest run, and writeUrlParams() preserves
// ?latest in the URL until the first real navigation replaces it.
let _latestMode = false;

export function readUrlParams(): { run?: string; game?: number; step?: number } {
    const params = new URLSearchParams(globalThis.location.search);
    // ?latest makes the app ignore saved run/game/step and auto-select the
    // newest run.  The param stays in the URL so the user can bookmark it
    // (iOS captures the loaded page URL for "Add to Home Screen").  The
    // first writeUrlParams() call clears it.
    if (params.has("latest")) {
        _latestMode = true;
        return {};
    }
    const run = params.get("run") ?? undefined;
    const game = params.has("game") ? Number(params.get("game")) : undefined;
    const step = params.has("step") ? Number(params.get("step")) : undefined;
    return {
        run: run || undefined,
        game: Number.isFinite(game) && game! > 0 ? game : undefined,
        step: Number.isFinite(step) && step! > 0 ? step : undefined,
    };
}

/** Batch-update URL search params via replaceState (no reload, no history). */
function writeUrlParams(run: string, game: number, step: number) {
    const params = new URLSearchParams();
    if (run) params.set("run", run);
    if (game > 0) params.set("game", String(game));
    if (step > 0) params.set("step", String(step));
    // Keep ?latest in the URL until a run is actually selected, so the
    // bookmarkable URL survives the initial empty-state render cycles.
    if (_latestMode) {
        if (!run) {
            globalThis.history.replaceState(
                null, "", `${globalThis.location.pathname}?latest`,
            );
            return;
        }
        _latestMode = false;
    }
    const qs = params.toString();
    const url = qs ? `${globalThis.location.pathname}?${qs}` : globalThis.location.pathname;
    globalThis.history.replaceState(null, "", url);
}

// ---------------------------------------------------------------------------
// sessionStorage helpers -- persist UI preferences across reloads
// ---------------------------------------------------------------------------

const SESSION_SPEED_KEY = "roc-dashboard-speed";
const SESSION_AUTOFOLLOW_KEY = "roc-dashboard-autofollow";

function readSessionSpeed(): number {
    try {
        const v = sessionStorage.getItem(SESSION_SPEED_KEY);
        if (v) {
            const n = Number(v);
            if (Number.isFinite(n) && n > 0) return n;
        }
    } catch { /* private browsing */ }
    return 200; // default
}

function writeSessionSpeed(speed: number) {
    try { sessionStorage.setItem(SESSION_SPEED_KEY, String(speed)); }
    catch { /* private browsing */ }
}

function readSessionAutoFollow(): boolean {
    try {
        const v = sessionStorage.getItem(SESSION_AUTOFOLLOW_KEY);
        if (v != null) return v === "true";
    } catch { /* private browsing */ }
    return initialPlayback.autoFollow;
}

function writeSessionAutoFollow(autoFollow: boolean) {
    try { sessionStorage.setItem(SESSION_AUTOFOLLOW_KEY, String(autoFollow)); }
    catch { /* private browsing */ }
}

interface DashboardState {
    run: string;
    setRun: (run: string) => void;
    game: number;
    setGame: (game: number) => void;
    step: number;
    setStep: (step: number | ((prev: number) => number)) => void;
    stepMin: number;
    stepMax: number;
    /** True once ``setStepRange`` has been called with ``max > 0`` for
     *  the current run.  Consumers use this to avoid firing data fetches
     *  with an unclamped step value before the range is known. */
    stepRangeReady: boolean;
    setStepRange: (min: number, max: number) => void;
    playing: boolean;
    setPlaying: (playing: boolean) => void;
    autoFollow: boolean;
    setAutoFollow: (autoFollow: boolean) => void;
    speed: number;
    setSpeed: (speed: number) => void;
}

export const DashboardContext = createContext<DashboardState | null>(null);

export function DashboardProvider({ children }: Readonly<{ children: ReactNode }>) {
    // Read URL params once on mount for initial state
    const initial = useRef(readUrlParams());

    const [run, rawSetRun] = useState(initial.current.run ?? "");
    const setRun = useCallback((r: string) => {
        rawSetRun((prev) => {
            if (prev !== r) setStepRangeReady(false);
            return r;
        });
    }, []);
    const [game, setGame] = useState(initial.current.game ?? 1);
    const [step, setStep] = useState(initial.current.step ?? 1);
    const [stepMin, setStepMin] = useState(1);
    const [stepMax, setStepMax] = useState(1);
    const [stepRangeReady, rawSetStepRangeReady] = useState(false);
    const setStepRangeReady = useCallback((v: boolean) => {
        // eslint-disable-next-line no-console
        console.log("[Context] stepRangeReady =", v);
        rawSetStepRangeReady(v);
    }, []);
    const [rawPlaying, setRawPlaying] = useState(initialPlayback.playing);
    const [rawAutoFollow, setRawAutoFollow] = useState(readSessionAutoFollow);
    const [rawSpeed, setRawSpeed] = useState(readSessionSpeed);

    // Wrap setters to also persist to sessionStorage
    const playing = rawPlaying;
    const setPlaying = useCallback((p: boolean) => {
        setRawPlaying(p);
    }, []);
    const autoFollow = rawAutoFollow;
    const setAutoFollow = useCallback((af: boolean) => {
        setRawAutoFollow(af);
        writeSessionAutoFollow(af);
    }, []);
    const speed = rawSpeed;
    const setSpeed = useCallback((s: number) => {
        setRawSpeed(s);
        writeSessionSpeed(s);
    }, []);

    const setStepRange = useCallback((min: number, max: number) => {
        setStepMin(min);
        setStepMax(max);
        // Clamp current step into the new range. Without this, navigating
        // via URL (?step=2500) or chart click to a step beyond the data
        // range leaves step state pointing at a non-existent step. The
        // slider clamps display but the underlying state stays out of
        // range, so the StatusBar shows "Step 2500 | Game 0" with no data.
        // Skip when max <= 0 (range not yet known) so we don't clobber an
        // in-progress URL navigation while the range query is loading.
        if (max > 0) {
            setStepRangeReady(true);
            setStep((current) => {
                if (current < min) return min;
                if (current > max) return max;
                return current;
            });
        }
    }, []);

    // Sync navigation state to URL. Debounced: during playback step changes
    // rapidly, so we batch via requestAnimationFrame.
    const rafRef = useRef(0);
    useEffect(() => {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = requestAnimationFrame(() => {
            writeUrlParams(run, game, step);
        });
        return () => cancelAnimationFrame(rafRef.current);
    }, [run, game, step]);

    // Scroll position persistence -- save on pagehide (fires reliably
    // before discard on both iOS and desktop), restore on mount.
    useEffect(() => {
        const onPageHide = () => {
            try {
                sessionStorage.setItem(
                    "roc-dashboard-scroll",
                    String(window.scrollY),
                );
            } catch { /* private browsing */ }
        };
        window.addEventListener("pagehide", onPageHide);

        // Restore scroll position from a previous session/discard.
        const saved = sessionStorage.getItem("roc-dashboard-scroll");
        if (saved) {
            const y = Number(saved);
            if (Number.isFinite(y) && y > 0) {
                requestAnimationFrame(() => window.scrollTo(0, y));
            }
        }

        return () => window.removeEventListener("pagehide", onPageHide);
    }, []);

    // Safety-net flush on `freeze` (Page Lifecycle API). The reactive
    // writes above handle the normal case; this catches the edge where
    // state changed in the same frame the browser decides to freeze.
    useEffect(() => {
        const flush = () => {
            try {
                writeSessionAutoFollow(autoFollow);
                sessionStorage.setItem(
                    "roc-dashboard-scroll",
                    String(window.scrollY),
                );
            } catch { /* private browsing */ }
        };
        document.addEventListener("freeze", flush);
        return () => document.removeEventListener("freeze", flush);
    }, [autoFollow]);

    // Expose setStep for e2e testing (Playwright can call globalThis.__testSetStep)
    useEffect(() => {
        const g = globalThis as unknown as Record<string, unknown>;
        g.__testSetStep = setStep;
        return () => { delete g.__testSetStep; };
    }, [setStep]);

    const contextValue = useMemo<DashboardState>(() => ({
        run,
        setRun,
        game,
        setGame,
        step,
        setStep,
        stepMin,
        stepMax,
        stepRangeReady,
        setStepRange,
        playing,
        setPlaying,
        autoFollow,
        setAutoFollow,
        speed,
        setSpeed,
    }), [
        run, setRun, game, setGame, step, setStep,
        stepMin, stepMax, stepRangeReady, setStepRange,
        playing, setPlaying, autoFollow, setAutoFollow,
        speed, setSpeed,
    ]);

    return (
        <DashboardContext.Provider
            value={contextValue}
        >
            {children}
        </DashboardContext.Provider>
    );
}

export function useDashboard(): DashboardState {
    const ctx = useContext(DashboardContext);
    if (!ctx) {
        throw new Error("useDashboard must be used within DashboardProvider");
    }
    return ctx;
}
