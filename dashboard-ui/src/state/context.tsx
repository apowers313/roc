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
    useReducer,
    useRef,
    useState,
    type Dispatch,
    type ReactNode,
} from "react";

import {
    playbackReducer,
    type PlaybackAction,
    type PlaybackState,
} from "./playback";

// ---------------------------------------------------------------------------
// URL param helpers -- persist navigation state across reloads
// ---------------------------------------------------------------------------

export function readUrlParams(): { run?: string; game?: number; step?: number } {
    const params = new URLSearchParams(globalThis.location.search);
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
    const qs = params.toString();
    const url = qs ? `${globalThis.location.pathname}?${qs}` : globalThis.location.pathname;
    globalThis.history.replaceState(null, "", url);
}

// ---------------------------------------------------------------------------
// sessionStorage helpers -- persist UI preferences across reloads
// ---------------------------------------------------------------------------

const SESSION_SPEED_KEY = "roc-dashboard-speed";

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

interface DashboardState {
    run: string;
    setRun: (run: string) => void;
    game: number;
    setGame: (game: number) => void;
    step: number;
    setStep: (step: number | ((prev: number) => number)) => void;
    stepMin: number;
    stepMax: number;
    setStepRange: (min: number, max: number) => void;
    playback: PlaybackState;
    dispatchPlayback: Dispatch<PlaybackAction>;
    playing: boolean;
    setPlaying: (playing: boolean) => void;
    speed: number;
    setSpeed: (speed: number) => void;
    /** The name of the currently-live run (set by live status poll). */
    liveRunName: string;
    setLiveRunName: (name: string) => void;
    /** The game number currently being played live. */
    liveGameNumber: number;
    setLiveGameNumber: (game: number) => void;
    /** Whether a live game is currently active (from status poll). */
    liveGameActive: boolean;
    setLiveGameActive: (active: boolean) => void;
}

export const DashboardContext = createContext<DashboardState | null>(null);

export function DashboardProvider({ children }: Readonly<{ children: ReactNode }>) {
    // Read URL params once on mount for initial state
    const initial = useRef(readUrlParams());

    const [run, setRun] = useState(initial.current.run ?? "");
    const [game, setGame] = useState(initial.current.game ?? 1);
    const [step, setStep] = useState(initial.current.step ?? 1);
    const [stepMin, setStepMin] = useState(1);
    const [stepMax, setStepMax] = useState(1);
    const [playing, setPlaying] = useState(false);
    const [rawSpeed, setRawSpeed] = useState(readSessionSpeed);
    const [liveRunName, setLiveRunName] = useState("");
    const [liveGameNumber, setLiveGameNumber] = useState(0);
    const [liveGameActive, setLiveGameActive] = useState(false);
    const [playback, dispatchPlayback] = useReducer(
        playbackReducer,
        "historical" as PlaybackState,
    );

    // Wrap setSpeed to also persist to sessionStorage
    const speed = rawSpeed;
    const setSpeed = useCallback((s: number) => {
        setRawSpeed(s);
        writeSessionSpeed(s);
    }, []);

    const setStepRange = useCallback((min: number, max: number) => {
        setStepMin(min);
        setStepMax(max);
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
        setStepRange,
        playback,
        dispatchPlayback,
        playing,
        setPlaying,
        speed,
        setSpeed,
        liveRunName,
        setLiveRunName,
        liveGameNumber,
        setLiveGameNumber,
        liveGameActive,
        setLiveGameActive,
    }), [
        run, setRun, game, setGame, step, setStep,
        stepMin, stepMax, setStepRange,
        playback, dispatchPlayback, playing, setPlaying,
        speed, setSpeed, liveRunName, setLiveRunName,
        liveGameNumber, setLiveGameNumber,
        liveGameActive, setLiveGameActive,
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
