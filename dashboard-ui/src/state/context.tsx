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
    useReducer,
    useState,
    type Dispatch,
    type ReactNode,
} from "react";

import {
    playbackReducer,
    type PlaybackAction,
    type PlaybackState,
} from "./playback";

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
}

const DashboardContext = createContext<DashboardState | null>(null);

export function DashboardProvider({ children }: { children: ReactNode }) {
    const [run, setRun] = useState("");
    const [game, setGame] = useState(0);
    const [step, setStep] = useState(1);
    const [stepMin, setStepMin] = useState(1);
    const [stepMax, setStepMax] = useState(1);
    const [playing, setPlaying] = useState(false);
    const [speed, setSpeed] = useState(200); // ms interval
    const [playback, dispatchPlayback] = useReducer(
        playbackReducer,
        "historical" as PlaybackState,
    );

    const setStepRange = useCallback((min: number, max: number) => {
        setStepMin(min);
        setStepMax(max);
    }, []);

    return (
        <DashboardContext.Provider
            value={{
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
            }}
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
