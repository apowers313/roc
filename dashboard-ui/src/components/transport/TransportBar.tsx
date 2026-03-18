/** Transport controls -- step slider, play/pause, speed, run/game selectors. */

import {
    ActionIcon,
    Group,
    Select,
    Slider,
    Text,
} from "@mantine/core";
import {
    ChevronFirst,
    ChevronLast,
    ChevronLeft,
    ChevronRight,
    Pause,
    Play,
} from "lucide-react";
import { useCallback, useEffect, useRef } from "react";

import { useGames, useRuns, useStepRange } from "../../api/queries";
import { useDashboard } from "../../state/context";

const SPEED_OPTIONS = [
    { value: "2000", label: "0.5x" },
    { value: "1000", label: "1x" },
    { value: "500", label: "2x" },
    { value: "200", label: "5x" },
    { value: "100", label: "10x" },
    { value: "50", label: "20x" },
    { value: "16", label: "60x" },
];

interface TransportBarProps {
    connected?: boolean;
    /** Ref that indicates whether the current step's data has loaded. */
    stepDataReadyRef?: React.RefObject<boolean>;
}

export function TransportBar({ connected, stepDataReadyRef }: TransportBarProps) {
    const {
        run,
        setRun,
        game,
        setGame,
        step,
        setStep,
        stepMin,
        stepMax,
        setStepRange,
        playing,
        setPlaying,
        speed,
        setSpeed,
        playback,
        dispatchPlayback,
        liveRunName,
    } = useDashboard();

    const { data: runs } = useRuns();
    const { data: games } = useGames(run);
    const { data: stepRangeData } = useStepRange(run, game || undefined);

    // Derive effective step range: prefer REST query data, fall back to context
    // (which is updated by live pushes). This avoids stale context values when
    // switching runs -- the REST response is authoritative.
    const effectiveMin = stepRangeData?.min ?? stepMin;
    const effectiveMax = stepRangeData?.max ?? stepMax;

    // Sync to context so other components (App.tsx) see the range
    useEffect(() => {
        if (stepRangeData) {
            setStepRange(stepRangeData.min, stepRangeData.max);
        }
    }, [stepRangeData, setStepRange]);

    // Auto-select first run (API returns newest-first)
    useEffect(() => {
        if (runs && runs.length > 0 && !run) {
            const name = runs[0]!.name;
            setRun(name);
            // Eagerly fetch step-range
            void fetch(`/api/runs/${encodeURIComponent(name)}/step-range?game=1`)
                .then((r) => r.json())
                .then((d: { min: number; max: number }) => {
                    if (d.max > 0) {
                        setStepRange(d.min, d.max);
                    }
                })
                .catch(() => {});
        }
    }, [runs, run, setRun, setStepRange]);

    // Auto-play timer -- uses setTimeout loop that waits for the current
    // step's data to load before advancing. This prevents request pileup
    // at high speeds (10x+) where the fetch time exceeds the interval.
    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const stepRef = useRef(step);
    stepRef.current = step;
    const stepMaxRef = useRef(stepMax);
    stepMaxRef.current = stepMax;
    const playingRef = useRef(playing);
    playingRef.current = playing;

    useEffect(() => {
        if (!playing) return;

        const advance = () => {
            if (!playingRef.current) return;

            // Wait for data before advancing: poll every 5ms until
            // stepDataReadyRef signals the fetch completed.
            if (stepDataReadyRef?.current === false) {
                timerRef.current = setTimeout(advance, 5);
                return;
            }

            const next = stepRef.current + 1;
            if (next > stepMaxRef.current) {
                setPlaying(false);
            } else {
                setStep(next);
                // Schedule next advance after the speed interval
                timerRef.current = setTimeout(advance, speed);
            }
        };

        timerRef.current = setTimeout(advance, speed);
        return () => {
            if (timerRef.current) clearTimeout(timerRef.current);
        };
    }, [playing, speed, setStep, setPlaying, stepDataReadyRef]);

    const togglePlay = useCallback(() => {
        if (playback === "historical") {
            dispatchPlayback({ type: "TOGGLE_PLAY" });
        }
        setPlaying(!playing);
    }, [playing, playback, setPlaying, dispatchPlayback]);

    const stepForward = useCallback(() => {
        setStep((prev) => (prev < effectiveMax ? prev + 1 : prev));
        if (playback === "live_following") {
            dispatchPlayback({ type: "USER_NAVIGATE" });
        }
    }, [effectiveMax, setStep, playback, dispatchPlayback]);

    const stepBack = useCallback(() => {
        setStep((prev) => (prev > effectiveMin ? prev - 1 : prev));
        if (playback === "live_following") {
            dispatchPlayback({ type: "USER_NAVIGATE" });
        }
    }, [effectiveMin, setStep, playback, dispatchPlayback]);

    const jumpToStart = useCallback(() => {
        setStep(effectiveMin);
        if (
            playback === "live_following" ||
            playback === "live_catchup"
        ) {
            dispatchPlayback({ type: "USER_NAVIGATE" });
        }
    }, [effectiveMin, setStep, playback, dispatchPlayback]);

    const jumpToEnd = useCallback(() => {
        setStep(effectiveMax);
        // Pure navigation -- just go to end of current game's range.
        // Use "L" or click "GO LIVE" badge to return to live-following.
        if (playback === "live_following") {
            dispatchPlayback({ type: "USER_NAVIGATE" });
        }
    }, [effectiveMax, setStep, playback, dispatchPlayback]);

    const handleSliderChange = useCallback(
        (value: number) => {
            setStep(value);
            if (playback === "live_following") {
                dispatchPlayback({ type: "USER_NAVIGATE" });
            }
        },
        [setStep, playback, dispatchPlayback],
    );

    const runOptions =
        runs?.map((r) => ({ value: r.name, label: r.name })) ?? [];
    const gameOptions =
        games?.map((g) => ({
            value: String(g.game_number),
            label: `Game ${g.game_number} (${g.steps} steps)`,
        })) ?? [];

    return (
        <div style={{ padding: "4px 8px" }}>
            <Group gap="sm" mb={4}>
                <Select
                    size="xs"
                    placeholder="Run"
                    searchable
                    value={run || null}
                    onChange={(v) => {
                        if (v) {
                            setRun(v);
                            setStep(1);
                            setGame(1);
                            // Switching away from live run enters historical mode
                            if (v !== liveRunName) {
                                dispatchPlayback({ type: "USER_NAVIGATE" });
                            }
                            // Eagerly fetch step-range for the new run so the
                            // slider updates immediately without waiting for
                            // TanStack Query to re-render.
                            void fetch(`/api/runs/${encodeURIComponent(v)}/step-range?game=1`)
                                .then((r) => r.json())
                                .then((d: { min: number; max: number }) => {
                                    if (d.max > 0) {
                                        setStepRange(d.min, d.max);
                                    }
                                })
                                .catch(() => {});
                        }
                    }}
                    data={runOptions}
                    style={{ width: 280 }}
                />
                <Select
                    size="xs"
                    placeholder="Game"
                    value={game ? String(game) : null}
                    onChange={(v) => {
                        if (v) {
                            const gameNum = Number(v);
                            setGame(gameNum);
                            // Selecting a specific game exits live-following
                            if (playback === "live_following") {
                                dispatchPlayback({ type: "USER_NAVIGATE" });
                            }
                            // Eagerly fetch step range for the game so
                            // we can jump to the game's first step.
                            void fetch(
                                `/api/runs/${encodeURIComponent(run)}/step-range?game=${gameNum}`,
                            )
                                .then((r) => r.json())
                                .then((d: { min: number; max: number }) => {
                                    if (d.max > 0) {
                                        setStepRange(d.min, d.max);
                                        setStep(d.min);
                                    }
                                })
                                .catch(() => {
                                    setStep(1);
                                });
                        }
                    }}
                    data={gameOptions}
                    style={{ width: 200 }}
                />
                <Select
                    size="xs"
                    value={String(speed)}
                    onChange={(v) => {
                        if (v) setSpeed(Number(v));
                    }}
                    data={SPEED_OPTIONS}
                    style={{ width: 80 }}
                />
                {connected !== undefined && (
                    <div
                        style={{
                            width: 8,
                            height: 8,
                            borderRadius: "50%",
                            background: connected ? "#3fb950" : "#f85149",
                        }}
                        title={connected ? "Connected" : "Disconnected"}
                    />
                )}
            </Group>

            <Group gap={4}>
                <ActionIcon
                    size="sm"
                    variant="subtle"
                    onClick={jumpToStart}
                    aria-label="First step"
                >
                    <ChevronFirst size={14} />
                </ActionIcon>
                <ActionIcon
                    size="sm"
                    variant="subtle"
                    onClick={stepBack}
                    aria-label="Previous step"
                >
                    <ChevronLeft size={14} />
                </ActionIcon>
                <ActionIcon
                    size="sm"
                    variant="subtle"
                    onClick={togglePlay}
                    aria-label={playing ? "Pause" : "Play"}
                >
                    {playing ? <Pause size={14} /> : <Play size={14} />}
                </ActionIcon>
                <ActionIcon
                    size="sm"
                    variant="subtle"
                    onClick={stepForward}
                    aria-label="Next step"
                >
                    <ChevronRight size={14} />
                </ActionIcon>
                <ActionIcon
                    size="sm"
                    variant="subtle"
                    onClick={jumpToEnd}
                    aria-label="Last step"
                >
                    <ChevronLast size={14} />
                </ActionIcon>

                {/* Intercept nav keys on the slider wrapper to prevent the
                    Slider's internal handler from also advancing the step.
                    We stop propagation + prevent default, then call the
                    appropriate step handler directly (since the event won't
                    reach the document-level useHotkeys listener). */}
                <div
                    style={{ flex: 1, minWidth: 200 }}
                    onKeyDownCapture={(e) => {
                        if (e.key === "ArrowRight" || e.key === "ArrowUp") {
                            e.stopPropagation();
                            e.preventDefault();
                            stepForward();
                        } else if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
                            e.stopPropagation();
                            e.preventDefault();
                            stepBack();
                        } else if (e.key === "Home") {
                            e.stopPropagation();
                            e.preventDefault();
                            jumpToStart();
                        } else if (e.key === "End") {
                            e.stopPropagation();
                            e.preventDefault();
                            jumpToEnd();
                        }
                    }}
                >
                    <Slider
                        size="xs"
                        min={effectiveMin}
                        max={effectiveMax}
                        value={step}
                        onChange={handleSliderChange}
                        label={(v) => String(v - effectiveMin + 1)}
                    />
                </div>

                <Text size="xs" c="dimmed" style={{ whiteSpace: "nowrap" }}>
                    {Math.min(step, effectiveMax) - effectiveMin + 1} / {effectiveMax - effectiveMin + 1}
                </Text>
            </Group>
        </div>
    );
}
