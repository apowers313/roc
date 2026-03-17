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
];

interface TransportBarProps {
    connected?: boolean;
}

export function TransportBar({ connected }: TransportBarProps) {
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
    } = useDashboard();

    const { data: runs } = useRuns();
    const { data: games } = useGames(run);
    const { data: stepRange } = useStepRange(run, game || undefined);

    // Update step range when data arrives
    useEffect(() => {
        if (stepRange) {
            setStepRange(stepRange.min, stepRange.max);
        }
    }, [stepRange, setStepRange]);

    // Auto-select first run (API returns newest-first)
    useEffect(() => {
        if (runs && runs.length > 0 && !run) {
            setRun(runs[0]!.name);
        }
    }, [runs, run, setRun]);

    // Auto-play timer -- uses refs to avoid stale closures
    const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const stepRef = useRef(step);
    stepRef.current = step;

    useEffect(() => {
        if (playing) {
            timerRef.current = setInterval(() => {
                const next = stepRef.current + 1;
                if (next > stepMax) {
                    setPlaying(false);
                } else {
                    setStep(next);
                }
            }, speed);
        }
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
        };
    }, [playing, speed, stepMax, setStep, setPlaying]);

    const togglePlay = useCallback(() => {
        if (playback === "historical") {
            dispatchPlayback({ type: "TOGGLE_PLAY" });
        }
        setPlaying(!playing);
    }, [playing, playback, setPlaying, dispatchPlayback]);

    const stepForward = useCallback(() => {
        setStep((prev) => (prev < stepMax ? prev + 1 : prev));
        if (playback === "live_following") {
            dispatchPlayback({ type: "USER_NAVIGATE" });
        }
    }, [stepMax, setStep, playback, dispatchPlayback]);

    const stepBack = useCallback(() => {
        setStep((prev) => (prev > stepMin ? prev - 1 : prev));
        if (playback === "live_following") {
            dispatchPlayback({ type: "USER_NAVIGATE" });
        }
    }, [stepMin, setStep, playback, dispatchPlayback]);

    const jumpToStart = useCallback(() => {
        setStep(stepMin);
        if (
            playback === "live_following" ||
            playback === "live_catchup"
        ) {
            dispatchPlayback({ type: "USER_NAVIGATE" });
        }
    }, [stepMin, setStep, playback, dispatchPlayback]);

    const jumpToEnd = useCallback(() => {
        setStep(stepMax);
        dispatchPlayback({ type: "JUMP_TO_END" });
    }, [stepMax, setStep, dispatchPlayback]);

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
                    value={run || null}
                    onChange={(v) => {
                        if (v) {
                            setRun(v);
                            setStep(1);
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
                            setGame(Number(v));
                            setStep(1);
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

                <Slider
                    size="xs"
                    min={stepMin}
                    max={stepMax}
                    value={step}
                    onChange={handleSliderChange}
                    style={{ flex: 1, minWidth: 200 }}
                    label={(v) => String(v)}
                />

                <Text size="xs" c="dimmed" style={{ whiteSpace: "nowrap" }}>
                    {step} / {stepMax}
                </Text>
            </Group>
        </div>
    );
}
