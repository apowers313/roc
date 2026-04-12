/** Transport controls -- step slider, play/pause, speed, run/game selectors. */

import {
    ActionIcon,
    Checkbox,
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
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { fetchStepRange } from "../../api/client";
import { useGames, useRuns, useStepRange } from "../../api/queries";
import { useDashboard } from "../../state/context";
import type { RunStatus, RunSummary } from "../../types/api";

const SPEED_OPTIONS = [
    { value: "2000", label: "0.5x" },
    { value: "1000", label: "1x" },
    { value: "500", label: "2x" },
    { value: "200", label: "5x" },
    { value: "100", label: "10x" },
    { value: "50", label: "20x" },
    { value: "16", label: "60x" },
];

/**
 * Render a RunSummary as a Mantine Select option.
 *
 * Exported for testing. The label format is:
 *   "[MM/DD HH:MM] adj-first-last (Ng, S steps)"
 *
 * Non-ok statuses are tagged with a trailing marker so the user can
 * see at a glance which runs are partial. We do NOT use color/icons
 * here because Mantine Select renders option text only -- enriching
 * options requires `renderOption`, which we add separately if needed.
 */
export function buildRunOption(r: RunSummary): { value: string; label: string } {
    const ts = r.name.slice(0, 14);
    const date =
        ts.length === 14
            ? `${ts.slice(4, 6)}/${ts.slice(6, 8)} ${ts.slice(8, 10)}:${ts.slice(10, 12)}`
            : "";
    const status: RunStatus = (r.status ?? "ok") as RunStatus;
    const suffix =
        r.games > 0 || r.steps > 0
            ? ` (${r.games}g, ${r.steps} steps)`
            : "";
    let tag = "";
    if (status === "short") tag = " [short]";
    else if (status === "empty") tag = " [empty]";
    else if (status === "corrupt") tag = " [corrupt]";
    else if (status === "missing") tag = " [missing]";
    return {
        value: r.name,
        label: `[${date}] ${r.name.slice(15)}${suffix}${tag}`,
    };
}

interface TransportBarProps {
    connected?: boolean;
    /** Ref that indicates whether the current step's data has loaded. */
    stepDataReadyRef?: React.RefObject<boolean>;
}

export function TransportBar({ connected, stepDataReadyRef }: Readonly<TransportBarProps>) {
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
        setAutoFollow,
        speed,
        setSpeed,
    } = useDashboard();

    const queryClient = useQueryClient();

    // "Show all runs" toggles between the default `ok`-only list and
    // the full list (including `short`/`empty`/`corrupt`). When the
    // user explicitly navigates to a run via URL param that the
    // server hides by default, we auto-flip this to `true` so the
    // dropdown reflects the active run instead of looking blank.
    // Persist across reloads so the user does not have to re-toggle.
    const [showAllRuns, setShowAllRuns] = useState<boolean>(() => {
        if (typeof window === "undefined") return false;
        return window.localStorage.getItem("roc.dashboard.showAllRuns") === "1";
    });
    useEffect(() => {
        if (typeof window === "undefined") return;
        window.localStorage.setItem(
            "roc.dashboard.showAllRuns",
            showAllRuns ? "1" : "0",
        );
    }, [showAllRuns]);

    const { data: runs } = useRuns(showAllRuns);
    const { data: games } = useGames(run);
    const { data: stepRangeData } = useStepRange(run, game || undefined);

    // Diagnostic logging: every time the run list arrives, log a
    // status breakdown so we can see from the iPad's remote logs
    // exactly what the dropdown is showing. Tagged "[Runs]" so it
    // is easy to grep for via remote-logger MCP.
    useEffect(() => {
        if (!runs) return;
        const breakdown: Record<string, number> = {};
        for (const r of runs) {
            const s = r.status ?? "ok";
            breakdown[s] = (breakdown[s] ?? 0) + 1;
        }
        // eslint-disable-next-line no-console
        console.log(
            "[Runs] received",
            JSON.stringify({
                total: runs.length,
                showAllRuns,
                breakdown,
                first: runs[0]?.name ?? null,
            }),
        );
    }, [runs, showAllRuns]);

    // If the URL pointed at a run that the default ("ok"-only) list
    // does not contain, automatically widen to include_all so the
    // user is not staring at a blank dropdown. We log this transition
    // so it is visible from the iPad's remote logs.
    useEffect(() => {
        if (!run) return;
        if (showAllRuns) return;
        if (!runs) return;
        const found = runs.some((r) => r.name === run);
        if (!found) {
            // eslint-disable-next-line no-console
            console.warn(
                "[Runs] active run not in default list; switching to include_all",
                JSON.stringify({ run, knownRuns: runs.length }),
            );
            setShowAllRuns(true);
        }
    }, [run, runs, showAllRuns]);

    // Phase 4: TanStack Query is the only data path. ``stepRangeData``
    // is always fresh -- Socket.io ``step_added`` events invalidate
    // the query via ``useRunSubscription`` and the refetch updates
    // this component's render. Prefer the query value over the context
    // fallback so a tail-growing run's slider tracks the live max even
    // when autoFollow is off (TC-GAME-004: after the user breaks
    // auto-follow, the slider must still show the growing max so
    // clicking GO LIVE snaps to the true head instead of a stale one).
    const effectiveMin = stepRangeData?.min ?? stepMin;
    const effectiveMax = stepRangeData?.max ?? stepMax;

    // Sync REST range to context so consumers that read from context
    // (App.tsx auto-follow effect, StatusBar, panels) see the same
    // value. TransportBar is the single owner of this sync -- App.tsx
    // used to dual-write from its auto-follow effect but now only
    // advances ``step``.
    useEffect(() => {
        if (stepRangeData) {
            setStepRange(stepRangeData.min, stepRangeData.max);
        }
    }, [stepRangeData, setStepRange]);

    // Auto-select first run (API returns newest-first).
    //
    // URL sovereignty (BUG-C2): the ``!run`` guard already protects
    // explicit URL runs -- when the URL has ``?run=X``, context init
    // populates ``run="X"`` so the effect bails. We deliberately do
    // NOT add an additional ``initialUrlRun`` ref guard here because
    // the ref would also block the legitimate "user clicked Browse
    // runs" path: after that click, ``run`` is cleared but a stale
    // ref would still hold the broken URL run, preventing the
    // dashboard from auto-selecting any new run. App.tsx's auto-select
    // -live-run effect uses ``initialUrlRun`` for a different reason
    // (to block live-takeover) where the stale ref behaviour is
    // actually correct.
    //
    // BUG-H1 fix: the eager step-range fetch must use the current
    // ``game`` from context, not a hardcoded ``?game=1``. If the URL
    // specified ``?game=N`` for N != 1, the prior code populated the
    // context range with game 1's bounds, then briefly rendered the
    // wrong slider (and queried step 1 of game N from the data fetcher,
    // which 404'd). Using the context's game keeps the eager fetch in
    // lock-step with the React Query that ``useStepRange(run, game)``
    // will eventually issue.
    useEffect(() => {
        if (runs && runs.length > 0 && !run) {
            // Prefer a run with data -- skip empty/corrupt/missing runs
            // so "Browse runs" doesn't land on an unusable run.
            const viable = runs.find(
                (r) =>
                    r.steps > 0 &&
                    (r.status === "ok" || r.status === "short" || !r.status),
            );
            const name = (viable ?? runs[0]!).name;
            setRun(name);
            // Eagerly prime the TanStack Query cache so ``useStepRange``
            // sees a cache hit on its next render.
            const targetGame = game > 0 ? game : 1;
            void queryClient
                .fetchQuery({
                    queryKey: ["step-range", name, targetGame],
                    queryFn: () => fetchStepRange(name, targetGame),
                })
                .then((d) => {
                    if (d.max > 0) {
                        setStepRange(d.min, d.max);
                    }
                })
                .catch(() => {});
        }
    }, [runs, run, game, setRun, setStepRange, queryClient]);

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
        setPlaying(!playing);
    }, [playing, setPlaying]);

    const stepForward = useCallback(() => {
        setStep((prev) => (prev < effectiveMax ? prev + 1 : prev));
        setAutoFollow(false);
    }, [effectiveMax, setStep, setAutoFollow]);

    const stepBack = useCallback(() => {
        setStep((prev) => (prev > effectiveMin ? prev - 1 : prev));
        setAutoFollow(false);
    }, [effectiveMin, setStep, setAutoFollow]);

    const jumpToStart = useCallback(() => {
        setStep(effectiveMin);
        setAutoFollow(false);
    }, [effectiveMin, setStep, setAutoFollow]);

    const jumpToEnd = useCallback(() => {
        setStep(effectiveMax);
        // Pure navigation -- just go to end of current game's range.
        // Use "L" or click "GO LIVE" badge to return to live-following.
        setAutoFollow(false);
    }, [effectiveMax, setStep, setAutoFollow]);

    const handleSliderChange = useCallback(
        (value: number) => {
            setStep(value);
            setAutoFollow(false);
        },
        [setStep, setAutoFollow],
    );

    const runOptions = useMemo(
        () => (runs ?? []).map(buildRunOption),
        [runs],
    );
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
                            // Switching runs drops autoFollow -- the new
                            // run's tail_growing flag (driven by
                            // useStepRange) determines whether GO LIVE
                            // applies.
                            setAutoFollow(false);
                            // Prime the TanStack Query cache so the slider
                            // updates immediately without waiting for
                            // ``useStepRange`` to re-render.
                            void queryClient
                                .fetchQuery({
                                    queryKey: ["step-range", v, 1],
                                    queryFn: () => fetchStepRange(v, 1),
                                })
                                .then((d) => {
                                    if (d.max > 0) {
                                        setStepRange(d.min, d.max);
                                    }
                                })
                                .catch(() => {});
                        }
                    }}
                    data={runOptions}
                    style={{ width: 420 }}
                />
                <Select
                    size="xs"
                    placeholder="Game"
                    value={game ? String(game) : null}
                    onChange={(v) => {
                        if (v) {
                            const gameNum = Number(v);
                            setGame(gameNum);
                            // Selecting a specific game drops autoFollow
                            setAutoFollow(false);
                            // Prime the TanStack Query cache with the new
                            // game's step range so we can jump to its first
                            // step without waiting for ``useStepRange`` to
                            // re-render.
                            void queryClient
                                .fetchQuery({
                                    queryKey: ["step-range", run, gameNum],
                                    queryFn: () => fetchStepRange(run, gameNum),
                                })
                                .then((d) => {
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
                <Checkbox
                    size="xs"
                    label="Show all"
                    checked={showAllRuns}
                    onChange={(e) => {
                        const next = e.currentTarget.checked;
                        setShowAllRuns(next);
                        // eslint-disable-next-line no-console
                        console.log(
                            "[Runs] showAllRuns toggled",
                            JSON.stringify({ next }),
                        );
                    }}
                    title="Show all runs including short, empty, and corrupt ones (default hides them)"
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
                            if (e.shiftKey) {
                                setStep((prev) => Math.min(prev + 10, effectiveMax));
                            } else {
                                stepForward();
                            }
                        } else if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
                            e.stopPropagation();
                            e.preventDefault();
                            if (e.shiftKey) {
                                setStep((prev) => Math.max(prev - 10, effectiveMin));
                            } else {
                                stepBack();
                            }
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
