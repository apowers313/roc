/** Main dashboard application layout. */

import {
    Accordion,
    AppShell,
    Grid,
    Text,
} from "@mantine/core";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useStepData, useGames } from "./api/queries";
import { ErrorBoundary } from "./components/common/ErrorBoundary";
import { useHighlight } from "./state/highlight";
import { ActionPanel } from "./components/panels/ActionPanel";
import { AllObjects } from "./components/panels/AllObjects";
import { AuralPerception } from "./components/panels/AuralPerception";
import { AttenuationPanel } from "./components/panels/AttenuationPanel";
import { BookmarkTable } from "./components/panels/BookmarkTable";
import { EventHistory } from "./components/panels/EventHistory";
import { EventSummary } from "./components/panels/EventSummary";
import { FeatureTable } from "./components/panels/FeatureTable";
import { FocusPoints } from "./components/panels/FocusPoints";
import { GameMetrics } from "./components/panels/GameMetrics";
import { GameScreen } from "./components/panels/GameScreen";
import { GraphHistory } from "./components/panels/GraphHistory";
import { IntrinsicsChart } from "./components/panels/IntrinsicsChart";
import { IntrinsicsPanel } from "./components/panels/IntrinsicsPanel";
import { InventoryPanel } from "./components/panels/InventoryPanel";
import { LogMessages } from "./components/panels/LogMessages";
import { ObjectInfo } from "./components/panels/ObjectInfo";
import { PipelineStatus } from "./components/panels/PipelineStatus";
import { ResolutionChart } from "./components/panels/ResolutionChart";
import { ResolutionInspector } from "./components/panels/ResolutionInspector";
import { SaliencyMap } from "./components/panels/SaliencyMap";
import { TransformPanel } from "./components/panels/TransformPanel";
import { StatusBar } from "./components/status/StatusBar";
import { BookmarkBar } from "./components/transport/BookmarkBar";
import { KeyboardHelp } from "./components/transport/KeyboardHelp";
import { TransportBar } from "./components/transport/TransportBar";
import { useBookmarks } from "./hooks/useBookmarks";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { usePrefetchWindow } from "./hooks/usePrefetchWindow";
import { useLiveUpdates } from "./hooks/useLiveUpdates";
import { useRemoteLogger } from "./hooks/useRemoteLogger";
import { useDashboard } from "./state/context";
import type { StepData } from "./types/step-data";

export function App() {
    const {
        run,
        setRun,
        step,
        setStep,
        game,
        setGame,
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
        setLiveGameNumber,
    } = useDashboard();

    const { data: games } = useGames(run);

    // Clear point highlights when step changes
    const { clear: clearHighlights } = useHighlight();
    useEffect(() => {
        clearHighlights();
    }, [step, clearHighlights]);

    useRemoteLogger();

    const isFollowing = playback === "live_following";

    // Bookmarks
    const bm = useBookmarks(run);

    // Keyboard help overlay
    const [helpOpen, setHelpOpen] = useState(false);

    // Accordion state -- persisted to sessionStorage
    const ACCORDION_KEY = "roc-dashboard-accordion";
    const ACCORDION_DEFAULT = useMemo(() => ["pipeline", "game-state", "perception"], []);
    const [openSections, setOpenSections] = useState<string[]>(() => {
        try {
            const saved = sessionStorage.getItem(ACCORDION_KEY);
            if (saved) {
                const parsed = JSON.parse(saved) as unknown;
                if (Array.isArray(parsed)) return parsed as string[];
            }
        } catch { /* private browsing */ }
        return ACCORDION_DEFAULT;
    });
    const handleAccordionChange = useCallback((value: string[]) => {
        setOpenSections(value);
        try { sessionStorage.setItem(ACCORDION_KEY, JSON.stringify(value)); }
        catch { /* private browsing */ }
    }, []);

    // Auto-open bookmarks section when bookmarks exist, collapse when empty
    const hasBookmarks = bm.bookmarks.length > 0;
    const prevHasBookmarks = useRef(hasBookmarks);
    useEffect(() => {
        if (hasBookmarks && !prevHasBookmarks.current) {
            // Bookmarks appeared -- open the section
            setOpenSections((prev) =>
                prev.includes("bookmarks") ? prev : [...prev, "bookmarks"],
            );
        } else if (!hasBookmarks && prevHasBookmarks.current) {
            // Bookmarks removed -- close the section
            setOpenSections((prev) => prev.filter((s) => s !== "bookmarks"));
        }
        prevHasBookmarks.current = hasBookmarks;
    }, [hasBookmarks]);

    // Live push data -- always updated from Socket.io push, regardless of
    // playback mode. This eliminates the race where isFollowingRef is stale
    // and liveData never gets populated.
    const [liveData, setLiveData] = useState<StepData | null>(null);

    // Track whether we've auto-selected the live run
    const liveRunSelected = useRef(false);

    // Use refs to avoid stale closures in the Socket.io callback
    const runRef = useRef(run);
    runRef.current = run;
    const gameRef = useRef(game);
    gameRef.current = game;
    const stepRef = useRef(step);
    stepRef.current = step;
    const stepMinRef = useRef(stepMin);
    stepMinRef.current = stepMin;
    const stepMaxRef = useRef(stepMax);
    stepMaxRef.current = stepMax;
    const isFollowingRef = useRef(isFollowing);
    isFollowingRef.current = isFollowing;
    const liveRunNameRef = useRef(liveRunName);
    liveRunNameRef.current = liveRunName;

    const onNewStep = useCallback(
        (pushData: StepData) => {
            // Always store the latest push data for live-following mode
            setLiveData(pushData);
            // Track which game is currently live
            setLiveGameNumber(pushData.game_number);

            const isViewingLiveRun =
                runRef.current === liveRunNameRef.current;
            const gameMatches =
                gameRef.current === pushData.game_number;

            if (isFollowingRef.current && isViewingLiveRun) {
                if (!gameMatches) {
                    // Live game changed (e.g. game 1 ended, game 2 started).
                    // Auto-switch to the new game so the user keeps following.
                    setGame(pushData.game_number);
                    setStepRange(pushData.step, pushData.step);
                } else {
                    // Update stepMax for the current game
                    setStepRange(stepMinRef.current, pushData.step);
                }
                // Advance step cursor to the live edge
                setStep(pushData.step);
            } else if (isViewingLiveRun && gameMatches) {
                // Paused on the same game -- update range and notify state machine
                setStepRange(stepMinRef.current, pushData.step);
                const atEdge = stepRef.current >= stepMaxRef.current;
                dispatchPlayback({ type: "PUSH_ARRIVED", atEdge });
            }
            // If viewing a different game than what's live, do nothing --
            // the user is browsing historical data for another game.
        },
        [setStep, setGame, setStepRange, dispatchPlayback, setLiveGameNumber],
    );

    const { connected, liveStatus } = useLiveUpdates({ onNewStep });

    // Keyboard shortcut handlers
    const stepForward = useCallback(() => {
        setStep((prev) => (prev < stepMax ? prev + 1 : prev));
        if (playback === "live_following") dispatchPlayback({ type: "USER_NAVIGATE" });
    }, [stepMax, setStep, playback, dispatchPlayback]);

    const stepBack = useCallback(() => {
        setStep((prev) => (prev > stepMin ? prev - 1 : prev));
        if (playback === "live_following") dispatchPlayback({ type: "USER_NAVIGATE" });
    }, [stepMin, setStep, playback, dispatchPlayback]);

    const jumpToStart = useCallback(() => {
        setStep(stepMin);
        if (playback === "live_following" || playback === "live_catchup") {
            dispatchPlayback({ type: "USER_NAVIGATE" });
        }
    }, [stepMin, setStep, playback, dispatchPlayback]);

    const jumpToEnd = useCallback(() => {
        setStep(stepMax);
        // Pure navigation -- just go to end of current game's range.
        // Use "L" or click "GO LIVE" badge to return to live-following.
        if (playback === "live_following") {
            dispatchPlayback({ type: "USER_NAVIGATE" });
        }
    }, [stepMax, setStep, playback, dispatchPlayback]);

    const togglePlay = useCallback(() => {
        if (playback === "historical") dispatchPlayback({ type: "TOGGLE_PLAY" });
        setPlaying(!playing);
    }, [playing, playback, setPlaying, dispatchPlayback]);

    const stepForward10 = useCallback(() => {
        setStep((prev) => Math.min(prev + 10, stepMax));
        if (playback === "live_following") dispatchPlayback({ type: "USER_NAVIGATE" });
    }, [stepMax, setStep, playback, dispatchPlayback]);

    const stepBack10 = useCallback(() => {
        setStep((prev) => Math.max(prev - 10, stepMin));
        if (playback === "live_following") dispatchPlayback({ type: "USER_NAVIGATE" });
    }, [stepMin, setStep, playback, dispatchPlayback]);

    const toggleBookmark = useCallback(() => {
        bm.toggleBookmark(step, game);
    }, [bm, step, game]);

    // Navigate to a bookmark -- switches game and fetches step range if needed.
    const navigateToBookmark = useCallback(
        (bookmark: { step: number; game: number }) => {
            if (bookmark.game !== game) {
                setGame(bookmark.game);
                if (playback === "live_following") {
                    dispatchPlayback({ type: "USER_NAVIGATE" });
                }
                void fetch(
                    `/api/runs/${encodeURIComponent(run)}/step-range?game=${bookmark.game}`,
                )
                    .then((r) => r.json())
                    .then((d: { min: number; max: number }) => {
                        if (d.max > 0) {
                            setStepRange(d.min, d.max);
                        }
                    })
                    .catch(() => {});
            }
            setStep(bookmark.step);
        },
        [game, run, setGame, setStep, setStepRange, playback, dispatchPlayback],
    );

    const goToNextBookmark = useCallback(() => {
        const next = bm.nextBookmark(step);
        if (next !== null) navigateToBookmark(next);
    }, [bm, step, navigateToBookmark]);

    const goToPrevBookmark = useCallback(() => {
        const prev = bm.prevBookmark(step);
        if (prev !== null) navigateToBookmark(prev);
    }, [bm, step, navigateToBookmark]);

    // Go live: jump to the live game's latest step and resume following.
    // Used by the "GO LIVE" badge click and the "L" keyboard shortcut.
    const goLive = useCallback(() => {
        if (!liveRunName) return;
        setRun(liveRunName);
        dispatchPlayback({ type: "GO_LIVE" });
        // The next Socket.io push will set the game, step, and range
        // via onNewStep's live_following path.
    }, [liveRunName, setRun, dispatchPlayback]);

    // Speed intervals ordered slow -> fast (matching SPEED_OPTIONS in TransportBar)
    const SPEED_VALUES = [2000, 1000, 500, 200, 100, 50, 16] as const;

    const speedUp = useCallback(() => {
        const idx = SPEED_VALUES.indexOf(speed as typeof SPEED_VALUES[number]);
        if (idx >= 0 && idx < SPEED_VALUES.length - 1) {
            setSpeed(SPEED_VALUES[idx + 1]!);
        } else if (idx < 0) {
            // Current speed not in list -- jump to fastest that's slower
            const next = SPEED_VALUES.find((v) => v < speed);
            if (next != null) setSpeed(next);
        }
    }, [speed, setSpeed]);

    const speedDown = useCallback(() => {
        const idx = SPEED_VALUES.indexOf(speed as typeof SPEED_VALUES[number]);
        if (idx > 0) {
            setSpeed(SPEED_VALUES[idx - 1]!);
        } else if (idx < 0) {
            // Current speed not in list -- jump to slowest that's faster
            const prev = [...SPEED_VALUES].reverse().find((v) => v > speed);
            if (prev != null) setSpeed(prev);
        }
    }, [speed, setSpeed]);

    const cycleGame = useCallback(() => {
        if (!games || games.length === 0) return;
        const gameNumbers = games.map((g) => g.game_number);
        const currentIdx = gameNumbers.indexOf(game);
        const nextIdx = (currentIdx + 1) % gameNumbers.length;
        const nextGame = gameNumbers[nextIdx]!;
        setGame(nextGame);
        if (playback === "live_following") {
            dispatchPlayback({ type: "USER_NAVIGATE" });
        }
        // Fetch step range for the new game
        void fetch(
            `/api/runs/${encodeURIComponent(run)}/step-range?game=${nextGame}`,
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
    }, [games, game, setGame, run, setStep, setStepRange, playback, dispatchPlayback]);

    useKeyboardShortcuts({
        stepForward,
        stepBack,
        togglePlay,
        jumpToStart,
        jumpToEnd,
        stepForward10,
        stepBack10,
        toggleHelp: useCallback(() => setHelpOpen((v) => !v), []),
        toggleBookmark,
        nextBookmark: goToNextBookmark,
        prevBookmark: goToPrevBookmark,
        goLive,
        speedUp,
        speedDown,
        cycleGame,
    });

    // Track the live run name and game number from status polls
    useEffect(() => {
        if (liveStatus?.active && liveStatus.run_name) {
            setLiveRunName(liveStatus.run_name);
            setLiveGameNumber(liveStatus.game_number);
        }
    }, [liveStatus, setLiveRunName, setLiveGameNumber]);

    // Auto-select the live run when a game starts.
    // Sets the run and game to match the live session, then GO_LIVE
    // so Socket.io pushes advance the step cursor.
    // The per-game step range is set by the useStepRange REST query
    // (authoritative), not from liveStatus (which reports global range).
    useEffect(() => {
        if (
            liveStatus?.active &&
            liveStatus.run_name &&
            !liveRunSelected.current
        ) {
            liveRunSelected.current = true;
            setRun(liveStatus.run_name);
            setGame(liveStatus.game_number);
            setStep(liveStatus.step);
            dispatchPlayback({ type: "GO_LIVE" });
        }
    }, [liveStatus, setRun, setGame, setStep, dispatchPlayback]);

    // REST data fetch. No debounce: with keepPreviousData the previous
    // step stays visible while the next one loads, so rapid step changes
    // don't cause flicker. DuckLake queries complete in ~8ms, well within
    // the playback interval.
    const { data: restData, isLoading, isPlaceholderData } = useStepData(
        run,
        step,
        game || undefined,
    );

    // Signal to the play timer that the current step's real data has
    // arrived (not just placeholder from the previous step).
    const stepDataReady = restData !== undefined && !isPlaceholderData;
    const stepDataReadyRef = useRef(stepDataReady);
    stepDataReadyRef.current = stepDataReady;

    // In live-following mode, prefer push data (instant) with REST fallback.
    // In historical/paused mode, use ONLY REST data -- never fall back to
    // liveData, which would cause a flicker of the live frame while the
    // historical step loads.
    const data = isFollowing ? liveData ?? restData : restData;

    // Click-to-navigate handler for history charts
    const handleChartStepClick = useCallback(
        (clickedStep: number) => {
            setStep(clickedStep);
            if (playback === "live_following") {
                dispatchPlayback({ type: "USER_NAVIGATE" });
            }
        },
        [setStep, playback, dispatchPlayback],
    );

    usePrefetchWindow(run, step, stepMin, stepMax, game || undefined);

    return (
        <AppShell header={{ height: 120 }} padding="xs">
            <AppShell.Header>
                <TransportBar connected={connected} stepDataReadyRef={stepDataReadyRef} />
                <BookmarkBar
                    bookmarks={bm.bookmarks}
                    currentStep={step}
                    stepMin={stepMin}
                    stepMax={stepMax}
                    isBookmarked={bm.isBookmarked(step)}
                    onToggle={toggleBookmark}
                    onNavigate={navigateToBookmark}
                    onAnnotate={bm.updateAnnotation}
                />
            </AppShell.Header>

            <KeyboardHelp opened={helpOpen} onClose={() => setHelpOpen(false)} />

            <AppShell.Main>
                <StatusBar data={data} playbackState={playback} onGoLive={goLive} />

                {isLoading && !data && (
                    <Text size="xs" c="dimmed" p="md">
                        Loading...
                    </Text>
                )}

                <Accordion
                    multiple
                    value={openSections}
                    onChange={handleAccordionChange}
                    variant="separated"
                >
                    <Accordion.Item value="pipeline">
                        <Accordion.Control>Pipeline Status</Accordion.Control>
                        <Accordion.Panel>
                            <ErrorBoundary>
                                <PipelineStatus data={data} />
                            </ErrorBoundary>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="bookmarks">
                        <Accordion.Control>Bookmarks</Accordion.Control>
                        <Accordion.Panel>
                            <ErrorBoundary>
                                <BookmarkTable
                                    bookmarks={bm.bookmarks}
                                    currentStep={step}
                                    onNavigate={navigateToBookmark}
                                    onUpdateBookmark={bm.updateBookmark}
                                />
                            </ErrorBoundary>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="game-state">
                        <Accordion.Control>Game State</Accordion.Control>
                        <Accordion.Panel>
                            <Grid>
                                <Grid.Col span={{ base: 12, md: 8 }}>
                                    <ErrorBoundary>
                                        <GameScreen data={data} />
                                    </ErrorBoundary>
                                </Grid.Col>
                                <Grid.Col span={{ base: 12, md: 4 }}>
                                    <ErrorBoundary>
                                        <GameMetrics data={data} />
                                    </ErrorBoundary>
                                </Grid.Col>
                            </Grid>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="log-messages">
                        <Accordion.Control>Log Messages</Accordion.Control>
                        <Accordion.Panel>
                            <ErrorBoundary>
                                <LogMessages data={data} />
                            </ErrorBoundary>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="intrinsics">
                        <Accordion.Control>Intrinsics & Significance</Accordion.Control>
                        <Accordion.Panel>
                            <Grid>
                                <Grid.Col span={{ base: 12, md: 6 }}>
                                    <ErrorBoundary>
                                        <IntrinsicsPanel data={data} />
                                    </ErrorBoundary>
                                </Grid.Col>
                                <Grid.Col span={{ base: 12, md: 6 }}>
                                    <ErrorBoundary>
                                        <IntrinsicsChart run={run} game={game || undefined} currentStep={step} onStepClick={handleChartStepClick} />
                                    </ErrorBoundary>
                                </Grid.Col>
                            </Grid>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="inventory">
                        <Accordion.Control>Inventory</Accordion.Control>
                        <Accordion.Panel>
                            <ErrorBoundary>
                                <InventoryPanel data={data} />
                            </ErrorBoundary>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="perception">
                        <Accordion.Control>Visual Perception</Accordion.Control>
                        <Accordion.Panel>
                            <Grid>
                                <Grid.Col span={{ base: 12, md: 4 }}>
                                    <ErrorBoundary>
                                        <FeatureTable data={data} />
                                    </ErrorBoundary>
                                </Grid.Col>
                                <Grid.Col span={{ base: 12, md: 8 }}>
                                    <ErrorBoundary>
                                        <ObjectInfo data={data} />
                                    </ErrorBoundary>
                                </Grid.Col>
                            </Grid>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="aural-perception">
                        <Accordion.Control>Aural Perception</Accordion.Control>
                        <Accordion.Panel>
                            <ErrorBoundary>
                                <AuralPerception data={data} />
                            </ErrorBoundary>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="attention">
                        <Accordion.Control>Visual Attention</Accordion.Control>
                        <Accordion.Panel>
                            <Grid>
                                <Grid.Col span={{ base: 12, md: 8 }}>
                                    <ErrorBoundary>
                                        <SaliencyMap data={data} />
                                    </ErrorBoundary>
                                </Grid.Col>
                                <Grid.Col span={{ base: 12, md: 4 }}>
                                    <ErrorBoundary>
                                        <AttenuationPanel data={data} />
                                        <div style={{ marginTop: 8 }}>
                                            <FocusPoints data={data} />
                                        </div>
                                    </ErrorBoundary>
                                </Grid.Col>
                            </Grid>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="object-resolution">
                        <Accordion.Control>
                            Object Resolution
                        </Accordion.Control>
                        <Accordion.Panel>
                            <Grid>
                                <Grid.Col span={{ base: 12, md: 6 }}>
                                    <ErrorBoundary>
                                        <ResolutionInspector data={data} />
                                    </ErrorBoundary>
                                </Grid.Col>
                                <Grid.Col span={{ base: 12, md: 6 }}>
                                    <ErrorBoundary>
                                        <ResolutionChart run={run} game={game || undefined} currentStep={step} onStepClick={handleChartStepClick} />
                                    </ErrorBoundary>
                                    <div style={{ marginTop: 8 }}>
                                        <ErrorBoundary>
                                            <EventSummary data={data} />
                                        </ErrorBoundary>
                                    </div>
                                </Grid.Col>
                            </Grid>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="all-objects">
                        <Accordion.Control>All Objects</Accordion.Control>
                        <Accordion.Panel>
                            <ErrorBoundary>
                                <AllObjects run={run} game={game || undefined} onStepClick={handleChartStepClick} />
                            </ErrorBoundary>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="transforms">
                        <Accordion.Control>Transforms & Prediction</Accordion.Control>
                        <Accordion.Panel>
                            <ErrorBoundary>
                                <TransformPanel data={data} />
                            </ErrorBoundary>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="actions">
                        <Accordion.Control>Actions</Accordion.Control>
                        <Accordion.Panel>
                            <ErrorBoundary>
                                <ActionPanel data={data} />
                            </ErrorBoundary>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="graph-events">
                        <Accordion.Control>Graph & Events</Accordion.Control>
                        <Accordion.Panel>
                            <Grid>
                                <Grid.Col span={{ base: 12, md: 6 }}>
                                    <ErrorBoundary>
                                        <GraphHistory run={run} game={game || undefined} currentStep={step} onStepClick={handleChartStepClick} />
                                    </ErrorBoundary>
                                </Grid.Col>
                                <Grid.Col span={{ base: 12, md: 6 }}>
                                    <ErrorBoundary>
                                        <EventHistory run={run} game={game || undefined} currentStep={step} onStepClick={handleChartStepClick} />
                                    </ErrorBoundary>
                                </Grid.Col>
                            </Grid>
                        </Accordion.Panel>
                    </Accordion.Item>
                </Accordion>
            </AppShell.Main>
        </AppShell>
    );
}
