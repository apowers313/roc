/** Main dashboard application layout. */

import {
    Accordion,
    Alert,
    AppShell,
    Button,
    Grid,
    Group,
    Text,
} from "@mantine/core";
import {
    Activity,
    ArrowRightLeft,
    Backpack,
    BarChart3,
    Bookmark,
    BrainCircuit,
    Database,
    Ear,
    Eye,
    Gamepad2,
    HeartPulse,
    Layers,
    MessageSquare,
    Network,
    ScanEye,
    Shapes,
    Bug,
    Table as TableIcon,
    Zap,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useQueryClient } from "@tanstack/react-query";

import { fetchStepRange } from "./api/client";
import { useStepData, useGames, useStepRange } from "./api/queries";
import { PopoutToolbar } from "./components/common/PopoutToolbar";
import { Section } from "./components/common/Section";
import { useHighlight } from "./state/highlight";
import { ActionHistogram } from "./components/panels/ActionHistogram";
import { ActionPanel } from "./components/panels/ActionPanel";
import { AllObjects } from "./components/panels/AllObjects";
import { AuralPerception } from "./components/panels/AuralPerception";
import { AttentionSpread } from "./components/panels/AttentionSpread";
import { AttentionCycleSummary, type CycleSummaryEntry } from "./components/panels/AttentionCycleSummary";
import { AttenuationPanel } from "./components/panels/AttenuationPanel";
import { BookmarkTable } from "./components/panels/BookmarkTable";
import { EventHistory } from "./components/panels/EventHistory";
import { EventSummary } from "./components/panels/EventSummary";
import { FeatureTable } from "./components/panels/FeatureTable";
import { FocusPoints } from "./components/panels/FocusPoints";
import { GameMetrics } from "./components/panels/GameMetrics";
import { GameScreen } from "./components/panels/GameScreen";
import { GraphHistory } from "./components/panels/GraphHistory";
import { GraphVisualization } from "./components/panels/GraphVisualization";
import { IntrinsicsChart } from "./components/panels/IntrinsicsChart";
import { IntrinsicsPanel } from "./components/panels/IntrinsicsPanel";
import { InventoryPanel } from "./components/panels/InventoryPanel";
import { LogMessages } from "./components/panels/LogMessages";
import { ObjectInfo } from "./components/panels/ObjectInfo";
import { PipelineStatus } from "./components/panels/PipelineStatus";
import { ResolutionChart } from "./components/panels/ResolutionChart";
import { ResolutionInspector } from "./components/panels/ResolutionInspector";
import { SaliencyMap } from "./components/panels/SaliencyMap";
import { SchemaPanel } from "./components/panels/SchemaPanel";
import { PredictionPanel } from "./components/panels/PredictionPanel";
import { SequencePanel } from "./components/panels/SequencePanel";
import { TransitionPanel } from "./components/panels/TransitionPanel";
import { StatusBar } from "./components/status/StatusBar";
import { BookmarkBar } from "./components/transport/BookmarkBar";
import { KeyboardHelp } from "./components/transport/KeyboardHelp";
import { MenuBar } from "./components/transport/MenuBar";
import { TransportBar } from "./components/transport/TransportBar";
import { useBookmarks } from "./hooks/useBookmarks";
import { useDebouncedValue } from "./hooks/useDebouncedValue";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { usePrefetchWindow } from "./hooks/usePrefetchWindow";
import { useRemoteLogger } from "./hooks/useRemoteLogger";
import { useRunSubscription, useGameState } from "./hooks/useRunSubscription";
import { useDashboard } from "./state/context";

/** Safely convert an unknown value to a display string, handling objects. */
function toDisplayString(value: unknown): string | undefined {
    if (value == null) return undefined;
    switch (typeof value) {
        case "string":
            return value;
        case "number":
        case "boolean":
        case "bigint":
            return String(value);
        case "object":
            return JSON.stringify(value);
        default:
            return JSON.stringify(value);
    }
}

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
        playing,
        setPlaying,
        autoFollow,
        setAutoFollow,
        speed,
        setSpeed,
    } = useDashboard();

    const queryClient = useQueryClient();

    // Game state is tracked via the Socket.io ``game_state_changed``
    // event. ``gameState`` carries the run name and active flag so the
    // auto-navigation effect can jump to a freshly started game.
    const gameState = useGameState();

    const {
        data: games,
        isError: gamesIsError,
        error: gamesError,
    } = useGames(run);
    const { data: stepRangeData } = useStepRange(run, game || undefined);

    // Debounce ``step`` for the heaviest panel (GraphVisualization).
    // That panel is wrapped in ``React.memo`` so a stable prop value
    // here skips its re-render during rapid scrubbing. The 200ms window
    // is empirically the smallest that prevents piling up useless
    // Cytoscape reconcile passes without making the graph feel
    // disconnected from the slider.
    const debouncedGraphStep = useDebouncedValue(step, 200);

    // Debounced step for history panels that only use ``currentStep``
    // to draw a marker/reference line (IntrinsicsChart, GraphHistory,
    // EventHistory, ResolutionChart). A 150ms window is imperceptible
    // for marker movement and cuts recharts reconciliation on rapid
    // scrubbing -- the bulk of the TC-PERF-001 budget was being spent
    // re-rendering these charts once per step when they only need the
    // final resting position.
    const debouncedHistoryStep = useDebouncedValue(step, 150);

    // Per-game step ranges, derived from the contiguous-step-numbering
    // convention (game N starts where game N-1 ended). Used by AllObjects
    // to attribute an object's canonical step_added to its origin game
    // when the user is filtering by a different game (BUG-M2).
    const gameStepRanges = useMemo(() => {
        if (!games) return [];
        const ranges: { game_number: number; min: number; max: number }[] = [];
        let cursor = 0;
        for (const g of games) {
            ranges.push({
                game_number: g.game_number,
                min: cursor + 1,
                max: cursor + g.steps,
            });
            cursor += g.steps;
        }
        return ranges;
    }, [games]);

    // Track whether we've auto-selected the live run
    const liveRunSelected = useRef(false);
    // Track the run name from the initial URL (if any). Used to avoid
    // overriding explicit URL navigation to a specific run/step, while
    // still allowing auto-navigation to a NEW game the user starts.
    // Declared up here (instead of further down) because
    // ``handleBrowseRuns`` clears it -- the callback's TDZ requires
    // the binding to be initialized before the useCallback runs.
    const initialUrlRun = useRef<string | null>(
        new URLSearchParams(globalThis.location.search).get("run"),
    );
    // BUG-M4: state for the bookmark guard. The guard itself uses
    // ``stepDataReadyRef`` (declared below alongside ``stepDataReady``)
    // because TanStack Query's ``placeholderData: keepPreviousData``
    // means raw ``data`` can be the previous run's data while a new
    // query is loading or errored -- so a "data != null" check would
    // pass even on a broken run. ``stepDataReady`` accounts for both
    // undefined data AND placeholder data, so it is the correct gate.
    const [bookmarkError, setBookmarkError] = useState<string | null>(null);

    // BUG-C2 follow-up (T2.6): when an explicit URL run cannot be
    // loaded (e.g. /games returns 500 or 404), show a banner so the
    // user knows WHY the dashboard is empty and has an escape hatch
    // back to the run picker. URL sovereignty preserves the broken run
    // name in the URL; this banner makes the failure visible instead
    // of showing a confusingly-empty dashboard. The "Browse runs"
    // button clears the URL and resets the context to its empty state
    // so the auto-select-first-run effect can take over.
    //
    // Empty-catalog case (post-Phase-7): some runs are on disk with
    // status="ok" in /api/runs but their DuckLake catalog has zero
    // tables (e.g., the writer crashed before any data was emitted).
    // For those, /games returns 200 with [] and /step-range returns
    // {min:0,max:0}. The Mantine Game Select with an empty data array
    // is non-interactive (clicks do nothing), so the user is stranded
    // -- they think the dropdown is broken. Surface the same banner so
    // the user can navigate away.
    const isEmptyCatalogRun =
        run !== "" &&
        games !== undefined &&
        games.length === 0 &&
        stepRangeData !== undefined &&
        stepRangeData.max === 0;
    const showLoadFailureBanner = (gamesIsError && run !== "") || isEmptyCatalogRun;
    const loadFailureMessage =
        gamesError instanceof Error
            ? gamesError.message
            : gamesError != null
                ? String(gamesError)
                : isEmptyCatalogRun
                    ? "This run has no recorded steps (empty catalog)."
                    : "";
    const handleBrowseRuns = useCallback(() => {
        // Clear the URL and reset context state. The auto-select-first
        // -run effect picks up the empty ``run`` and lands the user on
        // ``runs[0]`` automatically. Clearing ``initialUrlRun.current``
        // also re-enables the auto-select-live-run effect (its guard
        // would otherwise still hold the stale broken-URL run name and
        // block live-run auto-selection forever in this session).
        globalThis.history.replaceState(null, "", globalThis.location.pathname);
        initialUrlRun.current = null;
        setRun("");
        setGame(1);
        setStep(1);
    }, [setRun, setGame, setStep]);

    // Phase 4: subscribe to ``step_added`` invalidation events for the
    // current run. This is the single point where Socket.io meets
    // TanStack Query -- when the writer pushes a new step, the server
    // emits a tiny ``{run, step}`` payload to this client and the
    // matching step-range query invalidates. The slider, StatusBar,
    // and any other consumers re-render against the fresh range.
    useRunSubscription(run);

    // Clear point highlights when step changes
    const { clear: clearHighlights } = useHighlight();
    useEffect(() => {
        clearHighlights();
    }, [step, clearHighlights]);

    useRemoteLogger();

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

    // Phase 4: liveData/onNewStep/refs deleted. The data path now flows
    // through TanStack Query exclusively -- ``useStepData`` for the
    // current step, ``useStepRange`` for the slider range. Socket.io
    // ``step_added`` events invalidate those queries via
    // ``useRunSubscription``, which triggers a refetch.

    // Phase 5: the auto-follow effect now drives off the two-boolean
    // playback model. When the run is tail_growing AND autoFollow is
    // on, advance the step cursor to the new max as the range grows.
    // `autoFollow` defaults to true for new live runs and flips to
    // false on any explicit user navigation. Clicking GO LIVE flips
    // it back to true and snaps the cursor to the head.
    //
    // TransportBar is the single owner of ``setStepRange`` -- it syncs
    // the query data to context on every update. This effect only
    // advances ``step``; it never writes ``stepRange``. Keeping the
    // writes separate avoids a dead-band where a stale context
    // ``stepMax`` pins the slider behind the true head when the user
    // has auto-follow off (TC-GAME-004 regression).
    const tailGrowing = stepRangeData?.tail_growing ?? false;
    const restMax = stepRangeData?.max ?? 0;
    useEffect(() => {
        if (!tailGrowing) return;
        if (!autoFollow) return;
        if (restMax <= 0) return;
        if (step !== restMax) {
            setStep(restMax);
        }
    }, [tailGrowing, autoFollow, restMax, step, setStep]);

    const connected = true; // Socket.io connection managed by useRunSubscription

    // Keyboard shortcut handlers
    const stepForward = useCallback(() => {
        setStep((prev) => (prev < stepMax ? prev + 1 : prev));
        setAutoFollow(false);
    }, [stepMax, setStep, setAutoFollow]);

    const stepBack = useCallback(() => {
        setStep((prev) => (prev > stepMin ? prev - 1 : prev));
        setAutoFollow(false);
    }, [stepMin, setStep, setAutoFollow]);

    const jumpToStart = useCallback(() => {
        setStep(stepMin);
        setAutoFollow(false);
    }, [stepMin, setStep, setAutoFollow]);

    const jumpToEnd = useCallback(() => {
        setStep(stepMax);
        // Pure navigation -- just go to end of current game's range.
        // Use "L" or click "GO LIVE" badge to return to live-following.
        setAutoFollow(false);
    }, [stepMax, setStep, setAutoFollow]);

    const togglePlay = useCallback(() => {
        setPlaying(!playing);
    }, [playing, setPlaying]);

    const stepForward10 = useCallback(() => {
        setStep((prev) => Math.min(prev + 10, stepMax));
        setAutoFollow(false);
    }, [stepMax, setStep, setAutoFollow]);

    const stepBack10 = useCallback(() => {
        setStep((prev) => Math.max(prev - 10, stepMin));
        setAutoFollow(false);
    }, [stepMin, setStep, setAutoFollow]);

    const toggleBookmark = useCallback(() => {
        // BUG-M4: do not bookmark when no real step data is loaded. The
        // user sees a transient notification instead of a phantom
        // bookmark. ``stepDataReadyRef`` is the right gate (not raw
        // ``data``): TanStack Query's ``keepPreviousData`` means ``data``
        // can be a stale placeholder from a previous run while the new
        // run's query is loading or errored, so a plain null check would
        // miss the broken-run case.
        if (!stepDataReadyRef.current) {
            setBookmarkError("Cannot bookmark: no step loaded");
            return;
        }
        bm.toggleBookmark(step, game);
    }, [bm, step, game]);

    // Auto-clear the bookmark error after a few seconds so the user
    // does not see a stale notification stuck on screen.
    useEffect(() => {
        if (bookmarkError == null) return;
        const t = globalThis.setTimeout(() => setBookmarkError(null), 3000);
        return () => globalThis.clearTimeout(t);
    }, [bookmarkError]);

    // Navigate to a bookmark -- switches game and fetches step range if needed.
    const navigateToBookmark = useCallback(
        (bookmark: { step: number; game: number }) => {
            if (bookmark.game !== game) {
                setGame(bookmark.game);
                setAutoFollow(false);
                void queryClient
                    .fetchQuery({
                        queryKey: ["step-range", run, bookmark.game],
                        queryFn: () => fetchStepRange(run, bookmark.game),
                    })
                    .then((d) => {
                        if (d.max > 0) {
                            setStepRange(d.min, d.max);
                        }
                    })
                    .catch(() => {});
            }
            setStep(bookmark.step);
        },
        [game, run, setGame, setStep, setStepRange, setAutoFollow, queryClient],
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
    // Flip autoFollow on and snap step to the current max so the
    // StatusBar and slider catch up immediately. The auto-follow
    // effect then owns subsequent step advancement as range.max grows.
    const goLive = useCallback(() => {
        const runName = gameState?.run_name;
        if (!runName || gameState?.state !== "running") return;
        setRun(runName);
        setAutoFollow(true);
        if (restMax > 0) {
            setStep(restMax);
        }
    }, [gameState, setRun, setAutoFollow, restMax, setStep]);

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
        setAutoFollow(false);
        // Fetch step range for the new game through the TanStack Query cache
        // so ``useStepRange`` sees a cache hit on its next render.
        void queryClient
            .fetchQuery({
                queryKey: ["step-range", run, nextGame],
                queryFn: () => fetchStepRange(run, nextGame),
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
    }, [games, game, setGame, run, setStep, setStepRange, setAutoFollow, queryClient]);

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

    // Auto-select the live run when a game starts.
    // Sets the run and game to match the live session and flips
    // autoFollow on so the inline auto-follow effect advances the
    // step cursor as range.max grows.
    //
    // URL parameter sovereignty (BUG-C2): if the user navigated to
    // ``?run=X`` via URL, this effect must NEVER auto-navigate to a
    // different run, even if X's endpoints fail or a different live
    // game appears in the background. The user explicitly asked to look
    // at X; silently teleporting them away violates the documented
    // sovereignty invariant in ``dashboard-ui/CLAUDE.md``. Auto-select
    // is only allowed when the URL has no explicit run, OR when the
    // live run happens to be the same one the URL points at (in which
    // case "auto-select" just confirms the user's choice and lets us
    // flip autoFollow on).
    const prevLiveRunName = useRef<string | null>(null);
    useEffect(() => {
        if (!gameState || gameState.state !== "running" || !gameState.run_name) return;

        const isNewRun = gameState.run_name !== prevLiveRunName.current;
        prevLiveRunName.current = gameState.run_name;

        // URL sovereignty guard: bail entirely when an explicit URL run
        // exists and points at a *different* run from the live session.
        // We still allow the (URL run === live run) case so GO LIVE can
        // resume on the user's explicitly-chosen run.
        if (
            initialUrlRun.current &&
            gameState.run_name !== initialUrlRun.current
        ) {
            return;
        }

        // Auto-navigate when a new live run appears, OR on first load
        // when the URL has no explicit run. ``isNewRun`` covers the
        // "user started a new game during the session" case; the second
        // clause covers the bare-page first-load case.
        const shouldAutoNavigate =
            isNewRun || (!liveRunSelected.current && !initialUrlRun.current);

        if (shouldAutoNavigate) {
            liveRunSelected.current = true;
            setRun(gameState.run_name);
            setGame(1);
            setStep(1);
            setAutoFollow(true);
        }
    }, [gameState, setRun, setGame, setStep, setAutoFollow]);

    // REST data fetch. No debounce: with keepPreviousData the previous
    // step stays visible while the next one loads, so rapid step changes
    // don't cause flicker. DuckLake queries complete in ~8ms, well within
    // the playback interval.
    const {
        data: restData,
        isLoading,
        isPlaceholderData,
        isError: stepIsError,
        error: stepError,
    } = useStepData(run, step, game || undefined);

    // Surface fetch errors to the remote logger so failures show up in
    // the iPad debug session even when the user can't see the dev tools.
    // This is the trace for the recurring "errors not visible in UI"
    // class of bug -- the dashboard was silently absorbing 500s from
    // /api/runs/{run}/step/{n} and rendering "no data" indistinguishably
    // from a real missing-step state.
    useEffect(() => {
        if (stepIsError && stepError) {
            // eslint-disable-next-line no-console
            console.error(
                "[Step] fetch failed",
                JSON.stringify({
                    run,
                    step,
                    game,
                    message:
                        stepError instanceof Error
                            ? stepError.message
                            : String(stepError),
                }),
            );
        }
    }, [stepIsError, stepError, run, step, game]);

    // Signal to the play timer that the current step's real data has
    // arrived (not just placeholder from the previous step).
    const stepDataReady = restData !== undefined && !isPlaceholderData;
    const stepDataReadyRef = useRef(stepDataReady);
    stepDataReadyRef.current = stepDataReady;

    // Phase 4: only one data source -- REST via TanStack Query. The
    // ``useRunSubscription`` hook invalidates the relevant queries when
    // a step_added event arrives, triggering a fresh fetch through the
    // unified RunReader path.
    const data = restData;

    // Attention cycle stepper state -- reset when step changes
    const [selectedSaliencyCycle, setSelectedSaliencyCycle] = useState(0);
    useEffect(() => { setSelectedSaliencyCycle(0); }, [step]);

    // Build cycle summary entries from saliency_cycles data
    const cycleSummaryEntries: CycleSummaryEntry[] = useMemo(() => {
        const cycles = data?.saliency_cycles;
        if (!cycles) return [];
        return cycles.map((c) => ({
            preIorPeak: c.pre_ior_peak ?? { x: 0, y: 0, strength: 0 },
            postIorPeak: c.post_ior_peak ?? { x: 0, y: 0, strength: 0 },
            focusedPoint: c.focused_point ?? { x: 0, y: 0, strength: 0 },
        }));
    }, [data?.saliency_cycles]);

    // Click-to-navigate handler for history charts
    const handleChartStepClick = useCallback(
        (clickedStep: number) => {
            setStep(clickedStep);
            setAutoFollow(false);
        },
        [setStep, setAutoFollow],
    );

    usePrefetchWindow(run, step, stepMin, stepMax, game || undefined);

    return (
        <AppShell header={{ height: 140 }} padding="xs">
            <AppShell.Header>
                <MenuBar />
                <TransportBar connected={connected} stepDataReadyRef={stepDataReadyRef} />
                <BookmarkBar
                    bookmarks={bm.bookmarks}
                    currentStep={step}
                    stepMin={stepMin}
                    stepMax={stepMax}
                    isBookmarked={bm.isBookmarked(step)}
                    onToggle={toggleBookmark}
                    onNavigate={navigateToBookmark}
                />
            </AppShell.Header>

            <KeyboardHelp opened={helpOpen} onClose={() => setHelpOpen(false)} />

            <AppShell.Main>
                {showLoadFailureBanner && (
                    <Alert
                        color="red"
                        variant="filled"
                        radius={0}
                        mb={4}
                        title={`Run "${run}" could not be loaded`}
                    >
                        <Group justify="space-between" align="center">
                            <Text size="xs">
                                {loadFailureMessage || "Unknown error"}
                            </Text>
                            <Button
                                size="xs"
                                color="red"
                                variant="white"
                                onClick={handleBrowseRuns}
                            >
                                Browse runs
                            </Button>
                        </Group>
                    </Alert>
                )}
                {bookmarkError && (
                    <Alert
                        color="yellow"
                        variant="filled"
                        radius={0}
                        mb={4}
                    >
                        <Text size="xs">{bookmarkError}</Text>
                    </Alert>
                )}
                <StatusBar
                    data={data}
                    autoFollow={autoFollow}
                    onGoLive={goLive}
                    fetchError={stepIsError ? stepError : null}
                />

                {isLoading && !data && (
                    <Text size="xs" c="dimmed" p="md">
                        Loading...
                    </Text>
                )}

                <PopoutToolbar>
                    <PopoutToolbar.Button title="All Objects" icon={TableIcon}>
                        <AllObjects
                            run={run}
                            game={game || undefined}
                            gameStepRanges={gameStepRanges}
                            onStepClick={handleChartStepClick}
                        />
                    </PopoutToolbar.Button>
                    <PopoutToolbar.Button title="Graph & Events" icon={BarChart3}>
                        <GraphHistory run={run} game={game || undefined} currentStep={debouncedHistoryStep} onStepClick={handleChartStepClick} />
                        <div style={{ marginTop: 16 }}>
                            <EventHistory run={run} game={game || undefined} currentStep={debouncedHistoryStep} onStepClick={handleChartStepClick} />
                        </div>
                    </PopoutToolbar.Button>
                </PopoutToolbar>

                <Accordion
                    multiple
                    value={openSections}
                    onChange={handleAccordionChange}
                    variant="separated"
                >
                    <Section value="pipeline" title="Pipeline Status" icon={Activity} color="gray">
                        <PipelineStatus data={data} />
                    </Section>

                    <Section value="bookmarks" title="Bookmarks" icon={Bookmark} color="gray">
                        <BookmarkTable
                            bookmarks={bm.bookmarks}
                            currentStep={step}
                            onNavigate={navigateToBookmark}
                            onUpdateBookmark={bm.updateBookmark}
                        />
                    </Section>

                    <Section value="game-state" title="Game State" icon={Gamepad2} color="gray" toolbar={
                        <>
                            <PopoutToolbar.Button title="Graph & Events" icon={BarChart3}>
                                <GraphHistory run={run} game={game || undefined} currentStep={debouncedHistoryStep} onStepClick={handleChartStepClick} />
                                <div style={{ marginTop: 16 }}>
                                    <EventHistory run={run} game={game || undefined} currentStep={debouncedHistoryStep} onStepClick={handleChartStepClick} />
                                </div>
                            </PopoutToolbar.Button>
                            <PopoutToolbar.Button title="Schema" icon={Database}>
                                <SchemaPanel run={run} />
                            </PopoutToolbar.Button>
                        </>
                    }>
                        <Grid>
                            <Grid.Col span={{ base: 12, md: 8 }}>
                                <GameScreen data={data} />
                            </Grid.Col>
                            <Grid.Col span={{ base: 12, md: 4 }}>
                                <GameMetrics data={data} />
                            </Grid.Col>
                        </Grid>
                    </Section>

                    <Section value="log-messages" title="Log Messages" icon={MessageSquare} color="gray">
                        <LogMessages data={data} />
                    </Section>

                    <Section value="intrinsics" title="Intrinsics & Significance" icon={HeartPulse} color="orange">
                        <Grid>
                            <Grid.Col span={{ base: 12, md: 6 }}>
                                <IntrinsicsPanel data={data} />
                            </Grid.Col>
                            <Grid.Col span={{ base: 12, md: 6 }}>
                                <IntrinsicsChart run={run} game={game || undefined} currentStep={debouncedHistoryStep} onStepClick={handleChartStepClick} />
                            </Grid.Col>
                        </Grid>
                    </Section>

                    <Section value="inventory" title="Inventory" icon={Backpack} color="yellow">
                        <InventoryPanel data={data} />
                    </Section>

                    <Section value="perception" title="Visual Perception" icon={Eye} color="teal">
                        <Grid>
                            <Grid.Col span={{ base: 12, md: 4 }}>
                                <FeatureTable data={data} />
                            </Grid.Col>
                            <Grid.Col span={{ base: 12, md: 8 }}>
                                <ObjectInfo data={data} />
                            </Grid.Col>
                        </Grid>
                    </Section>

                    <Section value="aural-perception" title="Aural Perception" icon={Ear} color="teal">
                        <AuralPerception data={data} />
                    </Section>

                    <Section value="attention" title="Visual Attention" icon={ScanEye} color="violet" expmod={toDisplayString(data?.attenuation?.flavor)}>
                        <AttentionSpread data={data} />
                        {cycleSummaryEntries.length > 1 && (
                            <AttentionCycleSummary
                                cycles={cycleSummaryEntries}
                                selectedCycle={selectedSaliencyCycle}
                                onCycleChange={setSelectedSaliencyCycle}
                            />
                        )}
                        <Grid>
                            <Grid.Col span={{ base: 12, md: 8 }}>
                                <SaliencyMap
                                    data={data}
                                    cycleIndex={cycleSummaryEntries.length > 1 ? selectedSaliencyCycle : undefined}
                                />
                            </Grid.Col>
                            <Grid.Col span={{ base: 12, md: 4 }}>
                                <AttenuationPanel data={data} />
                                <div style={{ marginTop: 8 }}>
                                    <FocusPoints data={data} />
                                </div>
                            </Grid.Col>
                        </Grid>
                    </Section>

                    <Section value="object-resolution" title="Object Resolution" icon={Shapes} color="violet" expmod={toDisplayString(data?.resolution_metrics?.algorithm)} toolbar={
                        <>
                            <PopoutToolbar.Button title="All Objects" icon={TableIcon}>
                                <AllObjects
                                    run={run}
                                    game={game || undefined}
                                    gameStepRanges={gameStepRanges}
                                    onStepClick={handleChartStepClick}
                                />
                            </PopoutToolbar.Button>
                            <PopoutToolbar.Button title="Resolution Error Rate" icon={Bug}>
                                <ResolutionChart run={run} game={game || undefined} currentStep={debouncedHistoryStep} onStepClick={handleChartStepClick} />
                            </PopoutToolbar.Button>
                        </>
                    }>
                        <ResolutionInspector data={data} />
                        <div style={{ marginTop: 8 }}>
                            <EventSummary data={data} />
                        </div>
                    </Section>

                    <Section value="graph-visualization" title="Graph Visualization" icon={Network} color="cyan">
                        <GraphVisualization run={run} step={debouncedGraphStep} game={game || undefined} />
                    </Section>

                    <Section value="sequences" title="Sequences" icon={Layers} color="indigo">
                        <SequencePanel data={data} />
                    </Section>

                    <Section value="transitions" title="Transitions" icon={ArrowRightLeft} color="orange">
                        <TransitionPanel data={data} onStepClick={handleChartStepClick} />
                    </Section>

                    <Section value="prediction" title="Prediction" icon={BrainCircuit} color="cyan" expmod={[data?.prediction?.candidate_expmod, data?.prediction?.confidence_expmod].filter((v): v is string => v != null)}>
                        <PredictionPanel data={data} />
                    </Section>

                    <Section value="actions" title="Actions" icon={Zap} color="orange" expmod={data?.action_taken?.expmod_name} toolbar={
                        <PopoutToolbar.Button title="Action Histogram" icon={BarChart3}>
                            <ActionHistogram run={run} game={game || undefined} />
                        </PopoutToolbar.Button>
                    }>
                        <ActionPanel data={data} />
                    </Section>

                </Accordion>
            </AppShell.Main>
        </AppShell>
    );
}
