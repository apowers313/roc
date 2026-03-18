/** Main dashboard application layout. */

import {
    Accordion,
    AppShell,
    Grid,
    Text,
} from "@mantine/core";
import { useCallback, useEffect, useRef, useState } from "react";

import { usePrefetchAdjacentSteps, useStepData } from "./api/queries";
import { KVTable } from "./components/common/KVTable";
import { EventSummary } from "./components/panels/EventSummary";
import { FeatureTable } from "./components/panels/FeatureTable";
import { FocusPoints } from "./components/panels/FocusPoints";
import { GameMetrics } from "./components/panels/GameMetrics";
import { GameScreen } from "./components/panels/GameScreen";
import { GraphSummary } from "./components/panels/GraphSummary";
import { LogMessages } from "./components/panels/LogMessages";
import { ObjectInfo } from "./components/panels/ObjectInfo";
import { SaliencyMap } from "./components/panels/SaliencyMap";
import { StatusBar } from "./components/status/StatusBar";
import { TransportBar } from "./components/transport/TransportBar";
import { useLiveUpdates } from "./hooks/useLiveUpdates";
import { useDashboard } from "./state/context";
import type { StepData } from "./types/step-data";

export function App() {
    const {
        run,
        setRun,
        step,
        setStep,
        game,
        stepMax,
        setStepRange,
        playback,
        dispatchPlayback,
        liveRunName,
        setLiveRunName,
    } = useDashboard();

    const isFollowing = playback === "live_following";

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

            // Only update step range and advance when viewing the live run
            // without a game filter (or with the matching game selected).
            const isViewingLiveRun =
                runRef.current === liveRunNameRef.current;
            const gameMatches =
                gameRef.current === 0 ||
                gameRef.current === pushData.game_number;

            if (isViewingLiveRun && gameMatches) {
                if (gameRef.current === 0) {
                    // No game filter: global range
                    setStepRange(1, pushData.step);
                }
                // With a game filter the REST step-range query is authoritative
            }

            if (isFollowingRef.current && isViewingLiveRun) {
                // In live-following mode, advance the step cursor
                setStep(pushData.step);
            } else if (isViewingLiveRun) {
                // Notify the playback state machine
                const atEdge = stepRef.current >= stepMaxRef.current;
                dispatchPlayback({ type: "PUSH_ARRIVED", atEdge });
            }
        },
        [setStep, setStepRange, dispatchPlayback],
    );

    const { connected, liveStatus } = useLiveUpdates({ onNewStep });

    // Track the live run name from status polls
    useEffect(() => {
        if (liveStatus?.active && liveStatus.run_name) {
            setLiveRunName(liveStatus.run_name);
        }
    }, [liveStatus, setLiveRunName]);

    // Auto-select the live run when a game starts
    useEffect(() => {
        if (
            liveStatus?.active &&
            liveStatus.run_name &&
            !liveRunSelected.current
        ) {
            liveRunSelected.current = true;
            setRun(liveStatus.run_name);
            setStepRange(liveStatus.step_min, liveStatus.step_max);
            setStep(liveStatus.step_max);
            dispatchPlayback({ type: "GO_LIVE" });
        }
    }, [liveStatus, setRun, setStepRange, setStep, dispatchPlayback]);

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

    usePrefetchAdjacentSteps(run, step, game || undefined);

    return (
        <AppShell header={{ height: 100 }} padding="xs">
            <AppShell.Header>
                <TransportBar connected={connected} stepDataReadyRef={stepDataReadyRef} />
            </AppShell.Header>

            <AppShell.Main>
                <StatusBar data={data} playbackState={playback} />

                {isLoading && !data && (
                    <Text size="xs" c="dimmed" p="md">
                        Loading...
                    </Text>
                )}

                <Accordion
                    multiple
                    defaultValue={["game-state", "perception"]}
                    variant="separated"
                >
                    <Accordion.Item value="game-state">
                        <Accordion.Control>Game State</Accordion.Control>
                        <Accordion.Panel>
                            <Grid>
                                <Grid.Col span={8}>
                                    <GameScreen data={data} />
                                </Grid.Col>
                                <Grid.Col span={4}>
                                    <GameMetrics data={data} />
                                </Grid.Col>
                            </Grid>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="perception">
                        <Accordion.Control>Perception</Accordion.Control>
                        <Accordion.Panel>
                            <Grid>
                                <Grid.Col span={4}>
                                    <FeatureTable data={data} />
                                </Grid.Col>
                                <Grid.Col span={8}>
                                    <ObjectInfo data={data} />
                                </Grid.Col>
                            </Grid>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="attention">
                        <Accordion.Control>Attention</Accordion.Control>
                        <Accordion.Panel>
                            <Grid>
                                <Grid.Col span={8}>
                                    <SaliencyMap data={data} />
                                </Grid.Col>
                                <Grid.Col span={4}>
                                    <KVTable
                                        data={data?.attenuation}
                                        emptyText="No attenuation data"
                                        title="Attenuation"
                                    />
                                    <div style={{ marginTop: 8 }}>
                                        <FocusPoints data={data} />
                                    </div>
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
                                <Grid.Col span={4}>
                                    <KVTable
                                        data={data?.resolution_metrics}
                                        emptyText="No resolution data"
                                        title="Resolution"
                                    />
                                </Grid.Col>
                                <Grid.Col span={4}>
                                    <GraphSummary data={data} />
                                </Grid.Col>
                                <Grid.Col span={4}>
                                    <EventSummary data={data} />
                                </Grid.Col>
                            </Grid>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="log-messages">
                        <Accordion.Control>Log Messages</Accordion.Control>
                        <Accordion.Panel>
                            <LogMessages data={data} />
                        </Accordion.Panel>
                    </Accordion.Item>
                </Accordion>
            </AppShell.Main>
        </AppShell>
    );
}
