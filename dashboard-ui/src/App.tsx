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
import { FeatureTable } from "./components/panels/FeatureTable";
import { GameMetrics } from "./components/panels/GameMetrics";
import { GameScreen } from "./components/panels/GameScreen";
import { LogMessages } from "./components/panels/LogMessages";
import { SaliencyMap } from "./components/panels/SaliencyMap";
import { StatusBar } from "./components/status/StatusBar";
import { TransportBar } from "./components/transport/TransportBar";
import { useDebouncedValue } from "./hooks/useDebouncedValue";
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
    } = useDashboard();

    const isFollowing = playback === "live_following";

    // Live push data -- always updated from Socket.io push, regardless of
    // playback mode. This eliminates the race where isFollowingRef is stale
    // and liveData never gets populated.
    const [liveData, setLiveData] = useState<StepData | null>(null);

    // Track whether we've auto-selected the live run
    const liveRunSelected = useRef(false);

    // Use refs to avoid stale closures in the Socket.io callback
    const stepRef = useRef(step);
    stepRef.current = step;
    const stepMaxRef = useRef(stepMax);
    stepMaxRef.current = stepMax;
    const isFollowingRef = useRef(isFollowing);
    isFollowingRef.current = isFollowing;

    const onNewStep = useCallback(
        (pushData: StepData) => {
            // Always expand step range and store the latest push data
            setStepRange(1, pushData.step);
            setLiveData(pushData);

            if (isFollowingRef.current) {
                // In live-following mode, advance the step cursor
                setStep(pushData.step);
            } else {
                // Notify the playback state machine
                const atEdge = stepRef.current >= stepMaxRef.current;
                dispatchPlayback({ type: "PUSH_ARRIVED", atEdge });
            }
        },
        [setStep, setStepRange, dispatchPlayback],
    );

    const { connected, liveStatus } = useLiveUpdates({ onNewStep });

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

    // REST data is always fetched as a fallback. In live-following mode we
    // prefer the push data (instant, no round-trip) but REST is never disabled
    // so there's always a data source available.
    const debouncedStep = useDebouncedValue(step, 150);
    const { data: restData, isLoading } = useStepData(
        run,
        debouncedStep,
        game || undefined,
    );

    // In live-following mode, prefer push data (instant) with REST fallback.
    // In historical/paused mode, use ONLY REST data -- never fall back to
    // liveData, which would cause a flicker of the live frame while the
    // historical step loads.
    const data = isFollowing ? liveData ?? restData : restData;

    usePrefetchAdjacentSteps(run, debouncedStep, game || undefined);

    return (
        <AppShell header={{ height: 100 }} padding="xs">
            <AppShell.Header>
                <TransportBar connected={connected} />
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
                            <FeatureTable data={data} />
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
                                </Grid.Col>
                            </Grid>
                        </Accordion.Panel>
                    </Accordion.Item>

                    <Accordion.Item value="object-resolution">
                        <Accordion.Control>
                            Object Resolution
                        </Accordion.Control>
                        <Accordion.Panel>
                            <KVTable
                                data={data?.resolution_metrics}
                                emptyText="No resolution data"
                                title="Resolution"
                            />
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
