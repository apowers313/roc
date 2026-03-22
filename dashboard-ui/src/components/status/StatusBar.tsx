/** Status bar -- compact row of key game metrics + live badge. */

import { Badge, Group, Progress, Text } from "@mantine/core";

import { useDashboard } from "../../state/context";
import type { StepData } from "../../types/step-data";

interface StatusBarProps {
    data: StepData | undefined;
    playbackState: string;
    onGoLive?: () => void;
}

function hpColor(hp: number, hpMax: number): string {
    if (hpMax <= 0) return "gray";
    const ratio = hp / hpMax;
    if (ratio > 0.5) return "green";
    if (ratio > 0.25) return "yellow";
    return "red";
}

export function StatusBar({ data, playbackState, onGoLive }: StatusBarProps) {
    const { liveGameActive } = useDashboard();
    const metrics = data?.game_metrics;
    const isLive = playbackState === "live_following";
    // Show GO LIVE whenever a live game is running and we're not following it.
    // This covers historical mode (user navigated away), live_paused, and
    // live_catchup -- any state where clicking would return to live.
    const canGoLive = !isLive && liveGameActive;

    return (
        <Group gap="md" px={8} py={4}>
            {isLive && (
                <Badge color="red" size="xs" variant="filled">
                    LIVE
                </Badge>
            )}
            {canGoLive && (
                <Badge
                    color="yellow"
                    size="xs"
                    variant="filled"
                    style={{ cursor: "pointer" }}
                    onClick={onGoLive}
                >
                    GO LIVE
                </Badge>
            )}

            {metrics ? (
                <>
                    <Group gap={4}>
                        <Text size="xs" c="dimmed">
                            HP
                        </Text>
                        <Progress
                            value={
                                ((metrics.hp as number) /
                                    Math.max(
                                        metrics.hp_max as number,
                                        1,
                                    )) *
                                100
                            }
                            color={hpColor(
                                metrics.hp as number,
                                metrics.hp_max as number,
                            )}
                            size="xs"
                            style={{ width: 60 }}
                        />
                        <Text size="xs" fw={500}>
                            {String(metrics.hp)}/{String(metrics.hp_max)}
                        </Text>
                    </Group>
                    <StatItem label="Score" value={metrics.score} />
                    <StatItem label="Depth" value={metrics.depth} />
                    <StatItem label="Gold" value={metrics.gold} />
                    <StatItem label="Energy" value={metrics.energy} />
                    <StatItem label="Hunger" value={metrics.hunger} />
                </>
            ) : (
                <Text size="xs" c="dimmed">
                    Step {data?.step ?? "--"} | Game{" "}
                    {data?.game_number ?? "--"}
                </Text>
            )}
        </Group>
    );
}

function StatItem({
    label,
    value,
}: {
    label: string;
    value: unknown;
}) {
    return (
        <Group gap={4}>
            <Text size="xs" c="dimmed">
                {label}
            </Text>
            <Text size="xs" fw={500}>
                {value != null ? String(value) : "--"}
            </Text>
        </Group>
    );
}
