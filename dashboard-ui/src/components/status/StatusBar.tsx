/** Status bar -- compact row of key game metrics + live badge. */

import { Badge, Group, Progress, Text, Tooltip } from "@mantine/core";

import { useStepRange } from "../../api/queries";
import { useDashboard } from "../../state/context";
import type { StepData } from "../../types/step-data";

interface StatusBarProps {
    data: StepData | undefined;
    autoFollow: boolean;
    onGoLive?: () => void;
    /**
     * Error from the most recent step fetch, if any. Surfaced as a
     * red ERROR badge so failed REST calls are not silently rendered
     * as "no data" -- the recurring "errors not visible in the UI"
     * complaint. Pass `null` (or omit) when there is no error.
     */
    fetchError?: unknown;
}

function hpColor(hp: number, hpMax: number): string {
    if (hpMax <= 0) return "gray";
    const ratio = hp / hpMax;
    if (ratio > 0.5) return "green";
    if (ratio > 0.25) return "yellow";
    return "red";
}

export function StatusBar({
    data,
    autoFollow,
    onGoLive,
    fetchError,
}: Readonly<StatusBarProps>) {
    // tail_growing on the current run's step-range is the ONLY
    // liveness signal. autoFollow determines whether we render the
    // LIVE vs GO LIVE badge.
    const { run } = useDashboard();
    const { data: stepRange } = useStepRange(run);
    const tailGrowing = stepRange?.tail_growing ?? false;
    const metrics = data?.game_metrics;
    // LIVE badge shows when the run is tail-growing AND we are
    // following it. GO LIVE shows when the run is tail-growing but
    // the user has navigated away (autoFollow=false).
    const isLive = tailGrowing && autoFollow;
    const canGoLive = tailGrowing && !autoFollow;
    // The API returns game_number=0 with screen=null when no row exists for
    // the requested step (out-of-range URL/chart-click navigation). Surface
    // this as an explicit "no data" state instead of "Step N | Game 0",
    // which looks like a real reading and confuses users into thinking the
    // game has more steps than it does.
    const isMissingData =
        data != null && data.game_number === 0 && data.screen == null;
    const errorMessage =
        fetchError instanceof Error
            ? fetchError.message
            : fetchError != null
                ? String(fetchError)
                : null;

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
            {errorMessage && (
                <Tooltip label={errorMessage} withinPortal>
                    <Badge color="red" size="xs" variant="filled">
                        ERROR
                    </Badge>
                </Tooltip>
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
            ) : isMissingData ? (
                <Text size="xs" c="dimmed">
                    No data at step {data.step}
                </Text>
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
}: Readonly<{
    label: string;
    value: unknown;
}>) {
    let display: string;
    if (value == null) {
        display = "--";
    } else {
        switch (typeof value) {
            case "string":
                display = value;
                break;
            case "number":
            case "boolean":
            case "bigint":
                display = String(value);
                break;
            case "object":
                display = JSON.stringify(value);
                break;
            default:
                display = JSON.stringify(value);
                break;
        }
    }
    return (
        <Group gap={4}>
            <Text size="xs" c="dimmed">
                {label}
            </Text>
            <Text size="xs" fw={500}>
                {display}
            </Text>
        </Group>
    );
}
